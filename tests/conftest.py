"""Test-wide safeguards.

Issues #35 and #170: the runplz test suite must never invoke a real provider
CLI or contact Modal's real control plane. A plain `pytest` spinning up
a paid GPU box because one test forgot to mock a path is an
unacceptable footgun — especially when `pytest -n auto` multiplies
the blast radius and a killed test runner leaves orphan boxes
running.

This file installs an autouse fixture that replaces each backend
module's `subprocess` reference with a wrapper whose process-starting
calls raise on any of the banned CLIs. Tests that genuinely need live
infra must opt in via `@pytest.mark.live_brev` / `live_gcp` / `live_aws` /
`live_ssh` / `live_modal`.
A test can still opt one module out by patching that module's own
`subprocess` attribute (`mock.patch.object(backend.subprocess, "run")`);
patching the global `subprocess.run` does not, since the guard classifies
before it delegates.

Classification is deliberately conservative: the guard does not try to work
out which token is *the* program, because every wrapper it has not heard of
(`uv run modal`, `timeout 600 modal`, `nohup modal`) hides that answer, and
each gap in such an analysis silently *allows* a billed launch. It asks the
cheaper question instead — does this command mention anything billed? —
and fails closed on anything it cannot read.
"""

from __future__ import annotations

import inspect
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

_E2E_REMOTE_ENV = "RUNPLZ_E2E_REMOTE"


def pytest_addoption(parser):
    """Declare `--e2e-remote`, which used to be the `RUNPLZ_E2E_REMOTE` env var.

    An env var is invisible: it does not appear in `pytest --help`, it is easy
    to leave exported in a shell and then wonder why a later run behaves
    differently, and it was being read and validated independently here and in
    `sshd_harness`, so the two could disagree about what was asked for.
    """
    from sshd_harness import MODES

    parser.addoption(
        "--e2e-remote",
        choices=MODES,
        default="auto",
        help=(
            "Which ssh endpoint the e2e tier runs against: a real sshd on this "
            "machine (local), a Debian container matching production (docker), "
            "or auto -- Docker only where a local sshd would be the wrong "
            "platform. Default: auto."
        ),
    )


# CLI names that cost money or touch user-owned infrastructure.
_BILLED_COMMANDS = {
    "brev": "live_brev",
    "gcloud": "live_gcp",
    "aws": "live_aws",
    "ssh": "live_ssh",
    "rsync": "live_ssh",
    "modal": "live_modal",
}

# Running this package as a child interpreter is opaque: the child can reach any
# backend, and no in-process patch of ours applies inside it. One entry covers
# every self-spawn rather than one per entry point, which would have the same
# "a missing entry silently allows" failure mode as the wrapper tables below.
_BILLED_MODULES = {"runplz": "modal"}

# Programs that look a name up without executing it, so their operands are not
# launches. Listing one here can only *narrow* what the guard blocks and each
# entry is justified by "never execs its operand" — the opposite direction from
# a wrapper or interpreter-flag table, where every missing entry lets a billed
# launch through. `_require_brev_cli` runs `which brev` as a precondition check
# that cannot spend money.
_NON_EXECUTING_PROGRAMS = {"which", "whereis"}

# A leading interpreter-flag cluster ending in `m`, so `-mmodal` reads as the
# module `modal`. CPython takes the rest of the cluster after the first `m` as
# the module name; the non-greedy prefix reproduces exactly that.
_MODULE_FLAG = re.compile(r"^-[A-Za-z]*?m")

# Popen's signature is fixed for the life of the process; binding against it is
# how options are found whether they were passed positionally or by keyword.
_POPEN_SIGNATURE = inspect.signature(subprocess.Popen)

# Accepted by `run`/`call`/`check_output` but not by `Popen`.
_RUN_ONLY_KWARGS = frozenset({"input", "capture_output", "timeout", "check"})


def pytest_configure(config):
    # Refuse to run with the retired env var still exported. Ignoring it
    # silently is the worse failure: a stale `RUNPLZ_E2E_REMOTE=docker` in a
    # shell profile would quietly downgrade to `auto`, and the run would test a
    # different backend than the operator asked for without saying so.
    if os.environ.get(_E2E_REMOTE_ENV):
        raise pytest.UsageError(
            f"{_E2E_REMOTE_ENV} is no longer read; pass "
            f"--e2e-remote={os.environ[_E2E_REMOTE_ENV].lower()} instead."
        )
    for marker in set(_BILLED_COMMANDS.values()):
        config.addinivalue_line(
            "markers",
            f"{marker}: test is allowed to shell out to the real CLI. "
            f"Do not add without explicit need.",
        )


# Directories holding stub executables a test installed itself. A billed
# name resolving inside one of these is not the real CLI and cannot spend
# money, so it is allowed through without a live marker. Registered by the
# `sandbox_bin` fixture; the guard resolves the program on PATH and compares,
# so the real `gcloud` stays blocked even while a fake one is installed.
_SANDBOX_BINS: set = set()

_ENVIRONMENT_SKIP_PREFIX = "ENVIRONMENT_UNAVAILABLE:"


def _skip_environment(reason: str) -> None:
    """Emit a machine-distinguishable skip for unavailable infrastructure."""
    pytest.skip(f"{_ENVIRONMENT_SKIP_PREFIX} {reason}")


def _resolves_into_sandbox(program, env, cwd):
    """Whether `program` is a stub a test installed, not the real CLI."""
    if not _SANDBOX_BINS:
        return False
    if os.path.dirname(program):
        # subprocess executes a path-bearing program exactly as supplied,
        # relative to the *child's* working directory — which `cwd=` moves and
        # ours does not. Only bare names are looked up on PATH.
        if isinstance(cwd, int):
            # cwd may be a directory file descriptor, which names a directory
            # we cannot resolve a relative path against.
            return False
        found = Path(program) if cwd is None else Path(os.fsdecode(cwd)) / program
        # `shutil.which` makes this check for bare names, so a path must too:
        # otherwise a sandbox directory whose stub was never written earns the
        # exemption and the guard reports a confusing FileNotFoundError.
        if not (found.is_file() and os.access(found, os.X_OK)):
            return False
    else:
        try:
            search_path = os.pathsep.join(os.get_exec_path(env))
        except (TypeError, ValueError):
            # An environment mapping we cannot normalize may still change PATH,
            # so we cannot claim to know which file the child would find.
            return False
        located = shutil.which(program, path=search_path)
        if not located:
            return False
        found = Path(located)
    return any(sandbox in found.resolve().parents for sandbox in _SANDBOX_BINS)


def _words(text):
    """Every bare word in `text`, read with and without shell quoting.

    An argument may itself be a command line: `sh -c "modal run job"` and
    `env -S "modal run job"` both hide a launch inside a single token. Reading
    the words of every token catches those without the guard having to know
    which programs re-tokenize their arguments.
    """
    words = set()
    for word in text.split():
        # A value glued to a flag or a name (`--split-string=modal run job`,
        # `PATH=/x`) hides the half that names a program.
        for part in (word, word.partition("=")[2]):
            words.update({part, part.strip("\"'")})
    try:
        words.update(shlex.split(text))
    except ValueError:
        # Unbalanced quotes. The quote-stripped reading above still stands, so
        # a billed name cannot hide behind one.
        pass
    return words - {""}


def _command(cmd):
    """The program `cmd` names and every word it is built from, or None.

    None means a shape we cannot read at all, which the caller turns into a
    refusal rather than a silent delegation.
    """
    if isinstance(cmd, (str, bytes, os.PathLike)):
        # POSIX runs a shell-free string as one executable path; Windows parses
        # it as a command line. Read it both ways rather than picking one.
        text = os.fsdecode(cmd)
        first = text.split()
        return (first[0] if first else text), _words(text) | {text}
    if isinstance(cmd, (list, tuple)):
        tokens = [os.fsdecode(v) if isinstance(v, (bytes, os.PathLike)) else str(v) for v in cmd]
        words = set(tokens)
        for token in tokens:
            words |= _words(token)
        return (tokens[0] if tokens else ""), words
    return None


def _billed_name(word):
    """The billed CLI this word names, or None.

    A module name is billed by its root package, so `modal.cli.entry_point` and
    every future submodule are covered without naming them.
    """
    if os.path.basename(word) in _BILLED_COMMANDS:
        return os.path.basename(word)
    if os.sep in word or "/" in word:
        return None
    if word.startswith("-"):
        # An interpreter flag cluster: what follows the `m` is a module name.
        module, matched = _MODULE_FLAG.subn("", word, count=1)
        if not matched:
            return None
    elif "." in word:
        module = word
    else:
        # A bare word is a program name, already handled by basename above.
        return None
    root = module.partition(".")[0]
    return _BILLED_MODULES.get(root) or (root if root in _BILLED_COMMANDS else None)


def _reject_billed_commands(request, api, call_args, call_kwargs):
    """Raise unless every billed name in this command is allowed to run."""
    popen_kwargs = {k: v for k, v in call_kwargs.items() if k not in _RUN_ONLY_KWARGS}
    try:
        arguments = _POPEN_SIGNATURE.bind_partial(*call_args, **popen_kwargs).arguments
    except TypeError:
        raise RuntimeError(
            f"test {request.node.nodeid} called subprocess.{api} with arguments the provider "
            f"safety guard could not match against subprocess.Popen — pass explicit argv or "
            f"mock subprocess.{api}. args: {call_args!r}"
        ) from None

    cmd = arguments.get("args")

    # A shell can expand variables and execute arbitrarily many commands;
    # tokenizing its source would not tell us what it will launch. Tests
    # must use explicit argv or mock this boundary instead.
    if arguments.get("shell"):
        raise RuntimeError(
            f"test {request.node.nodeid} tried to use subprocess.{api}(shell=True), "
            "which the provider safety guard cannot inspect — pass explicit argv or mock "
            f"subprocess.{api}. cmd: {cmd!r}"
        )

    read = _command(cmd)
    if read is None:
        raise RuntimeError(
            f"test {request.node.nodeid} passed a command the provider safety guard cannot "
            f"read — pass explicit argv or mock subprocess.{api}. cmd: {cmd!r}"
        )
    program, words = read

    # On POSIX `executable=` replaces the program that is executed while argv[0]
    # stays whatever cmd[0] was, so it is what the exemption must be judged on.
    executable = arguments.get("executable")
    if executable is not None:
        program = os.fsdecode(executable)
        words |= {program}

    if os.path.basename(program) in _NON_EXECUTING_PROGRAMS:
        words = {program}

    # Sorted so a command naming two billed tools always reports the same one.
    for word in sorted(words):
        name = _billed_name(word)
        if name is None or request.node.get_closest_marker(_BILLED_COMMANDS[name]):
            continue
        # Only the program actually launched can be a test's own stub. A billed
        # name anywhere else is an argument to something we did not classify --
        # a wrapper, or an `env` whose assignments change lookup — so no PATH
        # of ours can vouch for what the child would find.
        if word == program and _resolves_into_sandbox(
            program, arguments.get("env"), arguments.get("cwd")
        ):
            continue
        raise RuntimeError(
            f"test {request.node.nodeid} tried to run `{name}` for "
            f"real — mock it, or mark the test `@pytest.mark.{_BILLED_COMMANDS[name]}` "
            f"if hitting live infra is intentional. cmd: {cmd!r}"
        )


def _make_guarded(request, api):
    def guarded(*args, **kwargs):
        _reject_billed_commands(request, api, args, kwargs)
        # Looked up at call time so a test's own patch of the real module is
        # still what runs.
        return getattr(subprocess, api)(*args, **kwargs)

    return guarded


@pytest.fixture
def sandbox_bin(tmp_path, monkeypatch):
    """A PATH directory for stub executables, exempted from the billing guard.

    Lets a test install a fake `gcloud`/`aws` and run the real provisioning
    code against it. The exemption is by resolved path, not by name, so a
    test that forgets to install its stub still hits the real guard rather
    than silently reaching the actual CLI.
    """
    bin_dir = (tmp_path / "sandbox-bin").resolve()
    bin_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    _SANDBOX_BINS.add(bin_dir)
    try:
        yield bin_dir
    finally:
        _SANDBOX_BINS.discard(bin_dir)


# Every documented way `subprocess` starts a process. Guarding `run` alone left
# the other four as unguarded routes to the same binaries.
# Every way `subprocess` starts a process. `getoutput`/`getstatusoutput` belong
# here too: they take a command line and run it through a shell, so leaving
# them to `__getattr__` handed back an unguarded `shell=True`.
_PROCESS_STARTING_APIS = (
    "run",
    "Popen",
    "call",
    "check_call",
    "check_output",
    "getoutput",
    "getstatusoutput",
)


class _GuardedSubprocessModule:
    """Thin wrapper over the real `subprocess` module.

    Delegates every attribute to the real module except the calls that start a
    process, each of which refuses billed CLIs first. This lets code keep
    using `subprocess.CalledProcessError`, `subprocess.TimeoutExpired`,
    `subprocess.DEVNULL`, etc. without us having to enumerate them.
    """

    def __init__(self, request):
        for api in _PROCESS_STARTING_APIS:
            setattr(self, api, _make_guarded(request, api))

    def __getattr__(self, name):
        return getattr(subprocess, name)


# Every module that calls subprocess.run needs its `subprocess`
# reference wrapped for the duration of each test.
_MODULES_TO_GUARD = (
    "runplz.backends.brev",
    "runplz.backends.provisioning",
    "runplz.runs",
    "runplz.backends.ssh_common",
    "runplz.backends.ssh",
    "runplz.backends.modal",
    # The Modal worker shells out to a *child interpreter* that builds a real
    # Modal client, so the in-process SDK guard cannot see it (#170 follow-up).
    "runplz.backends.modal_runs",
    "runplz.backends.local",
    # aws/gcp were missing until 3.25.0, so `list_jobs` reached the real
    # `aws` / `gcloud` binaries whenever a test drove `runplz ps` on a machine
    # with the provider env vars set — exactly the billed-CLI call this guard
    # exists to stop.
    "runplz.backends.aws",
    "runplz.backends.gcp",
)


@pytest.fixture(autouse=True)
def _block_real_provider_calls(request, monkeypatch):
    """Guard provider CLIs and every call to Modal's real control plane."""
    for mod_path in _MODULES_TO_GUARD:
        mod = __import__(mod_path, fromlist=["subprocess"])
        if not hasattr(mod, "subprocess"):
            # Skipping silently is how a listed module stops being guarded
            # without anything going red — `import subprocess as sp` keeps its
            # place in the list and every `sp.run(...)` in it reaches the real
            # binary.
            raise RuntimeError(
                f"{mod_path} is listed in _MODULES_TO_GUARD but has no `subprocess` "
                "attribute to replace; import it as plain `import subprocess`."
            )
        # A wrapper per module, so a test that patches one module's
        # `subprocess.run` does not silently unguard every other backend.
        monkeypatch.setattr(mod, "subprocess", _GuardedSubprocessModule(request))

    try:
        from modal.client import _Client
    except ImportError as exc:
        # `modal` is a hard runtime dependency (pyproject.toml), so this is not
        # "Modal is not installed" — it is the SDK having moved or renamed
        # `_Client`, the one case where continuing would leave every test in
        # the suite running with no control-plane guard at all.
        raise RuntimeError(
            "the Modal control-plane guard could not find `modal.client._Client`; "
            "point tests/conftest.py at the SDK's current channel boundary rather "
            "than running unguarded."
        ) from exc

    original_get_channel = _Client._get_channel

    async def guarded_get_channel(client, server_url):
        if not request.node.get_closest_marker("live_modal"):
            raise RuntimeError(
                f"test {request.node.nodeid} tried to contact the real Modal control plane — "
                "use an offline fake, or mark the test `@pytest.mark.live_modal` "
                "if live Modal access is intentional."
            )
        return await original_get_channel(client, server_url)

    monkeypatch.setattr(_Client, "_get_channel", guarded_get_channel)


@pytest.fixture(autouse=True)
def _isolate_brev_onboarding(monkeypatch, tmp_path):
    """_skip_onboarding writes ~/.brev/onboarding_step.json. Redirect
    to a tmp path per test so the developer's real Brev state stays
    untouched."""
    try:
        from runplz.backends import brev
    except ImportError:
        return
    if hasattr(brev, "_BREV_ONBOARDING"):
        monkeypatch.setattr(brev, "_BREV_ONBOARDING", tmp_path / ".brev-onboarding.json")


@pytest.fixture
def fast_clock(monkeypatch):
    """Make every backoff and poll interval instantaneous.

    `wait_until_ssh_reachable` polls every 15s and the retry loops back off,
    so a test that drives real provisioning code spends its whole runtime
    asleep. Patches the `time` reference inside each module that sleeps, not
    the global one.
    """
    from clock import FakeClock

    clock = FakeClock()
    for mod_path in ("runplz.backends.ssh_common", "runplz.backends.provisioning"):
        mod = __import__(mod_path, fromlist=["time"])
        monkeypatch.setattr(mod, "time", clock, raising=False)
    return clock


@pytest.fixture(scope="session")
def sshd_server(request, tmp_path_factory):
    """Shared, isolated SSH endpoint for local and cloud-handoff e2e tests.

    Auto mode may skip when this machine genuinely has no usable endpoint.
    An explicitly requested backend is a test contract and therefore fails
    loudly if it cannot start; CI cannot go green after silently testing zero
    SSH behavior.
    """
    from sshd_harness import select_backend

    mode = request.config.getoption("--e2e-remote")
    root = tmp_path_factory.mktemp("sshd")
    try:
        server, unavailable = select_backend(root, mode)
    except Exception as exc:
        message = f"could not initialize SSH test backend: {exc}"
        if mode == "auto":
            _skip_environment(message)
        pytest.fail(message)
    if server is None:
        if mode == "auto":
            _skip_environment(unavailable)
        pytest.fail(unavailable)
    try:
        server.start()
    except Exception as exc:
        server.stop()
        message = f"could not start {type(server).__name__}: {exc}"
        if mode == "auto":
            _skip_environment(message)
        pytest.fail(message)
    try:
        yield server
    finally:
        server.stop()
