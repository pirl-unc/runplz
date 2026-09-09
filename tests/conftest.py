"""Test-wide safeguards.

Issues #35 and #170: the runplz test suite must never invoke a real provider
CLI or contact Modal's real control plane. A plain `pytest` spinning up
a paid GPU box because one test forgot to mock a path is an
unacceptable footgun — especially when `pytest -n auto` multiplies
the blast radius and a killed test runner leaves orphan boxes
running.

This file installs an autouse fixture that hooks `subprocess.Popen` — the one
seam every spawn in the process goes through, whether it is spelled `run`,
`check_output`, `getoutput`, `from subprocess import run`, or an asyncio
subprocess — and refuses any command that names a banned CLI. Tests that
genuinely need live infra opt in via `@pytest.mark.live_brev` / `live_gcp` /
`live_aws` / `live_ssh` / `live_modal`; a test that must spawn a real child of
this package without live access uses the `real_child_processes` fixture. A
test can also take the boundary over with `mock.patch("<module>.subprocess.run")`,
which is process-wide because `subprocess` is one module.

Classification is deliberately conservative: the guard does not try to work
out which token is *the* program, because every wrapper it has not heard of
(`uv run modal`, `timeout 600 modal`, `nohup modal`) hides that answer, and
each gap in such an analysis silently *allows* a billed launch. It asks the
cheaper question instead — does this command mention anything billed? — and
fails closed on anything it cannot read.
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

import runplz

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

# Python modules whose execution reaches a billed provider, looked up by exact
# name and then by root package, so `modal.cli.entry_point` and every future
# submodule are covered without naming them. The Modal worker is here because
# it is spawned as a child interpreter that builds a real client, where no
# in-process patch of ours applies.
_BILLED_MODULES = {"modal": "modal", "runplz.backends.modal_runs": "modal"}

# The package's own CLI. A child running it can dispatch to any backend, so no
# single marker can vouch for it; it is refused outright, like `shell=True`.
# The module and file-path spellings are billed wherever they appear; the bare
# console-script name only as the launched program, because `runplz` is
# ordinary data everywhere else in this repo's own argv (`--labels=runplz=1`,
# `git config user.name "runplz test"`).
_SELF_SPAWNS = {"runplz.cli"}
_CONSOLE_SCRIPT = "runplz"
_PACKAGE_DIR = Path(runplz.__file__).resolve().parent

# Where one word ends inside an argument: whitespace, the shell's own
# separators (`cd /tmp;modal run`, `true&&modal`), and `=` (`--opt=modal`).
_SEPARATORS = re.compile(r"[\s;&|()<>=]+")
# A version specifier glued to a package name (`modal[aws]`, `modal~=1.5`).
_SPECIFIER = re.compile(r"[\[@~!<>=].*$")

# The signature options are read against. It is validated here because a
# `Popen.__init__` wrapper installed earlier without `functools.wraps` would
# degrade it to `(*args, **kwargs)`, after which nothing would bind and every
# command would pass — the one failure this guard must never have silently.
_ORIGINAL_POPEN_INIT = subprocess.Popen.__init__
_POPEN_INIT_SIGNATURE = inspect.signature(_ORIGINAL_POPEN_INIT)
assert {"args", "shell", "executable", "cwd", "env"} <= set(_POPEN_INIT_SIGNATURE.parameters), (
    "subprocess.Popen.__init__ has an unexpected signature; something wrapped it before "
    f"conftest imported. Parameters: {list(_POPEN_INIT_SIGNATURE.parameters)}"
)


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


def _child_path(word, cwd):
    """Where the child would find `word`, or None if we cannot know.

    subprocess resolves a path-bearing program relative to the *child's*
    working directory — which `cwd=` moves and ours does not.
    """
    try:
        base = Path.cwd() if cwd is None else Path(os.fsdecode(cwd))
    except TypeError:
        return None  # e.g. a directory file descriptor
    return base / word


def _resolves_into_sandbox(program, env, cwd):
    """Whether `program` is a stub a test installed, not the real CLI."""
    if not _SANDBOX_BINS:
        return False
    if os.path.dirname(program):
        found = _child_path(program, cwd)
        # `shutil.which` checks this for bare names, so a path must too:
        # otherwise a sandbox directory whose stub was never written earns the
        # exemption and the guard reports a confusing FileNotFoundError.
        if found is None or not (found.is_file() and os.access(found, os.X_OK)):
            return False
    else:
        try:
            search_path = os.pathsep.join(os.get_exec_path(env))
        except (AttributeError, TypeError, ValueError):
            # An environment we cannot normalize may still change PATH, so we
            # cannot claim to know which file the child would find.
            return False
        located = shutil.which(program, path=search_path)
        if not located:
            return False
        found = Path(located)
    return any(sandbox in found.resolve().parents for sandbox in _SANDBOX_BINS)


def _words(text):
    """Every bare word in `text`, read every way a shell or a tool might.

    An argument may itself be a command line: `sh -c "modal run job"` and
    `env -S "modal run job"` both hide a launch inside a single token. Reading
    the words of every token catches those without the guard having to know
    which programs re-tokenize their arguments.
    """
    words = set()
    for piece in _SEPARATORS.split(text):
        words.update({piece, piece.strip("\"'"), _SPECIFIER.sub("", piece)})
        if piece.startswith("-") and not piece.startswith("--"):
            # A value glued to a short option: `-Smodal run job` (env) or
            # `-mmodal` (python). Every suffix is cheaper than knowing which
            # letters take a value, and can only over-block.
            words.update(piece[i:] for i in range(2, len(piece)))
    try:
        words.update(shlex.split(text))
    except ValueError:
        pass  # Unbalanced quotes; the quote-stripped reading above still stands.
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
        return (first[0] if first else text), _words(text)
    if isinstance(cmd, (list, tuple)):
        tokens = [os.fsdecode(v) if isinstance(v, (bytes, os.PathLike)) else str(v) for v in cmd]
        words = set(tokens)
        for token in tokens:
            words |= _words(token)
        return (tokens[0] if tokens else ""), words
    return None


def _module_name(word, cwd):
    """The Python module `word` would run, if it names one of ours or theirs."""
    if word.endswith(".py"):
        # A file, not a module name: `python <pkg>/backends/modal_runs.py` is
        # the worker by path, while `modal.py` in the cwd is somebody's script.
        found = _child_path(word, cwd)
        try:
            relative = found.resolve().relative_to(_PACKAGE_DIR)
        except (AttributeError, ValueError):
            return None  # not one of ours
        return ".".join((_PACKAGE_DIR.name, *relative.with_suffix("").parts))
    if "/" in word or os.sep in word:
        return None
    return _SPECIFIER.sub("", word)


def _billed_name(word, cwd):
    """What `word` launches that costs money: a CLI name, or None.

    Matching is case-folded because the developer platform's filesystem is
    case-insensitive: `Modal` would exec the real binary.
    """
    name = os.path.basename(word).casefold()
    if name in _BILLED_COMMANDS:
        return name
    module = _module_name(word, cwd)
    if module is None:
        return None
    if module in _SELF_SPAWNS:
        return _CONSOLE_SCRIPT
    return _BILLED_MODULES.get(module) or _BILLED_MODULES.get(module.partition(".")[0])


def _reject_billed_commands(request, call_args, call_kwargs):
    """Raise unless every billed name in this Popen call is allowed to run."""
    try:
        arguments = _POPEN_INIT_SIGNATURE.bind_partial(None, *call_args, **call_kwargs).arguments
    except TypeError:
        raise RuntimeError(
            f"test {request.node.nodeid} called subprocess with arguments the provider "
            "safety guard could not match against Popen's signature — pass explicit argv "
            f"or mock subprocess.run. args: {call_args!r}"
        ) from None
    cmd = arguments.get("args")

    # A shell can expand variables and execute arbitrarily many commands;
    # tokenizing its source would not tell us what it will launch. This is
    # also where `getoutput`/`getstatusoutput` arrive. Tests must use explicit
    # argv or mock this boundary instead.
    if arguments.get("shell"):
        raise RuntimeError(
            f"test {request.node.nodeid} tried to use subprocess with shell=True, "
            "which the provider safety guard cannot inspect — pass explicit argv or mock "
            f"subprocess.run. cmd: {cmd!r}"
        )

    read = _command(cmd)
    if read is None:
        raise RuntimeError(
            f"test {request.node.nodeid} passed a command the provider safety guard cannot "
            f"read — pass explicit argv or mock subprocess.run. cmd: {cmd!r}"
        )
    program, words = read

    # On POSIX `executable=` replaces the program that is executed; argv[0] is
    # then only what the child sees as its own name, and is not launched.
    executable = arguments.get("executable")
    if executable is not None:
        words = (words - {program}) | {os.fsdecode(executable)}
        program = os.fsdecode(executable)

    cwd = arguments.get("cwd")
    # Sorted so a command naming two billed tools always reports the same one.
    for word in sorted(words):
        name = _billed_name(word, cwd)
        if (
            name is None
            and word == program
            and os.path.basename(word).casefold() == _CONSOLE_SCRIPT
        ):
            name = _CONSOLE_SCRIPT
        if name is None:
            continue
        marker = _BILLED_COMMANDS.get(name)
        if marker is None:
            raise RuntimeError(
                f"test {request.node.nodeid} tried to run the `runplz` CLI in a child process, "
                "which can dispatch to any backend where no in-process guard applies — call "
                "runplz.cli.main() in-process, or use the `real_child_processes` fixture with "
                f"a comment explaining why the child stays offline. cmd: {cmd!r}"
            )
        if request.node.get_closest_marker(marker):
            continue
        # Only the program actually launched can be a test's own stub. A billed
        # name anywhere else is an argument to something we did not classify —
        # a wrapper, or an `env` whose assignments change lookup — so no PATH
        # of ours can vouch for what the child would find.
        if word == program and _resolves_into_sandbox(program, arguments.get("env"), cwd):
            continue
        raise RuntimeError(
            f"test {request.node.nodeid} tried to run `{name}` for "
            f"real — mock it, or mark the test `@pytest.mark.{marker}` "
            f"if hitting live infra is intentional. cmd: {cmd!r}"
        )


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


@pytest.fixture(autouse=True)
def _block_real_provider_calls(request, monkeypatch):
    """Guard every process spawn and every call to Modal's real control plane."""

    def guarded_init(popen, *args, **kwargs):
        _reject_billed_commands(request, args, kwargs)
        _ORIGINAL_POPEN_INIT(popen, *args, **kwargs)

    monkeypatch.setattr(subprocess.Popen, "__init__", guarded_init)

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


@pytest.fixture
def real_child_processes(monkeypatch):
    """Let this test spawn a real child of this package without a live marker.

    For the few tests that must run the actual worker interpreter and can show
    it stays offline — a `prepared` receipt is answered from disk, an invalid
    one is rejected before any client exists. The test must say why in a
    comment; the SDK control-plane guard stays in force for the parent, and
    nothing in the child is guarded at all.
    """
    monkeypatch.setattr(subprocess.Popen, "__init__", _ORIGINAL_POPEN_INIT)


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
