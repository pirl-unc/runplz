"""Test-wide safeguards.

Issues #35 and #170: the runplz test suite must never invoke a real provider
CLI or contact Modal's real control plane. A plain `pytest` spinning up
a paid GPU box because one test forgot to mock a path is an
unacceptable footgun — especially when `pytest -n auto` multiplies
the blast radius and a killed test runner leaves orphan boxes
running.

This file installs an autouse fixture that replaces each backend
module's `subprocess` reference with a wrapper whose `.run` raises on
any of the banned CLIs. Tests that genuinely need live infra must opt
in via `@pytest.mark.live_brev` / `live_gcp` / `live_aws` / `live_ssh` /
`live_modal`.
Tests that already patch `subprocess.run` themselves are unaffected —
their patch overrides ours.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
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


def _resolves_into_sandbox(executable: str) -> bool:
    if not _SANDBOX_BINS:
        return False
    # subprocess executes paths containing a directory exactly as supplied;
    # only bare command names are resolved through PATH.
    found = executable if os.path.dirname(executable) else shutil.which(executable)
    if not found:
        return False
    found = Path(found).resolve()
    return any(sandbox in found.parents for sandbox in _SANDBOX_BINS)


def _make_guarded_run(request):
    def guarded(cmd, *args, **kwargs):
        if isinstance(cmd, str):
            try:
                command_args = shlex.split(cmd)
            except ValueError:
                command_args = []
        elif isinstance(cmd, (list, tuple)):
            command_args = [str(value) for value in cmd]
        else:
            command_args = []

        # `env` is the one common executable wrapper: peel off its options and
        # NAME=VALUE assignments so the program it launches remains visible.
        effective_args = command_args
        if command_args and os.path.basename(command_args[0]) == "env":
            command_index = 1
            while command_index < len(command_args):
                value = command_args[command_index]
                if value == "--":
                    command_index += 1
                    break
                if value in {
                    "-u",
                    "--unset",
                    "-C",
                    "--chdir",
                    "-S",
                    "--split-string",
                    "-P",
                    "-a",
                    "--argv0",
                }:
                    command_index += 2
                    continue
                if value.startswith(("--unset=", "--chdir=", "--split-string=", "--argv0=")):
                    command_index += 1
                    continue
                if value.startswith("-"):
                    command_index += 1
                    continue
                name, separator, _ = value.partition("=")
                is_assignment = (
                    bool(separator)
                    and bool(name)
                    and not name[0].isdigit()
                    and name.replace("_", "a").isalnum()
                )
                if is_assignment:
                    command_index += 1
                    continue
                break
            effective_args = command_args[command_index:]

        effective_executable = effective_args[0] if effective_args else ""
        executable_name = os.path.basename(effective_executable)
        billed_command = executable_name
        executable_is_python = executable_name.lower().startswith("python") or (
            bool(effective_args)
            and os.path.realpath(effective_executable) == os.path.realpath(sys.executable)
        )
        modal_module = False
        if executable_is_python:
            argument_index = 1
            while argument_index < len(effective_args):
                value = effective_args[argument_index]
                if value == "-m":
                    modal_module = (
                        argument_index + 1 < len(effective_args)
                        and effective_args[argument_index + 1] == "modal"
                    )
                    break
                if value.startswith("-m") and len(value) > 2:
                    modal_module = value[2:] == "modal"
                    break
                if value in {"-c", "-", "--"} or not value.startswith("-"):
                    break
                # These interpreter options consume the following argument;
                # joined forms such as `-Xdev` naturally consume only this one.
                if value in {"-W", "-X", "--check-hash-based-pycs"}:
                    argument_index += 2
                else:
                    argument_index += 1
        runs_modal_module = executable_is_python and modal_module
        if runs_modal_module:
            billed_command = "modal"

        required = _BILLED_COMMANDS.get(billed_command)
        if required and not request.node.get_closest_marker(required):
            # A sandboxed executable makes a direct CLI call safe. It cannot
            # make `python -m modal` safe: that imports the installed SDK.
            if not runs_modal_module and _resolves_into_sandbox(effective_executable):
                return subprocess.run(cmd, *args, **kwargs)
            raise RuntimeError(
                f"test {request.node.nodeid} tried to run `{billed_command}` for "
                f"real — mock it, or mark the test `@pytest.mark.{required}` "
                f"if hitting live infra is intentional. cmd: {cmd!r}"
            )
        return subprocess.run(cmd, *args, **kwargs)

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


class _GuardedSubprocessModule:
    """Thin wrapper over the real `subprocess` module.

    Delegates every attribute to the real module except `run`, which is
    replaced with a guard that refuses billed CLIs. This lets code keep
    using `subprocess.CalledProcessError`, `subprocess.TimeoutExpired`,
    `subprocess.DEVNULL`, etc. without us having to enumerate them.
    """

    def __init__(self, guarded_run):
        self.run = guarded_run

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
    "runplz.backends.local",
    # aws/gcp were missing until 3.25.0, so `list_jobs` reached the real
    # `aws` / `gcloud` binaries whenever a test drove `runplz ps` on a machine
    # with the provider env vars set — exactly the billed-CLI call this guard
    # exists to stop.
    "runplz.backends.aws",
    "runplz.backends.gcp",
    "runplz.cli",
)


@pytest.fixture(autouse=True)
def _block_real_provider_calls(request, monkeypatch):
    """Guard provider CLIs and every call to Modal's real control plane."""
    guarded = _make_guarded_run(request)
    wrapper = _GuardedSubprocessModule(guarded)
    for mod_path in _MODULES_TO_GUARD:
        try:
            mod = __import__(mod_path, fromlist=["subprocess"])
        except ImportError:
            continue
        if hasattr(mod, "subprocess"):
            monkeypatch.setattr(mod, "subprocess", wrapper, raising=False)

    try:
        from modal.client import _Client
    except ImportError:
        return

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
