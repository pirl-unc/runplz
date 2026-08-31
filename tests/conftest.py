"""Test-wide safeguards.

Issue #35: the runplz test suite must never invoke the real `brev`,
`gcloud`, `aws`, `ssh`, or `rsync` CLIs. A plain `pytest` spinning up
a paid GPU box because one test forgot to mock a path is an
unacceptable footgun — especially when `pytest -n auto` multiplies
the blast radius and a killed test runner leaves orphan boxes
running.

This file installs an autouse fixture that replaces each backend
module's `subprocess` reference with a wrapper whose `.run` raises on
any of the banned CLIs. Tests that genuinely need live infra must opt
in via `@pytest.mark.live_brev` / `live_gcp` / `live_aws` / `live_ssh`.
Tests that already patch `subprocess.run` themselves are unaffected —
their patch overrides ours.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

# CLI names that cost money or touch user-owned infrastructure.
_BILLED_COMMANDS = {
    "brev": "live_brev",
    "gcloud": "live_gcp",
    "aws": "live_aws",
    "ssh": "live_ssh",
    "rsync": "live_ssh",
}


def pytest_configure(config):
    for marker in set(_BILLED_COMMANDS.values()):
        config.addinivalue_line(
            "markers",
            f"{marker}: test is allowed to shell out to the real CLI. "
            f"Do not add without explicit need.",
        )


def _first_token(cmd) -> str:
    if isinstance(cmd, (list, tuple)):
        return os.path.basename(str(cmd[0])) if cmd else ""
    if isinstance(cmd, str):
        head = cmd.strip().split(None, 1)
        return os.path.basename(head[0]) if head else ""
    return ""


# Directories holding stub executables a test installed itself. A billed
# name resolving inside one of these is not the real CLI and cannot spend
# money, so it is allowed through without a live marker. Registered by the
# `sandbox_bin` fixture; the guard resolves the program on PATH and compares,
# so the real `gcloud` stays blocked even while a fake one is installed.
_SANDBOX_BINS: set = set()


def _resolves_into_sandbox(prog: str) -> bool:
    if not _SANDBOX_BINS:
        return False
    found = shutil.which(prog)
    if not found:
        return False
    found = Path(found).resolve()
    return any(sandbox in found.parents for sandbox in _SANDBOX_BINS)


def _make_guarded_run(request):
    def guarded(cmd, *args, **kwargs):
        prog = _first_token(cmd)
        required = _BILLED_COMMANDS.get(prog)
        if required and not request.node.get_closest_marker(required):
            if _resolves_into_sandbox(prog):
                return subprocess.run(cmd, *args, **kwargs)
            raise RuntimeError(
                f"test {request.node.nodeid} tried to run `{prog}` for "
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
    "runplz.cli",
)


@pytest.fixture(autouse=True)
def _block_real_brev_cli(request, monkeypatch):
    """Swap each backend module's `subprocess` for a guarded wrapper."""
    guarded = _make_guarded_run(request)
    wrapper = _GuardedSubprocessModule(guarded)
    for mod_path in _MODULES_TO_GUARD:
        try:
            mod = __import__(mod_path, fromlist=["subprocess"])
        except ImportError:
            continue
        if hasattr(mod, "subprocess"):
            monkeypatch.setattr(mod, "subprocess", wrapper, raising=False)


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
def sshd_server(tmp_path_factory):
    """Shared, isolated SSH endpoint for local and cloud-handoff e2e tests.

    Auto mode may skip when this machine genuinely has no usable endpoint.
    An explicitly requested backend is a test contract and therefore fails
    loudly if it cannot start; CI cannot go green after silently testing zero
    SSH behavior.
    """
    from sshd_harness import select_backend

    mode = os.environ.get("RUNPLZ_E2E_REMOTE", "auto").lower()
    root = tmp_path_factory.mktemp("sshd")
    try:
        server, unavailable = select_backend(root)
    except Exception as exc:
        message = f"could not initialize SSH test backend: {exc}"
        if mode == "auto":
            pytest.skip(message)
        pytest.fail(message)
    if server is None:
        if mode == "auto":
            pytest.skip(unavailable)
        pytest.fail(unavailable)
    try:
        server.start()
    except Exception as exc:
        server.stop()
        message = f"could not start {type(server).__name__}: {exc}"
        if mode == "auto":
            pytest.skip(message)
        pytest.fail(message)
    try:
        yield server
    finally:
        server.stop()
