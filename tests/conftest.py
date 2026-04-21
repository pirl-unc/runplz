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
import subprocess

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


def _make_guarded_run(request):
    def guarded(cmd, *args, **kwargs):
        prog = _first_token(cmd)
        required = _BILLED_COMMANDS.get(prog)
        if required and not request.node.get_closest_marker(required):
            raise RuntimeError(
                f"test {request.node.nodeid} tried to run `{prog}` for "
                f"real — mock it, or mark the test `@pytest.mark.{required}` "
                f"if hitting live infra is intentional. cmd: {cmd!r}"
            )
        return subprocess.run(cmd, *args, **kwargs)

    return guarded


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
    "runplz.backends._ssh_common",
    "runplz.backends.ssh",
    "runplz.backends.modal",
    "runplz.backends.local",
    "runplz._cli",
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
