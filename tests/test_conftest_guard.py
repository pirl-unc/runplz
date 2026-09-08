"""Proves the autouse guard refuses real provider CLI and SDK launches.

Issues #35 and #170.
"""

from inspect import getattr_static
from unittest import mock

import modal as modal_sdk
import pytest

from runplz.backends import brev, provisioning, ssh_common
from runplz.backends import modal as modal_backend


class _FakeModalMethod:
    """Original descriptor used to prove a live marker delegates safely."""

    def __get__(self, instance, owner):
        def sync(*args, **kwargs):
            return "sync", args, kwargs

        def aio(*args, **kwargs):
            return "aio", args, kwargs

        sync.aio = aio
        return sync


def test_guard_blocks_real_brev_ls():
    with pytest.raises(RuntimeError, match="tried to run `brev`"):
        brev.subprocess.run(["brev", "ls", "--json"], capture_output=True, text=True)


def test_guard_blocks_real_brev_create():
    with pytest.raises(RuntimeError, match="tried to run `brev`"):
        brev.subprocess.run(["brev", "create", "box", "--type", "gpu"], check=True)


def test_guard_blocks_real_rsync_via_ssh_common():
    # The guard also covers the shared SSH plumbing — that's where every
    # real rsync/ssh call lives in the 3.5+ architecture.
    with pytest.raises(RuntimeError, match="tried to run `rsync`"):
        ssh_common.subprocess.run(["rsync", "-az", "src/", "box:dest/"], check=True)


def test_guard_blocks_real_ssh_via_ssh_common():
    with pytest.raises(RuntimeError, match="tried to run `ssh`"):
        ssh_common.subprocess.run(["ssh", "box", "echo hi"], check=True)


def test_guard_blocks_real_gcloud_via_cloud_helper():
    # The GCP/AWS drivers shell out through provisioning.run_cli — using the
    # vendor CLIs rather than an SDK is precisely what keeps them inside
    # this guard. An SDK call would be invisible to it.
    with pytest.raises(RuntimeError, match="tried to run `gcloud`"):
        provisioning.subprocess.run(["gcloud", "compute", "instances", "create", "box"])


def test_guard_blocks_real_aws_via_cloud_helper():
    with pytest.raises(RuntimeError, match="tried to run `aws`"):
        provisioning.subprocess.run(
            ["aws", "ec2", "run-instances", "--instance-type", "p5.48xlarge"]
        )


@pytest.mark.parametrize(
    "cmd",
    [
        ["modal", "run", "job.py::main"],
        "modal deploy service.py",
    ],
)
def test_guard_blocks_real_modal_cli_launches(cmd):
    with pytest.raises(RuntimeError, match="tried to run `modal`"):
        modal_backend.subprocess.run(cmd, check=True)


def test_guard_allows_explicit_modal_cli_mock():
    result = mock.Mock(returncode=0)
    with mock.patch.object(modal_backend.subprocess, "run", return_value=result) as run:
        assert modal_backend.subprocess.run(["modal", "run", "job.py::main"]) is result
    run.assert_called_once_with(["modal", "run", "job.py::main"])


def test_guard_allows_sandboxed_modal_cli(sandbox_bin):
    executable = sandbox_bin / "modal"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)

    result = modal_backend.subprocess.run(["modal", "run", "fake.py::main"])

    assert result.returncode == 0


@pytest.mark.parametrize("use_aio", [False, True], ids=["sync", "aio"])
@pytest.mark.parametrize(
    ("owner_name", "method_name"),
    [
        ("Function", "remote"),
        ("Function", "remote_gen"),
        ("Function", "spawn"),
        ("Function", "map"),
        ("Function", "starmap"),
        ("Function", "for_each"),
        ("App", "run"),
        ("App", "deploy"),
        ("Sandbox", "create"),
    ],
)
def test_guard_blocks_real_modal_sdk_launches(owner_name, method_name, use_aio):
    if owner_name == "Function":
        target = modal_sdk.Function.from_name("runplz-guard-test", "job")
    elif owner_name == "App":
        target = modal_sdk.App("runplz-guard-test")
    else:
        target = modal_sdk.Sandbox
    method = getattr(target, method_name)
    call = method.aio if use_aio else method

    with pytest.raises(RuntimeError, match=rf"`{owner_name}\.{method_name}` for real"):
        call()


def test_guard_allows_explicit_modal_sdk_mock(monkeypatch):
    spawn = mock.Mock(return_value="mocked call")
    monkeypatch.setattr(modal_sdk.Function, "spawn", spawn)
    function = modal_sdk.Function.from_name("runplz-guard-test", "job")

    assert function.spawn() == "mocked call"
    spawn.assert_called_once_with()


def test_guard_leaves_read_only_modal_sdk_construction_available():
    function = modal_sdk.Function.from_name("runplz-guard-test", "job")
    volume = modal_sdk.Volume.from_name("runplz-guard-test")

    assert function is not None
    assert volume is not None
    for owner, method_name in (
        (modal_sdk.FunctionCall, "from_id"),
        (modal_sdk.Volume, "read_file"),
        (modal_sdk.Volume, "iterdir"),
    ):
        assert not hasattr(getattr_static(owner, method_name), "_operation")


def test_guard_lets_unrelated_commands_through():
    # `which` isn't in the billed set — should run (and normally succeed).
    r = brev.subprocess.run(["which", "true"], capture_output=True, text=True)
    assert r.returncode in (0, 1)


def test_guard_lets_docker_commands_through():
    # docker is intentionally not guarded — the local backend uses it for
    # real builds on the developer's machine, and the hit is bounded
    # (local resource, not billed).
    # We only verify the call reaches subprocess.run (it'll either succeed
    # or fail on `docker` being missing; either outcome is fine for this
    # test since the guard didn't trip).
    import subprocess as real

    assert brev.subprocess.run is not real.run  # guard wrapper installed
    # This just needs to not raise our RuntimeError.
    try:
        brev.subprocess.run(
            ["docker", "image", "inspect", "nonexistent:tag"],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        pass  # no docker on this host — still proves the guard didn't block it


def test_guard_subprocess_module_proxies_non_run_attributes():
    # Code that uses subprocess.TimeoutExpired / CalledProcessError /
    # DEVNULL through a guarded backend module must still work.
    assert brev.subprocess.TimeoutExpired is __import__("subprocess").TimeoutExpired
    assert brev.subprocess.CalledProcessError is __import__("subprocess").CalledProcessError
    assert brev.subprocess.DEVNULL == __import__("subprocess").DEVNULL


@pytest.mark.live_brev
def test_guard_allows_real_brev_when_opted_in():
    # The marker lets a test call brev for real. We don't actually issue
    # a billed call here; we just assert the guard's escape hatch works by
    # running `brev --version` (harmless, local-only, but still hits the
    # binary). If `brev` isn't installed on this machine the test's
    # FileNotFoundError is fine — the guard didn't raise, which is the
    # property we're verifying.
    try:
        r = brev.subprocess.run(["brev", "--version"], capture_output=True, text=True)
        assert r.returncode in (0, 1, 127)  # any non-guard result is fine
    except FileNotFoundError:
        pass


@pytest.mark.live_modal
def test_guard_allows_harmless_modal_cli_when_opted_in():
    # Version reporting is local-only but still proves the marker reaches the
    # real executable rather than the guard.
    try:
        result = modal_backend.subprocess.run(
            ["modal", "--version"], capture_output=True, text=True
        )
        assert result.returncode == 0
    except FileNotFoundError:
        pass


@pytest.mark.live_modal
def test_guard_allows_modal_sdk_delegation_when_opted_in(monkeypatch):
    # Replace the captured real descriptor so this tests the marker/delegation
    # branch without submitting any provider work.
    guard = getattr_static(modal_sdk.Function, "spawn")
    monkeypatch.setattr(guard, "_original", _FakeModalMethod())
    function = modal_sdk.Function.from_name("runplz-guard-test", "job")

    assert function.spawn(1, value=2) == ("sync", (1,), {"value": 2})
    assert function.spawn.aio(3, value=4) == ("aio", (3,), {"value": 4})
