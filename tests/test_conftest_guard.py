"""Proves the autouse guard refuses real provider CLI and SDK launches.

Issues #35 and #170.
"""

import asyncio
import shlex
import sys
from unittest import mock

import modal as modal_sdk
import pytest
from modal.client import _Client

from runplz.backends import brev, provisioning, ssh_common
from runplz.backends import modal as modal_backend


@pytest.fixture
def fake_modal_credentials(monkeypatch):
    """Let public SDK calls reach the guarded channel without local credentials."""
    monkeypatch.setenv("MODAL_TOKEN_ID", "ak-runplz-test")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "as-runplz-test")
    monkeypatch.setattr(_Client, "_client_from_env", None)
    monkeypatch.setattr(_Client, "_client_from_env_lock", None)


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
        ("modal", "deploy", "service.py"),
        "modal run job.py::main",
        ["/usr/local/bin/modal", "deploy", "service.py"],
        [sys.executable, "-m", "modal", "run", "job.py::main"],
        [sys.executable, "-u", "-m", "modal", "run", "job.py::main"],
        [sys.executable, "-mmodal", "run", "job.py::main"],
        ["env", "modal", "run", "job.py::main"],
        ["env", "RUNPLZ_GUARD_TEST=1", "modal", "run", "job.py::main"],
        ["env", "-u", "RUNPLZ_GUARD_TEST", "modal", "run", "job.py::main"],
        ["env", sys.executable, "-u", "-m", "modal", "run", "job.py::main"],
        f"{shlex.quote(sys.executable)} -m modal deploy service.py",
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

    for command in (
        ["modal", "run", "fake.py::main"],
        [str(executable), "run", "fake.py::main"],
        ["env", "modal", "run", "fake.py::main"],
    ):
        result = modal_backend.subprocess.run(command)
        assert result.returncode == 0


def test_sandboxed_name_does_not_exempt_an_explicit_modal_path(sandbox_bin):
    sandboxed = sandbox_bin / "modal"
    sandboxed.write_text("#!/bin/sh\nexit 0\n")
    sandboxed.chmod(0o755)

    outside_bin = sandbox_bin.parent / "outside-bin"
    outside_bin.mkdir()
    explicitly_invoked = outside_bin / "modal"
    explicitly_invoked.write_text("#!/bin/sh\nexit 0\n")
    explicitly_invoked.chmod(0o755)

    with pytest.raises(RuntimeError, match="tried to run `modal`"):
        modal_backend.subprocess.run([str(explicitly_invoked), "run", "job.py::main"])


def test_sandboxed_executable_does_not_exempt_python_modal_module(sandbox_bin):
    executable = sandbox_bin / "modal"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)

    with pytest.raises(RuntimeError, match="tried to run `modal`"):
        modal_backend.subprocess.run([sys.executable, "-m", "modal", "run", "job.py::main"])


def test_guard_allows_non_modal_python_module():
    result = modal_backend.subprocess.run(
        [sys.executable, "-u", "-m", "this"], capture_output=True, text=True
    )

    assert result.returncode == 0


@pytest.mark.parametrize("use_aio", [False, True], ids=["sync", "aio"])
def test_guard_blocks_modal_function_at_control_plane(use_aio, fake_modal_credentials):
    function = modal_sdk.Function.from_name("runplz-guard-test", "job")
    call = function.spawn.aio if use_aio else function.spawn

    with pytest.raises(RuntimeError, match="real Modal control plane"):
        if use_aio:
            asyncio.run(call())
        else:
            call()


@pytest.mark.parametrize("use_aio", [False, True], ids=["sync", "aio"])
def test_guard_blocks_modal_class_autoscaler_at_control_plane(use_aio, fake_modal_credentials):
    model = modal_sdk.Cls.from_name("runplz-guard-test", "Model")
    instance = model()
    call = instance.update_autoscaler.aio if use_aio else instance.update_autoscaler

    with pytest.raises(RuntimeError, match="real Modal control plane"):
        if use_aio:
            asyncio.run(call(buffer_containers=1))
        else:
            call(buffer_containers=1)


def test_guard_blocks_direct_modal_control_plane_access_before_connection():
    client = mock.Mock()

    with pytest.raises(RuntimeError, match="real Modal control plane"):
        asyncio.run(_Client._get_channel(client, "https://example.invalid"))

    assert client.mock_calls == []


def test_guard_allows_explicit_modal_sdk_mock(monkeypatch):
    spawn = mock.Mock(return_value="mocked call")
    monkeypatch.setattr(modal_sdk.Function, "spawn", spawn)
    function = modal_sdk.Function.from_name("runplz-guard-test", "job")

    assert function.spawn() == "mocked call"
    spawn.assert_called_once_with()


def test_guard_allows_offline_modal_object_construction():
    function = modal_sdk.Function.from_name("runplz-guard-test", "job")
    volume = modal_sdk.Volume.from_name("runplz-guard-test")
    model = modal_sdk.Cls.from_name("runplz-guard-test", "Model")

    assert function is not None
    assert volume is not None
    assert model is not None


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
@pytest.mark.parametrize(
    "cmd",
    [
        ["modal", "--version"],
        [sys.executable, "-m", "modal", "--version"],
    ],
)
def test_guard_allows_harmless_modal_cli_when_opted_in(cmd):
    # Version reporting is local-only but still proves the marker reaches the
    # real executable rather than the guard.
    try:
        result = modal_backend.subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0
    except FileNotFoundError:
        pass


@pytest.mark.live_modal
def test_guard_allows_modal_control_plane_delegation_when_opted_in():
    client = mock.Mock()
    client._reset_on_pid_change = mock.AsyncMock()
    expected_channel = object()
    client._connection_manager.get_or_create_channel = mock.AsyncMock(return_value=expected_channel)

    channel = asyncio.run(_Client._get_channel(client, "https://example.invalid"))

    assert channel is expected_channel
    client._reset_on_pid_change.assert_awaited_once_with()
    client._connection_manager.get_or_create_channel.assert_awaited_once_with(
        "https://example.invalid"
    )
