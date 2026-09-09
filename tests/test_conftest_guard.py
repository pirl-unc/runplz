"""Proves the autouse guard refuses real provider CLI and SDK launches.

Issues #35 and #170.
"""

import ast
import asyncio
import shlex
import sys
from pathlib import Path
from unittest import mock

import modal as modal_sdk
import pytest
from modal.client import _Client

import runplz
from runplz.backends import brev, modal_runs, provisioning, ssh_common
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
        ["env", sys.executable, "-u", "-m", "modal", "run", "job.py::main"],
        f"{shlex.quote(sys.executable)} -m modal deploy service.py",
    ],
)
def test_guard_blocks_real_modal_cli_launches(cmd):
    with pytest.raises(RuntimeError, match="tried to run `modal`"):
        modal_backend.subprocess.run(cmd, check=True)


def test_guard_rejects_shell_commands_before_execution(sandbox_bin):
    marker = sandbox_bin.parent / "shell-command-ran"
    executable = sandbox_bin / "modal"
    executable.write_text(f"#!/bin/sh\ntouch {shlex.quote(str(marker))}\n")
    executable.chmod(0o755)

    with pytest.raises(RuntimeError, match="shell=True"):
        modal_backend.subprocess.run(
            "cd /tmp && modal run job.py::main",
            shell=True,
            check=True,
        )

    assert not marker.exists()


def _sandboxed_modal_that_records_running(sandbox_bin, marker_name):
    marker = sandbox_bin.parent / marker_name
    executable = sandbox_bin / "modal"
    executable.write_text(f"#!/bin/sh\ntouch {shlex.quote(str(marker))}\n")
    executable.chmod(0o755)
    return marker


@pytest.mark.parametrize(
    "env_args",
    [
        ["-S", "modal run job.py::main"],
        ["--split-string", "modal run job.py::main"],
        ["--split-string=modal run job.py::main"],
    ],
)
def test_guard_rejects_env_options_that_hide_the_command(sandbox_bin, env_args):
    # `-S` re-tokenizes its argument into a command the guard never gets to
    # read, so there is no billed name to report — only the opaque layer.
    marker = _sandboxed_modal_that_records_running(sandbox_bin, "env-option-ran")

    with pytest.raises(RuntimeError, match="env options"):
        modal_backend.subprocess.run(["env", *env_args], check=True)

    assert not marker.exists()


@pytest.mark.parametrize(
    "env_args",
    [
        ["-u", "RUNPLZ_GUARD_TEST", "modal", "run", "job.py::main"],
        ["--future-option", "modal", "run", "job.py::main"],
    ],
)
def test_env_options_cannot_buy_a_sandbox_exemption(sandbox_bin, env_args):
    # Here the billed name is visible, so the guard names it. The option is
    # still what makes it unsafe: `env -u PATH modal` resolves `modal` from a
    # PATH we do not control, so the sandbox stub cannot stand in for it.
    marker = _sandboxed_modal_that_records_running(sandbox_bin, "env-option-ran")

    with pytest.raises(RuntimeError, match="tried to run `modal`"):
        modal_backend.subprocess.run(["env", *env_args], check=True)

    assert not marker.exists()


def test_guard_uses_executable_override_as_the_launched_program(tmp_path):
    explicitly_invoked = tmp_path / "modal"
    explicitly_invoked.write_text("#!/bin/sh\nexit 0\n")
    explicitly_invoked.chmod(0o755)

    with pytest.raises(RuntimeError, match="tried to run `modal`"):
        modal_backend.subprocess.run(
            ["harmless-argv-zero", "run", "job.py::main"],
            executable=explicitly_invoked,
        )


def test_guard_classifies_pathlike_and_space_containing_executables(tmp_path):
    outside_bin = tmp_path / "outside bin"
    outside_bin.mkdir()
    explicitly_invoked = outside_bin / "modal"
    explicitly_invoked.write_text("#!/bin/sh\nexit 0\n")
    explicitly_invoked.chmod(0o755)

    for command in (explicitly_invoked, str(explicitly_invoked)):
        with pytest.raises(RuntimeError, match="tried to run `modal`"):
            modal_backend.subprocess.run(command)


def test_guard_allows_sandboxed_executable_override(sandbox_bin):
    executable = sandbox_bin / "modal"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)

    result = modal_backend.subprocess.run(
        ["harmless-argv-zero", "run", "fake.py::main"],
        executable=executable,
    )

    assert result.returncode == 0


@pytest.mark.parametrize("separator", [[], ["--"]], ids=["plain", "after-double-dash"])
def test_guard_treats_every_env_name_value_operand_as_an_assignment(tmp_path, separator):
    explicitly_invoked = tmp_path / "modal"
    explicitly_invoked.write_text("#!/bin/sh\nexit 0\n")
    explicitly_invoked.chmod(0o755)

    with pytest.raises(RuntimeError, match="tried to run `modal`"):
        modal_backend.subprocess.run(
            ["env", *separator, "A.B=x", str(explicitly_invoked), "run", "job.py::main"]
        )


def test_guard_peels_repeated_env_wrappers(tmp_path):
    explicitly_invoked = tmp_path / "modal"
    explicitly_invoked.write_text("#!/bin/sh\nexit 0\n")
    explicitly_invoked.chmod(0o755)

    with pytest.raises(RuntimeError, match="tried to run `modal`"):
        modal_backend.subprocess.run(
            [
                "env",
                "OUTER=value",
                "env",
                "INNER=value",
                str(explicitly_invoked),
                "run",
                "job.py::main",
            ]
        )


def test_subprocess_env_path_controls_sandbox_resolution(sandbox_bin):
    sandboxed = sandbox_bin / "modal"
    sandboxed.write_text("#!/bin/sh\nexit 0\n")
    sandboxed.chmod(0o755)

    outside_bin = sandbox_bin.parent / "outside-path"
    outside_bin.mkdir()
    outside = outside_bin / "modal"
    outside.write_text("#!/bin/sh\nexit 0\n")
    outside.chmod(0o755)

    with pytest.raises(RuntimeError, match="tried to run `modal`"):
        modal_backend.subprocess.run(
            ["modal", "run", "job.py::main"],
            env={"PATH": str(outside_bin)},
        )


def test_env_path_assignment_cannot_reuse_outer_sandbox_resolution(sandbox_bin):
    sandboxed = sandbox_bin / "modal"
    sandboxed.write_text("#!/bin/sh\nexit 0\n")
    sandboxed.chmod(0o755)

    outside_bin = sandbox_bin.parent / "outside-env-path"
    outside_bin.mkdir()
    outside = outside_bin / "modal"
    outside.write_text("#!/bin/sh\nexit 0\n")
    outside.chmod(0o755)

    with pytest.raises(RuntimeError, match="tried to run `modal`"):
        modal_backend.subprocess.run(["env", f"PATH={outside_bin}", "modal", "run", "job.py::main"])


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


# --- Regression tests for the guard bypasses found in review ---------------
#
# Each of these ran for real against the previous classifier, which tried to
# identify *the* program by peeling `env` wrappers and parsing interpreter
# flags. Every spelling those two tables did not know about was a silent
# allow, which is why the guard now scans all of argv instead.


@pytest.mark.parametrize(
    "cmd",
    [
        # Clustered short options: CPython reads the first `m` in the cluster
        # as `-m`, so all of these really do execute the module.
        [sys.executable, "-um", "modal", "run", "job.py::main"],
        [sys.executable, "-Bm", "modal", "run", "job.py::main"],
        # A submodule entry point is the same SDK, spelled around an exact
        # string comparison against "modal".
        [sys.executable, "-m", "modal.cli.entry_point", "run", "job.py"],
        [sys.executable, "-mmodal.cli.entry_point", "run", "job.py"],
        # Wrappers. The old guard peeled `env` and nothing else, so each of
        # these was classified as `uv`/`timeout`/`nohup`/`sudo` and allowed.
        ["uv", "run", "modal", "run", "job.py::main"],
        ["uvx", "modal", "run", "job.py::main"],
        ["timeout", "600", "modal", "run", "job.py::main"],
        ["nohup", "modal", "run", "job.py::main"],
        ["sudo", "-E", "modal", "run", "job.py::main"],
        ["stdbuf", "-oL", "modal", "run", "job.py::main"],
        ["xargs", "-0", "modal", "run"],
    ],
)
def test_guard_blocks_modal_behind_flag_spellings_and_wrappers(cmd):
    with pytest.raises(RuntimeError, match="tried to run `modal`"):
        modal_backend.subprocess.run(cmd, check=True)


def test_guard_fails_closed_on_an_untokenizable_string_command():
    # shlex cannot split an unbalanced quote. Treating that as "no tokens"
    # delegated straight to the real subprocess, where Windows' own parser
    # would have found `modal` again.
    with pytest.raises(RuntimeError, match="cannot tokenize"):
        modal_backend.subprocess.run("/usr/bin/modal run 'unclosed")


def test_guard_reads_shell_passed_positionally(sandbox_bin):
    # run(*popenargs) forwards positionals to Popen, where shell is the 8th.
    # Reading shell only from kwargs meant a shell string reached the real
    # subprocess; a compound one hid `modal` behind an argv[0] of `cd`.
    marker = _sandboxed_modal_that_records_running(sandbox_bin, "positional-shell-ran")

    with pytest.raises(RuntimeError, match="shell=True"):
        modal_backend.subprocess.run(
            "cd /tmp && modal run job.py::main", -1, None, None, None, None, None, True, True
        )

    assert not marker.exists()


def test_guard_reads_executable_passed_positionally(tmp_path):
    explicitly_invoked = tmp_path / "modal"
    explicitly_invoked.write_text("#!/bin/sh\nexit 0\n")
    explicitly_invoked.chmod(0o755)

    with pytest.raises(RuntimeError, match="tried to run `modal`"):
        modal_backend.subprocess.run(["harmless-argv-zero"], -1, str(explicitly_invoked))


@pytest.mark.parametrize("api", ["Popen", "call", "check_call", "check_output"])
def test_guard_covers_every_process_starting_entry_point(api):
    # Guarding `run` alone left four documented ways to reach the same binary.
    with pytest.raises(RuntimeError, match="tried to run `modal`"):
        getattr(modal_backend.subprocess, api)(["modal", "run", "job.py::main"])


def test_sandbox_exemption_follows_the_childs_working_directory(sandbox_bin, tmp_path, monkeypatch):
    sandboxed = sandbox_bin / "modal"
    sandboxed.write_text("#!/bin/sh\nexit 0\n")
    sandboxed.chmod(0o755)
    real_tools = tmp_path / "real-tools"
    real_tools.mkdir()
    outside = real_tools / "modal"
    outside.write_text("#!/bin/sh\nexit 0\n")
    outside.chmod(0o755)
    monkeypatch.chdir(sandbox_bin)

    # Control: with no cwd= the relative path really is the sandboxed stub.
    assert modal_backend.subprocess.run(["./modal", "run", "job.py"]).returncode == 0

    # cwd= is where the *child* resolves "./modal", so the same argv now names
    # the real binary. Resolving it against our own cwd granted the exemption
    # while a different file got executed.
    with pytest.raises(RuntimeError, match="tried to run `modal`"):
        modal_backend.subprocess.run(["./modal", "run", "job.py"], cwd=str(real_tools))


def test_unwritten_sandbox_stub_does_not_earn_an_exemption(sandbox_bin):
    # `shutil.which` checks executability for bare names; the path branch
    # stopped doing so, so a stub a test forgot to chmod looked sandboxed.
    not_executable = sandbox_bin / "modal"
    not_executable.write_text("#!/bin/sh\nexit 0\n")

    with pytest.raises(RuntimeError, match="tried to run `modal`"):
        modal_backend.subprocess.run([str(not_executable), "run", "job.py::main"])


def test_guard_blocks_the_modal_worker_child():
    # The worker is a fresh interpreter that builds a real Modal client, so
    # the in-process SDK guard cannot see it. It has to be stopped here.
    with pytest.raises(RuntimeError, match="tried to run `modal`"):
        modal_runs.subprocess.run(
            [sys.executable, "-m", "runplz.backends.modal_runs", "probe", "/tmp/out"]
        )


def test_modal_worker_module_is_guarded():
    # The exact assertion that showed the module was missing from the list.
    assert type(modal_runs.subprocess).__name__ == "_GuardedSubprocessModule"


def test_every_module_that_shells_out_is_guarded():
    """Close the class of bug, not the one instance of it.

    `runplz.backends.modal_runs` called `subprocess.run` while absent from
    `_MODULES_TO_GUARD`, so its commands never reached the guard at all. The
    next module to shell out must not be able to repeat that silently.
    """
    from conftest import _MODULES_TO_GUARD

    package = Path(runplz.__file__).parent
    shells_out = set()
    for path in package.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import) and any(a.name == "subprocess" for a in node.names):
                shells_out.add(".".join(path.relative_to(package.parent).with_suffix("").parts))

    assert shells_out, "expected to find modules importing subprocess"
    assert not shells_out - set(_MODULES_TO_GUARD), (
        "these modules call subprocess but are not in _MODULES_TO_GUARD: "
        f"{sorted(shells_out - set(_MODULES_TO_GUARD))}"
    )


def test_control_plane_guard_is_actually_installed():
    # The SDK guard used to `return` on ImportError. `modal` is a hard runtime
    # dependency, so the only way that fired was the SDK moving `_Client` --
    # which would have left the whole suite unguarded and still green.
    assert _Client._get_channel.__name__ == "guarded_get_channel"


def test_guard_allows_looking_a_billed_cli_up_without_running_it():
    # `_require_brev_cli` runs this as a precondition check; `which` never
    # executes its operand, so blocking it would be a false positive.
    result = brev.subprocess.run(["which", "brev"], capture_output=True)
    assert result.returncode in (0, 1)
