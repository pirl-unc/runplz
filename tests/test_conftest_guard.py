"""Proves the autouse guard refuses real provider CLI and SDK launches.

Issues #35 and #170.
"""

import asyncio
import shlex
import subprocess
import sys
from pathlib import Path
from subprocess import run as bare_run
from types import SimpleNamespace
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
        # env's own options: ones that re-tokenize (-S), ones that change
        # lookup (-u PATH), and one nobody has invented yet.
        ["-S", "modal run job.py::main"],
        ["--split-string", "modal run job.py::main"],
        ["--split-string=modal run job.py::main"],
        ["-u", "PATH", "modal", "run", "job.py::main"],
        ["--future-option", "modal", "run", "job.py::main"],
        # Operands: a bare command, assignments (including a name outside
        # shell-identifier syntax), a `--` terminator, a repeated wrapper, and
        # an assignment that rewrites the PATH the child will search.
        ["modal", "run", "job.py::main"],
        ["RUNPLZ_GUARD_TEST=1", "modal", "run", "job.py::main"],
        ["A.B=x", "modal", "run", "job.py::main"],
        ["--", "modal", "run", "job.py::main"],
        ["OUTER=1", "env", "INNER=2", "modal", "run", "job.py::main"],
        ["PATH=/somewhere/else", "modal", "run", "job.py::main"],
    ],
)
def test_guard_blocks_env_wrapped_modal_however_it_is_spelled(sandbox_bin, env_args):
    """One rule, no `env` parser: the exemption covers the launched program.

    Behind `env` that program is `env`, so the sandbox stub cannot stand in for
    `modal` no matter which operand or option spelling puts it there — including
    the ones that re-tokenize or rewrite PATH, which no operand parser of ours
    would have modelled correctly anyway.
    """
    marker = _sandboxed_modal_that_records_running(sandbox_bin, "env-wrapped-ran")

    with pytest.raises(RuntimeError, match="tried to run `modal`"):
        modal_backend.subprocess.run(["env", *env_args], check=True)

    assert not marker.exists()


def test_guard_allows_explicit_modal_cli_mock():
    result = mock.Mock(returncode=0)
    with mock.patch.object(modal_backend.subprocess, "run", return_value=result) as run:
        assert modal_backend.subprocess.run(["modal", "run", "job.py::main"]) is result
    run.assert_called_once_with(["modal", "run", "job.py::main"])


def test_guard_allows_sandboxed_modal_cli(sandbox_bin):
    executable = sandbox_bin / "modal"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)

    # `env modal ...` is deliberately absent: the exemption covers the program
    # actually launched, and behind a wrapper that is the wrapper.
    for command in (
        ["modal", "run", "fake.py::main"],
        [str(executable), "run", "fake.py::main"],
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
    from conftest import _ORIGINAL_POPEN_INIT

    assert subprocess.Popen.__init__ is not _ORIGINAL_POPEN_INIT  # guard installed
    # This just needs to not raise our RuntimeError.
    try:
        brev.subprocess.run(
            ["docker", "image", "inspect", "nonexistent:tag"],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        pass  # no docker on this host — still proves the guard didn't block it


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


@pytest.mark.parametrize(
    "cmd",
    [
        "/usr/bin/modal run 'unclosed",  # shlex cannot split an unbalanced quote
        "'modal' run 'unclosed",  # ...and the name is quoted, so splitting on
        "'modal' run job.py",  # whitespace alone would not reveal it either
    ],
)
def test_guard_reads_a_string_command_shlex_cannot_split(cmd):
    # Treating an unsplittable string as "no tokens" delegated straight to the
    # real subprocess, where Windows' own parser would have found `modal`.
    with pytest.raises(RuntimeError, match="tried to run `modal`"):
        modal_backend.subprocess.run(cmd)


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


# --- Regression tests for the second review round --------------------------


@pytest.mark.parametrize("api", ["getoutput", "getstatusoutput"])
def test_guard_covers_the_shell_helpers(api):
    # These run a command line through a shell. Hooking Popen sees them arrive
    # as `shell=True`, so they get the shell refusal rather than a tokenizing
    # of shell source that `cd /tmp;modal run job` would have slipped past.
    with pytest.raises(RuntimeError, match="shell=True"):
        getattr(modal_backend.subprocess, api)("modal run job.py::main")


def test_guard_reads_a_bytes_command_line():
    # The `str` branch splits because Windows parses a string as a command
    # line; the same parsing applies to bytes, which used to arrive as one
    # undecoded token that hid every billed name.
    with pytest.raises(RuntimeError, match="tried to run `modal`"):
        modal_backend.subprocess.run(b"modal run job.py")


@pytest.mark.parametrize(
    "cmd",
    [
        ["sh", "-c", "modal run job.py::main"],
        ["bash", "-lc", "cd /tmp && modal run job.py::main"],
        [sys.executable, "-c", "import modal; modal.Function.from_name('a', 'b').spawn()"],
    ],
)
def test_guard_reads_commands_hidden_inside_an_argument(cmd):
    # An argument can itself be a command line. Reading the words of every
    # token catches that without the guard knowing which programs re-tokenize.
    with pytest.raises(RuntimeError, match="tried to run `modal`"):
        modal_backend.subprocess.run(cmd)


def test_guard_classifies_popen_args_passed_by_keyword():
    # `args` is Popen's documented parameter name, so this spelling is legal
    # and used to die on a TypeError naming a parameter the caller never used.
    with pytest.raises(RuntimeError, match="tried to run `modal`"):
        modal_backend.subprocess.Popen(args=["modal", "run", "job.py::main"])


# --- Regression tests for the third review round ---------------------------


def _classify(cmd, *markers, **popen_kwargs):
    """Run the guard's classifier alone, for cases a real spawn would need a
    live marker or a tool for. Raises exactly as the hook would."""
    from conftest import _reject_billed_commands

    node = SimpleNamespace(nodeid="classification", get_closest_marker=lambda m: m in markers)
    _reject_billed_commands(SimpleNamespace(node=node), (cmd,), popen_kwargs)


def test_the_seam_is_popen_so_every_spelling_of_a_spawn_is_guarded():
    # Hooking each module's `subprocess` name left every other route open:
    # a from-import, an alias, a test helper, asyncio. Popen is the one seam.
    cmd = ["modal", "run", "job.py::main"]
    with pytest.raises(RuntimeError, match="tried to run `modal`"):
        bare_run(cmd)
    with pytest.raises(RuntimeError, match="tried to run `modal`"):
        subprocess.check_output(cmd)
    with pytest.raises(RuntimeError, match="tried to run `modal`"):
        subprocess.Popen(args=cmd)
    with pytest.raises(RuntimeError, match="tried to run `modal`"):
        asyncio.run(asyncio.create_subprocess_exec(*cmd))


@pytest.mark.parametrize(
    "cmd",
    [
        ["uvx", "modal==1.5", "run", "job.py"],
        ["uvx", "modal[aws]", "run", "job.py"],
        ["uvx", "modal@latest", "run", "job.py"],
        ["pipx", "run", "modal>=1", "run", "job.py"],
        ["uv", "tool", "run", "modal~=1.5", "deploy", "s.py"],
    ],
)
def test_guard_reads_a_billed_name_through_a_version_specifier(cmd):
    # Install-and-run wrappers resolve the console script in a fresh env and
    # run it for real; the specifier glued to the name hid it from the scan.
    with pytest.raises(RuntimeError, match="tried to run `modal`"):
        subprocess.run(cmd)


@pytest.mark.parametrize(
    "cmd",
    [
        ["env", "-Smodal run job.py::main"],  # env's glued -S<string>
        ["sh", "-c", "cd /tmp;modal run job.py"],  # separator, plus a `/`
        ["sh", "-c", "true&&modal run job.py"],
        ["sh", "-c", "true|modal run job.py"],
        [sys.executable, "-c", "import modal;modal.run()"],
        ["Modal", "run", "job.py"],  # case-insensitive filesystem
        ["MODAL", "run", "job.py"],
    ],
)
def test_guard_reads_a_billed_name_glued_to_a_separator_or_case(cmd):
    with pytest.raises(RuntimeError, match="tried to run `modal`"):
        subprocess.run(cmd)


def test_guard_blocks_the_modal_worker_spelled_as_a_file_path():
    # The `-m` spelling was refused; the same worker by path was not.
    with pytest.raises(RuntimeError, match="tried to run `modal`"):
        subprocess.run([sys.executable, modal_runs.__file__, "probe", "/tmp/out"])


@pytest.mark.parametrize(
    "cmd",
    [
        ["runplz", "run", "job.py"],
        [sys.executable, "-m", "runplz.cli", "run", "job.py"],
        ["uv", "run", "python", "-m", "runplz.cli", "run", "job.py"],
        [sys.executable, str(Path(runplz.__file__).parent / "cli.py"), "run", "job.py"],
    ],
)
def test_a_child_running_the_cli_is_refused_whatever_the_marker(cmd):
    # It can dispatch to any backend, so no one marker can vouch for it.
    for markers in ((), ("live_modal",), ("live_brev", "live_ssh")):
        with pytest.raises(RuntimeError, match="`runplz` CLI in a child process"):
            _classify(cmd, *markers)


@pytest.mark.parametrize(
    "cmd, markers",
    [
        # runplz's own remote-command strings and docker argv name its
        # bootstrap module, which reaches no provider.
        (["ssh", "box", "bash -lc 'python -m runplz._bootstrap > last.log 2>&1'"], ("live_ssh",)),
        (["ssh", "box", "pkill -f 'runplz._bootstrap' || true"], ("live_ssh",)),
        (["docker", "run", "img", "python", "-m", "runplz._bootstrap"], ()),
        ([sys.executable, "-m", "runplz.bootstrap"], ()),
        # Product-embedded user data and ordinary filenames are not launches.
        (["docker", "run", "-e", 'RUNPLZ_KWARGS={"cfg": "aws.yaml"}', "img"], ()),
        (["cat", "aws.json"], ()),
        (["git", "config", "user.name", "runplz test"], ()),
        (["gcloud", "compute", "instances", "create", "box", "--labels=runplz=1"], ("live_gcp",)),
        (["docker", "build", "-t", "runplz.local:latest", "."], ()),
        (["git", "commit", "-m", "fix modal.py"], ()),
        # A short-option cluster is not `-m` unless a module name follows it.
        (["rsync", "-avzm", "src/", "dest/"], ("live_ssh",)),
        (["tar", "-xzmf", "a.tgz"], ()),
        # `-c` is not itself suspicious.
        ([sys.executable, "-c", "import time; print(time.strftime('%Y'))"], ()),
    ],
)
def test_guard_does_not_bill_words_that_merely_resemble_a_launch(cmd, markers):
    _classify(cmd, *markers)  # must not raise


def test_executable_override_can_grant_the_exemption_too(sandbox_bin):
    # POSIX execs `executable=`; argv[0] is then only the child's own name.
    stub = sandbox_bin / "modal"
    stub.write_text("#!/bin/sh\nexit 0\n")
    stub.chmod(0o755)

    result = subprocess.run(["modal", "run", "job.py"], executable=str(stub))

    assert result.returncode == 0


def test_real_child_processes_is_the_only_way_to_spawn_our_worker(real_child_processes):
    # The escape hatch exists for the worker tests; here it proves it is
    # scoped to the test that asks for it, by being the one that asks.
    result = subprocess.run(
        [sys.executable, "-m", "runplz.backends.modal_runs", "probe", "/nonexistent"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2  # rejected the receipt, offline
