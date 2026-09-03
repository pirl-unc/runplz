"""Public-contract tests for shared SSH staging and detached lifecycle."""

import os
import shlex
import signal
import subprocess
import time
from unittest import mock

import pytest

from runplz.backends import ssh_common


def _git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def _init_repo(repo):
    repo.mkdir()
    _git(repo, "init", "--quiet")
    _git(repo, "config", "user.name", "runplz test")
    _git(repo, "config", "user.email", "runplz@example.invalid")


def test_select_source_paths_includes_tracked_and_intentional_untracked_files(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / ".gitignore").write_text("ignored/\ntracked-ignored.txt\n")
    (repo / "tracked.txt").write_text("tracked\n")
    (repo / "tracked-ignored.txt").write_text("still tracked\n")
    (repo / "deleted.txt").write_text("delete me\n")
    (repo / "broken-link").symlink_to("missing-target")
    _git(repo, "add", ".gitignore", "tracked.txt", "deleted.txt", "broken-link")
    _git(repo, "add", "--force", "tracked-ignored.txt")

    (repo / "input.json").write_text("{}\n")
    (repo / "ignored").mkdir()
    (repo / "ignored" / "large-artifact.bin").write_bytes(b"artifact")
    (repo / "deleted.txt").unlink()

    assert set(ssh_common.select_source_paths(repo) or ()) == {
        ".gitignore",
        "broken-link",
        "input.json",
        "tracked-ignored.txt",
        "tracked.txt",
    }


def test_select_source_paths_returns_none_outside_git(tmp_path):
    assert ssh_common.select_source_paths(tmp_path) is None


def test_local_ssh_options_round_trip_and_stale_cleanup(tmp_path):
    host_out = tmp_path / "out"
    options = ssh_common.SshOptions(identity_file="/tmp/key", port=2222)
    ssh_common.write_local_ssh_options(host_out, options)

    assert ssh_common.read_local_ssh_options(host_out) == options
    recorded = host_out / ssh_common.REMOTE_META_DIRNAME / ssh_common.LOCAL_SSH_OPTIONS_FILENAME
    assert recorded.exists()

    ssh_common.write_local_ssh_options(host_out, None)
    assert not recorded.exists()
    assert ssh_common.read_local_ssh_options(host_out) == ssh_common.SshOptions()


def test_local_ssh_options_ignores_corrupt_private_metadata(tmp_path):
    metadata = tmp_path / ssh_common.REMOTE_META_DIRNAME
    metadata.mkdir()
    (metadata / ssh_common.LOCAL_SSH_OPTIONS_FILENAME).write_text("not-json")

    assert ssh_common.read_local_ssh_options(tmp_path) == ssh_common.SshOptions()


def test_select_source_paths_omits_sparse_checkout_entries(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "included").mkdir()
    (repo / "included" / "keep.txt").write_text("keep\n")
    (repo / "omitted").mkdir()
    (repo / "omitted" / "sparse.txt").write_text("absent after sparse checkout\n")
    _git(repo, "add", "included", "omitted")
    _git(repo, "commit", "--quiet", "-m", "sparse fixture")
    _git(repo, "sparse-checkout", "init", "--cone")
    _git(repo, "sparse-checkout", "set", "included")

    assert not (repo / "omitted" / "sparse.txt").exists()
    assert set(ssh_common.select_source_paths(repo) or ()) == {"included/keep.txt"}


def test_select_source_paths_recurses_into_initialized_submodules(tmp_path):
    child = tmp_path / "child"
    _init_repo(child)
    (child / ".gitignore").write_text("ignored/\n")
    (child / "tracked.py").write_text("print('tracked')\n")
    _git(child, "add", ".gitignore", "tracked.py")
    _git(child, "commit", "--quiet", "-m", "child fixture")

    repo = tmp_path / "repo"
    _init_repo(repo)
    _git(
        repo,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        "--quiet",
        str(child),
        "modules/child",
    )
    checkout = repo / "modules" / "child"
    (checkout / "input.json").write_text("{}\n")
    (checkout / "ignored").mkdir()
    (checkout / "ignored" / "checkpoint.pt").write_bytes(b"artifact")

    selected = set(ssh_common.select_source_paths(repo) or ())
    assert "modules/child" not in selected
    assert "modules/child/tracked.py" in selected
    assert "modules/child/input.json" in selected
    assert "modules/child/ignored/checkpoint.pt" not in selected

    _git(repo, "submodule", "deinit", "--force", "--", "modules/child")
    assert not any(
        path.startswith("modules/child/") for path in ssh_common.select_source_paths(repo) or ()
    )


def test_private_module_path_remains_a_compatibility_import():
    from runplz.backends import _ssh_common

    assert _ssh_common.rsync_up is ssh_common.rsync_up
    assert _ssh_common.select_source_paths is ssh_common.select_source_paths


@pytest.mark.parametrize("value", ["relative/path", "$HOME/$(touch pwned)", "/tmp/a b"])
def test_remote_path_validation_rejects_shell_metacharacters(value):
    with pytest.raises(ValueError):
        ssh_common.remote_shell_path(value)


def test_detached_probe_without_events_file_omits_start_marker():
    probe = ssh_common.build_detached_status_probe("$HOME/run/bootstrap.pid")
    assert "remote_command_start" not in probe


@pytest.mark.parametrize(
    "result, expected",
    [
        (
            mock.Mock(returncode=0, stdout="alive", stderr=""),
            ssh_common.DetachedProcessState.RUNNING,
        ),
        (
            mock.Mock(returncode=0, stdout="no-pid", stderr=""),
            ssh_common.DetachedProcessState.MISSING,
        ),
        (
            mock.Mock(returncode=0, stdout="mystery", stderr=""),
            ssh_common.DetachedProcessState.UNKNOWN,
        ),
        (
            mock.Mock(returncode=255, stdout="", stderr="refused"),
            ssh_common.DetachedProcessState.UNKNOWN,
        ),
    ],
)
def test_inspect_detached_run_preserves_transport_and_legacy_states(result, expected):
    with mock.patch("runplz.backends.ssh_common.subprocess.run", return_value=result):
        observed = ssh_common.inspect_detached_run("box", "$HOME/run/pid")
    assert observed.process_state is expected


def test_inspect_detached_run_maps_timeout_to_unknown():
    with mock.patch(
        "runplz.backends.ssh_common.subprocess.run",
        side_effect=subprocess.TimeoutExpired("ssh", 30),
    ):
        observed = ssh_common.inspect_detached_run("box", "$HOME/run/pid")
    assert observed == ssh_common.DetachedRunStatus(ssh_common.DetachedProcessState.UNKNOWN, False)


def test_rsync_up_passes_git_selection_as_nul_delimited_stdin(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / ".gitignore").write_text("ignored/\n")
    (repo / "tracked.py").write_text("print('tracked')\n")
    (repo / "input.csv").write_text("a,b\n")
    (repo / "ignored").mkdir()
    (repo / "ignored" / "checkpoint.pt").write_bytes(b"large")
    _git(repo, "add", ".gitignore", "tracked.py")
    captured = {}

    def fake_sh(cmd, *, stdin=None):
        captured.update(cmd=cmd, stdin=stdin)

    with mock.patch("runplz.backends.ssh_common.run_local", fake_sh):
        ssh_common.rsync_up(repo, "box")

    assert "--from0" in captured["cmd"]
    assert "--files-from=-" in captured["cmd"]
    assert "--recursive" in captured["cmd"]
    assert set(captured["stdin"].rstrip(b"\0").split(b"\0")) == {
        b".gitignore",
        b"input.csv",
        b"tracked.py",
    }


def test_inspect_local_repo_separates_tracked_untracked_and_ignored(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / ".gitignore").write_text("ignored/\n")
    (repo / "tracked.txt").write_text("clean\n")
    _git(repo, "add", ".gitignore", "tracked.txt")
    _git(repo, "commit", "--quiet", "-m", "initial")
    (repo / "input.json").write_text("{}\n")
    (repo / "ignored").mkdir()
    (repo / "ignored" / "run.bin").write_bytes(b"artifact")

    state = ssh_common.inspect_local_repo(repo)
    assert state.revision
    assert state.dirty is False
    assert state.untracked is True
    assert state.ignored is True
    remote_run = ssh_common.make_remote_run_context(
        backend="brev",
        target="box",
        function_name="train",
    )
    manifest = ssh_common.build_remote_run_manifest(
        remote_run=remote_run,
        repo=repo,
        outputs_dir="out",
        args=[],
        kwargs={},
        env={},
    )
    assert manifest["repo_dirty"] is False
    assert manifest["repo_untracked"] is True
    assert manifest["repo_ignored"] is True

    (repo / "tracked.txt").write_text("modified\n")
    assert ssh_common.inspect_local_repo(repo).dirty is True


def test_build_detached_status_probe_checks_for_zombies():
    probe = ssh_common.build_detached_status_probe("$HOME/run/bootstrap.pid", "$HOME/events")
    assert "kill -0" in probe
    assert "/proc/$runplz_pid/stat" in probe
    assert "ps -o stat=" not in probe
    assert "runplz_proc_tail=${runplz_proc_stat##*) }" in probe
    assert 'Z) runplz_state="zombie"' in probe
    assert "remote_command_start" in probe


def test_detached_status_probe_runs_without_procps(tmp_path):
    pid_file = tmp_path / "bootstrap.pid"
    pid_file.write_text(str(os.getpid()))
    probe = ssh_common.build_detached_status_probe(str(pid_file))
    result = subprocess.run(
        ["/bin/sh", "-c", probe],
        check=True,
        capture_output=True,
        text=True,
        env={"PATH": str(tmp_path / "empty-path")},
    )
    assert result.stdout.strip() == f"0 running {os.getpid()}"


def test_inspect_detached_run_reports_zombie_and_start_event():
    result = mock.Mock(returncode=0, stdout="1 zombie 4321\n", stderr="")
    with mock.patch("runplz.backends.ssh_common.subprocess.run", return_value=result):
        status = ssh_common.inspect_detached_run(
            "box",
            "$HOME/bootstrap.pid",
            events_file="$HOME/events.ndjson",
        )
        assert ssh_common.remote_pid_alive("box", "$HOME/bootstrap.pid") is False
    assert status == ssh_common.DetachedRunStatus(
        ssh_common.DetachedProcessState.ZOMBIE,
        True,
        4321,
    )


def test_wait_for_detached_start_returns_promptly_for_zombie():
    zombie = ssh_common.DetachedRunStatus(
        ssh_common.DetachedProcessState.ZOMBIE,
        False,
        4321,
    )
    with mock.patch("runplz.backends.ssh_common.inspect_detached_run", return_value=zombie):
        with mock.patch("runplz.backends.ssh_common.time.sleep") as sleep:
            status = ssh_common.wait_for_detached_start(
                "box",
                "$HOME/bootstrap.pid",
                "$HOME/events.ndjson",
            )
    assert status is zombie
    sleep.assert_not_called()


def test_wait_for_detached_start_returns_unknown_at_deadline():
    unknown = ssh_common.DetachedRunStatus(
        ssh_common.DetachedProcessState.UNKNOWN,
        False,
    )
    with mock.patch(
        "runplz.backends.ssh_common.inspect_detached_run", return_value=unknown
    ) as inspect:
        with mock.patch("runplz.backends.ssh_common.time.monotonic", side_effect=[0.0, 2.0]):
            with mock.patch("runplz.backends.ssh_common.time.sleep") as sleep:
                status = ssh_common.wait_for_detached_start(
                    "box",
                    "/run/bootstrap.pid",
                    "/run/events.ndjson",
                    timeout_s=1.0,
                )
    assert status is unknown
    inspect.assert_called_once_with(
        "box", "/run/bootstrap.pid", events_file="/run/events.ndjson", ssh_opts=None
    )
    sleep.assert_not_called()


def test_detached_launch_diagnostics_observes_remote_command():
    remote_run = ssh_common.make_remote_run_context(
        backend="brev", target="box", function_name="train"
    )
    capture = mock.Mock(return_value="detached bootstrap diagnostics:\nprocess state: dead\n")
    with mock.patch("runplz.backends.ssh_common.ssh_capture", capture):
        result = ssh_common.detached_launch_diagnostics("box", remote_run)
    assert "process state: dead" in result
    capture.assert_called_once()
    command = capture.call_args.args[1]
    assert "detached bootstrap diagnostics:" in command
    assert "run_driver.log" in command


def test_detached_launcher_ignores_hup_before_spawn_and_in_child(tmp_path):
    remote_run = ssh_common.make_remote_run_context(
        backend="brev",
        target="box",
        function_name="train",
    )
    marker = tmp_path / "survived-hup"
    launcher = ssh_common.build_detached_launcher(
        remote_run,
        f"sleep 1; printf survived > {shlex.quote(str(marker))}",
    )

    spawn_offset = launcher.index('bash "' + remote_run.meta_shell + '/run.sh"')
    assert launcher.count("trap '' HUP") == 2
    assert launcher.index("trap '' HUP") < spawn_offset
    assert launcher.rindex("trap '' HUP") < spawn_offset

    env = dict(os.environ, HOME=str(tmp_path))
    subprocess.run(["bash", "-c", launcher], check=True, env=env)
    pid_file = tmp_path / "runplz-runs" / remote_run.run_id / "out" / ".runplz" / "bootstrap.pid"
    pid = int(pid_file.read_text().strip())
    os.kill(pid, signal.SIGHUP)

    deadline = time.monotonic() + 5
    while not marker.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert marker.read_text() == "survived"


def test_launch_detached_failure_returns_nonzero_with_diagnostics(capsys):
    remote_run = ssh_common.make_remote_run_context(
        backend="brev",
        target="box",
        function_name="train",
    )
    zombie = ssh_common.DetachedRunStatus(
        ssh_common.DetachedProcessState.ZOMBIE,
        False,
        4321,
    )
    with mock.patch("runplz.backends.ssh_common.ssh_exec"):
        with mock.patch("runplz.backends.ssh_common.wait_for_detached_start", return_value=zombie):
            with mock.patch("runplz.backends.ssh_common._record_remote_event") as record:
                with mock.patch(
                    "runplz.backends.ssh_common.detached_launch_diagnostics",
                    return_value="driver log:\nnohup failed",
                ):
                    result = ssh_common.launch_detached_and_wait(
                        target="box",
                        wrapped_command="echo hi",
                        remote_run=remote_run,
                    )
    assert result == 1
    assert record.call_args.args[2] == "bootstrap_launch_failed"
    output = capsys.readouterr().out
    assert "state=zombie" in output
    assert "nohup failed" in output


def test_launch_detached_unknown_startup_enters_resilient_monitoring(capsys):
    remote_run = ssh_common.make_remote_run_context(
        backend="brev",
        target="box",
        function_name="train",
    )
    unknown = ssh_common.DetachedRunStatus(
        ssh_common.DetachedProcessState.UNKNOWN,
        False,
    )
    with mock.patch("runplz.backends.ssh_common.ssh_exec"):
        with mock.patch("runplz.backends.ssh_common.wait_for_detached_start", return_value=unknown):
            with mock.patch(
                "runplz.backends.ssh_common.tail_and_wait_for_detached",
                return_value=23,
            ) as monitor:
                with mock.patch("runplz.backends.ssh_common._record_remote_event") as record:
                    result = ssh_common.launch_detached_and_wait(
                        target="box",
                        wrapped_command="echo hi",
                        remote_run=remote_run,
                    )
    assert result == 23
    monitor.assert_called_once()
    record.assert_not_called()
    assert "entering resilient monitoring" in capsys.readouterr().out


def test_build_detached_log_command_stops_tail_for_zombie():
    command = ssh_common.build_detached_log_command("$HOME/bootstrap.pid", "$HOME/last.log")
    assert 'tail -n +1 -F "$HOME/last.log"' in command
    assert "/proc/$runplz_pid/stat" in command
    assert "ps -o stat=" not in command
    assert "missing|dead|zombie" in command
    assert "runplz_stop_tail" in command


def _spawn_line(launcher):
    """The line that actually backgrounds the bootstrap.

    Identified by the run script it launches, not by whichever tool happens to
    prefix the command -- `nohup` is now conditional, and a test that greps for
    it matches the probe instead of the spawn.
    """
    return next(line for line in launcher.splitlines() if '/run.sh"' in line and line.endswith("&"))


def test_a_refusing_nohup_does_not_stop_the_bootstrap(tmp_path):
    """Issue #92. macOS `nohup` refuses to detach in a non-interactive ssh
    session, and because bash forks the job before nohup execs and fails, the
    pid file is still written -- of a process that never ran. The run then
    looks exactly like a payload that crashed instantly.

    CI has no macOS runner, so the refusal is reproduced instead of the
    platform: a `nohup` on PATH that fails the way macOS fails. This test fails
    on the pre-4.0.3 launcher for the real reason -- no marker, because the
    payload never ran.
    """
    stub_dir = tmp_path / "stub-bin"
    stub_dir.mkdir()
    nohup = stub_dir / "nohup"
    nohup.write_text(
        "#!/bin/sh\n"
        'echo "nohup: can\'t detach from console: Inappropriate ioctl for device" >&2\n'
        "exit 1\n"
    )
    nohup.chmod(0o755)

    remote_run = ssh_common.make_remote_run_context(
        backend="ssh", target="box", function_name="train"
    )
    marker = tmp_path / "ran-anyway"
    launcher = ssh_common.build_detached_launcher(
        remote_run, f"printf ran > {shlex.quote(str(marker))}"
    )

    env = dict(os.environ, HOME=str(tmp_path), PATH=f"{stub_dir}{os.pathsep}{os.environ['PATH']}")
    subprocess.run(["bash", "-c", launcher], check=True, env=env)

    deadline = time.monotonic() + 5
    while not marker.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert marker.read_text() == "ran", "the bootstrap never ran when nohup refused"


def test_the_bootstrap_still_runs_under_a_working_nohup(tmp_path):
    """The control. Without it the test above would also pass if the probe had
    simply disabled the spawn entirely."""
    remote_run = ssh_common.make_remote_run_context(
        backend="ssh", target="box", function_name="train"
    )
    marker = tmp_path / "ran"
    launcher = ssh_common.build_detached_launcher(
        remote_run, f"printf ran > {shlex.quote(str(marker))}"
    )

    subprocess.run(["bash", "-c", launcher], check=True, env=dict(os.environ, HOME=str(tmp_path)))

    deadline = time.monotonic() + 5
    while not marker.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert marker.read_text() == "ran"


def test_the_run_id_is_on_the_spawn_line_whichever_way_nohup_resolves():
    """`/proc/<pid>/environ` is fixed at exec time, so the assignment has to be
    on the command that execs -- and there is now only one such line, rather
    than one per branch that could drift from the other."""
    remote_run = ssh_common.make_remote_run_context(
        backend="ssh", target="box", function_name="train"
    )
    launcher = ssh_common.build_detached_launcher(remote_run, "echo hi")
    spawn = _spawn_line(launcher)
    assert spawn.startswith(f"{ssh_common.RUN_ID_ENV_VAR}={remote_run.run_id} ")
    assert "${runplz_nohup}" in spawn
    assert launcher.count('/run.sh" </dev/null') == 1, "the spawn must not be duplicated"
