"""Public-contract tests for shared SSH staging and detached lifecycle."""

import subprocess
from unittest import mock

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
    _git(repo, "add", ".gitignore", "tracked.txt", "deleted.txt")
    _git(repo, "add", "--force", "tracked-ignored.txt")

    (repo / "input.json").write_text("{}\n")
    (repo / "ignored").mkdir()
    (repo / "ignored" / "large-artifact.bin").write_bytes(b"artifact")
    (repo / "deleted.txt").unlink()

    assert set(ssh_common.select_source_paths(repo) or ()) == {
        ".gitignore",
        "input.json",
        "tracked-ignored.txt",
        "tracked.txt",
    }


def test_select_source_paths_returns_none_outside_git(tmp_path):
    assert ssh_common.select_source_paths(tmp_path) is None


def test_private_module_path_remains_a_compatibility_import():
    from runplz.backends import _ssh_common

    assert _ssh_common.rsync_up is ssh_common.rsync_up
    assert _ssh_common.select_source_paths is ssh_common.select_source_paths


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

    with mock.patch("runplz.backends.ssh_common._sh", fake_sh):
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
    assert "ps -o stat=" in probe
    assert "Z*" in probe
    assert "remote_command_start" in probe


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
    with mock.patch("runplz.backends.ssh_common._ssh"):
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


def test_build_detached_log_command_stops_tail_for_zombie():
    command = ssh_common.build_detached_log_command("$HOME/bootstrap.pid", "$HOME/last.log")
    assert 'tail -n +1 -F "$HOME/last.log"' in command
    assert "ps -o stat=" in command
    assert 'Z*|"") break' in command
    assert "runplz_stop_tail" in command
