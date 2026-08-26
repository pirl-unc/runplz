"""Coverage for ``runplz kill`` / ``runplz cancel`` (issue #67)."""

import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from unittest import mock

import pytest

from runplz import _cli, _runs
from runplz.backends import ssh_common

HAS_PROCFS = Path("/proc/self/stat").is_file()


def _write_manifest(outputs_dir: Path, **overrides) -> dict:
    manifest = {
        "run_id": "20260826T010203Z-myhost-train-deadbeef",
        "backend": "brev",
        "target": "my-gpu-box",
        "function": "train",
        "remote_paths": {
            "meta": "~/runplz-runs/20260826T010203Z-myhost-train-deadbeef/out/.runplz",
        },
    }
    manifest.update(overrides)
    meta = outputs_dir / ".runplz"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "run.json").write_text(json.dumps(manifest))
    return manifest


def _summary(**fields) -> str:
    base = {
        "pid": "4242",
        "pgid": "4242",
        "container": "",
        "container_state": "none",
        "initial": "running",
        "final": "dead",
        "signalled": "1",
        "escalated": "0",
        "gpu_mem_used": "",
    }
    base.update(fields)
    body = "".join(f"{k}={v}\n" for k, v in base.items())
    return f"---SUMMARY---\n{body}---HEARTBEAT---\n---LOGTAIL---\n---END---\n"


# ---------------------------------------------------------------------------
# build_kill_command


def test_kill_command_reads_every_control_file():
    cmd = ssh_common.build_kill_command("$HOME/runs/r1/out/.runplz", run_id="r1")
    for name in ("bootstrap.pid", "bootstrap.pgid", "container"):
        assert f"$HOME/runs/r1/out/.runplz/{name}" in cmd
    assert "killed_by_user" in cmd
    assert '"r1"' in cmd


def test_kill_command_signals_group_pid_and_container():
    cmd = ssh_common.build_kill_command("$HOME/m", run_id="r")
    assert 'kill -"$1" "-$runplz_pgid"' in cmd
    assert 'kill -"$1" "$runplz_pid"' in cmd
    assert 'sudo docker kill --signal="$1" "$runplz_container"' in cmd


def test_kill_command_is_valid_posix_sh(tmp_path):
    script = tmp_path / "kill.sh"
    script.write_text(ssh_common.build_kill_command("$HOME/m", run_id="r"))
    assert subprocess.run(["sh", "-n", str(script)]).returncode == 0
    assert subprocess.run(["bash", "-n", str(script)]).returncode == 0


def test_kill_command_honors_no_escalate():
    escalating = ssh_common.build_kill_command("$HOME/m", escalate=True)
    passive = ssh_common.build_kill_command("$HOME/m", escalate=False)
    assert 'if [ "1" = "1" ]; then' in escalating
    assert 'if [ "0" = "1" ]; then' in passive


def test_kill_command_uses_requested_first_signal():
    assert 'runplz_signal "INT"' in ssh_common.build_kill_command("$HOME/m", first_signal="INT")


# ---------------------------------------------------------------------------
# The generated shell, actually executed


def _run_kill_script(home: Path, meta: Path, **kwargs) -> dict[str, str]:
    script = home / "kill.sh"
    script.write_text(ssh_common.build_kill_command(str(meta), run_id="r1", **kwargs))
    r = subprocess.run(
        ["bash", str(script)],
        capture_output=True,
        text=True,
        env=dict(os.environ, HOME=str(home)),
        timeout=120,
    )
    assert r.returncode == 0, r.stderr
    sections = _runs._parse_status_sections(r.stdout)
    return _runs._parse_kv_block(sections.get("SUMMARY", ""))


def test_kill_script_is_idempotent_on_a_run_that_never_existed(tmp_path):
    meta = tmp_path / "runs" / "r1" / "out" / ".runplz"
    meta.mkdir(parents=True)
    fields = _run_kill_script(tmp_path, meta)
    assert fields["initial"] == "missing"
    assert fields["signalled"] == "0"
    assert fields["escalated"] == "0"
    assert not (meta / "events.ndjson").exists()


def test_kill_script_refuses_to_signal_process_group_1(tmp_path):
    """`kill -TERM -1` would signal every process the user owns."""
    meta = tmp_path / "runs" / "r1" / "out" / ".runplz"
    meta.mkdir(parents=True)
    (meta / "bootstrap.pgid").write_text("1\n")
    fields = _run_kill_script(tmp_path, meta)
    assert fields["pgid"] == ""
    assert fields["signalled"] == "0"


def test_kill_script_rejects_non_numeric_pid_and_pgid(tmp_path):
    meta = tmp_path / "runs" / "r1" / "out" / ".runplz"
    meta.mkdir(parents=True)
    (meta / "bootstrap.pid").write_text("not-a-pid\n")
    (meta / "bootstrap.pgid").write_text("$(rm -rf /)\n")
    fields = _run_kill_script(tmp_path, meta)
    assert fields["pid"] == ""
    assert fields["pgid"] == ""
    assert fields["signalled"] == "0"


def test_kill_script_stops_the_whole_orphaned_process_group(tmp_path):
    """The case that motivated the issue: supervisor gone, workers orphaned.

    Spawns a bootstrap with two worker children in a session of its own, then
    lets the launcher exit so the group is reparented to init exactly as it is
    after ssh disconnects. Killing must take the workers with it, not just the
    pid named in ``bootstrap.pid``.
    """
    meta = tmp_path / "runs" / "r1" / "out" / ".runplz"
    meta.mkdir(parents=True)
    pid_file = meta / "bootstrap.pid"

    # start_new_session gives the launcher its own process group; the
    # backgrounded job inherits it and outlives the launcher, so the group ends
    # up orphaned to init. A nested `bash -c` rather than a subshell, because
    # $$ inside `( )` reports the parent's pid on the bash 3.2 macOS ships and
    # $BASHPID needs bash 4.
    inner = f"echo $$ > {pid_file}; sleep 300 & sleep 300 & wait"
    launcher = subprocess.Popen(
        ["bash", "-c", f"bash -c {shlex.quote(inner)} &"],
        start_new_session=True,
    )
    assert launcher.wait(timeout=30) == 0

    # Poll for content, not existence: the file appears (redirect) before the
    # pid lands in it, and the nested shell may not have started at all yet.
    def recorded_pid() -> str:
        try:
            return pid_file.read_text().strip()
        except FileNotFoundError:
            return ""

    deadline = time.time() + 10
    while time.time() < deadline and not recorded_pid():
        time.sleep(0.05)
    assert recorded_pid(), "bootstrap never wrote its pid"
    time.sleep(0.3)

    bootstrap_pid = int(recorded_pid())
    pgid = os.getpgid(bootstrap_pid)
    assert pgid != os.getpgid(0), "refusing to run against the test runner's own group"
    (meta / "bootstrap.pgid").write_text(f"{pgid}\n")

    def group_members() -> int:
        out = subprocess.run(
            ["ps", "-o", "pid=,pgid=", "-A"], capture_output=True, text=True
        ).stdout
        return sum(1 for line in out.split("\n") if line.split()[1:2] == [str(pgid)])

    assert group_members() == 3, "expected bootstrap + 2 workers"

    try:
        fields = _run_kill_script(tmp_path, meta)
    finally:
        try:
            os.killpg(pgid, 9)
        except (ProcessLookupError, PermissionError):
            pass

    assert fields["initial"] == "running"
    assert fields["signalled"] == "1"
    assert group_members() == 0, "workers survived the group kill"

    event = json.loads((meta / "events.ndjson").read_text().strip())
    assert event["event"] == "killed_by_user"
    assert event["run_id"] == "r1"


# ---------------------------------------------------------------------------
# launcher records the pgid


def test_launcher_records_pgid_next_to_the_pid(tmp_path):
    remote_run = ssh_common.make_remote_run_context(
        backend="ssh", target="box", function_name="train"
    )
    launcher = ssh_common.build_detached_launcher(remote_run, "sleep 5")
    # Never reintroduce setsid — see build_detached_launcher's docstring.
    assert "setsid" not in launcher

    subprocess.run(
        ["bash", "-c", launcher],
        check=True,
        env=dict(os.environ, HOME=str(tmp_path)),
        timeout=60,
    )
    meta = tmp_path / "runplz-runs" / remote_run.run_id / "out" / ".runplz"
    pid = int((meta / "bootstrap.pid").read_text().strip())
    try:
        pgid_file = meta / "bootstrap.pgid"
        if HAS_PROCFS:
            assert pgid_file.read_text().strip() == str(os.getpgid(pid))
        else:
            # No procfs (macOS dev box): the capture is a documented no-op and
            # must not break the launch or leave a bogus file behind.
            assert not pgid_file.exists()
    finally:
        try:
            os.kill(pid, 9)
        except ProcessLookupError:
            pass


def test_launcher_pgid_capture_does_not_disturb_the_pid_file():
    remote_run = ssh_common.make_remote_run_context(
        backend="ssh", target="box", function_name="train"
    )
    launcher = ssh_common.build_detached_launcher(remote_run, "echo hi")
    assert f'echo $! > "{remote_run.meta_shell}/bootstrap.pid"' in launcher
    assert f'"{remote_run.meta_shell}/bootstrap.pgid"' in launcher


# ---------------------------------------------------------------------------
# container name recording


def test_container_mode_records_container_name_for_kill():
    recorded = {}

    def fake_ssh(target, command, port=None):
        recorded["command"] = command

    function = mock.Mock(name="fn", env={})
    function.name = "train"
    remote_run = ssh_common.make_remote_run_context(
        backend="brev", target="box", function_name="train"
    )
    with mock.patch.object(ssh_common, "_ssh", fake_ssh):
        ssh_common._run_container_detached(
            target="box",
            container_name="runplz-train-abc12345",
            function=function,
            rel_script="job.py",
            args=[],
            kwargs={},
            gpu_flag="",
            app_name="app",
            remote_run=remote_run,
        )
    command = recorded["command"]
    assert f'> "{remote_run.meta_shell}/container"' in command
    assert "runplz-train-abc12345" in command
    # Written before the monitor loop so a kill arriving early still finds it.
    assert command.index("/container") < command.index("runplz_event container_started")


# ---------------------------------------------------------------------------
# _runs.kill


def test_kill_reports_a_clean_stop(tmp_path, capsys):
    _write_manifest(tmp_path)
    fake = mock.Mock(returncode=0, stdout=_summary(), stderr="")
    with mock.patch("runplz._runs.subprocess.run", return_value=fake) as run_mock:
        rc = _runs.kill(outputs_dir=tmp_path, host_override=None, run_id_override=None)
    assert rc == 0
    out = capsys.readouterr().out
    assert "target:     my-gpu-box" in out
    assert "stopped with SIGTERM" in out
    assert "running -> dead" in out
    cmd = run_mock.call_args.args[0]
    assert cmd[0] == "ssh"
    assert cmd[-2] == "my-gpu-box"
    assert "killed_by_user" in cmd[-1]


def test_kill_reports_escalation(tmp_path, capsys):
    _write_manifest(tmp_path)
    fake = mock.Mock(returncode=0, stdout=_summary(escalated="1"), stderr="")
    with mock.patch("runplz._runs.subprocess.run", return_value=fake):
        _runs.kill(outputs_dir=tmp_path, host_override=None, run_id_override=None)
    assert "escalated to SIGKILL" in capsys.readouterr().out


def test_kill_on_an_already_dead_run_is_a_success(tmp_path, capsys):
    _write_manifest(tmp_path)
    stdout = _summary(initial="dead", final="dead", signalled="0", pid="", pgid="")
    fake = mock.Mock(returncode=0, stdout=stdout, stderr="")
    with mock.patch("runplz._runs.subprocess.run", return_value=fake):
        rc = _runs.kill(outputs_dir=tmp_path, host_override=None, run_id_override=None)
    assert rc == 0
    assert "nothing to kill" in capsys.readouterr().out


def test_kill_flags_a_survivor_instead_of_claiming_success(tmp_path, capsys):
    _write_manifest(tmp_path)
    fake = mock.Mock(returncode=0, stdout=_summary(final="running"), stderr="")
    with mock.patch("runplz._runs.subprocess.run", return_value=fake):
        _runs.kill(outputs_dir=tmp_path, host_override=None, run_id_override=None)
    out = capsys.readouterr().out
    assert "inspect manually" in out


def test_kill_flags_a_surviving_container(tmp_path, capsys):
    _write_manifest(tmp_path)
    stdout = _summary(container="runplz-train-abc", container_state="running")
    fake = mock.Mock(returncode=0, stdout=stdout, stderr="")
    with mock.patch("runplz._runs.subprocess.run", return_value=fake):
        _runs.kill(outputs_dir=tmp_path, host_override=None, run_id_override=None)
    out = capsys.readouterr().out
    assert "container:  runplz-train-abc (running)" in out
    assert "inspect manually" in out


def test_kill_renders_gpu_memory_and_heartbeat(tmp_path, capsys):
    _write_manifest(tmp_path)
    hb = json.dumps({"ts": "2026-08-26T01:05:30Z", "run_id": "x"})
    stdout = (
        _summary(gpu_mem_used="0,512").replace("---HEARTBEAT---\n", f"---HEARTBEAT---\n{hb}\n")
    ).replace("---LOGTAIL---\n", "---LOGTAIL---\nepoch 3 loss 0.12\n")
    fake = mock.Mock(returncode=0, stdout=stdout, stderr="")
    with mock.patch("runplz._runs.subprocess.run", return_value=fake):
        _runs.kill(outputs_dir=tmp_path, host_override=None, run_id_override=None)
    out = capsys.readouterr().out
    assert "gpu memory: gpu0=0MiB, gpu1=512MiB" in out
    assert "2026-08-26T01:05:30Z" in out
    assert "epoch 3 loss 0.12" in out


def test_kill_returns_ssh_failure_code(tmp_path, capsys):
    _write_manifest(tmp_path)
    fake = mock.Mock(returncode=255, stdout="", stderr="ssh: connect refused")
    with mock.patch("runplz._runs.subprocess.run", return_value=fake):
        rc = _runs.kill(outputs_dir=tmp_path, host_override=None, run_id_override=None)
    assert rc == 255
    assert "ssh to my-gpu-box failed" in capsys.readouterr().err


def test_kill_survives_an_ssh_timeout(tmp_path, capsys):
    _write_manifest(tmp_path)
    with mock.patch(
        "runplz._runs.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="ssh", timeout=75),
    ):
        rc = _runs.kill(outputs_dir=tmp_path, host_override=None, run_id_override=None)
    assert rc == 2
    assert "timed out" in capsys.readouterr().err


def test_kill_ssh_timeout_outlives_the_remote_escalation_clock(tmp_path):
    _write_manifest(tmp_path)
    fake = mock.Mock(returncode=0, stdout=_summary(), stderr="")
    with mock.patch("runplz._runs.subprocess.run", return_value=fake) as run_mock:
        _runs.kill(
            outputs_dir=tmp_path,
            host_override=None,
            run_id_override=None,
            timeout_s=600,
        )
    assert run_mock.call_args.kwargs["timeout"] > 600 + ssh_common.KILL_SETTLE_S


# ---------------------------------------------------------------------------
# CLI


@pytest.mark.parametrize("verb", ["kill", "cancel"])
def test_cli_dispatches_both_verbs(tmp_path, verb):
    _write_manifest(tmp_path)
    with mock.patch.object(_runs, "kill", return_value=0) as kill_mock:
        rc = _cli.main([verb, "--outputs-dir", str(tmp_path)])
    assert rc == 0
    kwargs = kill_mock.call_args.kwargs
    assert kwargs["outputs_dir"] == tmp_path.resolve()
    assert kwargs["timeout_s"] == ssh_common.DEFAULT_KILL_TIMEOUT_S
    assert kwargs["escalate"] is True
    assert kwargs["first_signal"] == "TERM"


def test_cli_kill_plumbs_signal_and_timeout(tmp_path):
    _write_manifest(tmp_path)
    with mock.patch.object(_runs, "kill", return_value=0) as kill_mock:
        _cli.main(
            [
                "kill",
                "--outputs-dir",
                str(tmp_path),
                "--signal",
                "INT",
                "--timeout",
                "45",
                "--no-escalate",
            ]
        )
    kwargs = kill_mock.call_args.kwargs
    assert kwargs["first_signal"] == "INT"
    assert kwargs["timeout_s"] == 45
    assert kwargs["escalate"] is False


def test_cli_kill_rejects_a_negative_timeout(tmp_path):
    _write_manifest(tmp_path)
    with pytest.raises(SystemExit):
        _cli.main(["kill", "--outputs-dir", str(tmp_path), "--timeout", "-1"])


def test_cli_kill_surfaces_missing_manifest(tmp_path, capsys):
    rc = _cli.main(["kill", "--outputs-dir", str(tmp_path)])
    assert rc == 1
    assert "No run manifest" in capsys.readouterr().err


def test_cli_kill_run_id_without_host_errors(tmp_path, capsys):
    rc = _cli.main(["kill", "--outputs-dir", str(tmp_path), "--run-id", "xyz"])
    assert rc == 2
    assert "--run-id requires --host" in capsys.readouterr().err


def test_cli_kill_targets_an_explicit_host_and_run_id(tmp_path):
    with mock.patch.object(_runs, "kill", return_value=0) as kill_mock:
        _cli.main(["kill", "--host", "other-box", "--run-id", "rid-9"])
    kwargs = kill_mock.call_args.kwargs
    assert kwargs["host_override"] == "other-box"
    assert kwargs["run_id_override"] == "rid-9"


def test_kill_uses_run_id_override_when_there_is_no_manifest(tmp_path, capsys):
    fake = mock.Mock(returncode=0, stdout=_summary(), stderr="")
    with mock.patch("runplz._runs.subprocess.run", return_value=fake) as run_mock:
        rc = _runs.kill(
            outputs_dir=tmp_path,
            host_override="other-box",
            run_id_override="rid-9",
        )
    assert rc == 0
    remote_cmd = run_mock.call_args.args[0][-1]
    assert "runplz-runs/rid-9/out/.runplz/bootstrap.pid" in remote_cmd
    assert '"rid-9"' in remote_cmd
    assert "run:        rid-9" in capsys.readouterr().out


if sys.platform == "win32":  # pragma: no cover - runplz targets POSIX hosts
    pytest.skip("kill relies on POSIX process groups", allow_module_level=True)
