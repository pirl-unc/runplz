"""The opt-in inactivity watchdog for detached runs (issue #122).

Heartbeats prove the bootstrap is alive; they cannot tell productive work
from an application-level deadlock, and `max_runtime_seconds` says nothing
about a job that wedges early. The observed case was a 4xA100 run whose work
finished in ten seconds and then sat in `Pool.join()` for 31 minutes with the
heartbeat still ticking and every GPU idle.

These drive `tail_and_wait_for_detached` through a FakeClock, because the
watchdog is a wall-clock feature: patching `time.sleep` alone — what the
pre-existing tail tests do — cannot advance `monotonic()` and so can never
reach an expiry.
"""

import subprocess
from unittest import mock

import pytest

from runplz.backends import ssh_common as sc


def _remote_run():
    return sc.make_remote_run_context(backend="ssh", target="box", function_name="train")


class _Remote:
    """A scriptable remote for the monitor loop.

    `idle` is what the inactivity probe reports; set it per-test to say how
    long the application has been silent.
    """

    def __init__(self, *, idle=0, alive_calls=2, exit_code=0):
        self.idle = idle
        self.alive_calls = alive_calls
        self.exit_code = exit_code
        self.commands = []
        self.tail_calls = 0

    def run(self, argv, **kwargs):
        cmd = argv[-1] if isinstance(argv, list) else str(argv)
        self.commands.append(cmd)
        if "tail -n +1 -F" in cmd:
            self.tail_calls += 1
            timeout = kwargs.get("timeout")
            if timeout is None:
                # An unbounded tail cannot time out. It ends only when the
                # remote follower exits or ssh drops — so returning here is
                # the only faithful thing a fake can do.
                return mock.Mock(returncode=255, stdout="", stderr="")
            # A bounded tail is bounded *by the watchdog*: expiring is the
            # wake-up, not a failure.
            raise subprocess.TimeoutExpired(cmd, timeout)
        return mock.Mock(returncode=0, stdout="", stderr="")

    def capture(self, target, cmd, **kwargs):
        self.commands.append(cmd)
        if "---NOW---" in cmd:
            return f"---NOW---\n{self.idle}\n---LOG---\n0\n---OUT---\n0\n---END---\n"
        if "runplz_run_pids" in cmd or "alive_after" in cmd:
            return "---SUMMARY---\nalive_after=0\nsurvivors=\n---END---\n"
        return ""

    def alive(self, *a, **kw):
        self.alive_calls -= 1
        return self.alive_calls > 0


def _drive(remote, *, max_inactivity_seconds, action="diagnose", max_reconnects=20):
    """Run the monitor loop against a scripted remote."""
    rr = _remote_run()
    with (
        mock.patch.object(sc.subprocess, "run", side_effect=remote.run),
        mock.patch.object(sc, "ssh_capture", side_effect=remote.capture),
        mock.patch.object(sc, "remote_pid_alive", side_effect=remote.alive),
        mock.patch.object(sc, "read_remote_exit_code", return_value=remote.exit_code),
        mock.patch.object(sc, "_record_remote_event") as record,
    ):
        code = sc.tail_and_wait_for_detached(
            target="box",
            pid_file=f"{rr.meta_shell}/bootstrap.pid",
            log_file=f"{rr.meta_shell}/run_driver.log",
            events_file=rr.events_shell,
            max_inactivity_seconds=max_inactivity_seconds,
            inactivity_action=action,
            max_reconnects=max_reconnects,
            remote_run=rr,
        )
    return code, record


# ---------------------------------------------------------------------------
# the probe


def test_the_probe_reads_application_output_not_the_heartbeat():
    """The heartbeat loop is a background job of the wrapper shell, ticking on
    a timer whether or not the job progresses. Treating it as activity is
    exactly the mistake that made a 31-minute deadlock look healthy."""
    rr = _remote_run()
    probe = sc.build_inactivity_probe(rr)
    assert "run_driver.log" in probe
    assert rr.out_shell in probe
    assert "heartbeat" not in probe


def test_idle_seconds_come_from_the_remote_clock():
    """Both `now` and the mtimes are read on the box, so skew between the
    laptop and the remote cannot invent a stall or hide one."""
    out = "---NOW---\n1000\n---LOG---\n400\n---OUT---\n700\n---END---\n"
    # 700 is the more recent of the two signals.
    assert sc.seconds_since_activity(out) == 300.0


def test_an_unreadable_probe_is_not_a_stall():
    """Terminating a healthy job because one ssh round trip came back
    garbled would be a worse failure than the deadlock being watched for."""
    for bad in ("", "nonsense", "---NOW---\n\n---END---\n", "---LOG---\n5\n---END---\n"):
        assert sc.seconds_since_activity(bad) is None


# ---------------------------------------------------------------------------
# behaviour


def test_healthy_silence_below_the_threshold_changes_nothing(fast_clock):
    remote = _Remote(idle=10)
    code, record = _drive(remote, max_inactivity_seconds=1800)
    assert code == 0
    stalls = [c for c in record.call_args_list if c.args[2] == "remote_command_stalled"]
    assert stalls == []


def test_the_watchdog_is_off_unless_asked_for(fast_clock):
    """Build, download and compile phases are legitimately quiet. With no
    threshold the loop must not probe at all."""
    remote = _Remote(idle=10_000)
    _drive(remote, max_inactivity_seconds=None)
    assert not [c for c in remote.commands if "---NOW---" in c]


def test_diagnose_records_the_stall_and_keeps_monitoring(fast_clock):
    remote = _Remote(idle=3600, alive_calls=4, exit_code=7)
    code, record = _drive(remote, max_inactivity_seconds=1800, action="diagnose")

    stalls = [c for c in record.call_args_list if c.args[2] == "remote_command_stalled"]
    assert len(stalls) == 1
    assert stalls[0].kwargs["idle_seconds"] == 3600
    assert stalls[0].kwargs["action"] == "diagnose"
    # Kept monitoring: the run still produced its own exit code, and nothing
    # was killed.
    assert code == 7
    assert not [c for c in remote.commands if "runplz_run_pids" in c]


def test_diagnose_fires_once_per_episode_not_once_per_poll(fast_clock):
    """A job quiet for ten hours should warn once, not six hundred times."""
    remote = _Remote(idle=3600, alive_calls=8)
    _, record = _drive(remote, max_inactivity_seconds=1800, action="diagnose")
    stalls = [c for c in record.call_args_list if c.args[2] == "remote_command_stalled"]
    assert len(stalls) == 1


def test_terminate_stops_the_run_and_still_syncs_outputs(fast_clock):
    """The requirement `max_runtime_seconds` fails.

    `raise_for_runtime_cap` raises out of `dispatch_to_target` *before*
    `rsync_down`, so a capped run loses whatever it produced. The watchdog
    must not copy that: it stops the run and returns normally, leaving the
    completion path — and the outputs sync — intact.
    """
    remote = _Remote(idle=3600, alive_calls=6)
    code, record = _drive(remote, max_inactivity_seconds=1800, action="terminate")

    stalls = [c for c in record.call_args_list if c.args[2] == "remote_command_stalled"]
    assert stalls[0].kwargs["action"] == "terminate"
    # Run-scoped kill, not a broad pkill.
    kills = [c for c in remote.commands if "runplz_run_pids" in c]
    assert kills, remote.commands
    assert sc.RUN_ID_ENV_VAR in kills[0]
    # Returned rather than raised, which is what leaves rsync_down reachable.
    assert isinstance(code, int)


def test_watchdog_ticks_do_not_spend_the_reconnect_budget(fast_clock):
    """`reconnects` is cumulative for the run and capped at `max_reconnects`.
    If a poll counted as a reconnect, any legitimately quiet job would burn
    the budget and silently lose its live log stream."""
    remote = _Remote(idle=10, alive_calls=12)
    _drive(remote, max_inactivity_seconds=1800, max_reconnects=3)
    # More tails than the reconnect budget: the loop kept streaming.
    assert remote.tail_calls > 3, remote.tail_calls


def test_a_probe_failure_leaves_the_run_alone(fast_clock):
    remote = _Remote(idle=3600, alive_calls=4)

    def explode(target, cmd, **kwargs):
        if "---NOW---" in cmd:
            raise RuntimeError("ssh blew up")
        return remote.capture(target, cmd, **kwargs)

    rr = _remote_run()
    with (
        mock.patch.object(sc.subprocess, "run", side_effect=remote.run),
        mock.patch.object(sc, "ssh_capture", side_effect=explode),
        mock.patch.object(sc, "remote_pid_alive", side_effect=remote.alive),
        mock.patch.object(sc, "read_remote_exit_code", return_value=0),
        mock.patch.object(sc, "_record_remote_event") as record,
    ):
        sc.tail_and_wait_for_detached(
            target="box",
            pid_file="p",
            log_file="l",
            events_file=rr.events_shell,
            max_inactivity_seconds=1800,
            inactivity_action="terminate",
            remote_run=rr,
        )
    assert not [c for c in record.call_args_list if c.args[2] == "remote_command_stalled"]
    assert not [c for c in remote.commands if "runplz_run_pids" in c]


# ---------------------------------------------------------------------------
# diagnostics


def test_stall_diagnostics_report_this_run_s_processes_including_zombies(tmp_path):
    """The finding in #122 was one child wedged in `_finalize_join` with four
    zombie siblings. The kill scan skips zombies on purpose — counting one as
    alive would pin its wait loop — so the diagnostics need the opposite
    filter over the same marker."""
    rr = _remote_run()
    for pid, state, mine in (("101", "S", True), ("102", "Z", True), ("999", "R", False)):
        d = tmp_path / pid
        d.mkdir()
        marker = f"{sc.RUN_ID_ENV_VAR}={rr.run_id}" if mine else "OTHER=1"
        (d / "environ").write_bytes(f"PATH=/usr/bin\0{marker}\0".encode())
        (d / "stat").write_text(f"{pid} (python) {state} 1 1 0\n")

    out = subprocess.run(
        ["bash", "-c", sc._stall_context_shell(rr, proc_root=str(tmp_path))],
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stderr
    assert "101 S" in out.stdout
    assert "102 Z" in out.stdout, "a zombie child is the finding, not noise"
    assert "999" not in out.stdout, "another run's process must never be reported"
    assert "accelerators:" in out.stdout


def test_stall_context_is_opt_in_so_the_launch_path_is_unchanged():
    rr = _remote_run()
    with mock.patch.object(sc, "ssh_capture", return_value="") as capture:
        sc.detached_launch_diagnostics("box", rr)
    assert "run processes" not in capture.call_args.args[1]

    with mock.patch.object(sc, "ssh_capture", return_value="") as capture:
        sc.detached_launch_diagnostics("box", rr, include_stall_context=True)
    assert "run processes" in capture.call_args.args[1]
    assert "nvidia-smi" in capture.call_args.args[1]


@pytest.mark.parametrize("config_cls", ["BrevConfig", "SshConfig"])
def test_the_config_reaches_the_monitor(config_cls):
    """#124 landed the fields and their validation; nothing read them until
    now. This pins the whole path rather than the endpoints."""
    import inspect

    for fn in (
        sc.dispatch_to_target,
        sc.run_on_provisioned_vm,
        sc.run_container_mode,
        sc.run_native,
        sc.launch_detached_and_wait,
        sc.tail_and_wait_for_detached,
    ):
        params = inspect.signature(fn).parameters
        assert "max_inactivity_seconds" in params, fn.__name__
        assert "inactivity_action" in params, fn.__name__
