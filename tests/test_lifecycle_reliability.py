"""Regression tests for conservative stop evidence and durable cleanup (#166)."""

import json
import os
import shlex
import signal
import subprocess
import sys
from unittest import mock

import pytest

from runplz import App, Image, runs
from runplz.backends import ssh_common as sc


@pytest.mark.parametrize("shell", ["sh", "bash"])
@pytest.mark.parametrize(
    "initial,after,delivered,expected_alive,expected_signal,expected_event",
    [
        ("true", "error", False, "null", "0", "kill_attempted_by_user"),
        ("true", "error", True, "null", "1", "kill_attempted_by_user"),
        ("true", "true", False, "1", "0", "kill_attempted_by_user"),
        ("true", "false", True, "0", "1", "killed_by_user"),
        ("error", "error", False, "null", "0", "kill_attempted_by_user"),
        ("malformed", "malformed", False, "null", "0", "kill_attempted_by_user"),
        ("absent", "absent", False, "0", "0", None),
        ("false", "false", False, "0", "0", None),
    ],
)
def test_kill_measures_delivery_and_does_not_treat_probe_errors_as_stopped(
    tmp_path, shell, initial, after, delivered, expected_alive, expected_signal, expected_event
):
    # Execute the production shell; only the Docker boundary is fake. A kill
    # can fail just as a daemon becomes unavailable to subsequent inspection.
    prefix = f"""
audit_attempted=0
sudo() {{
  case "$2" in
    inspect)
      if [ "$audit_attempted" = 0 ]; then audit_result={shlex.quote(initial)};
      else audit_result={shlex.quote(after)}; fi
      case "$audit_result" in
        error) echo 'Cannot connect to Docker daemon' >&2; return 1 ;;
        absent) echo 'Error: No such object: audit-container' >&2; return 1 ;;
        *) printf '%s\\n' "$audit_result" ;;
      esac ;;
    kill) audit_attempted=1; return {0 if delivered else 1} ;;
  esac
}}
"""
    script = sc.build_kill_command(
        str(tmp_path),
        container="audit-container",
        run_id="audit-run",
        proc_root=str(tmp_path),
        timeout_s=0,
        escalate=False,
    )
    completed = subprocess.run(
        [shell, "-c", prefix + script], capture_output=True, text=True, timeout=10, check=True
    )
    sections = sc.parse_probe_sections(completed.stdout)
    fields = sc.parse_kv_block(sections["SUMMARY"])
    assert fields["alive_after"] == expected_alive
    assert fields["signalled"] == expected_signal
    if expected_alive == "null":
        assert fields["container_state"] == "unknown"
        assert "stop unconfirmed" in runs._format_kill(
            target="box", run_id="audit-run", fields=fields, sections=sections
        )
    path = tmp_path / "events.ndjson"
    if expected_event is None:
        assert not path.exists()
    else:
        event = json.loads(path.read_text())
        assert event["event"] == expected_event
        assert event["signalled"] is (expected_signal == "1")
        assert event["alive_after"] == {"0": 0, "1": 1, "null": None}[expected_alive]

    # The same measured summary is consumed by the CLI and runtime cap.
    with mock.patch.object(runs.subprocess, "run", return_value=completed):
        rc = runs.kill(outputs_dir=tmp_path, host_override="box", run_id_override="audit-run")
    assert rc == {"0": 0, "1": 3, "null": 2}[expected_alive]
    remote_run = sc.make_remote_run_context(backend="ssh", target="box", function_name="train")
    with (
        mock.patch.object(sc.subprocess, "run", return_value=completed),
        mock.patch.object(sc, "_record_remote_event") as record,
        pytest.raises(RuntimeError, match="max_runtime_seconds"),
    ):
        sc.raise_for_runtime_cap("box", 60, container_name="audit-container", remote_run=remote_run)
    cap_event = record.call_args.args[2]
    assert cap_event == (
        "killed_by_runtime_cap"
        if expected_signal == "1" and expected_alive == "0"
        else "runtime_cap_reached"
    )
    with (
        mock.patch.object(sc, "ssh_capture", return_value=""),
        mock.patch.object(sc, "seconds_since_activity", return_value=120),
        mock.patch.object(sc, "detached_launch_diagnostics", return_value=""),
        mock.patch.object(sc, "_terminate_stalled_run", return_value=fields),
        mock.patch.object(sc, "_record_remote_event") as record,
    ):
        sc._check_inactivity("box", remote_run, 60, "terminate", already_reported=False)
    expected_action = {
        ("1", "0"): "terminate",
        ("0", "0"): "nothing_to_stop",
    }.get(
        (expected_signal, expected_alive),
        "terminate_failed" if expected_alive == "1" else "terminate_unconfirmed",
    )
    assert record.call_args.kwargs["action"] == expected_action


def _app(tmp_path):
    app = App("audit")
    app.repo_root = tmp_path.resolve()
    job = tmp_path / "job.py"
    job.write_text("# offline test\n")

    @app.function(image=Image.from_registry("ubuntu:22.04"))
    def train():  # pragma: no cover - never executed
        pass

    function = app.functions["train"]
    function.module_file = str(job)
    return app, function


@pytest.mark.parametrize("backend", ["ssh", "brev", "gcp", "aws"])
@pytest.mark.parametrize("phase", ["tail", "sync", "remove", "event"])
def test_signals_during_cleanup_finish_salvage_and_persist_before_teardown(
    tmp_path, backend, phase
):
    app, function = _app(tmp_path)
    observed = []

    def step(name):
        observed.append(name)
        if name == phase:
            # Real installed handlers, including a repeated signal: neither
            # may interrupt the rest of this bounded cleanup action.
            os.kill(os.getpid(), signal.SIGTERM)
            os.kill(os.getpid(), signal.SIGINT)
        observed.append(name + "_finished")
        return ""

    def sync(*args, **kwargs):
        assert kwargs["timeout_s"] == sc._SALVAGE_TIMEOUT_S
        step("sync")

    interruption = sc.OrchestratorKilled("cancelled", signal_name="SIGTERM")
    with mock.patch.multiple(
        sc,
        wait_until_ssh_reachable=mock.DEFAULT,
        prepare_remote_run=mock.DEFAULT,
        ensure_remote_rsync=mock.DEFAULT,
        rsync_up=mock.DEFAULT,
        check_preconditions=mock.DEFAULT,
        ensure_docker=mock.DEFAULT,
        remote_has_nvidia=mock.Mock(return_value=False),
        build_image=mock.DEFAULT,
        run_container_detached=mock.DEFAULT,
        stream_and_wait=mock.Mock(side_effect=interruption),
        fetch_failure_tail=mock.Mock(side_effect=lambda **kw: step("tail")),
        rsync_down=sync,
        ssh_capture=mock.Mock(side_effect=lambda *a, **kw: step("remove")),
        _record_remote_event=mock.Mock(side_effect=lambda *a, **kw: step("event")),
    ):
        with pytest.raises(sc.OrchestratorKilled, match="cancelled"):
            sc.run_on_provisioned_vm(
                app=app,
                function=function,
                args=[],
                kwargs={},
                backend=backend,
                label="audit",
                provision=lambda: ("box", None),
                teardown=lambda: observed.append("teardown"),
            )

    assert all(name + "_finished" in observed for name in ("tail", "sync", "remove", "event"))
    assert observed[-1] == "teardown"
    local = tmp_path / "out" / ".runplz" / "events.ndjson"
    assert json.loads(local.read_text())["event"] == "orchestrator_signalled"


@pytest.mark.parametrize("phase", ["tail", "sync", "remove"])
def test_first_signal_raised_during_cleanup_cannot_skip_later_steps(tmp_path, phase):
    app, function = _app(tmp_path)
    observed = []
    interruption = sc.OrchestratorKilled("cleanup cancelled", signal_name="SIGTERM")

    def step(name):
        observed.append(name)
        if name == phase:
            raise interruption
        return ""

    with mock.patch.multiple(
        sc,
        prepare_remote_run=mock.DEFAULT,
        ensure_remote_rsync=mock.DEFAULT,
        rsync_up=mock.DEFAULT,
        check_preconditions=mock.DEFAULT,
        ensure_docker=mock.DEFAULT,
        remote_has_nvidia=mock.Mock(return_value=False),
        build_image=mock.DEFAULT,
        run_container_detached=mock.DEFAULT,
        stream_and_wait=mock.Mock(side_effect=RuntimeError("original failure")),
        fetch_failure_tail=mock.Mock(side_effect=lambda **kw: step("tail")),
        rsync_down=mock.Mock(side_effect=lambda *a, **kw: step("sync")),
        ssh_capture=mock.Mock(side_effect=lambda *a, **kw: step("remove")),
        _record_remote_event=mock.DEFAULT,
    ):
        with pytest.raises(sc.OrchestratorKilled) as raised:
            sc.dispatch_to_target(
                app=app, function=function, args=[], kwargs={}, target="box", backend="ssh"
            )
    assert raised.value is interruption
    assert observed == ["tail", "sync", "remove"]
    local = tmp_path / "out" / ".runplz" / "events.ndjson"
    assert json.loads(local.read_text())["signal"] == "SIGTERM"


def test_first_signal_after_successful_sync_is_still_persisted(tmp_path):
    app, function = _app(tmp_path)
    removed = []

    def remove(*a, **kw):
        os.kill(os.getpid(), signal.SIGTERM)
        removed.append(True)
        return ""

    with (
        sc.orchestrator_signal_cleanup("box"),
        mock.patch.multiple(
            sc,
            prepare_remote_run=mock.DEFAULT,
            ensure_remote_rsync=mock.DEFAULT,
            rsync_up=mock.DEFAULT,
            check_preconditions=mock.DEFAULT,
            ensure_docker=mock.DEFAULT,
            remote_has_nvidia=mock.Mock(return_value=False),
            build_image=mock.DEFAULT,
            run_container_detached=mock.DEFAULT,
            stream_and_wait=mock.Mock(return_value=0),
            rsync_down=mock.DEFAULT,
            ssh_capture=remove,
            _record_remote_event=mock.DEFAULT,
        ),
        pytest.raises(sc.OrchestratorKilled),
    ):
        sc.dispatch_to_target(
            app=app, function=function, args=[], kwargs={}, target="box", backend="ssh"
        )
    assert removed == [True]
    local = tmp_path / "out" / ".runplz" / "events.ndjson"
    assert json.loads(local.read_text())["signal"] == "SIGTERM"


def test_run_local_enforces_a_real_subprocess_deadline():
    with pytest.raises(subprocess.TimeoutExpired):
        sc.run_local([sys.executable, "-c", "import time; time.sleep(30)"], timeout_s=0.05)


def test_event_writes_use_a_bounded_subprocess_and_warn_on_timeout(capsys):
    remote_run = sc.make_remote_run_context(backend="brev", target="box", function_name="train")

    def timeout(cmd, **kwargs):
        assert kwargs["timeout"] == sc._EVENT_WRITE_TIMEOUT_S
        raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])

    with mock.patch.object(sc.subprocess, "run", side_effect=timeout):
        sc._record_remote_event("box", remote_run, "rsync_down_done")
    assert "failed to record" in capsys.readouterr().out


def test_salvage_timeout_is_recorded_locally(tmp_path):
    remote_run = sc.make_remote_run_context(backend="aws", target="box", function_name="train")

    def timeout(cmd, **kwargs):
        assert 0 < kwargs["timeout"] <= sc._SALVAGE_TIMEOUT_S
        assert f"--timeout={sc._RSYNC_IDLE_TIMEOUT_S}" in cmd
        raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])

    with (
        mock.patch.object(sc.subprocess, "run", side_effect=timeout),
        mock.patch.object(sc, "_record_remote_event"),
        pytest.raises(subprocess.TimeoutExpired),
    ):
        sc.rsync_down("box", tmp_path, remote_run=remote_run, timeout_s=sc._SALVAGE_TIMEOUT_S)
    event = json.loads((tmp_path / ".runplz" / "events.ndjson").read_text())
    assert event["event"] == "rsync_down_failed"
    assert event["error_type"] == "TimeoutExpired"


def test_healthy_download_has_idle_timeout_but_no_total_time_limit(tmp_path):
    with mock.patch.object(sc.subprocess, "run") as run:
        sc.rsync_down("box", tmp_path)
    assert run.call_args.kwargs["timeout"] is None
    assert f"--timeout={sc._RSYNC_IDLE_TIMEOUT_S}" in run.call_args.args[0]


@pytest.mark.parametrize("budget", [1, 60])
def test_salvage_retries_share_one_transfer_deadline(tmp_path, budget):
    now = [0]
    timeouts = []

    def run(cmd, **kwargs):
        timeouts.append(kwargs["timeout_s"])
        if len(timeouts) == 1:
            raise subprocess.CalledProcessError(12, cmd)

    def sleep(seconds):
        now[0] += seconds

    with (
        mock.patch.object(sc.time, "monotonic", side_effect=lambda: now[0]),
        mock.patch.object(sc.time, "sleep", side_effect=sleep),
        mock.patch.object(sc, "run_local", side_effect=run),
    ):
        if budget == 1:
            with pytest.raises(subprocess.TimeoutExpired):
                sc.rsync_down("box", tmp_path, timeout_s=budget)
            assert timeouts == [1]
        else:
            sc.rsync_down("box", tmp_path, timeout_s=budget)
            assert timeouts == [60, 58]


def test_cleanup_guard_leaves_embedding_app_signal_handlers_alone():
    before = {sig: signal.getsignal(sig) for sig in sc.CLEANUP_SIGNALS}
    with sc._defer_cleanup_signals() as pending:
        assert {sig: signal.getsignal(sig) for sig in sc.CLEANUP_SIGNALS} == before
    assert pending == []


def test_signal_outside_cleanup_cancels_immediately_and_restores_handlers():
    before = {sig: signal.getsignal(sig) for sig in sc.CLEANUP_SIGNALS}
    with pytest.raises(sc.OrchestratorKilled, match="SIGTERM"):
        with sc.orchestrator_signal_cleanup("audit"):
            os.kill(os.getpid(), signal.SIGTERM)
            pytest.fail("signal did not cancel")
    assert {sig: signal.getsignal(sig) for sig in sc.CLEANUP_SIGNALS} == before


def test_cleanup_guard_tolerates_off_main_thread_signal_registration():
    handler = sc._OrchestratorSignalHandler("audit")
    with (
        mock.patch.object(sc.signal, "getsignal", return_value=handler),
        mock.patch.object(sc.signal, "signal", side_effect=ValueError("not main thread")),
    ):
        with sc._defer_cleanup_signals() as pending:
            assert pending == []


@pytest.mark.parametrize("mode", ["native", "container"])
@pytest.mark.parametrize("outcome", ["done", "failed", "unconfirmed", "timeout", "signal"])
def test_environment_setup_reports_outcome_and_never_launches_after_failure(
    tmp_path, mode, outcome
):
    _, function = _app(tmp_path)
    remote_run = sc.make_remote_run_context(backend="ssh", target="box", function_name="train")
    errors = {
        "done": None,
        "failed": subprocess.CalledProcessError(1, "setup"),
        "unconfirmed": subprocess.CalledProcessError(255, "ssh"),
        "timeout": subprocess.TimeoutExpired("setup", 60),
        "signal": sc.OrchestratorKilled("setup cancelled"),
    }
    failure = errors[outcome]
    with (
        mock.patch.object(sc, "ssh_exec", side_effect=failure),
        mock.patch.object(sc, "retry_on_transport_failure", side_effect=lambda fn, **kw: fn()),
        mock.patch.object(sc, "_record_remote_event") as record,
        mock.patch.object(sc, "launch_detached_and_wait", return_value=0) as launch,
    ):
        runner = sc.run_native if mode == "native" else sc.run_container_mode
        kwargs = {"has_nvidia": False} if mode == "native" else {}

        def execute():
            return runner(
                target="box",
                function=function,
                rel_script="job.py",
                args=[],
                kwargs={},
                remote_run=remote_run,
                **kwargs,
            )

        if failure:
            with pytest.raises(type(failure)) as raised:
                execute()
            assert raised.value is failure
            launch.assert_not_called()
        else:
            assert execute() == 0
            launch.assert_called_once()
    names = [call.args[2] for call in record.call_args_list]
    expected = "unconfirmed" if outcome == "timeout" else outcome
    assert names == ["environment_setup_start"] + (
        [] if outcome == "signal" else [f"environment_setup_{expected}"]
    )
