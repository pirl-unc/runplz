"""Lifecycle events describe observed outcomes, not cleanup ordering."""

import json
import subprocess
from types import SimpleNamespace
from unittest import mock

import pytest

from runplz import App, Image, runs
from runplz.backends import ssh_common as sc


def _event(name, **fields):
    return json.dumps({"ts": "2026-09-04T12:00:00Z", "event": name, **fields}, sort_keys=True)


@pytest.mark.parametrize(
    ("events", "expected"),
    [
        (
            [
                _event("remote_command_exit", exit_code=137),
                _event("killed_by_runtime_cap", threshold_seconds=60),
                _event("rsync_down_start"),
                _event("rsync_down_done"),
            ],
            "killed_by_runtime_cap",
        ),
        (
            [
                _event(
                    "remote_command_stalled",
                    action="terminate",
                    signalled=True,
                    alive_after=False,
                ),
                _event("remote_command_exit", exit_code=137),
                _event("rsync_down_start"),
            ],
            "remote_command_stalled",
        ),
        (
            [
                _event("orchestrator_signalled", signal="SIGTERM"),
                _event("remote_command_exit", exit_code=137),
                _event("rsync_down_done"),
            ],
            "orchestrator_signalled",
        ),
        (
            [
                json.dumps(
                    {"ts": "2026-09-04T12:00:00Z", "event": "killed_by_user"},
                    separators=(",", ":"),
                ),
                json.dumps(
                    {
                        "ts": "2026-09-04T12:00:01Z",
                        "event": "remote_command_exit",
                        "exit_code": 137,
                    },
                    separators=(",", ":"),
                ),
                _event("rsync_down_done"),
            ],
            "remote_command_exit",
        ),
        (
            [
                _event("killed_by_user", signalled=True, alive_after=False),
                _event("remote_command_exit", exit_code=137),
                _event("rsync_down_done"),
            ],
            "killed_by_user",
        ),
        (
            [
                _event("killed_by_user", signalled=True, alive_after=True),
                _event("remote_command_exit", exit_code=0),
                _event("rsync_down_done"),
            ],
            "remote_command_exit",
        ),
        (
            [
                _event("runtime_cap_reached", action="terminate_failed", alive_after=True),
                _event("remote_command_exit", exit_code=137),
                _event("rsync_down_done"),
            ],
            "runtime_cap_reached",
        ),
        (
            [
                _event("runtime_cap_reached", action="cleanup_unconfirmed"),
                _event("remote_command_exit", exit_code=137),
                _event("rsync_down_done"),
            ],
            "runtime_cap_reached",
        ),
        (
            [
                _event("remote_command_stalled", action="terminate"),
                _event("remote_command_exit", exit_code=0),
                _event("rsync_down_done"),
            ],
            "remote_command_exit",
        ),
        (
            [
                _event("remote_command_stalled", action="diagnose"),
                _event("remote_command_exit", exit_code=0),
                _event("rsync_down_done"),
            ],
            "remote_command_exit",
        ),
    ],
)
def test_status_probe_selects_the_causal_run_event(tmp_path, events, expected):
    (tmp_path / "events.ndjson").write_text("\n".join(events) + "\n")
    (tmp_path / "heartbeat.ndjson").write_text("")

    completed = subprocess.run(
        ["bash", "-c", runs._status_probe_command(str(tmp_path))],
        check=True,
        capture_output=True,
        text=True,
    )
    sections = sc.parse_probe_sections(completed.stdout)
    sections.update(runs._status_event_sections(sections["EVENTS"]))

    assert json.loads(sections["LAST_EVENT"])["event"] == expected
    assert json.loads(sections["LAST_SYNC_EVENT"])["event"].startswith("rsync_down_")
    assert sections["EVENT_COUNT"] == str(len(events))


def test_status_renders_output_sync_separately_from_the_run_outcome():
    rendered = runs._format_status(
        target="box",
        manifest={},
        sections={
            "LAST_EVENT": _event("killed_by_runtime_cap", threshold_seconds=60),
            "LAST_SYNC_EVENT": _event("rsync_down_done"),
            "LAST_HEARTBEAT": "",
            "EVENT_COUNT": "4",
        },
    )

    assert "last event: killed_by_runtime_cap threshold_seconds=60" in rendered
    assert "output sync: completed" in rendered


def test_status_does_not_claim_a_start_only_sync_is_active():
    rendered = runs._format_status(
        target="box",
        manifest={},
        sections={
            "LAST_EVENT": _event("remote_command_exit", exit_code=0),
            "LAST_SYNC_EVENT": _event("rsync_down_start"),
            "LAST_HEARTBEAT": "",
            "EVENT_COUNT": "2",
        },
    )

    assert "output sync: started (completion unknown)" in rendered
    assert "in progress" not in rendered


def test_status_parser_tolerates_blank_malformed_and_non_object_records():
    sections = runs._status_event_sections('\nnot-json\n["valid JSON, wrong shape"]\n')

    assert sections == {
        "LAST_EVENT": '["valid JSON, wrong shape"]',
        "LAST_SYNC_EVENT": "",
        "EVENT_COUNT": "2",
    }

    rendered = runs._format_status(
        target="box",
        manifest={},
        sections={
            "LAST_EVENT": sections["LAST_EVENT"],
            "LAST_SYNC_EVENT": "[]",
            "EVENT_COUNT": sections["EVENT_COUNT"],
        },
    )
    assert "last event (unparsed)" in rendered
    assert "output sync (unparsed)" in rendered


def _remote_run():
    return sc.make_remote_run_context(backend="ssh", target="box", function_name="train")


def test_rsync_down_records_started_and_completed(tmp_path):
    remote_run = _remote_run()
    with (
        mock.patch.object(sc, "run_local"),
        mock.patch.object(sc, "_record_remote_event") as record,
    ):
        sc.rsync_down("box", tmp_path, remote_run=remote_run)

    assert [call.args[2] for call in record.call_args_list] == [
        "rsync_down_start",
        "rsync_down_done",
    ]


def test_rsync_down_records_failure_without_replacing_it(tmp_path):
    remote_run = _remote_run()
    failure = subprocess.CalledProcessError(23, "rsync")
    with (
        mock.patch.object(sc, "run_local", side_effect=failure),
        mock.patch.object(sc, "_record_remote_event") as record,
        pytest.raises(subprocess.CalledProcessError) as raised,
    ):
        sc.rsync_down("box", tmp_path, remote_run=remote_run)

    assert raised.value is failure
    assert [call.args[2] for call in record.call_args_list] == [
        "rsync_down_start",
        "rsync_down_failed",
    ]
    assert record.call_args.kwargs["exit_code"] == 23


def test_build_failure_closes_the_started_phase():
    remote_run = _remote_run()
    failure = subprocess.CalledProcessError(1, "ssh")
    image = Image.from_registry("ubuntu:22.04")
    with (
        mock.patch.object(sc, "ssh_exec", side_effect=failure),
        mock.patch.object(sc, "_record_remote_event") as record,
        pytest.raises(subprocess.CalledProcessError) as raised,
    ):
        sc.build_image("box", image, remote_run=remote_run)

    assert raised.value is failure
    assert [call.args[2] for call in record.call_args_list] == [
        "build_image_start",
        "build_image_failed",
    ]


def test_docker_run_failure_records_container_launch_failed():
    remote_run = _remote_run()
    failure = subprocess.CalledProcessError(125, "ssh")
    function = SimpleNamespace(name="train", env={})
    with (
        mock.patch.object(sc, "ssh_exec", side_effect=failure),
        mock.patch.object(sc, "_record_remote_event") as record,
        pytest.raises(subprocess.CalledProcessError) as raised,
    ):
        sc.run_container_detached(
            target="box",
            container_name="runplz-train-abc123",
            function=function,
            rel_script="job.py",
            args=[],
            kwargs={},
            gpu_flag="",
            remote_run=remote_run,
        )

    assert raised.value is failure
    assert record.call_args.args[2] == "container_launch_failed"
    assert record.call_args.kwargs["container"] == "runplz-train-abc123"


def _app(tmp_path, *, preconditions=None):
    repo = tmp_path / "repo"
    repo.mkdir()
    job = repo / "job.py"
    job.write_text("# job\n")
    app = App("demo")

    @app.function(
        image=Image.from_registry("ubuntu:22.04"),
        preconditions=preconditions,
    )
    def train():  # pragma: no cover - never executed
        pass

    function = app.functions["train"]
    function.module_file = str(job)
    app.repo_root = repo
    return app, function


def test_precondition_failure_is_recorded_and_salvaged(tmp_path):
    app, function = _app(tmp_path, preconditions={"gpu_count": 4})
    failure = sc.PreconditionFailed("gpu_count too small")
    with mock.patch.multiple(
        sc,
        prepare_remote_run=mock.DEFAULT,
        ensure_remote_rsync=mock.DEFAULT,
        rsync_up=mock.DEFAULT,
        check_preconditions=mock.Mock(side_effect=failure),
        fetch_failure_tail=mock.Mock(return_value=""),
        rsync_down=mock.DEFAULT,
        build_remote_run_manifest=mock.Mock(return_value={}),
        _record_remote_event=mock.DEFAULT,
    ) as patched:
        with pytest.raises(sc.PreconditionFailed) as raised:
            sc.dispatch_to_target(
                app=app,
                function=function,
                args=[],
                kwargs={},
                target="box",
                backend="ssh",
                mode="docker",
            )

    assert raised.value is failure
    patched["rsync_down"].assert_called_once()
    assert patched["_record_remote_event"].call_args.args[2] == "precondition_failed"


@pytest.mark.parametrize("backend", ["ssh", "brev", "gcp", "aws"])
def test_orchestrator_signal_is_salvaged_before_backend_teardown(tmp_path, backend):
    app, function = _app(tmp_path)
    interruption = sc.OrchestratorKilled("terminated", signal_name="SIGTERM")
    observed = []
    remote_events = []
    local_events = []

    def record(*args, **kwargs):
        observed.append(("event", args[2], kwargs))
        remote_events.append((args[2], kwargs))

    def sync(*args, **kwargs):
        observed.append(("sync",))
        local_events[:] = remote_events

    def capture(*args, **kwargs):
        observed.append(("container_cleanup",))
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
        stream_and_wait=mock.Mock(side_effect=interruption),
        fetch_failure_tail=mock.Mock(return_value=""),
        rsync_down=sync,
        ssh_capture=capture,
        build_remote_run_manifest=mock.Mock(return_value={}),
        _record_remote_event=record,
    ):
        with pytest.raises(sc.OrchestratorKilled) as raised:
            sc.dispatch_to_target(
                app=app,
                function=function,
                args=[],
                kwargs={},
                target="box",
                backend=backend,
                mode="docker",
            )

    assert raised.value is interruption
    assert [item[0] for item in observed] == ["event", "sync", "container_cleanup"]
    assert observed[0][1] == "orchestrator_signalled"
    assert observed[0][2]["signal"] == "SIGTERM"

    # Provisioned backends now tear the remote host down. The salvaged local
    # stream must already contain the signal even after its remote copy is gone.
    if backend != "ssh":
        remote_events.clear()
    assert local_events == [("orchestrator_signalled", {"ssh_opts": None, "signal": "SIGTERM"})]
