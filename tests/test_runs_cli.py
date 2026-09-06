"""Coverage for ``runplz tail`` / ``runplz status`` (issue #57)."""

import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import pytest

from runplz import cli, runs
from runplz.backends import ssh_common


@pytest.mark.parametrize(
    "timestamp", [123, [], {}, None, "", "2026-99-01T00:00:00Z", "2026-02-30T00:00:00Z"]
)
def test_malformed_timestamps_do_not_break_status_or_kill(timestamp):
    raw = json.dumps({"ts": timestamp, "event": "remote_command_exit", "exit_code": 0})
    rendered = runs._format_status(
        target="box",
        manifest={},
        sections={
            "LAST_EVENT": raw,
            "LAST_SYNC_EVENT": raw,
            "LAST_HEARTBEAT": raw,
        },
    )
    assert "last event: remote_command_exit exit_code=0" in rendered
    assert "last heartbeat:" in rendered
    assert runs._parse_iso_z(timestamp) is None
    assert "heartbeat:" in runs._format_kill(
        target="box", run_id="run", fields={}, sections={"HEARTBEAT": raw}
    )


@pytest.mark.parametrize("failure", ["connection", "timeout", "missing_ssh"])
@pytest.mark.parametrize("explicit", [False, True])
def test_status_falls_back_to_matching_local_snapshot(tmp_path, capsys, failure, explicit):
    manifest = _manifest()
    _write_manifest(tmp_path, manifest)
    meta = tmp_path / ".runplz"
    (meta / "events.ndjson").write_text(
        "bad-json\n[]\n"
        + json.dumps(
            {
                "event": "killed_by_user",
                "run_id": "another-run",
                "signalled": True,
                "alive_after": False,
            }
        )
        + "\n"
        + json.dumps(
            {"event": "orchestrator_signalled", "run_id": manifest["run_id"], "signal": "SIGTERM"}
        )
        + "\n"
        + json.dumps({"event": "rsync_down_done", "run_id": manifest["run_id"]})
        + "\n"
    )
    (meta / "heartbeat.ndjson").write_text(
        "bad-json\n[]\n"
        + json.dumps({"ts": "2026-09-05T01:02:03Z", "run_id": manifest["run_id"]})
        + "\n"
        + json.dumps({"ts": "1999-01-01T00:00:00Z", "run_id": "another-run"})
        + "\n"
    )

    def fail(cmd, **kwargs):
        assert kwargs["timeout"] == runs._STATUS_TIMEOUT_S
        if failure == "timeout":
            raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])
        if failure == "missing_ssh":
            raise FileNotFoundError("ssh")
        return mock.Mock(returncode=255, stdout="", stderr="host deleted")

    with mock.patch.object(runs.subprocess, "run", side_effect=fail):
        rc = runs.status(
            outputs_dir=tmp_path,
            host_override=manifest["target"] if explicit else None,
            run_id_override=manifest["run_id"] if explicit else None,
        )
    assert rc == 0
    rendered = capsys.readouterr().out
    assert "source: local snapshot" in rendered
    assert "not live status" in rendered
    assert "last event: orchestrator_signalled signal=SIGTERM" in rendered
    assert "output sync: completed" in rendered
    assert "events recorded: 2" in rendered
    assert "1999" not in rendered


@pytest.mark.parametrize("host,run_id", [("different-box", None), ("my-gpu-box", "different-run")])
def test_status_never_borrows_a_different_hosts_or_runs_snapshot(tmp_path, capsys, host, run_id):
    _write_manifest(tmp_path, _manifest())
    (tmp_path / ".runplz" / "events.ndjson").write_text('{"event":"remote_command_exit"}\n')
    with mock.patch.object(
        runs.subprocess, "run", return_value=mock.Mock(returncode=255, stderr="", stdout="")
    ):
        rc = runs.status(outputs_dir=tmp_path, host_override=host, run_id_override=run_id)
    assert rc == 255
    assert "source: local snapshot" not in capsys.readouterr().out


@pytest.mark.parametrize("explicit_run_id", [False, True])
@pytest.mark.parametrize(
    "recorded_port,extra_args,probe_port,allow_snapshot",
    [
        (2222, ["--ssh-port", "2223"], 2223, False),
        (2222, ["--ssh-port", "2222"], 2222, True),
        (2222, [], 2222, True),
        (None, ["--ssh-port", "2222"], 2222, False),
        # Unspecified ports come from SSH config, not necessarily port 22.
        (None, ["--ssh-port", "22"], 22, False),
        (None, [], None, True),
        (22, [], 22, True),
        (2222, ["--ssh-key", "/keys/replacement.pem"], 2222, True),
    ],
)
def test_cli_status_snapshot_requires_the_recorded_ssh_port(
    tmp_path, capsys, explicit_run_id, recorded_port, extra_args, probe_port, allow_snapshot
):
    manifest = _manifest(target="localhost")
    _write_manifest(tmp_path, manifest)
    ssh_common.write_local_ssh_options(tmp_path, ssh_common.SshOptions(port=recorded_port))
    (tmp_path / ".runplz" / "events.ndjson").write_text(
        json.dumps({"event": "remote_command_exit", "run_id": manifest["run_id"], "exit_code": 0})
        + "\n"
    )

    def unavailable(cmd, **kwargs):
        assert kwargs["timeout"] == runs._STATUS_TIMEOUT_S
        assert cmd[-2] == "localhost"
        if probe_port is None:
            assert "-p" not in cmd
        else:
            assert cmd[cmd.index("-p") + 1] == str(probe_port)
        return subprocess.CompletedProcess(cmd, 255, stdout="", stderr="connection refused")

    args = ["status", "--outputs-dir", str(tmp_path), "--host", "localhost", *extra_args]
    if explicit_run_id:
        args.extend(["--run-id", manifest["run_id"]])
    with mock.patch.object(runs.subprocess, "run", side_effect=unavailable):
        rc = cli.main(args)

    assert rc == (0 if allow_snapshot else 255)
    rendered = capsys.readouterr().out
    assert ("source: local snapshot" in rendered) is allow_snapshot
    assert ("last event: remote_command_exit" in rendered) is allow_snapshot


@pytest.mark.parametrize("events", ["", "not-json\n[]\n", '{"event":"old","run_id":"other"}\n'])
def test_status_rejects_empty_or_unrelated_snapshot(tmp_path, events):
    manifest = _manifest()
    _write_manifest(tmp_path, manifest)
    (tmp_path / ".runplz" / "events.ndjson").write_text(events)
    assert (
        runs._local_status_snapshot(
            tmp_path, manifest["target"], manifest["remote_paths"]["meta"], None
        )
        is None
    )


def test_status_snapshot_can_read_legacy_events_without_heartbeat(tmp_path):
    manifest = _manifest()
    _write_manifest(tmp_path, manifest)
    (tmp_path / ".runplz" / "events.ndjson").write_text('{"event":"remote_command_exit"}\n')
    saved, sections = runs._local_status_snapshot(
        tmp_path, manifest["target"], manifest["remote_paths"]["meta"], None
    )
    assert saved == manifest
    assert sections["LAST_HEARTBEAT"] == ""


def test_status_snapshot_rejects_inconsistent_manifest_run_id(tmp_path):
    manifest = _manifest(run_id="different-run")
    _write_manifest(tmp_path, manifest)
    assert (
        runs._local_status_snapshot(
            tmp_path, manifest["target"], manifest["remote_paths"]["meta"], "requested-run"
        )
        is None
    )


@pytest.mark.parametrize(
    "delta, expected",
    [
        (timedelta(seconds=10), "10s ago"),
        (timedelta(minutes=2, seconds=3), "2m 3s ago"),
        (timedelta(hours=2, minutes=4), "2h 4m ago"),
        (timedelta(seconds=-10), ""),
    ],
)
def test_age_rendering_uses_human_scale_boundaries(delta, expected):
    timestamp = (datetime.now(timezone.utc) - delta).strftime("%Y-%m-%dT%H:%M:%SZ")
    rendered = runs._age_str(timestamp)
    # Timestamps are serialized to whole seconds; a slow test process can
    # cross exactly one second between formatting and parsing. Accept that
    # representation-level rounding without weakening the unit boundaries.
    if delta == timedelta(seconds=10):
        assert rendered in {" (10s ago)", " (11s ago)"}
    else:
        assert expected in rendered


def _write_manifest(outputs_dir: Path, manifest: dict) -> Path:
    meta = outputs_dir / ".runplz"
    meta.mkdir(parents=True, exist_ok=True)
    p = meta / "run.json"
    p.write_text(json.dumps(manifest))
    return p


def _manifest(**overrides) -> dict:
    base = {
        "run_id": "20260427T010203Z-myhost-train-deadbeef",
        "started_at": "2026-04-27T01:02:03Z",
        "backend": "brev",
        "target": "my-gpu-box",
        "function": "train",
        "remote_paths": {
            "out": "~/runplz-runs/20260427T010203Z-myhost-train-deadbeef/out",
            "meta": "~/runplz-runs/20260427T010203Z-myhost-train-deadbeef/out/.runplz",
        },
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# resolve_target_and_meta


def test_resolve_uses_manifest_target_and_meta(tmp_path):
    _write_manifest(tmp_path, _manifest())
    target, meta, manifest = runs.resolve_target_and_meta(
        outputs_dir=tmp_path, host_override=None, run_id_override=None
    )
    assert target == "my-gpu-box"
    assert meta.endswith("/out/.runplz")
    assert manifest["function"] == "train"


def test_resolve_host_override_wins(tmp_path):
    _write_manifest(tmp_path, _manifest())
    target, _, _ = runs.resolve_target_and_meta(
        outputs_dir=tmp_path, host_override="other-box", run_id_override=None
    )
    assert target == "other-box"


def test_resolve_run_id_requires_host(tmp_path):
    with pytest.raises(RuntimeError, match="--run-id requires --host"):
        runs.resolve_target_and_meta(
            outputs_dir=tmp_path, host_override=None, run_id_override="xyz"
        )


def test_resolve_run_id_with_host_skips_manifest(tmp_path):
    target, meta, manifest = runs.resolve_target_and_meta(
        outputs_dir=tmp_path, host_override="some-box", run_id_override="rid-123"
    )
    assert target == "some-box"
    assert meta == "~/runplz-runs/rid-123/out/.runplz"
    assert manifest == {}


def test_resolve_falls_back_when_manifest_missing_meta(tmp_path):
    _write_manifest(tmp_path, _manifest(remote_paths={}))
    target, meta, _ = runs.resolve_target_and_meta(
        outputs_dir=tmp_path, host_override=None, run_id_override=None
    )
    assert target == "my-gpu-box"
    assert meta.endswith("/out/.runplz")


def test_resolve_raises_clean_when_no_manifest(tmp_path):
    with pytest.raises(runs.ManifestNotFound):
        runs.resolve_target_and_meta(outputs_dir=tmp_path, host_override=None, run_id_override=None)


@pytest.mark.parametrize(
    "manifest, message",
    [
        ({"target": "box", "remote_paths": {}, "run_id": ""}, "missing both"),
        ({"remote_paths": {}}, "no target host"),
    ],
)
def test_resolve_rejects_incomplete_manifest(tmp_path, manifest, message):
    _write_manifest(tmp_path, manifest)
    with pytest.raises(RuntimeError, match=message):
        runs.resolve_target_and_meta(outputs_dir=tmp_path, host_override=None, run_id_override=None)


# ---------------------------------------------------------------------------
# tail


def test_tail_invokes_remote_tail_n(tmp_path):
    _write_manifest(tmp_path, _manifest())
    fake = mock.Mock(returncode=0)
    with mock.patch("runplz.runs.subprocess.run", return_value=fake) as run_mock:
        rc = runs.tail(
            outputs_dir=tmp_path,
            host_override=None,
            run_id_override=None,
            lines=50,
            follow=False,
        )
    assert rc == 0
    cmd = run_mock.call_args.args[0]
    assert cmd[0] == "ssh"
    # remote_cmd is the last arg.
    assert "tail -n 50" in cmd[-1]
    assert "/out/.runplz/last.log" in cmd[-1]


def test_tail_follow_uses_dash_F(tmp_path):
    _write_manifest(tmp_path, _manifest())
    fake = mock.Mock(returncode=0)
    with mock.patch("runplz.runs.subprocess.run", return_value=fake) as run_mock:
        runs.tail(
            outputs_dir=tmp_path,
            host_override=None,
            run_id_override=None,
            lines=120,
            follow=True,
        )
    cmd = run_mock.call_args.args[0]
    assert "tail -F" in cmd[-1]


# ---------------------------------------------------------------------------
# status


def test_status_summarizes_last_event_and_heartbeat(tmp_path, capsys):
    _write_manifest(tmp_path, _manifest())
    last_event = json.dumps(
        {
            "ts": "2026-04-27T01:05:00Z",
            "run_id": "x",
            "event": "container_started",
        }
    )
    last_hb = json.dumps({"ts": "2026-04-27T01:05:30Z", "run_id": "x"})
    fake_stdout = f"---EVENTS---\n{last_event}\n---LAST_HEARTBEAT---\n{last_hb}\n---END---\n"
    fake = mock.Mock(returncode=0, stdout=fake_stdout, stderr="")
    with mock.patch("runplz.runs.subprocess.run", return_value=fake):
        rc = runs.status(outputs_dir=tmp_path, host_override=None, run_id_override=None)
    assert rc == 0
    out = capsys.readouterr().out
    assert "target: my-gpu-box" in out
    assert "function: train" in out
    assert "container_started" in out
    assert "events recorded: 1" in out


def test_status_handles_empty_event_log(tmp_path, capsys):
    _write_manifest(tmp_path, _manifest())
    fake_stdout = "---EVENTS---\n---LAST_HEARTBEAT---\n---END---\n"
    fake = mock.Mock(returncode=0, stdout=fake_stdout, stderr="")
    with mock.patch("runplz.runs.subprocess.run", return_value=fake):
        runs.status(outputs_dir=tmp_path, host_override=None, run_id_override=None)
    out = capsys.readouterr().out
    assert "last event: (none recorded)" in out
    assert "last heartbeat: (none yet)" in out


def test_status_formats_exit_code_and_trailing_section(capsys):
    rendered = runs._format_status(
        target="box",
        manifest={},
        sections={
            "LAST_EVENT": json.dumps({"event": "finished", "ts": "bad", "exit_code": 17}),
            "LAST_HEARTBEAT": "",
            "EVENT_COUNT": "3",
        },
    )
    assert "finished exit_code=17" in rendered
    assert "last heartbeat: (none yet)" in rendered
    assert ssh_common.parse_probe_sections("---TRAILING---\nvalue\n")["TRAILING"] == "value"


def test_status_returns_ssh_failure_code(tmp_path, capsys):
    _write_manifest(tmp_path, _manifest())
    fake = mock.Mock(returncode=255, stdout="", stderr="ssh: connect refused")
    with mock.patch("runplz.runs.subprocess.run", return_value=fake):
        rc = runs.status(outputs_dir=tmp_path, host_override=None, run_id_override=None)
    assert rc == 255
    out = capsys.readouterr().out
    assert "ssh to my-gpu-box failed" in out


@pytest.mark.parametrize(
    "heartbeat, expected",
    [
        ("not-json", "(unparsed)"),
        (json.dumps({}), "(unparsed)"),
        (json.dumps({"ts": "bad"}), "bad"),
    ],
)
def test_status_renders_malformed_heartbeat_without_failing(tmp_path, capsys, heartbeat, expected):
    _write_manifest(tmp_path, _manifest())
    fake = mock.Mock(
        returncode=0,
        stdout=f"---LAST_EVENT---\nnot-json\n---LAST_HEARTBEAT---\n{heartbeat}\n"
        "---EVENT_COUNT---\n0\n---END---\n",
        stderr="",
    )
    with mock.patch("runplz.runs.subprocess.run", return_value=fake):
        assert runs.status(outputs_dir=tmp_path, host_override=None, run_id_override=None) == 0
    out = capsys.readouterr().out
    assert "last event (unparsed)" in out
    assert f"last heartbeat: {expected}" in out


# ---------------------------------------------------------------------------
# CLI integration


def test_cli_tail_dispatch(tmp_path):
    _write_manifest(tmp_path, _manifest())
    with mock.patch.object(runs, "tail", return_value=0) as tail_mock:
        rc = cli.main(["tail", "--outputs-dir", str(tmp_path), "-n", "10"])
    assert rc == 0
    kwargs = tail_mock.call_args.kwargs
    assert kwargs["lines"] == 10
    assert kwargs["follow"] is False
    assert kwargs["outputs_dir"] == tmp_path.resolve()


def test_cli_tail_follow_flag(tmp_path):
    _write_manifest(tmp_path, _manifest())
    with mock.patch.object(runs, "tail", return_value=0) as tail_mock:
        cli.main(["tail", "--outputs-dir", str(tmp_path), "-f"])
    assert tail_mock.call_args.kwargs["follow"] is True


def test_cli_status_dispatch(tmp_path):
    _write_manifest(tmp_path, _manifest())
    with mock.patch.object(runs, "status", return_value=0) as status_mock:
        rc = cli.main(["status", "--outputs-dir", str(tmp_path)])
    assert rc == 0
    status_mock.assert_called_once()


def test_cli_tail_surfaces_missing_manifest(tmp_path, capsys):
    rc = cli.main(["tail", "--outputs-dir", str(tmp_path)])
    assert rc == 1
    assert "No run manifest" in capsys.readouterr().err


def test_cli_status_run_id_without_host_errors(tmp_path, capsys):
    rc = cli.main(["status", "--outputs-dir", str(tmp_path), "--run-id", "xyz"])
    assert rc == 2
    assert "--run-id requires --host" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# remote path expansion


def test_remote_shell_path_rewrites_tilde_to_home():
    """`~` expands before quoting, so a `~/` path is literal in the remote shell."""
    assert runs.remote_shell_path("~/runplz-runs/r1/out") == "$HOME/runplz-runs/r1/out"
    assert runs.remote_shell_path("/abs/path") == "/abs/path"
    assert runs.remote_shell_path("$HOME/already") == "$HOME/already"


@pytest.mark.parametrize(
    "call",
    [
        lambda d: runs.tail(
            outputs_dir=d, host_override=None, run_id_override=None, lines=5, follow=False
        ),
        lambda d: runs.status(outputs_dir=d, host_override=None, run_id_override=None),
        lambda d: runs.kill(outputs_dir=d, host_override=None, run_id_override=None),
    ],
    ids=["tail", "status", "kill"],
)
def test_remote_commands_never_ship_an_unexpandable_tilde(tmp_path, call):
    _write_manifest(tmp_path, _manifest())
    fake = mock.Mock(returncode=0, stdout="---SUMMARY---\n---END---\n", stderr="")
    with mock.patch("runplz.runs.subprocess.run", return_value=fake) as run_mock:
        call(tmp_path)
    remote_cmd = run_mock.call_args.args[0][-1]
    assert "~/" not in remote_cmd
    assert "$HOME/runplz-runs/" in remote_cmd


def test_resolve_rejects_a_run_id_that_could_escape_the_path():
    with pytest.raises(RuntimeError, match="not a valid run id"):
        runs.resolve_target_and_meta(
            outputs_dir=Path("."),
            host_override="box",
            run_id_override='x"; rm -rf ~; echo "',
        )


def test_kill_drops_a_tampered_run_id_from_the_remote_command(tmp_path):
    _write_manifest(tmp_path, _manifest(run_id='evil"; touch /tmp/pwned; echo "'))
    fake = mock.Mock(returncode=0, stdout="---SUMMARY---\n---END---\n", stderr="")
    with mock.patch("runplz.runs.subprocess.run", return_value=fake) as run_mock:
        runs.kill(outputs_dir=tmp_path, host_override=None, run_id_override=None)
    assert "pwned" not in run_mock.call_args.args[0][-1]
