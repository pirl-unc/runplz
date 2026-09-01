"""Coverage for ``runplz tail`` / ``runplz status`` (issue #57)."""

import json
from pathlib import Path
from unittest import mock

import pytest

from runplz import cli, runs


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
    fake_stdout = (
        "---LAST_EVENT---\n"
        f"{last_event}\n"
        "---LAST_HEARTBEAT---\n"
        f"{last_hb}\n"
        "---EVENT_COUNT---\n"
        "12\n"
        "---END---\n"
    )
    fake = mock.Mock(returncode=0, stdout=fake_stdout, stderr="")
    with mock.patch("runplz.runs.subprocess.run", return_value=fake):
        rc = runs.status(outputs_dir=tmp_path, host_override=None, run_id_override=None)
    assert rc == 0
    out = capsys.readouterr().out
    assert "target: my-gpu-box" in out
    assert "function: train" in out
    assert "container_started" in out
    assert "events recorded: 12" in out


def test_status_handles_empty_event_log(tmp_path, capsys):
    _write_manifest(tmp_path, _manifest())
    fake_stdout = "---LAST_EVENT---\n---LAST_HEARTBEAT---\n---EVENT_COUNT---\n0\n---END---\n"
    fake = mock.Mock(returncode=0, stdout=fake_stdout, stderr="")
    with mock.patch("runplz.runs.subprocess.run", return_value=fake):
        runs.status(outputs_dir=tmp_path, host_override=None, run_id_override=None)
    out = capsys.readouterr().out
    assert "last event: (none recorded)" in out
    assert "last heartbeat: (none yet)" in out


def test_status_returns_ssh_failure_code(tmp_path, capsys):
    _write_manifest(tmp_path, _manifest())
    fake = mock.Mock(returncode=255, stdout="", stderr="ssh: connect refused")
    with mock.patch("runplz.runs.subprocess.run", return_value=fake):
        rc = runs.status(outputs_dir=tmp_path, host_override=None, run_id_override=None)
    assert rc == 255
    out = capsys.readouterr().out
    assert "ssh to my-gpu-box failed" in out


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
