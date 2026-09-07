"""Offline lifecycle tests: execute the generated entrypoint, never launch Modal."""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import modal as modal_sdk
import pytest
from modal_proto import api_pb2

from runplz import App, Image, ModalConfig, cli, runs
from runplz.backends import modal as backend
from runplz.backends import modal_runs


@pytest.fixture
def sdk(monkeypatch):
    volume = mock.Mock(object_id="vo-original")
    volume.read_file.side_effect = FileNotFoundError
    volume.iterdir.return_value = []
    call = mock.Mock(object_id="fc-call")
    call.get.side_effect = TimeoutError
    fake = SimpleNamespace(
        Volume=mock.Mock(),
        FunctionCall=mock.Mock(),
        exception=modal_sdk.exception,
        Image=mock.Mock(),
    )
    fake.Volume.from_name.return_value = volume
    fake.FunctionCall.from_id.return_value = call
    monkeypatch.setitem(sys.modules, "modal", fake)
    monkeypatch.setitem(
        sys.modules, "modal.config", SimpleNamespace(config=SimpleNamespace(get=lambda key: "dev"))
    )
    monkeypatch.setitem(
        sys.modules,
        "modal.volume",
        SimpleNamespace(FileEntryType=SimpleNamespace(FILE=1, DIRECTORY=2)),
    )
    monkeypatch.setenv("MODAL_ENVIRONMENT", "dev")
    return fake, volume, call


@pytest.fixture
def receipt(tmp_path):
    path, data = modal_runs.prepare_run(tmp_path, "demo", "job", "outputs")
    data.update(
        app_id="ap-app",
        function_id="fu-function",
        call_id="fc-call",
        volume_id="vo-original",
        environment="dev",
        launch_state="submitted",
    )
    modal_runs._write(path, data)
    return path, data


@pytest.fixture
def staging(receipt):
    with tempfile.TemporaryDirectory(dir=receipt[0].parent) as directory:
        yield Path(directory)


def _app(tmp_path, *, detach=True, volumes=None):
    app = App("demo", modal_config=ModalConfig(detach=detach))
    app.repo_root = tmp_path

    @app.function(image=Image.from_registry("python:3.12"), volumes=volumes)
    def job():
        raise AssertionError("Never execute the user's job in a launcher test")

    job.module_file = str(tmp_path / "job.py")
    return app, job


def _entrypoint_runner(sdk, captured):
    fake, volume, call = sdk

    class FakeApp:
        app_id = "ap-app"

        def __init__(self, name):
            self.name = name

        def function(self, **kwargs):
            def wrap(body):
                runner = mock.Mock(object_id="fu-function")
                runner.spawn.return_value = call
                runner.remote.side_effect = AssertionError("Detached launch waited for the job")
                captured.update(body=body, runner=runner, options=kwargs)
                return runner

            return wrap

        def local_entrypoint(self):
            return lambda function: function

    fake.App = FakeApp

    def run(cmd, **kwargs):
        assert cmd[:3] == ["modal", "run", "--detach"]
        path = Path(cmd[-1].split("::")[0])
        namespace = {}
        source = path.read_text()
        exec(compile(source, str(path), "exec"), namespace)
        captured.update(path=path, source=source, namespace=namespace)
        namespace["main"]()
        return subprocess.CompletedProcess(cmd, 0)

    return run


def test_detached_launch_spawns_records_ids_and_never_downloads(tmp_path, sdk, monkeypatch):
    app, fn = _app(tmp_path, volumes={"/out": "outputs"})
    captured = {}
    launch = mock.Mock(side_effect=_entrypoint_runner(sdk, captured))
    monkeypatch.setattr(backend.subprocess, "run", launch)
    result = backend.run(app, fn, [1], {"epochs": 3})
    assert result == modal_runs._read(tmp_path / "out")
    assert result["call_id"] == "fc-call"
    assert result["volume_id"] == "vo-original"
    assert result["environment"] == "dev"
    assert result["launch_state"] == "submitted"
    assert captured["namespace"]["_CONTAINER_ENV"]["RUNPLZ_OUT"] == "/out" + result["remote_path"]
    captured["runner"].spawn.assert_called_once_with()
    captured["runner"].remote.assert_not_called()
    sdk[1].read_file.assert_not_called()
    launch.assert_called_once()
    assert not captured["path"].exists()


@pytest.mark.parametrize("code", [0, 17, -9])
def test_remote_wrapper_commits_success_and_partial_failed_outputs(
    tmp_path, sdk, monkeypatch, code
):
    app, fn = _app(tmp_path, volumes={"/out": "outputs"})
    captured = {}
    monkeypatch.setattr(backend.subprocess, "run", _entrypoint_runner(sdk, captured))
    receipt = backend.run(app, fn, [], {})
    namespace = captured["namespace"]
    remote_out = tmp_path / "remote"
    namespace["_CONTAINER_ENV"]["RUNPLZ_OUT"] = str(remote_out)
    namespace["subprocess"] = SimpleNamespace(run=lambda cmd: SimpleNamespace(returncode=code))
    assert captured["body"]() == code
    sdk[1].commit.assert_called_once_with()
    result = json.loads((remote_out / ".runplz/modal-result.json").read_text())
    assert result == {"run_id": receipt["run_id"], "exit_code": code}


@pytest.mark.parametrize("bootstrap_fails", [False, True])
def test_commit_error_is_not_success_and_does_not_mask_bootstrap_error(
    tmp_path, sdk, monkeypatch, bootstrap_fails
):
    app, fn = _app(tmp_path, volumes={"/out": "outputs"})
    captured = {}
    monkeypatch.setattr(backend.subprocess, "run", _entrypoint_runner(sdk, captured))
    backend.run(app, fn, [], {})
    namespace = captured["namespace"]
    namespace["_CONTAINER_ENV"]["RUNPLZ_OUT"] = str(tmp_path / "remote")
    namespace["subprocess"] = SimpleNamespace(
        run=mock.Mock(
            side_effect=OSError("bootstrap") if bootstrap_fails else None,
            return_value=SimpleNamespace(returncode=0),
        )
    )
    sdk[1].commit.side_effect = RuntimeError("commit")
    with pytest.raises((OSError, RuntimeError), match="bootstrap" if bootstrap_fails else "commit"):
        captured["body"]()


@pytest.mark.parametrize("volumes", [None, {"/data": "data"}, {"/out": "x", "/out/nested": "y"}])
def test_detach_requires_unambiguous_persistent_outputs(tmp_path, sdk, monkeypatch, volumes):
    app, fn = _app(tmp_path, volumes=volumes)
    launch = mock.Mock()
    monkeypatch.setattr(backend.subprocess, "run", launch)
    with pytest.raises(ValueError, match="require volumes"):
        backend.run(app, fn, [], {})
    launch.assert_not_called()
    assert not (tmp_path / "out/.runplz/run.json").exists()


def test_detach_rejects_output_env_override_and_non_bool(tmp_path, sdk):
    app, fn = _app(tmp_path, volumes={"/out": "outputs"})
    fn.env["RUNPLZ_OUT"] = "/somewhere/else"
    with pytest.raises(ValueError, match="RUNPLZ_OUT"):
        backend.run(app, fn, [], {})
    with pytest.raises(ValueError, match="bool"):
        backend.run(app, fn, [], {}, detach="yes")


@pytest.mark.parametrize("failure", [RuntimeError("connection lost"), KeyboardInterrupt()])
def test_interrupted_spawn_retains_identity_without_claiming_submission(receipt, sdk, failure):
    path, _ = receipt
    # Start with a prepared receipt, as the real launcher does.
    data = json.loads(path.read_text())
    data.pop("call_id")
    modal_runs._write(path, data)
    runner = mock.Mock(object_id="fu-function")
    runner.spawn.side_effect = failure
    with pytest.raises(type(failure)):
        modal_runs.record_launch(path, SimpleNamespace(app_id="ap-app"), runner, sdk[1])
    saved = modal_runs._read(path.parent.parent)
    assert saved["launch_state"] == "submitting"
    assert saved["app_id"] == "ap-app"
    assert "call_id" not in saved
    runner.spawn.assert_called_once()


@pytest.mark.parametrize(
    "failure", [subprocess.CalledProcessError(1, "modal"), KeyboardInterrupt()]
)
def test_launch_failure_preserves_receipt_and_removes_temporary_entrypoint(
    tmp_path, sdk, monkeypatch, failure
):
    app, fn = _app(tmp_path, volumes={"/out": "outputs"})
    paths = []

    def launch(cmd, **kwargs):
        paths.append(Path(cmd[-1].split("::")[0]))
        raise failure

    monkeypatch.setattr(backend.subprocess, "run", launch)
    with pytest.raises(type(failure)):
        backend.run(app, fn, [], {})
    assert modal_runs._read(tmp_path / "out")["launch_state"] == "prepared"
    assert not paths[0].exists()


def test_missing_submission_receipt_is_not_success(tmp_path, sdk, monkeypatch):
    app, fn = _app(tmp_path, volumes={"/out": "outputs"})
    monkeypatch.setattr(
        backend.subprocess, "run", mock.Mock(return_value=SimpleNamespace(returncode=0))
    )
    with pytest.raises(RuntimeError, match="unconfirmed"):
        backend.run(app, fn, [], {})


def test_existing_receipts_are_never_overwritten(receipt):
    path, data = receipt
    with pytest.raises(ValueError, match="new --outputs-dir"):
        modal_runs.prepare_run(path.parent.parent, "new", "new", "other")
    assert json.loads(path.read_text()) == data


def test_output_paths_are_isolated_between_runs(tmp_path):
    _, first = modal_runs.prepare_run(tmp_path / "first", "demo", "job", "outputs")
    _, second = modal_runs.prepare_run(tmp_path / "second", "demo", "job", "outputs")
    assert first["remote_path"] != second["remote_path"]


@pytest.mark.parametrize("value", [0, 1, "true", None])
def test_modal_config_validates_bool(value):
    with pytest.raises(ValueError, match="bool"):
        ModalConfig(detach=value)


def test_bind_detach_override_and_reset(tmp_path):
    app, _ = _app(tmp_path, volumes={"/out": "outputs"})
    app.bind("modal", detach=False)
    assert app._backend_kwargs["detach"] is False
    app.bind("modal")
    assert "detach" not in app._backend_kwargs
    assert app.modal_config.detach is True
    with pytest.raises(ValueError, match="bool"):
        app.bind("modal", detach=1)


@pytest.mark.parametrize("backend_name", ["local", "brev", "ssh", "gcp", "aws"])
@pytest.mark.parametrize("detach", [False, True])
def test_other_backends_reject_detach(tmp_path, backend_name, detach):
    app, _ = _app(tmp_path)
    with pytest.raises(ValueError, match="only applies to the modal"):
        app.bind(backend_name, detach=detach)


@pytest.mark.parametrize(
    "flag, expected", [([], None), (["--detach"], True), (["--no-detach"], False)]
)
def test_cli_threads_detach_override(tmp_path, monkeypatch, flag, expected):
    app, _ = _app(tmp_path)
    app.entrypoint = lambda: None
    script = tmp_path / "job.py"
    script.write_text("# loaded by test")
    monkeypatch.setattr(cli, "_load_app", lambda _: app)
    bind = mock.Mock()
    monkeypatch.setattr(app, "bind", bind)
    cli.main(["modal", str(script), "--no-log-file", *flag])
    assert bind.call_args.kwargs["detach"] is expected


@pytest.mark.parametrize(
    "change",
    [
        {"backend": "ssh"},
        {"mode": "attached"},
        {"schema_version": 2},
        {"run_id": []},
        {"remote_path": "/"},
        {"volume_name": []},
        {"environment": None},
        {"call_id": {}},
        {"app_id": "ap-x;stop"},
    ],
)
def test_malformed_receipts_rejected_before_provider_call(receipt, change):
    path, data = receipt
    data.update(change)
    modal_runs._write(path, data)
    with pytest.raises(ValueError):
        modal_runs._read(path.parent.parent)


@pytest.mark.parametrize("content", ["not-json", "[]"])
def test_invalid_receipt_document(tmp_path, content):
    path = tmp_path / ".runplz/run.json"
    path.parent.mkdir()
    path.write_text(content)
    with pytest.raises((ValueError, RuntimeError)):
        modal_runs._read(tmp_path)


def test_missing_receipt(tmp_path):
    with pytest.raises(RuntimeError, match="Cannot read"):
        modal_runs._read(tmp_path)


@pytest.mark.parametrize("code", [0, 13, -9])
def test_durable_result_survives_call_expiration(receipt, sdk, code):
    _, data = receipt
    sdk[1].read_file.side_effect = None
    sdk[1].read_file.return_value = [
        json.dumps({"run_id": data["run_id"], "exit_code": code}).encode()
    ]
    result = modal_runs._probe(data)
    assert result == {"state": "succeeded" if code == 0 else "failed", "exit_code": code}
    sdk[0].FunctionCall.from_id.assert_not_called()


@pytest.mark.parametrize(
    "result, state", [(0, "succeeded"), (11, "failed"), (TimeoutError(), "pending")]
)
def test_call_probe_does_not_wait(receipt, sdk, result, state):
    _, data = receipt
    call = sdk[2]
    call.get.side_effect = result if isinstance(result, Exception) else None
    call.get.return_value = result
    assert modal_runs._probe(data)["state"] == state
    call.get.assert_called_once_with(timeout=0)


@pytest.mark.parametrize(
    "error, state", [("FunctionTimeoutError", "failed"), ("OutputExpiredError", "expired")]
)
def test_modal_timeouts_are_not_pending(receipt, sdk, error, state):
    sdk[2].get.side_effect = getattr(sdk[0].exception, error)()
    assert modal_runs._probe(receipt[1])["state"] == state


def test_transport_error_is_not_a_job_failure(receipt, sdk):
    sdk[2].get.side_effect = RuntimeError("network unavailable")
    with pytest.raises(RuntimeError, match="network unavailable"):
        modal_runs._probe(receipt[1])


def test_unconfirmed_launch_can_recover_from_durable_result(receipt, sdk):
    _, data = receipt
    data.pop("call_id")
    assert modal_runs._probe(data)["state"] == "unconfirmed"
    sdk[1].read_file.side_effect = None
    sdk[1].read_file.return_value = [
        json.dumps({"run_id": data["run_id"], "exit_code": 0}).encode()
    ]
    assert modal_runs._probe(data)["state"] == "succeeded"
    data.pop("volume_id")
    assert modal_runs._probe(data)["state"] == "unconfirmed"


@pytest.mark.parametrize("result", [[], {"run_id": "wrong", "exit_code": 0}, {"exit_code": 0}])
def test_bad_completion_record(receipt, sdk, result):
    sdk[1].read_file.side_effect = None
    sdk[1].read_file.return_value = [json.dumps(result).encode()]
    with pytest.raises(ValueError, match="different run"):
        modal_runs._probe(receipt[1])


def test_oversized_completion_record(receipt, sdk):
    sdk[1].read_file.side_effect = None
    sdk[1].read_file.return_value = [b"x" * 16385]
    with pytest.raises(ValueError, match="Oversized"):
        modal_runs._probe(receipt[1])


@pytest.mark.parametrize("code", [None, True, "0", [], 999])
def test_invalid_exit_codes(code):
    with pytest.raises(ValueError, match="exit code"):
        modal_runs._outcome(code)


def test_environment_and_volume_identity_are_pinned(receipt, sdk):
    _, data = receipt
    data["environment"] = ""
    modal_runs._volume(data)
    assert os.environ["MODAL_ENVIRONMENT"] == ""
    sdk[0].Volume.from_name.assert_called_once_with("outputs", environment_name="")
    sdk[1].object_id = "vo-recreated"
    with pytest.raises(RuntimeError, match="identity changed"):
        modal_runs._probe(data)
    sdk[1].read_file.assert_not_called()


def _entry(data, suffix, kind=1):
    return SimpleNamespace(path=data["remote_path"].lstrip("/") + suffix, type=kind)


def test_download_scopes_outputs_preserves_metadata_and_is_repeatable(receipt, sdk, staging):
    path, data = receipt
    sdk[1].iterdir.return_value = [
        _entry(data, "", 2),
        _entry(data, "/checkpoints", 2),
        _entry(data, "/checkpoints/best.bin"),
        _entry(data, "/.runplz/run.json"),
        _entry(data, "/.RUNPLZ/run.json"),
    ]
    sdk[1].read_file_into_fileobj.side_effect = lambda remote, f: f.write(b"weights")
    for _ in range(2):
        assert modal_runs._download(data, path.parent.parent, staging_dir=staging) == {"files": 1}
    assert (path.parent.parent / "checkpoints/best.bin").read_bytes() == b"weights"
    assert json.loads(path.read_text()) == data
    sdk[1].iterdir.assert_called_with(data["remote_path"].lstrip("/"), recursive=True)


@pytest.mark.parametrize("suffix, kind", [("/../escape", 1), ("/link", 3)])
def test_unsafe_remote_files_rejected(receipt, sdk, staging, suffix, kind):
    sdk[1].iterdir.return_value = [_entry(receipt[1], suffix, kind)]
    with pytest.raises(ValueError):
        modal_runs._download(receipt[1], receipt[0].parent.parent, staging_dir=staging)


def test_volume_cannot_return_another_runs_files(receipt, sdk, staging):
    sdk[1].iterdir.return_value = [SimpleNamespace(path="runplz/other/file", type=1)]
    with pytest.raises(ValueError, match="escaped"):
        modal_runs._download(receipt[1], receipt[0].parent.parent, staging_dir=staging)


def test_symlink_destination_cannot_escape_collection(receipt, sdk, staging, tmp_path):
    sdk[1].iterdir.return_value = [_entry(receipt[1], "/link/file")]
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        modal_runs._download(receipt[1], tmp_path, staging_dir=staging)
    assert not list(outside.iterdir())


def test_failed_file_download_preserves_previous_complete_file(receipt, sdk, staging, tmp_path):
    sdk[1].iterdir.return_value = [_entry(receipt[1], "/weights")]
    sdk[1].read_file_into_fileobj.side_effect = RuntimeError("disconnected")
    (tmp_path / "weights").write_bytes(b"previous")
    with pytest.raises(RuntimeError, match="disconnected"):
        modal_runs._download(receipt[1], tmp_path, staging_dir=staging)
    assert (tmp_path / "weights").read_bytes() == b"previous"
    assert sorted(p.name for p in tmp_path.iterdir()) == [".runplz", "weights"]


@pytest.mark.parametrize(
    "failure", [OSError("missing python"), subprocess.TimeoutExpired("python", 20)]
)
def test_probe_bounds_network_waits(tmp_path, monkeypatch, failure):
    run = mock.Mock(side_effect=failure)
    monkeypatch.setattr(modal_runs.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="without relaunching"):
        modal_runs._bounded_worker("probe", tmp_path, timeout=20)
    assert run.call_args.kwargs["timeout"] == 20
    assert run.call_args.args[0][:4] == [
        sys.executable,
        "-m",
        "runplz.backends.modal_runs",
        "probe",
    ]


@pytest.mark.parametrize(
    "operation, stdout",
    [
        ("probe", "bad"),
        ("probe", "[]"),
        ("probe", '{"state": []}'),
        ("probe", '{"state": "invented"}'),
        ("download", '{"files": true}'),
    ],
)
def test_worker_protocol_validation(tmp_path, monkeypatch, operation, stdout):
    monkeypatch.setattr(
        modal_runs.subprocess,
        "run",
        mock.Mock(return_value=SimpleNamespace(returncode=0, stdout=stdout)),
    )
    with pytest.raises(RuntimeError, match="Invalid"):
        modal_runs._bounded_worker(operation, tmp_path, timeout=10)


def test_worker_failure_is_actionable(tmp_path, monkeypatch):
    monkeypatch.setattr(
        modal_runs.subprocess,
        "run",
        mock.Mock(return_value=SimpleNamespace(returncode=2, stderr="auth failed")),
    )
    with pytest.raises(RuntimeError, match="auth failed.*Receipt and remote outputs retained"):
        modal_runs._bounded_worker("probe", tmp_path, timeout=10)


@pytest.mark.parametrize(
    "state, rc, downloads",
    [
        ("pending", 3, False),
        ("unconfirmed", 2, False),
        ("succeeded", 0, True),
        ("failed", 1, True),
        ("expired", 2, True),
    ],
)
def test_collection_never_relaunches_or_waits_for_pending_job(
    receipt, monkeypatch, state, rc, downloads
):
    path, _ = receipt
    worker = mock.Mock(side_effect=[{"state": state}, {"files": 4}])
    monkeypatch.setattr(modal_runs, "_bounded_worker", worker)
    assert modal_runs.collect(path.parent.parent, timeout=1800) == rc
    assert worker.call_args_list[0].args[0] == "probe"
    assert worker.call_args_list[0].kwargs["timeout"] == 20
    assert worker.call_count == (2 if downloads else 1)
    if downloads:
        assert worker.call_args_list[1].args[0] == "download"
        assert worker.call_args_list[1].kwargs["timeout"] == 1800
        collection = json.loads(path.with_name("modal-collection.json").read_text())
        assert collection["outcome"]["state"] == state
        assert collection["run_id"] == receipt[1]["run_id"]
    else:
        assert not path.with_name("modal-collection.json").exists()
    assert modal_runs._read(path.parent.parent) == receipt[1]


def test_download_failure_does_not_record_success(receipt, monkeypatch):
    path, _ = receipt
    monkeypatch.setattr(
        modal_runs,
        "_bounded_worker",
        mock.Mock(side_effect=[{"state": "succeeded"}, RuntimeError("download interrupted")]),
    )
    with pytest.raises(RuntimeError, match="interrupted"):
        modal_runs.collect(path.parent.parent)
    assert not path.with_name("modal-collection.json").exists()


@pytest.mark.parametrize("timeout", [0, -1, True, 0.5])
def test_collect_validates_timeout(tmp_path, timeout):
    with pytest.raises(ValueError, match="positive integer"):
        modal_runs.collect(tmp_path, timeout=timeout)


@pytest.mark.parametrize(
    "result, rc",
    [
        ({"state": "succeeded", "exit_code": 0}, 0),
        ({"state": "pending"}, 0),
        ({"state": "expired", "detail": "expired"}, 2),
        ({"state": "unconfirmed"}, 2),
        (RuntimeError("offline"), 2),
    ],
)
def test_status_reports_ids_logs_and_truthful_outcome(receipt, monkeypatch, capsys, result, rc):
    path, _ = receipt
    worker = mock.Mock(
        side_effect=result if isinstance(result, Exception) else None, return_value=result
    )
    monkeypatch.setattr(modal_runs, "_bounded_worker", worker)
    assert modal_runs.status(path.parent.parent) == rc
    text = capsys.readouterr().out
    assert "logs: modal app logs ap-app" in text
    assert "stop: modal app stop ap-app" in text
    assert (
        "state: unknown" in text
        if isinstance(result, Exception)
        else f"state: {result['state']}" in text
    )


def test_status_routes_modal_receipts_without_ssh(receipt, monkeypatch):
    status = mock.Mock(return_value=0)
    monkeypatch.setattr(modal_runs, "status", status)
    out = receipt[0].parent.parent
    assert runs.status(outputs_dir=out, host_override=None, run_id_override=None) == 0
    status.assert_called_once_with(out)


def test_status_prepared_receipt(tmp_path, monkeypatch, capsys):
    modal_runs.prepare_run(tmp_path, "demo", "job", "outputs")
    monkeypatch.setattr(modal_runs, "_bounded_worker", lambda *a, **kw: {"state": "unconfirmed"})
    assert modal_runs.status(tmp_path) == 2
    assert "app: unconfirmed" in capsys.readouterr().out


def test_cli_collect(tmp_path, monkeypatch):
    collect = mock.Mock(return_value=3)
    monkeypatch.setattr(modal_runs, "collect", collect)
    assert cli.main(["collect", "--outputs-dir", str(tmp_path), "--timeout", "90"]) == 3
    collect.assert_called_once_with(tmp_path, timeout=90)
    collect.side_effect = ValueError("missing receipt")
    assert cli.main(["collect"]) == 2
    with pytest.raises(SystemExit):
        cli.main(["collect", "--timeout", "0"])


def test_worker_dispatch(receipt, sdk, staging):
    out = receipt[0].parent.parent
    assert modal_runs._worker("probe", out)["state"] == "pending"
    assert modal_runs._worker("download", out, staging) == {"files": 0}
    with pytest.raises(ValueError, match="parent-owned staging"):
        modal_runs._worker("download", out)
    with pytest.raises(ValueError, match="Unknown"):
        modal_runs._worker("oops", out)


def test_real_worker_process_rejects_invalid_receipt_without_network(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "runplz.backends.modal_runs", "probe", str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2
    assert "Cannot read Modal launch receipt" in result.stderr


def test_real_worker_process_reports_prepared_receipt_without_network(tmp_path):
    modal_runs.prepare_run(tmp_path, "demo", "job", "outputs")
    assert modal_runs._bounded_worker("probe", tmp_path, timeout=10) == {"state": "unconfirmed"}


def test_cli_status_routes_modal_receipt(receipt, monkeypatch):
    status = mock.Mock(return_value=0)
    monkeypatch.setattr(modal_runs, "status", status)
    out = receipt[0].parent.parent
    assert cli.main(["status", "--outputs-dir", str(out)]) == 0
    status.assert_called_once_with(out)


@pytest.mark.parametrize("command", ["tail", "kill"])
def test_modal_tail_and_kill_point_to_native_commands(receipt, capsys, command):
    assert cli.main([command, "--outputs-dir", str(receipt[0].parent.parent)]) == 2
    assert "modal app logs" in capsys.readouterr().err


def test_collection_cannot_erase_an_arriving_launch_acknowledgment(receipt, monkeypatch):
    path, data = receipt
    pending_receipt = {k: v for k, v in data.items() if k != "call_id"}
    modal_runs._write(path, pending_receipt)

    def worker(operation, outputs_dir, **kwargs):
        if operation == "probe":
            modal_runs._write(path, data)  # launcher receives the acknowledgment
            return {"state": "succeeded", "exit_code": 0}
        return {"files": 0}

    monkeypatch.setattr(modal_runs, "_bounded_worker", worker)
    assert modal_runs.collect(path.parent.parent) == 0
    assert modal_runs._read(path.parent.parent)["call_id"] == "fc-call"


def test_atomic_receipt_failure_preserves_previous_copy(receipt, monkeypatch):
    path, data = receipt
    monkeypatch.setattr(modal_runs.os, "replace", mock.Mock(side_effect=OSError("disk error")))
    with pytest.raises(OSError, match="disk error"):
        modal_runs._write(path, {"new": "data"})
    assert json.loads(path.read_text()) == data
    assert list(path.parent.iterdir()) == [path]


def test_explicit_attached_override_keeps_blocking_backend_path(tmp_path, sdk, monkeypatch):
    app, fn = _app(tmp_path)
    commands = []

    def launch(cmd, **kwargs):
        commands.append(cmd)
        assert cmd[:2] == ["modal", "run"]
        assert "--detach" not in cmd
        assert "_DETACHED_RECEIPT = None" in Path(cmd[-1].split("::")[0]).read_text()

    monkeypatch.setattr(backend.subprocess, "run", launch)
    extract = mock.Mock()
    monkeypatch.setattr(backend, "_extract_tar", extract)
    assert backend.run(app, fn, [], {}, detach=False) is None
    assert len(commands) == 1
    extract.assert_called_once()
    assert not (tmp_path / "out/.runplz/run.json").exists()


def test_provider_termination_allows_status_and_committed_artifact_salvage(
    receipt, sdk, staging, monkeypatch, capsys
):
    # Exercise the installed SDK's actual GenericResult decoding, not a made-up
    # exception. The only provider boundary is an offline gRPC response stub.
    response = api_pb2.FunctionGetOutputsResponse(
        outputs=[
            api_pb2.FunctionGetOutputsItem(
                result=api_pb2.GenericResult(
                    status=api_pb2.GenericResult.GENERIC_STATUS_TERMINATED,
                    exception="Function was terminated",
                ),
                data_format=api_pb2.DATA_FORMAT_PICKLE,
            )
        ],
    )
    rpc = mock.AsyncMock(return_value=response)
    client = SimpleNamespace(stub=SimpleNamespace(FunctionGetOutputs=rpc))
    call = modal_sdk.FunctionCall.from_id("fc-call", client=client)
    sdk[0].FunctionCall.from_id.return_value = call
    path, data = receipt
    sdk[1].iterdir.return_value = [_entry(data, "/checkpoint")]
    sdk[1].read_file_into_fileobj.side_effect = lambda remote, f: f.write(b"committed")
    monkeypatch.setattr(
        modal_runs,
        "_bounded_worker",
        lambda operation, outputs_dir, **kwargs: modal_runs._worker(
            operation, outputs_dir, staging
        ),
    )
    assert modal_runs.status(path.parent.parent) == 0
    assert "state: failed" in capsys.readouterr().out
    assert modal_runs.collect(path.parent.parent) == 1
    assert (path.parent.parent / "checkpoint").read_bytes() == b"committed"
    outcome = json.loads(path.with_name("modal-collection.json").read_text())["outcome"]
    assert outcome["state"] == "failed"
    assert "terminated" in outcome["detail"]
    assert "exit_code" not in outcome  # the provider did not supply one
    assert rpc.call_count == 2


_DOWNLOAD_WORKER_WITH_FAKE_VOLUME = """
import os
import runpy
import sys
import time
from types import SimpleNamespace

sys.modules["modal.volume"] = SimpleNamespace(
    FileEntryType=SimpleNamespace(FILE=1, DIRECTORY=2)
)

class Volume:
    object_id = "vo-original"

    def hydrate(self):
        return self

    def iterdir(self, root, **kwargs):
        yield SimpleNamespace(path=root + "/first", type=1)
        yield SimpleNamespace(path=root + "/weights", type=1)

    def read_file_into_fileobj(self, remote, f):
        if remote.endswith("/first"):
            f.write(b"first file completed")
            return
        mode = os.environ["RUNPLZ_TEST_DOWNLOAD_MODE"]
        if mode != "success":
            f.write(b"partial download")
            f.flush()
            print("partial-written", flush=True)
            if mode == "timeout":
                time.sleep(30)
            elif mode == "crash":
                os._exit(7)
        f.write(b"complete output")

sys.modules["modal"] = SimpleNamespace(
    Volume=SimpleNamespace(from_name=lambda name, **kwargs: Volume())
)
runpy.run_module("runplz.backends.modal_runs", run_name="__main__", alter_sys=True)
"""


@pytest.mark.parametrize("failure", ["timeout", "crash"])
def test_parent_cleans_partial_downloads_after_worker_death_and_retry(
    receipt, monkeypatch, failure
):
    path, data = receipt
    out = path.parent.parent
    weights = out / "weights"
    weights.write_bytes(b"previous complete output")
    first = out / "first"
    first.write_bytes(b"old first file")
    unrelated = path.parent / "download-unrelated"
    unrelated.mkdir()
    (unrelated / "user-file").write_bytes(b"keep me")
    original_paths = set(out.rglob("*"))
    real_run = subprocess.run
    observed_failures = []

    def run(cmd, **kwargs):
        # Replace only the provider with a fake. Keep the real bounded
        # subprocess, worker, file streaming, atomic writer, and parent cleanup.
        assert cmd[1:4] == ["-m", "runplz.backends.modal_runs", "download"]
        command = [cmd[0], "-c", _DOWNLOAD_WORKER_WITH_FAKE_VOLUME, *cmd[3:]]
        try:
            result = real_run(command, **kwargs)
        except subprocess.TimeoutExpired as exc:
            assert b"partial-written" in exc.stdout  # it really died mid-file
            observed_failures.append(failure)
            raise
        if result.returncode:
            assert "partial-written" in result.stdout
            observed_failures.append(failure)
        return result

    monkeypatch.setattr(modal_runs.subprocess, "run", run)
    monkeypatch.setenv("RUNPLZ_TEST_DOWNLOAD_MODE", failure)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="retry without relaunching"):
            modal_runs._bounded_worker("download", out, timeout=2)
        assert weights.read_bytes() == b"previous complete output"
        assert first.read_bytes() == b"first file completed"
        assert set(out.rglob("*")) == original_paths
        assert json.loads(path.read_text()) == data
    assert observed_failures == [failure, failure]
    monkeypatch.setenv("RUNPLZ_TEST_DOWNLOAD_MODE", "success")
    assert modal_runs._bounded_worker("download", out, timeout=10) == {"files": 2}
    assert weights.read_bytes() == b"complete output"
    assert set(out.rglob("*")) == original_paths
    assert (unrelated / "user-file").read_bytes() == b"keep me"


@pytest.mark.parametrize("boundary", ["volume_lookup", "volume_read", "call_lookup"])
def test_remote_errors_outside_result_lookup_remain_observation_errors(receipt, sdk, boundary):
    operation = {
        "volume_lookup": sdk[1].hydrate,
        "volume_read": sdk[1].read_file,
        "call_lookup": sdk[0].FunctionCall.from_id,
    }[boundary]
    operation.side_effect = modal_sdk.exception.RemoteError("observation failed")
    with pytest.raises(modal_sdk.exception.RemoteError, match="observation failed"):
        modal_runs._probe(receipt[1])
    sdk[2].get.assert_not_called()


@pytest.mark.parametrize("error", ["AuthError", "ConnectionError", "InternalFailure"])
def test_nonterminal_sdk_errors_remain_observation_errors(receipt, sdk, error):
    exception = getattr(modal_sdk.exception, error)
    sdk[2].get.side_effect = exception("observation failed")
    with pytest.raises(exception, match="observation failed"):
        modal_runs._probe(receipt[1])


def test_worker_refuses_download_without_parent_owned_staging(receipt):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "runplz.backends.modal_runs",
            "download",
            str(receipt[0].parent.parent),
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2
    assert "requires parent-owned staging" in result.stderr
