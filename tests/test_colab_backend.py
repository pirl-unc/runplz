"""Tests for the Google Colab backend (`runplz.backends.colab` + `ColabConfig`)."""

import io
import json
import subprocess
import tarfile
from pathlib import Path
from unittest import mock

import pytest

from runplz import App, ColabConfig, Image
from runplz.backends import colab, registry


def _make_app_and_fn(
    tmp_path: Path,
    *,
    colab_config: ColabConfig | None = None,
    image: Image | None = None,
    gpu: str | None = "T4",
    min_gpu_memory: float | None = None,
    min_memory: float | None = None,
    min_disk: float | None = None,
    min_gpus: int | None = None,
    env: dict | None = None,
    preconditions: dict | None = None,
):
    job_dir = tmp_path / "jobs"
    job_dir.mkdir(exist_ok=True)
    script = job_dir / "train.py"
    script.write_text("# dummy job\n")
    (tmp_path / ".env").write_text("SECRET=1\n")
    (tmp_path / "pkg.py").write_text("VALUE = 42\n")

    app = App("demo-app", colab_config=colab_config)
    app.repo_root = tmp_path
    img = image or (
        Image.from_registry("pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime")
        .apt_install("git")
        .pip_install("numpy", index_url="https://pypi.org/simple")
        .pip_install_local_dir(".", editable=True)
        .run_commands("echo ready")
    )

    @app.function(
        image=img,
        gpu=gpu,
        min_gpu_memory=min_gpu_memory,
        min_memory=min_memory,
        min_disk=min_disk,
        min_gpus=min_gpus,
        env=env,
        preconditions=preconditions,
    )
    def train(steps: int = 10):  # pragma: no cover
        return steps

    fn = app.functions["train"]
    fn.module_file = str(script)
    return app, fn


def _write_tar(path: Path, files: dict[str, bytes]) -> None:
    with tarfile.open(path, "w:gz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


# ---------------------------------------------------------------------------
# ColabConfig validation


def test_colab_config_defaults_and_valid_values():
    cfg = ColabConfig()
    assert cfg.auth == "oauth2"
    assert cfg.tpu is None
    assert cfg.high_mem is False
    assert cfg.on_finish == "stop"
    assert cfg.dry_run is False

    custom = ColabConfig(
        auth="adc",
        tpu="v6e1",
        high_mem=True,
        on_finish="leave",
        max_runtime_seconds=600,
        session_config_path="/tmp/colab-sessions.json",
        client_oauth_config="/tmp/oauth.json",
        dry_run=True,
    )
    assert custom.auth == "adc"
    assert custom.tpu == "v6e1"


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"auth": "magic"}, "ColabConfig.auth"),
        ({"tpu": "v4"}, "ColabConfig.tpu"),
        ({"tpu": 123}, "ColabConfig.tpu"),
        ({"high_mem": "yes"}, "ColabConfig.high_mem"),
        ({"on_finish": "delete"}, "ColabConfig.on_finish"),
        ({"max_runtime_seconds": 0}, "ColabConfig.max_runtime_seconds"),
        ({"max_runtime_seconds": "10"}, "ColabConfig.max_runtime_seconds"),
        ({"session_config_path": "  "}, "ColabConfig.session_config_path"),
        ({"client_oauth_config": ""}, "ColabConfig.client_oauth_config"),
        ({"exclude": ["data"]}, "ColabConfig.exclude"),
        ({"exclude": ("",)}, "ColabConfig.exclude"),
        ({"dry_run": 1}, "ColabConfig.dry_run"),
    ],
)
def test_colab_config_rejects_invalid_fields(kwargs, match):
    with pytest.raises(ValueError, match=match):
        ColabConfig(**kwargs)


# ---------------------------------------------------------------------------
# GPU / TPU mapping & guardrails


@pytest.mark.parametrize(
    ("gpu", "min_gpu_memory", "expected"),
    [
        (None, None, None),
        ("T4", None, "T4"),
        ("l4", None, "L4"),
        ("G4", None, "G4"),
        ("A100-40GB", None, "A100"),
        ("A100-80GB", None, "A100"),
        ("H100!", None, "H100"),
        ("H100-80GB", None, "H100"),
        (None, 12, "T4"),
        (None, 24, "L4"),
        (None, 40, "A100"),
        (None, 80, "H100"),
    ],
)
def test_gpu_flag_for_maps_colab_accelerators(gpu, min_gpu_memory, expected):
    assert colab.gpu_flag_for(gpu, min_gpu_memory) == expected


def test_gpu_flag_for_rejects_multi_gpu_unknown_and_excess_vram():
    with pytest.raises(ValueError, match="at most 1 GPU"):
        colab.gpu_flag_for("H100", None, num_gpus=2)
    with pytest.raises(ValueError, match="not supported on Google Colab"):
        colab.gpu_flag_for("H200", None)
    with pytest.raises(ValueError, match="exceeds the largest Google Colab GPU"):
        colab.gpu_flag_for(None, 141)


def test_colab_rejects_dockerfile_min_disk_and_gpu_plus_tpu(tmp_path):
    df = tmp_path / "Dockerfile"
    df.write_text("FROM ubuntu:22.04\n")
    app, fn = _make_app_and_fn(tmp_path, image=Image.from_dockerfile("Dockerfile"))
    with pytest.raises(ValueError, match="from_dockerfile"):
        colab.run(app, fn, [], {})

    app, fn = _make_app_and_fn(tmp_path, min_disk=100)
    with pytest.raises(ValueError, match="min_disk"):
        colab.run(app, fn, [], {})

    app, fn = _make_app_and_fn(tmp_path, gpu="T4", colab_config=ColabConfig(tpu="v5e1"))
    with pytest.raises(ValueError, match="cannot request both"):
        colab.run(app, fn, [], {})


def test_colab_guard_blocks_real_colab_cli():
    with pytest.raises(RuntimeError, match="tried to run `colab`"):
        colab.subprocess.run(["colab", "sessions"], capture_output=True, text=True)


def test_colab_missing_binary_raises_clear_error(tmp_path):
    app, fn = _make_app_and_fn(tmp_path)
    with mock.patch("runplz.backends.colab.shutil.which", return_value=None):
        with pytest.raises(RuntimeError, match="google-colab-cli"):
            colab.run(app, fn, [], {})


# ---------------------------------------------------------------------------
# End-to-end mocked dispatch, output collection, salvage & teardown


def test_colab_dry_run_prints_commands_without_spawning(tmp_path, capsys):
    cfg = ColabConfig(
        auth="adc",
        tpu="v6e1",
        high_mem=True,  # ignored for TPU
        session_config_path="/tmp/s.json",
        client_oauth_config="/tmp/c.json",
        dry_run=True,
    )
    app, fn = _make_app_and_fn(tmp_path, gpu=None, colab_config=cfg)
    with mock.patch("runplz.backends.colab.subprocess.run") as run_mock:
        colab.run(app, fn, [10], {"x": 1}, outputs_dir="out")
    run_mock.assert_not_called()
    out = capsys.readouterr().out
    assert "--auth=adc" in out
    assert "--config /tmp/s.json" in out
    assert "--client-oauth-config /tmp/c.json" in out
    assert "--tpu V6E1" in out
    assert "--high-mem" not in out
    assert " stop -s runplz-demo-app-train-" in out


def test_colab_run_happy_path_with_high_mem_and_excludes(tmp_path):
    cfg = ColabConfig(auth="oauth2", on_finish="stop", max_runtime_seconds=900)
    app, fn = _make_app_and_fn(
        tmp_path,
        gpu="H100!",
        min_memory=32,
        colab_config=cfg,
        env={"CUSTOM_FLAG": "1"},
        preconditions={"gpu_count": 1},
    )
    calls: list[list[str]] = []
    uploaded_members: list[str] = []
    runner_text = ""

    def fake_run(cmd, check=False, **kwargs):
        nonlocal runner_text
        calls.append(list(cmd))
        sub = next(tok for tok in cmd[1:] if not tok.startswith("-"))
        if sub == "upload":
            tar_src = Path(cmd[-2])
            with tarfile.open(tar_src, "r:gz") as tar:
                uploaded_members.extend(tar.getnames())
        elif sub == "exec":
            runner_path = Path(cmd[cmd.index("-f") + 1])
            runner_text = runner_path.read_text()
        elif sub == "download":
            remote_path, local_dest = cmd[-2], Path(cmd[-1])
            if remote_path.endswith("out.tar.gz"):
                _write_tar(local_dest, {"metrics.json": b'{"ok": true}'})
            elif remote_path.endswith("result.json"):
                local_dest.write_text(json.dumps({"exit_code": 0, "error": ""}))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    def fake_exec(cmd, *, cmd_timeout, hang_grace_s=5.0):
        nonlocal runner_text
        calls.append(list(cmd))
        runner_path = Path(cmd[cmd.index("-f") + 1])
        runner_text = runner_path.read_text()
        return 0, True

    with (
        mock.patch("runplz.backends.colab.shutil.which", return_value="/usr/local/bin/colab"),
        mock.patch("runplz.backends.colab.subprocess.run", side_effect=fake_run),
        mock.patch("runplz.backends.colab._exec_colab_once", side_effect=fake_exec),
    ):
        app.bind("colab", outputs_dir="out")
        fn.remote(25)

    assert (tmp_path / "out" / "metrics.json").read_text() == '{"ok": true}'
    assert ".env" not in uploaded_members
    assert "pkg.py" in uploaded_members
    assert "jobs/train.py" in uploaded_members
    assert any("--gpu" in c and "H100" in c and "--high-mem" in c for c in calls)
    assert any("exec" in c and "--timeout" in c and "0" in c for c in calls)
    assert any("stop" in c for c in calls)
    assert '"-m", "runplz._bootstrap"' in runner_text
    assert "CUSTOM_FLAG" in runner_text


def test_colab_run_l4_skips_high_mem_and_leave_skips_stop(tmp_path, capsys):
    app, fn = _make_app_and_fn(
        tmp_path,
        gpu="L4",
        min_memory=40,
        colab_config=ColabConfig(high_mem=True, on_finish="leave"),
    )
    calls: list[list[str]] = []

    def fake_run(cmd, check=False, **kwargs):
        calls.append(list(cmd))
        if "download" in cmd and cmd[-2].endswith("result.json"):
            Path(cmd[-1]).write_text(json.dumps({"exit_code": 0, "error": ""}))
        elif "download" in cmd and cmd[-2].endswith("out.tar.gz"):
            _write_tar(Path(cmd[-1]), {"done.txt": b"ok"})
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    with (
        mock.patch("runplz.backends.colab.shutil.which", return_value="/usr/local/bin/colab"),
        mock.patch("runplz.backends.colab.subprocess.run", side_effect=fake_run),
        mock.patch("runplz.backends.colab._exec_colab_once", return_value=(0, True)),
    ):
        colab.run(app, fn, [], {})

    new_cmd = calls[0]
    assert "--gpu" in new_cmd and "L4" in new_cmd
    assert "--high-mem" not in new_cmd
    assert not any("stop" in c for c in calls)
    assert "leaving Colab session" in capsys.readouterr().out


def test_colab_run_salvages_outputs_and_stops_on_nonzero_exit(tmp_path):
    app, fn = _make_app_and_fn(tmp_path, gpu="T4")
    stopped = []

    def fake_run(cmd, check=False, **kwargs):
        if "download" in cmd and cmd[-2].endswith("out.tar.gz"):
            _write_tar(Path(cmd[-1]), {"partial.log": b"step 100"})
        elif "download" in cmd and cmd[-2].endswith("result.json"):
            Path(cmd[-1]).write_text(json.dumps({"exit_code": 2, "error": "OOM"}))
        elif "stop" in cmd:
            stopped.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    with (
        mock.patch("runplz.backends.colab.shutil.which", return_value="/usr/local/bin/colab"),
        mock.patch("runplz.backends.colab.subprocess.run", side_effect=fake_run),
        mock.patch("runplz.backends.colab._exec_colab_once", return_value=(0, True)),
    ):
        with pytest.raises(RuntimeError, match="OOM"):
            colab.run(app, fn, [], {})

    assert (tmp_path / "out" / "partial.log").read_text() == "step 100"
    assert len(stopped) == 1


def test_colab_run_missing_receipt_and_stop_error_during_unwind(tmp_path, capsys):
    app, fn = _make_app_and_fn(tmp_path, gpu=None)

    def fake_run(cmd, check=False, **kwargs):
        if "download" in cmd:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="404")
        if "stop" in cmd:
            raise OSError("network down")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    with (
        mock.patch("runplz.backends.colab.shutil.which", return_value="/usr/local/bin/colab"),
        mock.patch("runplz.backends.colab.subprocess.run", side_effect=fake_run),
        mock.patch("runplz.backends.colab._exec_colab_once", return_value=(0, True)),
    ):
        with pytest.raises(RuntimeError, match="without a result receipt"):
            colab.run(app, fn, [], {})

    assert "warning: failed to stop Colab session" in capsys.readouterr().err


def test_colab_stop_error_propagates_when_run_succeeded(tmp_path):
    app, fn = _make_app_and_fn(tmp_path, gpu=None)

    def fake_run(cmd, check=False, **kwargs):
        if "download" in cmd and cmd[-2].endswith("result.json"):
            Path(cmd[-1]).write_text(json.dumps({"exit_code": 0, "error": ""}))
        elif "download" in cmd and cmd[-2].endswith("out.tar.gz"):
            _write_tar(Path(cmd[-1]), {"a.txt": b"1"})
        elif "stop" in cmd:
            raise subprocess.CalledProcessError(1, cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    with (
        mock.patch("runplz.backends.colab.shutil.which", return_value="/usr/local/bin/colab"),
        mock.patch("runplz.backends.colab.subprocess.run", side_effect=fake_run),
        mock.patch("runplz.backends.colab._exec_colab_once", return_value=(0, True)),
    ):
        with pytest.raises(subprocess.CalledProcessError):
            colab.run(app, fn, [], {})


def test_colab_pack_repo_honors_gitignore_and_always_includes_script(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / ".gitignore").write_text("ignored.bin\n")
    (tmp_path / "ignored.bin").write_bytes(b"secret")
    (tmp_path / "kept.py").write_text("x = 1\n")
    exp_dir = tmp_path / "experiments"
    exp_dir.mkdir()
    (exp_dir / "huge.dat").write_bytes(b"0" * 100)
    nested = tmp_path / "data" / "shared-2026"
    nested.mkdir(parents=True)
    (nested / "big.csv").write_text("a,b\n")
    script = exp_dir / "job.py"
    script.write_text("# job\n")

    tar_path = tmp_path / "repo.tar.gz"
    colab._pack_repo(
        tmp_path,
        tar_path,
        extra_excludes=("experiments", "data/shared-*"),
        always_include=Path("experiments/job.py"),
    )
    with tarfile.open(tar_path, "r:gz") as tar:
        names = set(tar.getnames())
    assert "kept.py" in names
    assert "experiments/job.py" in names
    assert "experiments/huge.dat" not in names
    assert "data/shared-2026/big.csv" not in names
    assert "ignored.bin" not in names


def test_colab_run_retries_transient_cli_failure():
    attempts = []

    def fake_run(cmd, check=False, timeout=None):
        attempts.append(cmd)
        rc = 1 if len(attempts) < 2 else 0
        return subprocess.CompletedProcess(cmd, rc, stdout="", stderr="")

    with (
        mock.patch("runplz.backends.colab.subprocess.run", side_effect=fake_run),
        mock.patch("runplz.backends.colab.time.sleep") as sleep_mock,
    ):
        res = colab._run_colab(["colab", "upload"], dry_run=False, retries=3)
    assert res.returncode == 0
    assert len(attempts) == 2
    sleep_mock.assert_called_once()

    with (
        mock.patch(
            "runplz.backends.colab.subprocess.run",
            return_value=subprocess.CompletedProcess(["colab"], 1),
        ),
        mock.patch("runplz.backends.colab.time.sleep"),
    ):
        with pytest.raises(subprocess.CalledProcessError):
            colab._run_colab(["colab", "upload"], dry_run=False, retries=2)


def test_colab_extract_tar_rejects_traversal_and_symlinks(tmp_path):
    dest = tmp_path / "dest"
    dest.mkdir()

    bad_path = tmp_path / "traversal.tar.gz"
    _write_tar(bad_path, {"../escape.txt": b"nope"})
    with pytest.raises(RuntimeError, match="unsafe tar member"):
        colab._extract_tar(bad_path, dest)

    sym_path = tmp_path / "symlink.tar.gz"
    with tarfile.open(sym_path, "w:gz") as tar:
        info = tarfile.TarInfo(name="link")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        tar.addfile(info)
    with pytest.raises(RuntimeError, match="tar link member"):
        colab._extract_tar(sym_path, dest)


# ---------------------------------------------------------------------------
# list_jobs & registry integration


def test_colab_list_jobs_parses_sessions_and_filters_prefix():
    stdout = "\n".join(
        [
            "[runplz-demo-app-train-a1b2c3] m-s-123 | Hardware: H100 "
            "| Shape: HIGH_RAM | Variant: GPU | Status: BUSY",
            "[runplz-simple] m-s-456 | Hardware: T4 | Shape: STANDARD | Variant: GPU",
            "[personal-nb] m-s-789 | Hardware: A100 | Shape: HIGH_RAM | Variant: GPU",
            "[?] m-s-orphan | Hardware: CPU | Shape: STANDARD | Variant: DEFAULT",
            "ignored header line",
        ]
    )
    fake = subprocess.CompletedProcess(["colab"], 0, stdout=stdout, stderr="")
    with (
        mock.patch("runplz.backends.colab.shutil.which", return_value="/usr/local/bin/colab"),
        mock.patch("runplz.backends.colab.subprocess.run", return_value=fake) as run_mock,
    ):
        jobs = registry.list_jobs("colab", auth="adc", config="/tmp/s.json")

    cmd = run_mock.call_args.args[0]
    assert cmd == ["colab", "--auth=adc", "--config", "/tmp/s.json", "sessions"]
    assert len(jobs) == 2
    assert jobs[0].backend == "colab"
    assert jobs[0].app == "demo-app"
    assert jobs[0].function == "train"
    assert jobs[0].status == "H100/HIGH_RAM (BUSY)"
    assert jobs[1].app == ""
    assert jobs[1].function == ""
    assert jobs[1].status == "T4/STANDARD"


def test_colab_list_jobs_errors_and_not_in_default_ps_fanout():
    assert "colab" in registry.names()
    assert "colab" in registry.listable_names()
    assert "colab" not in registry.ps_names()

    with mock.patch("runplz.backends.colab.shutil.which", return_value="/usr/local/bin/colab"):
        with pytest.raises(ValueError, match="colab auth"):
            colab.list_jobs(auth="bad")

    fake = subprocess.CompletedProcess(["colab"], 1, stdout="", stderr="401 unauthorized")
    with (
        mock.patch("runplz.backends.colab.shutil.which", return_value="/usr/local/bin/colab"),
        mock.patch("runplz.backends.colab.subprocess.run", return_value=fake),
    ):
        with pytest.raises(RuntimeError, match="colab sessions"):
            colab.list_jobs()


def test_colab_exec_reaps_wedged_disconnect_and_respects_started_sentinel():
    import sys

    # 1. Real child script that prints the google-colab-cli disconnect line and
    #    refuses to exit (simulating the leaked WebSocket thread). Must be reaped!
    wedged_cmd = [
        sys.executable,
        "-u",
        "-c",
        "import time; print('RuntimeError: Connection was lost.', flush=True); time.sleep(30)",
    ]
    rc, started = colab._exec_colab_once(wedged_cmd, cmd_timeout=10.0, hang_grace_s=0.15)
    assert rc == 1
    assert started is False

    # 2. Normal exit that emits the runner-started sentinel (avoid literal
    #    billed-CLI token in argv so conftest's offline guard allows the child).
    ok_cmd = [
        sys.executable,
        "-u",
        "-c",
        "print('[runplz] ' + 'co' + 'lab runner started', flush=True)",
    ]
    rc, started = colab._exec_colab_once(ok_cmd, cmd_timeout=10.0, hang_grace_s=0.15)
    assert rc == 0
    assert started is True

    # 3. Hard wall-clock timeout reaps a silent hung process.
    hung_cmd = [sys.executable, "-c", "import time; time.sleep(30)"]
    rc, started = colab._exec_colab_once(hung_cmd, cmd_timeout=0.2, hang_grace_s=5.0)
    assert rc == 124
    assert started is False

    # 4. _run_colab_exec retries if disconnect happened BEFORE runner started,
    #    and refuses to double-run if runner ALREADY started.
    seq = iter([(1, False), (0, True)])
    with (
        mock.patch("runplz.backends.colab._exec_colab_once", side_effect=lambda *a, **k: next(seq)),
        mock.patch("runplz.backends.colab.time.sleep") as sleep_mock,
    ):
        res = colab._run_colab_exec(["colab", "exec"], dry_run=False, cmd_timeout=60.0, retries=3)
    assert res.returncode == 0
    sleep_mock.assert_called_once()

    calls = []

    def fail_after_start(*a, **k):
        calls.append(1)
        return 1, True

    with (
        mock.patch("runplz.backends.colab._exec_colab_once", side_effect=fail_after_start),
        mock.patch("runplz.backends.colab.time.sleep"),
    ):
        with pytest.raises(subprocess.CalledProcessError):
            colab._run_colab_exec(["colab", "exec"], dry_run=False, cmd_timeout=60.0, retries=3)
    assert len(calls) == 1


def test_cli_entrypoint_supports_stringized_annotations():
    from runplz.cli import _parse_entrypoint_args

    ns: dict = {}
    exec(
        "from __future__ import annotations\ndef main(steps: int = 10, flag: bool = False): ...\n",
        ns,
    )
    parsed = _parse_entrypoint_args(ns["main"], ["--steps", "32", "--flag"], pytest.fail)
    assert parsed == {"steps": 32, "flag": True}


def test_colab_config_rejects_bad_fallback_to_standard_mem():
    with pytest.raises(ValueError, match="fallback_to_standard_mem"):
        ColabConfig(fallback_to_standard_mem="yes")


def test_colab_high_mem_fallback_to_standard_mem(tmp_path: Path):
    app, fn = _make_app_and_fn(
        tmp_path,
        colab_config=ColabConfig(high_mem=True, fallback_to_standard_mem=True),
        gpu="A100",
    )
    seen_new_cmds: list[list[str]] = []

    def fake_run(cmd, *, check=False, timeout=None, **kw):
        if "new" in cmd:
            seen_new_cmds.append(list(cmd))
            if "--high-mem" in cmd:
                return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="503")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if "download" in cmd:
            dest = Path(cmd[-1])
            if dest.name == "result.json":
                dest.write_text(json.dumps({"exit_code": 0, "error": None}))
            else:
                with tarfile.open(dest, "w:gz") as tar:
                    info = tarfile.TarInfo("ok.txt")
                    data = b"ok"
                    info.size = len(data)
                    tar.addfile(info, io.BytesIO(data))
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    with (
        mock.patch("runplz.backends.colab.shutil.which", return_value="/usr/bin/colab"),
        mock.patch("runplz.backends.colab.subprocess.run", side_effect=fake_run),
        mock.patch("runplz.backends.colab._exec_colab_once", return_value=(0, True)),
        mock.patch("runplz.backends.colab.time.sleep"),
    ):
        colab.run(app, fn, [], {}, outputs_dir="out")

    assert any("--high-mem" in c for c in seen_new_cmds)
    assert any("--high-mem" not in c for c in seen_new_cmds)
    assert (tmp_path / "out" / "ok.txt").read_text() == "ok"


def test_colab_high_mem_no_fallback_raises(tmp_path: Path):
    app, fn = _make_app_and_fn(
        tmp_path,
        colab_config=ColabConfig(high_mem=True, fallback_to_standard_mem=False),
        gpu="A100",
    )
    seen_new_cmds: list[list[str]] = []

    def fake_run(cmd, *, check=False, timeout=None, **kw):
        seen_new_cmds.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="503")

    with (
        mock.patch("runplz.backends.colab.shutil.which", return_value="/usr/bin/colab"),
        mock.patch("runplz.backends.colab.subprocess.run", side_effect=fake_run),
        mock.patch("runplz.backends.colab.time.sleep"),
    ):
        with pytest.raises(subprocess.CalledProcessError):
            colab.run(app, fn, [], {}, outputs_dir="out")

    assert all("--high-mem" in c for c in seen_new_cmds)


def test_colab_upload_survives_tls_warmup_retries(tmp_path: Path):
    """Fresh Colab reverse proxies can reject initial TLS handshakes for 15-25s."""
    app, fn = _make_app_and_fn(tmp_path, gpu="A100")
    upload_attempts = 0
    slept: list[float] = []

    def fake_run(cmd, *, check=False, timeout=None, **kw):
        nonlocal upload_attempts
        if "upload" in cmd:
            upload_attempts += 1
            if upload_attempts < 5:
                return subprocess.CompletedProcess(
                    cmd, 1, stdout="", stderr="SSL: WRONG_VERSION_NUMBER"
                )
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if "download" in cmd:
            dest = Path(cmd[-1])
            if dest.name == "result.json":
                dest.write_text(json.dumps({"exit_code": 0, "error": None}))
            else:
                with tarfile.open(dest, "w:gz") as tar:
                    info = tarfile.TarInfo("ok.txt")
                    data = b"ok"
                    info.size = len(data)
                    tar.addfile(info, io.BytesIO(data))
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    with (
        mock.patch("runplz.backends.colab.shutil.which", return_value="/usr/bin/colab"),
        mock.patch("runplz.backends.colab.subprocess.run", side_effect=fake_run),
        mock.patch("runplz.backends.colab._exec_colab_once", return_value=(0, True)),
        mock.patch("runplz.backends.colab.time.sleep", side_effect=slept.append),
    ):
        colab.run(app, fn, [], {}, outputs_dir="out")

    assert upload_attempts == 5
    assert slept == [5.0, 10.0, 15.0, 20.0]
    assert (tmp_path / "out" / "ok.txt").read_text() == "ok"
