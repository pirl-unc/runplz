"""Brev backend coverage — exercises every branch of brev.py with
subprocess/ssh/rsync mocked out, plus a full happy-path `run()` for
each of the three modes (docker vm, native vm, container).
"""

import json
import types
from pathlib import Path
from unittest import mock

import pytest

from runplz import App, BrevConfig, Image
from runplz.backends import brev

# -- helpers ---------------------------------------------------------------


class FakeSH:
    """Record all subprocess.run() calls and reply per a pre-programmed
    sequence. Each call's return value can be (returncode, stdout,
    stderr) or a callable that inspects the cmd."""

    def __init__(self, default=(0, "", ""), scripted=None):
        self.default = default
        self.scripted = list(scripted or [])
        self.calls = []

    def __call__(self, cmd, *args, **kwargs):
        self.calls.append((list(cmd), dict(kwargs)))
        if self.scripted:
            entry = self.scripted.pop(0)
            if callable(entry):
                entry = entry(cmd, kwargs)
            rc, out, err = entry
        else:
            rc, out, err = self.default
        return mock.Mock(returncode=rc, stdout=out, stderr=err)


def _app(tmp_path, *, cfg=None):
    app = App("panalle", brev=cfg or BrevConfig())
    app._repo_root = tmp_path
    return app


def _function(app, image, *, module_file=None, **extra):
    @app.function(image=image, **extra)
    def train():  # pragma: no cover — called only under runner control
        return "ok"

    fn = app.functions["train"]
    if module_file is not None:
        fn.module_file = str(module_file)
    return fn


def _job_inside(tmp_path, name="jobs/train.py"):
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("# fake\n")
    return p


# -- _validate_config already covered in test_runplz.py; skipping here ----


# -- _brev_gpu_name covered in test_runplz.py; skipping ------------------


# -- _render_ops_script already smoke-tested; add edges ------------------


def test_render_ops_script_run_commands_and_index_url(tmp_path):
    img = (
        Image.from_registry("ubuntu:22.04")
        .pip_install("torch", index_url="https://download.pytorch.org/whl/cu121")
        .run_commands("echo hi")
    )
    s = brev._render_ops_script(img)
    assert "--index-url https://download.pytorch.org/whl/cu121" in s
    assert "echo hi" in s


def test_render_ops_script_non_editable_local_dir(tmp_path):
    img = Image.from_registry("ubuntu:22.04").pip_install_local_dir(".", editable=False)
    s = brev._render_ops_script(img)
    # The quoted path appears literally with double quotes so $HOME expands.
    assert 'pip install --quiet "$HOME/runplz-repo"' in s
    assert "-e" not in s.split("pip install")[-1]


# -- _instance_exists -----------------------------------------------------


def test_instance_exists_true_when_name_present():
    with mock.patch(
        "runplz.backends.brev.subprocess.run",
        return_value=mock.Mock(returncode=0, stdout=json.dumps([{"name": "foo"}])),
    ):
        assert brev._instance_exists("foo") is True


def test_instance_exists_false_when_missing():
    with mock.patch(
        "runplz.backends.brev.subprocess.run",
        return_value=mock.Mock(returncode=0, stdout="[]"),
    ):
        assert brev._instance_exists("bar") is False


def test_instance_exists_handles_non_list_json():
    with mock.patch(
        "runplz.backends.brev.subprocess.run",
        return_value=mock.Mock(returncode=0, stdout=json.dumps({"instances": [{"name": "x"}]})),
    ):
        assert brev._instance_exists("x") is True


def test_instance_exists_false_on_cli_failure():
    with mock.patch(
        "runplz.backends.brev.subprocess.run",
        return_value=mock.Mock(returncode=1, stdout=""),
    ):
        assert brev._instance_exists("x") is False


def test_instance_exists_false_on_invalid_json():
    with mock.patch(
        "runplz.backends.brev.subprocess.run",
        return_value=mock.Mock(returncode=0, stdout="not-json"),
    ):
        assert brev._instance_exists("x") is False


# -- _require_brev_cli ----------------------------------------------------


def test_require_brev_cli_raises_when_missing():
    with mock.patch(
        "runplz.backends.brev.subprocess.run",
        return_value=mock.Mock(returncode=1),
    ):
        with pytest.raises(RuntimeError, match="brev` CLI not found"):
            brev._require_brev_cli()


def test_require_brev_cli_silent_when_present():
    with mock.patch(
        "runplz.backends.brev.subprocess.run",
        return_value=mock.Mock(returncode=0),
    ):
        brev._require_brev_cli()  # no raise


# -- _skip_onboarding -----------------------------------------------------


def test_skip_onboarding_writes_file_on_first_call(tmp_path, monkeypatch, capsys):
    fake_home = tmp_path / "home"
    monkeypatch.setattr(brev, "_BREV_ONBOARDING", fake_home / ".brev" / "onboarding_step.json")
    brev._skip_onboarding()
    text = (fake_home / ".brev" / "onboarding_step.json").read_text()
    parsed = json.loads(text)
    assert parsed["hasRunBrevShell"] is True
    assert "+ wrote" in capsys.readouterr().out


def test_skip_onboarding_tolerates_os_error(monkeypatch):
    monkeypatch.setattr(brev, "_BREV_ONBOARDING", Path("/nonexistent/root/not/allowed.json"))
    # Should not raise even if mkdir/write fails.
    brev._skip_onboarding()


# -- _pick_instance_type + _create_instance additional edges -------------


def test_pick_instance_type_handles_non_list_json():
    with mock.patch(
        "runplz.backends.brev.subprocess.run",
        return_value=mock.Mock(returncode=0, stdout=json.dumps({"type": "x"})),
    ):
        # `isinstance(results, list)` is False → returns None.
        assert (
            brev._pick_instance_type(
                types.SimpleNamespace(
                    gpu="T4",
                    min_cpu=None,
                    min_memory=None,
                    min_gpu_memory=None,
                    min_disk=None,
                )
            )
            is None
        )


def test_pick_instance_type_returns_none_on_non_zero_rc():
    with mock.patch(
        "runplz.backends.brev.subprocess.run",
        return_value=mock.Mock(returncode=1, stdout=""),
    ):
        assert (
            brev._pick_instance_type(
                types.SimpleNamespace(
                    gpu="T4",
                    min_cpu=None,
                    min_memory=None,
                    min_gpu_memory=None,
                    min_disk=None,
                )
            )
            is None
        )


def test_pick_instance_type_parses_Type_capital_key():
    with mock.patch(
        "runplz.backends.brev.subprocess.run",
        return_value=mock.Mock(returncode=0, stdout=json.dumps([{"Type": "alt-key"}])),
    ):
        result = brev._pick_instance_type(
            types.SimpleNamespace(
                gpu="T4",
                min_cpu=None,
                min_memory=None,
                min_gpu_memory=None,
                min_disk=None,
            )
        )
        assert result == "alt-key"


def test_create_instance_with_resource_request_uses_picker(tmp_path):
    app = _app(tmp_path)
    fn = _function(
        app,
        Image.from_registry("ubuntu:22.04"),
        module_file=_job_inside(tmp_path),
        gpu="T4",
        min_cpu=4,
        min_memory=26,
        min_disk=100,
    )

    recorded = {}

    def fake_sh(cmd):
        recorded["create_cmd"] = cmd

    with mock.patch("runplz.backends.brev._pick_instance_type", return_value="picked-type"):
        with mock.patch("runplz.backends.brev._sh", fake_sh):
            brev._create_instance("x", "fallback-type", cfg=app.brev, image=fn.image, function=fn)

    cmd = recorded["create_cmd"]
    assert cmd[:2] == ["brev", "create"]
    assert "picked-type" in cmd
    assert "fallback-type" not in cmd
    # min_disk forwarded to provisioning
    i = cmd.index("--min-disk")
    assert cmd[i + 1] == "100"


def test_create_instance_container_mode_adds_image_flag(tmp_path):
    cfg = BrevConfig(mode="container")
    app = _app(tmp_path, cfg=cfg)
    image = Image.from_registry("pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime")
    fn = _function(app, image, module_file=_job_inside(tmp_path))

    recorded = {}
    with mock.patch("runplz.backends.brev._sh", lambda c: recorded.setdefault("c", c)):
        brev._create_instance(
            "x", "n1-standard-4:nvidia-tesla-t4:1", cfg=cfg, image=image, function=fn
        )

    cmd = recorded["c"]
    assert "--mode" in cmd
    assert "container" in cmd
    assert "--container-image" in cmd
    assert "pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime" in cmd


def test_create_instance_container_mode_requires_from_registry(tmp_path):
    cfg = BrevConfig(mode="container")
    app = _app(tmp_path, cfg=cfg)
    image = Image.from_dockerfile("Dockerfile")
    fn = _function(app, image, module_file=_job_inside(tmp_path))

    with mock.patch("runplz.backends.brev._sh", lambda c: None):
        with pytest.raises(RuntimeError, match="from_registry"):
            brev._create_instance("x", "any-type", cfg=cfg, image=image, function=fn)


# -- _ensure_docker -------------------------------------------------------


def test_ensure_docker_happy_path():
    # subprocess.run returns 0 → wait script exit 0 → no fallback.
    with mock.patch(
        "runplz.backends.brev.subprocess.run",
        return_value=mock.Mock(returncode=0),
    ):
        brev._ensure_docker("some-box")  # no raise


def test_ensure_docker_falls_back_to_get_docker_sh(capsys):
    # First call (wait script) fails → fallback calls _sh with curl | sh.
    fallback_called = {}

    def fake_sh(cmd):
        fallback_called["cmd"] = cmd

    with mock.patch(
        "runplz.backends.brev.subprocess.run",
        return_value=mock.Mock(returncode=1),
    ):
        with mock.patch("runplz.backends.brev._sh", fake_sh):
            brev._ensure_docker("some-box")

    assert "get.docker.com" in fallback_called["cmd"][-1]
    assert "falling back" in capsys.readouterr().out


# -- _remote_has_nvidia ---------------------------------------------------


def test_remote_has_nvidia_true():
    with mock.patch(
        "runplz.backends.brev.subprocess.run",
        return_value=mock.Mock(returncode=0, stdout="y\n"),
    ):
        assert brev._remote_has_nvidia("box") is True


def test_remote_has_nvidia_false():
    with mock.patch(
        "runplz.backends.brev.subprocess.run",
        return_value=mock.Mock(returncode=0, stdout="n\n"),
    ):
        assert brev._remote_has_nvidia("box") is False


# -- _rsync_up / _rsync_down / _ssh / _sh ---------------------------------


def test_rsync_up_excludes_and_no_delete(tmp_path):
    recorded = {}
    with mock.patch("runplz.backends.brev._sh", lambda c: recorded.setdefault("c", c)):
        brev._rsync_up(tmp_path, "my-box")
    cmd = recorded["c"]
    assert cmd[:2] == ["rsync", "-az"]
    assert "--delete" not in cmd  # regression: --delete was removed
    assert "--exclude=.git" in cmd
    assert "--exclude=out" in cmd


def test_rsync_down_runs_correct_cmd(tmp_path):
    recorded = {}
    with mock.patch("runplz.backends.brev._sh", lambda c: recorded.setdefault("c", c)):
        brev._rsync_down("my-box", tmp_path)
    cmd = recorded["c"]
    assert cmd[:2] == ["rsync", "-az"]
    assert cmd[2].endswith(":runplz-out/")  # remote side
    assert cmd[3] == f"{tmp_path}/"


def test_ssh_packages_cmd_in_bash_lc():
    recorded = {}
    with mock.patch("runplz.backends.brev._sh", lambda c: recorded.setdefault("c", c)):
        brev._ssh("instance", "echo hi")
    assert recorded["c"][0] == "ssh"
    # Last arg is `bash -lc '<escaped cmd>'`.
    assert recorded["c"][-1].startswith("bash -lc ")
    assert "echo hi" in recorded["c"][-1]


def test_sh_prints_and_runs(capsys):
    with mock.patch(
        "runplz.backends.brev.subprocess.run",
        return_value=mock.Mock(returncode=0),
    ) as patched:
        brev._sh(["echo", "hello world"])
    assert patched.called
    # shlex-quoted print with + prefix
    assert "+ echo" in capsys.readouterr().out


def test_ssh_capture_runs_with_capture():
    with mock.patch(
        "runplz.backends.brev.subprocess.run",
        return_value=mock.Mock(returncode=0, stdout="stdout-content"),
    ) as patched:
        out = brev._ssh_capture("box", "echo hi")
    assert out == "stdout-content"
    assert patched.call_args.kwargs.get("capture_output") is True


# -- _ensure_remote_rsync -------------------------------------------------


def test_ensure_remote_rsync_sends_expected_cmd():
    recorded = {}
    with mock.patch(
        "runplz.backends.brev._ssh",
        lambda instance, cmd: recorded.update({"i": instance, "c": cmd}),
    ):
        brev._ensure_remote_rsync("box")
    assert recorded["i"] == "box"
    assert "apt-get install" in recorded["c"]
    assert "rsync" in recorded["c"]


# -- _build_image ---------------------------------------------------------


def test_build_image_dockerfile_uses_docker_build_f(tmp_path):
    img = Image.from_dockerfile("docker/Dockerfile")
    recorded = {}
    with mock.patch("runplz.backends.brev._ssh", lambda i, c: recorded.setdefault("c", c)):
        brev._build_image("box", img)
    assert "docker build -f docker/Dockerfile" in recorded["c"]
    assert "__EOF__" not in recorded["c"]  # no synthesized Dockerfile heredoc


def test_build_image_from_registry_pipes_synthesized_dockerfile():
    img = Image.from_registry("pytorch/pytorch:2.4.0").pip_install("numpy")
    recorded = {}
    with mock.patch("runplz.backends.brev._ssh", lambda i, c: recorded.setdefault("c", c)):
        brev._build_image("box", img)
    c = recorded["c"]
    assert "cat <<'__EOF__' | sudo docker build -f - -t" in c
    assert "FROM pytorch/pytorch:2.4.0" in c


# -- _run_container_detached ----------------------------------------------


def test_run_container_detached_builds_full_docker_run(tmp_path):
    app = _app(tmp_path)
    image = Image.from_registry("ubuntu:22.04")
    fn = _function(app, image, module_file=_job_inside(tmp_path), env={"USER_VAR": "1"})

    recorded = {}
    with mock.patch("runplz.backends.brev._ssh", lambda i, c: recorded.setdefault("c", c)):
        brev._run_container_detached(
            instance="box",
            container_name="runplz-train-abc123",
            function=fn,
            rel_script="jobs/train.py",
            args=[],
            kwargs={},
            gpu_flag="--gpus all",
        )
    c = recorded["c"]
    assert "sudo docker run -d --name runplz-train-abc123" in c
    assert "--network=host" in c
    assert "--gpus all" in c
    assert "RUNPLZ_RUNNER_SCRIPT" not in c  # old env name — must be RUNPLZ_SCRIPT
    assert "RUNPLZ_SCRIPT" in c
    assert "USER_VAR=1" in c


# -- _stream_and_wait + _container_running -------------------------------


def test_stream_and_wait_exits_when_container_done():
    # First log-stream ssh returns rc=0; _container_running says not running.
    seq = iter(
        [
            mock.Mock(returncode=0),  # docker logs -f
            mock.Mock(returncode=0, stdout="false"),  # docker inspect (not running)
            mock.Mock(returncode=0, stdout="0\n"),  # docker wait → exit 0
        ]
    )

    def fake_run(cmd, *a, **kw):
        return next(seq)

    with mock.patch("runplz.backends.brev.subprocess.run", fake_run):
        rc = brev._stream_and_wait("box", "c")
    assert rc == 0


def test_stream_and_wait_reconnects_then_gives_up(capsys):
    # docker logs -f keeps returning and container_running keeps saying
    # True. After max_reconnects we should bail out and try docker wait.
    ssh_returns = mock.Mock(returncode=255)  # logs -f dropped
    inspect_true = mock.Mock(returncode=0, stdout="true")  # still running
    wait_final = mock.Mock(returncode=0, stdout="137\n")

    order = []

    def fake_run(cmd, *a, **kw):
        # Classify which call this is based on cmd contents.
        joined = " ".join(str(x) for x in cmd)
        order.append(joined)
        if "docker logs" in joined:
            return ssh_returns
        if "docker inspect" in joined:
            return inspect_true
        if "docker wait" in joined:
            return wait_final
        return mock.Mock(returncode=0, stdout="")

    # Keep the retry ceiling small so the test finishes fast.
    with mock.patch("runplz.backends.brev.subprocess.run", fake_run):
        with mock.patch("runplz.backends.brev.time.sleep", lambda _s: None):
            rc = brev._stream_and_wait("box", "c", max_reconnects=3)
    assert rc == 137
    # Should have emitted the give-up message once max_reconnects hit.
    assert "giving up on log stream" in capsys.readouterr().out


def test_container_running_treats_ssh_failure_as_still_running():
    # subprocess.run raises TimeoutExpired → conservative return True.
    with mock.patch(
        "runplz.backends.brev.subprocess.run",
        side_effect=__import__("subprocess").TimeoutExpired(cmd="ssh", timeout=30),
    ):
        assert brev._container_running("box", "c") is True


def test_container_running_returns_false_on_inspect_false():
    with mock.patch(
        "runplz.backends.brev.subprocess.run",
        return_value=mock.Mock(returncode=0, stdout="false\n"),
    ):
        assert brev._container_running("box", "c") is False


def test_container_running_true_on_running_output():
    with mock.patch(
        "runplz.backends.brev.subprocess.run",
        return_value=mock.Mock(returncode=0, stdout="true\n"),
    ):
        assert brev._container_running("box", "c") is True


# -- run() full happy paths -----------------------------------------------


def test_run_fails_when_instance_missing_without_auto_create(tmp_path):
    cfg = BrevConfig(auto_create=False, mode="vm")
    app = _app(tmp_path, cfg=cfg)
    fn = _function(app, Image.from_registry("ubuntu:22.04"), module_file=_job_inside(tmp_path))

    with mock.patch("runplz.backends.brev._require_brev_cli"):
        with mock.patch("runplz.backends.brev._skip_onboarding"):
            with mock.patch("runplz.backends.brev._instance_exists", return_value=False):
                with pytest.raises(RuntimeError, match="not found"):
                    brev.run(app, fn, [], {}, instance="gone")


def test_run_vm_docker_mode_end_to_end(tmp_path):
    cfg = BrevConfig(mode="vm", use_docker=True)
    app = _app(tmp_path, cfg=cfg)
    fn = _function(app, Image.from_registry("ubuntu:22.04"), module_file=_job_inside(tmp_path))

    with mock.patch.multiple(
        "runplz.backends.brev",
        _require_brev_cli=mock.DEFAULT,
        _skip_onboarding=mock.DEFAULT,
        _instance_exists=mock.Mock(return_value=True),
        _refresh_ssh=mock.DEFAULT,
        _ensure_docker=mock.DEFAULT,
        _rsync_up=mock.DEFAULT,
        _remote_has_nvidia=mock.Mock(return_value=True),
        _build_image=mock.DEFAULT,
        _run_container_detached=mock.DEFAULT,
        _stream_and_wait=mock.Mock(return_value=0),
        _ssh_capture=mock.DEFAULT,
        _rsync_down=mock.DEFAULT,
    ):
        brev.run(app, fn, [], {}, instance="box")


def test_run_vm_docker_mode_nonzero_exit_raises(tmp_path):
    cfg = BrevConfig(mode="vm", use_docker=True)
    app = _app(tmp_path, cfg=cfg)
    fn = _function(app, Image.from_registry("ubuntu:22.04"), module_file=_job_inside(tmp_path))

    with mock.patch.multiple(
        "runplz.backends.brev",
        _require_brev_cli=mock.DEFAULT,
        _skip_onboarding=mock.DEFAULT,
        _instance_exists=mock.Mock(return_value=True),
        _refresh_ssh=mock.DEFAULT,
        _ensure_docker=mock.DEFAULT,
        _rsync_up=mock.DEFAULT,
        _remote_has_nvidia=mock.Mock(return_value=False),
        _build_image=mock.DEFAULT,
        _run_container_detached=mock.DEFAULT,
        _stream_and_wait=mock.Mock(return_value=137),
        _ssh_capture=mock.DEFAULT,
        _rsync_down=mock.DEFAULT,
    ):
        with pytest.raises(RuntimeError, match="exited with status 137"):
            brev.run(app, fn, [], {}, instance="box")


def test_run_vm_native_mode_end_to_end(tmp_path):
    cfg = BrevConfig(mode="vm", use_docker=False)
    app = _app(tmp_path, cfg=cfg)
    fn = _function(app, Image.from_registry("ubuntu:22.04"), module_file=_job_inside(tmp_path))

    with mock.patch.multiple(
        "runplz.backends.brev",
        _require_brev_cli=mock.DEFAULT,
        _skip_onboarding=mock.DEFAULT,
        _instance_exists=mock.Mock(return_value=True),
        _refresh_ssh=mock.DEFAULT,
        _rsync_up=mock.DEFAULT,
        _remote_has_nvidia=mock.Mock(return_value=False),
        _run_native=mock.Mock(return_value=0),
        _rsync_down=mock.DEFAULT,
    ):
        brev.run(app, fn, [], {}, instance="box")


def test_run_container_mode_end_to_end(tmp_path):
    cfg = BrevConfig(mode="container")
    app = _app(tmp_path, cfg=cfg)
    fn = _function(
        app,
        Image.from_registry("pytorch/pytorch:2.4.0").pip_install("numpy"),
        module_file=_job_inside(tmp_path),
    )

    with mock.patch.multiple(
        "runplz.backends.brev",
        _require_brev_cli=mock.DEFAULT,
        _skip_onboarding=mock.DEFAULT,
        _instance_exists=mock.Mock(return_value=True),
        _refresh_ssh=mock.DEFAULT,
        _ensure_remote_rsync=mock.DEFAULT,
        _rsync_up=mock.DEFAULT,
        _run_container_mode=mock.Mock(return_value=0),
        _rsync_down=mock.DEFAULT,
    ):
        brev.run(app, fn, [], {}, instance="box")


def test_run_creates_instance_when_auto_create(tmp_path):
    cfg = BrevConfig(auto_create=True, mode="vm")
    app = _app(tmp_path, cfg=cfg)
    fn = _function(app, Image.from_registry("ubuntu:22.04"), module_file=_job_inside(tmp_path))

    created = {}

    def fake_create(name, instance_type, *, cfg=None, image=None, function=None):
        created["name"] = name
        created["instance_type"] = instance_type

    with mock.patch.multiple(
        "runplz.backends.brev",
        _require_brev_cli=mock.DEFAULT,
        _skip_onboarding=mock.DEFAULT,
        _instance_exists=mock.Mock(return_value=False),
        _create_instance=fake_create,
        _refresh_ssh=mock.DEFAULT,
        _ensure_docker=mock.DEFAULT,
        _rsync_up=mock.DEFAULT,
        _remote_has_nvidia=mock.Mock(return_value=False),
        _build_image=mock.DEFAULT,
        _run_container_detached=mock.DEFAULT,
        _stream_and_wait=mock.Mock(return_value=0),
        _ssh_capture=mock.DEFAULT,
        _rsync_down=mock.DEFAULT,
    ):
        brev.run(app, fn, [], {}, instance="newbox")

    assert created["name"] == "newbox"
    assert created["instance_type"] == cfg.instance_type


# -- _run_native and _run_container_mode remote commands ------------------


def test_run_native_builds_expected_remote_cmd(tmp_path):
    app = _app(tmp_path)
    fn = _function(
        app,
        Image.from_registry("ubuntu:22.04"),
        module_file=_job_inside(tmp_path),
        env={"EXTRA": "ok"},
    )

    recorded = {}

    def fake_sh_ssh(instance, cmd):
        # First call is setup, second call (subprocess.run) is the actual run.
        recorded.setdefault("setup", cmd)

    def fake_sub_run(cmd, *a, **kw):
        recorded["run_cmd"] = cmd
        return mock.Mock(returncode=0)

    with mock.patch("runplz.backends.brev._ssh", fake_sh_ssh):
        with mock.patch("runplz.backends.brev.subprocess.run", fake_sub_run):
            rc = brev._run_native(
                instance="box",
                function=fn,
                rel_script="jobs/train.py",
                args=[],
                kwargs={},
                has_nvidia=True,
            )
    assert rc == 0
    # Setup installs python3-venv & pip.
    assert "python3-venv" in recorded["setup"]
    # CUDA 12.1 wheel index when has_nvidia=True.
    assert "cu121" in recorded["setup"]
    # Run command has user env + bootstrap invocation.
    joined = " ".join(recorded["run_cmd"])
    assert "EXTRA=ok" in joined or "EXTRA" in recorded["run_cmd"][-1]
    assert "runplz.runners._bootstrap" in joined or "runplz._bootstrap" in joined


def test_run_native_cpu_index_url(tmp_path):
    app = _app(tmp_path)
    fn = _function(app, Image.from_registry("ubuntu:22.04"), module_file=_job_inside(tmp_path))

    recorded = {}

    def fake_ssh(i, c):
        recorded.setdefault("setup", c)

    with mock.patch("runplz.backends.brev._ssh", fake_ssh):
        with mock.patch(
            "runplz.backends.brev.subprocess.run",
            return_value=mock.Mock(returncode=0),
        ):
            brev._run_native(
                instance="box",
                function=fn,
                rel_script="jobs/train.py",
                args=[],
                kwargs={},
                has_nvidia=False,
            )
    assert "cpu" in recorded["setup"]
    assert "cu121" not in recorded["setup"]


def test_run_container_mode_builds_expected_remote_cmd(tmp_path):
    app = _app(tmp_path, cfg=BrevConfig(mode="container"))
    fn = _function(
        app,
        Image.from_registry("pytorch/pytorch:2.4.0").pip_install("pandas"),
        module_file=_job_inside(tmp_path),
    )

    recorded = {}

    def fake_ssh(i, c):
        # First call = ops_script install; ignore.
        recorded["ops"] = c

    def fake_sub_run(cmd, *a, **kw):
        recorded["run"] = cmd
        return mock.Mock(returncode=0)

    with mock.patch("runplz.backends.brev._ssh", fake_ssh):
        with mock.patch("runplz.backends.brev.subprocess.run", fake_sub_run):
            rc = brev._run_container_mode(
                instance="box",
                function=fn,
                rel_script="jobs/train.py",
                args=[],
                kwargs={},
            )
    assert rc == 0
    assert "pip install --quiet pandas" in recorded["ops"]
    assert "runplz._bootstrap" in recorded["run"][-1]
