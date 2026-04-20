"""SSH backend coverage — SshConfig validation, ssh.run dispatch, and
the spec-mismatch probe.

Subprocess/ssh/rsync are mocked; we never touch a real remote.
"""

import types
from unittest import mock

import pytest

from runplz import App, Image, SshConfig
from runplz.backends import ssh  # noqa: I001

# --- helpers -------------------------------------------------------------


def _app(tmp_path, cfg=None):
    app = App("demo", ssh_config=cfg or SshConfig(on_finish="leave"))
    app._repo_root = tmp_path
    return app


def _function(app, image, *, module_file=None, **extra):
    @app.function(image=image, **extra)
    def train():  # pragma: no cover
        return "ok"

    fn = app.functions["train"]
    if module_file is not None:
        fn.module_file = str(module_file)
    return fn


def _job_inside(tmp_path):
    j = tmp_path / "jobs"
    j.mkdir(parents=True, exist_ok=True)
    f = j / "job.py"
    f.write_text("# fake\n")
    return f


# --- SshConfig validation ------------------------------------------------


def test_ssh_config_defaults():
    cfg = SshConfig()
    assert cfg.user is None
    assert cfg.port is None
    assert cfg.use_docker is True
    assert cfg.on_finish == "leave"
    assert cfg.max_runtime_seconds is None


def test_ssh_config_rejects_invalid_port():
    with pytest.raises(ValueError, match="valid TCP port"):
        SshConfig(port=0)
    with pytest.raises(ValueError, match="valid TCP port"):
        SshConfig(port=70000)


def test_ssh_config_accepts_valid_ports():
    SshConfig(port=22)
    SshConfig(port=65535)


def test_ssh_config_rejects_empty_user():
    with pytest.raises(ValueError, match="non-empty"):
        SshConfig(user="   ")


def test_ssh_config_only_allows_on_finish_leave():
    # User owns the VM lifecycle; stop/delete semantics don't apply.
    with pytest.raises(ValueError, match="on_finish must be one of"):
        SshConfig(on_finish="stop")
    with pytest.raises(ValueError, match="on_finish must be one of"):
        SshConfig(on_finish="delete")


def test_ssh_config_rejects_non_positive_max_runtime_seconds():
    with pytest.raises(ValueError, match="positive int"):
        SshConfig(max_runtime_seconds=0)
    with pytest.raises(ValueError, match="positive int"):
        SshConfig(max_runtime_seconds=-1)


# --- target string construction -----------------------------------------


def test_build_ssh_target_bare_host():
    assert ssh._build_ssh_target("my-box", user=None, port=None) == "my-box"


def test_build_ssh_target_user_from_config_wins():
    # Host has no user — user comes from cfg.
    assert ssh._build_ssh_target("my-box", user="alex", port=None) == "alex@my-box"


def test_build_ssh_target_user_preserved_when_in_host():
    # Host carries a user; cfg.user=None → keep what the user gave us.
    assert ssh._build_ssh_target("root@my-box", user=None, port=None) == "root@my-box"


def test_build_ssh_target_cfg_user_overrides_url_user(capsys):
    # Conflict: URL says root, cfg says alex. cfg wins — explicit over implicit.
    assert ssh._build_ssh_target("root@my-box", user="alex", port=None) == "alex@my-box"


def test_build_ssh_target_port_warning(capsys):
    ssh._build_ssh_target("my-box", user=None, port=2222)
    out = capsys.readouterr().out
    assert "port=2222" in out
    assert "warning" in out.lower()


# --- App.bind wiring -----------------------------------------------------


def test_app_bind_ssh_requires_host(tmp_path):
    app = _app(tmp_path)
    _function(app, Image.from_registry("ubuntu:22.04"), module_file=_job_inside(tmp_path))
    with pytest.raises(ValueError, match="host=... is required"):
        app.bind("ssh")


def test_app_bind_ssh_rejects_instance_kwarg(tmp_path):
    app = _app(tmp_path)
    _function(app, Image.from_registry("ubuntu:22.04"), module_file=_job_inside(tmp_path))
    with pytest.raises(ValueError, match="instance="):
        app.bind("ssh", host="foo", instance="bar")


def test_app_bind_ssh_rejects_host_on_non_ssh(tmp_path):
    app = _app(tmp_path)
    _function(app, Image.from_registry("ubuntu:22.04"), module_file=_job_inside(tmp_path))
    with pytest.raises(ValueError, match="host="):
        app.bind("local", host="foo")


def test_app_bind_ssh_threads_host_into_backend_kwargs(tmp_path):
    app = _app(tmp_path)
    _function(app, Image.from_registry("ubuntu:22.04"), module_file=_job_inside(tmp_path))
    app.bind("ssh", host="gpu.example.com")
    assert app._backend == "ssh"
    assert app._backend_kwargs["host"] == "gpu.example.com"


# --- spec-mismatch probe -------------------------------------------------


def _probe_output(nproc=None, mem_kb=None, gpus=None):
    """Build a fake _ssh_capture return mimicking the real probe payload."""
    lines = ["---NPROC---"]
    if nproc is not None:
        lines.append(str(nproc))
    lines.append("---MEMINFO---")
    if mem_kb is not None:
        lines.append(f"MemTotal:       {mem_kb} kB")
    lines.append("---NVIDIA---")
    for name, mib in gpus or []:
        lines.append(f"{name}, {mib} MiB")
    lines.append("---END---")
    return "\n".join(lines)


def _fn(min_cpu=None, min_memory=None, gpu=None, min_gpu_memory=None, num_gpus=1):
    return types.SimpleNamespace(
        min_cpu=min_cpu,
        min_memory=min_memory,
        gpu=gpu,
        min_gpu_memory=min_gpu_memory,
        num_gpus=num_gpus,
    )


def test_spec_probe_warns_on_low_cpu(capsys):
    probe = _probe_output(nproc=2, mem_kb=64 * 1024 * 1024, gpus=[])
    with mock.patch("runplz.backends.ssh._ssh_capture", return_value=probe):
        ssh._warn_on_spec_mismatch("box", _fn(min_cpu=8))
    out = capsys.readouterr().out
    assert "spec-mismatch warning" in out
    assert "min_cpu=8" in out
    assert "2 vCPUs" in out


def test_spec_probe_warns_on_low_memory(capsys):
    probe = _probe_output(nproc=16, mem_kb=8 * 1024 * 1024, gpus=[])  # 8GB
    with mock.patch("runplz.backends.ssh._ssh_capture", return_value=probe):
        ssh._warn_on_spec_mismatch("box", _fn(min_memory=32))
    out = capsys.readouterr().out
    assert "min_memory=32" in out
    assert "8.0 GB" in out


def test_spec_probe_warns_when_function_wants_gpu_but_none_found(capsys):
    probe = _probe_output(nproc=16, mem_kb=64 * 1024 * 1024, gpus=[])
    with mock.patch("runplz.backends.ssh._ssh_capture", return_value=probe):
        ssh._warn_on_spec_mismatch("box", _fn(gpu="A100"))
    out = capsys.readouterr().out
    assert "A100" in out
    assert "no GPUs" in out


def test_spec_probe_warns_on_wrong_gpu_model(capsys):
    probe = _probe_output(nproc=16, mem_kb=64 * 1024 * 1024, gpus=[("Tesla T4", 16384)])
    with mock.patch("runplz.backends.ssh._ssh_capture", return_value=probe):
        ssh._warn_on_spec_mismatch("box", _fn(gpu="A100"))
    out = capsys.readouterr().out
    assert "A100" in out
    assert "Tesla T4" in out


def test_spec_probe_warns_on_insufficient_vram(capsys):
    # Remote has a T4 (16GB); function wants 40GB of VRAM.
    probe = _probe_output(nproc=16, mem_kb=64 * 1024 * 1024, gpus=[("Tesla T4", 16384)])
    with mock.patch("runplz.backends.ssh._ssh_capture", return_value=probe):
        ssh._warn_on_spec_mismatch("box", _fn(gpu="T4", min_gpu_memory=40))
    out = capsys.readouterr().out
    assert "min_gpu_memory=40" in out
    assert "16.0 GB" in out


def test_spec_probe_warns_when_num_gpus_exceeds_available(capsys):
    """3.6: num_gpus=4 on a box with 1 GPU → warn."""
    probe = _probe_output(nproc=16, mem_kb=128 * 1024 * 1024, gpus=[("NVIDIA A100 80GB", 81920)])
    with mock.patch("runplz.backends.ssh._ssh_capture", return_value=probe):
        ssh._warn_on_spec_mismatch("box", _fn(gpu="A100", num_gpus=4))
    out = capsys.readouterr().out
    assert "num_gpus=4" in out
    assert "1 GPU" in out


def test_spec_probe_silent_when_everything_matches(capsys):
    probe = _probe_output(nproc=16, mem_kb=128 * 1024 * 1024, gpus=[("NVIDIA A100 80GB", 81920)])
    with mock.patch("runplz.backends.ssh._ssh_capture", return_value=probe):
        ssh._warn_on_spec_mismatch(
            "box", _fn(min_cpu=8, min_memory=64, gpu="A100", min_gpu_memory=40)
        )
    assert "spec-mismatch" not in capsys.readouterr().out


def test_spec_probe_never_raises_on_ssh_failure(capsys):
    def boom(*a, **kw):
        raise RuntimeError("ssh went away")

    with mock.patch("runplz.backends.ssh._ssh_capture", boom):
        ssh._warn_on_spec_mismatch("box", _fn(min_cpu=8))
    out = capsys.readouterr().out
    # Helper prints a soft warning and moves on — no exception escapes.
    assert "could not probe remote specs" in out


# --- ssh.run dispatch end-to-end -----------------------------------------


def test_ssh_run_vm_docker_happy_path(tmp_path):
    cfg = SshConfig()  # defaults: use_docker=True, on_finish="leave"
    app = _app(tmp_path, cfg)
    fn = _function(
        app,
        Image.from_registry("ubuntu:22.04"),
        module_file=_job_inside(tmp_path),
    )

    with mock.patch.multiple(
        "runplz.backends.ssh",
        _wait_until_ssh_reachable=mock.DEFAULT,
        _warn_on_spec_mismatch=mock.DEFAULT,
        _ensure_remote_rsync=mock.DEFAULT,
        _rsync_up=mock.DEFAULT,
        _ensure_docker=mock.DEFAULT,
        _remote_has_nvidia=mock.Mock(return_value=True),
        _build_image=mock.DEFAULT,
        _run_container_detached=mock.DEFAULT,
        _stream_and_wait=mock.Mock(return_value=0),
        _ssh_capture=mock.DEFAULT,
        _rsync_down=mock.DEFAULT,
    ):
        ssh.run(app, fn, [], {}, host="gpu.example.com")


def test_ssh_run_native_happy_path(tmp_path):
    cfg = SshConfig(use_docker=False)
    app = _app(tmp_path, cfg)
    fn = _function(app, Image.from_registry("ubuntu:22.04"), module_file=_job_inside(tmp_path))

    with mock.patch.multiple(
        "runplz.backends.ssh",
        _wait_until_ssh_reachable=mock.DEFAULT,
        _warn_on_spec_mismatch=mock.DEFAULT,
        _ensure_remote_rsync=mock.DEFAULT,
        _rsync_up=mock.DEFAULT,
        _remote_has_nvidia=mock.Mock(return_value=False),
        _run_native=mock.Mock(return_value=0),
        _rsync_down=mock.DEFAULT,
    ):
        ssh.run(app, fn, [], {}, host="user@gpu.example.com")


def test_ssh_run_nonzero_exit_raises_with_tail(tmp_path):
    """When the remote run fails, the RuntimeError must include the log
    tail just like the Brev backend."""
    cfg = SshConfig()
    app = _app(tmp_path, cfg)
    fn = _function(app, Image.from_registry("ubuntu:22.04"), module_file=_job_inside(tmp_path))

    with mock.patch.multiple(
        "runplz.backends.ssh",
        _wait_until_ssh_reachable=mock.DEFAULT,
        _warn_on_spec_mismatch=mock.DEFAULT,
        _ensure_remote_rsync=mock.DEFAULT,
        _rsync_up=mock.DEFAULT,
        _ensure_docker=mock.DEFAULT,
        _remote_has_nvidia=mock.Mock(return_value=False),
        _build_image=mock.DEFAULT,
        _run_container_detached=mock.DEFAULT,
        _stream_and_wait=mock.Mock(return_value=42),
        _ssh_capture=mock.DEFAULT,
        _rsync_down=mock.DEFAULT,
        _fetch_failure_tail=mock.Mock(return_value="KeyError: missing-key"),
    ):
        with pytest.raises(RuntimeError) as ei:
            ssh.run(app, fn, [], {}, host="gpu.example.com")

    msg = str(ei.value)
    assert "exited with status 42" in msg
    assert "KeyError: missing-key" in msg
