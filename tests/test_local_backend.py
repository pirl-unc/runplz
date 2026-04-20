"""Local backend coverage — mocks subprocess.run to inspect the
docker commands runplz builds without actually invoking docker.
"""

from unittest import mock

import pytest

from runplz import App, Image
from runplz.backends import local


def _app(tmp_path, repo_root=None):
    app = App("demo")
    # Simulate what the CLI sets up: attach a repo root.
    app._repo_root = repo_root or tmp_path
    return app


def _function(app, image, *, module_file=None, **extra):
    """Build a Function by going through the real decorator. When a
    test pins repo_root at tmp_path, we need module_file to live inside
    tmp_path too so `_container_path_for` can `relative_to(repo)` it."""

    @app.function(image=image, **extra)
    def train():  # pragma: no cover — called only under runner control
        return "ok"

    fn = app.functions["train"]
    if module_file is not None:
        fn.module_file = str(module_file)
    return fn


def _job_inside_repo(tmp_path):
    """Create a dummy job file inside tmp_path and return its path."""
    jobdir = tmp_path / "jobs"
    jobdir.mkdir(parents=True, exist_ok=True)
    job = jobdir / "job.py"
    job.write_text("# fake\n")
    return job


def _run_calls_capture():
    """Return (fake_run, calls_list). calls_list holds (cmd, kwargs)
    tuples for every subprocess.run call made through the backend."""
    calls = []

    def fake_run(cmd, *args, **kwargs):
        calls.append((cmd, kwargs))
        return mock.Mock(returncode=0, stdout="{}", stderr="")

    return fake_run, calls


def test_run_raises_without_repo_root(tmp_path):
    app = App("demo")
    fn = _function(app, Image.from_registry("ubuntu:22.04"))
    with pytest.raises(RuntimeError, match="repo_root"):
        local.run(app, fn, [], {})


def test_run_builds_image_from_registry_and_runs(tmp_path):
    app = _app(tmp_path)
    image = (
        Image.from_registry("pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime")
        .apt_install("bzip2")
        .pip_install("numpy")
    )
    fn = _function(app, image, env={"FOO": "bar"}, module_file=_job_inside_repo(tmp_path))
    fake_run, calls = _run_calls_capture()

    with mock.patch("runplz.backends.local.subprocess.run", fake_run):
        # Force _nvidia_available() → False so --gpus all is not added.
        with mock.patch("runplz.backends.local._nvidia_available", return_value=False):
            local.run(app, fn, [1, 2], {"k": "v"})

    # First call = docker build from stdin; second = docker run.
    build, run_ = calls[0], calls[1]
    assert build[0][:4] == ["docker", "build", "-f", "-"]
    # input= is a synthesized Dockerfile, not a path.
    assert "FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime" in build[1]["input"]
    assert build[1]["text"] is True

    run_cmd = run_[0]
    assert run_cmd[:3] == ["docker", "run", "--rm"]
    assert "--gpus" not in run_cmd
    env_kvs = [run_cmd[i + 1] for i, x in enumerate(run_cmd) if x == "-e"]
    assert "RUNPLZ_OUT=/out" in env_kvs
    assert any(kv.startswith("RUNPLZ_SCRIPT=/workspace/") for kv in env_kvs)
    assert "RUNPLZ_FUNCTION=train" in env_kvs
    assert "RUNPLZ_ARGS=[1, 2]" in env_kvs
    assert 'RUNPLZ_KWARGS={"k": "v"}' in env_kvs
    assert "FOO=bar" in env_kvs
    assert run_cmd[-4:] == [local.IMAGE_TAG_DEFAULT, "python", "-m", "runplz._bootstrap"]


def test_run_adds_gpus_all_when_nvidia_runtime_detected(tmp_path):
    app = _app(tmp_path)
    image = Image.from_registry("ubuntu:22.04")
    fn = _function(app, image, module_file=_job_inside_repo(tmp_path))
    fake_run, calls = _run_calls_capture()

    with mock.patch("runplz.backends.local.subprocess.run", fake_run):
        with mock.patch("runplz.backends.local._nvidia_available", return_value=True):
            local.run(app, fn, [], {})

    run_cmd = calls[1][0]
    i = run_cmd.index("--gpus")
    assert run_cmd[i + 1] == "all"


def test_run_skips_build_when_build_false(tmp_path):
    app = _app(tmp_path)
    image = Image.from_registry("ubuntu:22.04")
    fn = _function(app, image, module_file=_job_inside_repo(tmp_path))
    fake_run, calls = _run_calls_capture()

    with mock.patch("runplz.backends.local.subprocess.run", fake_run):
        with mock.patch("runplz.backends.local._nvidia_available", return_value=False):
            local.run(app, fn, [], {}, build=False)

    # `docker image inspect` for the reused-tag warning, then `docker run`.
    # No `docker build`.
    assert [c[0][:3] for c in calls] == [
        ["docker", "image", "inspect"],
        ["docker", "run", "--rm"],
    ]


def test_build_false_logs_reused_image_tag(tmp_path, capsys):
    """Issue #21: --no-build must tell the user which image it's reusing,
    so a stale image can't silently rerun."""
    app = _app(tmp_path)
    image = Image.from_registry("ubuntu:22.04")
    fn = _function(app, image, module_file=_job_inside_repo(tmp_path))

    def fake_run(cmd, *a, **kw):
        if cmd[:3] == ["docker", "image", "inspect"]:
            return mock.Mock(returncode=0, stdout="2026-04-20T12:34:56Z\n", stderr="")
        return mock.Mock(returncode=0, stdout="", stderr="")

    with mock.patch("runplz.backends.local.subprocess.run", fake_run):
        with mock.patch("runplz.backends.local._nvidia_available", return_value=False):
            local.run(app, fn, [], {}, build=False, image_tag="my-tag")

    out = capsys.readouterr().out
    assert "build=False" in out
    assert "'my-tag'" in out
    assert "2026-04-20T12:34:56Z" in out


def test_build_false_warns_when_image_not_found(tmp_path, capsys):
    app = _app(tmp_path)
    image = Image.from_registry("ubuntu:22.04")
    fn = _function(app, image, module_file=_job_inside_repo(tmp_path))

    def fake_run(cmd, *a, **kw):
        if cmd[:3] == ["docker", "image", "inspect"]:
            return mock.Mock(returncode=1, stdout="", stderr="No such image")
        return mock.Mock(returncode=0, stdout="", stderr="")

    with mock.patch("runplz.backends.local.subprocess.run", fake_run):
        with mock.patch("runplz.backends.local._nvidia_available", return_value=False):
            local.run(app, fn, [], {}, build=False, image_tag="ghost")

    out = capsys.readouterr().out
    assert "'ghost'" in out
    assert "not found locally" in out


def test_run_uses_user_dockerfile_when_from_dockerfile(tmp_path):
    # Create a real Dockerfile so image.resolve() succeeds.
    (tmp_path / "Dockerfile").write_text("FROM ubuntu:22.04\n")
    app = _app(tmp_path)
    image = Image.from_dockerfile("Dockerfile")
    fn = _function(app, image, module_file=_job_inside_repo(tmp_path))
    fake_run, calls = _run_calls_capture()

    with mock.patch("runplz.backends.local.subprocess.run", fake_run):
        with mock.patch("runplz.backends.local._nvidia_available", return_value=False):
            local.run(app, fn, [], {})

    build = calls[0][0]
    assert build[:2] == ["docker", "build"]
    # -f points at a real file, not "-".
    i = build.index("-f")
    assert build[i + 1] != "-"
    assert "text" not in calls[0][1]  # no stdin-piped Dockerfile


def test_nvidia_available_parses_docker_info():
    with mock.patch(
        "runplz.backends.local.subprocess.run",
        return_value=mock.Mock(returncode=0, stdout='{"nvidia":{}}'),
    ):
        assert local._nvidia_available() is True
    with mock.patch(
        "runplz.backends.local.subprocess.run",
        return_value=mock.Mock(returncode=0, stdout="{}"),
    ):
        assert local._nvidia_available() is False
    with mock.patch(
        "runplz.backends.local.subprocess.run",
        return_value=mock.Mock(returncode=1, stdout=""),
    ):
        assert local._nvidia_available() is False


def test_container_path_maps_relative_to_workspace(tmp_path):
    job = tmp_path / "jobs" / "train.py"
    job.parent.mkdir(parents=True)
    job.write_text("pass\n")
    assert local._container_path_for(str(job), tmp_path) == "/workspace/jobs/train.py"


def test_host_out_is_created(tmp_path):
    app = _app(tmp_path)
    fn = _function(app, Image.from_registry("ubuntu:22.04"), module_file=_job_inside_repo(tmp_path))
    fake_run, _ = _run_calls_capture()
    with mock.patch("runplz.backends.local.subprocess.run", fake_run):
        with mock.patch("runplz.backends.local._nvidia_available", return_value=False):
            local.run(app, fn, [], {}, outputs_dir="artifacts")
    assert (tmp_path / "artifacts").is_dir()


def test_print_cmd_is_flushed(capsys):
    local._print_cmd(["docker", "run", "--rm", "hello:world"])
    out = capsys.readouterr().out
    assert "+ docker run --rm hello:world" in out
