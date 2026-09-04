"""Modal backend coverage — test image rendering, env wiring,
entrypoint-template generation, and subprocess invocation without
actually shelling out to `modal` or hitting Modal's servers.
"""

import io
import json
import sys
import tarfile
import types
import warnings
from pathlib import Path
from unittest import mock

import pytest

from runplz import App, Image
from runplz.backends import modal as modal_backend

# --- _modal_gpu_string ----------------------------------------------------


def test_modal_gpu_string_passthrough_when_no_min_vram():
    assert modal_backend._modal_gpu_string("A100", None) == "A100"
    # No GPU constraints at all → no GPU.
    assert modal_backend._modal_gpu_string(None, None) is None
    # 3.14.0+: gpu=None with min_gpu_memory= now derives a default model
    # so the same script runs on Modal as on Brev. 80 GB → A100-80GB; the
    # VRAM suffix is already in the derived name so no double-suffixing.
    assert modal_backend._modal_gpu_string(None, 80) == "A100-80GB"


def test_list_jobs_falls_back_to_text_and_rejects_cli_failure(monkeypatch):
    monkeypatch.setitem(sys.modules, "modal", types.ModuleType("modal"))
    responses = [
        mock.Mock(returncode=1, stdout="not json", stderr="old cli"),
        mock.Mock(returncode=0, stdout="id  runplz-app-train  running", stderr=""),
    ]
    with mock.patch.object(modal_backend.subprocess, "run", side_effect=responses) as run:
        jobs = modal_backend.list_jobs()
    assert jobs[0].function == "train"
    assert run.call_count == 2

    with mock.patch.object(
        modal_backend.subprocess,
        "run",
        return_value=mock.Mock(returncode=2, stdout="", stderr="boom"),
    ):
        with pytest.raises(RuntimeError, match="failed"):
            modal_backend.list_jobs()


def test_list_jobs_requires_modal_package(monkeypatch):
    monkeypatch.setitem(sys.modules, "modal", None)
    with pytest.raises(RuntimeError, match=r"runplz\[modal\].*modal setup"):
        modal_backend.list_jobs()


def test_modal_gpu_string_appends_suffix():
    assert modal_backend._modal_gpu_string("A100", 80) == "A100-80GB"
    assert modal_backend._modal_gpu_string("H100", 40) == "H100-40GB"
    assert modal_backend._modal_gpu_string("T4", 16) == "T4-16GB"


def test_modal_gpu_string_respects_existing_suffix():
    # User pinned a size already — don't double-suffix.
    assert modal_backend._modal_gpu_string("A100-80GB", 40) == "A100-80GB"
    assert modal_backend._modal_gpu_string("L4-24gb", 16) == "L4-24gb"


def test_modal_gpu_string_appends_count_when_num_gpus_gt_1():
    """3.6: num_gpus > 1 maps to Modal's `:N` count suffix."""
    assert modal_backend._modal_gpu_string("A100", None, 4) == "A100:4"
    assert modal_backend._modal_gpu_string("A100", 80, 4) == "A100-80GB:4"
    assert modal_backend._modal_gpu_string("H100", None, 8) == "H100:8"


def test_modal_gpu_string_respects_existing_count_suffix():
    # User pinned "A100:2" already — don't double-suffix.
    assert modal_backend._modal_gpu_string("A100-80GB:2", 80, 4) == "A100-80GB:2"


def test_modal_gpu_string_num_gpus_one_omits_count():
    # Default num_gpus=1 shouldn't add ":1" noise.
    assert modal_backend._modal_gpu_string("A100", None, 1) == "A100"
    assert modal_backend._modal_gpu_string("A100", 80, 1) == "A100-80GB"


# --- render_modal_image ---------------------------------------------------


def test_render_modal_image_from_registry_emits_chain(tmp_path):
    img = (
        Image.from_registry("pytorch/pytorch:2.4.0")
        .apt_install("bzip2", "rsync")
        .pip_install("pandas>=2.0", "numpy")
        .pip_install_local_dir(".", editable=True)
    )
    src = modal_backend._render_modal_image(img, repo=tmp_path)
    assert "image = modal.Image.from_registry('pytorch/pytorch:2.4.0')" in src
    assert "image = image.apt_install('bzip2', 'rsync')" in src
    assert "image = image.pip_install('pandas>=2.0', 'numpy')" in src
    assert "image.add_local_dir(" in src
    assert "pip install -e /workspace" in src


def test_render_modal_image_from_dockerfile(tmp_path):
    (tmp_path / "Dockerfile.X").write_text("FROM ubuntu:22.04\n")
    img = Image.from_dockerfile("Dockerfile.X")
    src = modal_backend._render_modal_image(img, repo=tmp_path)
    assert "modal.Image.from_dockerfile(" in src
    assert "Dockerfile.X" in src


def test_render_modal_image_non_editable_install(tmp_path):
    img = Image.from_registry("ubuntu:22.04").pip_install_local_dir(".", editable=False)
    src = modal_backend._render_modal_image(img, repo=tmp_path)
    # Non-editable should produce `pip install /workspace`, no `-e`.
    assert "pip install /workspace" in src
    assert "pip install -e /workspace" not in src


def test_render_modal_image_passes_default_secret_ignores_to_add_local_dir(tmp_path):
    """Issue #18: .env / ssh keys / credentials.json must not be baked
    into the Modal image layer."""
    from runplz.excludes import DEFAULT_TRANSFER_EXCLUDES

    img = Image.from_registry("ubuntu:22.04").pip_install_local_dir(".", editable=False)
    src = modal_backend._render_modal_image(img, repo=tmp_path)

    assert "ignore=[" in src or "ignore=(" in src
    for pat in DEFAULT_TRANSFER_EXCLUDES:
        assert repr(pat) in src, f"missing {pat!r} from add_local_dir ignore list"


def test_render_modal_image_pip_install_with_index_url(tmp_path):
    img = Image.from_registry("ubuntu:22.04").pip_install(
        "torch", index_url="https://download.pytorch.org/whl/cu121"
    )
    src = modal_backend._render_modal_image(img, repo=tmp_path)
    assert "index_url='https://download.pytorch.org/whl/cu121'" in src


def test_render_modal_image_run_commands(tmp_path):
    img = Image.from_registry("ubuntu:22.04").run_commands("echo hi", "pip install more")
    src = modal_backend._render_modal_image(img, repo=tmp_path)
    assert "image = image.run_commands('echo hi', 'pip install more')" in src


def test_render_modal_image_requires_base_or_dockerfile(tmp_path):
    # An Image object with neither set (construct manually around the
    # frozen-dataclass validation that normally happens via constructors).
    bad = types.SimpleNamespace(base=None, dockerfile=None, ops=())
    with pytest.raises(ValueError, match="neither base nor dockerfile"):
        modal_backend._render_modal_image(bad, repo=tmp_path)


# --- _extract_tar ---------------------------------------------------------


def test_extract_tar_unpacks_to_dest(tmp_path):
    blob_buf = io.BytesIO()
    with tarfile.open(fileobj=blob_buf, mode="w:gz") as tar:
        content = b"hello from modal\n"
        info = tarfile.TarInfo("a/b.txt")
        info.size = len(content)
        tar.addfile(info, io.BytesIO(content))
    blob_path = tmp_path / "blob.tar.gz"
    blob_path.write_bytes(blob_buf.getvalue())

    dest = tmp_path / "unpacked"
    dest.mkdir()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        modal_backend._extract_tar(str(blob_path), dest)
    assert (dest / "a" / "b.txt").read_text() == "hello from modal\n"
    assert not [w for w in caught if issubclass(w.category, DeprecationWarning)]


def test_extract_tar_rejects_path_traversal(tmp_path):
    blob_buf = io.BytesIO()
    with tarfile.open(fileobj=blob_buf, mode="w:gz") as tar:
        content = b"boom\n"
        info = tarfile.TarInfo("../escape.txt")
        info.size = len(content)
        tar.addfile(info, io.BytesIO(content))
    blob_path = tmp_path / "blob.tar.gz"
    blob_path.write_bytes(blob_buf.getvalue())

    dest = tmp_path / "unpacked"
    dest.mkdir()
    with pytest.raises(RuntimeError, match="unsafe tar member"):
        modal_backend._extract_tar(str(blob_path), dest)
    assert not (tmp_path / "escape.txt").exists()


# --- run() end-to-end (mocked) -------------------------------------------


def _app_with_job(tmp_path):
    app = App("pan-allele")
    app.repo_root = tmp_path
    (tmp_path / "jobs").mkdir()
    job = tmp_path / "jobs" / "train.py"
    job.write_text("# fake\n")
    image = Image.from_registry("pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime").pip_install(
        "numpy"
    )

    @app.function(image=image, gpu="T4", min_cpu=4, min_memory=26, timeout=3600, env={"FOO": "bar"})
    def train():  # pragma: no cover
        pass

    fn = app.functions["train"]
    fn.module_file = str(job)
    return app, fn


def _fake_tarball_blob():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        info = tarfile.TarInfo("weights.bin")
        info.size = 5
        tar.addfile(info, io.BytesIO(b"abcde"))
    return buf.getvalue()


def test_run_requires_modal_package(tmp_path, monkeypatch):
    app, fn = _app_with_job(tmp_path)
    # Simulate `import modal` failing — override sys.modules lookup.
    monkeypatch.setitem(sys.modules, "modal", None)
    # Bypass the finder that would resolve modal; ImportError is what we
    # get when Python's import machinery finds `None` in sys.modules.
    # Simpler: patch `__import__` to raise on "modal".
    real_import = (
        __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__
    )

    def fake_import(name, *a, **kw):
        if name == "modal":
            raise ImportError("simulated modal-absent")
        return real_import(name, *a, **kw)

    with mock.patch("builtins.__import__", fake_import):
        with pytest.raises(RuntimeError, match="runplz\\[modal\\]|pip install modal|Modal backend"):
            modal_backend.run(app, fn, [], {})


def test_run_requires_repo_root(tmp_path):
    app, fn = _app_with_job(tmp_path)
    app.repo_root = None
    with pytest.raises(RuntimeError, match="repo_root"):
        modal_backend.run(app, fn, [], {})


def test_run_shells_modal_with_generated_entrypoint_and_extracts_tar(tmp_path):
    app, fn = _app_with_job(tmp_path)
    # Capture the entrypoint file that gets written + the modal run cmd.
    written_files = {}
    real_open = open

    def tracking_open(p, *args, **kwargs):
        handle = real_open(p, *args, **kwargs)
        if str(p).endswith("_modal_entry.py") and "w" in (
            args[0] if args else kwargs.get("mode", "")
        ):
            written_files[str(p)] = handle
        return handle

    calls = []
    blob_bytes = _fake_tarball_blob()

    def fake_run(cmd, *a, **kw):
        calls.append(cmd)
        # Simulate modal run having produced the tar blob at the path
        # embedded in the generated entrypoint. We can find that path by
        # scanning the last-written entry file for `_OUT_BLOB = ...`.
        entry_file = cmd[-1].split("::")[0]
        content = Path(entry_file).read_text()
        # Extract the blob path via regex-lite parse.
        for line in content.splitlines():
            if line.startswith("_OUT_BLOB = "):
                out_blob = line.split("=", 1)[1].strip().strip("'\"")
                Path(out_blob).write_bytes(blob_bytes)
                break
        return mock.Mock(returncode=0)

    with mock.patch("runplz.backends.modal.subprocess.run", fake_run):
        modal_backend.run(app, fn, [1], {"k": "v"})

    # Called `modal run <tmpfile>::main`
    assert len(calls) == 1
    cmd = calls[0]
    assert cmd[0] == "modal"
    assert cmd[1] == "run"
    assert cmd[2].endswith("::main")

    # The tar blob we planted should have been extracted to out/.
    assert (tmp_path / "out" / "weights.bin").read_bytes() == b"abcde"


def test_run_memory_gb_to_mb_conversion(tmp_path):
    app, fn = _app_with_job(tmp_path)  # min_memory=26 (GB)
    captured_src = {}

    def fake_run(cmd, *a, **kw):
        entry_file = cmd[-1].split("::")[0]
        captured_src["src"] = Path(entry_file).read_text()
        # Create the expected blob so _extract_tar doesn't crash.
        for line in captured_src["src"].splitlines():
            if line.startswith("_OUT_BLOB = "):
                out_blob = line.split("=", 1)[1].strip().strip("'\"")
                Path(out_blob).write_bytes(_fake_tarball_blob())
        return mock.Mock(returncode=0)

    with mock.patch("runplz.backends.modal.subprocess.run", fake_run):
        modal_backend.run(app, fn, [], {})

    # 26 GB → 26624 MB.
    assert "_MEMORY = 26624" in captured_src["src"]
    assert "_GPU = 'T4'" in captured_src["src"]
    assert "_CPU = 4" in captured_src["src"]
    # Our env/flags made it into container_env:
    assert "'RUNPLZ_OUT': '/out'" in captured_src["src"]
    assert "'RUNPLZ_FUNCTION': 'train'" in captured_src["src"]
    assert "'FOO': 'bar'" in captured_src["src"]


def test_run_min_gpu_memory_appends_suffix(tmp_path, capsys):
    app = App("x")
    app.repo_root = tmp_path
    (tmp_path / "jobs").mkdir()
    (tmp_path / "jobs" / "j.py").write_text("pass\n")

    @app.function(image=Image.from_registry("ubuntu:22.04"), gpu="A100", min_gpu_memory=80)
    def t():  # pragma: no cover
        pass

    fn = app.functions["t"]
    fn.module_file = str(tmp_path / "jobs" / "j.py")
    captured = {}

    def fake_run(cmd, *a, **kw):
        entry = cmd[-1].split("::")[0]
        captured["src"] = Path(entry).read_text()
        for line in captured["src"].splitlines():
            if line.startswith("_OUT_BLOB = "):
                Path(line.split("=", 1)[1].strip().strip("'\"")).write_bytes(_fake_tarball_blob())
        return mock.Mock(returncode=0)

    with mock.patch("runplz.backends.modal.subprocess.run", fake_run):
        modal_backend.run(app, fn, [], {})

    # min_gpu_memory=80 baked into the gpu string as -80GB suffix.
    assert "_GPU = 'A100-80GB'" in captured["src"]


def test_check_output_blob_size_warns_near_cap(tmp_path, capsys):
    """Issue #19: a tar approaching Modal's 256MB return-value cap should
    emit a loud warning so users switch to Volumes before they hit it."""
    blob = tmp_path / "out.tar.gz"
    blob.write_bytes(b"x" * (210 * 1024 * 1024))
    modal_backend._check_output_blob_size(str(blob))

    out = capsys.readouterr().out
    assert "warning" in out.lower()
    assert "Modal Volume" in out
    assert "210.0 MB" in out


def test_check_output_blob_size_raises_over_cap(tmp_path):
    """At or above 256MB we raise instead of unpacking — the tar may already
    be truncated and extracting it silently would lose data."""
    blob = tmp_path / "out.tar.gz"
    blob.write_bytes(b"x" * (260 * 1024 * 1024))
    with pytest.raises(RuntimeError) as ei:
        modal_backend._check_output_blob_size(str(blob))

    msg = str(ei.value)
    assert "260.0 MB" in msg
    assert "may be truncated" in msg
    assert "Modal Volume" in msg


def test_check_output_blob_size_silent_under_warn_threshold(tmp_path, capsys):
    blob = tmp_path / "out.tar.gz"
    blob.write_bytes(b"x" * (10 * 1024 * 1024))
    modal_backend._check_output_blob_size(str(blob))
    assert capsys.readouterr().out == ""


def test_run_min_disk_raises_on_modal(tmp_path):
    """Issue #20: min_disk used to print a warning and silently drop.
    Now it's a hard ValueError at dispatch so users can't believe their
    disk request was honored."""
    app = App("x")
    app.repo_root = tmp_path
    (tmp_path / "jobs").mkdir()
    (tmp_path / "jobs" / "j.py").write_text("pass\n")

    @app.function(image=Image.from_registry("ubuntu:22.04"), gpu="T4", min_disk=200)
    def t():  # pragma: no cover
        pass

    fn = app.functions["t"]
    fn.module_file = str(tmp_path / "jobs" / "j.py")

    with pytest.raises(ValueError) as ei:
        modal_backend.run(app, fn, [], {})

    msg = str(ei.value)
    assert "min_disk=200" in msg
    assert "not supported on the Modal backend" in msg
    assert "Modal Volume" in msg


def test_run_no_memory_emits_none(tmp_path):
    app = App("x")
    app.repo_root = tmp_path
    (tmp_path / "jobs").mkdir()
    (tmp_path / "jobs" / "j.py").write_text("pass\n")

    @app.function(image=Image.from_registry("ubuntu:22.04"))
    def t():  # pragma: no cover
        pass

    fn = app.functions["t"]
    fn.module_file = str(tmp_path / "jobs" / "j.py")
    captured = {}

    def fake_run(cmd, *a, **kw):
        entry = cmd[-1].split("::")[0]
        captured["src"] = Path(entry).read_text()
        for line in captured["src"].splitlines():
            if line.startswith("_OUT_BLOB = "):
                Path(line.split("=", 1)[1].strip().strip("'\"")).write_bytes(_fake_tarball_blob())
        return mock.Mock(returncode=0)

    with mock.patch("runplz.backends.modal.subprocess.run", fake_run):
        modal_backend.run(app, fn, [], {})

    assert "_MEMORY = None" in captured["src"]
    assert "_GPU = None" in captured["src"]


def test_run_cleans_up_entrypoint_and_blob_files(tmp_path):
    app, fn = _app_with_job(tmp_path)
    captured_paths = {}

    def fake_run(cmd, *a, **kw):
        entry_file = cmd[-1].split("::")[0]
        captured_paths["entry"] = entry_file
        src = Path(entry_file).read_text()
        for line in src.splitlines():
            if line.startswith("_OUT_BLOB = "):
                p = line.split("=", 1)[1].strip().strip("'\"")
                captured_paths["blob"] = p
                Path(p).write_bytes(_fake_tarball_blob())
        return mock.Mock(returncode=0)

    with mock.patch("runplz.backends.modal.subprocess.run", fake_run):
        modal_backend.run(app, fn, [], {})

    # Both temp files should be cleaned up after a successful run.
    assert not Path(captured_paths["entry"]).exists()
    assert not Path(captured_paths["blob"]).exists()


def test_list_jobs_reads_the_display_name_out_of_description(monkeypatch):
    """Modal client 1.1.4 reports the app display name under `Description`.

    The parser only knew `name` / `App Name` / `Name`, so the name came back
    empty, every app failed the `runplz-` prefix filter, and `runplz ps`
    reported no Modal jobs at all — silently, because an app that parses to no
    name is indistinguishable from someone else's app (#142). Fixture is the
    real 1.1.4 response shape.
    """
    payload = json.dumps(
        [
            {
                "App ID": "ap-example",
                "Description": "runplz-example-train",
                "State": "ephemeral (detached)",
                "Tasks": "1",
            },
            {"App ID": "ap-other", "Description": "someone-elses-app", "State": "deployed"},
        ]
    )
    monkeypatch.setitem(sys.modules, "modal", types.ModuleType("modal"))
    with mock.patch.object(
        modal_backend.subprocess,
        "run",
        return_value=mock.Mock(returncode=0, stdout=payload, stderr=""),
    ):
        jobs = modal_backend.list_jobs()

    assert [j.name for j in jobs] == ["runplz-example-train"]
    assert (jobs[0].app, jobs[0].function) == ("example", "train")
    assert jobs[0].status == "ephemeral (detached)"


def test_a_runplz_name_wins_over_an_unrelated_field_that_comes_first():
    """Field order is not the contract. If a future shape puts an app id in
    `Name` and the display name in `Description`, taking the first non-empty
    field would fail the prefix filter and lose the job the same way #142 did.
    """
    row = {"Name": "ap-0123456789", "Description": "runplz-demo-train"}
    assert modal_backend._display_name(row) == "runplz-demo-train"


def test_a_row_with_no_runplz_name_still_reports_something_identifiable():
    """Falling back to the first non-empty field keeps unrelated apps
    debuggable rather than blank — they are filtered out either way."""
    assert modal_backend._display_name({"Name": "someone-else"}) == "someone-else"
    assert modal_backend._display_name({}) == ""


def test_the_prefix_that_is_stamped_is_the_prefix_that_is_filtered():
    """One constant, so a rename cannot make runplz stop recognising its own
    apps — the shape of the bug this fixes."""
    assert modal_backend.APP_PREFIX == "runplz-"
    assert modal_backend._split_modal_app_name(f"{modal_backend.APP_PREFIX}demo-train") == (
        "demo",
        "train",
    )


# ---------------------------------------------------------------------------
# Volume-backed outputs (issue #143)


def _volume_app(tmp_path, volumes):
    from runplz import App, Image

    app = App("vision")

    @app.function(image=Image.from_registry("ubuntu:22.04"), volumes=volumes)
    def train():  # pragma: no cover - never executed locally
        pass

    fn = app.functions["train"]
    fn.module_file = str(tmp_path / "job.py")
    app.repo_root = tmp_path
    return app, fn


def test_the_readme_volume_example_is_accepted():
    """The contract README has documented since 3.24.31 as *the* answer to
    Modal's ~256 MB return cap. `App.function()` never accepted `volumes`, so
    the documented example raised TypeError (#143)."""
    from runplz import App, Image

    app = App("train")

    @app.function(
        image=Image.from_registry("ubuntu:22.04"), gpu="T4", volumes={"/out": "training-outputs"}
    )
    def train():  # pragma: no cover
        pass

    assert app.functions["train"].volumes == {"/out": "training-outputs"}


def test_a_volume_backed_run_keeps_outputs_off_the_function_return(tmp_path):
    """The point of the whole feature: a multi-GB outputs directory must not be
    tarred into a return value capped at ~256 MB. The generated runner returns
    empty for a volume-backed /out, and the local side downloads instead."""
    _, fn = _volume_app(tmp_path, {"/out": "training-outputs"})
    src = modal_backend._ENTRYPOINT_TEMPLATE.format(
        app_name="runplz-vision-train",
        gpu=None,
        cpu=None,
        memory=None,
        timeout=60,
        out_blob=str(tmp_path / "blob"),
        container_env={},
        image_construction="image = modal.Image.debian_slim()",
        volumes=fn.volumes,
        out_on_volume=modal_backend._outputs_are_volume_backed(fn.volumes),
    )
    compile(src, "<generated>", "exec")
    assert "_OUT_ON_VOLUME = True" in src
    assert "modal.Volume.from_name(name, create_if_missing=True)" in src
    assert "volumes=volumes" in src
    # The tar is still in the file for the non-volume path, but it is now
    # behind the guard rather than unconditional.
    assert src.index("if _OUT_ON_VOLUME:") < src.index("tarfile.open")


def test_without_a_volume_the_outputs_still_come_back_through_the_return(tmp_path):
    """Existing small-output behaviour is untouched."""
    _, fn = _volume_app(tmp_path, None)
    assert fn.volumes == {}
    assert modal_backend._outputs_are_volume_backed(fn.volumes) is False


def test_a_volume_mounted_elsewhere_does_not_divert_the_outputs(tmp_path):
    """Durable scratch at /data is not the same claim as durable *outputs*.
    Only a mount at the outputs directory takes them off the return path."""
    _, fn = _volume_app(tmp_path, {"/data": "datasets"})
    assert modal_backend._outputs_are_volume_backed(fn.volumes) is False


def test_the_volume_is_downloaded_after_the_run(tmp_path):
    out = tmp_path / "out"
    with mock.patch.object(
        modal_backend.subprocess, "run", return_value=mock.Mock(returncode=0, stderr="")
    ) as run:
        modal_backend._download_volume("training-outputs", out)
    assert run.call_args.args[0][:3] == ["modal", "volume", "get"]
    assert "training-outputs" in run.call_args.args[0]
    assert str(out) in run.call_args.args[0]


def test_a_failed_download_says_the_outputs_are_still_in_the_volume(tmp_path):
    """The run itself succeeded and the results are durable — the one thing a
    user must not conclude here is that their outputs are gone."""
    with mock.patch.object(
        modal_backend.subprocess,
        "run",
        return_value=mock.Mock(returncode=1, stderr="network unreachable"),
    ):
        with pytest.raises(RuntimeError, match="still in the volume"):
            modal_backend._download_volume("training-outputs", tmp_path / "out")


@pytest.mark.parametrize("backend, kwargs", [("local", {}), ("brev", {"instance": "box"})])
def test_a_backend_that_cannot_mount_refuses_rather_than_dropping_it(tmp_path, backend, kwargs):
    """A silently ignored volume would run the job and write its outputs to
    disk that disappears with the box — the failure arriving hours later, as
    missing results."""
    app, _ = _volume_app(tmp_path, {"/out": "training-outputs"})
    with pytest.raises(ValueError, match="cannot mount a volume"):
        app.bind(backend, **kwargs)


def test_a_live_volume_object_is_refused_with_the_reason(tmp_path):
    """The old README told people to pass `modal.Volume.from_name(...)`. That
    object cannot reach the generated entrypoint, so the error has to say so
    rather than surfacing as a generic type complaint."""
    with pytest.raises(ValueError, match="cannot reach it"):
        _volume_app(tmp_path, {"/out": object()})


def test_the_readme_volume_example_is_executable_as_written():
    """Not a paraphrase of the README — the README.

    #143 existed because the documented example had never been run against the
    code. Extracting and executing it means the docs cannot drift back into
    describing an API that does not exist.
    """
    from pathlib import Path as _Path

    from runplz import App, Image

    readme = (_Path(__file__).resolve().parents[1] / "README.md").read_text()
    section = readme.split("### Large / persistent outputs on Modal")[1]
    snippet = section.split("```python")[1].split("```")[0]
    assert "volumes=" in snippet, "the README stopped documenting a volume mount"

    namespace = {"app": App("demo"), "image": Image.from_registry("ubuntu:22.04")}
    # The body is illustrative torch; only the decorator is under test.
    body_free = snippet.replace("import torch", "pass").replace("model = ...", "model = None")
    body_free = body_free.replace('torch.save(model.state_dict(), "/out/weights.pt")', "pass")
    exec(compile(body_free, "<README>", "exec"), namespace)
    assert namespace["app"].functions["train"].volumes == {"/out": "training-outputs"}
