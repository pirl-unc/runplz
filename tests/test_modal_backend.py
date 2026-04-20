"""Modal backend coverage — test image rendering, env wiring,
entrypoint-template generation, and subprocess invocation without
actually shelling out to `modal` or hitting Modal's servers.
"""

import io
import sys
import tarfile
import types
from pathlib import Path
from unittest import mock

import pytest

from runplz import App, Image
from runplz.backends import modal as modal_backend

# --- _modal_gpu_string ----------------------------------------------------


def test_modal_gpu_string_passthrough_when_no_min_vram():
    assert modal_backend._modal_gpu_string("A100", None) == "A100"
    assert modal_backend._modal_gpu_string(None, None) is None
    assert modal_backend._modal_gpu_string(None, 80) is None


def test_modal_gpu_string_appends_suffix():
    assert modal_backend._modal_gpu_string("A100", 80) == "A100-80GB"
    assert modal_backend._modal_gpu_string("H100", 40) == "H100-40GB"
    assert modal_backend._modal_gpu_string("T4", 16) == "T4-16GB"


def test_modal_gpu_string_respects_existing_suffix():
    # User pinned a size already — don't double-suffix.
    assert modal_backend._modal_gpu_string("A100-80GB", 40) == "A100-80GB"
    assert modal_backend._modal_gpu_string("L4-24gb", 16) == "L4-24gb"


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
    from runplz._excludes import DEFAULT_TRANSFER_EXCLUDES

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
    modal_backend._extract_tar(str(blob_path), dest)
    assert (dest / "a" / "b.txt").read_text() == "hello from modal\n"


# --- run() end-to-end (mocked) -------------------------------------------


def _app_with_job(tmp_path):
    app = App("pan-allele")
    app._repo_root = tmp_path
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
    app._repo_root = None
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
    app._repo_root = tmp_path
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
    app._repo_root = tmp_path
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
    app._repo_root = tmp_path
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
