"""Modal backend.

Generates a per-invocation Python file with a module-scope `modal.App`
+ `@app.function` + `@app.local_entrypoint`, then runs
`modal run <generated>.py::main`. Module-scope decorators avoid the
Python-version-matching requirement that `serialized=True` carries.

Two image shapes supported:
- `Image.from_dockerfile(path, context=...)`: rendered as
  `modal.Image.from_dockerfile(path, context_dir=context)`.
- `Image.from_registry(ref).apt_install(...).pip_install(...)
  .pip_install_local_dir(".")`: rendered as a `modal.Image` op chain.
  All image build layers run on Modal's build cluster and are cached.

Outputs: the remote function tars `/out` and returns the bytes; the
local entrypoint writes them to a file we extract to the host.

TODO: Modal function return values are capped at ~256 MB. The tar-
return pattern works for smoketests, single-model training (~10 MB
weights + ~50 MB init info), and most pan-allele single runs, but
will fail on full 4-fold × 4-replicate ensemble runs (>1 GB of
weights). Switch to `modal.Volume.from_name(..., create_if_missing=
True)` mounted at /out, then download after the run via
`volume.batch_iter(...)` before flipping this on for heavy training.
"""

import io
import json
import os
import re
import subprocess
import tarfile
import tempfile
from pathlib import Path

from runplz._excludes import DEFAULT_TRANSFER_EXCLUDES

_ENTRYPOINT_TEMPLATE = '''\
"""Generated Modal entrypoint for runplz. Do not edit."""

import io
import os
import subprocess
import tarfile

import modal


_APP_NAME = {app_name!r}
_GPU = {gpu!r}
_CPU = {cpu!r}
_MEMORY = {memory!r}
_TIMEOUT = {timeout!r}
_OUT_BLOB = {out_blob!r}
_CONTAINER_ENV = {container_env!r}


{image_construction}


app = modal.App(_APP_NAME)


# Modal accepts None for cpu/memory — picks a default. Exact GPU string
# passed through.
@app.function(image=image, gpu=_GPU, cpu=_CPU, memory=_MEMORY, timeout=_TIMEOUT)
def runner() -> bytes:
    os.makedirs("/out", exist_ok=True)
    subprocess.run(
        ["python", "-m", "runplz._bootstrap"],
        check=True,
    )
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add("/out", arcname=".")
    return buf.getvalue()


@app.local_entrypoint()
def main():
    blob = runner.remote()
    with open(_OUT_BLOB, "wb") as f:
        f.write(blob)
    print(f"[runner] wrote {{len(blob)}} bytes to {{_OUT_BLOB}}")
'''


def list_jobs() -> list[dict]:
    """Return Modal apps created by runplz.

    Filters on the ``runplz-`` prefix that the entrypoint generator stamps on
    every app name (see :data:`_ENTRYPOINT_TEMPLATE`). Runs dispatched before
    the prefix was introduced won't show up — those eventually finish on their
    own.

    Uses ``modal app list --json`` when available and falls back to text
    parsing. The json flag has been supported for a while but this keeps the
    command usable against older Modal CLIs.
    """
    try:
        import modal  # noqa: F401
    except ImportError:
        return []

    r = subprocess.run(
        ["modal", "app", "list", "--json"],
        capture_output=True,
        text=True,
    )
    if r.returncode == 0 and r.stdout.strip().startswith(("[", "{")):
        return _jobs_from_modal_json(r.stdout)

    # Fallback: parse the plain-text table.
    r = subprocess.run(
        ["modal", "app", "list"],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(
            f"`modal app list` failed (rc={r.returncode}). stderr: {(r.stderr or '').strip()[:300]}"
        )
    return _jobs_from_modal_text(r.stdout)


def _jobs_from_modal_json(stdout: str) -> list[dict]:
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        return []
    rows = (
        data if isinstance(data, list) else data.get("apps", []) if isinstance(data, dict) else []
    )
    jobs = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = row.get("name") or row.get("App Name") or row.get("Name") or ""
        state = row.get("state") or row.get("State") or ""
        if not name.startswith("runplz-"):
            continue
        # Skip terminal states — we want running/active apps.
        if str(state).lower() in {"stopped", "finished", "terminated", "deleted"}:
            continue
        app_name, fn_name = _split_modal_app_name(name)
        jobs.append(
            {
                "backend": "modal",
                "name": name,
                "app": app_name,
                "function": fn_name,
                "started": row.get("created_at") or row.get("Created at") or "",
                "status": str(state),
            }
        )
    return jobs


def _jobs_from_modal_text(stdout: str) -> list[dict]:
    """Parse the plain-text `modal app list` table.

    Format is column-delimited; we look for lines containing ``runplz-`` and
    split on runs of whitespace. Best-effort — used only when ``--json`` isn't
    available. Columns we look for (order varies across Modal CLI versions):
    App ID, Name, State, Created at.
    """
    jobs = []
    for line in stdout.splitlines():
        if "runplz-" not in line:
            continue
        parts = [p.strip() for p in re.split(r"\s{2,}|\|", line.strip()) if p.strip()]
        name = next((p for p in parts if p.startswith("runplz-")), "")
        if not name:
            continue
        app_name, fn_name = _split_modal_app_name(name)
        # Pick the token that looks like a known state; leave blank otherwise.
        state_tokens = {"running", "ready", "stopped", "finished", "terminated", "deploying"}
        state = next((p for p in parts if p.lower() in state_tokens), "")
        if state.lower() in {"stopped", "finished", "terminated"}:
            continue
        jobs.append(
            {
                "backend": "modal",
                "name": name,
                "app": app_name,
                "function": fn_name,
                "started": "",
                "status": state,
            }
        )
    return jobs


def _split_modal_app_name(name: str) -> tuple[str, str]:
    """Reverse of ``runplz-{app}-{function}``. Takes the final hyphen segment
    as the function name and everything before it as the app name."""
    if not name.startswith("runplz-"):
        return ("", "")
    core = name[len("runplz-") :]
    parts = core.split("-")
    if len(parts) < 2:
        return ("", "")
    return ("-".join(parts[:-1]), parts[-1])


def run(app, function, args, kwargs, *, outputs_dir: str = "out"):
    try:
        import modal  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Modal backend requires `pip install modal` and `modal setup` (run once)."
        ) from exc

    repo = app._repo_root
    if repo is None:
        raise RuntimeError("App repo_root not set (CLI should have set this).")

    host_out = (repo / outputs_dir).resolve()
    host_out.mkdir(parents=True, exist_ok=True)

    rel_script = Path(function.module_file).resolve().relative_to(repo)
    container_env = {
        "RUNPLZ_OUT": "/out",
        "RUNPLZ_SCRIPT": f"/workspace/{rel_script}",
        "RUNPLZ_FUNCTION": function.name,
        "RUNPLZ_ARGS": json.dumps(args),
        "RUNPLZ_KWARGS": json.dumps(kwargs),
        **function.env,
    }

    image_src = _render_modal_image(function.image, repo=repo)
    image_src += "\nimage = image.env(_CONTAINER_ENV)"

    blob_path = tempfile.NamedTemporaryFile(
        suffix=".tar.gz", prefix="runplz-modal-", delete=False
    ).name

    # Modal's @app.function accepts memory in MB; our API uses GB. Convert.
    modal_memory = int(function.min_memory * 1024) if function.min_memory is not None else None

    # Modal expresses GPU memory by baking a suffix into the gpu string
    # (e.g. "A100-80GB"). Translate our min_gpu_memory constraint onto the
    # gpu string when set; leave the gpu string alone if it already carries
    # a size suffix.
    modal_gpu = _modal_gpu_string(
        function.gpu, function.min_gpu_memory, getattr(function, "num_gpus", 1) or 1
    )

    if function.min_disk is not None:
        # Issue #20: the previous `print()` let users believe their disk
        # request was honored. Modal has no per-function disk knob — if
        # you need durable or large storage, use a Modal Volume mount
        # from inside your function. Fail loud rather than silently drop.
        raise ValueError(
            f"min_disk={function.min_disk!r} is not supported on the Modal backend. "
            f"Modal manages container storage; there is no per-function disk-size kwarg. "
            f"Options: (a) remove min_disk for this function on Modal, "
            f"(b) mount a Modal Volume inside the function for large/durable storage, "
            f"or (c) run this function on brev/local where min_disk maps to real capacity."
        )

    entrypoint_src = _ENTRYPOINT_TEMPLATE.format(
        app_name=f"runplz-{app.name}-{function.name}",
        gpu=modal_gpu,
        cpu=function.min_cpu,
        memory=modal_memory,
        timeout=function.timeout,
        out_blob=blob_path,
        container_env=container_env,
        image_construction=image_src,
    )

    entry_file = tempfile.NamedTemporaryFile(
        suffix="_modal_entry.py", prefix="runplz-", delete=False, mode="w"
    )
    entry_file.write(entrypoint_src)
    entry_file.close()

    print(f"+ modal run {entry_file.name}::main", flush=True)
    try:
        subprocess.run(
            ["modal", "run", f"{entry_file.name}::main"],
            check=True,
        )
    finally:
        try:
            os.unlink(entry_file.name)
        except OSError:
            pass

    _check_output_blob_size(blob_path)
    _extract_tar(blob_path, host_out)
    try:
        os.unlink(blob_path)
    except OSError:
        pass
    print(f"Modal run complete. Outputs in {host_out}", flush=True)


# Modal function return values are capped at ~256 MB. We return outputs as
# a tar.gz blob, so large /out directories can silently overflow (issue #19).
# These thresholds apply to the compressed tar, not the raw /out tree.
_MODAL_OUTPUT_WARN_BYTES = 200 * 1024 * 1024
_MODAL_OUTPUT_LIMIT_BYTES = 256 * 1024 * 1024


def _check_output_blob_size(blob_path: str) -> None:
    """Warn or raise when the returned output tar is near / over Modal's
    ~256 MB return-value cap. At this point the blob has already been
    written locally — if it landed truncated we want to surface that
    instead of silently unpacking a broken archive (issue #19)."""
    try:
        size = os.path.getsize(blob_path)
    except OSError:
        return
    if size >= _MODAL_OUTPUT_LIMIT_BYTES:
        raise RuntimeError(
            f"Modal output tar is {size / 1024 / 1024:.1f} MB, at or above "
            f"Modal's ~{_MODAL_OUTPUT_LIMIT_BYTES // 1024 // 1024} MB "
            f"return-value cap. The archive may be truncated and extracting "
            f"it would silently lose data.\n"
            f"Fix: write large outputs to a Modal Volume instead of /out, "
            f"e.g. `modal.Volume.from_name('my-vol', create_if_missing=True)` "
            f"mounted at /out, then download after the run with "
            f"`volume.batch_iter(...)`. See Modal docs on Volumes."
        )
    if size >= _MODAL_OUTPUT_WARN_BYTES:
        print(
            f"+ warning: Modal output tar is {size / 1024 / 1024:.1f} MB, "
            f"approaching Modal's ~{_MODAL_OUTPUT_LIMIT_BYTES // 1024 // 1024} "
            f"MB return-value cap. Future runs with larger outputs will fail. "
            f"Consider writing large artifacts to a Modal Volume instead.",
            flush=True,
        )


_MODAL_GPU_VRAM_RE = re.compile(r"-\d+GB\b", re.IGNORECASE)
_MODAL_GPU_COUNT_RE = re.compile(r":\d+$")


def _modal_gpu_string(gpu, min_gpu_memory, num_gpus: int = 1):
    """Translate (gpu, min_gpu_memory, num_gpus) into Modal's gpu string.

    Modal's gpu string encodes VRAM as a "-NGB" suffix ("A100-80GB") and
    count as a ":N" suffix ("A100-80GB:4"). We build the string VRAM-first
    then count-last; if the user already pinned either suffix we leave it
    alone (explicit wins over our augmentation).
    """
    if gpu is None:
        return None
    result = gpu
    # VRAM check scans anywhere in the string so "A100-80GB:2" is recognized
    # as already-sized even though the ":2" count suffix sits at the end.
    if min_gpu_memory is not None and not _MODAL_GPU_VRAM_RE.search(result):
        result = f"{result}-{int(min_gpu_memory)}GB"
    if num_gpus > 1 and not _MODAL_GPU_COUNT_RE.search(result):
        result = f"{result}:{int(num_gpus)}"
    return result


def _render_modal_image(image, *, repo: Path) -> str:
    """Render an `image = ...` assignment using Modal's Image DSL.

    Maps our Image layer ops 1:1 onto modal.Image methods.
    """
    if image.dockerfile is not None:
        df, ctx = image.resolve(repo)
        return f"image = modal.Image.from_dockerfile({str(df)!r}, context_dir={str(ctx)!r})"
    if image.base is None:
        raise ValueError("Image has neither base nor dockerfile")
    lines = [f"image = modal.Image.from_registry({image.base!r})"]
    for op in image.ops:
        kw = op.kwargs_dict()
        if op.kind == "apt_install" and op.args:
            args = ", ".join(repr(a) for a in op.args)
            lines.append(f"image = image.apt_install({args})")
        elif op.kind == "pip_install" and op.args:
            args = ", ".join(repr(a) for a in op.args)
            extra = f", index_url={kw['index_url']!r}" if "index_url" in kw else ""
            lines.append(f"image = image.pip_install({args}{extra})")
        elif op.kind == "pip_install_local_dir":
            path = kw.get("path", ".")
            editable = kw.get("editable", "1") == "1"
            local_dir = (repo / path).resolve()
            flags = "-e " if editable else ""
            # Plumb the shared secret-exclude list into Modal's add_local_dir
            # so `.env` / ssh keys / credentials.json don't get baked into
            # an image layer and uploaded to Modal. See runplz/_excludes.py.
            ignore_list = list(DEFAULT_TRANSFER_EXCLUDES)
            lines.append(
                f"image = image.add_local_dir({str(local_dir)!r}, "
                f"remote_path='/workspace', copy=True, "
                f"ignore={ignore_list!r}).run_commands("
                f"'pip install {flags}/workspace')"
            )
        elif op.kind == "run" and op.args:
            args = ", ".join(repr(a) for a in op.args)
            lines.append(f"image = image.run_commands({args})")
    return "\n".join(lines)


def _extract_tar(blob_path: str, dest: Path):
    with open(blob_path, "rb") as f:
        blob = f.read()
    buf = io.BytesIO(blob)
    with tarfile.open(fileobj=buf, mode="r:gz") as tar:
        members = _validated_tar_members(tar, dest)
        try:
            tar.extractall(dest, members=members, filter="data")
        except TypeError:
            tar.extractall(dest, members=members)


def _validated_tar_members(tar: tarfile.TarFile, dest: Path) -> list[tarfile.TarInfo]:
    """Reject tar members that would escape `dest` or create links.

    The Modal backend round-trips `/out` through a returned tarball, so we
    treat that archive as untrusted input. Extracting `../x` or symlink
    members verbatim would let a malformed archive write outside the outputs
    dir. Python 3.14 also tightens tar extraction defaults in this direction.
    """
    base = dest.resolve()
    safe_members = []
    for member in tar.getmembers():
        target = (base / member.name).resolve()
        try:
            target.relative_to(base)
        except ValueError as exc:
            raise RuntimeError(
                f"Refusing to extract unsafe tar member {member.name!r} outside {dest}."
            ) from exc
        if member.issym() or member.islnk():
            raise RuntimeError(f"Refusing to extract tar link member {member.name!r}.")
        safe_members.append(member)
    return safe_members
