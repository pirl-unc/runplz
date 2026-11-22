"""Google Colab backend.

Shells out to the official `colab` CLI (`google-colab-cli`) to provision an
ephemeral Google Colab runtime (CPU, GPU: T4/L4/G4/A100/H100, or TPU:
v5e1/v6e1), upload the repository archive, apply declared `Image` layer ops
inline, invoke the runplz bootstrap module, download `/content/out` (including
partial outputs on failure), and release the VM on exit.

Colab sessions are Google-managed Linux VMs without nested Docker, so
`Image.from_dockerfile(...)` is rejected upfront and `Image.from_registry(...)`
ops (`apt_install`, `pip_install`, `pip_install_local_dir`, `run_commands`)
execute directly on the VM.

Authentication is owned by `google-colab-cli` (`--auth=oauth2` or
`--auth=adc`). runplz never stores or touches Google credentials.
"""

__all__ = [
    "run",
    "list_jobs",
    "gpu_flag_for",
    "SESSION_PREFIX",
    "REMOTE_WORKSPACE",
    "REMOTE_OUT",
]

import fnmatch
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from runplz.backends.listing import JobRecord
from runplz.excludes import DEFAULT_TRANSFER_EXCLUDES

SESSION_PREFIX = "runplz-"
REMOTE_WORKSPACE = "/content/workspace"
REMOTE_OUT = "/content/out"
_REMOTE_REPO_TAR = "/content/runplz-repo.tar.gz"
_REMOTE_OUT_TAR = "/content/runplz-out.tar.gz"
_REMOTE_RESULT_JSON = "/content/runplz-result.json"

# Colab accepts a single GPU per session from this exact set. Unrecognized
# names silently fall back to A100 inside colab-cli, so we validate upfront.
_COLAB_GPU_ALIASES = {
    "t4": "T4",
    "l4": "L4",
    "g4": "G4",
    "a100": "A100",
    "a100-40gb": "A100",
    "a100-80gb": "A100",
    "h100": "H100",
    "h100!": "H100",
    "h100-80gb": "H100",
}

_COLAB_VRAM_LADDER = (
    (16, "T4"),
    (24, "L4"),
    (40, "A100"),
    (80, "H100"),
)

# High-RAM-only accelerators emit a warning if --high-mem is passed; keep the
# flag clean when auto-enabling high_mem from min_memory.
_HIGH_MEM_ONLY_ACCELERATORS = frozenset({"L4"})

_SESSION_LINE_RE = re.compile(
    r"^\[(?P<name>[^\]]+)\]\s+(?P<endpoint>\S+)\s+\|\s+"
    r"Hardware:\s+(?P<hardware>[^|]+?)\s+\|\s+"
    r"Shape:\s+(?P<shape>[^|]+?)\s+\|\s+"
    r"Variant:\s+(?P<variant>[^|]+?)(?:\s+\|\s+Status:\s+(?P<status>.+))?$"
)

_RUNNER_TEMPLATE = '''\
"""Generated Google Colab runner for runplz. Do not edit."""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

_WORKSPACE = {workspace!r}
_OUT_DIR = {out_dir!r}
_REPO_TAR = {repo_tar!r}
_OUT_TAR = {out_tar!r}
_RESULT_JSON = {result_json!r}
_IMAGE_OPS = {image_ops!r}
_CONTAINER_ENV = {container_env!r}
_PRECONDITIONS = {preconditions!r}
_TIMEOUT = {timeout!r}


def _safe_extract(tar_path: str, dest: str) -> None:
    os.makedirs(dest, exist_ok=True)
    base = Path(dest).resolve()
    with tarfile.open(tar_path, "r:gz") as tar:
        members = []
        for member in tar.getmembers():
            target = (base / member.name).resolve()
            target.relative_to(base)
            if member.issym() or member.islnk():
                raise RuntimeError(f"Refusing link member {{member.name!r}}")
            members.append(member)
        try:
            tar.extractall(dest, members=members, filter="data")
        except TypeError:
            tar.extractall(dest, members=members)


def _stream(cmd, *, shell: bool = False, cwd=None, env=None, timeout=None) -> int:
    # Jupyter kernels only forward writes to Python's sys.stdout/sys.stderr over
    # WebSockets; child OS fds would otherwise vanish. Stream line-by-line so
    # pip/apt/bootstrap logs appear live in the caller's terminal.
    proc = subprocess.Popen(
        cmd,
        shell=shell,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
        return proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        raise


def _stream_check(cmd, *, shell: bool = False, cwd=None, env=None) -> None:
    rc = _stream(cmd, shell=shell, cwd=cwd, env=env)
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def _apply_image_ops() -> None:
    for op in _IMAGE_OPS:
        kind = op["kind"]
        args = op["args"]
        kw = op["kwargs"]
        if kind == "apt_install" and args:
            _stream_check(["apt-get", "update", "-qq"])
            _stream_check(
                ["apt-get", "install", "-y", "-qq", "--no-install-recommends", *args],
            )
        elif kind == "pip_install" and args:
            cmd = [sys.executable, "-m", "pip", "install", "-q"]
            if "index_url" in kw:
                cmd += ["--index-url", kw["index_url"]]
            cmd += list(args)
            _stream_check(cmd)
        elif kind == "pip_install_local_dir":
            rel = kw.get("path", ".")
            editable = kw.get("editable", "1") == "1"
            target = str((Path(_WORKSPACE) / rel).resolve())
            cmd = [sys.executable, "-m", "pip", "install", "-q"]
            if editable:
                cmd.append("-e")
            cmd.append(target)
            _stream_check(cmd)
        elif kind == "run" and args:
            for shell_cmd in args:
                _stream_check(shell_cmd, shell=True, cwd=_WORKSPACE)
    if importlib.util.find_spec("runplz") is None:
        _stream_check([sys.executable, "-m", "pip", "install", "-q", "runplz"])


def _check_preconditions() -> None:
    if not _PRECONDITIONS:
        return
    if "disk_free_gb" in _PRECONDITIONS:
        need = _PRECONDITIONS["disk_free_gb"]
        free_gb = shutil.disk_usage("/content").free / (1024 ** 3)
        if free_gb < need:
            raise RuntimeError(f"Precondition disk_free_gb failed: {{free_gb:.1f}} < {{need}}")
    if "shm_gb" in _PRECONDITIONS and os.path.exists("/dev/shm"):
        shm_gb = shutil.disk_usage("/dev/shm").total / (1024 ** 3)
        if shm_gb < _PRECONDITIONS["shm_gb"]:
            raise RuntimeError(
                f"Precondition shm_gb failed: {{shm_gb:.1f}} < {{_PRECONDITIONS['shm_gb']}}"
            )
    if "gpu_count" in _PRECONDITIONS or "gpu_memory_gb" in _PRECONDITIONS:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        )
        mems = [float(line.strip()) / 1024.0 for line in r.stdout.splitlines() if line.strip()]
        if "gpu_count" in _PRECONDITIONS and len(mems) < _PRECONDITIONS["gpu_count"]:
            raise RuntimeError(
                f"Precondition gpu_count failed: {{len(mems)}} < {{_PRECONDITIONS['gpu_count']}}"
            )
        if "gpu_memory_gb" in _PRECONDITIONS and (
            not mems or min(mems) < _PRECONDITIONS["gpu_memory_gb"]
        ):
            got = min(mems) if mems else 0.0
            need = _PRECONDITIONS["gpu_memory_gb"]
            raise RuntimeError(f"Precondition gpu_memory_gb failed: {{got:.1f}} < {{need}}")


def _main() -> None:
    print("[runplz] colab runner started", flush=True)
    exit_code = 1
    error_msg = ""
    os.makedirs(_OUT_DIR, exist_ok=True)
    try:
        if os.path.exists(_WORKSPACE):
            shutil.rmtree(_WORKSPACE)
        _safe_extract(_REPO_TAR, _WORKSPACE)
        _apply_image_ops()
        _check_preconditions()
        env = os.environ.copy()
        env.update(_CONTAINER_ENV)
        env["PYTHONUNBUFFERED"] = "1"
        exit_code = _stream(
            [sys.executable, "-m", "runplz._bootstrap"],
            cwd=_WORKSPACE,
            env=env,
            timeout=_TIMEOUT,
        )
        if exit_code != 0:
            error_msg = f"bootstrap exited with code {{exit_code}}"
    except subprocess.TimeoutExpired:
        exit_code = 124
        error_msg = f"job timed out after {{_TIMEOUT}}s"
        print(f"[runplz] {{error_msg}}", file=sys.stderr, flush=True)
    except BaseException as exc:
        exit_code = 1
        error_msg = f"{{type(exc).__name__}}: {{exc}}"
        print(f"[runplz] {{error_msg}}", file=sys.stderr, flush=True)
    finally:
        with open(_RESULT_JSON, "w") as f:
            json.dump({{"exit_code": exit_code, "error": error_msg}}, f)
        with tarfile.open(_OUT_TAR, "w:gz") as tar:
            tar.add(_OUT_DIR, arcname=".")


_main()
'''


def gpu_flag_for(
    gpu: Optional[str],
    min_gpu_memory: Optional[float],
    num_gpus: int = 1,
) -> Optional[str]:
    """Translate (gpu, min_gpu_memory, num_gpus) into a `colab --gpu` flag.

    Returns None when neither `gpu` nor `min_gpu_memory` is set (CPU runtime).
    Raises ValueError on multi-GPU requests or unsupported GPU models so users
    never hit colab-cli's silent fallback to A100.
    """
    if num_gpus > 1:
        raise ValueError(
            f"Google Colab supports at most 1 GPU per session (got num_gpus={num_gpus}). "
            "Use modal, brev, gcp, or aws for multi-GPU jobs."
        )
    if gpu is None and min_gpu_memory is None:
        return None
    if gpu is not None:
        key = gpu.strip().lower()
        mapped = _COLAB_GPU_ALIASES.get(key)
        if mapped is None:
            supported = ", ".join(sorted(set(_COLAB_GPU_ALIASES.values())))
            raise ValueError(
                f"GPU {gpu!r} is not supported on Google Colab. Supported Colab GPUs: {supported}."
            )
        return mapped
    for vram_gb, model in _COLAB_VRAM_LADDER:
        if min_gpu_memory <= vram_gb:
            return model
    raise ValueError(
        f"min_gpu_memory={min_gpu_memory!r} GB exceeds the largest Google Colab GPU "
        f"({_COLAB_VRAM_LADDER[-1][1]} = {_COLAB_VRAM_LADDER[-1][0]} GB)."
    )


def _colab_base_cmd(
    *,
    auth: str = "oauth2",
    session_config_path: Optional[str] = None,
    client_oauth_config: Optional[str] = None,
) -> list[str]:
    cmd = ["colab", f"--auth={auth}"]
    if session_config_path:
        cmd += ["--config", str(session_config_path)]
    if client_oauth_config:
        cmd += ["--client-oauth-config", str(client_oauth_config)]
    return cmd


def _ensure_colab_cli() -> None:
    if shutil.which("colab") is None:
        raise RuntimeError(
            "Google Colab backend requires `google-colab-cli` on PATH. "
            "Install with `pip install google-colab-cli` (or `uv tool install google-colab-cli`)."
        )


def _slug(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return cleaned or "job"


def _session_name(app_name: str, fn_name: str) -> str:
    suffix = uuid.uuid4().hex[:6]
    return f"{SESSION_PREFIX}{_slug(app_name)}-{_slug(fn_name)}-{suffix}"


def _split_session_name(name: str) -> tuple[str, str]:
    if not name.startswith(SESSION_PREFIX):
        return ("", "")
    core = name[len(SESSION_PREFIX) :]
    parts = core.split("-")
    if len(parts) < 3:
        return ("", "")
    return ("-".join(parts[:-2]), parts[-2])


def list_jobs(
    *,
    auth: Optional[str] = None,
    config: Optional[str] = None,
) -> list[JobRecord]:
    """Return active Google Colab sessions created by runplz."""
    _ensure_colab_cli()
    resolved_auth = auth or "oauth2"
    if resolved_auth not in ("oauth2", "adc"):
        raise ValueError(f"colab auth must be 'oauth2' or 'adc'; got {resolved_auth!r}.")
    cmd = [*_colab_base_cmd(auth=resolved_auth, session_config_path=config), "sessions"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        detail = (r.stderr or r.stdout or "").strip()[:300]
        raise RuntimeError(f"`colab sessions` failed (rc={r.returncode}): {detail}")
    return _parse_sessions_output(r.stdout)


def _parse_sessions_output(stdout: str) -> list[JobRecord]:
    jobs: list[JobRecord] = []
    for raw_line in stdout.splitlines():
        match = _SESSION_LINE_RE.match(raw_line.strip())
        if not match:
            continue
        name = match.group("name").strip()
        if not name.startswith(SESSION_PREFIX):
            continue
        app_name, fn_name = _split_session_name(name)
        hardware = match.group("hardware").strip()
        shape = match.group("shape").strip()
        status = (match.group("status") or "").strip()
        state = f"{hardware}/{shape}" + (f" ({status})" if status else "")
        jobs.append(
            JobRecord(
                backend="colab",
                name=name,
                app=app_name,
                function=fn_name,
                started="",
                status=state,
            )
        )
    return jobs


_NOISE_EXCLUDES = (
    ".git",
    ".venv",
    "__pycache__",
    "*.egg-info",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "build",
    "dist",
    "out",
)


def _is_excluded(rel_posix: str, name: str, extra_excludes: tuple[str, ...] = ()) -> bool:
    parts = Path(rel_posix).parts
    prefixes = tuple("/".join(parts[:i]) for i in range(1, len(parts) + 1))
    for pattern in (*_NOISE_EXCLUDES, *DEFAULT_TRANSFER_EXCLUDES, *extra_excludes):
        pat = pattern.rstrip("/")
        if (
            fnmatch.fnmatch(name, pat)
            or any(fnmatch.fnmatch(part, pat) for part in parts)
            or any(fnmatch.fnmatch(prefix, pat) for prefix in prefixes)
        ):
            return True
    return False


def _iter_repo_files(repo: Path) -> list[Path]:
    """List repo files honoring `.gitignore` when inside a git worktree."""
    if (repo / ".git").exists():
        r = subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "ls-files",
                "-z",
                "--cached",
                "--others",
                "--exclude-standard",
            ],
            capture_output=True,
        )
        if r.returncode == 0:
            files = []
            for raw in r.stdout.decode("utf-8", errors="ignore").split("\0"):
                if not raw:
                    continue
                fpath = repo / raw
                if fpath.is_file() and not fpath.is_symlink():
                    files.append(Path(raw))
            return sorted(files)
    collected: list[Path] = []
    for root, dirs, files in os.walk(repo):
        rel_root = Path(root).relative_to(repo)
        dirs[:] = sorted(
            d
            for d in dirs
            if not _is_excluded((rel_root / d).as_posix(), d) and not (Path(root) / d).is_symlink()
        )
        for fname in sorted(files):
            fpath = Path(root) / fname
            if not fpath.is_symlink():
                collected.append(rel_root / fname)
    return collected


def _pack_repo(
    repo: Path,
    tar_path: Path,
    *,
    extra_excludes: tuple[str, ...] = (),
    always_include: Optional[Path] = None,
) -> None:
    included: set[str] = set()
    with tarfile.open(tar_path, "w:gz") as tar:
        for rel in _iter_repo_files(repo):
            rel_posix = rel.as_posix()
            if _is_excluded(rel_posix, rel.name, extra_excludes):
                continue
            tar.add(repo / rel, arcname=rel_posix)
            included.add(rel_posix)
        if always_include is not None:
            rel_posix = always_include.as_posix()
            if rel_posix not in included and (repo / always_include).is_file():
                tar.add(repo / always_include, arcname=rel_posix)


def _extract_tar(tar_path: Path, dest: Path) -> None:
    blob = tar_path.read_bytes()
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        members = _validated_tar_members(tar, dest)
        try:
            tar.extractall(dest, members=members, filter="data")
        except TypeError:
            tar.extractall(dest, members=members)


def _validated_tar_members(tar: tarfile.TarFile, dest: Path) -> list[tarfile.TarInfo]:
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


_TRANSIENT_ATTEMPTS = 3
_TRANSIENT_BACKOFF_S = 2.0
_TRANSFER_ATTEMPTS = 6
_TRANSFER_BACKOFF_S = 5.0
_RUNNER_STARTED_SENTINEL = "[runplz] colab runner started"
_COLAB_HANG_PATTERNS = ("RuntimeError: Connection was lost.",)
_COLAB_HANG_GRACE_S = 5.0


def _exec_colab_once(
    cmd: list[str],
    *,
    cmd_timeout: float,
    hang_grace_s: float = _COLAB_HANG_GRACE_S,
) -> tuple[int, bool]:
    """Run `colab exec`, streaming output live and reaping wedged CLI children.

    `google-colab-cli` 0.8.0 can print `RuntimeError: Connection was lost.` and
    fail to exit because a non-daemon WebSocket thread stays alive after an
    exception. Watch the combined output stream so a disconnected `colab exec`
    is killed after `hang_grace_s` instead of holding a billable GPU VM open.

    Returns `(returncode, runner_started)`.
    """
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    state = {"started": False, "hang_seen_at": None}

    def _pump() -> None:
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if _RUNNER_STARTED_SENTINEL in line:
                state["started"] = True
            if state["hang_seen_at"] is None and any(pat in line for pat in _COLAB_HANG_PATTERNS):
                state["hang_seen_at"] = time.monotonic()

    reader = threading.Thread(target=_pump, daemon=True)
    reader.start()
    deadline = time.monotonic() + cmd_timeout
    rc: Optional[int] = None
    while True:
        rc = proc.poll()
        if rc is not None:
            break
        now = time.monotonic()
        if now >= deadline:
            print(
                f"[runplz] colab exec exceeded hard timeout ({cmd_timeout:.0f}s); terminating CLI",
                file=sys.stderr,
                flush=True,
            )
            proc.kill()
            proc.wait()
            rc = 124
            break
        if state["hang_seen_at"] is not None and (now - state["hang_seen_at"]) >= hang_grace_s:
            print(
                "[runplz] colab exec hung after WebSocket disconnect; reaping CLI process",
                file=sys.stderr,
                flush=True,
            )
            proc.kill()
            proc.wait()
            rc = 1
            break
        time.sleep(0.1)
    reader.join(timeout=2.0)
    return rc, bool(state["started"])


def _run_colab_exec(
    cmd: list[str],
    *,
    dry_run: bool,
    cmd_timeout: float,
    retries: int = _TRANSIENT_ATTEMPTS,
    hang_grace_s: float = _COLAB_HANG_GRACE_S,
) -> subprocess.CompletedProcess:
    print("+ " + " ".join(shlex.quote(part) for part in cmd), flush=True)
    if dry_run:
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
    last_rc = 1
    for attempt in range(1, max(1, retries) + 1):
        rc, started = _exec_colab_once(cmd, cmd_timeout=cmd_timeout, hang_grace_s=hang_grace_s)
        last_rc = rc
        if rc == 0 or started or attempt >= retries:
            break
        wait_s = _TRANSIENT_BACKOFF_S * attempt
        print(
            f"[runplz] colab exec disconnected before runner started (rc={rc}); "
            f"retrying ({attempt}/{retries - 1}) in {wait_s:.0f}s...",
            file=sys.stderr,
            flush=True,
        )
        time.sleep(wait_s)
    if last_rc != 0:
        raise subprocess.CalledProcessError(last_rc, cmd)
    return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")


def _run_colab(
    cmd: list[str],
    *,
    dry_run: bool,
    check: bool = True,
    retries: int = 1,
    cmd_timeout: Optional[float] = 300.0,
    backoff_s: float = _TRANSIENT_BACKOFF_S,
) -> subprocess.CompletedProcess:
    print("+ " + " ".join(shlex.quote(part) for part in cmd), flush=True)
    if dry_run:
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
    last_result: Optional[subprocess.CompletedProcess] = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            result = subprocess.run(cmd, check=False, timeout=cmd_timeout)
        except subprocess.TimeoutExpired:
            result = subprocess.CompletedProcess(cmd, 124, stdout="", stderr="timed out")
        last_result = result
        if result.returncode == 0 or attempt >= retries:
            break
        wait_s = backoff_s * attempt
        print(
            f"[runplz] colab command failed (rc={result.returncode}); "
            f"retrying ({attempt}/{retries - 1}) in {wait_s:.0f}s...",
            file=sys.stderr,
            flush=True,
        )
        time.sleep(wait_s)
    assert last_result is not None
    if check and last_result.returncode != 0:
        raise subprocess.CalledProcessError(last_result.returncode, cmd)
    return last_result


def _validate_function(function, cfg) -> tuple[Optional[str], Optional[str], bool]:
    if function.image.dockerfile is not None:
        raise ValueError(
            f"@app.function({function.name}): the Google Colab backend runs on Google-managed "
            "Linux VMs without nested Docker, so Image.from_dockerfile(...) is not supported. "
            "Use Image.from_registry(...) with .apt_install/.pip_install/.run_commands instead."
        )
    if function.min_disk is not None:
        raise ValueError(
            f"min_disk={function.min_disk!r} is not supported on the Google Colab backend. "
            "Colab manages VM disk capacity automatically; remove min_disk or use brev/gcp/aws."
        )
    gpu_flag = gpu_flag_for(
        function.gpu, function.min_gpu_memory, getattr(function, "num_gpus", 1) or 1
    )
    tpu_flag = cfg.tpu.strip().upper() if cfg.tpu else None
    if gpu_flag and tpu_flag:
        raise ValueError(
            f"@app.function({function.name}): cannot request both gpu={gpu_flag!r} and "
            f"ColabConfig(tpu={tpu_flag!r}) on the same Colab session."
        )
    want_high_mem = cfg.high_mem or (function.min_memory is not None and function.min_memory > 12)
    if want_high_mem and (tpu_flag is not None or gpu_flag in _HIGH_MEM_ONLY_ACCELERATORS):
        want_high_mem = False
    return gpu_flag, tpu_flag, want_high_mem


def run(app, function, args, kwargs, *, outputs_dir: str = "out"):
    """Provision a Colab session, run the function via bootstrap, and collect outputs."""
    cfg = app.colab_config
    gpu_flag, tpu_flag, want_high_mem = _validate_function(function, cfg)
    if not cfg.dry_run:
        _ensure_colab_cli()

    repo = app.require_repo_root(context="colab.run()")
    host_out = (repo / outputs_dir).resolve()
    host_out.mkdir(parents=True, exist_ok=True)

    rel_script = Path(function.module_file).resolve().relative_to(repo)
    container_env = {
        "RUNPLZ_OUT": REMOTE_OUT,
        "RUNPLZ_SCRIPT": f"{REMOTE_WORKSPACE}/{rel_script.as_posix()}",
        "RUNPLZ_FUNCTION": function.name,
        "RUNPLZ_ARGS": json.dumps(args),
        "RUNPLZ_KWARGS": json.dumps(kwargs),
        **function.env,
    }
    image_ops = [
        {"kind": op.kind, "args": list(op.args), "kwargs": op.kwargs_dict()}
        for op in function.image.ops
    ]
    timeout_s = cfg.max_runtime_seconds or function.timeout

    session = _session_name(app.name, function.name)
    base_cmd = _colab_base_cmd(
        auth=cfg.auth,
        session_config_path=cfg.session_config_path,
        client_oauth_config=cfg.client_oauth_config,
    )

    new_cmd = [*base_cmd, "new", "-s", session]
    if gpu_flag:
        new_cmd += ["--gpu", gpu_flag]
    if tpu_flag:
        new_cmd += ["--tpu", tpu_flag]
    if want_high_mem:
        new_cmd.append("--high-mem")

    runner_src = _RUNNER_TEMPLATE.format(
        workspace=REMOTE_WORKSPACE,
        out_dir=REMOTE_OUT,
        repo_tar=_REMOTE_REPO_TAR,
        out_tar=_REMOTE_OUT_TAR,
        result_json=_REMOTE_RESULT_JSON,
        image_ops=image_ops,
        container_env=container_env,
        preconditions=dict(function.preconditions),
        timeout=timeout_s,
    )

    with tempfile.TemporaryDirectory(prefix="runplz-colab-") as tmp:
        tmp_dir = Path(tmp)
        local_repo_tar = tmp_dir / "repo.tar.gz"
        local_runner = tmp_dir / "runner.py"
        local_out_tar = tmp_dir / "out.tar.gz"
        local_result = tmp_dir / "result.json"

        local_runner.write_text(runner_src)
        if not cfg.dry_run:
            out_excludes: list[str] = list(cfg.exclude)
            try:
                rel_out = host_out.relative_to(repo.resolve()).as_posix()
                if rel_out and rel_out != ".":
                    out_excludes.append(rel_out)
            except ValueError:
                pass
            _pack_repo(
                repo,
                local_repo_tar,
                extra_excludes=tuple(out_excludes),
                always_include=rel_script,
            )
            size_mb = local_repo_tar.stat().st_size / (1024 * 1024)
            print(f"+ packed repo archive: {size_mb:.2f} MB", flush=True)

        try:
            _run_colab(
                new_cmd,
                dry_run=cfg.dry_run,
                retries=_TRANSIENT_ATTEMPTS,
                cmd_timeout=300.0,
                backoff_s=5.0,
            )
        except subprocess.CalledProcessError:
            if not (want_high_mem and cfg.fallback_to_standard_mem):
                raise
            fallback_cmd = [part for part in new_cmd if part != "--high-mem"]
            print(
                "[runplz] colab --high-mem provisioning failed; retrying standard memory shape...",
                file=sys.stderr,
                flush=True,
            )
            _run_colab(
                fallback_cmd,
                dry_run=cfg.dry_run,
                retries=_TRANSIENT_ATTEMPTS,
                cmd_timeout=300.0,
                backoff_s=5.0,
            )
        try:
            _run_colab(
                [*base_cmd, "upload", "-s", session, str(local_repo_tar), _REMOTE_REPO_TAR],
                dry_run=cfg.dry_run,
                retries=_TRANSFER_ATTEMPTS,
                cmd_timeout=300.0,
                backoff_s=_TRANSFER_BACKOFF_S,
            )
            _run_colab_exec(
                [*base_cmd, "exec", "-s", session, "-f", str(local_runner), "--timeout", "0"],
                dry_run=cfg.dry_run,
                cmd_timeout=float(timeout_s) + 600.0,
                retries=_TRANSIENT_ATTEMPTS,
            )
            if cfg.dry_run:
                _run_colab(
                    [*base_cmd, "download", "-s", session, _REMOTE_RESULT_JSON, str(local_result)],
                    dry_run=True,
                )
                _run_colab(
                    [*base_cmd, "download", "-s", session, _REMOTE_OUT_TAR, str(local_out_tar)],
                    dry_run=True,
                )
                return

            # Always attempt to salvage /content/out before raising on failure.
            out_dl = _run_colab(
                [*base_cmd, "download", "-s", session, _REMOTE_OUT_TAR, str(local_out_tar)],
                dry_run=False,
                check=False,
                retries=_TRANSFER_ATTEMPTS,
                backoff_s=_TRANSFER_BACKOFF_S,
            )
            if out_dl.returncode == 0 and local_out_tar.is_file():
                _extract_tar(local_out_tar, host_out)

            res_dl = _run_colab(
                [*base_cmd, "download", "-s", session, _REMOTE_RESULT_JSON, str(local_result)],
                dry_run=False,
                check=False,
                retries=_TRANSFER_ATTEMPTS,
                backoff_s=_TRANSFER_BACKOFF_S,
            )
            if res_dl.returncode != 0 or not local_result.is_file():
                raise RuntimeError(
                    f"Colab session {session!r} finished without a result receipt "
                    f"({_REMOTE_RESULT_JSON}); partial outputs (if any) were saved to {host_out}."
                )
            receipt = json.loads(local_result.read_text())
            exit_code = int(receipt.get("exit_code", 1))
            if exit_code != 0:
                detail = receipt.get("error") or f"exit code {exit_code}"
                raise RuntimeError(
                    f"Colab job {function.name!r} failed ({detail}). "
                    f"Partial outputs (if any) were salvaged to {host_out}."
                )
        finally:
            if cfg.on_finish == "stop":
                unwinding = sys.exc_info()[0] is not None
                try:
                    _run_colab(
                        [*base_cmd, "stop", "-s", session],
                        dry_run=cfg.dry_run,
                        check=not unwinding,
                        cmd_timeout=120.0,
                    )
                except Exception:
                    if not unwinding:
                        raise
                    print(
                        f"[runplz] warning: failed to stop Colab session {session!r}",
                        file=sys.stderr,
                        flush=True,
                    )
            else:
                print(
                    f"[runplz] leaving Colab session {session!r} running (on_finish='leave'). "
                    f"Stop manually with: colab --auth={cfg.auth} stop -s {session}",
                    flush=True,
                )
    print(f"Colab run complete. Outputs in {host_out}", flush=True)
