"""SSH-layer plumbing shared by the Brev and SSH backends.

Everything in here is target-agnostic: it operates on an ssh alias /
host string and knows nothing about provisioning, billing, or CLI
lifecycle. Backend-specific concerns (`brev create/stop/delete`,
`brev search`, etc.) stay in their respective backend modules.

The parameter name `target` is used throughout — Brev calls it an
"instance," plain SSH calls it a "host," but it's the same thing:
whatever string ssh/rsync treat as a reachable endpoint.

Backends import the functions they need and call them directly.
Because Python `from … import name` binds a local reference, tests
patching `runplz.backends.brev.<name>` continue to work — brev.py
holds its own module-level reference to the imported function.
"""

import json
import shlex
import subprocess
import time
import uuid
from pathlib import Path
from typing import Callable, Optional

from runplz._excludes import DEFAULT_TRANSFER_EXCLUDES

# --- constants -----------------------------------------------------------

REMOTE_REPO_DIR = "runplz-repo"
REMOTE_OUT_DIR = "runplz-out"
REMOTE_IMAGE_TAG = "runplz-train:remote"

# container-mode / native paths tee the bootstrap's combined stdout+stderr
# into this file so we can `tail` it for failure context (issue #17). Lives
# under $HOME so no sudo needed and survives across ssh reconnects.
REMOTE_LAST_LOG = ".runplz-last.log"

# How many lines of remote log to include in a failure RuntimeError.
FAILURE_TAIL_LINES = 50

# Directories that are noise on every upload and exclusions we apply on
# top of DEFAULT_TRANSFER_EXCLUDES (which only covers secrets).
_RSYNC_NOISE_EXCLUDES = (
    ".git",
    ".venv",
    "__pycache__",
    "*.egg-info",
    "build",
    "dist",
    "out",
)

# Brev's managed ssh config sets `ControlMaster auto` (connection
# multiplexing). That's fast for short repeated calls but catastrophic
# for our workload: a long-lived `docker logs -f` ssh session holds the
# master, the underlying TCP goes stale (common on GCP N1/G2 GPU boxes),
# and every subsequent ssh call — including our `docker inspect` health
# probe — hangs for ~5 minutes waiting for the dead master to time out.
# Force a fresh TCP connection per ssh call to sidestep that entirely.
#
# ServerAliveInterval=30 + large ServerAliveCountMax keeps each
# individual session alive during idle stretches (docker image pulls,
# data downloads, between-epoch pauses).
SSH_OPTS = [
    "-o",
    "ControlMaster=no",
    "-o",
    "ControlPath=none",
    "-o",
    "ServerAliveInterval=30",
    "-o",
    "ServerAliveCountMax=240",
    "-o",
    "TCPKeepAlive=yes",
]

_NATIVE_VENV = "$HOME/runplz-venv"


# --- low-level ssh / sh / rsync ------------------------------------------


def _sh(cmd):
    print("+ " + " ".join(shlex.quote(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True)


def _ssh(target: str, remote_cmd: str):
    # Pass the whole pipeline as a SINGLE arg to ssh. If we pass
    # ["ssh", host, "bash", "-lc", cmd] instead, ssh space-joins the trailing
    # argv before sending to the remote shell, which then re-parses — turning
    # `bash -lc 'set -euo pipefail; X'` into `bash -lc set -euo pipefail; X`
    # (i.e. `set` runs with no args as the -c command, X runs in the outer
    # shell without errexit). Quoting with shlex.quote around the whole
    # command string avoids that.
    _sh(["ssh", *SSH_OPTS, target, f"bash -lc {shlex.quote(remote_cmd)}"])


def _ssh_capture(target: str, remote_cmd: str) -> str:
    r = subprocess.run(
        ["ssh", *SSH_OPTS, target, remote_cmd],
        capture_output=True,
        text=True,
        timeout=60,
    )
    return r.stdout


def _rsync_up(repo: Path, target: str):
    # Intentionally no --delete: a user who sshes in and leaves files under
    # ~/runplz-repo/ (logs, probe scripts, local edits) shouldn't have those
    # wiped by the next run. Stale files on the remote are cheap; accidental
    # user-data loss is not.
    cmd = ["rsync", "-az"]
    for pat in _RSYNC_NOISE_EXCLUDES:
        cmd.append(f"--exclude={pat}")
    # Safety: never ship local secrets / dotenv / SSH keys to a remote box.
    # See runplz/_excludes.py for the rationale.
    for pat in DEFAULT_TRANSFER_EXCLUDES:
        cmd.append(f"--exclude={pat}")
    cmd.extend([f"{repo}/", f"{target}:{REMOTE_REPO_DIR}/"])
    _sh(cmd)


def _rsync_down(target: str, local_out: Path):
    _sh(["rsync", "-az", f"{target}:{REMOTE_OUT_DIR}/", f"{local_out}/"])


# --- connectivity helpers ------------------------------------------------


def _wait_until_ssh_reachable(
    target: str,
    *,
    max_wait_s: int = 1200,
    probe_interval_s: int = 15,
    refresh_callback: Optional[Callable[[], None]] = None,
) -> None:
    """Block until an SSH session to `target` succeeds, or raise.

    Polls with short-timeout SSH probes. Every ~minute invokes
    `refresh_callback` (if provided) to let the caller repair any config
    drift — e.g. Brev passes a callback that runs `brev refresh` to pick
    up a new port when an instance transitions from the bootstrap-shim
    port to the real one. Plain SSH backends pass None.
    """
    print(
        f"+ waiting for {target} SSH to become reachable (up to {max_wait_s}s)...",
        flush=True,
    )
    deadline = time.time() + max_wait_s
    last_err = ""
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        probe = subprocess.run(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                f"ConnectTimeout={probe_interval_s}",
                *SSH_OPTS,
                target,
                "true",
            ],
            capture_output=True,
            text=True,
        )
        if probe.returncode == 0:
            print(f"+ {target} SSH ready (attempt {attempt})", flush=True)
            return
        last_err = (probe.stderr or probe.stdout or "").strip().splitlines()[-1:] or [""]
        last_err = last_err[0]
        if refresh_callback is not None and attempt % 4 == 0:
            print(
                f"+ {target} still unreachable after {attempt} probes "
                f"(last: {last_err}); running refresh callback...",
                flush=True,
            )
            try:
                refresh_callback()
            except Exception as exc:  # noqa: BLE001
                print(f"+ refresh callback raised: {exc}", flush=True)
        time.sleep(probe_interval_s)
    raise RuntimeError(
        f"SSH to {target} never became reachable within {max_wait_s}s (last error: {last_err!r})."
    )


def _ensure_remote_rsync(target: str):
    """Install rsync on the remote if missing (slim container images
    often don't ship with rsync)."""
    cmd = (
        "command -v rsync >/dev/null 2>&1 && exit 0; "
        "export DEBIAN_FRONTEND=noninteractive; "
        "sudo apt-get update -qq && "
        "sudo apt-get install -y -qq --no-install-recommends rsync"
    )
    _ssh(target, cmd)


def _ensure_docker(target: str, timeout_s: int = 420):
    """Wait for docker daemon to be reachable on the remote, installing
    docker via get.docker.com as a fallback if the daemon never appears."""
    print(f"+ waiting for docker daemon on {target} (up to {timeout_s}s)", flush=True)
    wait_script = (
        "for i in $(seq 1 60); do "
        "if command -v docker >/dev/null 2>&1 && "
        "   sudo docker info >/dev/null 2>&1; then exit 0; fi; "
        "sudo fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1 "
        "  && echo 'apt busy, waiting' || echo 'waiting for docker daemon'; "
        "sleep 7; "
        "done; exit 1"
    )
    r = subprocess.run(
        [
            "ssh",
            *SSH_OPTS,
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=accept-new",
            target,
            wait_script,
        ],
        timeout=timeout_s,
    )
    if r.returncode != 0:
        print(
            f"+ docker daemon not reachable on {target} after {timeout_s}s; "
            f"falling back to get-docker.sh",
            flush=True,
        )
        _sh(["ssh", target, "curl -fsSL https://get.docker.com | sudo sh"])


def _remote_has_nvidia(target: str) -> bool:
    # nvidia-smi is often pre-installed without a real GPU; the reliable
    # signal is /proc/driver/nvidia, which only exists when the kernel
    # module is loaded against real hardware.
    r = subprocess.run(
        ["ssh", *SSH_OPTS, target, "test -d /proc/driver/nvidia && echo y || echo n"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return r.returncode == 0 and r.stdout.strip() == "y"


# --- dispatch: container-mode / native / VM+docker -----------------------


def _render_ops_script(image) -> str:
    """Translate Image DSL ops into a bash script for container-mode
    dispatch — the remote box is already the user's image, so apt/pip ops
    run inline over ssh. Idempotent: apt/pip on already-present packages
    is a cheap no-op.

    Requires `Image.from_registry(...)` — Dockerfile images are rejected
    upstream by the dispatch-time validator.
    """
    lines = ["set -euo pipefail"]
    lines.append(
        "for i in $(seq 1 60); do "
        "  sudo fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1 "
        "    && { echo waiting for apt; sleep 10; } "
        "    || break; "
        "done"
    )
    lines.append("export DEBIAN_FRONTEND=noninteractive")
    lines.append("export PATH=/opt/conda/bin:$PATH")

    for op in image.ops:
        kw = op.kwargs_dict()
        if op.kind == "apt_install" and op.args:
            pkgs = " ".join(shlex.quote(p) for p in op.args)
            lines.append(
                f"sudo apt-get update -qq && sudo apt-get install -y -qq "
                f"--no-install-recommends {pkgs}"
            )
        elif op.kind == "pip_install" and op.args:
            pkgs = " ".join(shlex.quote(p) for p in op.args)
            idx = ""
            if "index_url" in kw:
                idx = f" --index-url {shlex.quote(kw['index_url'])}"
            lines.append(f"pip install --quiet{idx} {pkgs}")
        elif op.kind == "pip_install_local_dir":
            path = kw.get("path", ".")
            editable = kw.get("editable", "1") == "1"
            flags = "-e " if editable else ""
            rel = path.lstrip("./")
            sub = f"/{rel}" if rel else ""
            lines.append(f'pip install --quiet {flags}"$HOME/{REMOTE_REPO_DIR}{sub}"')
        elif op.kind == "run" and op.args:
            for cmd in op.args:
                lines.append(cmd)
    return "; ".join(lines)


def _run_container_mode(*, target, function, rel_script, args, kwargs, max_runtime_seconds=None):
    """Container-mode dispatch: the box IS the user's image. Apply Image
    DSL ops inline over ssh, then invoke the bootstrap. No docker-in-
    docker, no nvidia-container-toolkit."""
    ops_script = _render_ops_script(function.image)
    if ops_script:
        _ssh(target, ops_script)

    user_env_exports = " ".join(
        f"export {k}={shlex.quote(str(v))};" for k, v in function.env.items()
    )
    remote = (
        "set -euo pipefail; "
        "export PATH=/opt/conda/bin:$PATH; "
        f'export RUNPLZ_OUT="$HOME/{REMOTE_OUT_DIR}"; '
        f'export RUNPLZ_SCRIPT="$HOME/{REMOTE_REPO_DIR}/{rel_script}"; '
        f"export RUNPLZ_FUNCTION={shlex.quote(function.name)}; "
        f"export RUNPLZ_ARGS={shlex.quote(json.dumps(args))}; "
        f"export RUNPLZ_KWARGS={shlex.quote(json.dumps(kwargs))}; "
        f"{user_env_exports} "
        'mkdir -p "$RUNPLZ_OUT"; '
        f'cd "$HOME/{REMOTE_REPO_DIR}"; '
        # Tee combined stdout+stderr to a remote file so we can tail it for
        # failure context (issue #17). `pipefail` + `set -e` above ensure the
        # bootstrap's exit code (not tee's) is what escapes this block.
        f'python -m runplz._bootstrap 2>&1 | tee "$HOME/{REMOTE_LAST_LOG}"'
    )
    try:
        r = subprocess.run(
            ["ssh", *SSH_OPTS, target, f"bash -lc {shlex.quote(remote)}"],
            timeout=max_runtime_seconds,
        )
    except subprocess.TimeoutExpired:
        _raise_for_runtime_cap(target, max_runtime_seconds, container_name=None)
    return r.returncode


def _run_native(
    *, target, function, rel_script, args, kwargs, has_nvidia, max_runtime_seconds=None
):
    """Native dispatch: install python+torch+user code in a venv on the
    remote and run the bootstrap directly (no docker)."""
    torch_index = (
        "https://download.pytorch.org/whl/cu121"
        if has_nvidia
        else "https://download.pytorch.org/whl/cpu"
    )
    setup = (
        "set -euo pipefail; "
        "for i in $(seq 1 120); do "
        "  sudo fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1 "
        "    && { echo waiting for apt; sleep 10; } "
        "    || break; "
        "done; "
        "sudo apt-get update -qq; "
        "sudo apt-get install -y -qq --no-install-recommends "
        "  python3 python3-venv python3-pip bzip2 wget rsync build-essential; "
        f"python3 -m venv {_NATIVE_VENV}; "
        f"source {_NATIVE_VENV}/bin/activate; "
        "pip install --quiet --upgrade pip; "
        f"pip install --quiet torch --index-url {torch_index}; "
        f"pip install --quiet -e $HOME/{REMOTE_REPO_DIR}"
    )
    _ssh(target, setup)

    user_env_exports = " ".join(
        f"export {k}={shlex.quote(str(v))};" for k, v in function.env.items()
    )
    remote = (
        "set -euo pipefail; "
        f'source "$HOME/runplz-venv/bin/activate"; '
        f'export RUNPLZ_OUT="$HOME/{REMOTE_OUT_DIR}"; '
        f'export RUNPLZ_SCRIPT="$HOME/{REMOTE_REPO_DIR}/{rel_script}"; '
        f"export RUNPLZ_FUNCTION={shlex.quote(function.name)}; "
        f"export RUNPLZ_ARGS={shlex.quote(json.dumps(args))}; "
        f"export RUNPLZ_KWARGS={shlex.quote(json.dumps(kwargs))}; "
        f"{user_env_exports} "
        'mkdir -p "$RUNPLZ_OUT"; '
        f'cd "$HOME/{REMOTE_REPO_DIR}"; '
        f'python -m runplz._bootstrap 2>&1 | tee "$HOME/{REMOTE_LAST_LOG}"'
    )
    try:
        r = subprocess.run(
            ["ssh", *SSH_OPTS, target, f"bash -lc {shlex.quote(remote)}"],
            timeout=max_runtime_seconds,
        )
    except subprocess.TimeoutExpired:
        _raise_for_runtime_cap(target, max_runtime_seconds, container_name=None)
    return r.returncode


def _build_image(target: str, image):
    """Build a docker image on the remote — either from the user's
    Dockerfile or from a synthesized one (Image.from_registry + DSL ops)."""
    if image.dockerfile is not None:
        build = (
            f"set -euo pipefail; "
            f"cd ~/{REMOTE_REPO_DIR} && "
            f"sudo docker build -f {shlex.quote(image.dockerfile)} "
            f"-t {REMOTE_IMAGE_TAG} ."
        )
    else:
        df = image.render_dockerfile()
        build = (
            f"set -euo pipefail; "
            f"cd ~/{REMOTE_REPO_DIR} && "
            f"cat <<'__EOF__' | sudo docker build -f - -t {REMOTE_IMAGE_TAG} .\n"
            f"{df}\n"
            f"__EOF__"
        )
    _ssh(target, build)


def _run_container_detached(
    *, target, container_name, function, rel_script, args, kwargs, gpu_flag
):
    env_flags = " ".join(f"-e {shlex.quote(f'{k}={v}')}" for k, v in function.env.items())
    runner_env = (
        f"-e RUNPLZ_OUT=/out "
        f"-e RUNPLZ_SCRIPT={shlex.quote('/workspace/' + rel_script)} "
        f"-e RUNPLZ_FUNCTION={shlex.quote(function.name)} "
        f"-e RUNPLZ_ARGS={shlex.quote(json.dumps(args))} "
        f"-e RUNPLZ_KWARGS={shlex.quote(json.dumps(kwargs))}"
    )
    # --network=host: simpler networking, no NAT overhead. See the long
    # comment in the old brev.py for the GPU-SSH-wedging backstory.
    start = (
        f"set -euo pipefail; "
        f"mkdir -p ~/{REMOTE_OUT_DIR} && "
        f"sudo docker run -d --name {container_name} --network=host {gpu_flag} "
        f"-v $HOME/{REMOTE_OUT_DIR}:/out "
        f"{runner_env} {env_flags} "
        f"{REMOTE_IMAGE_TAG} python -m runplz._bootstrap"
    )
    _ssh(target, start)


def _stream_and_wait(
    target: str,
    container_name: str,
    max_reconnects: int = 20,
    max_runtime_seconds: Optional[int] = None,
) -> int:
    """Stream container logs and return its exit code.

    Reconnect-tolerant: if ssh drops mid-stream we re-attach with
    `--tail 0` to pick up where we left off, then call `docker wait` for
    the exit code. Gives up after `max_reconnects` consecutive reconnect
    attempts. Wall-clock cap from `max_runtime_seconds` tracked across
    reconnects so a streaming job can't dodge it.
    """
    print(f"+ streaming logs from {container_name} (resilient to reconnects)", flush=True)
    started = time.monotonic()

    def _remaining_s() -> Optional[float]:
        if max_runtime_seconds is None:
            return None
        left = max_runtime_seconds - (time.monotonic() - started)
        return max(1.0, left)

    tail = "all"
    reconnects = 0
    while True:
        cmd = f"sudo docker logs -f --tail {tail} {container_name}"
        try:
            r = subprocess.run(
                ["ssh", *SSH_OPTS, target, cmd],
                timeout=_remaining_s(),
            )
        except subprocess.TimeoutExpired:
            _raise_for_runtime_cap(target, max_runtime_seconds, container_name=container_name)
        running = _container_running(target, container_name)
        if not running:
            break
        if max_runtime_seconds is not None and (time.monotonic() - started) >= max_runtime_seconds:
            _raise_for_runtime_cap(target, max_runtime_seconds, container_name=container_name)
        reconnects += 1
        if reconnects > max_reconnects:
            print(
                f"+ ssh reconnected {max_reconnects} times without finishing; "
                f"giving up on log stream. Container {container_name} may "
                f"still be running on {target}.",
                flush=True,
            )
            break
        print(
            f"+ ssh disconnected (rc={r.returncode}); container still "
            f"running, reconnecting log stream "
            f"({reconnects}/{max_reconnects})",
            flush=True,
        )
        tail = "0"
        time.sleep(2)
    r = subprocess.run(
        ["ssh", *SSH_OPTS, target, f"sudo docker wait {container_name}"],
        capture_output=True,
        text=True,
    )
    try:
        return int(r.stdout.strip() or "1")
    except ValueError:
        return 1


def _container_running(target: str, container_name: str) -> bool:
    # Treat ssh hangs / errors as "assume still running" so the caller keeps
    # retrying the log stream instead of giving up.
    try:
        r = subprocess.run(
            [
                "ssh",
                *SSH_OPTS,
                target,
                f"sudo docker inspect --format '{{{{.State.Running}}}}' {container_name}",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return True
    if r.returncode != 0:
        return True
    return r.stdout.strip() == "true"


# --- failure context -----------------------------------------------------


def _fetch_failure_tail(*, target: str, container_name: Optional[str]) -> str:
    """Fetch the last N lines of remote output for a failed run.

    - VM + docker path (`container_name` set): `docker logs --tail N <name>`.
      Docker persists logs until `docker rm`, so this works post-crash.
    - container-mode / native paths: the bootstrap tee'd its combined
      stdout+stderr into `$HOME/{REMOTE_LAST_LOG}`. `tail` that.

    Best-effort: return a diagnostic string rather than raising so we
    never mask the real error in the caller.
    """
    try:
        if container_name is not None:
            cmd = f"sudo docker logs --tail {FAILURE_TAIL_LINES} {container_name} 2>&1"
        else:
            cmd = (
                f'if [ -f "$HOME/{REMOTE_LAST_LOG}" ]; then '
                f'tail -n {FAILURE_TAIL_LINES} "$HOME/{REMOTE_LAST_LOG}"; '
                f"fi"
            )
        out = _ssh_capture(target, cmd)
        return (out or "").rstrip()
    except Exception as exc:  # noqa: BLE001
        return f"[runplz: could not fetch remote log tail — {type(exc).__name__}: {exc}]"


def _raise_for_runtime_cap(target: str, cap_s, container_name):
    """Shared timeout-path cleanup + raise for issue #16.

    container_name: set for VM+docker mode (kill the container with docker kill);
    None for container-mode / native (pkill the bootstrap process tree).

    Best-effort cleanup: if the kill ssh hangs or fails, still raise — the
    on_finish action in the caller's finally block will nuke the box anyway.
    """
    if container_name is not None:
        cleanup = f"sudo docker kill {container_name}"
    else:
        cleanup = "pkill -f 'runplz._bootstrap' || true"
    try:
        subprocess.run(
            ["ssh", *SSH_OPTS, target, cleanup],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception:  # noqa: BLE001
        pass
    raise RuntimeError(
        f"Remote run exceeded max_runtime_seconds={cap_s}; "
        f"issued remote cleanup ({cleanup!r}). "
        f"Raise or remove max_runtime_seconds if the job legitimately "
        f"needs longer."
    )


def make_container_name(fn_name: str) -> str:
    """Unique container name, short enough to read in logs."""
    return f"runplz-{fn_name}-{uuid.uuid4().hex[:8]}"
