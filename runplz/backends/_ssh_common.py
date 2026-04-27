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
import re
import shlex
import subprocess
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from runplz._excludes import DEFAULT_TRANSFER_EXCLUDES

# --- constants -----------------------------------------------------------

REMOTE_REPO_DIR = "runplz-repo"
REMOTE_OUT_DIR = "runplz-out"
REMOTE_RUNS_DIR = "runplz-runs"
REMOTE_LATEST_LINK = "runplz-latest"
REMOTE_META_DIRNAME = ".runplz"
REMOTE_IMAGE_TAG = "runplz-train:remote"

# container-mode / native paths tee the bootstrap's combined stdout+stderr
# into this file so we can `tail` it for failure context (issue #17). Lives
# under $HOME so no sudo needed and survives across ssh reconnects.
REMOTE_LAST_LOG = ".runplz-last.log"

# How many lines of remote log to include in a failure RuntimeError.
FAILURE_TAIL_LINES = 50
HEARTBEAT_INTERVAL_S = 30

# Directories that are noise on every upload and exclusions we apply on
# top of DEFAULT_TRANSFER_EXCLUDES (which only covers secrets). The
# default outputs dir name "out" is excluded here so the common case
# works without extra plumbing; non-default outputs_dir values are
# threaded into _rsync_up explicitly via _outputs_dir_excludes.
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
_REMOTE_SLUG_RE = re.compile(r"[^a-z0-9]+")
_MASKED_ENV_TOKENS = ("SECRET", "TOKEN", "PASSWORD", "KEY", "CREDENTIAL", "AUTH")


@dataclass(frozen=True)
class RemoteRunContext:
    run_id: str
    backend: str
    target: str
    function_name: str
    run_root_rel: str
    repo_rel: str
    out_rel: str
    meta_rel: str
    run_json_rel: str
    events_rel: str
    heartbeat_rel: str
    last_log_rel: str

    def _shell_path(self, rel: str) -> str:
        return f"$HOME/{rel}"

    def _display_path(self, rel: str) -> str:
        return f"~/{rel}"

    @property
    def run_root_shell(self) -> str:
        return self._shell_path(self.run_root_rel)

    @property
    def repo_shell(self) -> str:
        return self._shell_path(self.repo_rel)

    @property
    def out_shell(self) -> str:
        return self._shell_path(self.out_rel)

    @property
    def meta_shell(self) -> str:
        return self._shell_path(self.meta_rel)

    @property
    def run_json_shell(self) -> str:
        return self._shell_path(self.run_json_rel)

    @property
    def events_shell(self) -> str:
        return self._shell_path(self.events_rel)

    @property
    def heartbeat_shell(self) -> str:
        return self._shell_path(self.heartbeat_rel)

    @property
    def last_log_shell(self) -> str:
        return self._shell_path(self.last_log_rel)

    @property
    def repo_display(self) -> str:
        return self._display_path(self.repo_rel)

    @property
    def out_display(self) -> str:
        return self._display_path(self.out_rel)

    @property
    def meta_display(self) -> str:
        return self._display_path(self.meta_rel)

    @property
    def repo_rsync(self) -> str:
        return self._display_path(self.repo_rel)

    @property
    def out_rsync(self) -> str:
        return self._display_path(self.out_rel)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _slug_for_remote_path(value: str, *, max_len: int = 18) -> str:
    slug = _REMOTE_SLUG_RE.sub("-", value.lower()).strip("-")
    if not slug:
        return "x"
    clipped = slug[:max_len].strip("-")
    return clipped or "x"


def make_remote_run_context(*, backend: str, target: str, function_name: str) -> RemoteRunContext:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = (
        f"{timestamp}-"
        f"{_slug_for_remote_path(target)}-"
        f"{_slug_for_remote_path(function_name)}-"
        f"{uuid.uuid4().hex[:8]}"
    )
    run_root_rel = f"{REMOTE_RUNS_DIR}/{run_id}"
    out_rel = f"{run_root_rel}/out"
    meta_rel = f"{out_rel}/{REMOTE_META_DIRNAME}"
    return RemoteRunContext(
        run_id=run_id,
        backend=backend,
        target=target,
        function_name=function_name,
        run_root_rel=run_root_rel,
        repo_rel=f"{run_root_rel}/repo",
        out_rel=out_rel,
        meta_rel=meta_rel,
        run_json_rel=f"{meta_rel}/run.json",
        events_rel=f"{meta_rel}/events.ndjson",
        heartbeat_rel=f"{meta_rel}/heartbeat.ndjson",
        last_log_rel=f"{meta_rel}/last.log",
    )


def _masked_env_for_manifest(env: dict[str, Any]) -> dict[str, str]:
    masked = {}
    for key, value in env.items():
        text = str(value)
        if any(token in key.upper() for token in _MASKED_ENV_TOKENS):
            masked[key] = "***"
        else:
            masked[key] = text
    return masked


def _local_repo_git_info(repo: Path) -> dict[str, Any]:
    info: dict[str, Any] = {"revision": None, "dirty": None}
    try:
        rev = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if rev.returncode == 0:
            info["revision"] = rev.stdout.strip() or None
        dirty = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if dirty.returncode == 0:
            info["dirty"] = bool(dirty.stdout.strip())
    except Exception:  # noqa: BLE001
        pass
    return info


def build_remote_run_manifest(
    *,
    remote_run: RemoteRunContext,
    repo: Path,
    outputs_dir: str,
    args: list,
    kwargs: dict,
    env: dict[str, Any],
) -> dict[str, Any]:
    git_info = _local_repo_git_info(repo)
    return {
        "run_id": remote_run.run_id,
        "started_at": _utc_now_iso(),
        "backend": remote_run.backend,
        "target": remote_run.target,
        "function": remote_run.function_name,
        "cwd": str(repo),
        "outputs_dir": outputs_dir,
        "repo_revision": git_info["revision"],
        "repo_dirty": git_info["dirty"],
        "args": args,
        "kwargs": kwargs,
        "env": _masked_env_for_manifest(env),
        "remote_paths": {
            "run_root": f"~/{remote_run.run_root_rel}",
            "repo": remote_run.repo_display,
            "out": remote_run.out_display,
            "meta": remote_run.meta_display,
            "latest": f"~/{REMOTE_LATEST_LINK}",
        },
    }


def _prepare_remote_run(
    target: str,
    remote_run: RemoteRunContext,
    *,
    manifest: dict[str, Any],
    port: Optional[int] = None,
) -> None:
    print(
        f"+ remote run {remote_run.run_id}: "
        f"repo={remote_run.repo_display} out={remote_run.out_display}",
        flush=True,
    )
    initial_event = json.dumps(
        {"ts": _utc_now_iso(), "run_id": remote_run.run_id, "event": "launch_prepared"},
        sort_keys=True,
    )
    manifest_json = json.dumps(manifest, indent=2, sort_keys=True)
    remote = (
        "set -euo pipefail\n"
        f'mkdir -p "{remote_run.run_root_shell}" "{remote_run.repo_shell}" '
        f'"{remote_run.out_shell}" "{remote_run.meta_shell}"\n'
        f'ln -sfn "{remote_run.run_root_shell}" "$HOME/{REMOTE_LATEST_LINK}"\n'
        f"cat <<'__RUNPLZ_MANIFEST__' > \"{remote_run.run_json_shell}\"\n"
        f"{manifest_json}\n"
        "__RUNPLZ_MANIFEST__\n"
        f"cat <<'__RUNPLZ_EVENTS__' > \"{remote_run.events_shell}\"\n"
        f"{initial_event}\n"
        "__RUNPLZ_EVENTS__\n"
        f': > "{remote_run.heartbeat_shell}"\n'
        f': > "{remote_run.last_log_shell}"\n'
    )
    _ssh(target, remote, port=port)


def _record_remote_event(
    target: str,
    remote_run: Optional[RemoteRunContext],
    event: str,
    *,
    port: Optional[int] = None,
    **fields: Any,
) -> None:
    if remote_run is None:
        return
    payload = {"ts": _utc_now_iso(), "run_id": remote_run.run_id, "event": event}
    payload.update({k: v for k, v in fields.items() if v is not None})
    line = json.dumps(payload, sort_keys=True)
    remote = (
        "set -euo pipefail; "
        f'mkdir -p "{remote_run.meta_shell}"; '
        f"printf '%s\\n' {shlex.quote(line)} >> \"{remote_run.events_shell}\""
    )
    try:
        _ssh(target, remote, port=port)
    except Exception as exc:  # noqa: BLE001
        print(
            f"+ warning: failed to record remote lifecycle event "
            f"{event!r} for {remote_run.run_id}: {type(exc).__name__}: {exc}",
            flush=True,
        )


def _remote_logging_shell(remote_run: RemoteRunContext) -> str:
    return (
        f'RUNPLZ_EVENTS="{remote_run.events_shell}"\n'
        f'RUNPLZ_HEARTBEAT="{remote_run.heartbeat_shell}"\n'
        f'RUNPLZ_LAST_LOG="{remote_run.last_log_shell}"\n'
        f'RUNPLZ_RUN_ID="{remote_run.run_id}"\n'
        "runplz_ts() {\n"
        "  date -u +%Y-%m-%dT%H:%M:%SZ\n"
        "}\n"
        "runplz_event() {\n"
        '  runplz_event_name="$1"\n'
        '  runplz_exit_code="${2:-null}"\n'
        '  printf \'{"ts":"%s","run_id":"%s","event":"%s","exit_code":%s}\\n\' \\\n'
        '    "$(runplz_ts)" "$RUNPLZ_RUN_ID" "$runplz_event_name" "$runplz_exit_code" \\\n'
        '    >> "$RUNPLZ_EVENTS"\n'
        "}\n"
        "runplz_heartbeat() {\n"
        '  printf \'{"ts":"%s","run_id":"%s","event":"heartbeat","pid":%s}\\n\' \\\n'
        '    "$(runplz_ts)" "$RUNPLZ_RUN_ID" "$$" >> "$RUNPLZ_HEARTBEAT"\n'
        "}\n"
    )


def _wrap_remote_command_for_logging(command: str, remote_run: RemoteRunContext) -> str:
    return (
        "set -euo pipefail\n"
        f"{_remote_logging_shell(remote_run)}"
        "runplz_heartbeat_loop() {\n"
        "  while true; do\n"
        "    runplz_heartbeat\n"
        f"    sleep {HEARTBEAT_INTERVAL_S}\n"
        "  done\n"
        "}\n"
        "runplz_heartbeat_loop &\n"
        "runplz_hb_pid=$!\n"
        "runplz_cleanup() {\n"
        "  runplz_status=$?\n"
        '  kill "$runplz_hb_pid" >/dev/null 2>&1 || true\n'
        '  wait "$runplz_hb_pid" >/dev/null 2>&1 || true\n'
        '  runplz_event remote_command_exit "$runplz_status"\n'
        "}\n"
        "trap 'runplz_cleanup' EXIT\n"
        "runplz_event remote_command_start\n"
        "runplz_event bootstrap_start\n"
        f"{command}\n"
    )


# --- ssh-opts / rsync-transport builders --------------------------------


def _ssh_cmd_opts(port: Optional[int] = None) -> list:
    """Return SSH_OPTS plus `-p <port>` when a non-default port is pinned."""
    if port:
        return [*SSH_OPTS, "-p", str(int(port))]
    return list(SSH_OPTS)


def _rsync_ssh_transport(port: Optional[int] = None) -> str:
    """Build the argument rsync expects behind `-e`: the ssh invocation
    it should use for the transport. Shell-quoted so rsync splits it back
    into argv correctly."""
    parts = ["ssh", *_ssh_cmd_opts(port)]
    return " ".join(shlex.quote(p) for p in parts)


# --- low-level ssh / sh / rsync ------------------------------------------


def _sh(cmd):
    print("+ " + " ".join(shlex.quote(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True)


def _ssh(target: str, remote_cmd: str, *, port: Optional[int] = None):
    # Pass the whole pipeline as a SINGLE arg to ssh. If we pass
    # ["ssh", host, "bash", "-lc", cmd] instead, ssh space-joins the trailing
    # argv before sending to the remote shell, which then re-parses — turning
    # `bash -lc 'set -euo pipefail; X'` into `bash -lc set -euo pipefail; X`
    # (i.e. `set` runs with no args as the -c command, X runs in the outer
    # shell without errexit). Quoting with shlex.quote around the whole
    # command string avoids that.
    _sh(["ssh", *_ssh_cmd_opts(port), target, f"bash -lc {shlex.quote(remote_cmd)}"])


def _ssh_capture(target: str, remote_cmd: str, *, port: Optional[int] = None) -> str:
    r = subprocess.run(
        ["ssh", *_ssh_cmd_opts(port), target, remote_cmd],
        capture_output=True,
        text=True,
        timeout=60,
    )
    return r.stdout


def _remote_repo_shell(remote_run: Optional[RemoteRunContext]) -> str:
    if remote_run is not None:
        return remote_run.repo_shell
    return f"$HOME/{REMOTE_REPO_DIR}"


def _remote_out_shell(remote_run: Optional[RemoteRunContext]) -> str:
    if remote_run is not None:
        return remote_run.out_shell
    return f"$HOME/{REMOTE_OUT_DIR}"


def _remote_last_log_shell(remote_run: Optional[RemoteRunContext]) -> str:
    if remote_run is not None:
        return remote_run.last_log_shell
    return f"$HOME/{REMOTE_LAST_LOG}"


def _remote_repo_rsync(target: str, remote_run: Optional[RemoteRunContext]) -> str:
    if remote_run is not None:
        return f"{target}:{remote_run.repo_rsync}/"
    return f"{target}:{REMOTE_REPO_DIR}/"


def _remote_out_rsync(target: str, remote_run: Optional[RemoteRunContext]) -> str:
    if remote_run is not None:
        return f"{target}:{remote_run.out_rsync}/"
    return f"{target}:{REMOTE_OUT_DIR}/"


def _rsync_up(
    repo: Path,
    target: str,
    *,
    outputs_dir: Optional[str] = None,
    remote_run: Optional[RemoteRunContext] = None,
    port: Optional[int] = None,
):
    # Intentionally no --delete: a user who sshes in and leaves files under
    # ~/runplz-repo/ (logs, probe scripts, local edits) shouldn't have those
    # wiped by the next run. Stale files on the remote are cheap; accidental
    # user-data loss is not.
    cmd = ["rsync", "-az"]
    if port:
        cmd += ["-e", _rsync_ssh_transport(port)]
    for pat in _RSYNC_NOISE_EXCLUDES:
        cmd.append(f"--exclude={pat}")
    # Safety: never ship local secrets / dotenv / SSH keys to a remote box.
    # See runplz/_excludes.py for the rationale.
    for pat in DEFAULT_TRANSFER_EXCLUDES:
        cmd.append(f"--exclude={pat}")
    # Don't re-upload the outputs we'll rsync_down later. _RSYNC_NOISE_EXCLUDES
    # already covers `out/`; this catches a user-configured `outputs_dir`
    # (issue #55 — a 15 GB local outputs tree was getting shipped on every
    # launch when outputs_dir != "out").
    for pat in _outputs_dir_excludes(outputs_dir, repo):
        cmd.append(f"--exclude={pat}")
    cmd.extend([f"{repo}/", _remote_repo_rsync(target, remote_run)])
    _sh(cmd)
    _record_remote_event(target, remote_run, "rsync_up_done", port=port)


def _outputs_dir_excludes(outputs_dir: Optional[str], repo: Path) -> list[str]:
    """Translate the App's ``outputs_dir`` into rsync `--exclude` patterns.

    Returns an empty list for the default ``"out"`` (already in
    :data:`_RSYNC_NOISE_EXCLUDES`) and for absolute paths that don't live
    inside the repo (rsync's source root). Otherwise emits one anchored
    pattern (``/<rel>/``) plus, when the path is a single segment, the
    unanchored basename — matching the existing ``out`` convention so a
    nested ``foo/out`` would also be excluded.
    """
    if not outputs_dir:
        return []
    raw = str(outputs_dir).strip()
    if not raw or raw == "out":
        return []
    p = Path(raw)
    if p.is_absolute():
        try:
            rel = p.resolve().relative_to(repo.resolve())
        except ValueError:
            # outputs_dir lives outside the repo — rsync source root won't
            # see it anyway. Nothing to exclude.
            return []
    else:
        rel = p
    rel_posix = rel.as_posix().strip("/")
    if not rel_posix:
        return []
    patterns = [f"/{rel_posix}/"]
    if "/" not in rel_posix:
        patterns.append(rel_posix)
    return patterns


def _rsync_down(
    target: str,
    local_out: Path,
    *,
    remote_run: Optional[RemoteRunContext] = None,
    port: Optional[int] = None,
):
    _record_remote_event(target, remote_run, "rsync_down_start", port=port)
    cmd = ["rsync", "-az"]
    if port:
        cmd += ["-e", _rsync_ssh_transport(port)]
    cmd.extend([_remote_out_rsync(target, remote_run), f"{local_out}/"])
    _sh(cmd)


# --- connectivity helpers ------------------------------------------------


def _wait_until_ssh_reachable(
    target: str,
    *,
    max_wait_s: int = 1800,
    probe_interval_s: int = 15,
    refresh_callback: Optional[Callable[[], None]] = None,
    port: Optional[int] = None,
) -> None:
    """Block until an SSH session to `target` succeeds, or raise.

    Default budget: 1800s (30 min). Bumped from 1200s in 3.7.2 because
    8×A100/H100 shapes on Denvr / OCI consistently take 15-18 min to
    boot — the old 20-min cap tripped on healthy provisioning and left
    the freshly-created billed box running (see issues #29 / #34).
    Callers can still override via the backend config
    (BrevConfig.ssh_ready_wait_seconds, SshConfig.ssh_ready_wait_seconds).

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
                *_ssh_cmd_opts(port),
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
            except BaseException as exc:
                # If the callback raised on purpose (e.g. the instance
                # entered a terminal FAILURE state and we should bail
                # early instead of probing a dead box), let it through.
                # Only swallow plain exceptions from general-purpose
                # refresh logic (auth blip, etc.) — those are best-
                # effort. Signal exceptions (_OrchestratorKilled,
                # KeyboardInterrupt) propagate regardless.
                name = type(exc).__name__
                if name in ("_OrchestratorKilled", "BrevInstanceFailed"):
                    raise
                print(f"+ refresh callback raised: {exc}", flush=True)
        time.sleep(probe_interval_s)
    raise RuntimeError(
        f"SSH to {target} never became reachable within {max_wait_s}s (last error: {last_err!r})."
    )


def _ensure_remote_rsync(target: str, *, port: Optional[int] = None):
    """Install rsync on the remote if missing (slim container images
    often don't ship with rsync)."""
    cmd = (
        "command -v rsync >/dev/null 2>&1 && exit 0; "
        "export DEBIAN_FRONTEND=noninteractive; "
        "sudo apt-get update -qq && "
        "sudo apt-get install -y -qq --no-install-recommends rsync"
    )
    _ssh(target, cmd, port=port)


def _ensure_docker(target: str, timeout_s: int = 420, *, port: Optional[int] = None):
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
            *_ssh_cmd_opts(port),
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
        _sh(["ssh", *_ssh_cmd_opts(port), target, "curl -fsSL https://get.docker.com | sudo sh"])


def _remote_has_nvidia(target: str, *, port: Optional[int] = None) -> bool:
    # nvidia-smi is often pre-installed without a real GPU; the reliable
    # signal is /proc/driver/nvidia, which only exists when the kernel
    # module is loaded against real hardware.
    r = subprocess.run(
        [
            "ssh",
            *_ssh_cmd_opts(port),
            target,
            "test -d /proc/driver/nvidia && echo y || echo n",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return r.returncode == 0 and r.stdout.strip() == "y"


# --- dispatch: container-mode / native / VM+docker -----------------------


def _render_ops_script(image, *, remote_run: Optional[RemoteRunContext] = None) -> str:
    """Translate Image DSL ops into a bash script for container-mode
    dispatch — the remote box is already the user's image, so apt/pip ops
    run inline over ssh. Idempotent: apt/pip on already-present packages
    is a cheap no-op.

    Requires `Image.from_registry(...)` — Dockerfile images are rejected
    upstream by the dispatch-time validator.
    """
    remote_repo = _remote_repo_shell(remote_run)
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
            lines.append(f'pip install --quiet {flags}"{remote_repo}{sub}"')
        elif op.kind == "run" and op.args:
            for cmd in op.args:
                lines.append(cmd)
    return "; ".join(lines)


def _run_container_mode(
    *,
    target,
    function,
    rel_script,
    args,
    kwargs,
    remote_run: Optional[RemoteRunContext] = None,
    max_runtime_seconds=None,
    port=None,
):
    """Container-mode dispatch: the box IS the user's image. Apply Image
    DSL ops inline over ssh, then invoke the bootstrap. No docker-in-
    docker, no nvidia-container-toolkit.

    The bootstrap is launched detached (``setsid`` + ``nohup`` + stdio
    redirected to files) so a flaky client-side network connection can't
    kill the remote training job. Local streaming + completion tracking
    runs through a reconnect-tolerant tail-and-poll loop that mirrors
    the docker-mode ``_stream_and_wait`` pattern.
    """
    ops_script = _render_ops_script(function.image, remote_run=remote_run)
    if ops_script:
        _ssh(target, ops_script, port=port)

    user_env_exports = " ".join(
        f"export {k}={shlex.quote(str(v))};" for k, v in function.env.items()
    )
    inner = (
        "set -euo pipefail; "
        "export PATH=/opt/conda/bin:$PATH; "
        f'export RUNPLZ_OUT="{_remote_out_shell(remote_run)}"; '
        f'export RUNPLZ_SCRIPT="{_remote_repo_shell(remote_run)}/{rel_script}"; '
        f"export RUNPLZ_FUNCTION={shlex.quote(function.name)}; "
        f"export RUNPLZ_ARGS={shlex.quote(json.dumps(args))}; "
        f"export RUNPLZ_KWARGS={shlex.quote(json.dumps(kwargs))}; "
        f"{user_env_exports} "
        'mkdir -p "$RUNPLZ_OUT"; '
        f'cd "{_remote_repo_shell(remote_run)}"; '
        # Direct file redirect — no ``tee`` + no pipe. The previous
        # ``python ... 2>&1 | tee last.log`` pipeline chained python's
        # stdout through tee whose OWN stdout was the ssh socket. When
        # ssh dropped, tee took a SIGPIPE and the pipeline unwound back
        # through python, killing training. With a plain file redirect
        # inside a detached session, nothing in the pipeline is tethered
        # to the client's network.
        f'python -m runplz._bootstrap > "{_remote_last_log_shell(remote_run)}" 2>&1'
    )
    wrapped = _wrap_remote_command_for_logging(inner, remote_run) if remote_run else inner
    return _launch_detached_and_wait(
        target=target,
        wrapped_command=wrapped,
        remote_run=remote_run,
        max_runtime_seconds=max_runtime_seconds,
        port=port,
    )


def _run_native(
    *,
    target,
    function,
    rel_script,
    args,
    kwargs,
    has_nvidia,
    remote_run: Optional[RemoteRunContext] = None,
    max_runtime_seconds=None,
    port=None,
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
        f"pip install --quiet -e {_remote_repo_shell(remote_run)}"
    )
    _ssh(target, setup, port=port)

    user_env_exports = " ".join(
        f"export {k}={shlex.quote(str(v))};" for k, v in function.env.items()
    )
    # Launch detached and stream with reconnect tolerance — same pattern
    # as ``_run_container_mode``. See that function's docstring for the
    # SIGPIPE / SSH-drop rationale.
    inner = (
        "set -euo pipefail; "
        f'source "$HOME/runplz-venv/bin/activate"; '
        f'export RUNPLZ_OUT="{_remote_out_shell(remote_run)}"; '
        f'export RUNPLZ_SCRIPT="{_remote_repo_shell(remote_run)}/{rel_script}"; '
        f"export RUNPLZ_FUNCTION={shlex.quote(function.name)}; "
        f"export RUNPLZ_ARGS={shlex.quote(json.dumps(args))}; "
        f"export RUNPLZ_KWARGS={shlex.quote(json.dumps(kwargs))}; "
        f"{user_env_exports} "
        'mkdir -p "$RUNPLZ_OUT"; '
        f'cd "{_remote_repo_shell(remote_run)}"; '
        f'python -m runplz._bootstrap > "{_remote_last_log_shell(remote_run)}" 2>&1'
    )
    wrapped = _wrap_remote_command_for_logging(inner, remote_run) if remote_run else inner
    return _launch_detached_and_wait(
        target=target,
        wrapped_command=wrapped,
        remote_run=remote_run,
        max_runtime_seconds=max_runtime_seconds,
        port=port,
    )


def _build_image(
    target: str,
    image,
    *,
    remote_run: Optional[RemoteRunContext] = None,
    port: Optional[int] = None,
):
    """Build a docker image on the remote — either from the user's
    Dockerfile or from a synthesized one (Image.from_registry + DSL ops)."""
    remote_repo = _remote_repo_shell(remote_run)
    _record_remote_event(target, remote_run, "build_image_start", port=port)
    if image.dockerfile is not None:
        context = image.context or "."
        build = (
            f"set -euo pipefail; "
            f'cd "{remote_repo}" && '
            f"sudo docker build -f {shlex.quote(image.dockerfile)} "
            f"-t {REMOTE_IMAGE_TAG} {shlex.quote(context)}"
        )
    else:
        df = image.render_dockerfile()
        build = (
            f"set -euo pipefail; "
            f'cd "{remote_repo}" && '
            f"cat <<'__EOF__' | sudo docker build -f - -t {REMOTE_IMAGE_TAG} .\n"
            f"{df}\n"
            f"__EOF__"
        )
    _ssh(target, build, port=port)
    _record_remote_event(target, remote_run, "build_image_done", port=port)


def _run_container_detached(
    *,
    target,
    container_name,
    function,
    rel_script,
    args,
    kwargs,
    gpu_flag,
    app_name: Optional[str] = None,
    remote_run: Optional[RemoteRunContext] = None,
    port=None,
):
    env_flags = " ".join(f"-e {shlex.quote(f'{k}={v}')}" for k, v in function.env.items())
    label_flags = "--label runplz=1 "
    if app_name is not None:
        label_flags += f"--label {shlex.quote(f'runplz-app={app_name}')} "
    label_flags += f"--label {shlex.quote(f'runplz-function={function.name}')}"
    runner_env = (
        f"-e RUNPLZ_OUT=/out "
        f"-e RUNPLZ_SCRIPT={shlex.quote('/workspace/' + rel_script)} "
        f"-e RUNPLZ_FUNCTION={shlex.quote(function.name)} "
        f"-e RUNPLZ_ARGS={shlex.quote(json.dumps(args))} "
        f"-e RUNPLZ_KWARGS={shlex.quote(json.dumps(kwargs))}"
    )
    out_dir = _remote_out_shell(remote_run)
    monitor = ""
    if remote_run is not None:
        monitor = (
            f"{_remote_logging_shell(remote_run)}"
            f"runplz_event remote_command_start; "
            f"runplz_event container_started; "
            f"("
            f"  ("
            f"    while sudo docker inspect --format '{{{{.State.Running}}}}' {container_name} "
            f"      2>/dev/null | grep -qx true; do "
            f"      runplz_heartbeat; "
            f"      sleep {HEARTBEAT_INTERVAL_S}; "
            f"    done"
            f"  ) & "
            f"  runplz_hb_pid=$!; "
            f"  runplz_status=$(sudo docker wait {container_name} 2>/dev/null || echo null); "
            f'  kill "$runplz_hb_pid" >/dev/null 2>&1 || true; '
            f'  wait "$runplz_hb_pid" >/dev/null 2>&1 || true; '
            f'  runplz_event remote_command_exit "$runplz_status"'
            f") >/dev/null 2>&1 & "
        )
    # --network=host: simpler networking, no NAT overhead. See the long
    # comment in the old brev.py for the GPU-SSH-wedging backstory.
    start = (
        f"set -euo pipefail; "
        f'mkdir -p "{out_dir}" && '
        f"sudo docker run -d --name {container_name} {label_flags} "
        f"--network=host {gpu_flag} "
        f'-v "{out_dir}:/out" '
        f"{runner_env} {env_flags} "
        f"{REMOTE_IMAGE_TAG} python -m runplz._bootstrap >/dev/null; "
        f"{monitor}"
    )
    _ssh(target, start, port=port)


def _launch_detached_and_wait(
    *,
    target: str,
    wrapped_command: str,
    remote_run: Optional["RemoteRunContext"] = None,
    max_runtime_seconds: Optional[int] = None,
    port: Optional[int] = None,
    max_reconnects: int = 20,
) -> int:
    """Launch a bash command in a detached session and stream+wait locally.

    Core SSH-drop-survival path for container-mode and native backends.
    Before this helper, ``_run_container_mode`` and ``_run_native`` ran
    the bootstrap as a foreground ssh command whose stdout pipeline
    ended with ``tee`` — which meant any ssh drop SIGPIPEd tee and
    cascaded SIGPIPE / BrokenPipeError back through the whole pipeline,
    killing training.

    Detaching requires three elements, and missing any one of them
    leaves the process tethered to the client:

    - ``setsid`` — new process session. The old session's ``pg`` getting
      HUP'd doesn't propagate here.
    - ``nohup`` — SIGHUP is ignored explicitly (belt-and-suspenders with
      setsid) and stdin is wired to /dev/null.
    - Explicit stdout/stderr redirection to a file — nothing pipe-ward
      toward the ssh socket. Without this, even a detached session
      can SIGPIPE when the socket closes.

    Backend-agnostic: ``wrapped_command`` is arbitrary bash — typically
    the output of ``_wrap_remote_command_for_logging`` so events.ndjson
    records command start / exit with their real exit codes. We read
    the exit code back from that events file after the remote PID
    disappears.

    If ``remote_run`` is ``None`` (no events file available), falls
    back to the previous blocking ssh path. This keeps old call sites
    (tests, ad-hoc harnesses) working.
    """
    if remote_run is None:
        # Pre-remote_run call sites (early code paths) stay synchronous —
        # the whole point of the detach/poll path is the events file +
        # meta-dir it provides for PID / exit-code bookkeeping.
        try:
            r = subprocess.run(
                ["ssh", *_ssh_cmd_opts(port), target, f"bash -lc {shlex.quote(wrapped_command)}"],
                timeout=max_runtime_seconds,
            )
        except subprocess.TimeoutExpired:
            _raise_for_runtime_cap(target, max_runtime_seconds, container_name=None, port=port)
        return r.returncode

    meta = remote_run.meta_shell
    pid_file = f"{meta}/bootstrap.pid"
    run_script = f"{meta}/run.sh"
    driver_log = f"{meta}/run_driver.log"
    log_file = remote_run.last_log_shell
    events_file = remote_run.events_shell

    # Heredoc delimiter unique enough to guarantee no collision with
    # anything inside ``wrapped_command``. Single-quote the delimiter so
    # the heredoc body is passed through verbatim (no $-expansion, no
    # backtick execution, no \-escapes).
    delim = f"__RUNPLZ_CMD_{uuid.uuid4().hex}__"
    # All paths here start with ``$HOME/...`` (via ``RemoteRunContext._shell_path``).
    # ``shlex.quote`` would wrap the whole path in single quotes, which
    # suppresses ``$HOME`` expansion — ``cat > '$HOME/runplz-runs/…'``
    # tries to write to a literal ``$HOME/…`` path in cwd and fails.
    # Use double quotes instead so the remote shell expands ``$HOME``
    # while still tolerating characters like spaces in the run-id slug.
    # The paths themselves are all constructed from a fixed
    # ``RemoteRunContext`` template so they can't contain ``"`` or ``$``
    # sequences outside of the intentional ``$HOME`` prefix.
    launcher = (
        "set -euo pipefail\n"
        f'mkdir -p "{meta}"\n'
        f"cat > \"{run_script}\" << '{delim}'\n"
        f"{wrapped_command}\n"
        f"{delim}\n"
        f'chmod +x "{run_script}"\n'
        # Detach. setsid makes a new session leader (SIGHUP-immune),
        # nohup re-redirects stdin from /dev/null and ignores SIGHUP
        # for belt-and-suspenders, and the explicit >>/2>& redirects
        # give stdio fresh file destinations so nothing in the pipeline
        # reaches back to the ssh socket.
        f'nohup setsid bash "{run_script}" </dev/null '
        f'>> "{driver_log}" 2>&1 & '
        f'echo $! > "{pid_file}"\n'
        "disown || true\n"
    )
    # Launch ssh returns quickly once the detached job is running + PID
    # recorded. Anything that follows in this function is local polling.
    _ssh(target, launcher, port=port)

    return _tail_and_wait_for_detached(
        target=target,
        pid_file=pid_file,
        log_file=log_file,
        events_file=events_file,
        max_runtime_seconds=max_runtime_seconds,
        max_reconnects=max_reconnects,
        port=port,
    )


def _tail_and_wait_for_detached(
    *,
    target: str,
    pid_file: str,
    log_file: str,
    events_file: str,
    max_runtime_seconds: Optional[int] = None,
    max_reconnects: int = 20,
    port: Optional[int] = None,
) -> int:
    """Stream log_file via ssh ``tail -F`` and return remote exit code.

    Mirrors ``_stream_and_wait``'s reconnect pattern but uses a PID file
    + events file instead of docker commands for the "is the job still
    alive" and "what was the exit code" checks.

    If ssh drops mid-stream but the remote PID is still alive, reconnect
    and keep tailing. If the PID is gone, read the exit code from the
    events file and return it.
    """
    print(
        "+ streaming detached remote log (resilient to ssh reconnects)",
        flush=True,
    )
    started = time.monotonic()

    def _remaining_s() -> Optional[float]:
        if max_runtime_seconds is None:
            return None
        left = max_runtime_seconds - (time.monotonic() - started)
        return max(1.0, left)

    reconnects = 0
    while True:
        # Double-quoted for $HOME expansion (see launcher).
        cmd = f'tail -n +1 -F "{log_file}"'
        try:
            r = subprocess.run(
                ["ssh", *_ssh_cmd_opts(port), target, cmd],
                timeout=_remaining_s(),
            )
        except subprocess.TimeoutExpired:
            _raise_for_runtime_cap(target, max_runtime_seconds, container_name=None, port=port)
        if not _remote_pid_alive(target, pid_file, port=port):
            break
        if max_runtime_seconds is not None and (time.monotonic() - started) >= max_runtime_seconds:
            _raise_for_runtime_cap(target, max_runtime_seconds, container_name=None, port=port)
        reconnects += 1
        if reconnects > max_reconnects:
            print(
                f"+ ssh reconnected {max_reconnects} times without the remote "
                f"job finishing; giving up on live log stream and waiting for "
                f"remote exit only. The detached job on {target} is still "
                f"running and will finish on its own.",
                flush=True,
            )
            break
        print(
            f"+ ssh disconnected (rc={r.returncode}); remote job still "
            f"alive, reconnecting log stream ({reconnects}/{max_reconnects})",
            flush=True,
        )
        time.sleep(2)

    # If we gave up streaming while the remote was still alive, block
    # here until the pid file clears (so the caller sees the real exit
    # code, not a premature "unknown").
    while _remote_pid_alive(target, pid_file, port=port):
        if max_runtime_seconds is not None and (time.monotonic() - started) >= max_runtime_seconds:
            _raise_for_runtime_cap(target, max_runtime_seconds, container_name=None, port=port)
        time.sleep(min(30, HEARTBEAT_INTERVAL_S))

    return _read_remote_exit_code(target, events_file, port=port)


def _remote_pid_alive(target: str, pid_file: str, *, port: Optional[int] = None) -> bool:
    """Return True if the PID in ``pid_file`` is still running on the remote.

    Conservative on ssh errors: if we can't reach the box right now,
    assume the job is still alive so the caller keeps polling instead
    of prematurely declaring the job done. A real dead job will surface
    next poll once ssh recovers.
    """
    # Double-quoted path, not shlex.quote'd: ``pid_file`` starts with
    # ``$HOME/...`` and must expand on the remote shell. Same rationale
    # as the launcher in ``_launch_detached_and_wait``.
    probe = (
        f'pid=$(cat "{pid_file}" 2>/dev/null || true); '
        f'if [ -z "$pid" ]; then echo "no-pid"; exit 0; fi; '
        f'if kill -0 "$pid" 2>/dev/null; then echo "alive"; else echo "dead"; fi'
    )
    try:
        r = subprocess.run(
            ["ssh", *_ssh_cmd_opts(port), target, probe],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return True
    if r.returncode != 0:
        return True
    return r.stdout.strip() == "alive"


def _read_remote_exit_code(target: str, events_file: str, *, port: Optional[int] = None) -> int:
    """Parse the last ``remote_command_exit`` entry from events.ndjson.

    Returns 1 when the events file is missing, unreadable, or has no
    exit entry yet — treating "unknown" as failure keeps a silent
    exit-code regression from masquerading as success.
    """
    # Same expansion rule: ``events_file`` is a $HOME-relative path.
    probe = f"grep -F 'remote_command_exit' \"{events_file}\" 2>/dev/null | tail -n 1 || true"
    try:
        r = subprocess.run(
            ["ssh", *_ssh_cmd_opts(port), target, probe],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return 1
    line = r.stdout.strip()
    if not line:
        return 1
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return 1
    ec = obj.get("exit_code")
    if isinstance(ec, bool):
        # bool is a subclass of int in Python; explicit check keeps
        # ``true`` / ``false`` exit codes from sneaking through.
        return 1
    if isinstance(ec, int):
        return ec
    if isinstance(ec, str):
        try:
            return int(ec)
        except ValueError:
            return 1
    return 1


def _stream_and_wait(
    target: str,
    container_name: str,
    max_reconnects: int = 20,
    max_runtime_seconds: Optional[int] = None,
    port: Optional[int] = None,
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
                ["ssh", *_ssh_cmd_opts(port), target, cmd],
                timeout=_remaining_s(),
            )
        except subprocess.TimeoutExpired:
            _raise_for_runtime_cap(
                target, max_runtime_seconds, container_name=container_name, port=port
            )
        running = _container_running(target, container_name, port=port)
        if not running:
            break
        if max_runtime_seconds is not None and (time.monotonic() - started) >= max_runtime_seconds:
            _raise_for_runtime_cap(
                target, max_runtime_seconds, container_name=container_name, port=port
            )
        reconnects += 1
        if reconnects > max_reconnects:
            print(
                f"+ ssh reconnected {max_reconnects} times without finishing; "
                f"giving up on log stream and waiting for container exit "
                f"only. Container {container_name} is still running on "
                f"{target}.",
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
    try:
        r = subprocess.run(
            ["ssh", *_ssh_cmd_opts(port), target, f"sudo docker wait {container_name}"],
            capture_output=True,
            text=True,
            timeout=_remaining_s(),
        )
    except subprocess.TimeoutExpired:
        _raise_for_runtime_cap(
            target, max_runtime_seconds, container_name=container_name, port=port
        )
    try:
        return int(r.stdout.strip() or "1")
    except ValueError:
        return 1


def _container_running(target: str, container_name: str, *, port: Optional[int] = None) -> bool:
    # Treat ssh hangs / errors as "assume still running" so the caller keeps
    # retrying the log stream instead of giving up.
    try:
        r = subprocess.run(
            [
                "ssh",
                *_ssh_cmd_opts(port),
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


def _fetch_failure_tail(
    *,
    target: str,
    container_name: Optional[str],
    remote_run: Optional[RemoteRunContext] = None,
    port: Optional[int] = None,
) -> str:
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
                f'if [ -f "{_remote_last_log_shell(remote_run)}" ]; then '
                f'tail -n {FAILURE_TAIL_LINES} "{_remote_last_log_shell(remote_run)}"; '
                f"fi"
            )
        out = _ssh_capture(target, cmd, port=port)
        return (out or "").rstrip()
    except Exception as exc:  # noqa: BLE001
        return f"[runplz: could not fetch remote log tail — {type(exc).__name__}: {exc}]"


def _raise_for_runtime_cap(target: str, cap_s, container_name, *, port: Optional[int] = None):
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
            ["ssh", *_ssh_cmd_opts(port), target, cleanup],
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
