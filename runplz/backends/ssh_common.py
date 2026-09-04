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

import contextlib
import dataclasses
import json
import os
import re
import shlex
import signal
import subprocess
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

from runplz.backends import docker
from runplz.backends.provisioning import RetryPolicy, retry_budget_spent
from runplz.excludes import DEFAULT_TRANSFER_EXCLUDES

__all__ = [
    # --- running one function on a reachable box -------------------------
    "dispatch_to_target",
    "run_on_provisioned_vm",
    "DispatchResult",
    "format_remote_failure",
    # --- the dispatch pipeline, in the order dispatch_to_target runs it ---
    # Public because a backend may need to drive a stage on its own, and
    # because these are the seams the test suite patches.
    "prepare_remote_run",
    "ensure_remote_rsync",
    "check_preconditions",
    "PreconditionFailed",
    "ensure_docker",
    "remote_has_nvidia",
    "build_image",
    "run_container_detached",
    "run_container_mode",
    "run_native",
    "stream_and_wait",
    "fetch_failure_tail",
    "apt_lock_wait_shell",
    # --- provisioning-backend lifecycle ----------------------------------
    "orchestrator_signal_cleanup",
    "OrchestratorKilled",
    "CLEANUP_SIGNALS",
    "wait_until_ssh_reachable",
    # --- per-run identity and layout on the remote -----------------------
    "RemoteRunContext",
    "make_remote_run_context",
    "build_remote_run_manifest",
    "LocalRepoState",
    "inspect_local_repo",
    "select_source_paths",
    # --- remote filesystem layout (a cross-version on-disk contract) -----
    "REMOTE_REPO_DIR",
    "REMOTE_OUT_DIR",
    "REMOTE_RUNS_DIR",
    "REMOTE_LATEST_LINK",
    "REMOTE_META_DIRNAME",
    "REMOTE_IMAGE_TAG",
    "REMOTE_LAST_LOG",
    "BOOTSTRAP_PID_FILENAME",
    "CONTAINER_FILENAME",
    "LOCAL_SSH_OPTIONS_FILENAME",
    # --- moving code and results -----------------------------------------
    "rsync_up",
    "rsync_down",
    "rsync_ssh_transport",
    "rsync_transport_flags",
    "SshOptions",
    "read_local_ssh_options",
    "write_local_ssh_options",
    # --- talking to the box ----------------------------------------------
    "ssh_exec",
    "retry_on_transport_failure",
    "is_ssh_transport_failure",
    "SSH_TRANSPORT_EXIT",
    "SSH_RESULTS_POLICY",
    "SSH_PREP_POLICY",
    "SSH_OPTS",
    "RSYNC_TRANSPORT_EXITS",
    "ssh_capture",
    "ssh_cmd_opts",
    "run_local",
    "parse_probe_sections",
    "container_running",
    "raise_for_runtime_cap",
    "render_image_ops_script",
    # --- detached bootstrap ----------------------------------------------
    "build_detached_launcher",
    "build_detached_log_command",
    "build_detached_status_probe",
    "inspect_detached_run",
    "wait_for_detached_start",
    "detached_launch_diagnostics",
    "INACTIVITY_POLL_INTERVAL_S",
    "build_inactivity_probe",
    "seconds_since_activity",
    "launch_detached_and_wait",
    "tail_and_wait_for_detached",
    "remote_pid_alive",
    "LAUNCH_CLAIM_FILENAME",
    "container_exists",
    "detached_run_started",
    "read_remote_exit_code",
    "DetachedProcessState",
    "DetachedRunStatus",
    "HEARTBEAT_INTERVAL_S",
    "DETACHED_START_TIMEOUT_S",
    "DETACHED_START_POLL_INTERVAL_S",
    # --- stopping a run ---------------------------------------------------
    "build_kill_command",
    "make_container_name",
    "RUN_ID_ENV_VAR",
    "FAILURE_TAIL_LINES",
    "DEFAULT_KILL_TIMEOUT_S",
    "KILL_SETTLE_S",
    # --- validating untrusted values headed for a remote shell -----------
    "remote_shell_path",
    "validate_remote_path",
    "validate_run_id",
    "is_safe_run_id",
]

# --- constants -----------------------------------------------------------

REMOTE_REPO_DIR = "runplz-repo"
REMOTE_OUT_DIR = "runplz-out"
REMOTE_RUNS_DIR = "runplz-runs"
REMOTE_LATEST_LINK = "runplz-latest"
REMOTE_META_DIRNAME = ".runplz"
REMOTE_IMAGE_TAG = "runplz-train:remote"

# Per-run control files inside the meta dir. The pid is what every probe
# keys off; the container file names the docker container in VM+docker mode,
# which is a child of dockerd and has to be signalled separately.
BOOTSTRAP_PID_FILENAME = "bootstrap.pid"
# Written *before* the bootstrap is spawned. The pid can only be recorded
# after, which leaves a window in which a launch that already landed still
# looks like "nothing here" — and a retry in that window starts a second
# job. This marker closes it (issue #84).
LAUNCH_CLAIM_FILENAME = "launch.claimed"
CONTAINER_FILENAME = "container"

# The bootstrap is exec'd with its run id in the environment, so every
# descendant inherits it and `runplz kill` can identify a run's processes
# exactly -- including workers orphaned to init after the supervisor dies,
# which is the case that motivated issue #67. A process group would be the
# obvious handle, but bash disables job control in non-interactive shells, so
# the bootstrap never becomes a group leader and its pgid is the *launching
# shell's* -- not unique to the run, and unsafe to signal wholesale. A run id
# is unique, survives reparenting, and cannot be recycled by PID wraparound.
RUN_ID_ENV_VAR = "RUNPLZ_RUN_ID"

# container-mode / native paths tee the bootstrap's combined stdout+stderr
# into this file so we can `tail` it for failure context (issue #17). Lives
# under $HOME so no sudo needed and survives across ssh reconnects.
REMOTE_LAST_LOG = ".runplz-last.log"

# How many lines of remote log to include in a failure RuntimeError.
FAILURE_TAIL_LINES = 50
HEARTBEAT_INTERVAL_S = 30
DETACHED_START_TIMEOUT_S = 15
DETACHED_START_POLL_INTERVAL_S = 1

# `runplz kill` sends SIGTERM, gives the job this long to unwind (flush
# checkpoints, close writers), then escalates to SIGKILL and allows a
# short settle window for the kernel to reap the tree.
DEFAULT_KILL_TIMEOUT_S = 10
KILL_SETTLE_S = 5

# Directories that are noise on every upload and exclusions we apply on
# top of DEFAULT_TRANSFER_EXCLUDES (which only covers secrets). The
# default outputs dir name "out" is excluded here so the common case
# works without extra plumbing; non-default outputs_dir values are
# threaded into rsync_up explicitly via _outputs_dir_excludes.
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
# A tuple, not a list: this is exported, and a mutable module-level
# default would let one caller's `.append()` change every later ssh
# invocation in the process. Build your own list from it.
SSH_OPTS = (
    # Ephemeral cloud boxes are a new host key every time, and every probe
    # runs with BatchMode=yes — which cannot answer OpenSSH's default
    # "are you sure?" prompt. Without this, a fresh EC2 IP fails host-key
    # verification on every attempt for the full ssh_ready_wait_seconds
    # while the instance bills. `accept-new` trusts a *first* sighting only;
    # a key that later *changes* still fails, which is the point.
    "-o",
    "StrictHostKeyChecking=accept-new",
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
)


@dataclass(frozen=True)
class SshOptions:
    """Everything that shapes how runplz reaches a box over ssh.

    This exists as one object rather than a pile of parallel parameters
    because the plumbing threads it through ~50 call sites: `port` alone was
    already threaded that far, and adding a second scalar beside it would
    have doubled that and made a third worse.

    Backends that manage their own ssh config need none of it — brev writes
    `~/.brev/ssh_config` and gcp's `config-ssh` writes an `IdentityFile`
    entry — so the default is "whatever ssh would do on its own".
    """

    port: Optional[int] = None
    # Private key to authenticate with. EC2 key pairs are conventionally
    # saved as ~/.ssh/my-key.pem, which ssh will not offer unless it is
    # agent-loaded or named in a Host block — and a Host block cannot be
    # written in advance because the instance IP does not exist yet.
    identity_file: Optional[str] = None
    # Override OpenSSH's known-hosts database. Primarily useful for isolated
    # harnesses, where accepting a throwaway host key must not mutate the
    # developer's ~/.ssh/known_hosts. Kept in the same object so ssh and
    # rsync cannot accidentally use different trust stores.
    known_hosts_file: Optional[str] = None

    @classmethod
    def coerce(cls, value) -> "SshOptions":
        """Accept None, a bare port int, or an SshOptions."""
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, int) and not isinstance(value, bool):
            return cls(port=value)
        raise TypeError(f"expected SshOptions, an int port, or None; got {value!r}")

    def argv(self) -> list:
        opts = list(SSH_OPTS)
        if self.port:
            opts += ["-p", str(int(self.port))]
        if self.identity_file:
            # IdentitiesOnly stops a loaded agent from offering its other
            # keys first and tripping the server's MaxAuthTries before ours
            # is ever tried.
            opts += [
                "-i",
                os.path.expanduser(str(self.identity_file)),
                "-o",
                "IdentitiesOnly=yes",
            ]
        if self.known_hosts_file:
            opts += [
                "-o",
                f"UserKnownHostsFile={os.path.expanduser(str(self.known_hosts_file))}",
            ]
        return opts

    def to_dict(self) -> dict:
        """The subset worth recording so `runplz tail/status/kill` can reach
        the same box later. Written to a *local* sidecar, never uploaded —
        see :func:`write_local_ssh_options`."""
        out = {}
        if self.port:
            out["port"] = int(self.port)
        if self.identity_file:
            out["identity_file"] = str(self.identity_file)
        if self.known_hosts_file:
            out["known_hosts_file"] = str(self.known_hosts_file)
        return out

    @classmethod
    def from_dict(cls, data) -> "SshOptions":
        data = data or {}
        return cls(
            port=data.get("port"),
            identity_file=data.get("identity_file"),
            known_hosts_file=data.get("known_hosts_file"),
        )


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


@dataclass(frozen=True)
class LocalRepoState:
    """Git state recorded alongside a staged remote source snapshot.

    ``dirty`` is deliberately limited to changes to tracked files. Untracked
    inputs are staged, but reported separately; ignored artifacts are neither
    staged nor allowed to make the tracked checkout look dirty.
    """

    revision: Optional[str]
    dirty: Optional[bool]
    untracked: Optional[bool]
    ignored: Optional[bool]


class DetachedProcessState(str, Enum):
    """Observable state of a detached bootstrap process on a remote host."""

    MISSING = "missing"
    RUNNING = "running"
    ZOMBIE = "zombie"
    DEAD = "dead"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class DetachedRunStatus:
    """Startup-event and process state returned by one remote probe."""

    process_state: DetachedProcessState
    started: bool
    pid: Optional[int] = None


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


def inspect_local_repo(repo: Path) -> LocalRepoState:
    """Inspect tracked, untracked, and ignored Git state independently.

    A non-Git directory returns all-``None`` state so callers can preserve
    full-tree fallback behavior without treating Git inspection failures as a
    clean checkout.
    """

    revision: Optional[str] = None
    try:
        rev = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if rev.returncode != 0:
            return LocalRepoState(None, None, None, None)
        revision = rev.stdout.strip() or None
        status = subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "status",
                "--porcelain=v1",
                "-z",
                "--ignored=matching",
                "--untracked-files=normal",
                "--",
                ".",
            ],
            capture_output=True,
            timeout=15,
        )
        if status.returncode != 0:
            return LocalRepoState(revision, None, None, None)
        codes = [entry[:2] for entry in status.stdout.split(b"\0") if len(entry) >= 3]
        return LocalRepoState(
            revision=revision,
            dirty=any(code not in (b"??", b"!!") for code in codes),
            untracked=any(code == b"??" for code in codes),
            ignored=any(code == b"!!" for code in codes),
        )
    except Exception:  # noqa: BLE001
        return LocalRepoState(revision, None, None, None)


def select_source_paths(repo: Path) -> Optional[tuple[str, ...]]:
    """Return the Git-selected files that make up a remote source snapshot.

    The selection is tracked files plus untracked files not matched by Git's
    normal ignore stack (repository, info/exclude, and global excludes).
    Tracked paths absent from the working tree (deleted or sparse) are removed
    so rsync does not fail trying to stage them. Initialized submodules are
    expanded through their own Git selections rather than copied as opaque
    directories, preserving ignore behavior at every repository boundary.

    ``None`` means Git could not provide a selection; callers should fall back
    to their documented non-Git directory behavior. An empty tuple is a valid
    empty Git snapshot.
    """

    base = ["git", "-C", str(repo), "ls-files", "-z"]
    try:
        selected = subprocess.run(
            [*base, "--cached", "--others", "--exclude-standard", "--", "."],
            capture_output=True,
            timeout=30,
        )
        if selected.returncode != 0:
            return None
        staged = subprocess.run(
            [*base, "--stage", "--", "."],
            capture_output=True,
            timeout=30,
        )
        if staged.returncode != 0:
            return None
    except (OSError, subprocess.TimeoutExpired):
        return None

    gitlinks = set()
    for entry in staged.stdout.split(b"\0"):
        metadata, separator, path = entry.partition(b"\t")
        if separator and metadata.startswith(b"160000 ") and path:
            gitlinks.add(path)

    paths = []
    for path in selected.stdout.split(b"\0"):
        if not path or path in gitlinks:
            continue
        decoded = os.fsdecode(path)
        if os.path.lexists(repo / decoded):
            paths.append(decoded)

    for gitlink in sorted(gitlinks):
        prefix = os.fsdecode(gitlink)
        submodule = repo / prefix
        # An uninitialized submodule may exist as an empty directory. Requiring
        # its own .git marker prevents Git from walking back up into the
        # superproject and mistaking the gitlink for submodule contents.
        if not os.path.lexists(submodule / ".git"):
            continue
        nested = select_source_paths(submodule)
        if nested is None:
            continue
        paths.extend(f"{prefix}/{path}" for path in nested)

    return tuple(paths)


LOCAL_SSH_OPTIONS_FILENAME = "ssh.json"


def write_local_ssh_options(host_out: Path, ssh_opts=None) -> None:
    """Record how to reach this box, in a file that never leaves this machine.

    `runplz tail/status/kill` need the port and key to follow a run on a host
    that isn't in your ssh config — an EC2 box whose IP didn't exist when you
    wrote it. That belongs beside the run manifest, but *not* inside it: the
    manifest is heredoc'd onto the remote box, and a key path there would
    disclose the local username and key filename to a rented, possibly
    multi-tenant host. The codebase already masks anything env-shaped whose
    name contains KEY; this keeps the same promise.

    rsync_down does not use --delete, so this survives the outputs sync.
    """
    options = SshOptions.coerce(ssh_opts)
    meta = Path(host_out) / REMOTE_META_DIRNAME
    meta.mkdir(parents=True, exist_ok=True)
    path = meta / LOCAL_SSH_OPTIONS_FILENAME
    recorded = options.to_dict()
    if not recorded:
        # Nothing worth saying; don't leave a stale file from an earlier run.
        if path.exists():
            path.unlink()
        return
    path.write_text(json.dumps(recorded, indent=2, sort_keys=True))


def read_local_ssh_options(outputs_dir: Path) -> SshOptions:
    """Read back what :func:`write_local_ssh_options` recorded, if anything."""
    path = Path(outputs_dir) / REMOTE_META_DIRNAME / LOCAL_SSH_OPTIONS_FILENAME
    try:
        return SshOptions.from_dict(json.loads(path.read_text()))
    except (OSError, ValueError):
        return SshOptions()


def build_remote_run_manifest(
    *,
    remote_run: RemoteRunContext,
    repo: Path,
    outputs_dir: str,
    args: list,
    kwargs: dict,
    env: dict[str, Any],
) -> dict[str, Any]:
    git_state = inspect_local_repo(repo)
    return {
        "run_id": remote_run.run_id,
        "started_at": _utc_now_iso(),
        "backend": remote_run.backend,
        "target": remote_run.target,
        "function": remote_run.function_name,
        "cwd": str(repo),
        "outputs_dir": outputs_dir,
        "repo_revision": git_state.revision,
        "repo_dirty": git_state.dirty,
        "repo_untracked": git_state.untracked,
        "repo_ignored": git_state.ignored,
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


def prepare_remote_run(
    target: str,
    remote_run: RemoteRunContext,
    *,
    manifest: dict[str, Any],
    ssh_opts: Optional[SshOptions] = None,
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
    # Writes the run dir and overwrites the manifest, so running it twice
    # leaves the same state. This is the call that failed in issue #84.
    retry_on_transport_failure(
        lambda: ssh_exec(target, remote, ssh_opts=ssh_opts),
        label=f"prepare remote run {remote_run.run_id}",
    )


def _record_remote_event(
    target: str,
    remote_run: Optional[RemoteRunContext],
    event: str,
    *,
    ssh_opts: Optional[SshOptions] = None,
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
        ssh_exec(target, remote, ssh_opts=ssh_opts)
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


def ssh_cmd_opts(ssh_opts=None) -> list:
    """Return the ssh flags for `ssh_opts` (an SshOptions, a port, or None)."""
    return SshOptions.coerce(ssh_opts).argv()


def rsync_ssh_transport(ssh_opts=None) -> str:
    """Build the argument rsync expects behind `-e`: the ssh invocation
    it should use for the transport. Shell-quoted so rsync splits it back
    into argv correctly."""
    parts = ["ssh", *ssh_cmd_opts(ssh_opts)]
    return " ".join(shlex.quote(p) for p in parts)


def rsync_transport_flags(ssh_opts=None) -> list:
    """`-e <ssh ...>` for rsync, or nothing when there is nothing to say.

    A bare `-e ssh ...` on every rsync would be churn with no effect, so the
    override is only emitted when the options actually differ from ssh's own
    defaults.
    """
    if SshOptions.coerce(ssh_opts) == SshOptions():
        return []
    return ["-e", rsync_ssh_transport(ssh_opts)]


# --- transient ssh transport failures -------------------------------------
#
# Issue #84: a run reached SSH readiness, then died in prepare_remote_run
# with `Can't assign requested address` / `client_loop: send disconnect:
# Broken pipe`. The remote held only run.json and a launch_prepared event —
# nothing staged, no bootstrap — and the identical command succeeded on the
# next try. The transport blinked; the run did not have to die.

# ssh reserves 255 for its own failures, as distinct from the remote
# command's exit code. That covers both "never connected" and "the session
# dropped mid-command", so it is a signal that the command did not *complete*
# — not that it never started. Retrying is therefore safe for steps that
# converge on the same state when repeated, and the steps below are chosen on
# that basis rather than on any claim of atomicity.
SSH_TRANSPORT_EXIT = 255
# docker's wording for "that container is not here" — the only non-zero
# `docker inspect` exit that actually answers the question.
_DOCKER_NO_SUCH_RE = re.compile(r"no such (object|container)", re.I)
# rsync's transport codes. 12 is the one a mid-transfer connection drop
# actually produces ("error in rsync protocol data stream") — 30 needs a
# --timeout we never pass and 35 is daemon-mode only, so without 12 the rsync
# retry would do nothing for the failure it exists for. 12 is broader than
# transport (a missing remote rsync binary reports it too), but
# ensure_remote_rsync runs first and the budget is bounded, so that
# ambiguity costs seconds on an already-doomed run.
RSYNC_TRANSPORT_EXITS = (12, 30, 35, SSH_TRANSPORT_EXIT)
# Staging is cheap to repeat and the job already spent minutes getting here.
SSH_PREP_POLICY = RetryPolicy(waits=(0, 2, 5, 10), deadline_s=300)
# Pulling results back runs on the teardown side, often just before a paid
# box is stopped. Same reasoning as the cloud teardown policies: keep it
# short so a box that has already gone away costs seconds, not half a minute.
SSH_RESULTS_POLICY = RetryPolicy(waits=(0, 2), deadline_s=60)


def is_ssh_transport_failure(returncode: Optional[int]) -> bool:
    """True when ssh itself failed rather than the remote command."""
    return returncode == SSH_TRANSPORT_EXIT


def retry_on_transport_failure(
    action: Callable[[], Any],
    *,
    label: str,
    policy: RetryPolicy = SSH_PREP_POLICY,
    retriable_exits: tuple = (SSH_TRANSPORT_EXIT,),
    can_retry: Optional[Callable[[], bool]] = None,
    sleep=None,
):
    """Run `action`, retrying it when the *transport* fails.

    Only for operations that converge on the same state when repeated.
    `can_retry` is the guard for the ones that do not: a launch whose delivery
    is ambiguous asks the remote whether a bootstrap marker exists, and
    declines the retry if one does (issue #84).

    `policy.waits` is the sleep *before* each attempt, matching
    :class:`~runplz.backends.provisioning.RetryPolicy` and the loop in
    `run_with_retries`, so a policy opening with a non-zero wait is honoured
    here too. The `can_retry` question is asked after that backoff, never
    before: a probe issued the instant the transport failed hits the same
    broken link, answers "can't tell", and (failing closed) vetoes every
    retry.
    """
    if not policy.waits:
        raise ValueError(f"{label}: retry policy must contain at least one attempt")
    nap = sleep or time.sleep
    started_all = time.monotonic()
    last = None
    for attempt, wait_s in enumerate(policy.waits, start=1):
        # Budget first, so an already-spent one does not burn the wait only
        # to discover it should have stopped.
        if attempt > 1 and retry_budget_spent(policy, started_all, time.monotonic()):
            break
        if wait_s:
            nap(wait_s)
        if attempt > 1:
            # Asked *after* the backoff, never before: a probe issued the
            # instant the transport failed hits the same broken link,
            # answers "can't tell", and (failing closed) vetoes every retry.
            if can_retry is not None and not can_retry():
                print(
                    f"+ {label} failed at the transport layer, but the remote "
                    f"already shows work in progress — not retrying",
                    flush=True,
                )
                break
            print(
                f"+ {label} retrying after a transient ssh transport failure "
                f"(attempt {attempt}/{len(policy.waits)})",
                flush=True,
            )
        try:
            return action()
        except subprocess.CalledProcessError as exc:
            last = exc
            if exc.returncode not in retriable_exits:
                raise
    assert last is not None  # the loop only exits here after a failure
    raise last


# --- low-level ssh / sh / rsync ------------------------------------------


def run_local(cmd, *, stdin: Optional[bytes] = None):
    print("+ " + " ".join(shlex.quote(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True, input=stdin)


def ssh_exec(target: str, remote_cmd: str, *, ssh_opts: Optional[SshOptions] = None):
    # Pass the whole pipeline as a SINGLE arg to ssh. If we pass
    # ["ssh", host, "bash", "-lc", cmd] instead, ssh space-joins the trailing
    # argv before sending to the remote shell, which then re-parses — turning
    # `bash -lc 'set -euo pipefail; X'` into `bash -lc set -euo pipefail; X`
    # (i.e. `set` runs with no args as the -c command, X runs in the outer
    # shell without errexit). Quoting with shlex.quote around the whole
    # command string avoids that.
    run_local(["ssh", *ssh_cmd_opts(ssh_opts), target, f"bash -lc {shlex.quote(remote_cmd)}"])


def ssh_capture(target: str, remote_cmd: str, *, ssh_opts: Optional[SshOptions] = None) -> str:
    r = subprocess.run(
        ["ssh", *ssh_cmd_opts(ssh_opts), target, remote_cmd],
        capture_output=True,
        text=True,
        timeout=60,
    )
    return r.stdout


class PreconditionFailed(RuntimeError):
    """Remote state failed a declared precondition. Raised before bootstrap
    so we don't waste paid GPU minutes on a doomed run (issue #56)."""


# Below the declared minimum → warn. Below this fraction of the minimum →
# hard-fail. Matches the rule the issue proposes.
_PRECONDITION_FAIL_FRACTION = 0.5


def check_preconditions(
    target: str, preconditions: dict, *, ssh_opts: Optional[SshOptions] = None
) -> None:
    """Probe declared preconditions on the remote box and warn-or-fail.

    No-op when ``preconditions`` is empty. Fires a single ssh round-trip
    that emits a labelled section per probe; we parse and compare each
    against the declared minimum. Below the minimum prints a warning;
    below half the minimum raises :class:`PreconditionFailed` so the
    dispatch path bails before bootstrap.
    """
    if not preconditions:
        return
    probe = (
        "echo '---SHM_BYTES---'; df -B1 /dev/shm 2>/dev/null | tail -n 1 "
        "| awk '{print $4}' || true; "
        "echo '---HOME_FREE_BYTES---'; df -B1 \"$HOME\" 2>/dev/null | tail -n 1 "
        "| awk '{print $4}' || true; "
        "echo '---GPU_COUNT---'; nvidia-smi -L 2>/dev/null | wc -l || echo 0; "
        "echo '---GPU_MIN_VRAM_MIB---'; "
        "nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits "
        "2>/dev/null | sort -n | head -n 1 || echo 0; "
        "echo '---END---'"
    )
    out = ssh_capture(target, probe, ssh_opts=ssh_opts)
    sections = parse_probe_sections(out)

    failures: list[str] = []
    warnings: list[str] = []

    def _check(key, observed_label, observed_value, threshold):
        """Compare ``observed_value`` to ``threshold`` and route the result.

        ``observed_value`` and ``threshold`` are both in the user-facing unit
        the precondition key declares (GB for shm_gb / disk_free_gb /
        gpu_memory_gb, count for gpu_count). One callsite per key keeps the
        warn/fail logic uniform and the messages legible.
        """
        if observed_value is None:
            warnings.append(
                f"could not probe {key} on {target} (no value reported); "
                f"declared minimum {threshold} not enforced."
            )
            return
        if observed_value < threshold * _PRECONDITION_FAIL_FRACTION:
            failures.append(
                f"{key}: {observed_label}={observed_value:g} is below "
                f"{int(_PRECONDITION_FAIL_FRACTION * 100)}% of the declared "
                f"minimum ({threshold}). Refusing to bootstrap a doomed run."
            )
        elif observed_value < threshold:
            warnings.append(
                f"{key}: {observed_label}={observed_value:g} is below the "
                f"declared minimum ({threshold}). Job will continue but may fail."
            )

    if "shm_gb" in preconditions:
        observed = _bytes_to_gb(_first_int(sections.get("SHM_BYTES")))
        _check("shm_gb", "free_shm_gb", observed, preconditions["shm_gb"])
    if "disk_free_gb" in preconditions:
        observed = _bytes_to_gb(_first_int(sections.get("HOME_FREE_BYTES")))
        _check("disk_free_gb", "free_home_gb", observed, preconditions["disk_free_gb"])
    if "gpu_count" in preconditions:
        count = _first_int(sections.get("GPU_COUNT"))
        observed = float(count) if count is not None else None
        _check("gpu_count", "gpu_count", observed, preconditions["gpu_count"])
    if "gpu_memory_gb" in preconditions:
        mib = _first_int(sections.get("GPU_MIN_VRAM_MIB"))
        observed = (mib / 1024.0) if mib is not None and mib > 0 else None
        _check("gpu_memory_gb", "min_gpu_memory_gb", observed, preconditions["gpu_memory_gb"])

    for w in warnings:
        print(f"+ precondition warning: {w}", flush=True)
    if failures:
        joined = "\n  - ".join(failures)
        raise PreconditionFailed(f"remote preconditions failed on {target}:\n  - {joined}")


def parse_probe_sections(stdout: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    current = None
    buf: list[str] = []
    for line in (stdout or "").splitlines():
        line = line.rstrip()
        if line.startswith("---") and line.endswith("---"):
            if current:
                sections[current] = "\n".join(buf).strip()
                buf = []
            current = line.strip("-").strip()
        else:
            buf.append(line)
    if current and current != "END":
        sections[current] = "\n".join(buf).strip()
    return sections


def _first_int(text: Optional[str]) -> Optional[int]:
    """Pull the first integer out of a probe section's output. Tolerates
    leading whitespace, trailing units, and probes that emitted nothing."""
    if not text:
        return None
    for token in text.split():
        digits = "".join(ch for ch in token if ch.isdigit())
        if digits:
            try:
                return int(digits)
            except ValueError:
                continue
    return None


def _bytes_to_gb(b: Optional[int]) -> Optional[float]:
    if b is None or b <= 0:
        return None
    return b / (1024.0**3)


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


def rsync_up(
    repo: Path,
    target: str,
    *,
    outputs_dir: Optional[str] = None,
    remote_run: Optional[RemoteRunContext] = None,
    ssh_opts: Optional[SshOptions] = None,
):
    # Intentionally no --delete: a user who sshes in and leaves files under
    # ~/runplz-repo/ (logs, probe scripts, local edits) shouldn't have those
    # wiped by the next run. Stale files on the remote are cheap; accidental
    # user-data loss is not.
    selected_paths = select_source_paths(repo)
    cmd = ["rsync", "-az", *rsync_transport_flags(ssh_opts)]
    if selected_paths is not None:
        # --files-from changes rsync's -a implication: recursion must be
        # requested explicitly. NUL delimiters preserve every valid Git path,
        # including names containing whitespace or newlines.
        cmd += ["--from0", "--files-from=-", "--recursive"]
    for pat in _RSYNC_NOISE_EXCLUDES:
        cmd.append(f"--exclude={pat}")
    # Safety: never ship local secrets / dotenv / SSH keys to a remote box.
    # See runplz/excludes.py for the rationale.
    for pat in DEFAULT_TRANSFER_EXCLUDES:
        cmd.append(f"--exclude={pat}")
    # Don't re-upload the outputs we'll rsync_down later. _RSYNC_NOISE_EXCLUDES
    # already covers `out/`; this catches a user-configured `outputs_dir`
    # (issue #55 — a 15 GB local outputs tree was getting shipped on every
    # launch when outputs_dir != "out").
    for pat in _outputs_dir_excludes(outputs_dir, repo):
        cmd.append(f"--exclude={pat}")
    cmd.extend([f"{repo}/", _remote_repo_rsync(target, remote_run)])
    stdin = None
    if selected_paths is not None:
        stdin = b"".join(os.fsencode(path) + b"\0" for path in selected_paths)

    # rsync is a sync: repeating it converges on the same result, and a
    # half-transferred tree is exactly what a retry is for.
    def _send():
        # Keep the historical call shape: `stdin` is only passed when there
        # is one, so anything that fakes run_local keeps working.
        if stdin is None:
            run_local(cmd)
        else:
            run_local(cmd, stdin=stdin)

    retry_on_transport_failure(
        _send,
        label=f"rsync up to {target}",
        retriable_exits=RSYNC_TRANSPORT_EXITS,
    )
    _record_remote_event(target, remote_run, "rsync_up_done", ssh_opts=ssh_opts)


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


def rsync_down(
    target: str,
    local_out: Path,
    *,
    remote_run: Optional[RemoteRunContext] = None,
    ssh_opts: Optional[SshOptions] = None,
):
    _record_remote_event(target, remote_run, "rsync_down_start", ssh_opts=ssh_opts)
    cmd = ["rsync", "-az", *rsync_transport_flags(ssh_opts)]
    cmd.extend([_remote_out_rsync(target, remote_run), f"{local_out}/"])
    # rsync is a sync: repeating it converges on the same result.
    retry_on_transport_failure(
        lambda: run_local(cmd),
        label=f"rsync down from {target}",
        retriable_exits=RSYNC_TRANSPORT_EXITS,
        policy=SSH_RESULTS_POLICY,
    )


# --- connectivity helpers ------------------------------------------------


def wait_until_ssh_reachable(
    target: str,
    *,
    max_wait_s: int = 1800,
    probe_interval_s: int = 15,
    refresh_callback: Optional[Callable[[], None]] = None,
    ssh_opts: Optional[SshOptions] = None,
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
                *ssh_cmd_opts(ssh_opts),
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
                # entered a terminal FAILURE state and we should bail early
                # instead of probing a dead box), let it through. Only
                # swallow plain exceptions from general-purpose refresh
                # logic (auth blip, etc.) — those are best-effort.
                #
                # Matched by class, not by name: this guard is what carries
                # a SIGTERM out of the ssh wait so teardown can run, and a
                # string comparison silently disarmed it once when the
                # exception was renamed — leaking the billed box that
                # issue #38 exists to prevent.
                if isinstance(exc, (OrchestratorKilled, KeyboardInterrupt, SystemExit)):
                    raise
                # BrevInstanceFailed lives in the brev backend, which cannot
                # be imported here without a cycle, so it stays a name match.
                if type(exc).__name__ == "BrevInstanceFailed":
                    raise
                print(f"+ refresh callback raised: {exc}", flush=True)
        time.sleep(probe_interval_s)
    raise RuntimeError(
        f"SSH to {target} never became reachable within {max_wait_s}s (last error: {last_err!r})."
    )


def apt_lock_wait_shell(iterations: int = 60, sleep_s: int = 5) -> str:
    """Shell that waits out an apt/dpkg lock, then repairs an interrupted one.

    Four copies of this loop had drifted apart with three different bounds,
    and the `dpkg --configure -a` repair reached only one — which is how a
    retried transport blip could land straight in "dpkg was interrupted"
    (exit 100, not retriable) on the other three.

    `fuser` ships in psmisc, which the slim images this code exists for do
    not carry; its absence used to end the loop on its first iteration. The
    lock is checked without it when it is missing.
    """
    return (
        "export DEBIAN_FRONTEND=noninteractive; "
        f"for _i in $(seq 1 {iterations}); do "
        "  if command -v fuser >/dev/null 2>&1; then "
        "    sudo fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1 || break; "
        "  elif command -v lsof >/dev/null 2>&1; then "
        "    sudo lsof /var/lib/dpkg/lock-frontend >/dev/null 2>&1 || break; "
        "  else "
        "    sudo flock -n /var/lib/dpkg/lock-frontend true >/dev/null 2>&1 && break; "
        "  fi; "
        "  echo 'apt busy, waiting'; "
        f"  sleep {sleep_s}; "
        "done; "
        # An interrupted install leaves dpkg half-configured; without this the
        # retry meets `dpkg was interrupted` instead of the blip it retries.
        "sudo dpkg --configure -a >/dev/null 2>&1 || true; "
    )


def ensure_remote_rsync(target: str, *, ssh_opts: Optional[SshOptions] = None):
    """Install rsync on the remote if missing (slim container images
    often don't ship with rsync)."""
    cmd = (
        "command -v rsync >/dev/null 2>&1 && exit 0; "
        + apt_lock_wait_shell(iterations=30, sleep_s=2)
        + "sudo apt-get update -qq && "
        "sudo apt-get install -y -qq --no-install-recommends rsync"
    )
    retry_on_transport_failure(
        lambda: ssh_exec(target, cmd, ssh_opts=ssh_opts),
        label=f"ensure rsync on {target}",
    )


def ensure_docker(target: str, timeout_s: int = 420, *, ssh_opts: Optional[SshOptions] = None):
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
            *ssh_cmd_opts(ssh_opts),
            "-o",
            "BatchMode=yes",
            target,
            wait_script,
        ],
        timeout=timeout_s,
    )
    if is_ssh_transport_failure(r.returncode):
        # The wait script never ran. Piping a remote installer onto a box
        # that already has working docker because the link blinked is not a
        # fallback, it is a second problem — surface the blip instead.
        raise subprocess.CalledProcessError(r.returncode, "ssh")
    if r.returncode != 0:
        print(
            f"+ docker daemon not reachable on {target} after {timeout_s}s; "
            f"falling back to get-docker.sh",
            flush=True,
        )
        retry_on_transport_failure(
            lambda: run_local(
                [
                    "ssh",
                    *ssh_cmd_opts(ssh_opts),
                    target,
                    "curl -fsSL https://get.docker.com | sudo sh",
                ]
            ),
            label=f"install docker on {target}",
        )


def remote_has_nvidia(target: str, *, ssh_opts: Optional[SshOptions] = None) -> bool:
    # nvidia-smi is often pre-installed without a real GPU; the reliable
    # signal is /proc/driver/nvidia, which only exists when the kernel
    # module is loaded against real hardware.
    def _probe():
        r = subprocess.run(
            [
                "ssh",
                *ssh_cmd_opts(ssh_opts),
                target,
                "test -d /proc/driver/nvidia && echo y || echo n",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if is_ssh_transport_failure(r.returncode):
            # A blip here used to read as "no GPU", silently dropping
            # `--gpus all` and running a multi-hour job on CPU on a paid GPU
            # box. Make it a failure the retry can see instead.
            raise subprocess.CalledProcessError(r.returncode, "ssh", r.stdout, r.stderr)
        return r

    r = retry_on_transport_failure(_probe, label=f"probe GPUs on {target}")
    return r.returncode == 0 and r.stdout.strip() == "y"


# --- dispatch: container-mode / native / VM+docker -----------------------


def render_image_ops_script(image, *, remote_run: Optional[RemoteRunContext] = None) -> str:
    """Translate Image DSL ops into a bash script for container-mode
    dispatch — the remote box is already the user's image, so apt/pip ops
    run inline over ssh. Idempotent: apt/pip on already-present packages
    is a cheap no-op.

    Requires `Image.from_registry(...)` — Dockerfile images are rejected
    upstream by the dispatch-time validator.
    """
    remote_repo = _remote_repo_shell(remote_run)
    lines = ["set -euo pipefail"]
    # Retried on a transport blip, so it needs the dpkg repair too.
    lines.append(apt_lock_wait_shell(iterations=60, sleep_s=10).rstrip("; "))
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


def run_container_mode(
    *,
    target,
    function,
    rel_script,
    args,
    kwargs,
    remote_run: Optional[RemoteRunContext] = None,
    max_runtime_seconds=None,
    max_inactivity_seconds=None,
    inactivity_action="diagnose",
    ssh_opts=None,
):
    """Container-mode dispatch: the box IS the user's image. Apply Image
    DSL ops inline over ssh, then invoke the bootstrap. No docker-in-
    docker, no nvidia-container-toolkit.

    The bootstrap is launched detached (``nohup`` + stdio redirected to
    files) so a flaky client-side network connection can't
    kill the remote training job. Local streaming + completion tracking
    runs through a reconnect-tolerant tail-and-poll loop that mirrors
    the docker-mode ``stream_and_wait`` pattern.
    """
    ops_script = render_image_ops_script(function.image, remote_run=remote_run)
    if ops_script:
        # Container mode's equivalent of build_image: apt/pip layers applied
        # inline. brev's container mode never reaches build_image, so without
        # this the mode where the box *is* the user's image had no cover at
        # all for a #84-shaped blip.
        retry_on_transport_failure(
            lambda: ssh_exec(target, ops_script, ssh_opts=ssh_opts),
            label=f"apply image ops on {target}",
        )

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
    return launch_detached_and_wait(
        target=target,
        wrapped_command=wrapped,
        remote_run=remote_run,
        max_runtime_seconds=max_runtime_seconds,
        max_inactivity_seconds=max_inactivity_seconds,
        inactivity_action=inactivity_action,
        ssh_opts=ssh_opts,
    )


def run_native(
    *,
    target,
    function,
    rel_script,
    args,
    kwargs,
    has_nvidia,
    remote_run: Optional[RemoteRunContext] = None,
    max_runtime_seconds=None,
    max_inactivity_seconds=None,
    inactivity_action="diagnose",
    ssh_opts=None,
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
        + apt_lock_wait_shell(iterations=120, sleep_s=10)
        + "sudo apt-get update -qq; "
        "sudo apt-get install -y -qq --no-install-recommends "
        "  python3 python3-venv python3-pip bzip2 wget rsync build-essential; "
        f"python3 -m venv {_NATIVE_VENV}; "
        f"source {_NATIVE_VENV}/bin/activate; "
        "pip install --quiet --upgrade pip; "
        f"pip install --quiet torch --index-url {torch_index}; "
        f"pip install --quiet -e {_remote_repo_shell(remote_run)}"
    )
    # A superset of ensure_remote_rsync's apt-get, which is retried — the
    # two adjacent calls should not behave differently for the same failure.
    retry_on_transport_failure(
        lambda: ssh_exec(target, setup, ssh_opts=ssh_opts),
        label=f"prepare native environment on {target}",
    )

    user_env_exports = " ".join(
        f"export {k}={shlex.quote(str(v))};" for k, v in function.env.items()
    )
    # Launch detached and stream with reconnect tolerance — same pattern
    # as ``run_container_mode``. See that function's docstring for the
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
    return launch_detached_and_wait(
        target=target,
        wrapped_command=wrapped,
        remote_run=remote_run,
        max_runtime_seconds=max_runtime_seconds,
        max_inactivity_seconds=max_inactivity_seconds,
        inactivity_action=inactivity_action,
        ssh_opts=ssh_opts,
    )


def build_image(
    target: str,
    image,
    *,
    remote_run: Optional[RemoteRunContext] = None,
    ssh_opts: Optional[SshOptions] = None,
):
    """Build a docker image on the remote — either from the user's
    Dockerfile or from a synthesized one (Image.from_registry + DSL ops)."""
    remote_repo = _remote_repo_shell(remote_run)
    _record_remote_event(target, remote_run, "build_image_start", ssh_opts=ssh_opts)
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
    retry_on_transport_failure(
        lambda: ssh_exec(target, build, ssh_opts=ssh_opts),
        label=f"build image on {target}",
    )
    _record_remote_event(target, remote_run, "build_image_done", ssh_opts=ssh_opts)


def run_container_detached(
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
    ssh_opts=None,
):
    env_flags = " ".join(f"-e {shlex.quote(f'{k}={v}')}" for k, v in function.env.items())
    label_flags = docker.label_flags(app_name, function.name)
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
            # Name the container so `runplz kill` can signal it. It is a child
            # of dockerd, not of the bootstrap, so it sits outside the process
            # group every other stop path relies on.
            f'mkdir -p "{remote_run.meta_shell}"; '
            f"printf '%s\\n' {shlex.quote(container_name)} "
            f'> "{remote_run.meta_shell}/{CONTAINER_FILENAME}"; '
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
    # Same ambiguity as the detached launcher: `docker run -d` may have
    # started the container before the transport dropped. Ask docker before
    # retrying rather than starting a second one (issue #84).
    retry_on_transport_failure(
        lambda: ssh_exec(target, start, ssh_opts=ssh_opts),
        label=f"start container {container_name}",
        can_retry=lambda: not container_exists(target, container_name, ssh_opts=ssh_opts),
    )


def detached_run_started(
    target: str,
    pid_file: str,
    events_file: str,
    *,
    ssh_opts: Optional[SshOptions] = None,
) -> bool:
    """True when a launch for this run has already been claimed remotely.

    The no-duplicate guarantee for issue #84. Asks about the claim marker as
    well as the pid file and the start event, because the launcher can only
    record its pid *after* spawning — so between the spawn and that write, a
    job that is already running looks like one that never started.

    Anything other than a confirmed "clear" counts as landed. Declining a
    retry costs one failed run; a wrong retry costs a second training job on
    the same GPU.
    """
    claim_file = f"{os.path.dirname(pid_file)}/{LAUNCH_CLAIM_FILENAME}"
    probe = (
        f'if [ -e "{claim_file}" ] || [ -s "{pid_file}" ] || '
        f"grep -Fq 'remote_command_start' \"{events_file}\" 2>/dev/null; "
        f"then echo landed; else echo clear; fi"
    )
    try:
        r = subprocess.run(
            ["ssh", *ssh_cmd_opts(ssh_opts), target, probe],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return True
    if r.returncode != 0:
        return True
    return (r.stdout or "").strip() != "clear"


def container_exists(
    target: str, container_name: str, *, ssh_opts: Optional[SshOptions] = None
) -> bool:
    """True when a container of this name is already present on the remote.

    Only docker's own "no such object" counts as absence. Everything else —
    an unreachable box, a hung probe, a dockerd hiccup, sudo refusing on a
    non-interactive session — means the question was not answered, and an
    unanswered question is not proof that nothing landed. Guessing "absent"
    there is exactly what would let a retry start a second container.
    """
    probed = _docker_inspect(target, container_name, "{{.Name}}", ssh_opts=ssh_opts)
    if probed is None:
        return True
    returncode, stdout, stderr = probed
    if returncode == 0:
        return bool(stdout.strip())
    if _DOCKER_NO_SUCH_RE.search(stderr) or _DOCKER_NO_SUCH_RE.search(stdout):
        return False
    return True


def build_detached_launcher(remote_run: RemoteRunContext, wrapped_command: str) -> str:
    """Build the portable remote shell used to launch one detached run.

    The parent ignores SIGHUP before spawning so SSH teardown cannot win the
    fork-to-spawn race. The child script repeats the ignore as defense in
    depth. Those two traps are the whole SIGHUP guarantee (#74); ``nohup``
    contributes nothing to it, and neither does it provide the redirections —
    ``</dev/null`` and the driver log are written here explicitly.

    What ``nohup`` is still for is PID stability: it execs bash in place, so
    the PID captured by ``$!`` remains the bootstrap PID, which is why it was
    chosen over ``setsid`` (which may fork when invoked by a process group
    leader, leaving callers monitoring a short-lived wrapper). A plain
    backgrounded ``bash`` is equally PID-stable, so ``nohup`` is preferred but
    never required.

    Hence the probe. macOS ``nohup`` refuses to detach in a non-interactive ssh
    session — ``can't detach from console: Inappropriate ioctl for device`` —
    and bash forks the job before nohup execs and fails, so ``$!`` records a
    pid that is already dead and the run looks like a payload that crashed
    instantly (#92). Asking ``nohup`` whether it will work costs one process
    and is right for any platform whose nohup refuses, where sniffing for
    Darwin would only be right for Darwin.
    """

    meta = remote_run.meta_shell
    pid_file = f"{meta}/{BOOTSTRAP_PID_FILENAME}"
    run_script = f"{meta}/run.sh"
    claim_file = f"{meta}/{LAUNCH_CLAIM_FILENAME}"
    driver_log = f"{meta}/run_driver.log"
    delim = f"__RUNPLZ_CMD_{uuid.uuid4().hex}__"
    return (
        "set -euo pipefail\n"
        "trap '' HUP\n"
        f'mkdir -p "{meta}"\n'
        f"cat > \"{run_script}\" << '{delim}'\n"
        "trap '' HUP\n"
        f"{wrapped_command}\n"
        f"{delim}\n"
        f'chmod +x "{run_script}"\n'
        # Claim the run before anything can be spawned, so an ambiguous
        # launch is never mistaken for one that never happened.
        f': > "{claim_file}"\n'
        # Kept after the claim so the claim-before-spawn ordering still holds.
        "runplz_nohup=nohup\n"
        "nohup true >/dev/null 2>&1 || runplz_nohup=\n"
        f"{_run_id_env_assignment(remote_run.run_id)}"
        f'${{runplz_nohup}} bash "{run_script}" </dev/null >> "{driver_log}" 2>&1 &\n'
        f'echo $! > "{pid_file}"\n'
    )


_SAFE_RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]*$")
# `$HOME/...`, `~/...` or an absolute path, made of path-safe characters only.
# The meta path comes from a manifest that is rsynced down off the remote box,
# so it is untrusted input that lands inside a remote shell command.
_SAFE_REMOTE_PATH_RE = re.compile(r"^(\$HOME/|~/|/)[A-Za-z0-9._/-]*$")
_KILL_SIGNALS = ("TERM", "INT", "HUP", "QUIT", "KILL")


def _run_id_env_assignment(run_id: str) -> str:
    """Shell prefix that puts the run id in a command's exec environment."""
    validate_run_id(run_id)
    if not run_id:
        return ""
    return f"{RUN_ID_ENV_VAR}={run_id} "


def validate_remote_path(path: str, *, what: str = "path") -> str:
    """Reject a remote path that could break out of the word it lands in.

    Paths reach us from a run manifest that is written on the remote box and
    rsynced back down, so they are untrusted input headed for a double-quoted
    shell word — where ``$(...)`` and backticks still run.
    """
    if not _SAFE_REMOTE_PATH_RE.match(path or ""):
        raise ValueError(
            f"refusing to use {what} {path!r}: expected an absolute or "
            f"$HOME-relative path made of plain path characters."
        )
    return path


def validate_run_id(run_id: str, *, what: str = "run id") -> str:
    """Reject a run id that could break out of the shell word it lands in."""
    if not _SAFE_RUN_ID_RE.match(run_id or ""):
        raise ValueError(f"unsafe {what} for remote shell: {run_id!r}")
    return run_id


def is_safe_run_id(run_id: str) -> bool:
    """True when `run_id` is safe to interpolate into a remote command."""
    return bool(_SAFE_RUN_ID_RE.match(run_id or ""))


def remote_shell_path(path: str) -> str:
    """Make a manifest path usable inside a remote shell command.

    The manifest records display paths like ``~/runplz-runs/<id>/out/.runplz``.
    Tilde expansion happens before any quoting, so a ``~/`` path is literal in
    both ``'...'`` and ``"..."`` and the file is simply never found. ``$HOME``
    does expand inside double quotes, so callers can interpolate the result
    into ``"..."`` and get the directory they meant.

    Validates on the way through — see :func:`validate_remote_path`.
    """
    if path.startswith("~/"):
        path = f"$HOME/{path[2:]}"
    return validate_remote_path(path, what="remote path from the run manifest")


def build_kill_command(
    meta: str,
    *,
    run_id: str = "",
    timeout_s: int = DEFAULT_KILL_TIMEOUT_S,
    escalate: bool = True,
    first_signal: str = "TERM",
    proc_root: str = "/proc",
) -> str:
    """Build the remote shell that stops one detached run in a single hop.

    Does the whole signal/poll/escalate dance remotely so the escalation
    timeout is measured against the job rather than against ssh latency, and
    so a flaky link can't strand the run half-signalled.

    A run's processes are identified by the run id in their environment
    (see :data:`RUN_ID_ENV_VAR`), which is exact: it finds workers orphaned
    to init after the supervisor died, and it cannot match anything but this
    run. The recorded bootstrap pid is only a fallback, for runs launched
    before the marker existed or on a host without procfs -- and it is used
    only while the run has no terminal exit event, because a pid outlives its
    process and can be recycled by PID wraparound.

    The container in VM+docker mode is signalled separately: it is a child of
    dockerd, so it is in neither the marker set nor the pid's descendants.

    Every value interpolated here is validated first -- the meta path in
    particular arrives from a manifest rsynced off the remote box.

    ``proc_root`` exists so the marker scan can be exercised on a machine
    without procfs (a macOS dev box): point it at a tree of ``<pid>/environ``
    files naming real processes. Production always uses ``/proc``.
    """

    meta = validate_remote_path(meta, what="meta path")
    proc_root = validate_remote_path(proc_root, what="proc root").rstrip("/")
    validate_run_id(run_id)
    if first_signal not in _KILL_SIGNALS:
        raise ValueError(f"unsupported signal {first_signal!r}; expected one of {_KILL_SIGNALS}")
    timeout_s = int(timeout_s)
    if timeout_s < 0:
        raise ValueError(f"timeout_s must be >= 0, got {timeout_s}")

    pid_file = f"{meta}/{BOOTSTRAP_PID_FILENAME}"
    container_file = f"{meta}/{CONTAINER_FILENAME}"
    events_file = f"{meta}/events.ndjson"
    heartbeat_file = f"{meta}/heartbeat.ndjson"
    log_file = f"{meta}/last.log"
    settle_s = KILL_SETTLE_S
    return f"""\
runplz_run_id='{run_id}'
runplz_pid=""
runplz_container=""
if [ -r "{pid_file}" ]; then IFS= read -r runplz_pid < "{pid_file}" || true; fi
if [ -r "{container_file}" ]; then IFS= read -r runplz_container < "{container_file}" || true; fi

# Numeric comparison, not a string case: "0000001" is pid 1 to kill(1), and
# `kill -TERM 0` signals every process in our own group -- i.e. this script.
case "$runplz_pid" in ''|*[!0-9]*) runplz_pid="" ;; esac
if [ -n "$runplz_pid" ] && [ "$runplz_pid" -le 1 ]; then runplz_pid=""; fi

# A finished run's pid may already have been recycled by another job, so the
# pid fallback is only trusted while no terminal event has been recorded.
runplz_finished=0
if grep -Fq 'remote_command_exit' "{events_file}" 2>/dev/null; then
  runplz_finished=1
fi

runplz_scan=0
if [ -n "$runplz_run_id" ] && [ -d "{proc_root}" ]; then runplz_scan=1; fi

# Every *live* process this run started, found by the run id in its
# environment. This script does not carry the marker, so it can never signal
# itself. The `kill -0` filter keeps a dead pid from pinning the wait loop.
runplz_is_zombie() {{
  runplz_zstat=""
  [ -r "{proc_root}/$1/stat" ] || return 1
  IFS= read -r runplz_zstat < "{proc_root}/$1/stat" || return 1
  runplz_zrest=${{runplz_zstat##*) }}
  case "${{runplz_zrest%% *}}" in Z) return 0 ;; esac
  return 1
}}

runplz_run_pids() {{
  [ "$runplz_scan" = "1" ] || return 0
  for runplz_entry in "{proc_root}"/[0-9]*; do
    [ -r "$runplz_entry/environ" ] || continue
    runplz_cand=${{runplz_entry##*/}}
    kill -0 "$runplz_cand" 2>/dev/null || continue
    # A zombie answers `kill -0` but is already dead; counting it as alive
    # would pin the wait loop and force a pointless SIGKILL escalation.
    if runplz_is_zombie "$runplz_cand"; then continue; fi
    if tr '\\0' '\\n' < "$runplz_entry/environ" 2>/dev/null \
      | grep -qxF "{RUN_ID_ENV_VAR}=$runplz_run_id"; then
      printf '%s\n' "$runplz_cand"
    fi
  done
}}

runplz_pid_state() {{
  {_detached_process_state_shell()}printf '%s' "$runplz_state"
}}

runplz_pid_usable() {{
  [ -n "$runplz_pid" ] && [ "$runplz_finished" = "0" ]
}}

runplz_container_running() {{
  [ -n "$runplz_container" ] || return 1
  sudo docker inspect --format '{{{{.State.Running}}}}' "$runplz_container" 2>/dev/null \
    | grep -qx true
}}

runplz_alive() {{
  [ -n "$(runplz_run_pids)" ] && return 0
  if runplz_pid_usable; then
    case "$(runplz_pid_state)" in running) return 0 ;; esac
  fi
  runplz_container_running
}}

runplz_signal() {{
  for runplz_victim in $(runplz_run_pids); do
    kill -"$1" "$runplz_victim" 2>/dev/null || true
  done
  if runplz_pid_usable; then kill -"$1" "$runplz_pid" 2>/dev/null || true; fi
  if [ -n "$runplz_container" ]; then
    sudo docker kill --signal="$1" "$runplz_container" >/dev/null 2>&1 || true
  fi
}}

runplz_wait_until_dead() {{
  runplz_waited=0
  while [ "$runplz_waited" -lt "$1" ]; do
    runplz_alive || return 0
    sleep 1
    runplz_waited=$((runplz_waited + 1))
  done
  runplz_alive && return 1
  return 0
}}

runplz_initial="$(runplz_pid_state)"
runplz_alive_before=0
runplz_escalated=0
runplz_signalled=0
if runplz_alive; then
  runplz_alive_before=1
  runplz_signalled=1
  runplz_signal "{first_signal}"
  if ! runplz_wait_until_dead {timeout_s}; then
    if [ "{1 if escalate else 0}" = "1" ]; then
      runplz_escalated=1
      runplz_signal KILL
      runplz_wait_until_dead {settle_s} || true
    fi
  fi
fi

runplz_final="$(runplz_pid_state)"
runplz_alive_after=0
if runplz_alive; then runplz_alive_after=1; fi
runplz_survivors="$(runplz_run_pids | tr '\n' ' ' | sed 's/  *$//')"
if runplz_container_running; then runplz_container_state=running; \
elif [ -n "$runplz_container" ]; then runplz_container_state=stopped; \
else runplz_container_state=none; fi

if [ "$runplz_signalled" = "1" ]; then
  mkdir -p "{meta}" 2>/dev/null || true
  printf '{{"ts":"%s","run_id":"%s","event":"killed_by_user","escalated":%s}}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "{run_id}" "$runplz_escalated" \
    >> "{events_file}" 2>/dev/null || true
fi

runplz_gpu=""
if command -v nvidia-smi >/dev/null 2>&1; then
  runplz_gpu="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
    | tr '\n' ',' | sed 's/,$//')"
fi

printf '%s\n' '---SUMMARY---'
printf 'pid=%s\n' "${{runplz_pid:-}}"
printf 'container=%s\n' "${{runplz_container:-}}"
printf 'container_state=%s\n' "$runplz_container_state"
printf 'scan=%s\n' "$runplz_scan"
printf 'finished=%s\n' "$runplz_finished"
printf 'initial=%s\n' "$runplz_initial"
printf 'final=%s\n' "$runplz_final"
printf 'alive_before=%s\n' "$runplz_alive_before"
printf 'alive_after=%s\n' "$runplz_alive_after"
printf 'survivors=%s\n' "${{runplz_survivors:-}}"
printf 'signal=%s\n' '{first_signal}'
printf 'signalled=%s\n' "$runplz_signalled"
printf 'escalated=%s\n' "$runplz_escalated"
printf 'gpu_mem_used=%s\n' "${{runplz_gpu:-}}"
printf '%s\n' '---HEARTBEAT---'
tail -n 1 "{heartbeat_file}" 2>/dev/null || true
printf '%s\n' '---LOGTAIL---'
# Prefixed so a log line that happens to look like ---SECTION--- cannot be
# mistaken for one by the section parser.
tail -n 10 "{log_file}" 2>/dev/null | sed 's/^/| /' || true
printf '%s\n' '---END---'
"""


def _detached_process_state_shell() -> str:
    """Return shell that sets ``runplz_state`` for ``$runplz_pid``.

    Brev container images can be minimal and need not include procps. Linux
    ``/proc/<pid>/stat`` is available in the supported remote environments and
    exposes zombie state directly. If procfs is unavailable, ``kill -0`` still
    gives a conservative live/dead answer without making ``ps`` a dependency.
    """

    return (
        'runplz_state="missing"; '
        'if [ -n "$runplz_pid" ]; then '
        '  if [ -r "/proc/$runplz_pid/stat" ]; then '
        '    runplz_proc_stat=""; '
        '    IFS= read -r runplz_proc_stat < "/proc/$runplz_pid/stat" || true; '
        "    runplz_proc_tail=${runplz_proc_stat##*) }; "
        "    runplz_proc_state=${runplz_proc_tail%% *}; "
        '    case "$runplz_proc_state" in '
        '      Z) runplz_state="zombie" ;; '
        '      "") if kill -0 "$runplz_pid" 2>/dev/null; '
        '          then runplz_state="running"; else runplz_state="dead"; fi ;; '
        '      *) runplz_state="running" ;; '
        "    esac; "
        '  elif kill -0 "$runplz_pid" 2>/dev/null; then '
        '    runplz_state="running"; '
        "  else "
        '    runplz_state="dead"; '
        "  fi; "
        "fi; "
    )


def build_detached_status_probe(pid_file: str, events_file: Optional[str] = None) -> str:
    """Build a remote probe that distinguishes running, dead, and zombie PIDs."""

    started_probe = "runplz_started=0; "
    if events_file is not None:
        started_probe += (
            f"if grep -Fq 'remote_command_start' \"{events_file}\" 2>/dev/null; "
            "then runplz_started=1; fi; "
        )
    return (
        f"{started_probe}"
        'runplz_pid=""; '
        f'if [ -r "{pid_file}" ]; then IFS= read -r runplz_pid < "{pid_file}" || true; fi; '
        f"{_detached_process_state_shell()}"
        'printf \'%s %s %s\\n\' "$runplz_started" "$runplz_state" "$runplz_pid"'
    )


def inspect_detached_run(
    target: str,
    pid_file: str,
    *,
    events_file: Optional[str] = None,
    ssh_opts: Optional[SshOptions] = None,
) -> DetachedRunStatus:
    """Inspect one detached run without conflating zombies with live jobs."""

    probe = build_detached_status_probe(pid_file, events_file)
    try:
        result = subprocess.run(
            ["ssh", *ssh_cmd_opts(ssh_opts), target, probe],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return DetachedRunStatus(DetachedProcessState.UNKNOWN, False)
    if result.returncode != 0:
        return DetachedRunStatus(DetachedProcessState.UNKNOWN, False)

    parts = result.stdout.strip().split()
    if len(parts) < 2:
        # Accept the historical one-word probe output for compatibility with
        # callers that fake SSH at this boundary.
        legacy = {
            "alive": DetachedProcessState.RUNNING,
            "dead": DetachedProcessState.DEAD,
            "no-pid": DetachedProcessState.MISSING,
            "zombie": DetachedProcessState.ZOMBIE,
        }
        state = legacy.get(parts[0] if parts else "", DetachedProcessState.UNKNOWN)
        return DetachedRunStatus(state, False)
    try:
        state = DetachedProcessState(parts[1])
    except ValueError:
        state = DetachedProcessState.UNKNOWN
    try:
        pid = int(parts[2]) if len(parts) >= 3 and parts[2] else None
    except ValueError:
        pid = None
    return DetachedRunStatus(state, parts[0] == "1", pid)


def wait_for_detached_start(
    target: str,
    pid_file: str,
    events_file: str,
    *,
    timeout_s: float = DETACHED_START_TIMEOUT_S,
    poll_interval_s: float = DETACHED_START_POLL_INTERVAL_S,
    ssh_opts: Optional[SshOptions] = None,
) -> DetachedRunStatus:
    """Wait for the bootstrap's first lifecycle event or a terminal PID state."""

    deadline = time.monotonic() + timeout_s
    last = DetachedRunStatus(DetachedProcessState.UNKNOWN, False)
    while True:
        last = inspect_detached_run(
            target,
            pid_file,
            events_file=events_file,
            ssh_opts=ssh_opts,
        )
        if last.started or last.process_state in {
            DetachedProcessState.MISSING,
            DetachedProcessState.DEAD,
            DetachedProcessState.ZOMBIE,
        }:
            return last
        if time.monotonic() >= deadline:
            return last
        time.sleep(poll_interval_s)


# How often the watchdog wakes to check for silence. Bounded well under any
# useful inactivity budget so expiry is noticed promptly, and floored so a
# small budget cannot turn the monitor into a polling loop.
INACTIVITY_POLL_INTERVAL_S = 60


def build_inactivity_probe(
    remote_run: RemoteRunContext, container_name: Optional[str] = None
) -> str:
    """Shell that reports how long the *application* has been silent.

    Deliberately not the heartbeat. `runplz_heartbeat_loop` runs on a timer as
    a background job of the wrapper shell, independent of the user's command,
    so its mtime stays fresh while the job is wedged — proving liveness, which
    is the exact signal issue #122 says failed to distinguish a deadlock.

    What does move only when the application moves: the driver log, and the
    outputs directory. The directory is checked non-recursively — a new
    checkpoint file updates its mtime — so this stays one cheap `stat` rather
    than a walk of a multi-gigabyte tree every minute.

    ``container_name`` adds the third signal, for VM+docker mode (#153). That
    path writes no driver log — the application's output goes to `docker logs`
    — so without this leg the probe would watch only the outputs directory and
    call a job that prints steadily but writes no files stalled. An empty
    LogPath (a log driver other than `json-file`) stats to 0, which reads as
    "never written" and loses the `max` to the outputs directory, so an unusual
    driver degrades to the weaker signal rather than inventing a stall.

    Both timestamps and `now` come from the remote, so idle time is computed
    entirely in the box's own clock; skew against the laptop cannot invent a
    stall or hide one.
    """
    driver_log = f"{remote_run.meta_shell}/run_driver.log"
    outputs_shell = remote_run.out_shell
    # `stat -c` is GNU, `stat -f` is BSD. Try both, then fall back to 0, which
    # reads as "never written" rather than as fresh activity.
    # `printf '%s\\n' '---NAME---'` rather than `printf '---NAME---\\n'`:
    # a leading `--` is parsed as options by printf, which the section
    # markers are made entirely of.
    docker_leg = ""
    if container_name is not None:
        quoted = shlex.quote(container_name)
        docker_leg = (
            "printf '%s\\n' '---DLOG---'; "
            "runplz_mtime "
            f"\"$(sudo docker inspect --format '{{{{.LogPath}}}}' {quoted} 2>/dev/null)\"; "
            "printf '%s\\n' ''; "
        )
    return (
        "runplz_mtime() { "
        'stat -c %Y "$1" 2>/dev/null || stat -f %m "$1" 2>/dev/null || printf 0; '
        "}; "
        "printf '%s\\n' '---NOW---'; date +%s; "
        "printf '%s\\n' '---LOG---'; "
        f"runplz_mtime \"{driver_log}\"; printf '%s\\n' ''; "
        "printf '%s\\n' '---OUT---'; "
        f"runplz_mtime \"{outputs_shell}\"; printf '%s\\n' ''; "
        f"{docker_leg}"
        "printf '%s\\n' '---END---'"
    )


def seconds_since_activity(probe_output: str) -> Optional[float]:
    """Idle seconds from :func:`build_inactivity_probe` output, or None.

    None means "cannot tell" — a truncated or unparseable probe. The caller
    must treat that as *not* a stall: an unreadable probe is a fact about the
    ssh round trip, and killing a healthy job over one would be worse than the
    deadlock the watchdog exists to catch.
    """
    sections = parse_probe_sections(probe_output)

    def _stamp(name: str) -> Optional[int]:
        raw = (sections.get(name) or "").strip().splitlines()
        try:
            return int(raw[0]) if raw else None
        except ValueError:
            return None

    now = _stamp("NOW")
    if now is None:
        return None
    legs = (_stamp("LOG"), _stamp("OUT"), _stamp("DLOG"))
    latest = max((t for t in legs if t is not None), default=None)
    if latest is None:
        return None
    return max(0.0, float(now - latest))


def _stall_context_shell(remote_run: RemoteRunContext, proc_root: str = "/proc") -> str:
    """Every process of this run, with its state, plus accelerator use.

    Not `ps`: `tasks/lessons.md` records that remote process tracking has to
    work in minimal container images and must not assume procps. This uses the
    same `/proc/<pid>/environ` run-id marker `build_kill_command` uses, which
    is exact — it finds workers orphaned to init and cannot match another run.

    Unlike the kill scan, zombies are *kept*. That scan skips them because
    counting a zombie as alive would pin its wait loop; here they are the
    finding — the stall in #122 was one child wedged in `_finalize_join` with
    four zombie siblings.
    """
    return (
        "printf '%s\\n' 'run processes (pid state):'; "
        f'for runplz_entry in "{proc_root}"/[0-9]*; do '
        '  [ -r "$runplz_entry/environ" ] || continue; '
        "  runplz_cand=${runplz_entry##*/}; "
        f"  if tr '\\0' '\\n' < \"$runplz_entry/environ\" 2>/dev/null "
        f'    | grep -qxF "{RUN_ID_ENV_VAR}={remote_run.run_id}"; then '
        '    runplz_st="?"; '
        '    if [ -r "$runplz_entry/stat" ]; then '
        '      IFS= read -r runplz_line < "$runplz_entry/stat" || true; '
        "      runplz_rest=${runplz_line##*) }; runplz_st=${runplz_rest%% *}; "
        "    fi; "
        '    printf \'%s %s\\n\' "$runplz_cand" "$runplz_st"; '
        "  fi; "
        "done; "
        "printf '%s\\n' 'accelerators:'; "
        "nvidia-smi --query-gpu=index,utilization.gpu,memory.used "
        "--format=csv,noheader 2>/dev/null || printf '%s\\n' '(no nvidia-smi)'"
    )


def detached_launch_diagnostics(
    target: str,
    remote_run: RemoteRunContext,
    *,
    ssh_opts: Optional[SshOptions] = None,
    include_stall_context: bool = False,
) -> str:
    """Fetch compact process and lifecycle context for a failed startup.

    ``include_stall_context`` adds this run's process table and accelerator
    use, which a stall needs and a failed launch does not — at launch failure
    there are no processes to table and no GPU work to report. Opt-in so the
    launch path's output is unchanged.
    """

    meta = remote_run.meta_shell
    pid_file = f"{meta}/{BOOTSTRAP_PID_FILENAME}"
    driver_log = f"{meta}/run_driver.log"
    command = (
        'runplz_pid=""; '
        f'if [ -r "{pid_file}" ]; then IFS= read -r runplz_pid < "{pid_file}" || true; fi; '
        f"{_detached_process_state_shell()}"
        "printf '%s\\n' 'detached bootstrap diagnostics:'; "
        "printf 'pid: %s\\n' \"${runplz_pid:-missing}\"; "
        "printf 'process state: %s\\n' \"$runplz_state\"; "
        "printf '%s\\n' 'recent events:'; "
        f'tail -n 10 "{remote_run.events_shell}" 2>/dev/null || true; '
        "printf '%s\\n' 'recent heartbeats:'; "
        f'tail -n 3 "{remote_run.heartbeat_shell}" 2>/dev/null || true; '
        "printf '%s\\n' 'driver log:'; "
        f'tail -n {FAILURE_TAIL_LINES} "{driver_log}" 2>/dev/null || true'
    )
    if include_stall_context:
        command += "; " + _stall_context_shell(remote_run)
    return ssh_capture(target, command, ssh_opts=ssh_opts).rstrip()


def build_detached_log_command(pid_file: str, log_file: str) -> str:
    """Build a log follower that exits when its bootstrap is dead or zombie."""

    return (
        "set -u; "
        'runplz_pid=""; '
        f'if [ -r "{pid_file}" ]; then IFS= read -r runplz_pid < "{pid_file}" || true; fi; '
        'if [ -z "$runplz_pid" ]; then exit 1; fi; '
        f'tail -n +1 -F "{log_file}" & '
        "runplz_tail_pid=$!; "
        "runplz_stop_tail() { "
        '  kill "$runplz_tail_pid" >/dev/null 2>&1 || true; '
        '  wait "$runplz_tail_pid" >/dev/null 2>&1 || true; '
        "}; "
        "trap 'runplz_stop_tail' EXIT HUP INT TERM; "
        "while true; do "
        f"  {_detached_process_state_shell()}"
        '  case "$runplz_state" in missing|dead|zombie) break ;; esac; '
        "  sleep 1; "
        "done; "
        # Give tail one final polling interval to flush lines written during
        # bootstrap shutdown before the EXIT trap stops it.
        "sleep 1"
    )


def launch_detached_and_wait(
    *,
    target: str,
    wrapped_command: str,
    remote_run: Optional["RemoteRunContext"] = None,
    max_runtime_seconds: Optional[int] = None,
    max_inactivity_seconds: Optional[int] = None,
    inactivity_action: str = "diagnose",
    ssh_opts: Optional[SshOptions] = None,
    max_reconnects: int = 20,
) -> int:
    """Launch a nohup-protected bash command and stream/wait locally.

    Core SSH-drop-survival path for container-mode and native backends.
    Before this helper, ``run_container_mode`` and ``run_native`` ran
    the bootstrap as a foreground ssh command whose stdout pipeline
    ended with ``tee`` — which meant any ssh drop SIGPIPEd tee and
    cascaded SIGPIPE / BrokenPipeError back through the whole pipeline,
    killing training.

    The launching shell ignores SIGHUP before it forks, closing the race before
    ``nohup`` can install the same disposition. The child wrapper retains that
    ignore, stdin comes from ``/dev/null``, and stdout/stderr go only to a
    per-run driver log. Avoiding ``setsid`` keeps ``$!`` tied to bash on
    implementations that fork for process-group leaders.

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
                [
                    "ssh",
                    *ssh_cmd_opts(ssh_opts),
                    target,
                    f"bash -lc {shlex.quote(wrapped_command)}",
                ],
                timeout=max_runtime_seconds,
            )
        except subprocess.TimeoutExpired:
            raise_for_runtime_cap(
                target,
                max_runtime_seconds,
                container_name=None,
                ssh_opts=ssh_opts,
                remote_run=remote_run,
            )
        return r.returncode

    meta = remote_run.meta_shell
    pid_file = f"{meta}/{BOOTSTRAP_PID_FILENAME}"
    log_file = remote_run.last_log_shell
    events_file = remote_run.events_shell
    launcher = build_detached_launcher(remote_run, wrapped_command)
    # Launch ssh returns quickly once the detached job is running + PID
    # recorded. Anything that follows in this function is local polling.
    #
    # A transport failure here is ambiguous: the launcher may have run and
    # detached before the connection dropped. So the retry asks the remote
    # first — if a bootstrap pid or a start event exists, the job is already
    # running and a second launch would double it (issue #84).
    retry_on_transport_failure(
        lambda: ssh_exec(target, launcher, ssh_opts=ssh_opts),
        label=f"launch detached run {remote_run.run_id}",
        can_retry=lambda: (
            not detached_run_started(target, pid_file, events_file, ssh_opts=ssh_opts)
        ),
    )

    startup = wait_for_detached_start(
        target,
        pid_file,
        events_file,
        ssh_opts=ssh_opts,
    )
    terminal_startup_failure = startup.process_state in {
        DetachedProcessState.MISSING,
        DetachedProcessState.DEAD,
        DetachedProcessState.ZOMBIE,
    }
    if not startup.started and terminal_startup_failure:
        _record_remote_event(
            target,
            remote_run,
            "bootstrap_launch_failed",
            ssh_opts=ssh_opts,
            process_state=startup.process_state.value,
            pid=startup.pid,
        )
        diagnostics = detached_launch_diagnostics(target, remote_run, ssh_opts=ssh_opts)
        print(
            f"+ detached bootstrap failed to start on {target} "
            f"(state={startup.process_state.value})\n{diagnostics}",
            flush=True,
        )
        return 1
    if not startup.started:
        print(
            f"+ detached bootstrap startup not yet confirmed on {target} "
            f"(state={startup.process_state.value}); entering resilient monitoring",
            flush=True,
        )

    return tail_and_wait_for_detached(
        target=target,
        pid_file=pid_file,
        log_file=log_file,
        events_file=events_file,
        max_runtime_seconds=max_runtime_seconds,
        max_inactivity_seconds=max_inactivity_seconds,
        inactivity_action=inactivity_action,
        max_reconnects=max_reconnects,
        ssh_opts=ssh_opts,
        # So a runtime-cap timeout stops this run precisely instead of
        # pkill-ing every runplz bootstrap on the box.
        remote_run=remote_run,
    )


def _check_inactivity(
    target: str,
    remote_run: Optional[RemoteRunContext],
    max_inactivity_seconds: Optional[int],
    inactivity_action: str,
    *,
    already_reported: bool,
    container_name: Optional[str] = None,
    ssh_opts: Optional[SshOptions] = None,
) -> tuple:
    """One watchdog tick. Returns ``(stalled_now, reported)``.

    An unreadable probe is not a stall. It is a fact about the ssh round
    trip, and terminating a healthy job over one would be a worse failure
    than the deadlock this exists to catch — so anything short of a
    confidently-measured silence leaves the run alone.

    ``container_name`` is set in VM+docker mode, where the application's
    output is the container's log rather than a driver log on the box.
    """
    if remote_run is None or max_inactivity_seconds is None:
        return False, already_reported
    try:
        idle = seconds_since_activity(
            ssh_capture(
                target,
                build_inactivity_probe(remote_run, container_name),
                ssh_opts=ssh_opts,
            )
        )
    except Exception as exc:  # noqa: BLE001
        print(f"+ warning: inactivity probe failed: {type(exc).__name__}: {exc}", flush=True)
        return False, already_reported
    if idle is None or idle < max_inactivity_seconds:
        # Work resumed (or was never absent): re-arm, so a later stall is
        # reported as its own episode rather than suppressed by this one.
        return False, False
    if already_reported:
        return True, True

    _record_remote_event(
        target,
        remote_run,
        "remote_command_stalled",
        ssh_opts=ssh_opts,
        idle_seconds=int(idle),
        threshold_seconds=int(max_inactivity_seconds),
        action=inactivity_action,
    )
    print(
        f"+ WARNING: no application output on {target} for {int(idle)}s "
        f"(max_inactivity_seconds={int(max_inactivity_seconds)}). The process is "
        f"alive — the runplz heartbeat proves only that — so this is silence, "
        f"not necessarily a hang.",
        flush=True,
    )
    try:
        print(
            detached_launch_diagnostics(
                target, remote_run, ssh_opts=ssh_opts, include_stall_context=True
            ),
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"+ warning: stall diagnostics failed: {type(exc).__name__}: {exc}", flush=True)

    if inactivity_action == "terminate":
        print(f"+ inactivity_action=terminate: stopping run {remote_run.run_id}", flush=True)
        _terminate_stalled_run(target, remote_run, ssh_opts=ssh_opts)
    else:
        print("+ inactivity_action=diagnose: leaving the run alone", flush=True)
    return True, True


def _terminate_stalled_run(
    target: str,
    remote_run: RemoteRunContext,
    *,
    ssh_opts: Optional[SshOptions] = None,
) -> None:
    """Stop exactly this run, and report what it actually stopped.

    `build_kill_command` already emits a bounded SUMMARY / HEARTBEAT / LOGTAIL
    block — including gpu_mem_used — which `raise_for_runtime_cap` captures and
    then throws away. Printing it costs nothing and is precisely the evidence
    an operator needs about a stall.
    """
    cleanup = build_kill_command(remote_run.meta_shell, run_id=remote_run.run_id, timeout_s=5)
    try:
        summary = ssh_capture(target, cleanup, ssh_opts=ssh_opts)
    except Exception as exc:  # noqa: BLE001
        print(f"+ warning: failed to stop stalled run: {type(exc).__name__}: {exc}", flush=True)
        return
    if summary.strip():
        print(summary.rstrip(), flush=True)


def tail_and_wait_for_detached(
    *,
    target: str,
    pid_file: str,
    log_file: str,
    events_file: str,
    max_runtime_seconds: Optional[int] = None,
    max_inactivity_seconds: Optional[int] = None,
    inactivity_action: str = "diagnose",
    max_reconnects: int = 20,
    ssh_opts: Optional[SshOptions] = None,
    remote_run: Optional[RemoteRunContext] = None,
) -> int:
    """Stream log_file via ssh ``tail -F`` and return remote exit code.

    Mirrors ``stream_and_wait``'s reconnect pattern but uses a PID file
    + events file instead of docker commands for the "is the job still
    alive" and "what was the exit code" checks.

    If ssh drops mid-stream but the remote PID is still alive, reconnect
    and keep tailing. If the PID is gone, read the exit code from the
    events file and return it.

    ``max_inactivity_seconds`` adds an opt-in watchdog on *application*
    silence, independent of the total-runtime cap (#122). Off by default,
    because build, download and compile phases are legitimately quiet and
    silence alone must not change what an existing job does.
    """
    print(
        "+ streaming detached remote log (resilient to ssh reconnects)",
        flush=True,
    )
    started = time.monotonic()
    watching = max_inactivity_seconds is not None and remote_run is not None
    # Fires once per stall, re-armed when work resumes, so a job that is quiet
    # for ten hours warns once rather than every poll interval.
    stall_reported = False

    def _remaining_s() -> Optional[float]:
        bounds = []
        if max_runtime_seconds is not None:
            bounds.append(max(1.0, max_runtime_seconds - (time.monotonic() - started)))
        if watching:
            # The tail blocks until the job ends, so without a bound there is
            # no moment at which silence could be noticed. This is what wakes
            # the loop; it is not a timeout on anything.
            bounds.append(float(min(INACTIVITY_POLL_INTERVAL_S, max_inactivity_seconds)))
        return min(bounds) if bounds else None

    def _runtime_cap_reached() -> bool:
        return (
            max_runtime_seconds is not None and (time.monotonic() - started) >= max_runtime_seconds
        )

    reconnects = 0
    while True:
        cmd = build_detached_log_command(pid_file, log_file)
        try:
            r = subprocess.run(
                ["ssh", *ssh_cmd_opts(ssh_opts), target, cmd],
                timeout=_remaining_s(),
            )
        except subprocess.TimeoutExpired:
            # Two things end the tail early, and they are not the same event.
            # Only an exhausted runtime budget is the cap; otherwise this is
            # the watchdog's own wake-up.
            if _runtime_cap_reached() or not watching:
                raise_for_runtime_cap(
                    target,
                    max_runtime_seconds,
                    container_name=None,
                    ssh_opts=ssh_opts,
                    remote_run=remote_run,
                )
            if not remote_pid_alive(target, pid_file, ssh_opts=ssh_opts):
                break
            stalled, stall_reported = _check_inactivity(
                target,
                remote_run,
                max_inactivity_seconds,
                inactivity_action,
                already_reported=stall_reported,
                ssh_opts=ssh_opts,
            )
            if stalled and inactivity_action == "terminate":
                # Deliberately not raise_for_runtime_cap's shape: that raises
                # out of dispatch_to_target before rsync_down, losing whatever
                # the job produced. #122 requires outputs preserved, so stop
                # the run and let the normal completion path collect them.
                break
            # A watchdog tick is not a dropped connection. Falling through
            # would spend one of `max_reconnects` per poll and silently kill
            # the live log stream on any legitimately quiet job.
            continue
        if not remote_pid_alive(target, pid_file, ssh_opts=ssh_opts):
            break
        if max_runtime_seconds is not None and (time.monotonic() - started) >= max_runtime_seconds:
            raise_for_runtime_cap(
                target,
                max_runtime_seconds,
                container_name=None,
                ssh_opts=ssh_opts,
                remote_run=remote_run,
            )
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
    while remote_pid_alive(target, pid_file, ssh_opts=ssh_opts):
        if max_runtime_seconds is not None and (time.monotonic() - started) >= max_runtime_seconds:
            raise_for_runtime_cap(
                target,
                max_runtime_seconds,
                container_name=None,
                ssh_opts=ssh_opts,
                remote_run=remote_run,
            )
        time.sleep(min(30, HEARTBEAT_INTERVAL_S))

    return read_remote_exit_code(target, events_file, ssh_opts=ssh_opts)


def remote_pid_alive(target: str, pid_file: str, *, ssh_opts: Optional[SshOptions] = None) -> bool:
    """Return True if a PID is live, treating zombies as completed.

    Conservative on ssh errors: if we can't reach the box right now,
    assume the job is still alive so the caller keeps polling instead
    of prematurely declaring the job done. A real dead job will surface
    next poll once ssh recovers.
    """
    state = inspect_detached_run(target, pid_file, ssh_opts=ssh_opts).process_state
    return state in {DetachedProcessState.RUNNING, DetachedProcessState.UNKNOWN}


def read_remote_exit_code(
    target: str, events_file: str, *, ssh_opts: Optional[SshOptions] = None
) -> int:
    """Parse the last ``remote_command_exit`` entry from events.ndjson.

    Returns 1 when the events file is missing, unreadable, or has no
    exit entry yet — treating "unknown" as failure keeps a silent
    exit-code regression from masquerading as success.
    """
    # Same expansion rule: ``events_file`` is a $HOME-relative path.
    probe = f"grep -F 'remote_command_exit' \"{events_file}\" 2>/dev/null | tail -n 1 || true"
    try:
        r = subprocess.run(
            ["ssh", *ssh_cmd_opts(ssh_opts), target, probe],
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


def stream_and_wait(
    target: str,
    container_name: str,
    max_reconnects: int = 20,
    max_runtime_seconds: Optional[int] = None,
    max_inactivity_seconds: Optional[int] = None,
    inactivity_action: str = "diagnose",
    remote_run: Optional[RemoteRunContext] = None,
    ssh_opts: Optional[SshOptions] = None,
) -> int:
    """Stream container logs and return its exit code.

    Reconnect-tolerant: if ssh drops mid-stream we re-attach with
    `--tail 0` to pick up where we left off, then call `docker wait` for
    the exit code. Gives up after `max_reconnects` consecutive reconnect
    attempts. Wall-clock cap from `max_runtime_seconds` tracked across
    reconnects so a streaming job can't dodge it.

    ``max_inactivity_seconds`` is the same opt-in application-silence
    watchdog the detached monitor runs (#122), on the mode that is actually
    the default — `use_docker=True` on ssh/gcp/aws all land here, and until
    #153 this function did not take the parameter at all, so setting it did
    nothing at all on a default config.
    """
    print(f"+ streaming logs from {container_name} (resilient to reconnects)", flush=True)
    started = time.monotonic()
    watching = max_inactivity_seconds is not None and remote_run is not None
    # Fires once per stall, re-armed when work resumes, so a job that is quiet
    # for ten hours warns once rather than every poll interval.
    stall_reported = False

    def _runtime_remaining_s() -> Optional[float]:
        if max_runtime_seconds is None:
            return None
        return max(1.0, max_runtime_seconds - (time.monotonic() - started))

    def _remaining_s() -> Optional[float]:
        bounds = []
        runtime_left = _runtime_remaining_s()
        if runtime_left is not None:
            bounds.append(runtime_left)
        if watching:
            # `docker logs -f` blocks until the container ends, so without a
            # bound there is no moment at which silence could be noticed. This
            # is what wakes the loop; it is not a timeout on anything.
            bounds.append(float(min(INACTIVITY_POLL_INTERVAL_S, max_inactivity_seconds)))
        return min(bounds) if bounds else None

    def _runtime_cap_reached() -> bool:
        return (
            max_runtime_seconds is not None and (time.monotonic() - started) >= max_runtime_seconds
        )

    tail = "all"
    reconnects = 0
    while True:
        cmd = f"sudo docker logs -f --tail {tail} {container_name}"
        try:
            r = subprocess.run(
                ["ssh", *ssh_cmd_opts(ssh_opts), target, cmd],
                timeout=_remaining_s(),
            )
        except subprocess.TimeoutExpired:
            # Two things end the stream early, and they are not the same
            # event. Only an exhausted runtime budget is the cap; otherwise
            # this is the watchdog's own wake-up.
            if _runtime_cap_reached() or not watching:
                raise_for_runtime_cap(
                    target,
                    max_runtime_seconds,
                    container_name=container_name,
                    ssh_opts=ssh_opts,
                    remote_run=remote_run,
                )
            if not container_running(target, container_name, ssh_opts=ssh_opts):
                break
            stalled, stall_reported = _check_inactivity(
                target,
                remote_run,
                max_inactivity_seconds,
                inactivity_action,
                already_reported=stall_reported,
                container_name=container_name,
                ssh_opts=ssh_opts,
            )
            if stalled and inactivity_action == "terminate":
                # Deliberately not raise_for_runtime_cap's shape: that raises
                # out of dispatch_to_target, and `docker rm -f` in its finally
                # then takes the container's logs with it. Stopping the run and
                # falling through to `docker wait` keeps the normal completion
                # path — and the outputs sync — intact, matching the detached
                # monitor.
                break
            # A watchdog tick is not a dropped connection. Falling through
            # would spend one of `max_reconnects` per poll and silently kill
            # the live log stream on any legitimately quiet job. Reattaching
            # with `--tail 0` for the same reason a reconnect does: `all`
            # would reprint the entire log every poll.
            tail = "0"
            continue
        running = container_running(target, container_name, ssh_opts=ssh_opts)
        if not running:
            break
        if max_runtime_seconds is not None and (time.monotonic() - started) >= max_runtime_seconds:
            raise_for_runtime_cap(
                target,
                max_runtime_seconds,
                container_name=container_name,
                ssh_opts=ssh_opts,
                remote_run=remote_run,
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
            ["ssh", *ssh_cmd_opts(ssh_opts), target, f"sudo docker wait {container_name}"],
            capture_output=True,
            text=True,
            # The runtime budget only. `_remaining_s()` is the min of that and
            # the watchdog's poll interval, and there is nothing to poll here —
            # using it would turn a 60s tick into a spurious cap raise on a
            # container this call is merely waiting on.
            timeout=_runtime_remaining_s(),
        )
    except subprocess.TimeoutExpired:
        raise_for_runtime_cap(
            target,
            max_runtime_seconds,
            container_name=container_name,
            ssh_opts=ssh_opts,
            remote_run=remote_run,
        )
    try:
        return int(r.stdout.strip() or "1")
    except ValueError:
        return 1


def _docker_inspect(
    target: str,
    container_name: str,
    fmt: str,
    *,
    ssh_opts: Optional[SshOptions] = None,
    timeout_s: int = 30,
):
    """Ask the remote docker about one container.

    Returns ``(returncode, stdout, stderr)``, or ``None`` when the question
    could not be asked at all — a hung ssh or a transport failure. Both
    callers treat "could not ask" as the pessimistic answer.
    """
    try:
        r = subprocess.run(
            [
                "ssh",
                *ssh_cmd_opts(ssh_opts),
                target,
                f"sudo docker inspect --format '{fmt}' {container_name}",
            ],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if is_ssh_transport_failure(r.returncode):
        return None
    return (r.returncode, r.stdout or "", r.stderr or "")


def container_running(
    target: str, container_name: str, *, ssh_opts: Optional[SshOptions] = None
) -> bool:
    # Treat ssh hangs / errors as "assume still running" so the caller keeps
    # retrying the log stream instead of giving up.
    probed = _docker_inspect(target, container_name, "{{.State.Running}}", ssh_opts=ssh_opts)
    if probed is None:
        return True
    returncode, stdout, _stderr = probed
    if returncode != 0:
        return True
    return stdout.strip() == "true"


# --- failure context -----------------------------------------------------


def fetch_failure_tail(
    *,
    target: str,
    container_name: Optional[str],
    remote_run: Optional[RemoteRunContext] = None,
    ssh_opts: Optional[SshOptions] = None,
) -> str:
    """Fetch the last N lines of remote output for a failed run.

    - VM + docker path (`container_name` set): `docker logs --tail N <name>`.
      Docker persists logs until `docker rm`, so this works post-crash.
    - container-mode / native paths: include the detached launcher driver
      log and the bootstrap's combined stdout/stderr file.

    Best-effort: return a diagnostic string rather than raising so we
    never mask the real error in the caller.
    """
    try:
        if container_name is not None:
            cmd = f"sudo docker logs --tail {FAILURE_TAIL_LINES} {container_name} 2>&1"
        else:
            if remote_run is None:
                cmd = (
                    f'if [ -f "{_remote_last_log_shell(None)}" ]; then '
                    f'tail -n {FAILURE_TAIL_LINES} "{_remote_last_log_shell(None)}"; '
                    f"fi"
                )
            else:
                driver_log = f"{remote_run.meta_shell}/run_driver.log"
                cmd = (
                    "printf '%s\\n' 'detached driver log:'; "
                    f'tail -n {FAILURE_TAIL_LINES} "{driver_log}" 2>/dev/null || true; '
                    "printf '%s\\n' 'bootstrap log:'; "
                    f'tail -n {FAILURE_TAIL_LINES} "{remote_run.last_log_shell}" '
                    "2>/dev/null || true"
                )
        out = ssh_capture(target, cmd, ssh_opts=ssh_opts)
        return (out or "").rstrip()
    except Exception as exc:  # noqa: BLE001
        return f"[runplz: could not fetch remote log tail — {type(exc).__name__}: {exc}]"


def raise_for_runtime_cap(
    target: str,
    cap_s,
    container_name,
    *,
    ssh_opts: Optional[SshOptions] = None,
    remote_run: Optional[RemoteRunContext] = None,
):
    """Shared timeout-path cleanup + raise for issue #16.

    container_name: set for VM+docker mode (kill the container with docker kill);
    None for container-mode / native.

    When the run context is known, stop exactly this run's processes via
    :func:`build_kill_command`. The fallback -- ``pkill -f runplz._bootstrap``
    -- matches on a cmdline substring, so on a box running two jobs it takes
    the innocent one down too; it is only used when we have no run to scope to.

    Best-effort cleanup: if the kill ssh hangs or fails, still raise — the
    on_finish action in the caller's finally block will nuke the box anyway.
    """
    if container_name is not None:
        cleanup = f"sudo docker kill {container_name}"
    elif remote_run is not None:
        cleanup = build_kill_command(
            remote_run.meta_shell,
            run_id=remote_run.run_id,
            timeout_s=5,
        )
    else:
        cleanup = "pkill -f 'runplz._bootstrap' || true"
    try:
        subprocess.run(
            ["ssh", *ssh_cmd_opts(ssh_opts), target, cleanup],
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


# --- Shared VM dispatch + lifecycle --------------------------------------
#
# Every backend that ends up with an ssh-reachable box runs the same
# sequence, and the provisioning ones wrap it in the same try/finally. The
# pieces below are that shared middle, so a new cloud driver only has to
# answer three questions: how do I create a box, what target do I hand
# over, and how do I tear it down.

CLEANUP_SIGNALS = (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)


class OrchestratorKilled(RuntimeError):
    """Raised in place of a termination signal so cleanup still runs."""


@contextlib.contextmanager
def orchestrator_signal_cleanup(label: str):
    """Turn termination signals into an exception so a `finally` can run.

    Without this, `kill -TERM <runplz pid>` exits cleanly while leaving a
    freshly provisioned box running — no teardown, no on_finish. A leaked
    8xA100 adds up fast (issue #38).

    Only works on the main thread (``signal.signal`` is main-thread-only).
    Off-main — a test runner worker, say — the handlers are not installed and
    cleanup degrades to Ctrl-C, which is acceptable: signal-driven teardown is
    a main-process concern anyway.
    """
    previous = {}

    def _handler(signum, _frame):
        signame = signal.Signals(signum).name
        print(
            f"+ runplz received {signame} — triggering cleanup for {label!r}",
            flush=True,
        )
        raise OrchestratorKilled(
            f"runplz orchestrator killed by {signame}; cleaning up {label!r} before exit."
        )

    try:
        for sig in CLEANUP_SIGNALS:
            try:
                previous[sig] = signal.signal(sig, _handler)
            except (ValueError, OSError):
                # Not the main thread, or unsupported on this platform.
                pass
        yield
    finally:
        for sig, prev in previous.items():
            try:
                signal.signal(sig, prev)
            except (ValueError, OSError):
                pass


@dataclass
class DispatchResult:
    """A record of what one dispatch did.

    Not a cleanup handle — `dispatch_to_target` removes its own container and
    captures its own failure tail before returning, because both have to
    happen before a provisioning caller tears the box down. This is for
    callers that want to report on the run.
    """

    exit_code: Optional[int]
    container_name: Optional[str]
    remote_run: Optional[RemoteRunContext]
    failure_tail: str = ""


def dispatch_to_target(
    *,
    app,
    function,
    args: list,
    kwargs: dict,
    target: str,
    backend: str,
    outputs_dir: str = "out",
    mode: str = "docker",
    max_runtime_seconds: Optional[int] = None,
    max_inactivity_seconds: Optional[int] = None,
    inactivity_action: str = "diagnose",
    ssh_opts: Optional[SshOptions] = None,
) -> DispatchResult:
    """Run one function on an already-reachable box, start to finish.

    Stages the repo, probes preconditions, runs the job, streams its output
    and brings the outputs back. Raises ``RuntimeError`` with a log tail if
    the remote command exited non-zero.

    ``mode`` picks the runner:

    - ``"docker"``   build/pull an image on the box and run the bootstrap in
      a container. The default, and what ssh/gcp/aws use.
    - ``"native"``   install a python venv on the box and run the bootstrap
      directly. No docker involved.
    - ``"container"`` the box *is* the user's image already (brev's
      container mode), so the Image DSL ops are applied inline over ssh and
      the bootstrap is invoked without docker-in-docker.

    Owns its own container cleanup and failure-tail capture, because both have
    to happen before a caller tears the box down — afterwards the logs are
    gone. Provisioning backends wrap this in their own try/finally for the box
    itself; see :func:`run_on_provisioned_vm`.
    """
    repo = app.require_repo_root(context="dispatch_to_target()")
    host_out = (repo / outputs_dir).resolve()
    host_out.mkdir(parents=True, exist_ok=True)

    write_local_ssh_options(host_out, ssh_opts)

    remote_run = make_remote_run_context(
        backend=backend,
        target=target,
        function_name=function.name,
    )
    prepare_remote_run(
        target,
        remote_run,
        manifest=build_remote_run_manifest(
            remote_run=remote_run,
            repo=repo,
            outputs_dir=outputs_dir,
            args=args,
            kwargs=kwargs,
            env=function.env,
        ),
        ssh_opts=ssh_opts,
    )

    # Pre-built images (and minimal cloud images) need not ship rsync.
    ensure_remote_rsync(target, ssh_opts=ssh_opts)
    rsync_up(repo, target, outputs_dir=outputs_dir, remote_run=remote_run, ssh_opts=ssh_opts)

    # Probe declared preconditions (issue #56) before bootstrap, so a
    # misprovisioned box fails fast instead of burning paid GPU minutes.
    check_preconditions(target, function.preconditions, ssh_opts=ssh_opts)

    rel_script = Path(function.module_file).resolve().relative_to(repo)

    if mode not in ("docker", "native", "container"):
        raise ValueError(f"mode must be 'docker', 'native' or 'container'; got {mode!r}")

    container_name: Optional[str] = None
    exit_code: Optional[int] = None
    failure_tail = ""
    synced = False
    try:
        if mode == "container":
            exit_code = run_container_mode(
                target=target,
                function=function,
                rel_script=str(rel_script),
                args=args,
                kwargs=kwargs,
                remote_run=remote_run,
                max_runtime_seconds=max_runtime_seconds,
                max_inactivity_seconds=max_inactivity_seconds,
                inactivity_action=inactivity_action,
                ssh_opts=ssh_opts,
            )
        elif mode == "docker":
            ensure_docker(target, ssh_opts=ssh_opts)
            gpu_flag = "--gpus all" if remote_has_nvidia(target, ssh_opts=ssh_opts) else ""
            container_name = make_container_name(function.name)
            build_image(target, function.image, remote_run=remote_run, ssh_opts=ssh_opts)
            run_container_detached(
                target=target,
                container_name=container_name,
                function=function,
                rel_script=str(rel_script),
                args=args,
                kwargs=kwargs,
                gpu_flag=gpu_flag,
                app_name=app.name,
                remote_run=remote_run,
                ssh_opts=ssh_opts,
            )
            exit_code = stream_and_wait(
                target,
                container_name,
                max_runtime_seconds=max_runtime_seconds,
                max_inactivity_seconds=max_inactivity_seconds,
                inactivity_action=inactivity_action,
                remote_run=remote_run,
                ssh_opts=ssh_opts,
            )
        else:
            exit_code = run_native(
                target=target,
                function=function,
                rel_script=str(rel_script),
                args=args,
                kwargs=kwargs,
                has_nvidia=remote_has_nvidia(target, ssh_opts=ssh_opts),
                remote_run=remote_run,
                max_runtime_seconds=max_runtime_seconds,
                max_inactivity_seconds=max_inactivity_seconds,
                inactivity_action=inactivity_action,
                ssh_opts=ssh_opts,
            )
        # On the success path a failed sync is a real error: the run produced
        # results and we could not collect them.
        rsync_down(target, host_out, remote_run=remote_run, ssh_opts=ssh_opts)
        synced = True
    finally:
        # Grab the log tail before the container goes away — `docker rm`
        # wipes it, and a provisioning caller is about to delete the box.
        if exit_code is None or exit_code != 0:
            failure_tail = fetch_failure_tail(
                target=target,
                container_name=container_name,
                remote_run=remote_run,
                ssh_opts=ssh_opts,
            )
        if not synced:
            # Something is already unwinding — a runtime cap, a stream error,
            # a box reclaimed mid-run. Whatever the job managed to write is
            # often the only evidence of what went wrong, and `max_runtime_seconds`
            # exists precisely to salvage a wedged run (#150). Everything inside
            # this `try` happens after rsync_up proved the box reachable, so a
            # sync here is worth attempting.
            #
            # Best-effort on purpose: an exception is on its way out and must
            # not be replaced by an rsync error naming the wrong problem.
            try:
                rsync_down(target, host_out, remote_run=remote_run, ssh_opts=ssh_opts)
            except Exception as exc:  # noqa: BLE001
                print(
                    f"+ warning: could not collect outputs after failure: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
            if failure_tail:
                print(
                    f"--- last {FAILURE_TAIL_LINES} lines of remote output ---\n"
                    f"{failure_tail}\n"
                    f"--- end remote output ---",
                    flush=True,
                )
        if container_name is not None:
            try:
                ssh_capture(
                    target,
                    f"sudo docker rm -f {container_name} >/dev/null 2>&1 || true",
                    ssh_opts=ssh_opts,
                )
            except Exception as exc:  # noqa: BLE001
                print(
                    f"+ warning: failed to remove container {container_name}: {exc}",
                    flush=True,
                )

    result = DispatchResult(
        exit_code=exit_code,
        container_name=container_name,
        remote_run=remote_run,
        failure_tail=failure_tail,
    )
    if exit_code != 0:
        raise RuntimeError(format_remote_failure(exit_code, failure_tail))
    return result


def format_remote_failure(exit_code: Optional[int], failure_tail: str) -> str:
    msg = f"Remote run exited with status {exit_code}"
    if failure_tail:
        msg += (
            f"\n--- last {FAILURE_TAIL_LINES} lines of remote output ---\n"
            f"{failure_tail}\n"
            f"--- end remote output ---"
        )
    return msg


def run_on_provisioned_vm(
    *,
    app,
    function,
    args: list,
    kwargs: dict,
    backend: str,
    label: str,
    provision: Callable[[], tuple],
    teardown: Callable[[], None],
    outputs_dir: str = "out",
    mode: str = "docker",
    max_runtime_seconds: Optional[int] = None,
    max_inactivity_seconds: Optional[int] = None,
    inactivity_action: str = "diagnose",
    ssh_ready_wait_seconds: int = 1800,
    refresh_callback: Optional[Callable[[], None]] = None,
    ssh_opts: Optional[SshOptions] = None,
) -> None:
    """Provision a box, run one function on it, then always tear it down.

    ``provision`` returns ``(target, ssh_opts_or_None)`` — the second slot
    is for details a driver only learns after creating the box. ``teardown`` runs in a
    ``finally`` and must be best-effort: it can never mask the real error from
    the run, but a silent failure there is a billing leak, so it should shout.

    Anything that raises after ``provision`` has been called leaks a paid box
    unless teardown runs, which is why the try block opens *before* provision
    rather than after it, and why teardown runs even when provision itself
    failed (issue #29).
    """
    with orchestrator_signal_cleanup(label):
        try:
            target, provisioned_opts = provision()
            # A driver that learns its ssh details only after creating the
            # box (an EC2 public IP, say) returns them here; otherwise the
            # caller's options stand.
            if provisioned_opts is not None:
                # Merge rather than replace: a caller-supplied option the
                # driver knows nothing about must survive the box's own
                # details being filled in.
                learned = SshOptions.coerce(provisioned_opts)
                base = SshOptions.coerce(ssh_opts)
                ssh_opts = dataclasses.replace(
                    base,
                    **{k: v for k, v in learned.to_dict().items() if v is not None},
                )
            wait_until_ssh_reachable(
                target,
                ssh_opts=ssh_opts,
                max_wait_s=ssh_ready_wait_seconds,
                refresh_callback=refresh_callback,
            )
            dispatch_to_target(
                app=app,
                function=function,
                args=args,
                kwargs=kwargs,
                target=target,
                backend=backend,
                outputs_dir=outputs_dir,
                mode=mode,
                max_runtime_seconds=max_runtime_seconds,
                max_inactivity_seconds=max_inactivity_seconds,
                inactivity_action=inactivity_action,
                ssh_opts=ssh_opts,
            )
        finally:
            # Unconditional, including when provision() itself raised
            # partway. A create that failed *after* allocating the box is
            # precisely the leak issue #29 was about, and only teardown can
            # tell the difference — so it always runs and is responsible for
            # handling "there was nothing to clean up" quietly.
            teardown()
