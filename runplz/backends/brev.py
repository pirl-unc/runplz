"""Brev backend: thin wrapper over the shared SSH plumbing.

Provides Brev CLI lifecycle (create / stop / delete / ls / refresh) +
the instance-type picker. Everything else — rsync, ssh, docker build,
stream-and-wait, failure-tail, runtime-cap — lives in the public
`runplz.backends.ssh_common` module and is shared with the SSH backend.

Assumes `brev` CLI is installed and `brev login` has been run. Uses Brev's
managed SSH config (`brev refresh` populates ~/.brev/ssh_config, which
~/.ssh/config Includes).
"""

import dataclasses
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Optional

from runplz.backends.provisioning import (
    CloudCliError,
    RetryPolicy,
    apply_teardown,
    make_instance_name,
    run_with_retries,
    split_instance_name,
)
from runplz.backends.ssh_common import (
    CLEANUP_SIGNALS as ssh_common_CLEANUP_SIGNALS,
)
from runplz.backends.ssh_common import (
    OrchestratorKilled,
    orchestrator_signal_cleanup,
    run_on_provisioned_vm,
)
from runplz.selector import Candidate

__all__ = ["run"]

# Names that lived on this module from 3.5 until 3.20.0. brev never called
# them -- they were aliases of the shared layer -- so they are gone from
# brev's own imports, but they were plain public names on a public module,
# and a minor bump should not ImportError on `from runplz.backends.brev
# import rsync_up`. Forwarded with a warning; drop in 4.0.
_MOVED_TO_SSH_COMMON = frozenset(
    {
        "FAILURE_TAIL_LINES",
        "build_image",
        "build_remote_run_manifest",
        "check_preconditions",
        "container_running",
        "ensure_docker",
        "ensure_remote_rsync",
        "fetch_failure_tail",
        "make_container_name",
        "make_remote_run_context",
        "prepare_remote_run",
        "raise_for_runtime_cap",
        "remote_has_nvidia",
        "render_image_ops_script",
        "rsync_down",
        "rsync_up",
        "run_container_detached",
        "run_container_mode",
        "run_local",
        "run_native",
        "ssh_capture",
        "ssh_exec",
        "stream_and_wait",
        "wait_until_ssh_reachable",
    }
)


def __getattr__(name):
    if name in _MOVED_TO_SSH_COMMON:
        import warnings

        warnings.warn(
            f"runplz.backends.brev.{name} moved to runplz.backends.ssh_common "
            f"in 3.20.0; import it from there. This alias goes away in 4.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        from runplz.backends import ssh_common

        return getattr(ssh_common, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Brev instance names must be slug-ish. Lowercase, ASCII, hyphen-separated.
# Some providers cap names around 30-40 chars; keep the generated part short
# enough that typical app/function names fit comfortably.
# Instance naming is the shared provisioning contract — see
# runplz.backends.provisioning. Kept as module-level aliases because both have
# been part of brev's surface since 3.9 and the tests reach them here.
_make_ephemeral_name = make_instance_name
_split_ephemeral_name = split_instance_name


# Brev's CLI hangs on an interactive walkthrough once an instance exists in the
# org, blocking `brev ls`/`brev refresh`. Overwriting this file with the
# completed state skips it. The file lives under the user's home — cheap and
# harmless.
_BREV_ONBOARDING = Path.home() / ".brev" / "onboarding_step.json"
_BREV_ONBOARDING_DONE = {
    "step": 999,
    "hasRunBrevShell": True,
    "hasRunBrevOpen": True,
}


# Signals that should trigger the ephemeral-cleanup path. SIGINT we
# translate to KeyboardInterrupt ourselves too (Python does it by default
# on the main thread, but installing our handler makes the behavior
# explicit and consistent across platforms).
# Signal-driven cleanup lives in ssh_common now, shared with every
# provisioning backend. These names are kept so existing callers and tests
# keep working.
CLEANUP_SIGNALS = ssh_common_CLEANUP_SIGNALS
_OrchestratorKilled = OrchestratorKilled
_orchestrator_signal_cleanup = orchestrator_signal_cleanup


def dispatch_mode(cfg) -> str:
    """Map BrevConfig's two knobs onto the shared dispatcher's one."""
    if cfg.mode == "container":
        return "container"
    return "docker" if cfg.use_docker else "native"


def run(
    app,
    function,
    args,
    kwargs,
    *,
    instance: Optional[str] = None,
    outputs_dir: str = "out",
):
    _require_brev_cli()
    _skip_onboarding()

    cfg = app.brev_config
    from runplz.app import validate_image_vs_brev_mode

    validate_image_vs_brev_mode(fn_name=function.name, image=function.image, brev_config=cfg)

    # Ephemeral mode: no name pinned by the caller. Generate one, force
    # auto-create (there's nothing existing to target), and switch on_finish
    # to "delete" so we don't leak a billed stopped box — nothing will
    # ever reuse this name. If the user has explicitly asked for "leave"
    # they're probably debugging; respect that.
    if instance is None:
        instance = _make_ephemeral_name(app.name, function.name)
        overrides = {"auto_create_instances": True}
        if cfg.on_finish == "stop":
            overrides["on_finish"] = "delete"
        cfg = dataclasses.replace(cfg, **overrides)
        print(
            f"+ ephemeral mode: instance={instance!r}, "
            f"auto_create_instances=True, on_finish={cfg.on_finish!r}",
            flush=True,
        )

    # Typo guard (pre-provision): raise BEFORE handing off to the shared
    # lifecycle so its teardown doesn't run — we haven't touched any
    # billable state yet.
    existed = _instance_exists(instance)
    if not existed and not cfg.auto_create_instances:
        raise RuntimeError(
            f"Brev instance {instance!r} not found. Nothing was created.\n"
            f"  - If you mistyped the name, run `brev ls` to see your boxes.\n"
            f"  - If you want runplz to create it for you, pass "
            f"`BrevConfig(auto_create_instances=True)` on your App.\n"
            f"  - Or pre-create it yourself: "
            f"`brev create {instance} --mode container --type <TYPE> "
            f"--container-image <IMAGE>` (or --type <TYPE> for vm mode)."
        )

    # Refresh callback does two things per invocation:
    # 1. Runs `brev refresh` so ~/.brev/ssh_config picks up any port
    #    changes when the instance transitions from the bootstrap-shim
    #    port to the real one.
    # 2. Checks `brev ls` for terminal failure states and raises
    #    BrevInstanceFailed early, so a box stuck in FAILURE / DEAD /
    #    DEPLOYING_FAILED doesn't burn the full budget.
    def _poll_refresh_and_check():
        _refresh_ssh()
        _check_terminal_state(instance)

    def provision():
        if not existed:
            _create_instance(instance, cfg=cfg, image=function.image, function=function)
        else:
            # Existing instance — may have been stopped by a previous run's
            # `on_finish="stop"`. Resume it before SSH.
            _start_instance_if_stopped(instance)
        _refresh_ssh()
        # Brev publishes its own ssh alias, so the instance name IS the
        # target and there is no port to pin.
        return (instance, None)

    def teardown():
        _apply_on_finish(instance=instance, cfg=cfg)

    # `brev create` has its own wait-for-ready loop, but on some providers
    # (8-GPU boxes, slow pull of large container images) it returns before
    # SSH is actually reachable — hence the explicit poll with the refresh
    # callback above.
    run_on_provisioned_vm(
        app=app,
        function=function,
        args=args,
        kwargs=kwargs,
        backend="brev",
        label=instance,
        provision=provision,
        teardown=teardown,
        outputs_dir=outputs_dir,
        mode=dispatch_mode(cfg),
        max_runtime_seconds=cfg.max_runtime_seconds,
        ssh_ready_wait_seconds=cfg.ssh_ready_wait_seconds,
        refresh_callback=_poll_refresh_and_check,
    )


# --- Brev CLI lifecycle --------------------------------------------------


def _require_brev_cli():
    if subprocess.run(["which", "brev"], capture_output=True).returncode != 0:
        raise RuntimeError(
            "`brev` CLI not found. Install via `brew install brev` (macOS) "
            "or the script at https://developer.nvidia.com/brev, then run "
            "`brev login`."
        )


def _skip_onboarding():
    try:
        _BREV_ONBOARDING.parent.mkdir(parents=True, exist_ok=True)
        existed = _BREV_ONBOARDING.exists()
        _BREV_ONBOARDING.write_text(json.dumps(_BREV_ONBOARDING_DONE))
        if not existed:
            print(
                f"+ wrote {_BREV_ONBOARDING} to skip Brev CLI walkthrough "
                f"(prevents `brev ls` from hanging once instances exist)",
                flush=True,
            )
    except OSError:
        pass


def list_jobs() -> list[dict]:
    """Return Brev instances runplz created for ephemeral runs.

    Matches the full shape :func:`_make_ephemeral_name` generates —
    ``runplz-<app>-<fn>-<uuid8>`` where uuid8 is 8 lowercase hex chars. A
    user-named ``--instance runplz-mygpu`` box won't match (no uuid suffix),
    so we won't falsely report it as a live job. Jobs dispatched to a reused
    ``--instance`` box are not included — from ``brev ls`` alone we can't
    tell whether a job is actively running inside such a box.
    """
    r = _brev_capture(["brev", "ls", "--json"], label="brev ls --json (ps)")
    if r.returncode != 0:
        raise RuntimeError(
            f"`brev ls --json` failed with exit code {r.returncode}. "
            f"stderr: {(r.stderr or '').strip()[:300]}"
        )
    return _jobs_from_brev_rows(_parse_brev_ls_rows(r.stdout))


_EPHEMERAL_NAME_RE = re.compile(r"^runplz-.+-[0-9a-f]{8}$")


def _jobs_from_brev_rows(rows: list[dict]) -> list[dict]:
    jobs = []
    for row in rows:
        name = row.get("name") or ""
        if not _EPHEMERAL_NAME_RE.match(name):
            continue
        app_name, fn_name = _split_ephemeral_name(name)
        jobs.append(
            {
                "backend": "brev",
                "name": name,
                "app": app_name,
                "function": fn_name,
                "started": row.get("createdAt") or row.get("created_at") or "",
                "status": _snapshot_status(row) or "",
            }
        )
    return jobs


def _instance_exists(name: str) -> bool:
    """True iff `brev ls` lists an instance with this name.

    Raises RuntimeError if the `brev` CLI itself failed (bad auth, network,
    Brev API outage, malformed JSON) — we refuse to silently treat a degraded
    listing as "instance doesn't exist," because the caller's fallback is to
    auto-create a *new billed box*, which may duplicate one the user already
    has. Transient errors are retried before escalating.

    Returns False only when `brev ls` succeeded but the target name is
    definitively not in the list (including the documented `null` / empty
    shapes Brev returns for empty orgs).
    """
    r = _brev_capture(["brev", "ls", "--json"], label="brev ls --json")
    if r.returncode != 0:
        raise RuntimeError(
            f"`brev ls --json` failed with exit code {r.returncode} after "
            f"{len(_BREV_DEFAULT_RETRIES)} attempts. Refusing to continue — if "
            f"we assumed the instance was missing we'd auto-create a duplicate "
            f"billed box. Run `brev login` / check the `brev` CLI and retry. "
            f"stderr: {(r.stderr or '').strip()[:500]}"
        )
    instances = _parse_brev_ls_rows(r.stdout)
    return any(i.get("name") == name for i in instances)


# Field names we accept when pulling a power-state string out of a
# `brev ls --json` row. Brev has used more than one over time.
_BREV_STATUS_FIELDS = ("status", "state", "power_state", "lifecycle_status")

# Status values treated as "needs a `brev start` first." Lower-cased for
# case-insensitive matching.
_BREV_STOPPED_STATES = {"stopped", "paused", "hibernated", "suspended"}

# Terminal failure states — the instance isn't coming back. Probing SSH
# against these is wasted budget (observed in the wild: H100 Nebius
# workspaces that enter FAILURE after provisioning and never leave).
# Lower-cased for case-insensitive matching.
_BREV_TERMINAL_FAILED_STATES = {
    "failed",
    "failure",
    "deploying_failed",
    "create_failed",
    "terminated",
    "dead",
    "error",
}


def _parse_brev_ls_rows(stdout: str) -> list[dict]:
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"`brev ls --json` returned unparseable JSON. stdout head: "
            f"{(stdout or '').strip()[:200]!r}"
        ) from exc
    if data is None:
        return []
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        rows = data.get("instances", []) or []
    else:
        raise RuntimeError(
            f"`brev ls --json` returned an unexpected shape ({type(data).__name__})."
        )
    return [row for row in rows if isinstance(row, dict)]


def _snapshot_status(snapshot: Optional[dict]) -> Optional[str]:
    if snapshot is None:
        return None
    for key in _BREV_STATUS_FIELDS:
        value = snapshot.get(key)
        if value:
            return str(value)
    return None


def _format_instance_snapshot(snapshot: Optional[dict]) -> str:
    if snapshot is None:
        return "<missing>"
    parts = []
    for key in ("name", "status", "state", "power_state", "lifecycle_status", "provider", "id"):
        value = snapshot.get(key)
        if value:
            parts.append(f"{key}={value}")
    return ", ".join(parts) if parts else json.dumps(snapshot, sort_keys=True)


def _instance_snapshot(name: str) -> Optional[dict]:
    r = _brev_capture(["brev", "ls", "--json"], label=f"brev ls --json (snapshot {name})")
    if r.returncode != 0:
        raise RuntimeError(
            f"`brev ls --json` failed while checking {name!r}. "
            f"stderr: {(r.stderr or '').strip()[:500]}"
        )
    for row in _parse_brev_ls_rows(r.stdout):
        if row.get("name") == name:
            return row
    return None


def _verify_post_action_state(
    action: str,
    name: str,
    *,
    timeout_s: int = 20,
    poll_interval_s: int = 5,
) -> None:
    deadline = time.monotonic() + timeout_s
    last_snapshot: Optional[dict] = None
    while True:
        try:
            last_snapshot = _instance_snapshot(name)
        except Exception as exc:  # noqa: BLE001
            print(
                f"+ warning: could not verify `brev {action} {name}` via `brev ls`: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            return
        status = (_snapshot_status(last_snapshot) or "").strip().lower()
        if action == "create" and last_snapshot is not None:
            print(
                f"+ verified create: {_format_instance_snapshot(last_snapshot)}",
                flush=True,
            )
            return
        if action == "start" and last_snapshot is not None and status not in _BREV_STOPPED_STATES:
            print(
                f"+ verified start: {_format_instance_snapshot(last_snapshot)}",
                flush=True,
            )
            return
        if action == "stop" and (
            last_snapshot is None or status in _BREV_STOPPED_STATES or status == "deleted"
        ):
            suffix = _format_instance_snapshot(last_snapshot)
            print(f"+ verified stop: {suffix}", flush=True)
            return
        if action == "delete" and last_snapshot is None:
            print(f"+ verified delete: {name!r} no longer listed by `brev ls`", flush=True)
            return
        if time.monotonic() >= deadline:
            break
        time.sleep(poll_interval_s)

    print(
        f"+ warning: `brev {action} {name}` returned success but post-action "
        f"state is still {_format_instance_snapshot(last_snapshot)}",
        flush=True,
    )


def _instance_status(name: str) -> Optional[str]:
    """Return the raw status string for `name` from `brev ls --json`, or
    None if the instance isn't listed or no recognized status field is
    present. Best-effort — used only to decide whether we need to call
    `brev start` before SSH. Transient errors are retried; any final
    failure quietly returns None (the SSH reachability loop will surface
    a real problem if there is one)."""
    try:
        return _snapshot_status(_instance_snapshot(name))
    except Exception:  # noqa: BLE001
        return None


class BrevInstanceFailed(RuntimeError):
    """Brev reports the instance in a terminal failure state (FAILURE /
    DEAD / DEPLOYING_FAILED). Raised during the SSH-ready poll so we
    bail early instead of waiting out the full 30-minute budget.
    Distinct exception type so `brev.run()`'s finally block can tell
    "provisioning failed" apart from "dispatch failed" when shaping the
    user-facing error."""


def _check_terminal_state(name: str) -> None:
    """Raise BrevInstanceFailed if `brev ls` reports a terminal failure
    state for this instance. Called periodically during
    wait_until_ssh_reachable so we stop probing dead boxes early.

    Silent no-op if status isn't recognizable — the reachability loop
    handles ambiguous cases by timing out normally.
    """
    status = _instance_status(name)
    if status is None:
        return
    if status.strip().lower() in _BREV_TERMINAL_FAILED_STATES:
        raise BrevInstanceFailed(
            f"Brev instance {name!r} is in terminal state {status!r}. "
            f"Provisioning at the cloud-provider layer failed (check "
            f"`brev ls` / provider console for details). Runplz will "
            f"not waste the SSH-reachability budget probing a dead box."
        )


def _start_instance_if_stopped(name: str) -> None:
    """If the Brev box for `name` is in a stopped / paused state, run
    `brev start` before the dispatch tries to SSH.

    The 3.2 default of `on_finish="stop"` means every previous runplz run
    leaves the box powered off — without this, the next run silently hangs
    in `wait_until_ssh_reachable` until the 20-minute deadline. Best-
    effort: if Brev's schema doesn't expose a status we can recognize, we
    skip and let the SSH reachability loop figure it out (or time out).
    """
    status = _instance_status(name)
    if status is None:
        return
    if status.strip().lower() not in _BREV_STOPPED_STATES:
        return
    print(f"+ instance {name!r} is {status!r}; running `brev start {name}`", flush=True)
    try:
        r = _brev_capture(
            ["brev", "start", name],
            timeout=600,
            label=f"brev start {name}",
        )
    except Exception as exc:  # noqa: BLE001
        print(
            f"+ warning: `brev start {name}` raised {type(exc).__name__}: {exc}. "
            f"SSH reachability probe will decide whether to continue.",
            flush=True,
        )
        return
    if r.returncode != 0:
        print(
            f"+ warning: `brev start {name}` exited {r.returncode}. "
            f"stderr: {(r.stderr or '').strip()[:500]}. "
            f"SSH reachability probe will decide whether to continue.",
            flush=True,
        )
        return
    _verify_post_action_state("start", name)


def _create_instance(name: str, *, cfg=None, image=None, function=None):
    """Provision a Brev instance.

    Picks the instance type in this order:
    1. `cfg.instance_type` — explicit user override (if set).
    2. Cheapest match from `brev search` driven by `function`'s resource
       constraints, with the 5% cost-tolerance + availability tiebreaker
       applied via `runplz.selector.pick_machine`.

    Raises only if the picker finds no matches.
    """
    # Build the list of candidate types to pass to `brev create`. Brev's
    # CLI natively supports repeated `--type` flags for multi-provider
    # fallback (if A fails on Nebius, it tries B on OCI, etc.). We feed
    # the selector's top-N ranked candidates when auto-picking; a
    # user-pinned `instance_type` is always the one-and-only.
    if cfg is not None and cfg.instance_type is not None:
        # User pinned the type explicitly — respect it even if it's on the
        # provider blocklist. Their pin, their consequences.
        instance_types = [cfg.instance_type]
    else:
        n = cfg.instance_type_fallback_count if cfg is not None else 1
        exclude = cfg.exclude_providers if cfg is not None else ()
        instance_types = _pick_instance_types(function, n=n, exclude_providers=exclude)
        if not instance_types:
            raise RuntimeError(
                "`brev search` returned no matching instances. Loosen the "
                "function's resource constraints, pass an explicit "
                "`instance_type=...` on BrevConfig, or pre-create the instance."
            )

    cmd = ["brev", "create", name]
    for t in instance_types:
        cmd += ["--type", t]
    if function is not None and function.min_disk is not None:
        cmd += ["--min-disk", str(function.min_disk)]
    if cfg is not None and cfg.mode == "container":
        # `image.base` is guaranteed to be set here — Dockerfile images are
        # rejected at function-decoration time by runplz.app's validator.
        cmd += ["--mode", "container", "--container-image", image.base]

    # `brev create` can take a while; 10 minutes per attempt gives the API
    # enough room on slow providers. Retry transient errors (HTTP 500, EOF,
    # context deadline) — see the 3.8 report for real-world signatures.
    print("+ " + " ".join(str(c) for c in cmd), flush=True)
    r = _brev_capture(cmd, timeout=600, label=f"brev create {name}")
    if r.returncode == 0:
        _verify_post_action_state("create", name)
        return

    err_str = ((r.stderr or "") + (r.stdout or "")).strip()
    # Idempotency guard: if we retried and Brev says "already exists", the
    # first attempt succeeded under the hood (HTTP 500 *after* create was
    # registered, which happens). Verify the instance is really there and
    # proceed instead of failing.
    if _looks_already_exists(err_str):
        print(
            f"+ `brev create {name}` reports already exists; verifying via "
            f"`brev ls` (idempotent-retry path)",
            flush=True,
        )
        try:
            if _instance_exists(name):
                print(f"+ {name!r} confirmed created — treating as success", flush=True)
                _verify_post_action_state("create", name)
                return
        except RuntimeError:
            pass  # fall through to the hard error below

    # Non-retriable org/config gaps (missing OCI cred, provider not enabled,
    # quota exceeded, auth expired) get a reframed error pointing the user
    # at the actual fix — see _reframe_brev_create_error. Skip the snapshot
    # probe since we know nothing was created. (Issue #62.)
    if _looks_non_retriable(err_str):
        raise RuntimeError(_reframe_brev_create_error(name, instance_types, err_str))

    try:
        snapshot_text = _format_instance_snapshot(_instance_snapshot(name))
    except Exception as exc:  # noqa: BLE001
        snapshot_text = f"[snapshot unavailable: {type(exc).__name__}: {exc}]"

    raise RuntimeError(
        f"`brev create {name}` failed after {len(_BREV_DEFAULT_RETRIES)} attempt(s) "
        f"with exit {r.returncode}. stderr: {err_str[:500]}. "
        f"Final instance snapshot: {snapshot_text}"
    )


_BREV_TRANSIENT_PATTERNS = (
    # API / gRPC layer (brevapi.us-west-2-prod.control-plane.brev.dev).
    "context deadline exceeded",
    "rpc error",
    "connection reset",
    "connection refused",
    "i/o timeout",
    "temporary failure in name resolution",
    # HTTP status codes — 500/502/503/504 are all transient; Brev sometimes
    # surfaces 500 Internal Server Error on otherwise-valid `brev create`.
    "internal server error",
    "service unavailable",
    "bad gateway",
    "gateway timeout",
    "500",
    "502",
    "503",
    "504",
    # Transport-level hiccups.
    "eof",
    "unexpected eof",
    "http2: server sent goaway",
    "read: connection closed",
    "broken pipe",
    # Shadeform broker — the "not_found" case from attempt 5 is NOT
    # transient (broker said no), but an empty "list failed" often is.
    "external nodes: skipping (list failed):",
)

# Error signatures meaning "Brev already created this instance but we lost
# the response" — the retry attempt then races into `name already exists`.
# Caller treats this as success after verifying the instance actually exists.
_BREV_ALREADY_EXISTS_PATTERNS = (
    "already exists",
    "name is taken",
    "workspace with this name",
    "instance with name",
    "conflict",
)

# Error signatures from the Brev API that are NOT transient — retrying them
# burns the whole retry budget on guaranteed-fail attempts and delays the
# real failure by ~21s for nothing. These are all org/config gaps that
# require human action (Brev admin console / web UI / support ticket); no
# amount of CLI tweaking from runplz makes them succeed. (Issue #62.)
_BREV_NON_RETRIABLE_PATTERNS = (
    # OCI launchpad path: org has no OCI cloud credential registered.
    "cloudcredid or workspacegroupid must be specified",
    # Provider integration not enabled at the org level.
    "provider not enabled",
    "provider is not configured",
    # Server-side quota gates — retrying won't conjure capacity.
    "quota exceeded",
    "quota has been exceeded",
    # Auth gone — no point retrying with the same expired token.
    "unauthorized",
    "401 unauthorized",
    "403 forbidden",
)


def _looks_non_retriable(err: str) -> bool:
    """True iff ``err`` matches a Brev error pattern we know retrying can't fix."""
    low = (err or "").lower()
    return any(pat in low for pat in _BREV_NON_RETRIABLE_PATTERNS)


def _reframe_brev_create_error(name: str, types: list, err_str: str) -> str:
    """Translate a known Brev API error into a message a user can act on.

    Falls through to the raw error if the pattern isn't one we recognize.
    The known patterns all have the same shape — "this won't work until your
    Brev org is reconfigured" — so the suggestions are roughly the same.
    """
    low = err_str.lower()
    type_hint = ", ".join(str(t) for t in types) if types else "<unknown>"
    if "cloudcredid or workspacegroupid must be specified" in low:
        return (
            f"`brev create {name} --type {type_hint}` failed: this Brev org has "
            f"no cloud credential registered for the provider that hosts the "
            f"requested instance type. The credential is configured server-side "
            f"in the Brev console — runplz cannot pass it as a CLI flag.\n\n"
            f"Fix options:\n"
            f"  - Pick a --type whose provider is already configured for your "
            f"org (run `brev search gpu` to see what your org can provision).\n"
            f"  - Pre-create the instance once via the Brev web UI, which walks "
            f"through cred setup. Subsequent `runplz brev --instance <name>` "
            f"calls will reuse that box.\n"
            f"  - Ask your Brev admin / support to register a cloudCredId for "
            f"the provider that hosts {type_hint}.\n\n"
            f"The instance was NOT created. No charge. Raw API error: {err_str[:300]}"
        )
    if "provider not enabled" in low or "provider is not configured" in low:
        return (
            f"`brev create {name} --type {type_hint}` failed: the requested "
            f"provider is not enabled on your Brev org. Pick a different --type "
            f"or have your Brev admin enable the provider in the console. "
            f"No instance created. Raw API error: {err_str[:300]}"
        )
    if "quota" in low and "exceeded" in low:
        return (
            f"`brev create {name} --type {type_hint}` failed: provider quota "
            f"exceeded. This is server-side capacity — wait, request a quota "
            f"increase, or pick a --type from a different provider. "
            f"No instance created. Raw API error: {err_str[:300]}"
        )
    if "unauthorized" in low or "403 forbidden" in low:
        return (
            f"`brev create {name} --type {type_hint}` failed: Brev rejected the "
            f"call as unauthorized. Re-run `brev login` and try again. "
            f"No instance created. Raw API error: {err_str[:300]}"
        )
    return f"`brev create {name}` failed (non-retriable). Raw API error: {err_str[:500]}"


# Attempts: (0, 3, 6, 12) = 4 tries, ~21s total retry budget per CLI call.
# Enough to ride out typical Brev API blips without delaying hard-failures
# by too much. Every brev call uses this unless explicitly overridden.
_BREV_DEFAULT_RETRIES = (0, 3, 6, 12)

# Per-attempt timeout for brev CLI calls. 90s gives slow control-plane calls
# (brev ls with many instances, brev create kicking off provisioning) room
# without waiting forever on a genuinely hung subprocess.
_BREV_DEFAULT_TIMEOUT_S = 90


def _looks_transient(err: str) -> bool:
    low = err.lower()
    return any(pat in low for pat in _BREV_TRANSIENT_PATTERNS)


def _looks_already_exists(err: str) -> bool:
    low = err.lower()
    return any(pat in low for pat in _BREV_ALREADY_EXISTS_PATTERNS)


BREV_RETRY_POLICY = RetryPolicy(
    waits=_BREV_DEFAULT_RETRIES,
    is_transient=_looks_transient,
    is_non_retriable=_looks_non_retriable,
)


def _brev_capture(
    cmd: list,
    *,
    timeout: int = _BREV_DEFAULT_TIMEOUT_S,
    retry_waits: tuple = _BREV_DEFAULT_RETRIES,
    label: Optional[str] = None,
) -> subprocess.CompletedProcess:
    """Run a `brev` subcommand with transient-error retries.

    The attempt loop is shared (see
    :func:`runplz.backends.provisioning.run_with_retries`); what stays here
    is Brev's own classification — which API strings are worth another try,
    and which org/config gaps retrying can never fix (issue #62).

    Non-transient failures return the CompletedProcess for caller
    inspection; a non-zero exit is not always fatal to the caller.
    `subprocess.TimeoutExpired` on every attempt raises.
    """
    label = label or " ".join(str(c) for c in cmd[:3])
    policy = dataclasses.replace(BREV_RETRY_POLICY, waits=retry_waits)
    return run_with_retries(cmd, label=label, timeout=timeout, policy=policy)


def _brev_sh(
    cmd: list,
    *,
    timeout: int = _BREV_DEFAULT_TIMEOUT_S,
    retry_waits: tuple = _BREV_DEFAULT_RETRIES,
    label: Optional[str] = None,
):
    """Analogue of `run_local` for brev CLI calls: runs with retries, prints the
    command, and raises on final non-zero exit."""
    import shlex as _shlex

    label = label or " ".join(str(c) for c in cmd[:3])
    print("+ " + " ".join(_shlex.quote(str(c)) for c in cmd), flush=True)
    r = _brev_capture(cmd, timeout=timeout, retry_waits=retry_waits, label=label)
    if r.returncode != 0:
        err = (r.stderr or r.stdout or "").strip()[:500]
        raise RuntimeError(
            f"`{label}` failed after {len(retry_waits)} attempt(s) "
            f"with exit {r.returncode}. stderr: {err}"
        )


def _refresh_ssh():
    """Run `brev refresh`, retrying on transient Brev API errors.

    Issue #28: `brev refresh` periodically returns `rpc error: context
    deadline exceeded` when the Brev backend API is slow (common during
    8×A100 Denvr/OCI provisioning). Without retry, that used to be fatal
    and leaked a billed box. 3.8.0 unified this with every other brev
    CLI call through _brev_sh.
    """
    _brev_sh(["brev", "refresh"], label="brev refresh")


def _apply_on_finish(*, instance: str, cfg) -> None:
    """Stop / delete / leave the Brev box per `cfg.on_finish`.

    Runs under the shared teardown contract (see
    :func:`runplz.backends.provisioning.apply_teardown`): never raises, never fails
    quietly. Brev adds two things the other providers don't have — retries on
    transient API errors via :func:`_brev_capture`, so a single flaky call
    doesn't leak a billed box, and a post-action state check that confirms the
    box really reached the state we asked for.
    """

    def _act(action: str) -> None:
        print(f"+ on_finish={action}: running `brev {action} {instance}`", flush=True)
        r = _brev_capture(
            ["brev", action, instance],
            timeout=120,
            label=f"brev {action} {instance}",
        )
        if r.returncode != 0:
            raise CloudCliError(
                f"`brev {action} {instance}` exited {r.returncode}. "
                f"stderr: {(r.stderr or '').strip()[:500]}"
            )
        _verify_post_action_state(action, instance)

    apply_teardown(
        on_finish=cfg.on_finish,
        target=instance,
        run_action=_act,
        check_hint="brev ls",
    )


# --- Brev instance-type picker -------------------------------------------


def _brev_gpu_name(modal_name: str) -> str:
    """Translate Modal-style GPU labels to Brev `--gpu-name` filter strings.

    Modal accepts things like "A100-40GB"; brev search wants a base name
    like "A100" and separately filters by VRAM. Stripping the suffix is
    good enough for matching.
    """
    n = modal_name.upper()
    for suffix in ("-40GB", "-80GB", "-16GB", "-24GB"):
        if n.endswith(suffix):
            return n[: -len(suffix)]
    return n


# Candidate field names we tolerate across `brev search --json` schema
# drift. Ordered most-specific → least.
_BREV_PRICE_FIELDS = (
    "hourly_price",
    "price_per_hour",
    "usd_per_hour",
    "price",
    "hourly_usd",
    "estimated_hourly",
)
_BREV_AVAILABILITY_FIELDS = (
    "estimated_start_seconds",
    "eta_seconds",
    "eta_s",
    "queue_wait_seconds",
    "availability_rank",
)
_BREV_REGION_FIELDS = ("region", "zone", "location", "provider_region")


def _brev_row_type(row: dict) -> Optional[str]:
    return row.get("type") or row.get("Type") or row.get("name")


def _candidate_from_brev_row(row: dict) -> Optional[Candidate]:
    """Map a single `brev search --json` row onto a selector Candidate."""
    if not isinstance(row, dict):
        return None
    name = _brev_row_type(row)
    if not name:
        return None

    price = None
    for key in _BREV_PRICE_FIELDS:
        v = row.get(key)
        if v is None:
            continue
        try:
            price = float(v)
            break
        except (TypeError, ValueError):
            continue

    hint = None
    for key in _BREV_AVAILABILITY_FIELDS:
        v = row.get(key)
        if v is None:
            continue
        try:
            hint = float(v)
            break
        except (TypeError, ValueError):
            continue

    region = None
    for key in _BREV_REGION_FIELDS:
        v = row.get(key)
        if v:
            region = str(v)
            break

    return Candidate(name=name, hourly_usd=price, availability_hint=hint, region=region, raw=row)


def _pick_instance_types(function, *, n: int = 1, exclude_providers: tuple = ()) -> list:
    """Run `brev search` with filters from `function`'s resource requests
    and return up to `n` ranked TYPE strings for multi-type fallback
    dispatch (issue #44).

    Brev's own `--sort price` gives us cheapest-first. We post-process
    through `pick_machines` so the top pick applies the 5% cost-
    tolerance + availability tiebreaker, and the tail provides fallback
    candidates cheapest-first. If no price/availability fields are
    exposed, falls back to the top N rows in `brev search` order.

    Returns `[]` on no match. When n == 1, returns a single-element list
    (or []).
    """
    from runplz.selector import pick_machines

    # gpu mode whenever the user named a model OR set any GPU-shaped
    # constraint (min_gpu_memory, multi-GPU). This is what makes
    # `min_gpu_memory=24` without `gpu=` actually search GPU instances
    # rather than silently falling through to a CPU box.
    needs_gpu_search = (
        function.gpu is not None
        or function.min_gpu_memory is not None
        or (getattr(function, "num_gpus", 1) or 1) > 1
    )
    mode = "gpu" if needs_gpu_search else "cpu"
    cmd = ["brev", "search", mode, "--json", "--sort", "price"]
    if function.gpu:
        cmd += ["--gpu-name", _brev_gpu_name(function.gpu)]
    num_gpus = getattr(function, "num_gpus", 1) or 1
    if num_gpus > 1:
        cmd += ["--min-gpus", str(num_gpus)]
    if function.min_gpu_memory is not None:
        cmd += ["--min-vram", str(function.min_gpu_memory)]
    if function.min_cpu is not None:
        cmd += ["--min-vcpu", str(int(function.min_cpu))]
    if function.min_memory is not None:
        cmd += ["--min-ram", str(function.min_memory)]
    if function.min_disk is not None:
        cmd += ["--min-disk", str(function.min_disk)]
    print("+ " + " ".join(str(c) for c in cmd), flush=True)
    r = _brev_capture(cmd, label=f"brev search {mode}")
    if r.returncode != 0:
        return []
    try:
        results = json.loads(r.stdout)
    except json.JSONDecodeError:
        return []
    if not isinstance(results, list) or not results:
        return []

    # Drop rows whose type matches an excluded provider prefix BEFORE the
    # selector ranks anything (issue #62 follow-up: OCI is the canonical
    # offender — its launchpad path fails server-side on most orgs and
    # there's nothing runplz can do about it from the client).
    if exclude_providers:
        before = len(results)
        results = [
            row
            for row in results
            if not _matches_excluded_provider(_brev_row_type(row), exclude_providers)
        ]
        dropped = before - len(results)
        if dropped:
            print(
                f"+ selector excluded {dropped} candidate(s) matching "
                f"BrevConfig.exclude_providers={list(exclude_providers)}",
                flush=True,
            )
        if not results:
            return []

    candidates = [_candidate_from_brev_row(row) for row in results]
    priced = [c for c in candidates if c is not None and c.hourly_usd is not None]
    if priced:
        choices = pick_machines(priced, n=n)
        if choices:
            names = [c.name for c in choices]
            print(
                f"+ selector picked {names!r}: top={choices[0].reason}",
                flush=True,
            )
            return names

    # Fallback: price/name fields not exposed in this `brev search` shape.
    # Take the first N rows in Brev's own order (it already sorted by
    # price server-side).
    fallback = []
    for row in results[:n]:
        t = _brev_row_type(row)
        if t:
            fallback.append(t)
    return fallback


def _matches_excluded_provider(type_str: Optional[str], exclude_providers: tuple) -> bool:
    """True iff ``type_str`` starts with any excluded provider prefix.

    Match is case-insensitive and provider-segment-aware: a prefix ``"oci"``
    matches ``"oci.a100x8.sxm.brev-dgxc"`` and ``"OCI_..."`` but not
    ``"ocifoo"`` (so a type happening to share leading letters doesn't get
    nuked). The boundary characters we accept are ``.`` and ``_`` because
    those are what Brev uses to delimit the provider segment in practice
    (``oci.a100x8...``, ``massedcompute_A100_...``).
    """
    if not type_str or not exclude_providers:
        return False
    low = type_str.lower()
    for prefix in exclude_providers:
        p = prefix.lower()
        if low == p:
            return True
        if low.startswith(p) and len(low) > len(p) and low[len(p)] in (".", "_", "-"):
            return True
    return False


def _pick_instance_type(function) -> Optional[str]:
    """Single-type picker. Back-compat wrapper around
    `_pick_instance_types(function, n=1)` — kept so older code /
    tests that expect a single TYPE string still work."""
    types = _pick_instance_types(function, n=1)
    return types[0] if types else None
