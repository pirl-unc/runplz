"""Brev backend: thin wrapper over the shared SSH plumbing.

Provides Brev CLI lifecycle (create / stop / delete / ls / refresh) +
the instance-type picker. Everything else — rsync, ssh, docker build,
stream-and-wait, failure-tail, runtime-cap — lives in
`runplz/backends/_ssh_common.py` and is shared with the SSH backend.

Assumes `brev` CLI is installed and `brev login` has been run. Uses Brev's
managed SSH config (`brev refresh` populates ~/.brev/ssh_config, which
~/.ssh/config Includes).
"""

import contextlib
import dataclasses
import json
import re
import signal
import subprocess
import time
import uuid
from pathlib import Path
from typing import Optional

from runplz._selector import Candidate
from runplz.backends._ssh_common import (
    FAILURE_TAIL_LINES,
    _build_image,
    _check_preconditions,
    _container_running,
    _ensure_docker,
    _ensure_remote_rsync,
    _fetch_failure_tail,
    _prepare_remote_run,
    _raise_for_runtime_cap,
    _remote_has_nvidia,
    _render_ops_script,
    _rsync_down,
    _rsync_up,
    _run_container_detached,
    _run_container_mode,
    _run_native,
    _sh,
    _ssh,
    _ssh_capture,
    _stream_and_wait,
    _wait_until_ssh_reachable,
    build_remote_run_manifest,
    make_container_name,
    make_remote_run_context,
)

# Re-exports so older test patches and external code that patched these
# keep working without a hard rename. 3.8's _brev_capture / _brev_sh
# replaced `_sh` for brev CLI calls, but tests still patch `brev._sh`
# in a few places — the re-export keeps the test surface stable.
_ = (  # noqa: F841 — held for test-mocking compatibility
    _container_running,
    _raise_for_runtime_cap,
    _render_ops_script,
    _sh,
    _ssh,
)

__all__ = ["run"]


# Brev instance names must be slug-ish. Lowercase, ASCII, hyphen-separated.
# Some providers cap names around 30-40 chars; keep the generated part short
# enough that typical app/function names fit comfortably.
_BREV_NAME_SAFE_RE = re.compile(r"[^a-z0-9-]+")


def _make_ephemeral_name(app_name: str, fn_name: str) -> str:
    """Generate a Brev-safe instance name for an ephemeral run.

    Shape: ``runplz-<app>-<fn>-<uuid8>``. Trailing uuid makes the name
    unique per dispatch so two concurrent runs don't collide.
    """

    def _slug(s: str) -> str:
        return _BREV_NAME_SAFE_RE.sub("-", s.lower()).strip("-") or "x"

    return f"runplz-{_slug(app_name)}-{_slug(fn_name)}-{uuid.uuid4().hex[:8]}"


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
_CLEANUP_SIGNALS = (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)


class _OrchestratorKilled(RuntimeError):
    """Raised when the runplz orchestrator receives SIGTERM / SIGHUP /
    SIGINT. Propagates through the dispatch try/finally so _apply_on_finish
    fires. Issue #38."""


@contextlib.contextmanager
def _orchestrator_signal_cleanup(instance: str):
    """Install translators that convert termination signals into an
    exception so brev.run()'s finally block can clean up the Brev box.

    Without this, `kill -TERM <runplz pid>` used to exit cleanly while
    leaving the freshly-provisioned ephemeral box running — no on_finish
    action, no remote cleanup. A leaked A100 at $1.49/hr adds up fast.

    Only runs on the main thread (signal.signal is main-thread-only). If
    called off-main (e.g. from a test runner worker) the handlers aren't
    installed; cleanup degrades to Ctrl-C only, which is acceptable since
    signal-driven teardown is a main-process concern anyway.
    """
    previous = {}

    def _handler(signum, _frame):
        signame = signal.Signals(signum).name
        print(
            f"+ runplz received {signame} — triggering Brev cleanup for {instance!r}",
            flush=True,
        )
        raise _OrchestratorKilled(
            f"runplz orchestrator killed by {signame}; "
            f"running on_finish for {instance!r} before exit."
        )

    try:
        for sig in _CLEANUP_SIGNALS:
            try:
                previous[sig] = signal.signal(sig, _handler)
            except (ValueError, OSError):
                # Not the main thread, or signal not supported on this
                # platform. Skip — cleanup on that signal is unavailable,
                # but the rest of dispatch still works.
                pass
        yield
    finally:
        for sig, prev in previous.items():
            try:
                signal.signal(sig, prev)
            except (ValueError, OSError):
                pass


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

    # Typo guard (pre-provision): raise BEFORE the try/finally so the
    # cleanup path doesn't run — we haven't touched any billable state yet.
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

    container_name: Optional[str] = None
    exit_code: Optional[int] = None
    remote_run = None
    # Signal handlers: SIGTERM / SIGHUP on the orchestrator used to exit
    # cleanly without firing the finally's _apply_on_finish, leaking a
    # billed box (issue #38). Install translators that convert the signal
    # into a RuntimeError so the finally runs. Original handlers are
    # restored on exit.
    with _orchestrator_signal_cleanup(instance):
        try:
            # Everything inside this block can leak a billed box if it
            # raises — widen the scope beyond just the dispatch (issue #29).
            if not existed:
                _create_instance(instance, cfg=cfg, image=function.image, function=function)
            else:
                # Existing instance — may have been stopped by a previous
                # run's `on_finish="stop"`. Resume it before SSH.
                _start_instance_if_stopped(instance)
            _refresh_ssh()

            # `brev create` has its own internal wait-for-ready loop, but
            # on some providers (8-GPU boxes, slow pull of large container
            # images) that loop times out before SSH is actually reachable.
            # Poll explicitly with a refresh callback so ssh_common can
            # re-run `brev refresh` mid-poll when the instance transitions
            # from bootstrap-shim port to real port.
            # Refresh callback does two things per invocation:
            # 1. Runs `brev refresh` so ~/.brev/ssh_config picks up any
            #    port changes when the instance transitions from the
            #    bootstrap-shim port to the real one.
            # 2. Checks `brev ls` for terminal failure states and raises
            #    BrevInstanceFailed early, so a box stuck in FAILURE /
            #    DEAD / DEPLOYING_FAILED doesn't burn the full budget.
            def _poll_refresh_and_check():
                _refresh_ssh()
                _check_terminal_state(instance)

            _wait_until_ssh_reachable(
                instance,
                refresh_callback=_poll_refresh_and_check,
                max_wait_s=cfg.ssh_ready_wait_seconds,
            )

            repo = app._repo_root
            host_out = (repo / outputs_dir).resolve()
            host_out.mkdir(parents=True, exist_ok=True)
            remote_run = make_remote_run_context(
                backend="brev",
                target=instance,
                function_name=function.name,
            )
            _prepare_remote_run(
                instance,
                remote_run,
                manifest=build_remote_run_manifest(
                    remote_run=remote_run,
                    repo=repo,
                    outputs_dir=outputs_dir,
                    args=args,
                    kwargs=kwargs,
                    env=function.env,
                ),
            )

            if cfg.mode == "container":
                # Pre-built container images (e.g. pytorch/pytorch) don't
                # ship with rsync. Install it before the first rsync call.
                _ensure_remote_rsync(instance)
            _rsync_up(repo, instance, outputs_dir=outputs_dir, remote_run=remote_run)

            # Probe declared remote preconditions (issue #56) before bootstrap.
            # See _ssh_common._check_preconditions for the warn/fail rule.
            _check_preconditions(instance, function.preconditions)

            rel_script = Path(function.module_file).resolve().relative_to(repo)

            if cfg.mode == "container":
                exit_code = _run_container_mode(
                    target=instance,
                    function=function,
                    rel_script=str(rel_script),
                    args=args,
                    kwargs=kwargs,
                    remote_run=remote_run,
                    max_runtime_seconds=cfg.max_runtime_seconds,
                )
            elif cfg.use_docker:
                _ensure_docker(instance)
                gpu_flag = "--gpus all" if _remote_has_nvidia(instance) else ""
                container_name = make_container_name(function.name)
                _build_image(instance, function.image, remote_run=remote_run)
                _run_container_detached(
                    target=instance,
                    container_name=container_name,
                    function=function,
                    rel_script=str(rel_script),
                    args=args,
                    kwargs=kwargs,
                    gpu_flag=gpu_flag,
                    app_name=app.name,
                    remote_run=remote_run,
                )
                exit_code = _stream_and_wait(
                    instance, container_name, max_runtime_seconds=cfg.max_runtime_seconds
                )
            else:
                # Legacy native path. Skips docker; installs python + torch
                # + user code into a venv on a plain VM-mode Brev box.
                exit_code = _run_native(
                    target=instance,
                    function=function,
                    rel_script=str(rel_script),
                    args=args,
                    kwargs=kwargs,
                    has_nvidia=_remote_has_nvidia(instance),
                    remote_run=remote_run,
                    max_runtime_seconds=cfg.max_runtime_seconds,
                )
            _rsync_down(instance, host_out, remote_run=remote_run)
        finally:
            # Fetch a log tail BEFORE container/box cleanup — afterwards
            # the logs are gone (docker rm wipes container state; brev
            # stop/delete makes the box unreachable). Only do this on
            # failure.
            failure_tail = ""
            if exit_code is not None and exit_code != 0:
                failure_tail = _fetch_failure_tail(
                    target=instance,
                    container_name=container_name,
                    remote_run=remote_run,
                )
            if container_name is not None:
                try:
                    _ssh_capture(
                        instance,
                        f"sudo docker rm -f {container_name} >/dev/null 2>&1 || true",
                    )
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"+ warning: failed to remove container {container_name}: {exc}",
                        flush=True,
                    )
            _apply_on_finish(instance=instance, cfg=cfg)
    if exit_code != 0:
        msg = f"Remote run exited with status {exit_code}"
        if failure_tail:
            msg += (
                f"\n--- last {FAILURE_TAIL_LINES} lines of remote output ---\n"
                f"{failure_tail}\n"
                f"--- end remote output ---"
            )
        raise RuntimeError(msg)


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


def _split_ephemeral_name(name: str) -> tuple[str, str]:
    """Best-effort reverse of :func:`_make_ephemeral_name`: ``runplz-<app>-<fn>-<uuid8>``.

    The user's app / function names can themselves contain hyphens (they're
    slugified but hyphens survive), so we can't perfectly unambiguously split.
    We trim the ``runplz-`` prefix and the trailing uuid, then take the final
    remaining segment as the function name and everything before it as the app
    name — the common convention for ephemeral runs. Returns empty strings if
    the shape doesn't match.
    """
    if not name.startswith("runplz-"):
        return ("", "")
    core = name[len("runplz-") :]
    parts = core.split("-")
    if len(parts) < 3:
        return ("", "")
    # Drop the uuid8 suffix.
    parts = parts[:-1]
    if len(parts) < 2:
        return ("", "")
    fn = parts[-1]
    app = "-".join(parts[:-1])
    return (app, fn)


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
    _wait_until_ssh_reachable so we stop probing dead boxes early.

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
    in `_wait_until_ssh_reachable` until the 20-minute deadline. Best-
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
       applied via `runplz._selector.pick_machine`.

    Raises only if the picker finds no matches.
    """
    # Build the list of candidate types to pass to `brev create`. Brev's
    # CLI natively supports repeated `--type` flags for multi-provider
    # fallback (if A fails on Nebius, it tries B on OCI, etc.). We feed
    # the selector's top-N ranked candidates when auto-picking; a
    # user-pinned `instance_type` is always the one-and-only.
    if cfg is not None and cfg.instance_type is not None:
        instance_types = [cfg.instance_type]
    else:
        n = cfg.instance_type_fallback_count if cfg is not None else 1
        instance_types = _pick_instance_types(function, n=n)
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


def _brev_capture(
    cmd: list,
    *,
    timeout: int = _BREV_DEFAULT_TIMEOUT_S,
    retry_waits: tuple = _BREV_DEFAULT_RETRIES,
    label: Optional[str] = None,
) -> subprocess.CompletedProcess:
    """Run a `brev` subcommand with transient-error retries.

    Transient error patterns are checked against combined stdout+stderr
    (see _BREV_TRANSIENT_PATTERNS). Non-transient failures terminate
    immediately with the CompletedProcess returned for caller inspection.
    A transient failure on the final attempt returns the last
    CompletedProcess too — the caller can decide whether that's fatal.

    `subprocess.TimeoutExpired` is always treated as transient.
    """
    label = label or " ".join(str(c) for c in cmd[:3])
    last: Optional[subprocess.CompletedProcess] = None
    total_attempts = len(retry_waits)
    for attempt, wait_s in enumerate(retry_waits, start=1):
        if wait_s:
            time.sleep(wait_s)
        started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        started = time.monotonic()
        print(
            f"+ {label} attempt {attempt}/{total_attempts} started {started_at}",
            flush=True,
        )
        try:
            last = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            elapsed_s = time.monotonic() - started
            print(
                f"+ {label} attempt {attempt}/{total_attempts} timed out "
                f"after {elapsed_s:.1f}s (timeout={timeout}s)",
                flush=True,
            )
            if attempt < total_attempts:
                print(
                    f"+ {label} attempt {attempt}/{total_attempts} will retry",
                    flush=True,
                )
                continue
            raise RuntimeError(
                f"`{label}` timed out after {timeout}s on all {total_attempts} attempts."
            ) from None
        elapsed_s = time.monotonic() - started
        stdout = str(last.stdout or "")
        stderr = str(last.stderr or "")
        print(
            f"+ {label} attempt {attempt}/{total_attempts} finished "
            f"rc={last.returncode} elapsed={elapsed_s:.1f}s",
            flush=True,
        )
        if (last.returncode != 0 or attempt > 1) and stdout.strip():
            print(f"+ {label} attempt {attempt} stdout:\n{stdout.rstrip()}", flush=True)
        if (last.returncode != 0 or attempt > 1) and stderr.strip():
            print(f"+ {label} attempt {attempt} stderr:\n{stderr.rstrip()}", flush=True)
        if last.returncode == 0:
            if attempt > 1:
                print(f"+ {label} succeeded on attempt {attempt}", flush=True)
            return last
        # Coerce to str in case stderr/stdout are Mocks (test harness) or
        # None (some subprocess configurations) — we only want the text.
        err = stderr + stdout
        # Early-bail on org/config-gap errors that retrying can never fix —
        # otherwise we burn the full retry budget on guaranteed-fail attempts
        # and delay the real failure (issue #62). Caller still gets the
        # CompletedProcess so it can produce a reframed message.
        if _looks_non_retriable(err):
            print(
                f"+ {label} attempt {attempt}/{total_attempts} hit "
                f"non-retriable error; bailing out (no point retrying org/config gaps)",
                flush=True,
            )
            return last
        if _looks_transient(err) and attempt < total_attempts:
            print(
                f"+ {label} attempt {attempt}/{total_attempts} hit transient error; retrying",
                flush=True,
            )
            continue
        # Non-transient, or final attempt of a transient: return for caller.
        return last
    # Unreachable — the loop either returns or raises.
    assert last is not None  # for type checkers
    return last


def _brev_sh(
    cmd: list,
    *,
    timeout: int = _BREV_DEFAULT_TIMEOUT_S,
    retry_waits: tuple = _BREV_DEFAULT_RETRIES,
    label: Optional[str] = None,
):
    """Analogue of `_sh` for brev CLI calls: runs with retries, prints the
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

    Always best-effort: we never want box-cleanup to swallow a real error
    from the try block. Transient Brev API errors get retries (via
    _brev_capture) so a single flaky call doesn't leak a billed box;
    any final failure prints a loud warning and moves on.
    """
    if cfg.on_finish == "leave":
        return
    action = cfg.on_finish  # "stop" or "delete"
    print(f"+ on_finish={action}: running `brev {action} {instance}`", flush=True)
    try:
        r = _brev_capture(
            ["brev", action, instance],
            timeout=120,
            label=f"brev {action} {instance}",
        )
    except Exception as exc:  # noqa: BLE001
        print(
            f"+ warning: `brev {action} {instance}` raised {type(exc).__name__}: {exc}. "
            f"The box may still be running — check `brev ls`.",
            flush=True,
        )
        return
    if r.returncode != 0:
        # Don't raise — we're in a finally block and must not mask the real
        # error. But DO shout: a silent non-zero here is a billing leak.
        print(
            f"+ warning: `brev {action} {instance}` exited {r.returncode}. "
            f"The box may still be running — check `brev ls`. "
            f"stderr: {(r.stderr or '').strip()[:500]}",
            flush=True,
        )
        return
    _verify_post_action_state(action, instance)


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


def _pick_instance_types(function, *, n: int = 1) -> list:
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
    from runplz._selector import pick_machines

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


def _pick_instance_type(function) -> Optional[str]:
    """Single-type picker. Back-compat wrapper around
    `_pick_instance_types(function, n=1)` — kept so older code /
    tests that expect a single TYPE string still work."""
    types = _pick_instance_types(function, n=1)
    return types[0] if types else None
