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
import uuid
from pathlib import Path
from typing import Optional

from runplz._selector import Candidate, pick_machine
from runplz.backends._ssh_common import (
    FAILURE_TAIL_LINES,
    _build_image,
    _container_running,
    _ensure_docker,
    _ensure_remote_rsync,
    _fetch_failure_tail,
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
    make_container_name,
)

# Re-exports so older test patches and external code that patched these
# keep working without a hard rename.
_ = (  # noqa: F841 — held for test-mocking compatibility
    _container_running,
    _raise_for_runtime_cap,
    _render_ops_script,
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
            _wait_until_ssh_reachable(instance, refresh_callback=_refresh_ssh)

            repo = app._repo_root
            host_out = (repo / outputs_dir).resolve()
            host_out.mkdir(parents=True, exist_ok=True)

            if cfg.mode == "container":
                # Pre-built container images (e.g. pytorch/pytorch) don't
                # ship with rsync. Install it before the first rsync call.
                _ensure_remote_rsync(instance)
            _rsync_up(repo, instance)
            rel_script = Path(function.module_file).resolve().relative_to(repo)

            if cfg.mode == "container":
                exit_code = _run_container_mode(
                    target=instance,
                    function=function,
                    rel_script=str(rel_script),
                    args=args,
                    kwargs=kwargs,
                    max_runtime_seconds=cfg.max_runtime_seconds,
                )
            elif cfg.use_docker:
                _ensure_docker(instance)
                gpu_flag = "--gpus all" if _remote_has_nvidia(instance) else ""
                container_name = make_container_name(function.name)
                _build_image(instance, function.image)
                _run_container_detached(
                    target=instance,
                    container_name=container_name,
                    function=function,
                    rel_script=str(rel_script),
                    args=args,
                    kwargs=kwargs,
                    gpu_flag=gpu_flag,
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
                    max_runtime_seconds=cfg.max_runtime_seconds,
                )
            _rsync_down(instance, host_out)
        finally:
            # Fetch a log tail BEFORE container/box cleanup — afterwards
            # the logs are gone (docker rm wipes container state; brev
            # stop/delete makes the box unreachable). Only do this on
            # failure.
            failure_tail = ""
            if exit_code is not None and exit_code != 0:
                failure_tail = _fetch_failure_tail(target=instance, container_name=container_name)
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


def _instance_exists(name: str) -> bool:
    """True iff `brev ls` lists an instance with this name.

    Raises RuntimeError if the `brev` CLI itself failed (bad auth, network,
    Brev API outage, malformed JSON) — we refuse to silently treat a degraded
    listing as "instance doesn't exist," because the caller's fallback is to
    auto-create a *new billed box*, which may duplicate one the user already
    has.

    Returns False only when `brev ls` succeeded but the target name is
    definitively not in the list (including the documented `null` / empty
    shapes Brev returns for empty orgs).
    """
    r = subprocess.run(["brev", "ls", "--json"], capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        raise RuntimeError(
            f"`brev ls --json` failed with exit code {r.returncode}. "
            f"Refusing to continue — if we assumed the instance was missing we'd "
            f"auto-create a duplicate billed box. Run `brev login` / check the "
            f"`brev` CLI and retry. stderr: {(r.stderr or '').strip()[:500]}"
        )
    try:
        data = json.loads(r.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"`brev ls --json` returned unparseable JSON. Refusing to continue "
            f"(see _instance_exists docstring for why). stdout head: "
            f"{(r.stdout or '').strip()[:200]!r}"
        ) from exc
    if data is None:
        return False
    if isinstance(data, list):
        instances = data
    elif isinstance(data, dict):
        instances = data.get("instances", []) or []
    else:
        raise RuntimeError(
            f"`brev ls --json` returned an unexpected shape ({type(data).__name__}). "
            f"Refusing to continue (see _instance_exists docstring)."
        )
    return any(i.get("name") == name for i in instances)


# Field names we accept when pulling a power-state string out of a
# `brev ls --json` row. Brev has used more than one over time.
_BREV_STATUS_FIELDS = ("status", "state", "power_state", "lifecycle_status")

# Status values treated as "needs a `brev start` first." Lower-cased for
# case-insensitive matching.
_BREV_STOPPED_STATES = {"stopped", "paused", "hibernated", "suspended"}


def _instance_status(name: str) -> Optional[str]:
    """Return the raw status string for `name` from `brev ls --json`, or
    None if the instance isn't listed or no recognized status field is
    present. Best-effort — used only to decide whether we need to call
    `brev start` before SSH."""
    try:
        r = subprocess.run(["brev", "ls", "--json"], capture_output=True, text=True, timeout=60)
    except Exception:  # noqa: BLE001
        return None
    if r.returncode != 0:
        return None
    try:
        data = json.loads(r.stdout)
    except json.JSONDecodeError:
        return None
    if data is None:
        return None
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        rows = data.get("instances", []) or []
    else:
        return None
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get("name") != name:
            continue
        for key in _BREV_STATUS_FIELDS:
            v = row.get(key)
            if v:
                return str(v)
        return None
    return None


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
        r = subprocess.run(
            ["brev", "start", name],
            check=False,
            capture_output=True,
            text=True,
            timeout=600,
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


def _create_instance(name: str, *, cfg=None, image=None, function=None):
    """Provision a Brev instance.

    Picks the instance type in this order:
    1. `cfg.instance_type` — explicit user override (if set).
    2. Cheapest match from `brev search` driven by `function`'s resource
       constraints, with the 5% cost-tolerance + availability tiebreaker
       applied via `runplz._selector.pick_machine`.

    Raises only if the picker finds no matches.
    """
    instance_type: Optional[str] = cfg.instance_type if cfg is not None else None
    if instance_type is None:
        instance_type = _pick_instance_type(function)
        if instance_type is None:
            raise RuntimeError(
                "`brev search` returned no matching instances. Loosen the "
                "function's resource constraints, pass an explicit "
                "`instance_type=...` on BrevConfig, or pre-create the instance."
            )

    cmd = ["brev", "create", name, "--type", instance_type]
    if function is not None and function.min_disk is not None:
        cmd += ["--min-disk", str(function.min_disk)]
    if cfg is not None and cfg.mode == "container":
        # `image.base` is guaranteed to be set here — Dockerfile images are
        # rejected at function-decoration time by runplz.app's validator.
        cmd += ["--mode", "container", "--container-image", image.base]
    _sh(cmd)


_BREV_TRANSIENT_PATTERNS = (
    "context deadline exceeded",
    "rpc error",
    "connection reset",
    "connection refused",
    "i/o timeout",
    "temporary failure in name resolution",
    "service unavailable",
    "503",
    "504",
    "eof",
)

# Attempts, seconds to wait before each retry. First attempt has no wait;
# each subsequent wait grows. A total of ~21s spent on retries is enough to
# ride out the Brev API hiccups seen in the wild without blocking for too
# long if the error is genuine.
_REFRESH_RETRY_WAITS = (0, 3, 6, 12)


def _refresh_ssh():
    """Run `brev refresh`, retrying on transient Brev API errors.

    Issue #28: `brev refresh` periodically returns `rpc error: context
    deadline exceeded` when the Brev backend API is slow (common during
    8×A100 Denvr/OCI provisioning). That used to be fatal, leaving a
    freshly-provisioned billed box running. Retry up to 4 attempts total
    across ~21 seconds; only raise if every attempt fails with what looks
    like a transient error, or if we hit a non-transient failure.
    """
    import time

    last_stderr = ""
    for attempt, wait_s in enumerate(_REFRESH_RETRY_WAITS, start=1):
        if wait_s:
            time.sleep(wait_s)
        r = subprocess.run(
            ["brev", "refresh"],
            capture_output=True,
            text=True,
        )
        if r.returncode == 0:
            if attempt > 1:
                print(f"+ brev refresh succeeded on attempt {attempt}", flush=True)
            return
        err = ((r.stderr or "") + (r.stdout or "")).lower()
        last_stderr = (r.stderr or r.stdout or "").strip()
        if _looks_transient(err) and attempt < len(_REFRESH_RETRY_WAITS):
            print(
                f"+ brev refresh attempt {attempt}/{len(_REFRESH_RETRY_WAITS)} "
                f"hit transient error; retrying. stderr: {last_stderr[:200]}",
                flush=True,
            )
            continue
        break
    raise RuntimeError(
        f"`brev refresh` failed after {len(_REFRESH_RETRY_WAITS)} attempts. "
        f"Last stderr: {last_stderr[:500]}"
    )


def _looks_transient(err: str) -> bool:
    low = err.lower()
    return any(pat in low for pat in _BREV_TRANSIENT_PATTERNS)


def _apply_on_finish(*, instance: str, cfg) -> None:
    """Stop / delete / leave the Brev box per `cfg.on_finish`.

    Always best-effort: we never want box-cleanup to swallow a real error
    from the try block. Failures here print a warning and move on.
    """
    if cfg.on_finish == "leave":
        return
    action = cfg.on_finish  # "stop" or "delete"
    print(f"+ on_finish={action}: running `brev {action} {instance}`", flush=True)
    try:
        r = subprocess.run(
            ["brev", action, instance],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
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


def _pick_instance_type(function) -> Optional[str]:
    """Run `brev search gpu` (or cpu) with filters from `function`'s
    resource requests; return the best matching TYPE string.

    Brev's own `--sort price` gives us cheapest-first; we post-process
    through `pick_machine` so that when the top few candidates are
    within 5% on price, we prefer whichever exposes the lowest
    availability signal (ETA / supply). If no price or availability
    fields are present we fall back to the original first-row behavior.

    Returns None if no match.
    """
    import shlex as _shlex

    mode = "gpu" if function.gpu is not None else "cpu"
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
    print("+ " + " ".join(_shlex.quote(c) for c in cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        return None
    try:
        results = json.loads(r.stdout)
    except json.JSONDecodeError:
        return None
    if not isinstance(results, list) or not results:
        return None

    candidates = [_candidate_from_brev_row(row) for row in results]
    priced = [c for c in candidates if c is not None and c.hourly_usd is not None]
    if priced:
        choice = pick_machine(priced)
        if choice is not None:
            print(f"+ selector picked {choice.name!r}: {choice.reason}", flush=True)
            return choice.name

    # Fallback: price/name fields not exposed in this `brev search` shape.
    return _brev_row_type(results[0])
