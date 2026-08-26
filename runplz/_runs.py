"""Local-side helpers for the ``runplz tail`` / ``runplz status`` CLI.

Reads the ``run.json`` manifest the orchestrator writes into the local
outputs dir (via ``_rsync_down``), pulls out the target host + remote
meta path, and shells to ``ssh`` to fetch ``last.log`` /
``events.ndjson`` / ``heartbeat.ndjson``.

This module owns nothing the dispatch path needs — it's pure consumer
of the data persisted by ``ssh_common`` so the CLI doesn't have to
remember run IDs or reconstruct ssh commands by hand (issue #57).
"""

import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from runplz.backends.ssh_common import (
    DEFAULT_KILL_TIMEOUT_S,
    KILL_SETTLE_S,
    REMOTE_META_DIRNAME,
    REMOTE_RUNS_DIR,
    _ssh_cmd_opts,
    build_kill_command,
)

META_FILENAME = "run.json"

# Run ids are generated as <ts>-<slug>-<slug>-<hex>, but --run-id is user
# input and lands inside a remote shell command, so pin it to a charset that
# cannot break out of the path it is interpolated into.
_SAFE_RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")


class ManifestNotFound(RuntimeError):
    """Raised when no local run manifest can be located."""


def find_local_manifest(outputs_dir: Path) -> Path:
    """Return the path to the most recent ``run.json`` under ``outputs_dir``.

    The orchestrator writes ``<remote_run.out>/.runplz/run.json`` and
    ``_rsync_down`` brings that into ``<outputs_dir>/.runplz/run.json``. We
    only ever have *one* such file locally (rsync overwrites on the next
    run), which is fine — the user wants the most-recent run by default.
    """
    p = outputs_dir / REMOTE_META_DIRNAME / META_FILENAME
    if not p.is_file():
        raise ManifestNotFound(
            f"No run manifest at {p}. Has the run started yet? "
            f"`runplz tail`/`status` reads the manifest produced by the most "
            f"recent dispatch into this outputs dir."
        )
    return p


def read_manifest(outputs_dir: Path) -> dict:
    return json.loads(find_local_manifest(outputs_dir).read_text())


def resolve_target_and_meta(
    *,
    outputs_dir: Path,
    host_override: Optional[str],
    run_id_override: Optional[str],
) -> tuple[str, str, dict]:
    """Resolve ``(target, remote_meta_path, manifest_or_none)``.

    Precedence: explicit ``--run-id`` requires an explicit ``--host``
    (otherwise we have no idea where to ssh). Without ``--run-id`` we read
    the local manifest to pick up the host + meta dir from the most-recent
    dispatch.
    """
    if run_id_override:
        if not host_override:
            raise RuntimeError("--run-id requires --host (no manifest lookup possible)")
        if not _SAFE_RUN_ID_RE.match(run_id_override):
            raise RuntimeError(
                f"--run-id {run_id_override!r} is not a valid run id "
                f"(letters, digits, dot, dash, underscore only)"
            )
        meta = f"~/{REMOTE_RUNS_DIR}/{run_id_override}/out/{REMOTE_META_DIRNAME}"
        return (host_override, meta, {})
    manifest = read_manifest(outputs_dir)
    target = host_override or manifest.get("target") or ""
    if not target:
        raise RuntimeError(f"manifest at {outputs_dir} has no target host; pass --host to override")
    meta = (manifest.get("remote_paths") or {}).get("meta")
    if not meta:
        run_id = manifest.get("run_id") or ""
        if not run_id:
            raise RuntimeError("manifest is missing both remote_paths.meta and run_id")
        meta = f"~/{REMOTE_RUNS_DIR}/{run_id}/out/{REMOTE_META_DIRNAME}"
    return (target, meta, manifest)


def remote_shell_path(path: str) -> str:
    """Make a manifest path usable inside a remote shell command.

    The manifest records display paths like ``~/runplz-runs/<id>/out/.runplz``.
    Tilde expansion happens before any quoting, so a ``~/`` path is literal in
    both ``'...'`` and ``"..."`` — the file is simply never found. ``$HOME``
    does expand inside double quotes, so callers can interpolate the result
    into ``"..."`` and get the directory they meant.
    """
    if path.startswith("~/"):
        return f"$HOME/{path[2:]}"
    return path


def tail(
    *,
    outputs_dir: Path,
    host_override: Optional[str],
    run_id_override: Optional[str],
    lines: int,
    follow: bool,
    port: Optional[int] = None,
) -> int:
    """Stream the remote ``last.log`` to stdout. Returns ssh's exit code."""
    target, meta, _ = resolve_target_and_meta(
        outputs_dir=outputs_dir,
        host_override=host_override,
        run_id_override=run_id_override,
    )
    log_path = remote_shell_path(f"{meta}/last.log")
    flags = "-F" if follow else f"-n {int(lines)}"
    remote_cmd = f'tail {flags} "{log_path}"'
    cmd = ["ssh", *_ssh_cmd_opts(port), target, remote_cmd]
    return subprocess.run(cmd).returncode


def status(
    *,
    outputs_dir: Path,
    host_override: Optional[str],
    run_id_override: Optional[str],
    port: Optional[int] = None,
) -> int:
    """Print a one-screen summary of the most recent run's state."""
    target, meta, manifest = resolve_target_and_meta(
        outputs_dir=outputs_dir,
        host_override=host_override,
        run_id_override=run_id_override,
    )
    # One ssh round-trip pulls last events line, last heartbeat line, and
    # an event count so we don't pay 3x ssh latency for a status check.
    ev_q = f'"{remote_shell_path(f"{meta}/events.ndjson")}"'
    hb_q = f'"{remote_shell_path(f"{meta}/heartbeat.ndjson")}"'
    remote_cmd = (
        f"echo '---LAST_EVENT---'; tail -n 1 {ev_q} 2>/dev/null || true; "
        f"echo '---LAST_HEARTBEAT---'; tail -n 1 {hb_q} 2>/dev/null || true; "
        f"echo '---EVENT_COUNT---'; wc -l < {ev_q} 2>/dev/null || echo 0; "
        f"echo '---END---'"
    )
    cmd = ["ssh", *_ssh_cmd_opts(port), target, remote_cmd]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"ssh to {target} failed (rc={r.returncode})")
        if r.stderr:
            print(r.stderr.strip())
        return r.returncode
    sections = _parse_status_sections(r.stdout)
    print(_format_status(target=target, manifest=manifest, sections=sections))
    return 0


def kill(
    *,
    outputs_dir: Path,
    host_override: Optional[str],
    run_id_override: Optional[str],
    timeout_s: int = DEFAULT_KILL_TIMEOUT_S,
    escalate: bool = True,
    first_signal: str = "TERM",
    port: Optional[int] = None,
) -> int:
    """Stop a detached run and report what happened to it.

    Idempotent by design (issue #67): killing a run that already exited is a
    normal outcome of "make sure nothing is still burning GPU hours", not an
    error, so an already-dead run prints its state and returns 0.
    """
    target, meta, manifest = resolve_target_and_meta(
        outputs_dir=outputs_dir,
        host_override=host_override,
        run_id_override=run_id_override,
    )
    run_id = manifest.get("run_id") or run_id_override or ""
    # The manifest is a file on disk; don't let a hand-edited run_id reach the
    # remote shell. It is only used to label the killed_by_user event.
    if not _SAFE_RUN_ID_RE.match(run_id or ""):
        run_id = ""
    remote_cmd = build_kill_command(
        remote_shell_path(meta),
        run_id=run_id,
        timeout_s=timeout_s,
        escalate=escalate,
        first_signal=first_signal,
    )
    cmd = ["ssh", *_ssh_cmd_opts(port), target, remote_cmd]
    # The remote side owns the signal/poll/escalate clock; give ssh enough
    # headroom to outlive the worst case rather than cutting it off mid-dance.
    ssh_timeout = timeout_s + KILL_SETTLE_S + 60
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=ssh_timeout)
    except subprocess.TimeoutExpired:
        print(f"ssh to {target} timed out after {ssh_timeout}s", file=sys.stderr)
        return 2
    if r.returncode != 0:
        print(f"ssh to {target} failed (rc={r.returncode})", file=sys.stderr)
        if r.stderr:
            print(r.stderr.strip(), file=sys.stderr)
        return r.returncode
    sections = _parse_status_sections(r.stdout)
    fields = _parse_kv_block(sections.get("SUMMARY", ""))
    print(_format_kill(target=target, run_id=run_id, fields=fields, sections=sections))
    return 0


def _parse_kv_block(block: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in block.splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            out[key.strip()] = value.strip()
    return out


def _format_kill(
    *, target: str, run_id: str, fields: dict[str, str], sections: dict[str, str]
) -> str:
    initial = fields.get("initial", "unknown")
    final = fields.get("final", "unknown")
    signalled = fields.get("signalled") == "1"
    escalated = fields.get("escalated") == "1"
    container = fields.get("container", "")
    container_state = fields.get("container_state", "none")

    lines = [f"target:     {target}"]
    if run_id:
        lines.append(f"run:        {run_id}")
    pid = fields.get("pid") or "-"
    pgid = fields.get("pgid") or "-"
    lines.append(f"bootstrap:  pid={pid} pgid={pgid}")
    if container:
        lines.append(f"container:  {container} ({container_state})")

    if not signalled:
        lines.append(f"action:     nothing to kill — process was already {initial}")
    elif final in {"dead", "missing", "zombie"} and container_state != "running":
        how = "SIGTERM, escalated to SIGKILL" if escalated else "SIGTERM"
        lines.append(f"action:     stopped with {how}")
    else:
        lines.append(
            f"action:     signalled, but the run still reports {final!r} — inspect manually"
        )
    lines.append(f"state:      {initial} -> {final}")

    gpu = fields.get("gpu_mem_used", "")
    if gpu:
        used = [g for g in gpu.split(",") if g]
        pretty = ", ".join(f"gpu{i}={m}MiB" for i, m in enumerate(used))
        lines.append(f"gpu memory: {pretty}")

    heartbeat = (sections.get("HEARTBEAT") or "").strip()
    if heartbeat:
        lines.append(f"heartbeat:  {_heartbeat_age(heartbeat)}")

    logtail = (sections.get("LOGTAIL") or "").strip()
    if logtail:
        lines.append("last log:")
        lines.extend(f"  {line}" for line in logtail.splitlines())
    return "\n".join(lines)


def _heartbeat_age(line: str) -> str:
    """Render the last heartbeat as ``<ts> (<age> ago)``, matching `status`."""
    try:
        ts = json.loads(line).get("ts", "")
    except (json.JSONDecodeError, AttributeError):
        return line[:200]
    if not ts:
        return line[:200]
    return f"{ts}{_age_str(ts)}"


def _parse_status_sections(stdout: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    current = None
    buf: list[str] = []
    for line in stdout.splitlines():
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


_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


def _parse_iso_z(ts: str) -> Optional[datetime]:
    if not ts or not _ISO_RE.match(ts):
        return None
    return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def _format_status(*, target: str, manifest: dict, sections: dict[str, str]) -> str:
    """Render the user-visible status output.

    Format is intentionally one-line-per-fact so a human can scan it without
    parsing JSON or inventing a UI.
    """
    last_event_raw = sections.get("LAST_EVENT", "")
    last_hb_raw = sections.get("LAST_HEARTBEAT", "")
    count_raw = (sections.get("EVENT_COUNT", "") or "0").strip()

    lines = [
        f"target: {target}",
        f"run_id: {manifest.get('run_id') or '(unknown — no manifest)'}",
        f"backend: {manifest.get('backend') or '(unknown)'}",
        f"function: {manifest.get('function') or '(unknown)'}",
    ]

    if last_event_raw:
        try:
            evt = json.loads(last_event_raw)
            ts = evt.get("ts", "")
            evt_name = evt.get("event", "?")
            age = _age_str(ts)
            extra = ""
            if "exit_code" in evt:
                extra = f" exit_code={evt['exit_code']}"
            lines.append(f"last event: {evt_name}{extra} at {ts}{age}")
        except json.JSONDecodeError:
            lines.append(f"last event (unparsed): {last_event_raw[:200]}")
    else:
        lines.append("last event: (none recorded)")

    if last_hb_raw:
        try:
            hb = json.loads(last_hb_raw)
            ts = hb.get("ts", "")
            age = _age_str(ts)
            lines.append(f"last heartbeat: {ts}{age}")
        except json.JSONDecodeError:
            lines.append(f"last heartbeat (unparsed): {last_hb_raw[:200]}")
    else:
        lines.append("last heartbeat: (none yet)")

    lines.append(f"events recorded: {count_raw}")
    return "\n".join(lines)


def _age_str(iso: str) -> str:
    parsed = _parse_iso_z(iso)
    if parsed is None:
        return ""
    delta = datetime.now(timezone.utc) - parsed
    secs = int(delta.total_seconds())
    if secs < 0:
        return ""
    if secs < 60:
        return f" ({secs}s ago)"
    if secs < 3600:
        return f" ({secs // 60}m {secs % 60}s ago)"
    return f" ({secs // 3600}h {(secs % 3600) // 60}m ago)"
