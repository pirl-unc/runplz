"""Local-side helpers for the ``runplz tail`` / ``runplz status`` CLI.

Reads the ``run.json`` manifest the orchestrator writes into the local
outputs dir (via ``_rsync_down``), pulls out the target host + remote
meta path, and shells to ``ssh`` to fetch ``last.log`` /
``events.ndjson`` / ``heartbeat.ndjson``.

This module owns nothing the dispatch path needs — it's pure consumer
of the data persisted by ``_ssh_common`` so the CLI doesn't have to
remember run IDs or reconstruct ssh commands by hand (issue #57).
"""

import json
import re
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from runplz.backends._ssh_common import (
    REMOTE_META_DIRNAME,
    REMOTE_RUNS_DIR,
    _ssh_cmd_opts,
)

META_FILENAME = "run.json"


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
    log_path = f"{meta}/last.log"
    flags = "-F" if follow else f"-n {int(lines)}"
    remote_cmd = f"tail {flags} {shlex.quote(log_path)}"
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
    events_path = f"{meta}/events.ndjson"
    heartbeat_path = f"{meta}/heartbeat.ndjson"
    ev_q = shlex.quote(events_path)
    hb_q = shlex.quote(heartbeat_path)
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
