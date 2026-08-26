"""Docker vocabulary shared by every backend that runs a container.

The labels runplz stamps on its containers are how `runplz ps` finds them
again. They used to be string literals in four files — written in
`local.py` and `ssh_common.py`, read back in `local.py` and `ssh.py` — so
renaming one would have quietly broken `ps` rather than failing anything.
Producer and consumer now share these constants.

`local` runs docker on this machine and `ssh`/`brev`/`gcp`/`aws` run it
over ssh, but the container shape and the `docker ps` output are identical
either way, which is why the parsing lives here rather than in
`ssh_common` (which `local` has no business importing).
"""

import json
import shlex
from typing import Optional

# The marker label. Every `docker ps` lookup filters on it so runplz never
# reports — or cleans up — a container the user started by hand.
RUNPLZ_LABEL = "runplz=1"
PS_FILTER = f"label={RUNPLZ_LABEL}"
APP_LABEL_KEY = "runplz-app"
FUNCTION_LABEL_KEY = "runplz-function"

# `docker ps` emits one JSON object per line in this format.
PS_FORMAT = "{{json .}}"


def label_args(app_name: Optional[str], fn_name: str) -> list:
    """`docker run` label flags as argv, for local execution."""
    args = ["--label", RUNPLZ_LABEL]
    if app_name is not None:
        args += ["--label", f"{APP_LABEL_KEY}={app_name}"]
    args += ["--label", f"{FUNCTION_LABEL_KEY}={fn_name}"]
    return args


def label_flags(app_name: Optional[str], fn_name: str) -> str:
    """The same flags as a shell string, for `docker run` over ssh."""
    parts = [f"--label {RUNPLZ_LABEL}"]
    if app_name is not None:
        parts.append(f"--label {shlex.quote(f'{APP_LABEL_KEY}={app_name}')}")
    parts.append(f"--label {shlex.quote(f'{FUNCTION_LABEL_KEY}={fn_name}')}")
    return " ".join(parts)


def ps_args() -> list:
    """The `docker ps` arguments used by every backend's `list_jobs`."""
    return ["ps", "--filter", PS_FILTER, "--format", PS_FORMAT]


def parse_labels(labels: str) -> dict:
    """Parse docker's comma-separated `k=v` label string."""
    out = {}
    for part in (labels or "").split(","):
        if "=" in part:
            key, value = part.split("=", 1)
            out[key.strip()] = value.strip()
    return out


def parse_ps_rows(stdout: str, *, backend: str, name_prefix: str = "") -> list[dict]:
    """Turn `docker ps --format '{{json .}}'` output into `runplz ps` rows.

    Malformed lines are skipped rather than raising: `ps` is a read-only
    convenience, and one unparseable line should not cost the user the
    whole table.

    ``name_prefix`` distinguishes otherwise-identical container names across
    hosts — the ssh backend passes the target so two boxes running `train`
    don't render as the same row.
    """
    rows = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            continue
        labels = parse_labels(raw.get("Labels", ""))
        name = raw.get("Names", "") or raw.get("ID", "")
        rows.append(
            {
                "backend": backend,
                "name": f"{name_prefix}{name}",
                "app": labels.get(APP_LABEL_KEY, ""),
                "function": labels.get(FUNCTION_LABEL_KEY, ""),
                "started": raw.get("CreatedAt", "") or raw.get("RunningFor", ""),
                "status": raw.get("Status", ""),
            }
        )
    return rows


def looks_like_daemon_down(err: str) -> bool:
    """Match the canonical 'docker daemon not running' stderr signatures.

    Worth distinguishing from a real failure: on a dev machine docker is
    often simply off, and `runplz ps` reporting "0 local jobs" is the
    correct answer there, not an error.
    """
    low = (err or "").lower()
    return any(
        sig in low
        for sig in (
            "cannot connect to the docker daemon",
            "is the docker daemon running",
            "error during connect",
            "docker desktop is not running",
        )
    )
