"""Pieces shared by the direct-cloud provisioning backends (issue #25).

GCP and AWS differ in almost everything about *how* you ask for a box, and
almost nothing about *what* runplz needs from one: a name, a shape that
matches the function's resource request, and a way to make it go away
again. This module holds that common middle — naming, GPU-shape lookup,
CLI invocation with useful errors, and dry-run rendering — so each driver
is left with only its own cloud's vocabulary.

Both drivers shell out to the vendor CLI rather than a Python SDK. The core
stays dependency-free, and `tests/conftest.py` already guards `gcloud` and
`aws` as billed commands, so a test that forgets to mock cannot quietly
launch a paid instance.
"""

import json
import re
import shlex
import subprocess
import uuid
from typing import Optional

# Instance names have to be DNS-ish on both clouds: lowercase alphanumerics
# and dashes, starting with a letter.
_NAME_SAFE_RE = re.compile(r"[^a-z0-9-]+")


class CloudCliError(RuntimeError):
    """A vendor CLI call failed. Carries the stderr that explains why."""


def make_instance_name(app_name: str, fn_name: str) -> str:
    """Build a unique, DNS-safe instance name tagged as runplz's.

    The `runplz-` prefix and the random suffix both matter: the prefix makes
    a leaked box obvious in a console full of unrelated instances, and the
    suffix means two concurrent runs of the same function never collide on a
    name (which on both clouds is an error, not a queue).
    """

    def slug(value: str, limit: int) -> str:
        out = _NAME_SAFE_RE.sub("-", (value or "").lower()).strip("-")
        return (out[:limit].strip("-")) or "x"

    return f"runplz-{slug(app_name, 18)}-{slug(fn_name, 18)}-{uuid.uuid4().hex[:8]}"


def render_command(cmd: list) -> str:
    """Render an argv list as a copy-pasteable shell command."""
    return " ".join(shlex.quote(str(part)) for part in cmd)


def run_cli(
    cmd: list,
    *,
    label: str,
    timeout: int = 300,
    dry_run: bool = False,
    parse_json: bool = False,
    check: bool = True,
):
    """Invoke a vendor CLI, printing the command so runs are auditable.

    On `dry_run` the command is printed and *not* executed, and the call
    returns None — the driver keeps walking its own logic so you can see the
    whole sequence a config would produce without creating anything.

    Failures raise `CloudCliError` carrying stderr. The vendor CLIs put the
    actionable part (quota exceeded, no capacity in this zone, missing
    permission) on stderr and nothing useful in the exit code, so swallowing
    it would turn a fixable problem into a mystery.
    """
    printable = render_command(cmd)
    if dry_run:
        print(f"+ [dry-run] {printable}", flush=True)
        return None

    print(f"+ {printable}", flush=True)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise CloudCliError(f"`{label}` timed out after {timeout}s.") from exc
    except FileNotFoundError as exc:
        raise CloudCliError(
            f"`{cmd[0]}` not found on PATH. Install the {cmd[0]} CLI and "
            f"authenticate it before using this backend."
        ) from exc

    if check and r.returncode != 0:
        raise CloudCliError(
            f"`{label}` exited {r.returncode}.\n"
            f"  command: {printable}\n"
            f"  stderr: {(r.stderr or '').strip()[:2000]}"
        )
    if parse_json:
        text = (r.stdout or "").strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise CloudCliError(
                f"`{label}` returned output that is not JSON: {text[:500]}"
            ) from exc
    return r


# --- GPU shape lookup ----------------------------------------------------
#
# Deliberately a static table rather than a live pricing/availability query.
# A lookup would be one more failure mode on the critical path, and #25
# explicitly defers the cost-minimising selector. Users who want an exact
# shape pass machine_type / instance_type and skip all of this.

# Function.gpu label -> (gcp accelerator, gcp machine family, vram GB)
GCP_GPUS = {
    "T4": ("nvidia-tesla-t4", "n1-standard", 16),
    "V100": ("nvidia-tesla-v100", "n1-standard", 16),
    "L4": ("nvidia-l4", "g2-standard", 24),
    "A100-40GB": ("nvidia-tesla-a100", "a2-highgpu", 40),
    "A100-80GB": ("nvidia-a100-80gb", "a2-ultragpu", 80),
    "H100": ("nvidia-h100-80gb", "a3-highgpu", 80),
}

# Function.gpu label -> (aws instance family, vram GB)
AWS_GPUS = {
    "T4": ("g4dn", 16),
    "V100": ("p3", 16),
    "A10G": ("g5", 24),
    "L4": ("g6", 24),
    "L40S": ("g6e", 48),
    "A100-40GB": ("p4d", 40),
    "A100-80GB": ("p4de", 80),
    "H100": ("p5", 80),
}


def resolve_gpu_label(function, table: dict) -> Optional[str]:
    """Pick the GPU label to provision for, honouring min_gpu_memory.

    `gpu=` wins when set. Otherwise `min_gpu_memory` selects the smallest
    GPU in the table that fits, so `min_gpu_memory=40` doesn't silently
    land on a T4 that cannot hold the model.
    """
    if getattr(function, "gpu", None):
        label = function.gpu
        # Modal-style "A100-80GB:4" count suffix — the count is carried by
        # min_gpus, so strip it here.
        label = label.split(":", 1)[0]
        if label not in table:
            raise CloudCliError(
                f"gpu={function.gpu!r} has no mapping for this cloud. "
                f"Known: {', '.join(sorted(table))}. "
                f"Pass an explicit machine type to use a shape runplz "
                f"doesn't know about."
            )
        return label
    need_vram = getattr(function, "min_gpu_memory", None)
    if not need_vram:
        return None
    fitting = sorted((vram, name) for name, (*_rest, vram) in table.items() if vram >= need_vram)
    if not fitting:
        raise CloudCliError(
            f"min_gpu_memory={need_vram} GB exceeds every GPU runplz knows "
            f"for this cloud (largest: {max(v for *_r, v in table.values())} GB)."
        )
    return fitting[0][1]


def gpu_count(function) -> int:
    return max(1, int(getattr(function, "min_gpus", None) or 1))
