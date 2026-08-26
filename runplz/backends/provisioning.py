"""Pieces shared by the backends that provision a machine.

Brev, GCP and AWS differ in almost everything about *how* you ask for a box,
and almost nothing about *what* runplz needs from one: a name, a shape that
matches the function's resource request, and a way to make it go away again.
This module holds that common middle — naming, GPU-shape lookup, CLI
invocation with useful errors, and dry-run rendering — so each driver is
left with only its own provider's vocabulary.

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
from typing import NamedTuple, Optional

__all__ = [
    "CloudCliError",
    # Naming: two halves of one contract, they must agree.
    "INSTANCE_PREFIX",
    "make_instance_name",
    "split_instance_name",
    # Talking to a vendor CLI.
    "run_cli",
    "render_command",
    # Choosing a machine for a function's resource request.
    "GpuShapes",
    "GCP_GPUS",
    "AWS_GPUS",
    "resolve_gpu_label",
    "gpu_count",
    "pick_shape",
    # Giving the machine back.
    "apply_teardown",
]

# Instance names have to be DNS-ish on both clouds: lowercase alphanumerics
# and dashes, starting with a letter.
_NAME_SAFE_RE = re.compile(r"[^a-z0-9-]+")


class CloudCliError(RuntimeError):
    """A vendor CLI call failed. Carries the stderr that explains why."""


INSTANCE_PREFIX = "runplz-"
_SLUG_LIMIT = 18


def make_instance_name(app_name: str, fn_name: str) -> str:
    """Build a unique, DNS-safe instance name tagged as runplz's.

    Shape: ``runplz-<app>-<fn>-<uuid8>``. The prefix and the random suffix
    both matter — the prefix makes a leaked box obvious in a console full of
    unrelated instances, and the suffix means two concurrent runs of the same
    function never collide on a name (which every provider treats as an
    error, not a queue).

    Paired with :func:`split_instance_name`, which every provider's job
    listing uses to read an app/function back out of a name. The two must
    agree, so they live together.
    """

    def slug(value: str, limit: int) -> str:
        out = _NAME_SAFE_RE.sub("-", (value or "").lower()).strip("-")
        return (out[:limit].strip("-")) or "x"

    return (
        f"{INSTANCE_PREFIX}{slug(app_name, _SLUG_LIMIT)}-"
        f"{slug(fn_name, _SLUG_LIMIT)}-{uuid.uuid4().hex[:8]}"
    )


def split_instance_name(name: str) -> tuple:
    """Best-effort reverse of :func:`make_instance_name`.

    App and function names keep their hyphens through slugification, so the
    split can't be unambiguous. Trim the prefix and the trailing uuid, then
    treat the last remaining segment as the function and everything before it
    as the app — the common shape for a provisioned run. Returns empty
    strings when the name isn't one of ours.
    """
    if not name.startswith(INSTANCE_PREFIX):
        return ("", "")
    parts = name[len(INSTANCE_PREFIX) :].split("-")
    if len(parts) < 3:
        return ("", "")
    parts = parts[:-1]  # drop the uuid8 suffix
    if len(parts) < 2:
        return ("", "")
    return ("-".join(parts[:-1]), parts[-1])


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


class GpuShapes(NamedTuple):
    """What one GPU model looks like on one provider."""

    # GCP needs an accelerator name for shapes that don't bundle a GPU.
    # AWS has no equivalent, so it is None there.
    accelerator: Optional[str]
    family: str
    vram_gb: int
    # GPU count -> the exact machine type that sells it. None means the
    # family takes a separately-attached accelerator instead.
    shapes: Optional[dict]


# Machine types are enumerated, never derived. Arithmetic gets this wrong in
# two different ways: `f"{family}.xlarge"` invents p3.xlarge and
# g4dn.24xlarge, which do not exist, and g2-standard's numeric suffix is
# vCPUs rather than GPUs, so `4 * count` names a real machine that quietly
# carries one GPU instead of the eight that were asked for.
GCP_GPUS = {
    "T4": GpuShapes("nvidia-tesla-t4", "n1-standard", 16, None),
    "V100": GpuShapes("nvidia-tesla-v100", "n1-standard", 16, None),
    "L4": GpuShapes(
        "nvidia-l4",
        "g2-standard",
        24,
        {1: "g2-standard-4", 2: "g2-standard-24", 4: "g2-standard-48", 8: "g2-standard-96"},
    ),
    "A100-40GB": GpuShapes(
        "nvidia-tesla-a100",
        "a2-highgpu",
        40,
        {1: "a2-highgpu-1g", 2: "a2-highgpu-2g", 4: "a2-highgpu-4g", 8: "a2-highgpu-8g"},
    ),
    "A100-80GB": GpuShapes(
        "nvidia-a100-80gb",
        "a2-ultragpu",
        80,
        {1: "a2-ultragpu-1g", 2: "a2-ultragpu-2g", 4: "a2-ultragpu-4g", 8: "a2-ultragpu-8g"},
    ),
    "H100": GpuShapes("nvidia-h100-80gb", "a3-highgpu", 80, {8: "a3-highgpu-8g"}),
}

AWS_GPUS = {
    "T4": GpuShapes(None, "g4dn", 16, {1: "g4dn.xlarge", 4: "g4dn.12xlarge", 8: "g4dn.metal"}),
    "V100": GpuShapes(None, "p3", 16, {1: "p3.2xlarge", 4: "p3.8xlarge", 8: "p3.16xlarge"}),
    "A10G": GpuShapes(None, "g5", 24, {1: "g5.xlarge", 4: "g5.12xlarge", 8: "g5.48xlarge"}),
    "L4": GpuShapes(None, "g6", 24, {1: "g6.xlarge", 4: "g6.12xlarge", 8: "g6.48xlarge"}),
    "L40S": GpuShapes(None, "g6e", 48, {1: "g6e.xlarge", 4: "g6e.12xlarge", 8: "g6e.48xlarge"}),
    "A100-40GB": GpuShapes(None, "p4d", 40, {8: "p4d.24xlarge"}),
    "A100-80GB": GpuShapes(None, "p4de", 80, {8: "p4de.24xlarge"}),
    "H100": GpuShapes(None, "p5", 80, {8: "p5.48xlarge"}),
}


def pick_shape(label: str, count: int, table: dict, *, cloud: str) -> str:
    """Return the machine type selling exactly `count` of `label`'s GPU.

    Refuses a count the family does not offer rather than fabricating a name:
    a wrong-but-plausible type either fails deep inside the provider CLI or,
    worse, launches a box with fewer GPUs than the job asked for.
    """
    shapes = table[label].shapes
    if shapes is None:
        raise KeyError(label)
    if count not in shapes:
        raise CloudCliError(
            f"{cloud} sells {label} in {sorted(shapes)} GPU counts, not {count}. "
            f"Pass an explicit machine type to use a shape runplz doesn't know."
        )
    return shapes[count]


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
    fitting = sorted(
        (entry.vram_gb, name) for name, entry in table.items() if entry.vram_gb >= need_vram
    )
    if not fitting:
        raise CloudCliError(
            f"min_gpu_memory={need_vram} GB exceeds every GPU runplz knows "
            f"for this cloud (largest: {max(e.vram_gb for e in table.values())} GB)."
        )
    return fitting[0][1]


def gpu_count(function) -> int:
    return max(1, int(getattr(function, "min_gpus", None) or 1))


def apply_teardown(
    *,
    on_finish: str,
    target: str,
    run_action,
    check_hint: str,
    where: str = "",
) -> None:
    """Run a provisioning backend's teardown under one safety contract.

    Every provider spells the action differently (`brev delete`, `gcloud
    compute instances delete`, `aws ec2 terminate-instances`), but the rules
    around it are identical and are what actually protect the user's bill:

    - ``on_finish="leave"`` announces and does nothing.
    - Teardown never raises. It runs inside a ``finally``, so raising here
      would mask whatever really went wrong with the run.
    - But it never fails *quietly* either. A teardown that silently didn't
      happen is a box that bills until someone notices, so a failure prints
      loudly and tells the user exactly how to check.

    ``run_action`` receives the on_finish value and does the provider-specific
    work; raising from it is how it reports failure.
    """
    if on_finish == "leave":
        print(f"+ on_finish=leave: {target} left running{where}", flush=True)
        return
    try:
        run_action(on_finish)
    except Exception as exc:  # noqa: BLE001 - teardown must never propagate
        print(
            f"+ warning: on_finish={on_finish} failed for {target}: "
            f"{type(exc).__name__}: {exc}\n"
            f"  It may still exist and still be billing. Check: {check_hint}",
            flush=True,
        )
