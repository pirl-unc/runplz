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

import dataclasses
import json
import re
import shlex
import subprocess
import time
import uuid
from dataclasses import dataclass
from typing import Callable, NamedTuple, Optional

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
    "AWS_RETRY_POLICY",
    "AWS_TEARDOWN_POLICY",
    "GCP_TEARDOWN_POLICY",
    "GCP_RETRY_POLICY",
    "run_with_retries",
    "retry_budget_spent",
    "NO_RETRIES",
    "RetryPolicy",
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


# --- retrying a vendor CLI ------------------------------------------------
#
# Every provider's control plane is flaky in its own vocabulary. The attempt
# loop is identical; the *classification* is not — Brev's org/config gaps,
# GCP's ZONE_RESOURCE_POOL_EXHAUSTED and AWS's throttling look nothing like
# each other. So the loop lives here and each backend brings its own tables.


@dataclass(frozen=True)
class RetryPolicy:
    """When to try a vendor CLI again, and when not to bother.

    `waits` is the sleep *before* each attempt, so its length is the attempt
    count and its first entry is normally 0.

    `is_non_retriable` matters as much as `is_transient`: without it a run
    burns the whole budget on guaranteed-fail attempts and delays the real
    error by minutes (issue #62). An unclassified failure is neither — it
    stops immediately, because retrying an error we don't understand is a
    guess made at the user's expense.
    """

    waits: tuple = (0,)
    is_transient: Optional[Callable[[str], bool]] = None
    is_non_retriable: Optional[Callable[[str], bool]] = None
    # Stop starting new attempts once this much wall time has passed. Without
    # it a 3-attempt policy triples the worst case: a hung 900s create becomes
    # 45 minutes, and a hung teardown holds the process long after the job is
    # done.
    deadline_s: Optional[int] = None
    # A timeout carries no output to classify, so it is retried on the
    # assumption that the command is idempotent. Both provisioning creates
    # are: aws sends --client-token, and gcp treats "already exists" as
    # success. Set False for anything that is not.
    retry_timeouts: bool = True

    @property
    def attempts(self) -> int:
        return len(self.waits)

    def transient(self, err: str) -> bool:
        return bool(self.is_transient and self.is_transient(err))

    def non_retriable(self, err: str) -> bool:
        return bool(self.is_non_retriable and self.is_non_retriable(err))


NO_RETRIES = RetryPolicy()


def retry_budget_spent(policy: RetryPolicy, started_all: float) -> bool:
    """True once the policy's overall wall-clock budget is spent."""
    if policy.deadline_s is None:
        return False
    if time.monotonic() - started_all < policy.deadline_s:
        return False
    print(f"+ giving up: retry budget of {policy.deadline_s}s is spent", flush=True)
    return True


def run_with_retries(
    cmd: list,
    *,
    label: str,
    timeout: int,
    policy: RetryPolicy = NO_RETRIES,
    sleep=None,
    announce: Optional[str] = None,
):
    """Run a vendor CLI with `policy`, returning the last CompletedProcess.

    Does not raise on a non-zero exit — some callers treat that as data (a
    `brev ls` that lists nothing is not an error). It raises only when every
    attempt timed out, because there is no CompletedProcess to hand back.
    """
    if not policy.waits:
        raise CloudCliError(
            f"`{label}` was given a retry policy with no attempts; "
            f"RetryPolicy.waits must contain at least one entry."
        )
    total = policy.attempts
    if announce:
        print(f"+ {announce}", flush=True)
    last = None
    started_all = time.monotonic()
    # Resolved per call, not bound as a default: a default would capture
    # time.sleep at import and quietly ignore any later patch of it, which
    # is how a mocked test suite ends up really sleeping.
    nap = sleep or time.sleep
    for attempt, wait_s in enumerate(policy.waits, start=1):
        if wait_s:
            nap(wait_s)
        started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        started = time.monotonic()
        if total > 1:
            print(f"+ {label} attempt {attempt}/{total} started {started_at}", flush=True)
        try:
            last = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            elapsed_s = time.monotonic() - started
            print(
                f"+ {label} attempt {attempt}/{total} timed out "
                f"after {elapsed_s:.1f}s (timeout={timeout}s)",
                flush=True,
            )
            if (
                attempt < total
                and policy.retry_timeouts
                and not retry_budget_spent(policy, started_all)
            ):
                print(f"+ {label} attempt {attempt}/{total} will retry", flush=True)
                continue
            raise CloudCliError(
                f"`{label}` timed out after {timeout}s on all {total} attempts."
            ) from None
        elapsed_s = time.monotonic() - started
        # Coerce in case stdout/stderr are Mocks (test harness) or None.
        stdout = str(last.stdout or "")
        stderr = str(last.stderr or "")
        if total > 1:
            print(
                f"+ {label} attempt {attempt}/{total} finished "
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
        err = stderr + stdout
        if policy.non_retriable(err):
            print(
                f"+ {label} attempt {attempt}/{total} hit a non-retriable "
                f"error; bailing out rather than burning the retry budget",
                flush=True,
            )
            return last
        if (
            policy.transient(err)
            and attempt < total
            and not retry_budget_spent(policy, started_all)
        ):
            print(
                f"+ {label} attempt {attempt}/{total} hit transient error; retrying",
                flush=True,
            )
            continue
        return last
    assert last is not None  # for type checkers; the loop returns or raises
    return last


# Cloud control planes fail in ways that clear on their own — a 503 from the
# API, a throttle, a lock on a resource still settling — and in ways that
# never will. Retrying the first costs seconds; retrying the second costs
# minutes of a user's time before showing them the error that mattered.

# Written against the strings the CLIs actually print, not from memory: an
# earlier cut had "internalerror" (no space) and a contiguous "quota
# exceeded", neither of which gcloud ever emits, so the GCP half retried
# nothing. Substrings are kept specific enough not to fire on an instance
# name — a bare "503" matched runplz-app-fn-a503bcd1.

GCP_TRANSIENT = (
    "internal error",  # "Internal error. Please try again or contact Google Support."
    "internal_error",
    "backend error",
    "backenderror",
    "service unavailable",
    "serviceunavailable",
    "http 503",
    "code: 503",
    "try again",  # gcloud's own advice when it means it
    "rate limit",
    "ratelimitexceeded",
    "is not ready",  # "The resource 'projects/...' is not ready"
    "operation is already in progress",
    "connection reset",
    "connection refused",
    "i/o timeout",
    "read timed out",
    "deadline exceeded",
    "unavailable_error",
)
GCP_NON_RETRIABLE = (
    "zone_resource_pool_exhausted",
    "resource_exhausted",
    "permission",
    "billing account",
    "invalid value",
    "invalid_argument",
    "was not found",
    "required 'compute",
    "constraint",
)

AWS_TRANSIENT = (
    "throttl",  # Throttling, ThrottlingException, RequestThrottled
    "rate exceeded",
    "requestlimitexceeded",
    "serviceunavailable",
    "service unavailable",
    "internal error",
    "internalerror",
    "http 503",
    "please try again",
    "connection reset",
    "could not connect to the endpoint",
    "timed out",
    # Eventual consistency right after run-instances — the single most
    # likely describe-instances failure, and the reason it is retried.
    "invalidinstanceid.notfound",
)
AWS_NON_RETRIABLE = (
    "insufficientinstancecapacity",
    "unauthorizedoperation",
    "authfailure",
    "invalidkeypair.notfound",
    "invalidami",
    "invalidsubnetid",
    "invalidgroup.notfound",
    "invalidparametervalue",
    "vcpulimitexceeded",
    "instancelimitexceeded",
    "optinrequired",
)

# gcloud reports a create that already landed this way. It means the box
# exists and must be torn down, so it is neither transient nor a plain
# failure — see gcp.provision.
ALREADY_EXISTS = "already exists"


def _matcher(patterns: tuple):
    def matches(err: str) -> bool:
        low = (err or "").lower()
        return any(pat in low for pat in patterns)

    return matches


def _gcp_non_retriable(err: str) -> bool:
    low = (err or "").lower()
    if _matcher(GCP_NON_RETRIABLE)(low):
        return True
    # "Quota 'NVIDIA_T4_GPUS' exceeded. Limit: 1.0" — the two words are never
    # adjacent, so a substring match misses the most common final error.
    return "quota" in low and "exceeded" in low


GCP_RETRY_POLICY = RetryPolicy(
    waits=(0, 3, 8),
    is_transient=_matcher(GCP_TRANSIENT),
    is_non_retriable=_gcp_non_retriable,
    deadline_s=1800,
)
AWS_RETRY_POLICY = RetryPolicy(
    waits=(0, 3, 8),
    is_transient=_matcher(AWS_TRANSIENT),
    is_non_retriable=_matcher(AWS_NON_RETRIABLE),
    deadline_s=1800,
)
# Teardown runs in a finally, and every second of backoff is a second in
# which a Ctrl-C abandons the delete and leaves the box billing. One quick
# retry for a blip, then give up and shout.
GCP_TEARDOWN_POLICY = dataclasses.replace(GCP_RETRY_POLICY, waits=(0, 2), deadline_s=90)
AWS_TEARDOWN_POLICY = dataclasses.replace(AWS_RETRY_POLICY, waits=(0, 2), deadline_s=90)


def run_cli(
    cmd: list,
    *,
    label: str,
    timeout: int = 300,
    dry_run: bool = False,
    parse_json: bool = False,
    check: bool = True,
    policy: RetryPolicy = NO_RETRIES,
):
    """Invoke a vendor CLI, printing the command so runs are auditable.

    A multi-attempt `policy` retries transient control-plane failures and
    gives up immediately on the ones that will never clear — see
    :class:`RetryPolicy`.

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

    try:
        # Always through the one loop, and always announcing the full argv:
        # the machine type, accelerator, disk size and network that were
        # actually requested belong in the log of a *successful* run, not
        # only in the error message of a failed one.
        r = run_with_retries(
            cmd,
            label=label,
            timeout=timeout,
            policy=policy,
            announce=printable,
        )
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
