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
from typing import Callable, Optional

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
    "MachineOffering",
    "GpuShapes",
    "AWS_CPU_SHAPES",
    "GCP_CPU_SHAPES",
    "GCP_GPUS",
    "AWS_GPUS",
    "resolve_gpu_label",
    "gpu_count",
    "select_machine",
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
    # --- retry classification tables -------------------------------------
    "GCP_TRANSIENT",
    "GCP_NON_RETRIABLE",
    "AWS_TRANSIENT",
    "AWS_NON_RETRIABLE",
    # gcp.py reads this to tell a benign "already exists" (a retried create
    # whose first attempt landed) from a real failure.
    "ALREADY_EXISTS",
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


def retry_budget_spent(policy: RetryPolicy, started_all: float, now: float = None) -> bool:
    """True once the policy's overall wall-clock budget is spent.

    `now` lets a caller supply its own clock, so a loop that measures elapsed
    time with one clock does not compare it against another's.
    """
    if policy.deadline_s is None:
        return False
    if now is None:
        now = time.monotonic()
    if now - started_all < policy.deadline_s:
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


@dataclass(frozen=True)
class MachineOffering:
    """One provider machine type and the resources it actually supplies.

    Selection and validation consume the same records. That prevents the old
    split-brain design where production built a plausible name arithmetically
    while tests separately asserted a hand-written list of names.
    """

    name: str
    vcpus: int
    memory_gb: float
    gpus: int = 0

    def satisfies(self, *, min_cpu: float, min_memory: float, gpus: int) -> bool:
        return self.gpus == gpus and self.vcpus >= min_cpu and self.memory_gb >= min_memory


@dataclass(frozen=True)
class GpuShapes:
    """All machine offerings for one GPU model on one provider."""

    # GCP needs an accelerator name for shapes with separately attached GPUs.
    # AWS and GCP's accelerator-optimized machine families bundle the GPU.
    accelerator: Optional[str]
    family: str
    vram_gb: int
    offerings: tuple[MachineOffering, ...]
    attached: bool = False

    @property
    def gpu_counts(self) -> tuple[int, ...]:
        return tuple(sorted({offering.gpus for offering in self.offerings}))


def _offering(name: str, vcpus: int, memory_gb: float, gpus: int = 0) -> MachineOffering:
    return MachineOffering(name, vcpus, memory_gb, gpus)


# Exact capabilities from the provider machine-type specifications. Ordering is
# deliberately smallest-first: select_machine returns the first offering that
# satisfies every declared minimum. A request beyond the last entry raises; it
# never clamps to an undersized but syntactically valid machine.
AWS_CPU_SHAPES = tuple(
    _offering(name, vcpus, vcpus * 4)
    for name, vcpus in (
        ("m6i.large", 2),
        ("m6i.xlarge", 4),
        ("m6i.2xlarge", 8),
        ("m6i.4xlarge", 16),
        ("m6i.8xlarge", 32),
        ("m6i.12xlarge", 48),
        ("m6i.16xlarge", 64),
        ("m6i.24xlarge", 96),
        ("m6i.32xlarge", 128),
    )
)

GCP_CPU_SHAPES = tuple(
    _offering(f"n2-standard-{vcpus}", vcpus, vcpus * 4)
    for vcpus in (2, 4, 8, 16, 32, 48, 64, 80, 96, 128)
)


def _n1_offerings(gpu_counts_and_sizes: dict) -> tuple[MachineOffering, ...]:
    """Build the finite standard N1 shapes allowed for attached GPUs.

    GCE publishes different maximum vCPU ranges for each GPU/count pair.
    N1 standard shapes carry 3.75 GB per vCPU (not 4 GB); keeping that fact
    here prevents memory minima from silently receiving a smaller machine.
    """
    return tuple(
        _offering(f"n1-standard-{vcpus}", vcpus, vcpus * 3.75, gpus)
        for gpus, sizes in gpu_counts_and_sizes.items()
        for vcpus in sizes
    )


GCP_GPUS = {
    "T4": GpuShapes(
        "nvidia-tesla-t4",
        "n1-standard",
        16,
        _n1_offerings({1: (8, 16, 32), 2: (8, 16, 32), 4: (16, 32, 64, 96)}),
        attached=True,
    ),
    "V100": GpuShapes(
        "nvidia-tesla-v100",
        "n1-standard",
        16,
        _n1_offerings({1: (8,), 2: (8, 16), 4: (16, 32), 8: (32, 64, 96)}),
        attached=True,
    ),
    "L4": GpuShapes(
        "nvidia-l4",
        "g2-standard",
        24,
        (
            _offering("g2-standard-4", 4, 16, 1),
            _offering("g2-standard-8", 8, 32, 1),
            _offering("g2-standard-12", 12, 48, 1),
            _offering("g2-standard-16", 16, 64, 1),
            _offering("g2-standard-32", 32, 128, 1),
            _offering("g2-standard-24", 24, 96, 2),
            _offering("g2-standard-48", 48, 192, 4),
            _offering("g2-standard-96", 96, 384, 8),
        ),
    ),
    "A100-40GB": GpuShapes(
        "nvidia-tesla-a100",
        "a2-highgpu",
        40,
        tuple(
            _offering(f"a2-highgpu-{gpus}g", vcpus, memory, gpus)
            for gpus, vcpus, memory in (
                (1, 12, 85),
                (2, 24, 170),
                (4, 48, 340),
                (8, 96, 680),
            )
        ),
    ),
    "A100-80GB": GpuShapes(
        "nvidia-a100-80gb",
        "a2-ultragpu",
        80,
        tuple(
            _offering(f"a2-ultragpu-{gpus}g", vcpus, memory, gpus)
            for gpus, vcpus, memory in (
                (1, 12, 170),
                (2, 24, 340),
                (4, 48, 680),
                (8, 96, 1360),
            )
        ),
    ),
    "H100": GpuShapes(
        "nvidia-h100-80gb",
        "a3-highgpu",
        80,
        (_offering("a3-highgpu-8g", 208, 1872, 8),),
    ),
}


def _aws_scale(family: str, one_gpu_memory: int) -> tuple[MachineOffering, ...]:
    """Common G4dn/G5/G6/G6e layout for 1, 4 and 8 GPU offerings."""
    one_gpu = tuple(
        _offering(f"{family}.{size}", vcpus, memory, 1)
        for size, vcpus, memory in (
            ("xlarge", 4, one_gpu_memory),
            ("2xlarge", 8, one_gpu_memory * 2),
            ("4xlarge", 16, one_gpu_memory * 4),
            ("8xlarge", 32, one_gpu_memory * 8),
            ("16xlarge", 64, one_gpu_memory * 16),
        )
    )
    four_gpu = (
        _offering(f"{family}.12xlarge", 48, one_gpu_memory * 12, 4),
        _offering(f"{family}.24xlarge", 96, one_gpu_memory * 24, 4),
    )
    eight_gpu = (_offering(f"{family}.48xlarge", 192, one_gpu_memory * 48, 8),)
    if family == "g4dn":
        four_gpu = (_offering("g4dn.12xlarge", 48, 192, 4),)
        eight_gpu = (_offering("g4dn.metal", 96, 384, 8),)
    return one_gpu + four_gpu + eight_gpu


AWS_GPUS = {
    "T4": GpuShapes(None, "g4dn", 16, _aws_scale("g4dn", 16)),
    "V100": GpuShapes(
        None,
        "p3",
        16,
        (
            _offering("p3.2xlarge", 8, 61, 1),
            _offering("p3.8xlarge", 32, 244, 4),
            _offering("p3.16xlarge", 64, 488, 8),
        ),
    ),
    "A10G": GpuShapes(None, "g5", 24, _aws_scale("g5", 16)),
    "L4": GpuShapes(None, "g6", 24, _aws_scale("g6", 16)),
    "L40S": GpuShapes(None, "g6e", 48, _aws_scale("g6e", 32)),
    "A100-40GB": GpuShapes(None, "p4d", 40, (_offering("p4d.24xlarge", 96, 1152, 8),)),
    "A100-80GB": GpuShapes(None, "p4de", 80, (_offering("p4de.24xlarge", 96, 1152, 8),)),
    "H100": GpuShapes(
        None,
        "p5",
        80,
        (
            _offering("p5.4xlarge", 16, 256, 1),
            _offering("p5.48xlarge", 192, 2048, 8),
        ),
    ),
}


def _resource_minimums(function) -> tuple[float, float]:
    min_cpu = float(getattr(function, "min_cpu", None) or 0) or 2
    min_memory = float(getattr(function, "min_memory", None) or 0)
    return min_cpu, min_memory


def select_machine(
    function,
    offerings: tuple[MachineOffering, ...],
    *,
    cloud: str,
    gpus: int = 0,
    gpu_label: Optional[str] = None,
) -> MachineOffering:
    """Choose the smallest known offering satisfying every resource minimum.

    The function is intentionally provider-neutral. Provider drivers supply
    only their ordered catalogue and vocabulary; count validation, CPU/RAM
    validation and the fail-instead-of-clamp contract live here once.
    """
    min_cpu, min_memory = _resource_minimums(function)
    matching_count = tuple(offering for offering in offerings if offering.gpus == gpus)
    if not matching_count:
        counts = sorted({offering.gpus for offering in offerings})
        resource = gpu_label or "GPU"
        raise CloudCliError(
            f"{cloud} sells {resource} in {counts} GPU counts, not {gpus}. "
            f"Pass an explicit machine type to use a shape runplz doesn't know."
        )
    for offering in matching_count:
        if offering.satisfies(min_cpu=min_cpu, min_memory=min_memory, gpus=gpus):
            return offering
    largest = max(matching_count, key=lambda item: (item.vcpus, item.memory_gb))
    resource = f"{gpu_label} with {gpus} GPU(s)" if gpu_label else "CPU-only"
    raise CloudCliError(
        f"{cloud} has no known {resource} machine satisfying "
        f"min_cpu={min_cpu:g}, min_memory={min_memory:g} GB. "
        f"Largest known is {largest.name} ({largest.vcpus} vCPU, "
        f"{largest.memory_gb:g} GB). Pass an explicit machine type to use "
        f"a shape runplz doesn't know."
    )


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
