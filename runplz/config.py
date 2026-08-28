"""Per-backend config objects.

Validation runs at construction time so invalid configs raise before the
job ever starts. Defaults rely on each tool's native auth (`brev login`,
`~/.modal.toml`).

There's no shared base config today — see README "Why not one unified
config?". Brev has several meaningful provisioning knobs; Modal has
nothing we expose. If that ever changes (e.g. both grow a shared
concept like per-App secrets), factor into a `BaseConfig` then.
"""

__all__ = [
    "BrevConfig",
    "SshConfig",
    "ModalConfig",
    "GcpConfig",
    "AwsConfig",
]


import os
from dataclasses import dataclass
from typing import Optional

_VALID_BREV_MODES = ("vm", "container")
_VALID_BREV_ON_FINISH = ("stop", "delete", "leave")

# SSH-backend on_finish: user owns the VM lifecycle, so runplz never
# stops/deletes on our end. "leave" is the only valid option.
_VALID_SSH_ON_FINISH = ("leave",)


@dataclass(frozen=True)
class BrevConfig:
    # When True and `--instance <name>` points at a box that does not exist,
    # runplz provisions it via `brev create`. When False (default), a missing
    # instance is a hard error so a typoed instance name can't silently create
    # a brand-new billed box — opt in explicitly when you actually want
    # programmatic creation.
    auto_create_instances: bool = False
    # Optional explicit instance type (e.g. "n1-standard-4:nvidia-tesla-t4:1").
    # When set, takes precedence over the constraint-based picker. When None
    # (default), `brev search` is driven by the function's resource
    # constraints and the cheapest matching type is picked.
    instance_type: Optional[str] = None
    # Provisioning mode:
    # - "container" (default): `brev create --mode container --container-image
    #   <base>` where `<base>` comes from Image.from_registry(...). The box
    #   *is* the user's image; our backend skips docker entirely and runs the
    #   declared apt/pip layer ops inline over ssh. Lighter host footprint
    #   and sidesteps the `docker run --gpus all` path that historically
    #   wedged SSH on Brev GPU boxes (see docs/brev-ssh-bug-report.md).
    #   Requires Image.from_registry(...) — Dockerfile images don't
    #   translate to inline installs.
    # - "vm": `brev create` provisions a full VM with Brev's sidecar stack
    #   (grafana, influxdb, jupyter, cloudflared). User code runs inside
    #   `docker run --gpus all ...` on top of that. Use when you need a
    #   user Dockerfile (`Image.from_dockerfile`) or the legacy native path
    #   (`use_docker=False`).
    mode: str = "container"
    # Legacy escape hatch for VM mode — skip docker, install the training
    # environment natively (apt + python3-venv + pip) and run the user's job
    # directly over ssh. Kept for boxes where mode="container" isn't an
    # option (different provider / legacy flow).
    use_docker: bool = True
    # What to do with the Brev box when the App exits (success OR failure).
    # Matches Modal's ephemeral-compute model by default.
    # - "stop" (default): `brev stop <instance>`. Disk + image cached so the
    #   next run starts fast. Incurs a small disk charge while stopped.
    # - "delete": `brev delete <instance>`. Zero ongoing cost; full rebuild
    #   on next run.
    # - "leave": never touch the box. Opt-in for interactive workflows where
    #   you're using the same box for dev + jobs. Current pre-3.2 behavior.
    on_finish: str = "stop"
    # Wall-clock cap on the remote run. When None (default), the job runs
    # until it finishes or the user Ctrl-Cs. When set, runplz kills the
    # remote container/process after this many seconds and raises
    # RuntimeError, so a wedged/infinite-loop job can't keep billing
    # forever. Distinct from `Function(timeout=...)` which applies only
    # to Modal — this is a Brev-specific kill-switch enforced by runplz.
    max_runtime_seconds: Optional[int] = None
    # How long to wait for the freshly-provisioned Brev box to become
    # SSH-reachable before giving up. Default 1800s (30 min) covers
    # 8×A100/H100 cold boots on Denvr / OCI (observed 15-18 min). Bump
    # to 2400+ for exotic shapes that take longer (issue #34).
    ssh_ready_wait_seconds: int = 1800
    # How many fallback instance types to pass to `brev create`. When
    # > 1, runplz feeds the selector's top-N ranked candidates via
    # repeated `--type` flags and Brev's own retry loop tries them in
    # order — if Nebius fails on type A, Brev transparently falls back
    # to type B on e.g. OCI. Only applies to auto-picked types; when
    # `instance_type` is pinned explicitly, that one type is still the
    # only one passed. Set to 1 for the pre-3.9 behavior (single type).
    instance_type_fallback_count: int = 3
    # Provider prefixes to drop from `brev search` results before the
    # selector ranks candidates. Match is case-insensitive against the
    # type string's leading dotted/underscored segment(s) — e.g. the
    # default `("oci",)` filters out types like `oci.a100x8.sxm.brev-dgxc`.
    # Default blocks OCI because Brev support has confirmed (issue #62)
    # that the OCI launchpad path is broken at their backend layer
    # (cloudCredId/workspaceGroupId rejection on every create call), so
    # silently feeding it to runplz costs the user a confusing failure
    # that no client-side fix can resolve. Set to () to disable, or
    # extend if other providers misbehave for your org.
    exclude_providers: tuple = ("oci",)

    def __post_init__(self):
        if self.mode not in _VALID_BREV_MODES:
            raise ValueError(
                f"BrevConfig.mode must be one of {_VALID_BREV_MODES}; got {self.mode!r}."
            )
        if self.mode == "container" and not self.use_docker:
            raise ValueError(
                "BrevConfig(mode='container', use_docker=False) is contradictory. "
                "use_docker only applies to mode='vm'. In mode='container' the "
                "box itself is the user's image — there's no inner docker to skip."
            )
        if self.instance_type is not None and not self.instance_type.strip():
            raise ValueError("BrevConfig.instance_type must be a non-empty string (or None).")
        _validate_remote_common("BrevConfig", self, valid_on_finish=_VALID_BREV_ON_FINISH)
        if (
            not isinstance(self.instance_type_fallback_count, int)
            or self.instance_type_fallback_count < 1
        ):
            raise ValueError(
                f"BrevConfig.instance_type_fallback_count must be a positive int "
                f"(1 = no fallback); got {self.instance_type_fallback_count!r}."
            )
        if not isinstance(self.exclude_providers, tuple):
            raise ValueError(
                f"BrevConfig.exclude_providers must be a tuple of strings; "
                f"got {type(self.exclude_providers).__name__}."
            )
        for prefix in self.exclude_providers:
            if not isinstance(prefix, str) or not prefix.strip():
                raise ValueError(
                    f"BrevConfig.exclude_providers entries must be non-empty strings; "
                    f"got {prefix!r}."
                )


@dataclass(frozen=True)
class SshConfig:
    """Config for the `ssh` backend — dispatch to a user-owned remote box.

    The actual ssh target (hostname / alias) isn't stored here; it comes
    through `App.bind("ssh", host=...)` or `runplz ssh --host ...`, same
    shape as `BrevConfig` + `--instance`. This dataclass holds everything
    that's shape-of-the-remote, not which remote.

    Lifecycle is minimal: runplz never provisions and never tears down.
    We assume the user manages their own box. `on_finish` is pinned to
    `"leave"` for that reason.
    """

    # Optional ssh user. When None, the local ssh config / URL is used.
    user: Optional[str] = None
    # Optional ssh port. When None, ssh's default (22 or whatever the user's
    # config sets).
    port: Optional[int] = None
    # Private key to authenticate with. None → ssh's own resolution (agent,
    # ~/.ssh/id_*, or an IdentityFile entry). Set it when the box uses a
    # key that ssh would not otherwise offer.
    ssh_key_path: Optional[str] = None
    # True (default): build/pull a docker image on the remote and `docker
    # run` the bootstrap. Mirrors BrevConfig(mode="vm", use_docker=True).
    # False: install a python venv on the remote and run the bootstrap
    # natively. Mirrors BrevConfig(mode="vm", use_docker=False).
    use_docker: bool = True
    # The user owns the box; runplz never stops or deletes it.
    on_finish: str = "leave"
    # Wall-clock kill-switch on the remote run. Same semantics as
    # BrevConfig.max_runtime_seconds.
    max_runtime_seconds: Optional[int] = None
    # How long to wait for the SSH box to become reachable before giving
    # up. Default 1800s (30 min). Mostly matters when the user is
    # booting the box themselves just before the runplz invocation;
    # for always-on dev boxes the ssh probe succeeds on the first try.
    ssh_ready_wait_seconds: int = 1800

    def __post_init__(self):
        if self.port is not None and not (0 < self.port < 65536):
            raise ValueError(
                f"SshConfig.port must be a valid TCP port (1-65535) or None; got {self.port!r}."
            )
        _validate_ssh_key_path("SshConfig", self.ssh_key_path)
        if self.user is not None and not self.user.strip():
            raise ValueError("SshConfig.user must be a non-empty string (or None).")
        _validate_remote_common(
            "SshConfig",
            self,
            valid_on_finish=_VALID_SSH_ON_FINISH,
            on_finish_note=" (runplz never stops/deletes a user-owned SSH box.)",
        )


@dataclass(frozen=True)
class ModalConfig:
    """Modal has nothing to configure today.

    Modal reads auth from `~/.modal.toml` and schedules resources from
    `@app.function(gpu=..., cpu=..., memory=...)`. This class exists as
    a slot in `App(modal_config=...)` so we don't break the signature
    when we add real fields. Until then, `ModalConfig()` is a no-op.
    """

    pass


_VALID_CLOUD_ON_FINISH = ("delete", "stop", "leave")


@dataclass(frozen=True)
class GcpConfig:
    """Config for the `gcp` backend — provision a GCE VM, run, tear down.

    runplz shells out to `gcloud` rather than the Python SDK: the core
    stays dependency-free, and the test suite's billed-CLI guard already
    covers `gcloud` (see tests/conftest.py), so a test that forgets to mock
    cannot quietly launch a paid instance.

    Auth is whatever `gcloud` already has — ADC or `gcloud auth login`.
    runplz never handles credentials and never writes them anywhere.

    Networking is pinned, never created: pass an existing `network` /
    `subnet` if the default VPC isn't what you want. runplz will not create
    firewall rules for you.
    """

    # Required. The GCP project and zone to create the instance in.
    project: str = ""
    zone: str = ""
    # Machine type (e.g. "a2-highgpu-1g"). None → derived from the
    # function's gpu / min_gpus / min_cpu / min_memory.
    machine_type: Optional[str] = None
    # Accelerator type (e.g. "nvidia-tesla-a100"). None → derived from
    # `Function.gpu`. Ignored when the machine type has GPUs built in
    # (a2/a3 shapes ship with their accelerators attached).
    accelerator: Optional[str] = None
    # Boot image. Defaults to Google's Deep Learning VM, which ships with
    # NVIDIA drivers and docker already installed — without those, every
    # run would pay a multi-minute driver install before it could start.
    image_family: str = "common-cu123-py310"
    image_project: str = "deeplearning-platform-release"
    # Boot disk. None → Function.min_disk, else GCP's default.
    boot_disk_gb: Optional[int] = None
    boot_disk_type: Optional[str] = None
    # Existing network / subnet to attach to. None → project default.
    network: Optional[str] = None
    subnet: Optional[str] = None
    # Service account + scopes for the instance. None → project default.
    service_account: Optional[str] = None
    scopes: Optional[str] = None
    # Spot (preemptible) capacity. Cheaper, can be reclaimed mid-run.
    # No retry-on-reclaim loop yet — see issue #25's deferred list.
    spot: bool = False
    # What to do with the instance when the run ends. "delete" is the
    # default because an ephemeral box nothing will reuse is a pure
    # billing leak if left running.
    on_finish: str = "delete"
    # Build/pull a docker image on the instance (True) vs installing a
    # python venv natively (False). Same meaning as SshConfig.use_docker.
    use_docker: bool = True
    max_runtime_seconds: Optional[int] = None
    ssh_ready_wait_seconds: int = 1800
    # Print every gcloud command instead of running it. Nothing is created,
    # nothing is billed — for checking what a config would actually do.
    dry_run: bool = False

    def __post_init__(self):
        _validate_remote_common("GcpConfig", self, valid_on_finish=_VALID_CLOUD_ON_FINISH)
        if not (self.project or "").strip():
            raise ValueError(
                "GcpConfig.project is required, e.g. GcpConfig(project='my-proj', "
                "zone='us-central1-a'). runplz does not guess it from your "
                "gcloud config — an instance billed to the wrong project is "
                "worse than an error."
            )
        if not (self.zone or "").strip():
            raise ValueError(
                "GcpConfig.zone is required, e.g. GcpConfig(project='my-proj', "
                "zone='us-central1-a'). GPU availability is per-zone, so there "
                "is no safe default."
            )


@dataclass(frozen=True)
class AwsConfig:
    """Config for the `aws` backend — provision an EC2 instance, run, tear down.

    Shells out to the `aws` CLI for the same reasons as :class:`GcpConfig`.
    Auth is whatever the CLI already has: profile, env vars, or an
    instance/SSO role.

    Unlike GCP, EC2 has no `config-ssh` equivalent that publishes an ssh
    alias for you, so `key_name` is required and the target is built from
    the instance's public IP.

    Point `ssh_key_path` at the private half of that key pair; runplz passes
    it as `-i` (with `IdentitiesOnly=yes`, so a loaded agent cannot offer its
    other keys first and trip the server's MaxAuthTries). Leave it None if
    the key is already agent-loaded or named in your ssh config.

    Networking is pinned, never created: pass an existing `subnet_id` /
    `security_group_id`. The security group must allow inbound TCP 22 from
    wherever you are running runplz, or the run will hang waiting for ssh.
    """

    # Required. Region to launch in.
    region: str = ""
    # Instance type (e.g. "g5.xlarge"). None → derived from the function's
    # gpu / min_gpus / min_cpu / min_memory.
    instance_type: Optional[str] = None
    # AMI id. None → resolve the current Deep Learning AMI from SSM, which
    # ships NVIDIA drivers + docker. AMI ids are region-specific and roll
    # monthly, so hardcoding one would rot.
    ami: Optional[str] = None
    # Required: the EC2 key pair whose private half you hold locally.
    # Without it the instance launches with no way to ssh in.
    key_name: Optional[str] = None
    # Login user for the AMI. Deep Learning AMIs are Ubuntu-based.
    ssh_user: str = "ubuntu"
    # Private half of `key_name`, e.g. "~/.ssh/my-key.pem". None → ssh's own
    # resolution (agent, default name, or an IdentityFile entry). EC2 keys
    # are conventionally saved under a name ssh will not offer on its own,
    # and no Host block can be written in advance because the instance IP
    # does not exist yet — so setting this is usually what you want.
    ssh_key_path: Optional[str] = None
    # Existing VPC bits to attach to. None → account default VPC.
    subnet_id: Optional[str] = None
    security_group_id: Optional[str] = None
    iam_instance_profile: Optional[str] = None
    # Root volume size. None → Function.min_disk, else the AMI's default.
    volume_gb: Optional[int] = None
    # Spot capacity. Cheaper, can be reclaimed mid-run. No retry loop yet.
    spot: bool = False
    on_finish: str = "delete"
    use_docker: bool = True
    max_runtime_seconds: Optional[int] = None
    ssh_ready_wait_seconds: int = 1800
    dry_run: bool = False

    def __post_init__(self):
        _validate_remote_common("AwsConfig", self, valid_on_finish=_VALID_CLOUD_ON_FINISH)
        _validate_ssh_key_path("AwsConfig", self.ssh_key_path)
        if not (self.region or "").strip():
            raise ValueError(
                "AwsConfig.region is required, e.g. AwsConfig(region='us-east-1', "
                "key_name='my-key'). GPU availability and pricing are "
                "per-region, so there is no safe default."
            )


def _validate_ssh_key_path(cls_name: str, path) -> None:
    """Reject a key path that ssh will silently ignore.

    `IdentitiesOnly=yes` accompanies `-i`, which is what stops a loaded agent
    from offering its other keys first — but it also means a typo'd path
    doesn't fall back to the agent, it just fails with `Permission denied`.
    On a provisioning backend that surfaces as a 30-minute ssh wait on a
    billed box, so catch it here instead.
    """
    if path is None:
        return
    if not str(path).strip():
        raise ValueError(f"{cls_name}.ssh_key_path must be a non-empty path (or None).")
    if not os.path.exists(os.path.expanduser(str(path))):
        raise ValueError(
            f"{cls_name}.ssh_key_path={path!r} does not exist. runplz passes it "
            f"to ssh as `-i` with IdentitiesOnly=yes, so a wrong path fails "
            f"authentication outright rather than falling back to your agent."
        )


def _validate_remote_common(
    cls_name: str, cfg, *, valid_on_finish, on_finish_note: str = ""
) -> None:
    """Field validation every remote backend config shares.

    All four remote configs carry `on_finish`, `max_runtime_seconds` and
    `ssh_ready_wait_seconds` with identical rules; only the set of valid
    `on_finish` values differs (SSH allows "leave" alone, since runplz never
    owns a user's box). One implementation means the message a user sees has
    the same shape whichever backend they mistyped.
    """
    if cfg.on_finish not in valid_on_finish:
        raise ValueError(
            f"{cls_name}.on_finish must be one of {valid_on_finish}; "
            f"got {cfg.on_finish!r}.{on_finish_note}"
        )
    if cfg.max_runtime_seconds is not None and cfg.max_runtime_seconds <= 0:
        raise ValueError(
            f"{cls_name}.max_runtime_seconds must be a positive int (or None); "
            f"got {cfg.max_runtime_seconds!r}."
        )
    if not isinstance(cfg.ssh_ready_wait_seconds, int) or cfg.ssh_ready_wait_seconds <= 0:
        raise ValueError(
            f"{cls_name}.ssh_ready_wait_seconds must be a positive int; "
            f"got {cfg.ssh_ready_wait_seconds!r}."
        )
