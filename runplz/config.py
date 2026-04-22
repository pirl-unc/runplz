"""Per-backend config objects.

Validation runs at construction time so invalid configs raise before the
job ever starts. Defaults rely on each tool's native auth (`brev login`,
`~/.modal.toml`).

There's no shared base config today — see README "Why not one unified
config?". Brev has several meaningful provisioning knobs; Modal has
nothing we expose. If that ever changes (e.g. both grow a shared
concept like per-App secrets), factor into a `BaseConfig` then.
"""

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
        if self.on_finish not in _VALID_BREV_ON_FINISH:
            raise ValueError(
                f"BrevConfig.on_finish must be one of {_VALID_BREV_ON_FINISH}; "
                f"got {self.on_finish!r}."
            )
        if self.max_runtime_seconds is not None and self.max_runtime_seconds <= 0:
            raise ValueError(
                f"BrevConfig.max_runtime_seconds must be a positive int (or None); "
                f"got {self.max_runtime_seconds!r}."
            )
        if not isinstance(self.ssh_ready_wait_seconds, int) or self.ssh_ready_wait_seconds <= 0:
            raise ValueError(
                f"BrevConfig.ssh_ready_wait_seconds must be a positive int; "
                f"got {self.ssh_ready_wait_seconds!r}."
            )
        if (
            not isinstance(self.instance_type_fallback_count, int)
            or self.instance_type_fallback_count < 1
        ):
            raise ValueError(
                f"BrevConfig.instance_type_fallback_count must be a positive int "
                f"(1 = no fallback); got {self.instance_type_fallback_count!r}."
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
        if self.user is not None and not self.user.strip():
            raise ValueError("SshConfig.user must be a non-empty string (or None).")
        if self.on_finish not in _VALID_SSH_ON_FINISH:
            raise ValueError(
                f"SshConfig.on_finish must be one of {_VALID_SSH_ON_FINISH}; "
                f"got {self.on_finish!r}. (runplz never stops/deletes a "
                f"user-owned SSH box.)"
            )
        if self.max_runtime_seconds is not None and self.max_runtime_seconds <= 0:
            raise ValueError(
                f"SshConfig.max_runtime_seconds must be a positive int (or None); "
                f"got {self.max_runtime_seconds!r}."
            )
        if not isinstance(self.ssh_ready_wait_seconds, int) or self.ssh_ready_wait_seconds <= 0:
            raise ValueError(
                f"SshConfig.ssh_ready_wait_seconds must be a positive int; "
                f"got {self.ssh_ready_wait_seconds!r}."
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
