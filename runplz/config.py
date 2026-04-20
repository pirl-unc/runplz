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


@dataclass(frozen=True)
class BrevConfig:
    # When True (default) and `--instance <name>` points at a box that does
    # not exist, runplz provisions it via `brev create`. When False, a
    # missing instance is a hard error.
    auto_create_instances: bool = True
    # Optional explicit instance type (e.g. "n1-standard-4:nvidia-tesla-t4:1").
    # When set, takes precedence over the constraint-based picker. When None
    # (default), `brev search` is driven by the function's resource
    # constraints and the cheapest matching type is picked.
    instance_type: Optional[str] = None
    # Provisioning mode:
    # - "vm" (default): `brev create` provisions a full VM with Brev's
    #   sidecar stack (grafana, influxdb, jupyter, cloudflared). User code
    #   runs inside `docker run --gpus all ...` on top of that.
    # - "container": `brev create --mode container --container-image <base>`
    #   where `<base>` comes from Image.from_registry(...). The box *is*
    #   the user's image; our backend skips docker entirely and runs the
    #   declared apt/pip layer ops inline over ssh. Lighter host footprint
    #   and sidesteps the `docker run --gpus all` path that historically
    #   wedged SSH on Brev GPU boxes (see docs/brev-ssh-bug-report.md).
    mode: str = "vm"
    # Legacy escape hatch for VM mode — skip docker, install the training
    # environment natively (apt + python3-venv + pip) and run the user's job
    # directly over ssh. Kept for boxes where mode="container" isn't an
    # option (different provider / legacy flow).
    use_docker: bool = True

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


@dataclass(frozen=True)
class ModalConfig:
    """Modal has nothing to configure today.

    Modal reads auth from `~/.modal.toml` and schedules resources from
    `@app.function(gpu=..., cpu=..., memory=...)`. This class exists as
    a slot in `App(modal=...)` so we don't break the signature when we
    add real fields. Until then, `ModalConfig()` is a no-op.
    """

    pass
