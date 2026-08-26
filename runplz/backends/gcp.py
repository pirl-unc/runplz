"""GCP backend: provision a GCE VM, run one function on it, tear it down.

Shells out to `gcloud`. Auth is whatever the CLI already has — ADC or
`gcloud auth login`; runplz never touches credentials.

SSH access rides on `gcloud compute config-ssh`, which writes a
`NAME.ZONE.PROJECT` alias per instance into `~/.ssh/config`. That is the
direct analogue of `brev refresh`, so it plugs into the existing
`wait_until_ssh_reachable` refresh callback and no new ssh plumbing is
needed — plain `ssh` and `rsync` work against the alias.

Networking is pinned, never created. Pass an existing `network`/`subnet`
if the default VPC isn't right; runplz will not open firewall rules for
you, and a VPC with no inbound 22 will simply hang the ssh wait.
"""

from typing import Optional

from runplz.backends.provisioning import (
    GCP_GPUS,
    apply_teardown,
    gpu_count,
    make_instance_name,
    resolve_gpu_label,
    run_cli,
)
from runplz.backends.ssh_common import run_on_provisioned_vm

__all__ = ["run"]


def run(app, function, args, kwargs, *, outputs_dir: str = "out"):
    cfg = app.gcp_config
    name = make_instance_name(app.name, function.name)
    machine_type, accelerator = resolve_shape(cfg, function)

    print(
        f"+ gcp: instance={name} zone={cfg.zone} machine-type={machine_type}"
        + (f" accelerator={accelerator}" if accelerator else "")
        + f" on_finish={cfg.on_finish}",
        flush=True,
    )

    def provision():
        run_cli(
            build_create_command(
                cfg,
                function,
                name=name,
                machine_type=machine_type,
                accelerator=accelerator,
            ),
            label=f"gcloud compute instances create {name}",
            timeout=900,
            dry_run=cfg.dry_run,
        )
        _config_ssh(cfg)
        return (ssh_alias(cfg, name), None)

    def teardown():
        apply_on_finish(cfg, name)

    if cfg.dry_run:
        # Walk the provisioning and teardown commands without dispatching:
        # there is no box to ssh to, and the point is to show what a config
        # would do before it costs anything.
        provision()
        teardown()
        print("+ [dry-run] nothing was created; no dispatch attempted", flush=True)
        return

    run_on_provisioned_vm(
        app=app,
        function=function,
        args=args,
        kwargs=kwargs,
        backend="gcp",
        label=name,
        provision=provision,
        teardown=teardown,
        outputs_dir=outputs_dir,
        mode="docker" if cfg.use_docker else "native",
        max_runtime_seconds=cfg.max_runtime_seconds,
        ssh_ready_wait_seconds=cfg.ssh_ready_wait_seconds,
        # Re-run config-ssh while waiting: the alias only resolves once the
        # instance has an external IP, which lags creation.
        refresh_callback=lambda: _config_ssh(cfg, quiet=True),
    )


def ssh_alias(cfg, name: str) -> str:
    """The alias `gcloud compute config-ssh` writes for this instance."""
    return f"{name}.{cfg.zone}.{cfg.project}"


def resolve_shape(cfg, function) -> tuple:
    """Return (machine_type, accelerator_flag_value or None).

    An explicit `machine_type` wins outright — that's the escape hatch for
    shapes runplz has no table entry for.
    """
    label = resolve_gpu_label(function, GCP_GPUS)
    count = gpu_count(function)

    if cfg.machine_type:
        accelerator = cfg.accelerator
        if accelerator is None and label and not _has_builtin_gpus(cfg.machine_type):
            accelerator, _family, _vram = GCP_GPUS[label]
        return (cfg.machine_type, _accelerator_value(accelerator, count))

    if label is None:
        # CPU-only run. n2-standard scales cleanly with min_cpu.
        return (f"n2-standard-{_cpu_size(function)}", None)

    accel, family, _vram = GCP_GPUS[label]
    if family in ("a2-highgpu", "a2-ultragpu", "a3-highgpu"):
        # A2/A3 shapes ship their accelerators attached; asking for
        # --accelerator on top of them is an error, not a no-op.
        return (f"{family}-{count}g", None)
    if family == "g2-standard":
        # L4 shapes bundle the GPU too; vCPUs scale with GPU count.
        return (f"g2-standard-{4 * count}", None)
    return (f"{family}-{max(8, 4 * count)}", _accelerator_value(cfg.accelerator or accel, count))


def _has_builtin_gpus(machine_type: str) -> bool:
    return machine_type.startswith(("a2-", "a3-", "g2-"))


def _accelerator_value(accelerator: Optional[str], count: int) -> Optional[str]:
    if not accelerator:
        return None
    return f"type={accelerator},count={count}"


def _cpu_size(function) -> int:
    """Round a min_cpu request up to a valid n2-standard size."""
    want = int(getattr(function, "min_cpu", None) or 2)
    for size in (2, 4, 8, 16, 32, 48, 64, 80, 96, 128):
        if size >= want:
            return size
    return 128


def build_create_command(cfg, function, *, name, machine_type, accelerator) -> list:
    """Assemble the `gcloud compute instances create` argv."""
    cmd = [
        "gcloud",
        "compute",
        "instances",
        "create",
        name,
        f"--project={cfg.project}",
        f"--zone={cfg.zone}",
        f"--machine-type={machine_type}",
        f"--image-family={cfg.image_family}",
        f"--image-project={cfg.image_project}",
        # runplz labels every box it makes so a leak is greppable in the
        # console and in billing exports.
        "--labels=runplz=1",
        "--format=json",
        "--quiet",
    ]
    if accelerator:
        cmd.append(f"--accelerator={accelerator}")
        # GPU instances cannot live-migrate; without this GCP rejects the
        # create outright.
        cmd.append("--maintenance-policy=TERMINATE")
    disk_gb = cfg.boot_disk_gb or getattr(function, "min_disk", None)
    if disk_gb:
        cmd.append(f"--boot-disk-size={int(disk_gb)}GB")
    if cfg.boot_disk_type:
        cmd.append(f"--boot-disk-type={cfg.boot_disk_type}")
    if cfg.network:
        cmd.append(f"--network={cfg.network}")
    if cfg.subnet:
        cmd.append(f"--subnet={cfg.subnet}")
    if cfg.service_account:
        cmd.append(f"--service-account={cfg.service_account}")
    if cfg.scopes:
        cmd.append(f"--scopes={cfg.scopes}")
    if cfg.spot:
        cmd.append("--provisioning-model=SPOT")
        # Spot instances that GCP reclaims should stay down; a restart
        # would resume a box with no job on it and bill for it.
        cmd.append("--no-restart-on-failure")
    return cmd


def _config_ssh(cfg, *, quiet: bool = False) -> None:
    """Publish ssh aliases for the project's instances into ~/.ssh/config."""
    run_cli(
        ["gcloud", "compute", "config-ssh", f"--project={cfg.project}", "--quiet"],
        label="gcloud compute config-ssh",
        timeout=120,
        dry_run=cfg.dry_run,
        # During the ssh wait this runs every poll; a transient failure
        # there should not abort a run that is otherwise coming up fine.
        check=not quiet,
    )


def apply_on_finish(cfg, name: str) -> None:
    """Stop / delete / leave the instance per `cfg.on_finish`."""

    def _act(on_finish: str) -> None:
        verb = "delete" if on_finish == "delete" else "stop"
        cmd = [
            "gcloud",
            "compute",
            "instances",
            verb,
            name,
            f"--project={cfg.project}",
            f"--zone={cfg.zone}",
            "--quiet",
        ]
        if verb == "delete":
            # Belt and braces on the "must not leave disks behind" criterion:
            # the boot disk is auto-delete by default, but say so rather than
            # trusting the default.
            cmd.append("--delete-disks=all")
        run_cli(
            cmd,
            label=f"gcloud compute instances {verb} {name}",
            timeout=600,
            # Without this a dry run would really delete things — the one
            # thing dry-run exists to prevent.
            dry_run=cfg.dry_run,
        )

    apply_teardown(
        on_finish=cfg.on_finish,
        target=name,
        run_action=_act,
        where=f" in {cfg.zone}",
        check_hint=(
            f"gcloud compute instances list --project={cfg.project} --filter='name={name}'"
        ),
    )
