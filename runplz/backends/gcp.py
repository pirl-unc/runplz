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

import json
import os
import subprocess
from typing import Optional

from runplz.backends.provisioning import (
    ALREADY_EXISTS,
    GCP_CPU_SHAPES,
    GCP_GPUS,
    GCP_RETRY_POLICY,
    GCP_TEARDOWN_POLICY,
    CloudCliError,
    apply_teardown,
    gpu_count,
    make_instance_name,
    resolve_gpu_label,
    run_cli,
    select_machine,
    split_instance_name,
)
from runplz.backends.ssh_common import run_on_provisioned_vm

# `run` and `list_jobs` are the driver contract the registry calls; the
# rest is this driver's own testable surface.
__all__ = [
    "run",
    "ssh_alias",
    "resolve_shape",
    "build_create_command",
    "apply_on_finish",
    "list_jobs",
]


def list_jobs(*, project: str | None = None, zone: str | None = None) -> list[dict]:
    """List active runplz-labelled GCE instances through the real gcloud CLI."""
    project = project or os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCLOUD_PROJECT")
    zone = zone or os.environ.get("CLOUDSDK_COMPUTE_ZONE")
    if not project:
        raise RuntimeError("GCP project is required; pass --project or set GOOGLE_CLOUD_PROJECT")
    cmd = [
        "gcloud",
        "compute",
        "instances",
        "list",
        f"--project={project}",
        "--filter=labels.runplz=1",
        "--format=json",
    ]
    if zone:
        cmd.append(f"--zones={zone}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(
            f"gcloud instances list failed (rc={r.returncode}): {(r.stderr or '').strip()[:300]}"
        )
    try:
        instances = json.loads(r.stdout)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("gcloud instances list returned malformed JSON") from exc
    rows = []
    for instance in instances if isinstance(instances, list) else []:
        name = instance.get("name", "")
        app_name, fn_name = split_instance_name(name)
        rows.append(
            {
                "backend": "gcp",
                "name": name,
                "app": app_name,
                "function": fn_name,
                "started": instance.get("creationTimestamp", ""),
                "status": instance.get("status", ""),
            }
        )
    return rows


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

    created = {"ok": False}

    def provision():
        try:
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
                policy=GCP_RETRY_POLICY,
            )
        except CloudCliError as exc:
            if ALREADY_EXISTS not in str(exc).lower():
                raise
            # A retried create whose first attempt actually landed. The box
            # exists and is billing, so it must be torn down — reporting
            # "was never created" here would be actively wrong.
            print(
                f"+ gcp: {name} already exists — an earlier attempt landed; continuing with it",
                flush=True,
            )
        created["ok"] = True
        _config_ssh(cfg)
        return (ssh_alias(cfg, name), None)

    def teardown():
        # run_on_provisioned_vm always calls teardown, including when
        # provision() died before the create landed. Deleting a name that
        # was never created would warn about a phantom billing leak.
        if not created["ok"]:
            print(
                f"+ on_finish={cfg.on_finish}: {name} was never created; nothing to tear down.",
                flush=True,
            )
            return
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
        if _has_builtin_gpus(cfg.machine_type):
            # a2/a3/g2 ship their accelerators attached. Passing
            # --accelerator on top is rejected by GCE, so drop it even when
            # the user set one explicitly — which is what the config says
            # happens ("ignored when the machine type has GPUs built in").
            if accelerator:
                print(
                    f"+ gcp: ignoring accelerator={accelerator!r} — "
                    f"{cfg.machine_type} already includes its GPUs",
                    flush=True,
                )
            return (cfg.machine_type, None)
        if accelerator is None and label:
            accelerator = GCP_GPUS[label].accelerator
        return (cfg.machine_type, _accelerator_value(accelerator, count))

    if label is None:
        return (select_machine(function, GCP_CPU_SHAPES, cloud="GCE").name, None)

    entry = GCP_GPUS[label]
    selected = select_machine(
        function,
        entry.offerings,
        cloud="GCE",
        gpus=count,
        gpu_label=label,
    )
    accelerator = None
    if entry.attached:
        accelerator = _accelerator_value(cfg.accelerator or entry.accelerator, count)
    return (selected.name, accelerator)


def _has_builtin_gpus(machine_type: str) -> bool:
    return machine_type.startswith(("a2-", "a3-", "g2-"))


def _accelerator_value(accelerator: Optional[str], count: int) -> Optional[str]:
    if not accelerator:
        return None
    return f"type={accelerator},count={count}"


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
            # One quick retry for a blip; every second of backoff here is
            # a second in which a Ctrl-C abandons the delete.
            policy=GCP_TEARDOWN_POLICY,
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
