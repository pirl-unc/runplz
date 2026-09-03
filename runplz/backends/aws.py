"""AWS backend: provision an EC2 instance, run one function on it, terminate it.

Shells out to the `aws` CLI. Auth is whatever the CLI already has — a
profile, environment variables, or an SSO/instance role.

EC2 has no `gcloud compute config-ssh` equivalent, so unlike the GCP
backend there is no alias to lean on: `key_name` is required and the ssh
target is built from the instance's public IP. Point
`AwsConfig.ssh_key_path` at the private half of that key pair; runplz passes
it to ssh and to rsync's transport.

Networking is pinned, never created. The security group in play must allow
inbound TCP 22 from wherever runplz is running, or the run will sit in the
ssh wait until it times out.
"""

import json
import os
import subprocess

from runplz.backends.provisioning import (
    AWS_CPU_SHAPES,
    AWS_GPUS,
    AWS_RETRY_POLICY,
    AWS_TEARDOWN_POLICY,
    CloudCliError,
    apply_teardown,
    gpu_count,
    make_instance_name,
    resolve_gpu_label,
    run_cli,
    select_machine,
)
from runplz.backends.ssh_common import SshOptions, run_on_provisioned_vm

# `run` and `list_jobs` are the driver contract the registry calls; the
# rest is this driver's own testable surface.
__all__ = [
    "run",
    "resolve_instance_type",
    "instance_type_has_gpu",
    "resolve_ami",
    "build_run_instances_command",
    "apply_on_finish",
    "DEFAULT_ROOT_DEVICE",
    "list_jobs",
]

# Deep Learning AMI ids are region-specific and roll monthly, so resolve the
# current one from SSM instead of hardcoding something that rots.
_DLAMI_SSM_PARAM = (
    "/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id"
)
# CPU-only runs don't need the (large, GPU-driver-laden) DLAMI.
# Both AMIs runplz resolves are Ubuntu-based and root on /dev/sda1.
DEFAULT_ROOT_DEVICE = "/dev/sda1"
_UBUNTU_SSM_PARAM = (
    "/aws/service/canonical/ubuntu/server/22.04/stable/current/amd64/hvm/ebs-gp2/ami-id"
)


def list_jobs(*, region: str | None = None) -> list[dict]:
    """List active runplz-tagged EC2 instances through the real AWS CLI."""
    region = region or os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION")
    if not region:
        raise RuntimeError("AWS region is required; pass --region or set AWS_DEFAULT_REGION")
    r = subprocess.run(
        [
            "aws",
            "ec2",
            "describe-instances",
            "--region",
            region,
            "--filters",
            "Name=tag:runplz,Values=1",
            "Name=instance-state-name,Values=pending,running,stopping,stopped",
            "--output",
            "json",
        ],
        capture_output=True,
        text=True,
    )
    if r.returncode:
        raise RuntimeError(
            f"aws describe-instances failed (rc={r.returncode}): {(r.stderr or '').strip()[:300]}"
        )
    try:
        reservations = json.loads(r.stdout).get("Reservations", [])
    except (TypeError, ValueError) as exc:
        raise RuntimeError("aws describe-instances returned malformed JSON") from exc
    rows = []
    for reservation in reservations:
        for instance in reservation.get("Instances", []):
            tags = {tag.get("Key"): tag.get("Value") for tag in instance.get("Tags", [])}
            name = tags.get("Name") or instance.get("InstanceId", "")
            parts = name.split("-", 2)
            rows.append(
                {
                    "backend": "aws",
                    "name": name,
                    "app": parts[1] if len(parts) > 1 else "",
                    "function": parts[2] if len(parts) > 2 else "",
                    "started": instance.get("LaunchTime", ""),
                    "status": instance.get("State", {}).get("Name", ""),
                }
            )
    return rows


def run(app, function, args, kwargs, *, outputs_dir: str = "out"):
    cfg = app.aws_config
    if not (cfg.key_name or "").strip():
        # Raise before anything is created: no billable state exists yet, so
        # this must not go through the teardown path.
        raise RuntimeError(
            "AwsConfig.key_name is required — EC2 gives you no way to ssh in "
            "without a key pair, so runplz would provision a box it cannot "
            "reach.\n"
            "  - List your key pairs: aws ec2 describe-key-pairs "
            f"--region {cfg.region or '<region>'}\n"
            "  - Then: AwsConfig(region=..., key_name='my-key', "
            "ssh_key_path='~/.ssh/my-key.pem')\n"
            "  - ssh_key_path is optional if the key is already agent-loaded "
            "or named in your ~/.ssh/config."
        )

    name = make_instance_name(app.name, function.name)
    instance_type = resolve_instance_type(cfg, function)
    state = {"instance_id": None}

    print(
        f"+ aws: name={name} region={cfg.region} instance-type={instance_type} "
        f"on_finish={cfg.on_finish}",
        flush=True,
    )

    def provision():
        ami = cfg.ami or resolve_ami(cfg, function)
        created = run_cli(
            build_run_instances_command(
                cfg,
                function,
                name=name,
                instance_type=instance_type,
                ami=ami,
                # Known only for the AMIs runplz resolves itself.
                root_device=None if cfg.ami else DEFAULT_ROOT_DEVICE,
            ),
            label=f"aws ec2 run-instances ({name})",
            timeout=900,
            policy=AWS_RETRY_POLICY,
            dry_run=cfg.dry_run,
            parse_json=True,
        )
        instance_id = _instance_id_from(created)
        state["instance_id"] = instance_id
        _wait_running(cfg, instance_id)
        ip = _public_ip(cfg, instance_id)
        # The target is only knowable after the box exists, so the options
        # travel back with it.
        return (f"{cfg.ssh_user}@{ip}", SshOptions(identity_file=cfg.ssh_key_path))

    def teardown():
        apply_on_finish(cfg, state["instance_id"], name=name)

    if cfg.dry_run:
        # Show the whole command sequence without creating or dispatching
        # anything. state["instance_id"] stays None and the renderers below
        # substitute a placeholder.
        provision()
        teardown()
        print("+ [dry-run] nothing was created; no dispatch attempted", flush=True)
        return

    run_on_provisioned_vm(
        app=app,
        function=function,
        args=args,
        kwargs=kwargs,
        backend="aws",
        label=name,
        provision=provision,
        teardown=teardown,
        outputs_dir=outputs_dir,
        mode="docker" if cfg.use_docker else "native",
        max_runtime_seconds=cfg.max_runtime_seconds,
        ssh_ready_wait_seconds=cfg.ssh_ready_wait_seconds,
    )


def resolve_instance_type(cfg, function) -> str:
    """Pick an EC2 instance type for the function's resource request."""
    if cfg.instance_type:
        return cfg.instance_type

    label = resolve_gpu_label(function, AWS_GPUS)
    if label is None:
        return select_machine(function, AWS_CPU_SHAPES, cloud="EC2").name
    return select_machine(
        function,
        AWS_GPUS[label].offerings,
        cloud="EC2",
        gpus=gpu_count(function),
        gpu_label=label,
    ).name


def instance_type_has_gpu(instance_type: str) -> bool:
    """True when an EC2 instance type is a GPU shape."""
    family = (instance_type or "").split(".", 1)[0]
    return any(family == entry.family for entry in AWS_GPUS.values())


def resolve_ami(cfg, function) -> str:
    """Resolve the current Deep Learning (or plain Ubuntu) AMI via SSM."""
    # Follow the instance type we are actually launching, not just the
    # function's gpu= field: an explicitly pinned GPU instance_type with no
    # gpu= would otherwise get a driverless Ubuntu AMI and run on CPU.
    wants_gpu = bool(getattr(function, "gpu", None) or getattr(function, "min_gpu_memory", None))
    if not wants_gpu:
        wants_gpu = instance_type_has_gpu(resolve_instance_type(cfg, function))
    param = _DLAMI_SSM_PARAM if wants_gpu else _UBUNTU_SSM_PARAM
    result = run_cli(
        [
            "aws",
            "ssm",
            "get-parameter",
            "--region",
            cfg.region,
            "--name",
            param,
            "--query",
            "Parameter.Value",
            "--output",
            "text",
        ],
        label="aws ssm get-parameter (resolve AMI)",
        timeout=120,
        policy=AWS_RETRY_POLICY,
        dry_run=cfg.dry_run,
    )
    if result is None:  # dry-run
        return "<ami-resolved-at-run-time>"
    ami = (result.stdout or "").strip()
    if not ami.startswith("ami-"):
        raise CloudCliError(
            f"Could not resolve an AMI from SSM parameter {param!r} in "
            f"{cfg.region}; got {ami[:200]!r}. Pass AwsConfig(ami='ami-...') "
            f"to pin one yourself."
        )
    return ami


def build_run_instances_command(
    cfg, function, *, name, instance_type, ami, root_device=None
) -> list:
    """Assemble the `aws ec2 run-instances` argv."""
    cmd = [
        "aws",
        "ec2",
        "run-instances",
        "--region",
        cfg.region,
        "--image-id",
        ami,
        "--instance-type",
        instance_type,
        "--key-name",
        cfg.key_name,
        "--count",
        "1",
        # Makes RunInstances idempotent for 24h. Without it, a retry after a
        # launch whose *response* was lost starts a second instance that
        # nothing tears down — an extra p4d.24xlarge billing forever. The
        # run name is unique per dispatch, which is exactly the scope wanted.
        "--client-token",
        name,
        # runplz tags every instance it makes so a leak is greppable in the
        # console and in Cost Explorer.
        "--tag-specifications",
        (f"ResourceType=instance,Tags=[{{Key=Name,Value={name}}},{{Key=runplz,Value=1}}]"),
        "--output",
        "json",
    ]
    if cfg.subnet_id:
        cmd += ["--subnet-id", cfg.subnet_id]
    if cfg.security_group_id:
        cmd += ["--security-group-ids", cfg.security_group_id]
    if cfg.iam_instance_profile:
        cmd += ["--iam-instance-profile", f"Name={cfg.iam_instance_profile}"]

    volume_gb = cfg.volume_gb or getattr(function, "min_disk", None)
    # The mapping is what pins DeleteOnTermination, so on_finish="delete"
    # takes the disk rather than leaving a billed EBS volume behind.
    #
    # It only works when DeviceName matches the AMI's actual root device:
    # naming the wrong one attaches a *second* volume and leaves the real
    # root on the AMI's own settings. We know the device for the Deep
    # Learning / Ubuntu AMIs runplz resolves; for a user-supplied `ami=` we
    # don't, so send nothing rather than silently attaching a stray disk.
    if cfg.ami and root_device is None:
        if volume_gb:
            print(
                f"+ warning: ignoring disk size {int(volume_gb)}GB — runplz "
                f"cannot tell the root device name of a user-supplied AMI "
                f"({cfg.ami}). Bake the size into the AMI, or drop ami= to "
                f"let runplz resolve one.",
                flush=True,
            )
    else:
        cmd += [
            "--block-device-mappings",
            (
                "[{"
                f'"DeviceName":"{root_device or DEFAULT_ROOT_DEVICE}",'
                '"Ebs":{'
                + (f'"VolumeSize":{int(volume_gb)},' if volume_gb else "")
                + '"DeleteOnTermination":true,"VolumeType":"gp3"'
                "}}]"
            ),
        ]
    if cfg.spot:
        cmd += [
            "--instance-market-options",
            '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time"}}',
        ]
    return cmd


def _instance_id_from(created) -> str:
    if created is None:  # dry-run
        return "<instance-id-assigned-at-run-time>"
    try:
        instance_id = created["Instances"][0]["InstanceId"]
    except (KeyError, IndexError, TypeError) as exc:
        raise CloudCliError(
            f"`aws ec2 run-instances` returned no instance id: {str(created)[:500]}"
        ) from exc
    print(f"+ aws: launched {instance_id}", flush=True)
    return instance_id


def _wait_running(cfg, instance_id: str) -> None:
    run_cli(
        [
            "aws",
            "ec2",
            "wait",
            "instance-running",
            "--region",
            cfg.region,
            "--instance-ids",
            instance_id,
        ],
        label=f"aws ec2 wait instance-running {instance_id}",
        timeout=900,
        dry_run=cfg.dry_run,
    )


def _public_ip(cfg, instance_id: str) -> str:
    result = run_cli(
        [
            "aws",
            "ec2",
            "describe-instances",
            "--region",
            cfg.region,
            "--instance-ids",
            instance_id,
            "--query",
            "Reservations[0].Instances[0].PublicIpAddress",
            "--output",
            "text",
        ],
        label=f"aws ec2 describe-instances {instance_id}",
        timeout=120,
        policy=AWS_RETRY_POLICY,
        dry_run=cfg.dry_run,
    )
    if result is None:  # dry-run
        return "<public-ip-assigned-at-run-time>"
    ip = (result.stdout or "").strip()
    if not ip or ip == "None":
        raise CloudCliError(
            f"{instance_id} came up with no public IP. runplz reaches the box "
            f"over ssh, so it needs one: launch into a subnet that "
            f"auto-assigns public IPs, or pass a subnet_id that does."
        )
    print(f"+ aws: {instance_id} reachable at {ip}", flush=True)
    return ip


def apply_on_finish(cfg, instance_id, *, name: str) -> None:
    """Terminate / stop / leave the instance per `cfg.on_finish`."""
    if instance_id is None and not cfg.dry_run:
        # run-instances never got far enough to return an id. Nothing to
        # clean up, but say so — silence here reads as "cleaned up".
        print(
            f"+ on_finish={cfg.on_finish}: no instance id recorded for {name}; "
            f"nothing to tear down. If run-instances did launch something, "
            f"check: aws ec2 describe-instances --region {cfg.region} "
            f"--filters Name=tag:Name,Values={name}",
            flush=True,
        )
        return

    def _act(on_finish: str) -> None:
        action = "terminate-instances" if on_finish == "delete" else "stop-instances"
        run_cli(
            [
                "aws",
                "ec2",
                action,
                "--region",
                cfg.region,
                "--instance-ids",
                instance_id,
                "--output",
                "json",
            ],
            label=f"aws ec2 {action} {instance_id}",
            timeout=600,
            # One quick retry for a blip; every second of backoff here is
            # a second in which a Ctrl-C abandons the terminate.
            policy=AWS_TEARDOWN_POLICY,
            dry_run=cfg.dry_run,
        )

    apply_teardown(
        on_finish=cfg.on_finish,
        target=instance_id,
        run_action=_act,
        where=f" in {cfg.region}",
        check_hint=(
            f"aws ec2 describe-instances --region {cfg.region} --instance-ids {instance_id}"
        ),
    )
