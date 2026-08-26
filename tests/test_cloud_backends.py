"""Coverage for the direct GCP / AWS provisioning backends (issue #25).

Nothing here touches a real cloud. `tests/conftest.py` guards `gcloud` and
`aws` behind the `live_gcp` / `live_aws` markers, and no test in this file
opts in — everything asserts on the argv runplz *would* send.
"""

import json
from pathlib import Path
from unittest import mock

import pytest

from runplz import App, AwsConfig, GcpConfig, Image
from runplz.backends import aws, gcp, provisioning


def _app(tmp_path, **configs) -> App:
    app = App("vision-train", **configs)
    app._repo_root = tmp_path
    return app


def _function(app, **spec):
    job = tmp_job(app._repo_root)

    @app.function(image=Image.from_registry("ubuntu:22.04"), **spec)
    def train():
        pass

    train.module_file = str(job)
    return train


def tmp_job(repo: Path) -> Path:
    job = repo / "job.py"
    if not job.exists():
        job.write_text("# job\n")
    return job


def _gcp(**kw):
    kw.setdefault("project", "my-proj")
    kw.setdefault("zone", "us-central1-a")
    return GcpConfig(**kw)


def _aws(**kw):
    kw.setdefault("region", "us-east-1")
    kw.setdefault("key_name", "my-key")
    return AwsConfig(**kw)


# ---------------------------------------------------------------------------
# config validation


@pytest.mark.parametrize(
    "ctor,kw,match",
    [
        (GcpConfig, {"zone": "us-central1-a"}, "project is required"),
        (GcpConfig, {"project": "p"}, "zone is required"),
        (AwsConfig, {}, "region is required"),
        (GcpConfig, {"project": "p", "zone": "z", "on_finish": "nuke"}, "on_finish"),
        (AwsConfig, {"region": "r", "on_finish": "nuke"}, "on_finish"),
        (GcpConfig, {"project": "p", "zone": "z", "max_runtime_seconds": 0}, "max_runtime"),
        (AwsConfig, {"region": "r", "ssh_ready_wait_seconds": 0}, "ssh_ready_wait"),
    ],
)
def test_cloud_configs_reject_bad_input(ctor, kw, match):
    with pytest.raises(ValueError, match=match):
        ctor(**kw)


def test_cloud_backends_need_their_config(tmp_path):
    app = _app(tmp_path)
    _function(app)
    with pytest.raises(ValueError, match="needs App"):
        app.bind("gcp")
    with pytest.raises(ValueError, match="needs App"):
        app.bind("aws")


def test_cloud_backends_reject_instance_and_host(tmp_path):
    app = _app(tmp_path, gcp_config=_gcp())
    _function(app)
    with pytest.raises(ValueError, match="only applies to the brev backend"):
        app.bind("gcp", instance="some-box")
    with pytest.raises(ValueError, match="only applies to the ssh backend"):
        app.bind("gcp", host="some-host")


# ---------------------------------------------------------------------------
# instance naming


def test_instance_names_are_dns_safe_prefixed_and_unique():
    a = provisioning.make_instance_name("My App!", "Train_Model")
    b = provisioning.make_instance_name("My App!", "Train_Model")
    assert a.startswith("runplz-")
    assert a != b, "two concurrent runs would collide on the name"
    assert all(c.islower() or c.isdigit() or c == "-" for c in a), a
    assert "--" not in a.strip("-")


def test_instance_name_survives_an_all_punctuation_input():
    name = provisioning.make_instance_name("!!!", "???")
    assert name.startswith("runplz-x-x-")


# ---------------------------------------------------------------------------
# GPU shape resolution


@pytest.mark.parametrize(
    "spec,expect_machine",
    [
        ({"gpu": "A100-80GB", "min_gpus": 8}, "a2-ultragpu-8g"),
        ({"gpu": "A100-40GB", "min_gpus": 2}, "a2-highgpu-2g"),
        ({"gpu": "H100", "min_gpus": 8}, "a3-highgpu-8g"),
        ({"gpu": "L4"}, "g2-standard-4"),
        ({"gpu": "T4"}, "n1-standard-8"),
    ],
)
def test_gcp_machine_type_derived_from_gpu(tmp_path, spec, expect_machine):
    app = _app(tmp_path, gcp_config=_gcp())
    fn = _function(app, **spec)
    machine, _accel = gcp.resolve_shape(app.gcp_config, fn)
    assert machine == expect_machine


def test_gcp_bundled_gpu_shapes_do_not_also_request_an_accelerator(tmp_path):
    """a2/a3/g2 ship their GPUs attached; --accelerator on top is an error."""
    for spec in (
        {"gpu": "A100-80GB", "min_gpus": 8},
        {"gpu": "H100", "min_gpus": 8},
        {"gpu": "L4"},
    ):
        app = _app(tmp_path, gcp_config=_gcp())
        fn = _function(app, **spec)
        _machine, accel = gcp.resolve_shape(app.gcp_config, fn)
        assert accel is None, spec


def test_gcp_t4_attaches_an_accelerator_with_the_right_count(tmp_path):
    app = _app(tmp_path, gcp_config=_gcp())
    fn = _function(app, gpu="T4", min_gpus=4)
    _machine, accel = gcp.resolve_shape(app.gcp_config, fn)
    assert accel == "type=nvidia-tesla-t4,count=4"


def test_explicit_machine_type_wins(tmp_path):
    app = _app(tmp_path, gcp_config=_gcp(machine_type="custom-96-1024"))
    fn = _function(app, gpu="T4")
    machine, _accel = gcp.resolve_shape(app.gcp_config, fn)
    assert machine == "custom-96-1024"


def test_min_gpu_memory_picks_the_smallest_gpu_that_fits(tmp_path):
    app = _app(tmp_path, gcp_config=_gcp())
    fn = _function(app, min_gpu_memory=40)
    assert provisioning.resolve_gpu_label(fn, provisioning.GCP_GPUS) == "A100-40GB"


def test_min_gpu_memory_beyond_every_known_gpu_is_an_error(tmp_path):
    app = _app(tmp_path, gcp_config=_gcp())
    fn = _function(app, min_gpu_memory=500)
    with pytest.raises(provisioning.CloudCliError, match="exceeds every GPU"):
        provisioning.resolve_gpu_label(fn, provisioning.GCP_GPUS)


def test_unknown_gpu_label_names_the_alternatives(tmp_path):
    app = _app(tmp_path, gcp_config=_gcp())
    fn = _function(app, gpu="A10G")  # valid on AWS, absent from the GCP table
    with pytest.raises(provisioning.CloudCliError, match="no mapping for this cloud"):
        provisioning.resolve_gpu_label(fn, provisioning.GCP_GPUS)


def test_no_gpu_request_yields_a_cpu_shape(tmp_path):
    app = _app(tmp_path, gcp_config=_gcp())
    fn = _function(app, min_cpu=6)
    machine, accel = gcp.resolve_shape(app.gcp_config, fn)
    assert machine == "n2-standard-8"
    assert accel is None


@pytest.mark.parametrize(
    "spec,expect",
    [
        ({"gpu": "T4"}, "g4dn.xlarge"),
        ({"gpu": "A10G", "min_gpus": 4}, "g5.12xlarge"),
        ({"gpu": "H100", "min_gpus": 8}, "p5.48xlarge"),
        ({"gpu": "A100-80GB", "min_gpus": 8}, "p4de.24xlarge"),
        ({"min_cpu": 3}, "m6i.xlarge"),
    ],
)
def test_aws_instance_type_derived_from_gpu(tmp_path, spec, expect):
    app = _app(tmp_path, aws_config=_aws())
    fn = _function(app, **spec)
    assert aws.resolve_instance_type(app.aws_config, fn) == expect


def test_aws_rejects_an_impossible_gpu_count_before_ec2_does(tmp_path):
    app = _app(tmp_path, aws_config=_aws())
    fn = _function(app, gpu="H100", min_gpus=2)
    with pytest.raises(provisioning.CloudCliError, match=r"sells H100 in \[8\] GPU counts"):
        aws.resolve_instance_type(app.aws_config, fn)


# ---------------------------------------------------------------------------
# rendered commands


def test_gcp_create_command_shape(tmp_path):
    cfg = _gcp(network="my-vpc", subnet="my-subnet", service_account="sa@x.iam", spot=True)
    app = _app(tmp_path, gcp_config=cfg)
    fn = _function(app, gpu="T4", min_disk=200)
    machine, accel = gcp.resolve_shape(cfg, fn)
    cmd = gcp.build_create_command(cfg, fn, name="box-1", machine_type=machine, accelerator=accel)

    assert cmd[:5] == ["gcloud", "compute", "instances", "create", "box-1"]
    assert "--project=my-proj" in cmd
    assert "--zone=us-central1-a" in cmd
    assert "--accelerator=type=nvidia-tesla-t4,count=1" in cmd
    # GPU boxes cannot live-migrate; GCP rejects the create without this.
    assert "--maintenance-policy=TERMINATE" in cmd
    assert "--boot-disk-size=200GB" in cmd
    assert "--network=my-vpc" in cmd
    assert "--subnet=my-subnet" in cmd
    assert "--service-account=sa@x.iam" in cmd
    assert "--provisioning-model=SPOT" in cmd
    assert "--labels=runplz=1" in cmd


def test_gcp_cpu_only_create_omits_gpu_flags(tmp_path):
    cfg = _gcp()
    app = _app(tmp_path, gcp_config=cfg)
    fn = _function(app, min_cpu=4)
    machine, accel = gcp.resolve_shape(cfg, fn)
    cmd = gcp.build_create_command(cfg, fn, name="b", machine_type=machine, accelerator=accel)
    assert not any(c.startswith("--accelerator") for c in cmd)
    assert "--maintenance-policy=TERMINATE" not in cmd


def test_gcp_ssh_alias_matches_config_ssh_naming(tmp_path):
    cfg = _gcp()
    assert gcp.ssh_alias(cfg, "box-1") == "box-1.us-central1-a.my-proj"


def test_aws_run_instances_command_shape(tmp_path):
    cfg = _aws(
        subnet_id="subnet-1", security_group_id="sg-1", iam_instance_profile="prof", spot=True
    )
    app = _app(tmp_path, aws_config=cfg)
    fn = _function(app, gpu="T4", min_disk=300)
    cmd = aws.build_run_instances_command(
        cfg, fn, name="box-1", instance_type="g4dn.xlarge", ami="ami-123"
    )
    assert cmd[:3] == ["aws", "ec2", "run-instances"]
    assert "--key-name" in cmd and "my-key" in cmd
    assert "ami-123" in cmd
    assert "subnet-1" in cmd
    assert "sg-1" in cmd
    assert "Name=prof" in cmd
    assert any("MarketType" in c for c in cmd)
    assert any("Key=runplz,Value=1" in c for c in cmd)

    mapping = json.loads(cmd[cmd.index("--block-device-mappings") + 1])
    assert mapping[0]["Ebs"]["VolumeSize"] == 300
    # This is what makes on_finish="delete" actually take the disk with it.
    assert mapping[0]["Ebs"]["DeleteOnTermination"] is True


def test_aws_always_pins_delete_on_termination_even_without_a_size(tmp_path):
    """Otherwise on_finish='delete' leaves a billed EBS volume behind."""
    cfg = _aws()
    app = _app(tmp_path, aws_config=cfg)
    fn = _function(app)
    cmd = aws.build_run_instances_command(cfg, fn, name="b", instance_type="m6i.large", ami="ami-1")
    mapping = json.loads(cmd[cmd.index("--block-device-mappings") + 1])
    assert mapping[0]["Ebs"]["DeleteOnTermination"] is True
    assert "VolumeSize" not in mapping[0]["Ebs"]


def test_aws_gpu_run_resolves_the_deep_learning_ami(tmp_path):
    cfg = _aws()
    app = _app(tmp_path, aws_config=cfg)
    fn = _function(app, gpu="T4")
    with mock.patch.object(provisioning.subprocess, "run") as run_mock:
        run_mock.return_value = mock.Mock(returncode=0, stdout="ami-abc123\n", stderr="")
        assert aws.resolve_ami(cfg, fn) == "ami-abc123"
    sent = run_mock.call_args.args[0]
    assert "deeplearning" in " ".join(sent)


def test_aws_cpu_run_does_not_pull_the_gpu_ami(tmp_path):
    cfg = _aws()
    app = _app(tmp_path, aws_config=cfg)
    fn = _function(app, min_cpu=2)
    with mock.patch.object(provisioning.subprocess, "run") as run_mock:
        run_mock.return_value = mock.Mock(returncode=0, stdout="ami-plain\n", stderr="")
        aws.resolve_ami(cfg, fn)
    sent = " ".join(run_mock.call_args.args[0])
    assert "deeplearning" not in sent
    assert "canonical/ubuntu" in sent


def test_aws_rejects_ssm_output_that_is_not_an_ami(tmp_path):
    cfg = _aws()
    app = _app(tmp_path, aws_config=cfg)
    fn = _function(app, gpu="T4")
    with mock.patch.object(provisioning.subprocess, "run") as run_mock:
        run_mock.return_value = mock.Mock(returncode=0, stdout="None\n", stderr="")
        with pytest.raises(provisioning.CloudCliError, match="Could not resolve an AMI"):
            aws.resolve_ami(cfg, fn)


def test_aws_requires_a_key_name(tmp_path):
    app = _app(tmp_path, aws_config=AwsConfig(region="us-east-1"))
    fn = _function(app)
    with pytest.raises(RuntimeError, match="key_name is required"):
        aws.run(app, fn, [], {})


def test_aws_missing_public_ip_is_a_clear_error(tmp_path):
    cfg = _aws()
    with mock.patch.object(provisioning.subprocess, "run") as run_mock:
        run_mock.return_value = mock.Mock(returncode=0, stdout="None\n", stderr="")
        with pytest.raises(provisioning.CloudCliError, match="no public IP"):
            aws._public_ip(cfg, "i-123")


# ---------------------------------------------------------------------------
# teardown


@pytest.mark.parametrize(
    "on_finish,expect_verb",
    [("delete", "delete"), ("stop", "stop")],
)
def test_gcp_on_finish_runs_the_right_verb(on_finish, expect_verb):
    cfg = _gcp(on_finish=on_finish)
    with mock.patch.object(provisioning.subprocess, "run") as run_mock:
        run_mock.return_value = mock.Mock(returncode=0, stdout="", stderr="")
        gcp.apply_on_finish(cfg, "box-1")
    sent = run_mock.call_args.args[0]
    assert sent[:4] == ["gcloud", "compute", "instances", expect_verb]
    if expect_verb == "delete":
        assert "--delete-disks=all" in sent


def test_gcp_on_finish_leave_touches_nothing():
    with mock.patch.object(provisioning.subprocess, "run") as run_mock:
        gcp.apply_on_finish(_gcp(on_finish="leave"), "box-1")
    run_mock.assert_not_called()


def test_gcp_teardown_failure_warns_loudly_but_does_not_raise(capsys):
    """A raise here would mask the real error from the run itself."""
    with mock.patch.object(provisioning.subprocess, "run") as run_mock:
        run_mock.return_value = mock.Mock(returncode=1, stdout="", stderr="quota exceeded")
        gcp.apply_on_finish(_gcp(), "box-1")
    out = capsys.readouterr().out
    assert "warning" in out
    assert "may still exist and still be billing" in out


@pytest.mark.parametrize(
    "on_finish,expect",
    [("delete", "terminate-instances"), ("stop", "stop-instances")],
)
def test_aws_on_finish_runs_the_right_action(on_finish, expect):
    cfg = _aws(on_finish=on_finish)
    with mock.patch.object(provisioning.subprocess, "run") as run_mock:
        run_mock.return_value = mock.Mock(returncode=0, stdout="{}", stderr="")
        aws.apply_on_finish(cfg, "i-123", name="box-1")
    assert run_mock.call_args.args[0][2] == expect


def test_aws_teardown_without_an_instance_id_says_so(capsys):
    """Silence here would read as 'cleaned up' when nothing was checked."""
    with mock.patch.object(provisioning.subprocess, "run") as run_mock:
        aws.apply_on_finish(_aws(), None, name="box-1")
    run_mock.assert_not_called()
    assert "nothing to tear down" in capsys.readouterr().out


def test_aws_teardown_failure_warns_loudly_but_does_not_raise(capsys):
    with mock.patch.object(provisioning.subprocess, "run") as run_mock:
        run_mock.return_value = mock.Mock(returncode=1, stdout="", stderr="boom")
        aws.apply_on_finish(_aws(), "i-123", name="box-1")
    assert "still be billing" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# dry run: nothing may execute


@pytest.mark.parametrize("backend", ["gcp", "aws"])
def test_dry_run_executes_absolutely_nothing(tmp_path, backend, capsys):
    """The one thing dry-run must guarantee. Teardown included — an early
    version of this backend really did delete during a dry run."""
    configs = {
        "gcp": {"gcp_config": _gcp(dry_run=True)},
        "aws": {"aws_config": _aws(dry_run=True)},
    }[backend]
    app = _app(tmp_path, **configs)
    fn = _function(app, gpu="T4", min_disk=100)
    app.bind(backend)

    with mock.patch.object(provisioning.subprocess, "run") as run_mock:
        fn.remote()
    run_mock.assert_not_called()

    out = capsys.readouterr().out
    assert "[dry-run]" in out
    assert "nothing was created" in out
    # Both the create and the teardown must be shown, not just the create.
    assert ("instances create" in out) or ("run-instances" in out)
    assert ("instances delete" in out) or ("terminate-instances" in out)


# ---------------------------------------------------------------------------
# CLI invocation helper


def test_run_cli_surfaces_stderr_on_failure():
    with mock.patch.object(provisioning.subprocess, "run") as run_mock:
        run_mock.return_value = mock.Mock(
            returncode=1, stdout="", stderr="ZONE_RESOURCE_POOL_EXHAUSTED"
        )
        with pytest.raises(provisioning.CloudCliError, match="ZONE_RESOURCE_POOL_EXHAUSTED"):
            provisioning.run_cli(["gcloud", "x"], label="gcloud x")


def test_run_cli_explains_a_missing_cli():
    with mock.patch.object(provisioning.subprocess, "run", side_effect=FileNotFoundError()):
        with pytest.raises(provisioning.CloudCliError, match="not found on PATH"):
            provisioning.run_cli(["gcloud", "x"], label="gcloud x")


def test_run_cli_check_false_tolerates_failure():
    with mock.patch.object(provisioning.subprocess, "run") as run_mock:
        run_mock.return_value = mock.Mock(returncode=1, stdout="", stderr="transient")
        assert provisioning.run_cli(["gcloud", "x"], label="x", check=False) is not None


def test_run_cli_rejects_non_json_when_json_expected():
    with mock.patch.object(provisioning.subprocess, "run") as run_mock:
        run_mock.return_value = mock.Mock(returncode=0, stdout="not json", stderr="")
        with pytest.raises(provisioning.CloudCliError, match="not JSON"):
            provisioning.run_cli(["aws", "x"], label="x", parse_json=True)


# ---------------------------------------------------------------------------
# shared teardown contract


def test_teardown_leave_announces_and_does_nothing(capsys):
    calls = []
    provisioning.apply_teardown(
        on_finish="leave",
        target="box-1",
        run_action=calls.append,
        check_hint="check me",
        where=" in us-east-1",
    )
    assert calls == []
    assert "left running in us-east-1" in capsys.readouterr().out


def test_teardown_never_raises_but_always_shouts(capsys):
    """It runs in a finally: raising masks the real error, silence leaks a box."""

    def boom(_on_finish):
        raise RuntimeError("provider said no")

    provisioning.apply_teardown(
        on_finish="delete",
        target="box-1",
        run_action=boom,
        check_hint="gcloud compute instances list",
    )
    out = capsys.readouterr().out
    assert "warning" in out
    assert "provider said no" in out
    assert "still be billing" in out
    assert "gcloud compute instances list" in out


def test_teardown_passes_the_on_finish_value_through():
    seen = []
    provisioning.apply_teardown(
        on_finish="stop", target="b", run_action=seen.append, check_hint="x"
    )
    assert seen == ["stop"]


# ---------------------------------------------------------------------------
# naming is a round-trip contract shared by every provisioning backend


def test_instance_name_round_trips_through_split():
    name = provisioning.make_instance_name("vision-train", "train")
    assert provisioning.split_instance_name(name) == ("vision-train", "train")


def test_split_rejects_names_that_are_not_ours():
    for name in ("my-own-box", "runplz", "runplz-x", ""):
        assert provisioning.split_instance_name(name) == ("", "")


def test_brev_and_cloud_backends_share_one_naming_contract():
    """A drift here would make `runplz ps` stop recognising its own boxes."""
    from runplz.backends import brev

    assert brev._make_ephemeral_name is provisioning.make_instance_name
    assert brev._split_ephemeral_name is provisioning.split_instance_name


# ---------------------------------------------------------------------------
# machine types are real, and carry the GPU count that was asked for
#
# The first cut derived them arithmetically, which invented p3.xlarge and
# g4dn.24xlarge (neither exists) and — worse, because it launches — mapped
# min_gpus=8 onto g5.24xlarge and g2-standard-32, which carry 4 and 1 GPUs.


@pytest.mark.parametrize(
    "gpu,count,expect",
    [
        ("V100", 1, "p3.2xlarge"),
        ("V100", 4, "p3.8xlarge"),
        ("V100", 8, "p3.16xlarge"),
        ("T4", 1, "g4dn.xlarge"),
        ("T4", 4, "g4dn.12xlarge"),
        ("A10G", 8, "g5.48xlarge"),
        ("L40S", 4, "g6e.12xlarge"),
        ("H100", 8, "p5.48xlarge"),
    ],
)
def test_aws_instance_types_exist_and_carry_the_requested_gpus(tmp_path, gpu, count, expect):
    app = _app(tmp_path, aws_config=_aws())
    fn = _function(app, gpu=gpu, min_gpus=count)
    assert aws.resolve_instance_type(app.aws_config, fn) == expect


@pytest.mark.parametrize(
    "gpu,count,expect",
    [
        ("H100", 8, "a3-highgpu-8g"),
        ("A100-40GB", 2, "a2-highgpu-2g"),
        ("A100-80GB", 8, "a2-ultragpu-8g"),
        ("L4", 1, "g2-standard-4"),
        ("L4", 2, "g2-standard-24"),
        ("L4", 4, "g2-standard-48"),
        ("L4", 8, "g2-standard-96"),
    ],
)
def test_gcp_machine_types_exist_and_carry_the_requested_gpus(tmp_path, gpu, count, expect):
    app = _app(tmp_path, gcp_config=_gcp())
    fn = _function(app, gpu=gpu, min_gpus=count)
    machine, _accel = gcp.resolve_shape(app.gcp_config, fn)
    assert machine == expect


@pytest.mark.parametrize(
    "gpu,count",
    [("H100", 1), ("H100", 2), ("A100-40GB", 3), ("L4", 3), ("T4", 2)],
)
def test_gcp_refuses_a_gpu_count_the_family_does_not_sell(tmp_path, gpu, count):
    """Better than a fabricated name that gcloud rejects — or worse, accepts."""
    app = _app(tmp_path, gcp_config=_gcp())
    fn = _function(app, gpu=gpu, min_gpus=count)
    entry = provisioning.GCP_GPUS[gpu]
    if entry.shapes is None or count in entry.shapes:
        pytest.skip("this count is sold")
    with pytest.raises(provisioning.CloudCliError, match="GPU counts"):
        gcp.resolve_shape(app.gcp_config, fn)


@pytest.mark.parametrize("gpu,count", [("H100", 2), ("A100-40GB", 4), ("T4", 3)])
def test_aws_refuses_a_gpu_count_the_family_does_not_sell(tmp_path, gpu, count):
    app = _app(tmp_path, aws_config=_aws())
    fn = _function(app, gpu=gpu, min_gpus=count)
    if count in provisioning.AWS_GPUS[gpu].shapes:
        pytest.skip("this count is sold")
    with pytest.raises(provisioning.CloudCliError, match="GPU counts"):
        aws.resolve_instance_type(app.aws_config, fn)


def test_min_memory_sizes_the_box_on_both_clouds(tmp_path):
    """Both configs document min_memory as an input; it used to be ignored."""
    gapp = _app(tmp_path, gcp_config=_gcp())
    gfn = _function(gapp, min_memory=256)
    machine, _ = gcp.resolve_shape(gapp.gcp_config, gfn)
    assert machine == "n2-standard-64", "n2-standard gives 4GB per vCPU"

    aapp = App("x", aws_config=_aws())
    aapp._repo_root = tmp_path

    @aapp.function(image=Image.from_registry("ubuntu:22.04"), min_memory=128)
    def big():
        pass

    big.module_file = str(tmp_job(tmp_path))
    assert aws.resolve_instance_type(aapp.aws_config, big) == "m6i.8xlarge"


def test_explicit_accelerator_is_dropped_on_a_bundled_gpu_shape(tmp_path, capsys):
    """GCE rejects --accelerator on a2/a3/g2; the config says it is ignored."""
    cfg = _gcp(machine_type="a2-highgpu-1g", accelerator="nvidia-tesla-a100")
    app = _app(tmp_path, gcp_config=cfg)
    fn = _function(app, gpu="A100-40GB")
    machine, accel = gcp.resolve_shape(cfg, fn)
    assert machine == "a2-highgpu-1g"
    assert accel is None
    assert "already includes its GPUs" in capsys.readouterr().out


def test_pinned_gpu_instance_type_still_gets_the_gpu_ami(tmp_path):
    """Without gpu=, an explicit g5 shape used to get a driverless AMI."""
    cfg = _aws(instance_type="g5.12xlarge")
    app = _app(tmp_path, aws_config=cfg)
    fn = _function(app)  # no gpu= at all
    with mock.patch.object(provisioning.subprocess, "run") as run_mock:
        run_mock.return_value = mock.Mock(returncode=0, stdout="ami-gpu\n", stderr="")
        aws.resolve_ami(cfg, fn)
    assert "deeplearning" in " ".join(run_mock.call_args.args[0])


def test_user_supplied_ami_gets_no_block_device_mapping(tmp_path, capsys):
    """Naming the wrong root device attaches a stray second volume."""
    cfg = _aws(ami="ami-custom")
    app = _app(tmp_path, aws_config=cfg)
    fn = _function(app, min_disk=500)
    cmd = aws.build_run_instances_command(
        cfg, fn, name="b", instance_type="m6i.large", ami="ami-custom", root_device=None
    )
    assert "--block-device-mappings" not in cmd
    assert "cannot tell the root device name" in capsys.readouterr().out


def test_resolved_ami_still_pins_delete_on_termination(tmp_path):
    cfg = _aws()
    app = _app(tmp_path, aws_config=cfg)
    fn = _function(app, min_disk=200)
    cmd = aws.build_run_instances_command(
        cfg,
        fn,
        name="b",
        instance_type="m6i.large",
        ami="ami-1",
        root_device=aws.DEFAULT_ROOT_DEVICE,
    )
    mapping = json.loads(cmd[cmd.index("--block-device-mappings") + 1])
    assert mapping[0]["DeviceName"] == aws.DEFAULT_ROOT_DEVICE
    assert mapping[0]["Ebs"]["DeleteOnTermination"] is True


def test_gcp_teardown_skips_a_vm_that_was_never_created(tmp_path, capsys):
    """Teardown is unconditional now; it must not delete a phantom."""
    cfg = _gcp()
    app = _app(tmp_path, gcp_config=cfg)
    fn = _function(app, gpu="T4")

    calls = []

    def fake_run(cmd, **kw):
        calls.append(list(cmd))
        if "create" in cmd:
            return mock.Mock(returncode=1, stdout="", stderr="QUOTA_EXCEEDED")
        return mock.Mock(returncode=0, stdout="", stderr="")

    with mock.patch.object(provisioning.subprocess, "run", fake_run):
        with pytest.raises(provisioning.CloudCliError):
            gcp.run(app, fn, [], {})

    assert not any("delete" in c for c in calls), "deleted a VM that never existed"
    out = capsys.readouterr().out
    assert "was never created" in out
    assert "still be billing" not in out, "phantom billing-leak warning"
