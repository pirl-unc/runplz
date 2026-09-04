"""Cloud-control-plane consistency failures through real CLI subprocesses."""

import subprocess
from pathlib import Path

import fake_cloud
import pytest


def test_command_observation_contract_rejects_silent_paths(tmp_path):
    """The fidelity helper itself must fail when a path never invoked the CLI."""
    log = tmp_path / "aws-calls.jsonl"
    log.write_text('{"argv": ["ec2", "describe-instances"]}\n')
    with pytest.raises(AssertionError, match="observed no command"):
        fake_cloud.assert_observed(log, "terminate-instances")


def test_describe_eventual_consistency_is_scriptable(sandbox_bin):
    log = fake_cloud.install(
        sandbox_bin,
        name="aws",
        fail_times={"ec2/describe-instances": 2},
        fail_message="InvalidInstanceID.NotFound",
    )
    create = subprocess.run(
        [
            "aws",
            "ec2",
            "run-instances",
            "--region",
            "us-east-1",
            "--image-id",
            "ami-x",
            "--instance-type",
            "m6i.large",
            "--key-name",
            "k",
            "--count",
            "1",
            "--client-token",
            "seed",
            "--tag-specifications",
            "x",
            "--output",
            "json",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    instance_id = "i-fake0000000001"
    assert instance_id in create.stdout
    command = [
        "aws",
        "ec2",
        "describe-instances",
        "--region",
        "us-east-1",
        "--instance-ids",
        instance_id,
        "--query",
        "Reservations",
        "--output",
        "text",
    ]
    first = subprocess.run(command, capture_output=True, text=True)
    second = subprocess.run(command, capture_output=True, text=True)
    third = subprocess.run(command, capture_output=True, text=True)
    assert first.returncode != 0
    assert second.returncode != 0
    assert third.returncode == 0
    assert third.stdout.strip() == "127.0.0.1"
    fake_cloud.assert_observed(log, "describe-instances", count=3)


def test_fake_describe_address_is_derived_from_requested_instance(sandbox_bin):
    log = fake_cloud.install(sandbox_bin, name="aws")
    common = [
        "aws",
        "ec2",
        "run-instances",
        "--region",
        "us-east-1",
        "--image-id",
        "ami-x",
        "--instance-type",
        "m6i.large",
        "--key-name",
        "k",
        "--count",
        "1",
        "--client-token",
    ]
    for token in ("one", "two"):
        subprocess.run(
            common + [token, "--tag-specifications", "x", "--output", "json"], check=True
        )
    for instance_id, expected in (
        ("i-fake0000000001", "127.0.0.1"),
        ("i-fake0000000002", "127.0.0.2"),
    ):
        result = subprocess.run(
            [
                "aws",
                "ec2",
                "describe-instances",
                "--region",
                "us-east-1",
                "--instance-ids",
                instance_id,
                "--query",
                "Reservations",
                "--output",
                "text",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        assert result.stdout.strip() == expected
    fake_cloud.assert_observed(log, "describe-instances", count=2)


def test_malformed_create_response_is_visible(sandbox_bin):
    fake_cloud.install(
        sandbox_bin,
        name="aws",
        malformed={"ec2/run-instances": "{not-json"},
    )
    result = subprocess.run(
        [
            "aws",
            "ec2",
            "run-instances",
            "--region",
            "us-east-1",
            "--image-id",
            "ami-x",
            "--instance-type",
            "m6i.large",
            "--key-name",
            "k",
            "--count",
            "1",
            "--client-token",
            "token",
            "--tag-specifications",
            "x",
            "--output",
            "json",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert result.stdout == "{not-json"


def test_aws_create_is_idempotent_and_distinguishes_tokens(sandbox_bin):
    log = fake_cloud.install(sandbox_bin, name="aws")
    base = [
        "aws",
        "ec2",
        "run-instances",
        "--region",
        "us-east-1",
        "--image-id",
        "ami-x",
        "--instance-type",
        "m6i.large",
        "--key-name",
        "k",
        "--count",
        "1",
        "--tag-specifications",
        "x",
        "--output",
        "json",
    ]

    def create(token):
        result = subprocess.run(
            [*base, "--client-token", token], capture_output=True, text=True, check=True
        )
        return result.stdout

    first = create("same-token")
    retry = create("same-token")
    distinct = create("new-token")

    assert first == retry
    assert first != distinct
    assert len(fake_cloud.calls_matching(log, "run-instances")) == 3


def test_fake_aws_rejects_unknown_options_instead_of_accepting_any_argv(sandbox_bin):
    log = fake_cloud.install(sandbox_bin, name="aws")
    result = subprocess.run(
        [
            "aws",
            "ec2",
            "describe-instances",
            "--region",
            "us-east-1",
            "--instance-ids",
            "i-fake",
            "--query",
            "Reservations",
            "--output",
            "text",
            "--definitely-not-an-aws-option",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "unknown option" in result.stderr
    assert len(fake_cloud.calls_matching(log, "describe-instances")) == 1


def test_fake_aws_rejects_missing_required_options(sandbox_bin):
    log = fake_cloud.install(sandbox_bin, name="aws")
    result = subprocess.run(
        ["aws", "ec2", "run-instances", "--region", "us-east-1"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "missing required option" in result.stderr
    assert len(fake_cloud.calls_matching(log, "run-instances")) == 1


# ---------------------------------------------------------------------------
# the stub's vocabulary is per route, not one set shared by both CLIs (#96)


@pytest.mark.parametrize(
    "name, argv, rejected",
    [
        # `--instance-type` is EC2's; gcloud sizes with `--machine-type`.
        (
            "gcloud",
            [
                "gcloud",
                "compute",
                "instances",
                "create",
                "x",
                "--project=p",
                "--zone=z",
                "--machine-type=m",
                "--image-family=f",
                "--image-project=i",
                "--instance-type=t3.micro",
            ],
            "--instance-type",
        ),
        # `--zone` is gcloud's; EC2 has regions and availability zones in the
        # subnet, never a bare --zone.
        (
            "aws",
            [
                "aws",
                "ec2",
                "run-instances",
                "--region",
                "r",
                "--image-id",
                "a",
                "--instance-type",
                "t",
                "--key-name",
                "k",
                "--count",
                "1",
                "--client-token",
                "c",
                "--tag-specifications",
                "x",
                "--output",
                "json",
                "--zone",
                "z",
            ],
            "--zone",
        ),
    ],
)
def test_each_stub_rejects_the_other_cli_s_flags(sandbox_bin, name, argv, rejected):
    """A single flat vocabulary accepted both, so a production command that
    reached for the wrong provider's flag was validated by nobody."""
    fake_cloud.install(sandbox_bin, name=name)
    result = subprocess.run(argv, capture_output=True, text=True)
    assert result.returncode == 2
    assert f"unknown option(s): {rejected}" in result.stderr


def test_instances_list_takes_zones_not_zone(sandbox_bin):
    """The confusion this file's own comment warned about while accepting it:
    `instances list` narrows with `--zones`, and `--zone` is a different flag
    belonging to a different subcommand."""
    fake_cloud.install(sandbox_bin, name="gcloud")
    base = [
        "gcloud",
        "compute",
        "instances",
        "list",
        "--project=p",
        "--filter=labels.runplz=1",
        "--format=json",
    ]

    typo = subprocess.run(base + ["--zone=z"], capture_output=True, text=True)
    assert typo.returncode == 2
    assert "unknown option(s): --zone" in typo.stderr

    correct = subprocess.run(base + ["--zones=z"], capture_output=True, text=True)
    assert correct.returncode == 0, correct.stderr


def test_a_required_option_is_never_also_an_unknown_one(sandbox_bin):
    """Required and allowed are separate tables merged at install time. If they
    drifted, the stub would demand an option and then reject it — refusing the
    exact argv it asks for."""
    fake_cloud.install(sandbox_bin, name="aws")
    result = subprocess.run(
        [
            "aws",
            "ec2",
            "describe-instances",
            "--region",
            "r",
            "--instance-ids",
            "i-1",
            "--query",
            "Q",
            "--output",
            "text",
        ],
        capture_output=True,
        text=True,
    )
    assert "unknown option" not in result.stderr


def test_a_rejected_command_line_leaves_no_state_behind(sandbox_bin):
    """Validation used to run *after* the stateful mutation, so an argv the
    stub refused had already allocated a client token and an instance entry —
    seeding the next call's state from a command that supposedly never ran."""
    log = fake_cloud.install(sandbox_bin, name="aws")
    rejected = subprocess.run(
        ["aws", "ec2", "run-instances", "--region", "us-east-1"],
        capture_output=True,
        text=True,
    )
    assert rejected.returncode == 2
    assert "missing required option" in rejected.stderr

    state = Path(str(log) + ".state")
    tokens = Path(str(log) + ".tokens")
    assert not state.exists() or state.read_text() == "{}", state.read_text()
    assert not tokens.exists() or tokens.read_text() == "{}", tokens.read_text()


# ---------------------------------------------------------------------------
# the stub validates option *values*, not just option names (#154)
#
# Per-route vocabularies (#140) made the stub reject the wrong provider's
# flags, but it still took anything at all inside a flag it recognised. That
# left the first of the two failures this harness was built for -- invented
# machine types -- outside what it could catch, which is what its own module
# docstring claims it exists for.


def _create_argv(**overrides) -> list:
    """A `gcloud compute instances create` runplz would actually emit."""
    options = {
        "--project": "p",
        "--zone": "us-central1-a",
        "--machine-type": "a2-ultragpu-8g",
        "--image-family": "f",
        "--image-project": "i",
        "--format": "json",
        **overrides,
    }
    return ["gcloud", "compute", "instances", "create", "runplz-x"] + [
        f"{name}={value}" for name, value in options.items()
    ]


def _run_instances_argv(**overrides) -> list:
    options = {
        "--region": "us-east-1",
        "--image-id": "ami-0fake00000000000",
        "--instance-type": "p4d.24xlarge",
        "--key-name": "k",
        "--count": "1",
        "--client-token": "tok",
        "--tag-specifications": "x",
        "--output": "json",
        **overrides,
    }
    argv = ["aws", "ec2", "run-instances"]
    for name, value in options.items():
        argv += [name, value]
    return argv


def test_an_invented_machine_type_is_rejected(sandbox_bin):
    """#154's first example. runplz picks machine types from its own
    catalogue, and the whole point of the executable tier is that a name which
    drifts out of the provider's real vocabulary fails a test rather than a
    paid run."""
    fake_cloud.install(sandbox_bin, name="gcloud")
    result = subprocess.run(
        _create_argv(**{"--machine-type": "totally-invented-9000"}),
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "--machine-type" in result.stderr, result.stderr


def test_an_invented_instance_type_is_rejected(sandbox_bin):
    fake_cloud.install(sandbox_bin, name="aws")
    result = subprocess.run(
        _run_instances_argv(**{"--instance-type": "p99d.9000xlarge"}),
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "--instance-type" in result.stderr, result.stderr


def test_a_non_numeric_count_is_rejected(sandbox_bin):
    """#154's second example. `--count abc` exited 0."""
    fake_cloud.install(sandbox_bin, name="aws")
    result = subprocess.run(
        _run_instances_argv(**{"--count": "abc"}), capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "--count" in result.stderr, result.stderr


@pytest.mark.parametrize(
    "option, value",
    [
        ("--provisioning-model", "CHEAP"),
        ("--maintenance-policy", "SOMETIMES"),
        ("--boot-disk-size", "loads"),
        ("--boot-disk-type", "pd-imaginary"),
        ("--format", "interpretive-dance"),
    ],
)
def test_enumerated_and_numeric_gcloud_options_reject_invented_values(sandbox_bin, option, value):
    fake_cloud.install(sandbox_bin, name="gcloud")
    result = subprocess.run(_create_argv(**{option: value}), capture_output=True, text=True)
    assert result.returncode != 0
    assert option in result.stderr, result.stderr


def test_an_invented_accelerator_is_rejected(sandbox_bin):
    """The accelerator name comes from the same catalogue the machine type
    does, and drifts the same way."""
    fake_cloud.install(sandbox_bin, name="gcloud")
    result = subprocess.run(
        _create_argv(**{"--accelerator": "type=nvidia-imaginary-9000,count=8"}),
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "--accelerator" in result.stderr, result.stderr


def test_a_malformed_accelerator_is_rejected(sandbox_bin):
    """gcloud's shape is `type=<name>,count=<n>`; a bare name is not it."""
    fake_cloud.install(sandbox_bin, name="gcloud")
    result = subprocess.run(
        _create_argv(**{"--accelerator": "nvidia-tesla-a100"}),
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "--accelerator" in result.stderr, result.stderr


def test_the_argv_runplz_actually_generates_still_passes(sandbox_bin):
    """The other half of the contract. Validation that rejected real commands
    would be worse than validation that accepts invented ones, because it
    would fail on the paths the e2e tier exists to prove."""
    fake_cloud.install(sandbox_bin, name="gcloud")
    created = subprocess.run(
        _create_argv(
            **{
                "--accelerator": "type=nvidia-a100-80gb,count=8",
                "--maintenance-policy": "TERMINATE",
                "--boot-disk-size": "500GB",
                "--provisioning-model": "SPOT",
                "--labels": "runplz=1",
            }
        )
        + ["--quiet", "--no-restart-on-failure"],
        capture_output=True,
        text=True,
    )
    assert created.returncode == 0, created.stderr

    fake_cloud.install(sandbox_bin, name="aws")
    launched = subprocess.run(_run_instances_argv(), capture_output=True, text=True)
    assert launched.returncode == 0, launched.stderr


def test_a_value_rejection_leaves_no_state_behind(sandbox_bin):
    """Same requirement the missing-option check already meets: an argv the
    stub refuses must not have allocated a client token first."""
    log = fake_cloud.install(sandbox_bin, name="aws")
    rejected = subprocess.run(
        _run_instances_argv(**{"--count": "abc"}), capture_output=True, text=True
    )
    assert rejected.returncode == 2

    state = Path(str(log) + ".state")
    tokens = Path(str(log) + ".tokens")
    assert not state.exists() or state.read_text() == "{}", state.read_text()
    assert not tokens.exists() or tokens.read_text() == "{}", tokens.read_text()


def test_the_vocabulary_tracks_the_live_catalogue(sandbox_bin):
    """Declared from `provisioning`, not copied beside it. A shape removed
    upstream and dropped from the table must stop being accepted here, which a
    hand-maintained duplicate would not do."""
    from runplz.backends.provisioning import AWS_CPU_SHAPES

    accepted = fake_cloud._values_for("aws")["ec2/run-instances"]["--instance-type"]
    assert set(offering.name for offering in AWS_CPU_SHAPES) <= set(accepted["choices"])


@pytest.mark.parametrize(
    "option, value",
    [
        ("--image-id", "my-favourite-ami"),
        ("--image-id", ""),
        ("--instance-ids", "arn:aws:ec2:us-east-1:1:instance/i-1"),
    ],
)
def test_ec2_resource_ids_must_look_like_ec2_resource_ids(sandbox_bin, option, value):
    """A lookup that returns a name, an ARN, or nothing at all is a real bug
    shape -- `ssm get-parameter` resolves the AMI, and an empty result would
    otherwise sail through as a launch with no image."""
    fake_cloud.install(sandbox_bin, name="aws")
    argv = (
        _run_instances_argv(**{option: value})
        if option == "--image-id"
        else [
            "aws",
            "ec2",
            "describe-instances",
            "--region",
            "us-east-1",
            "--instance-ids",
            value,
            "--query",
            "Reservations",
            "--output",
            "text",
        ]
    )
    result = subprocess.run(argv, capture_output=True, text=True)
    assert result.returncode != 0
    assert option in result.stderr, result.stderr
