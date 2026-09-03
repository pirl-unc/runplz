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
            "t3.micro",
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
        "t3.micro",
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
            "t3.micro",
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
        "t3.micro",
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
