"""Cloud-control-plane consistency failures through real CLI subprocesses."""

import subprocess

import fake_cloud


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
    assert len(fake_cloud.calls_matching(log, "describe-instances")) == 3


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
