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
    result = subprocess.run(
        ["aws", "ec2", "describe-instances", "--region", "us-east-1", "--instance-ids", "i-x"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert len(fake_cloud.calls_matching(log, "describe-instances")) == 1


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
