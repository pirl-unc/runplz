"""Lifecycle edge cases against the stateful fake cloud control plane."""

import json
import subprocess

import fake_cloud


def _aws_create_args():
    return [
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
    ]


def test_terminating_an_already_terminated_instance_fails(sandbox_bin):
    log = fake_cloud.install(sandbox_bin, name="aws")
    create = subprocess.run(_aws_create_args(), capture_output=True, text=True)
    assert create.returncode == 0
    instance_id = json.loads(create.stdout)["Instances"][0]["InstanceId"]
    terminate = [
        "aws",
        "ec2",
        "terminate-instances",
        "--region",
        "us-east-1",
        "--instance-ids",
        instance_id,
        "--output",
        "json",
    ]
    assert subprocess.run(terminate, capture_output=True).returncode == 0
    assert subprocess.run(terminate, capture_output=True).returncode != 0
    assert len(fake_cloud.calls_matching(log, "terminate-instances")) == 2
