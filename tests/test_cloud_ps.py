"""Provider-neutral job rows from AWS/GCP listing commands."""

import json
from unittest import mock

import pytest

from runplz.backends import aws, gcp


def test_aws_list_jobs_normalizes_tagged_instances(monkeypatch):
    response = {
        "Reservations": [
            {
                "Instances": [
                    {
                        "InstanceId": "i-1",
                        "LaunchTime": "2026-01-01T00:00:00Z",
                        "State": {"Name": "running"},
                        "Tags": [{"Key": "Name", "Value": "runplz-app-fn-x"}],
                    }
                ]
            }
        ]
    }
    with mock.patch.object(
        aws.subprocess,
        "run",
        return_value=mock.Mock(returncode=0, stdout=json.dumps(response), stderr=""),
    ) as run:
        rows = aws.list_jobs(region="us-east-1")
    assert rows[0]["backend"] == "aws"
    assert rows[0]["status"] == "running"
    assert "Name=tag:runplz,Values=1" in run.call_args.args[0]


def test_gcp_list_jobs_normalizes_labelled_instances():
    response = [
        {
            "name": "runplz-app-fn-x",
            "creationTimestamp": "2026-01-01T00:00:00Z",
            "status": "RUNNING",
        }
    ]
    with mock.patch.object(
        gcp.subprocess,
        "run",
        return_value=mock.Mock(returncode=0, stdout=json.dumps(response), stderr=""),
    ) as run:
        rows = gcp.list_jobs(project="p", zone="z")
    assert rows[0]["backend"] == "gcp"
    assert rows[0]["status"] == "RUNNING"
    assert "--filter=labels.runplz=1" in run.call_args.args[0]


@pytest.mark.parametrize(
    "backend, kwargs, message",
    [
        (aws, {}, "AWS region is required"),
        (gcp, {}, "GCP project is required"),
    ],
)
def test_cloud_list_jobs_requires_scope(backend, kwargs, message, monkeypatch):
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("GCLOUD_PROJECT", raising=False)
    with pytest.raises(RuntimeError, match=message):
        backend.list_jobs(**kwargs)


@pytest.mark.parametrize("backend, kwargs", [(aws, {"region": "r"}), (gcp, {"project": "p"})])
def test_cloud_list_jobs_rejects_provider_failures(backend, kwargs):
    with mock.patch.object(
        backend.subprocess, "run", return_value=mock.Mock(returncode=1, stdout="", stderr="denied")
    ):
        with pytest.raises(RuntimeError, match="failed"):
            backend.list_jobs(**kwargs)


def test_aws_list_jobs_rejects_malformed_json():
    with mock.patch.object(
        aws.subprocess, "run", return_value=mock.Mock(returncode=0, stdout="oops", stderr="")
    ):
        with pytest.raises(RuntimeError, match="malformed JSON"):
            aws.list_jobs(region="r")
