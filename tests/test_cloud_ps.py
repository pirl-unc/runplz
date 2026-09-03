"""Provider-neutral job records, and the capability layer that guards them.

Two halves. The drivers still own parsing a provider's response into
`JobRecord`s. What they no longer own is deciding whether they may be called
at all — scope resolution moved up into the registry's `ListingSpec`, so the
tests for "aws needs a region" moved with it, and now assert the thing that
actually matters: that the check happens *before* the provider CLI is
spawned.
"""

import json
from unittest import mock

import pytest

from runplz.backends import aws, gcp, registry
from runplz.backends.listing import (
    JobRecord,
    ListingSpec,
    ListingUnsupported,
    MissingScope,
    ScopeField,
)

# ---------------------------------------------------------------------------
# drivers normalize their provider's response


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
    assert rows == [
        JobRecord(
            backend="aws",
            name="runplz-app-fn-x",
            app="app",
            function="fn",
            started="2026-01-01T00:00:00Z",
            status="running",
        )
    ]
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
    assert rows == [
        JobRecord(
            backend="gcp",
            name="runplz-app-fn-x",
            app="app",
            function="fn",
            started="2026-01-01T00:00:00Z",
            status="RUNNING",
        )
    ]
    assert "--filter=labels.runplz=1" in run.call_args.args[0]


@pytest.mark.parametrize("backend, kwargs", [(aws, {"region": "r"}), (gcp, {"project": "p"})])
def test_cloud_list_jobs_rejects_provider_failures(backend, kwargs):
    with mock.patch.object(
        backend.subprocess, "run", return_value=mock.Mock(returncode=1, stdout="", stderr="denied")
    ):
        with pytest.raises(RuntimeError, match="failed"):
            backend.list_jobs(**kwargs)


@pytest.mark.parametrize(
    "backend, kwargs", [(aws, {"region": "r"}), (gcp, {"project": "p", "zone": "z"})]
)
def test_cloud_list_jobs_rejects_malformed_json(backend, kwargs):
    with mock.patch.object(
        backend.subprocess, "run", return_value=mock.Mock(returncode=0, stdout="oops", stderr="")
    ):
        with pytest.raises(RuntimeError, match="malformed JSON"):
            backend.list_jobs(**kwargs)


# ---------------------------------------------------------------------------
# the capability layer decides whether a driver may be called at all


@pytest.fixture
def no_cloud_env(monkeypatch):
    for var in (
        "AWS_DEFAULT_REGION",
        "AWS_REGION",
        "GOOGLE_CLOUD_PROJECT",
        "GCLOUD_PROJECT",
        "CLOUDSDK_COMPUTE_ZONE",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.mark.parametrize(
    "backend, flag, message",
    [
        ("aws", "--region", "aws region is required"),
        ("gcp", "--project", "gcp project is required"),
    ],
)
def test_missing_scope_is_refused_before_the_provider_cli_runs(
    backend, flag, message, no_cloud_env
):
    """The point of moving this check up a layer.

    It used to live inside the driver, which meant the driver had already
    been imported and entered before anyone asked whether it could answer.
    Now nothing below the registry is reached — asserted by watching
    `subprocess.run`, not just by the exception type, since an exception
    raised *after* a wasted API call would look identical from the outside.
    """
    module = registry.load(backend)
    with mock.patch.object(module.subprocess, "run") as run:
        with pytest.raises(MissingScope, match=message):
            registry.list_jobs(backend)
    run.assert_not_called()


@pytest.mark.parametrize(
    "backend, env_var, kwarg",
    [
        ("aws", "AWS_DEFAULT_REGION", "region"),
        ("aws", "AWS_REGION", "region"),
        ("gcp", "GOOGLE_CLOUD_PROJECT", "project"),
        ("gcp", "GCLOUD_PROJECT", "project"),
    ],
)
def test_each_declared_env_var_satisfies_required_scope(
    backend, env_var, kwarg, monkeypatch, no_cloud_env
):
    """Every fallback in the chain, not just the first.

    `AWS_REGION` and `GCLOUD_PROJECT` are the second entries and were
    reachable only when the first was unset, so a chain that stopped early
    would still pass a single-variable test.
    """
    monkeypatch.setenv(env_var, "from-env")
    with mock.patch.object(registry.load(backend), "list_jobs", return_value=[]) as list_jobs:
        registry.list_jobs(backend)
    assert list_jobs.call_args.kwargs[kwarg] == "from-env"


def test_an_explicit_value_beats_the_environment(monkeypatch):
    monkeypatch.setenv("AWS_DEFAULT_REGION", "from-env")
    with mock.patch.object(aws, "list_jobs", return_value=[]) as list_jobs:
        registry.list_jobs("aws", region="explicit")
    assert list_jobs.call_args.kwargs["region"] == "explicit"


def test_an_empty_environment_variable_does_not_count_as_scope(monkeypatch, no_cloud_env):
    """`AWS_DEFAULT_REGION=` exports the name with no value. Treating that as
    a region sends `--region ''` to the provider instead of saying what is
    missing."""
    monkeypatch.setenv("AWS_DEFAULT_REGION", "")
    with pytest.raises(MissingScope, match="aws region is required"):
        registry.list_jobs("aws")


def test_optional_scope_is_passed_through_as_none_when_unset(no_cloud_env, monkeypatch):
    """Unset optional scope reaches the driver as None rather than being
    dropped, so a driver sees one call shape instead of two."""
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "p")
    with mock.patch.object(gcp, "list_jobs", return_value=[]) as list_jobs:
        registry.list_jobs("gcp")
    assert list_jobs.call_args.kwargs == {"project": "p", "zone": None}


def test_a_backend_that_cannot_list_says_so_rather_than_returning_empty(monkeypatch):
    """The distinction the issue is about: "I can't tell you" is not "none".

    No shipped backend declares `listing=None`, so this drives the contract
    through a synthetic entry — the guarantee is for whichever backend is
    added next, and an empty list would silently tell the user their jobs
    are gone.
    """
    spec = registry.BackendSpec(name="fictional", module="runplz.backends.local", listing=None)
    monkeypatch.setitem(registry.BACKENDS, "fictional", spec)
    with pytest.raises(ListingUnsupported, match="cannot list jobs"):
        registry.list_jobs("fictional")
    assert "fictional" not in registry.listable_names()
    assert "fictional" not in registry.ps_names()


def test_provider_errors_reach_the_caller_unwrapped():
    """`runplz ps` prints the exception's class name, so re-raising a provider
    failure as something else would relabel every warning line."""
    boom = RuntimeError("aws describe-instances failed (rc=255)")
    with mock.patch.object(aws, "list_jobs", side_effect=boom):
        with pytest.raises(RuntimeError) as caught:
            registry.list_jobs("aws", region="r")
    assert caught.value is boom


# ---------------------------------------------------------------------------
# the declarations themselves


def test_scope_fields_are_deduplicated_and_attributed_to_their_backends():
    by_flag = {field.flag: backends for field, backends in registry.scope_fields()}
    assert by_flag["--region"] == ("aws",)
    assert by_flag["--project"] == ("gcp",)
    assert by_flag["--host"] == ("ssh",)


def test_two_backends_sharing_a_flag_get_one_flag_and_both_names(monkeypatch):
    """The reason `scope_fields` deduplicates at all.

    No two shipped backends want `--region` yet, but the CLI must offer one
    flag rather than a duplicate argparse registration that fails at import,
    and its help has to say which backends read it — `[aws|second]`, so the
    user can tell what a flag reaches.
    """
    shared = registry.get("aws").listing.scope[0]
    second = registry.BackendSpec(
        name="second",
        module="runplz.backends.local",
        listing=ListingSpec(scope=(shared,)),
    )
    monkeypatch.setitem(registry.BACKENDS, "second", second)
    by_flag = {field.flag: backends for field, backends in registry.scope_fields()}
    assert by_flag["--region"] == ("aws", "second")
    assert sum(1 for field, _ in registry.scope_fields() if field.flag == "--region") == 1


def test_scope_fields_refuses_two_backends_that_disagree_about_a_flag(monkeypatch):
    """A shared flag feeds one keyword. Two backends spelling `--region`
    differently would mean one of them silently receives the other's value,
    so the registry refuses to build the CLI at all."""
    clashing = registry.BackendSpec(
        name="clashing",
        module="runplz.backends.local",
        listing=ListingSpec(
            scope=(ScopeField(name="area", flag="--region", help="A different --region."),)
        ),
    )
    monkeypatch.setitem(registry.BACKENDS, "clashing", clashing)
    with pytest.raises(ValueError, match=r"--region is declared differently"):
        registry.scope_fields()


def test_every_listable_backend_exposes_list_jobs_and_the_rest_do_not():
    """The registry's claim and the driver's surface must agree — a backend
    that declares a ListingSpec but has no `list_jobs` is a dispatch that
    fails at the worst moment."""
    for name in registry.names():
        if name == "modal":
            continue  # optional extra; may not be installed
        module = registry.load(name)
        declared = registry.get(name).listing is not None
        assert hasattr(module, "list_jobs") == declared, name


def test_ssh_is_listable_but_stays_out_of_the_default_fan_out():
    assert "ssh" in registry.listable_names()
    assert "ssh" not in registry.ps_names()
    assert set(registry.ps_names()) <= set(registry.listable_names())
