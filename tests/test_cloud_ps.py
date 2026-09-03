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


def test_scope_fields_refuses_one_keyword_claimed_under_two_flags(monkeypatch):
    """The other half of a field's identity.

    The CLI keys argparse's `dest` — and the scope dict — by `name` while
    keying the option by `flag`, so guarding only the flag lets two backends
    claim `region` under `--region` and `--dup-region`. argparse accepts that
    and then feeds whichever option was parsed last into both backends.
    """
    clashing = registry.BackendSpec(
        name="clashing",
        module="runplz.backends.local",
        listing=ListingSpec(
            scope=(ScopeField(name="region", flag="--dup-region", help="Another region."),)
        ),
    )
    monkeypatch.setitem(registry.BACKENDS, "clashing", clashing)
    with pytest.raises(ValueError, match=r"scope 'region' is declared differently"):
        registry.scope_fields()


def test_list_jobs_refuses_scope_the_backend_does_not_declare():
    """A dropped keyword becomes "aws region is required" — an error naming
    the right fix while hiding that a value was passed and thrown away."""
    with pytest.raises(TypeError, match="does not take regoin"):
        registry.list_jobs("aws", regoin="us-east-1")


def test_list_jobs_refuses_scope_for_a_backend_that_takes_none():
    with pytest.raises(TypeError, match="accepts no scope"):
        registry.list_jobs("local", host="somewhere")


def test_a_backend_kept_out_of_the_fan_out_stays_out_without_required_scope(monkeypatch):
    """`default_fan_out=False` has to mean it.

    `invited_by` asks "did the user's flags request this backend", and a
    backend with no required fields has no flag to request it with. Answering
    True by vacuous `all([])` returned it to the fan-out it opted out of —
    queried unprompted on every bare `runplz ps`.
    """
    from runplz import cli

    opted_out = registry.BackendSpec(
        name="optout",
        module="runplz.backends.local",
        listing=ListingSpec(default_fan_out=False),
    )
    monkeypatch.setitem(registry.BACKENDS, "optout", opted_out)
    assert "optout" not in registry.ps_names()
    assert "optout" not in cli._ps_selection(None, {})
    # Naming it positionally is still how you get it.
    assert cli._ps_selection("optout", {}) == ["optout"]


def test_the_unasked_note_names_a_command_when_there_is_no_flag_to_pass(monkeypatch, capsys):
    """Such a backend would otherwise be told to "pass  to include it"."""
    from runplz import cli

    opted_out = registry.BackendSpec(
        name="optout",
        module="runplz.backends.local",
        listing=ListingSpec(default_fan_out=False),
    )
    monkeypatch.setitem(registry.BACKENDS, "optout", opted_out)
    cli._note_backends_not_listed(list(registry.ps_names()))
    err = capsys.readouterr().err
    assert "note: optout was not listed; run `runplz ps optout` to include it" in err


def test_an_environment_value_is_parsed_by_the_field_type(monkeypatch):
    """Environment values arrive as strings however the field is declared, so
    a typed field has to apply its parser here or the driver gets "2222"."""
    monkeypatch.setenv("MYPORT", "2222")
    field = ScopeField(name="port", flag="--p", help="", type=int, env=("MYPORT",))
    assert field.resolve(None) == 2222


def test_a_multiple_field_is_split_however_it_was_supplied(monkeypatch):
    """Splitting lives with the field, not on the flag path, so a value that
    arrives from the environment is split the same way one typed as a flag
    is — otherwise `a.box,b.box` becomes a single host named `a.box,b.box`."""
    monkeypatch.setenv("MYHOSTS", "a.box, b.box")
    field = ScopeField(name="host", flag="--host", help="", multiple=True, env=("MYHOSTS",))
    assert field.resolve_all(None) == ["a.box", "b.box"]
    assert field.resolve_all("c.box,d.box") == ["c.box", "d.box"]


def test_the_port_constraint_travels_with_the_field_not_the_cli():
    """A check keyed off the literal string "port" in the CLI stops running
    the day the field is renamed. Declared here, it moves with it."""
    field = next(f for f in registry.get("ssh").listing.scope if f.flag == "--ssh-port")
    assert field.validate is not None
    field.validate(22)
    for bad in (0, 70000):
        with pytest.raises(ValueError, match="valid TCP port"):
            field.validate(bad)


def test_aws_rejects_valid_json_that_is_not_an_object():
    """`[]` and `null` parse fine and then die on `.get` with an
    AttributeError that escapes the malformed-JSON handler."""
    for payload in ("[]", "null"):
        with mock.patch.object(
            aws.subprocess, "run", return_value=mock.Mock(returncode=0, stdout=payload, stderr="")
        ):
            with pytest.raises(RuntimeError, match="malformed JSON"):
                aws.list_jobs(region="r")


def test_a_single_valued_field_resolves_to_exactly_one_target():
    """`resolve_all` is the one path the CLI fans out over, so a non-multiple
    field has to answer in the same shape — one target, not a split of it.
    A `--ssh-key` naming a path with a comma in it must stay one path."""
    field = ScopeField(name="ssh_key_path", flag="--ssh-key", help="")
    assert field.resolve_all("/keys/a,b.pem") == ["/keys/a,b.pem"]
    assert field.resolve_all(None) == []


def test_a_field_constraint_is_enforced_on_the_entry_point_not_just_the_flag():
    """`registry.list_jobs` is the public way in, and a value that arrives
    from the environment or from a direct call never passes an argparse
    check. The declared constraint has to hold on every path."""
    from runplz.backends import ssh
    from runplz.backends.listing import InvalidScope

    with mock.patch.object(ssh, "list_jobs") as list_jobs:
        with pytest.raises(InvalidScope, match=r"ssh port must be a valid TCP port"):
            registry.list_jobs("ssh", host="a.box", port=99999)
    list_jobs.assert_not_called()


def test_a_valid_value_passes_the_constraint_untouched():
    from runplz.backends import ssh

    with mock.patch.object(ssh, "list_jobs", return_value=[]) as list_jobs:
        registry.list_jobs("ssh", host="a.box", port=2200)
    assert list_jobs.call_args.kwargs["port"] == 2200
