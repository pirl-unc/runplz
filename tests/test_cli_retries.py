"""Retrying a vendor CLI (issue #81).

The attempt loop is shared; the classification is not. Brev's org/config
gaps, GCP's ZONE_RESOURCE_POOL_EXHAUSTED and AWS's throttling look nothing
like each other, so each backend brings its own tables to a common loop.
"""

import subprocess
from unittest import mock

import pytest

from runplz.backends import aws, brev, gcp
from runplz.backends import provisioning as prov


def _runner(outcomes):
    """A fake subprocess.run yielding (returncode, stderr) in order."""
    calls = {"n": 0}

    def fake_run(cmd, **kw):
        i = calls["n"]
        calls["n"] += 1
        rc, err = outcomes[min(i, len(outcomes) - 1)]
        return mock.Mock(returncode=rc, stdout="", stderr=err)

    return fake_run, calls


def _run(policy, outcomes, **kw):
    fake, calls = _runner(outcomes)
    with mock.patch.object(prov.subprocess, "run", fake):
        with mock.patch.object(prov.time, "sleep", lambda _s: None):
            result = prov.run_with_retries(
                ["vendor", "x"], label="x", timeout=5, policy=policy, **kw
            )
    return result, calls["n"]


# ---------------------------------------------------------------------------
# the loop


def test_no_policy_means_exactly_one_attempt():
    _r, attempts = _run(prov.NO_RETRIES, [(1, "boom")])
    assert attempts == 1


def test_success_short_circuits():
    _r, attempts = _run(prov.GCP_RETRY_POLICY, [(0, "")])
    assert attempts == 1


def test_transient_is_retried_until_it_clears():
    result, attempts = _run(prov.GCP_RETRY_POLICY, [(1, "Error 503: backendError"), (0, "")])
    assert attempts == 2
    assert result.returncode == 0


def test_transient_gives_up_after_the_budget():
    result, attempts = _run(prov.GCP_RETRY_POLICY, [(1, "backendError")])
    assert attempts == prov.GCP_RETRY_POLICY.attempts
    assert result.returncode == 1


def test_unclassified_failure_is_not_retried():
    """Retrying an error we don't understand is a guess at the user's expense."""
    _r, attempts = _run(prov.GCP_RETRY_POLICY, [(1, "something nobody has seen")])
    assert attempts == 1


def test_all_attempts_timing_out_raises():
    with mock.patch.object(prov.subprocess, "run", side_effect=subprocess.TimeoutExpired("x", 5)):
        with mock.patch.object(prov.time, "sleep", lambda _s: None):
            with pytest.raises(prov.CloudCliError, match="on all 3 attempts"):
                prov.run_with_retries(
                    ["vendor", "x"], label="x", timeout=5, policy=prov.GCP_RETRY_POLICY
                )


def test_a_timeout_that_clears_is_retried():
    seq = [subprocess.TimeoutExpired("x", 5), mock.Mock(returncode=0, stdout="", stderr="")]
    with mock.patch.object(prov.subprocess, "run", side_effect=seq):
        with mock.patch.object(prov.time, "sleep", lambda _s: None):
            r = prov.run_with_retries(
                ["vendor", "x"], label="x", timeout=5, policy=prov.GCP_RETRY_POLICY
            )
    assert r.returncode == 0


def test_waits_are_honoured_between_attempts():
    slept = []
    fake, _calls = _runner([(1, "backendError")])
    with mock.patch.object(prov.subprocess, "run", fake):
        prov.run_with_retries(
            ["vendor", "x"],
            label="x",
            timeout=5,
            policy=prov.GCP_RETRY_POLICY,
            sleep=slept.append,
        )
    assert slept == [w for w in prov.GCP_RETRY_POLICY.waits if w]


def test_sleep_is_resolved_per_call_not_bound_at_import():
    """A bound default would ignore a patched time.sleep and really sleep."""
    fake, _calls = _runner([(1, "backendError")])
    naps = []
    with mock.patch.object(prov.subprocess, "run", fake):
        with mock.patch.object(prov.time, "sleep", naps.append):
            prov.run_with_retries(
                ["vendor", "x"], label="x", timeout=5, policy=prov.GCP_RETRY_POLICY
            )
    assert naps, "patched time.sleep was ignored — the suite would really sleep"


# ---------------------------------------------------------------------------
# per-provider classification


@pytest.mark.parametrize(
    "policy,transient,final",
    [
        (prov.GCP_RETRY_POLICY, "Error 503: backendError", "QUOTA_EXCEEDED"),
        (prov.GCP_RETRY_POLICY, "rate limit exceeded", "ZONE_RESOURCE_POOL_EXHAUSTED"),
        (prov.AWS_RETRY_POLICY, "RequestLimitExceeded", "InsufficientInstanceCapacity"),
        (prov.AWS_RETRY_POLICY, "Throttling", "UnauthorizedOperation"),
    ],
)
def test_each_provider_classifies_its_own_errors(policy, transient, final):
    assert policy.transient(transient)
    assert not policy.non_retriable(transient)
    assert policy.non_retriable(final)
    assert not policy.transient(final)


def test_providers_do_not_borrow_each_others_vocabulary():
    """A shared table would retry GCP on an AWS string and vice versa."""
    assert not prov.GCP_RETRY_POLICY.non_retriable("InsufficientInstanceCapacity")
    assert not prov.AWS_RETRY_POLICY.non_retriable("ZONE_RESOURCE_POOL_EXHAUSTED")


def test_brev_keeps_its_own_classification():
    assert brev.BREV_RETRY_POLICY.is_transient is brev._looks_transient
    assert brev.BREV_RETRY_POLICY.is_non_retriable is brev._looks_non_retriable
    assert brev.BREV_RETRY_POLICY.waits == brev._BREV_DEFAULT_RETRIES


# ---------------------------------------------------------------------------
# the drivers actually ask for retries


def _capture_policies():
    """Record the `policy=` each run_cli call asks for."""
    seen = []

    def fake_run_cli(cmd, **kw):
        seen.append(kw.get("policy"))
        return mock.Mock(returncode=0, stdout="{}", stderr="")

    return seen, fake_run_cli


def test_gcp_teardown_retries_briefly():
    """A teardown that gives up on a blip leaks a box — but long backoff in
    a finally widens the window where a Ctrl-C abandons the delete."""
    from runplz.config import GcpConfig

    seen, fake = _capture_policies()
    with mock.patch.object(gcp, "run_cli", fake):
        gcp.apply_on_finish(GcpConfig(project="p", zone="z"), "box-1")
    assert prov.GCP_TEARDOWN_POLICY in seen
    assert prov.GCP_TEARDOWN_POLICY.deadline_s < prov.GCP_RETRY_POLICY.deadline_s


def test_aws_teardown_retries_briefly():
    from runplz.config import AwsConfig

    seen, fake = _capture_policies()
    with mock.patch.object(aws, "run_cli", fake):
        aws.apply_on_finish(AwsConfig(region="us-east-1", key_name="k"), "i-123", name="b")
    assert prov.AWS_TEARDOWN_POLICY in seen


def test_the_create_paths_ask_for_retries(tmp_path):
    """The riskiest wiring in the PR, and what the issue actually asked for."""
    from runplz.config import AwsConfig, GcpConfig

    class App:
        name = "vision"
        gcp_config = GcpConfig(project="p", zone="us-central1-a")
        aws_config = AwsConfig(region="us-east-1", key_name="k")
        _repo_root = tmp_path

    class Fn:
        name = "train"
        gpu = "T4"
        min_gpus = 1
        min_cpu = None
        min_memory = None
        min_gpu_memory = None
        min_disk = None

    for module, expected in ((gcp, prov.GCP_RETRY_POLICY), (aws, prov.AWS_RETRY_POLICY)):
        seen, fake = _capture_policies()
        with mock.patch.object(module, "run_cli", fake):
            with mock.patch.object(module, "run_on_provisioned_vm", lambda **kw: kw["provision"]()):
                try:
                    module.run(App(), Fn(), [], {})
                except Exception:  # noqa: BLE001 - the fakes stop short of a real run
                    pass
        assert expected in seen, f"{module.__name__} create is not retried"


def test_run_cli_single_attempt_output_is_unchanged(capsys):
    """No attempt narration for a one-shot call — that would be new noise."""
    with mock.patch.object(prov.subprocess, "run") as run_mock:
        run_mock.return_value = mock.Mock(returncode=0, stdout="", stderr="")
        prov.run_cli(["gcloud", "x"], label="gcloud x")
    out = capsys.readouterr().out
    assert out.startswith("+ gcloud x")
    assert "attempt" not in out


def test_run_cli_with_a_policy_narrates_attempts(capsys):
    fake, _calls = _runner([(1, "backendError"), (0, "")])
    with mock.patch.object(prov.subprocess, "run", fake):
        with mock.patch.object(prov.time, "sleep", lambda _s: None):
            prov.run_cli(["gcloud", "x"], label="gcloud x", policy=prov.GCP_RETRY_POLICY)
    out = capsys.readouterr().out
    assert "attempt 1/3" in out
    assert "succeeded on attempt 2" in out


# ---------------------------------------------------------------------------
# retrying a create must not create twice


def test_aws_create_is_idempotent():
    """Retrying a launch whose response was lost must not start a second
    instance — nothing would ever tear the first one down."""
    from runplz.config import AwsConfig

    class Fn:
        gpu = None
        min_gpus = 1
        min_cpu = None
        min_memory = None
        min_gpu_memory = None
        min_disk = None

    cmd = aws.build_run_instances_command(
        AwsConfig(region="us-east-1", key_name="k"),
        Fn(),
        name="runplz-app-fn-ab12cd34",
        instance_type="m6i.large",
        ami="ami-1",
    )
    assert "--client-token" in cmd
    assert cmd[cmd.index("--client-token") + 1] == "runplz-app-fn-ab12cd34"


def test_gcp_treats_an_already_existing_box_as_created(tmp_path, capsys):
    """A retried create whose first attempt landed: the box exists and bills,
    so teardown must run rather than report 'was never created'."""
    from runplz.config import GcpConfig

    class App:
        name = "vision"
        gcp_config = GcpConfig(project="p", zone="us-central1-a")
        _repo_root = tmp_path

    class Fn:
        name = "train"
        gpu = "T4"
        min_gpus = 1
        min_cpu = None
        min_memory = None
        min_gpu_memory = None
        min_disk = None

    torn_down = []

    def fake_run_cli(cmd, **kw):
        if "create" in cmd:
            raise prov.CloudCliError("The resource 'projects/p/...' already exists")
        return mock.Mock(returncode=0, stdout="", stderr="")

    def fake_lifecycle(**kw):
        kw["provision"]()
        kw["teardown"]()

    with mock.patch.object(gcp, "run_cli", fake_run_cli):
        with mock.patch.object(gcp, "apply_on_finish", lambda cfg, n: torn_down.append(n)):
            with mock.patch.object(gcp, "run_on_provisioned_vm", fake_lifecycle):
                gcp.run(App(), Fn(), [], {})

    assert torn_down, "an existing, billing box was left to run"
    out = capsys.readouterr().out
    assert "already exists" in out
    assert "was never created" not in out


# ---------------------------------------------------------------------------
# loop guards


def test_non_retriable_stops_after_exactly_one_attempt():
    _r, attempts = _run(prov.GCP_RETRY_POLICY, [(1, "QUOTA_EXCEEDED")])
    assert attempts == 1


def test_a_spent_deadline_stops_further_attempts():
    import dataclasses

    policy = dataclasses.replace(prov.GCP_RETRY_POLICY, deadline_s=0)
    _r, attempts = _run(policy, [(1, "Internal error. Please try again")])
    assert attempts == 1, "the overall budget was ignored"


def test_a_policy_with_no_attempts_is_refused():
    """An empty schedule used to fall through to a bare assert."""
    import dataclasses

    with pytest.raises(prov.CloudCliError, match="at least one"):
        prov.run_with_retries(
            ["vendor", "x"],
            label="x",
            timeout=5,
            policy=dataclasses.replace(prov.NO_RETRIES, waits=()),
        )


def test_timeouts_can_be_declared_non_idempotent():
    import dataclasses

    policy = dataclasses.replace(prov.GCP_RETRY_POLICY, retry_timeouts=False)
    with mock.patch.object(prov.subprocess, "run", side_effect=subprocess.TimeoutExpired("x", 5)):
        with mock.patch.object(prov.time, "sleep", lambda _s: None):
            with pytest.raises(prov.CloudCliError):
                prov.run_with_retries(["vendor", "x"], label="x", timeout=5, policy=policy)


def test_run_cli_keeps_the_argv_in_the_log_even_when_retrying(capsys):
    """The requested machine type belongs in the log of a *successful* run,
    not only in the error message of a failed one."""
    fake, _calls = _runner([(0, "")])
    with mock.patch.object(prov.subprocess, "run", fake):
        prov.run_cli(
            ["gcloud", "compute", "instances", "create", "box", "--machine-type=a2-highgpu-8g"],
            label="create box",
            policy=prov.GCP_RETRY_POLICY,
        )
    assert "--machine-type=a2-highgpu-8g" in capsys.readouterr().out
