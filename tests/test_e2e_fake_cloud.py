"""Cloud lifecycle against stub CLIs: real subprocess, real argv, no cloud.

Everything between `run()` and the CLI boundary used to be exercised only
through `mock.patch`, which means those tests assert what the author
believed `gcloud`/`aws` do. These run the actual code against actual
executables (see tests/fake_cloud.py) so argv, JSON parsing, exit-code
handling, retry classification and teardown are checked against something
that can disagree with the author.

The billing guard permits this because the stubs resolve inside the
`sandbox_bin` fixture's directory -- a real `gcloud` on PATH stays blocked.
"""

import subprocess
from pathlib import Path

import fake_cloud
import pytest

from runplz import App, AwsConfig, GcpConfig, Image
from runplz.backends import aws, gcp
from runplz.backends import ssh_common as sc


def _app(tmp_path, **cfg):
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "job.py").write_text("# job\n")
    app = App("vision", **cfg)

    @app.function(image=Image.from_registry("ubuntu:22.04"))
    def train():
        pass

    train.module_file = str(repo / "job.py")
    app.repo_root = repo
    return app, train


@pytest.fixture(autouse=True)
def _no_real_sleeping(fast_clock):
    """Every test here drives real provisioning code, which polls and backs off."""
    return fast_clock


def _run_until_dispatch_fails(module, app, fn):
    """Provision for real, let dispatch die, return without raising."""
    try:
        module.run(app, fn, [], {})
    except BaseException:
        pass


# ---------------------------------------------------------------------------
# GCP


def test_gcp_create_argv_is_what_gcloud_actually_receives(tmp_path, sandbox_bin):
    log = fake_cloud.install(sandbox_bin, name="gcloud")
    fake_cloud.install_unreachable_ssh(sandbox_bin)
    app, fn = _app(
        tmp_path,
        gcp_config=GcpConfig(
            project="proj", zone="us-central1-a", ssh_ready_wait_seconds=1, spot=True
        ),
    )
    _run_until_dispatch_fails(gcp, app, fn)

    creates = fake_cloud.calls_matching(log, "create")
    assert len(creates) == 1, fake_cloud.calls(log)
    argv = creates[0]
    assert "--project=proj" in argv
    assert "--zone=us-central1-a" in argv
    assert "--format=json" in argv
    assert "--labels=runplz=1" in argv, "runplz must label what it creates"
    assert "--provisioning-model=SPOT" in argv
    assert "--no-restart-on-failure" in argv, "a reclaimed spot box must stay down"


def test_gcp_tears_down_even_though_dispatch_failed(tmp_path, sandbox_bin):
    """The billing guarantee, against a real CLI instead of a mock."""
    log = fake_cloud.install(sandbox_bin, name="gcloud")
    fake_cloud.install_unreachable_ssh(sandbox_bin)
    app, fn = _app(
        tmp_path,
        gcp_config=GcpConfig(project="p", zone="z", ssh_ready_wait_seconds=1),
    )
    _run_until_dispatch_fails(gcp, app, fn)

    deletes = fake_cloud.calls_matching(log, "delete")
    assert len(deletes) == 1, f"instance was not deleted: {fake_cloud.calls(log)}"
    assert "--delete-disks=all" in deletes[0], "a leaked disk still bills"


@pytest.mark.parametrize(
    "on_finish, expected, forbidden",
    [("delete", "delete", "stop"), ("stop", "stop", "delete"), ("leave", None, "delete")],
)
def test_gcp_on_finish_modes(tmp_path, sandbox_bin, on_finish, expected, forbidden):
    log = fake_cloud.install(sandbox_bin, name="gcloud")
    fake_cloud.install_unreachable_ssh(sandbox_bin)
    app, fn = _app(
        tmp_path,
        gcp_config=GcpConfig(project="p", zone="z", on_finish=on_finish, ssh_ready_wait_seconds=1),
    )
    _run_until_dispatch_fails(gcp, app, fn)

    if expected:
        assert fake_cloud.calls_matching(log, expected)
    assert not fake_cloud.calls_matching(log, forbidden)


def test_gcp_retries_a_transient_create_failure(tmp_path, sandbox_bin):
    """The retry table is strings matched against real stderr, not a mock.

    An earlier version of these tables was written from memory and matched
    nothing gcloud actually prints.
    """
    log = fake_cloud.install(
        sandbox_bin,
        name="gcloud",
        fail_times={"compute/instances/create": 1},
        fail_message="Internal error. Please try again or contact Google Support.",
    )
    fake_cloud.install_unreachable_ssh(sandbox_bin)
    app, fn = _app(tmp_path, gcp_config=GcpConfig(project="p", zone="z", ssh_ready_wait_seconds=1))
    _run_until_dispatch_fails(gcp, app, fn)

    creates = fake_cloud.calls_matching(log, "create")
    assert len(creates) >= 2, f"transient failure was not retried: {creates}"
    assert fake_cloud.calls_matching(log, "delete"), "a retried create still needs teardown"


def test_gcp_does_not_retry_a_permanent_failure(tmp_path, sandbox_bin):
    log = fake_cloud.install(
        sandbox_bin,
        name="gcloud",
        fail_times={"compute/instances/create": 99},
        fail_message="Required 'compute.instances.create' permission for 'projects/p'",
    )
    fake_cloud.install_unreachable_ssh(sandbox_bin)
    app, fn = _app(tmp_path, gcp_config=GcpConfig(project="p", zone="z", ssh_ready_wait_seconds=1))
    _run_until_dispatch_fails(gcp, app, fn)

    creates = fake_cloud.calls_matching(log, "create")
    assert len(creates) == 1, f"a permissions error must not be retried: {creates}"


# ---------------------------------------------------------------------------
# AWS


def test_aws_run_instances_argv_and_teardown(tmp_path, sandbox_bin):
    log = fake_cloud.install(sandbox_bin, name="aws")
    fake_cloud.install_unreachable_ssh(sandbox_bin)
    app, fn = _app(
        tmp_path,
        aws_config=AwsConfig(region="us-east-1", key_name="mykey", ssh_ready_wait_seconds=1),
    )
    _run_until_dispatch_fails(aws, app, fn)

    launches = fake_cloud.calls_matching(log, "run-instances")
    assert len(launches) == 1, fake_cloud.calls(log)
    assert "us-east-1" in launches[0]
    assert "mykey" in launches[0], "key_name must reach the CLI"

    terminates = fake_cloud.calls_matching(log, "terminate-instances")
    assert terminates, f"instance was not terminated: {fake_cloud.calls(log)}"
    assert any(value.startswith("i-fake") for value in terminates[0]), (
        "must terminate the id run-instances returned"
    )


def test_aws_reads_the_instance_id_out_of_real_cli_json(tmp_path, sandbox_bin):
    """Parsing is against the CLI's actual output shape, not a mock's."""
    log = fake_cloud.install(sandbox_bin, name="aws")
    fake_cloud.install_unreachable_ssh(sandbox_bin)
    app, fn = _app(
        tmp_path,
        aws_config=AwsConfig(region="us-east-1", key_name="k", ssh_ready_wait_seconds=1),
    )
    _run_until_dispatch_fails(aws, app, fn)

    descriptions = fake_cloud.calls_matching(log, "describe-instances")
    assert descriptions, fake_cloud.calls(log)
    for argv in descriptions:
        assert any(value.startswith("i-fake") for value in argv)


def test_the_billing_guard_still_blocks_a_real_cli(tmp_path):
    """The sandbox exemption must be by resolved path, not by name.

    Without `sandbox_bin` there is no stub installed, so `gcloud` resolves
    to the real thing (or nothing) and the guard must refuse.
    """
    app, fn = _app(tmp_path, gcp_config=GcpConfig(project="p", zone="z"))
    with pytest.raises((RuntimeError, FileNotFoundError, Exception)) as excinfo:
        gcp.run(app, fn, [], {})
    assert "gcloud" in str(excinfo.value) or "brev" in str(excinfo.value)


def test_stub_is_the_binary_that_actually_ran(sandbox_bin):
    """Guards the harness itself: prove PATH resolution reaches the stub."""
    import shutil

    fake_cloud.install(sandbox_bin, name="gcloud")
    resolved = Path(shutil.which("gcloud")).resolve()
    assert resolved.parent == sandbox_bin, resolved


def test_stub_rejects_unknown_and_missing_cli_arguments(sandbox_bin):
    """The fake must fail on malformed commands, like the vendor CLI does."""
    fake_cloud.install(sandbox_bin, name="gcloud")

    unknown = subprocess.run(
        ["gcloud", "compute", "instances", "create", "--not-a-real-flag=x"],
        capture_output=True,
        text=True,
    )
    assert unknown.returncode != 0
    assert "unknown option" in unknown.stderr

    missing_name = subprocess.run(
        [
            "gcloud",
            "compute",
            "instances",
            "create",
            "--project=p",
            "--zone=z",
            "--machine-type=n2-standard-2",
            "--image-family=ubuntu",
            "--image-project=p",
        ],
        capture_output=True,
        text=True,
    )
    assert missing_name.returncode != 0
    assert "requires an instance name" in missing_name.stderr

    duplicate = subprocess.run(
        [
            "gcloud",
            "compute",
            "instances",
            "create",
            "fake",
            "--project=p",
            "--project=q",
            "--zone=z",
            "--machine-type=n2-standard-2",
            "--image-family=ubuntu",
            "--image-project=p",
        ],
        capture_output=True,
        text=True,
    )
    assert duplicate.returncode != 0
    assert "duplicate option" in duplicate.stderr


@pytest.mark.live_ssh
def test_fake_aws_control_plane_hands_off_to_live_ssh(
    tmp_path, sandbox_bin, sshd_server, monkeypatch
):
    """Provisioning output, SSH options, and a real endpoint must agree."""
    log = fake_cloud.install(sandbox_bin, name="aws")
    server = sshd_server
    user = (
        server.target.split("@", 1)[0] if "@" in server.target else __import__("getpass").getuser()
    )
    app, fn = _app(
        tmp_path,
        aws_config=AwsConfig(
            region="us-east-1",
            key_name="k",
            ssh_user=user,
            ssh_ready_wait_seconds=30,
        ),
    )

    original_lifecycle = aws.run_on_provisioned_vm

    def run_with_harness_options(**kwargs):
        kwargs["ssh_opts"] = server.ssh_options()
        return original_lifecycle(**kwargs)

    monkeypatch.setattr(aws, "run_on_provisioned_vm", run_with_harness_options)
    handed_off = []

    def real_ssh_dispatch(**kwargs):
        handed_off.append(kwargs)
        assert "HANDOFF_OK" in sc.ssh_capture(
            kwargs["target"], "echo HANDOFF_OK", ssh_opts=kwargs["ssh_opts"]
        )

    monkeypatch.setattr(sc, "dispatch_to_target", real_ssh_dispatch)
    aws.run(app, fn, [], {})

    assert len(handed_off) == 1
    assert handed_off[0]["target"].endswith("127.0.0.1")
    assert fake_cloud.calls_matching(log, "run-instances")
    assert fake_cloud.calls_matching(log, "terminate-instances")


@pytest.mark.live_ssh
def test_fake_gcp_control_plane_hands_off_to_live_ssh(
    tmp_path, sandbox_bin, sshd_server, monkeypatch
):
    """The gcloud config-ssh alias must resolve to the same live endpoint."""
    log = fake_cloud.install(sandbox_bin, name="gcloud")
    server = sshd_server
    user = (
        server.target.split("@", 1)[0] if "@" in server.target else __import__("getpass").getuser()
    )
    home = tmp_path / "home"
    config_path = home / ".ssh" / "config"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("RUNPLZ_FAKE_SSH_CONFIG", str(config_path))
    monkeypatch.setenv(
        "RUNPLZ_FAKE_SSH_HOST", server.host if hasattr(server, "host") else "127.0.0.1"
    )
    monkeypatch.setenv("RUNPLZ_FAKE_SSH_USER", user)
    monkeypatch.setenv("RUNPLZ_FAKE_SSH_PORT", str(server.port))
    monkeypatch.setenv("RUNPLZ_FAKE_SSH_IDENTITY", str(server.identity))
    # OpenSSH resolves the user's config from the account database rather than
    # trusting HOME, so point the production option list explicitly at the
    # throwaway config that the fake `config-ssh` command writes.
    monkeypatch.setattr(sc, "SSH_OPTS", ("-F", str(config_path)))
    app, fn = _app(
        tmp_path,
        gcp_config=GcpConfig(project="p", zone="z", ssh_ready_wait_seconds=30),
    )

    original_lifecycle = gcp.run_on_provisioned_vm

    def run_with_harness_options(**kwargs):
        kwargs["ssh_opts"] = server.ssh_options()
        return original_lifecycle(**kwargs)

    monkeypatch.setattr(gcp, "run_on_provisioned_vm", run_with_harness_options)
    handed_off = []

    def real_ssh_dispatch(**kwargs):
        handed_off.append(kwargs)
        assert "HANDOFF_OK" in sc.ssh_capture(
            kwargs["target"], "echo HANDOFF_OK", ssh_opts=kwargs["ssh_opts"]
        )

    monkeypatch.setattr(sc, "dispatch_to_target", real_ssh_dispatch)
    gcp.run(app, fn, [], {})

    assert len(handed_off) == 1
    assert config_path.exists()
    assert fake_cloud.calls_matching(log, "create")
    assert fake_cloud.calls_matching(log, "delete")


# ---------------------------------------------------------------------------
# listing (issue #135)
#
# Until 4.0.2 the stubs could not reach `list_jobs` at all: `--filter`,
# `--filters` and `--zones` were outside the option vocabulary, and
# `ec2 describe-instances` was modelled as one call when production makes two
# of it. So every listing test in the suite was `mock.patch`-based, asserting
# argv the author believed the CLIs accept. These assert argv they do.


def test_gcp_lists_the_instance_it_just_created(tmp_path, sandbox_bin):
    """The round trip, not a fixture: the stub reports what it created, and
    the app and function names come back out of the generated instance name."""
    fake_cloud.install(sandbox_bin, name="gcloud")
    fake_cloud.install_unreachable_ssh(sandbox_bin)
    app, fn = _app(
        tmp_path,
        gcp_config=GcpConfig(project="p", zone="z", on_finish="leave", ssh_ready_wait_seconds=1),
    )
    _run_until_dispatch_fails(gcp, app, fn)

    rows = gcp.list_jobs(project="p")
    assert [(r.backend, r.app, r.function) for r in rows] == [("gcp", "vision", "train")]
    assert rows[0].name.startswith("runplz-vision-train-")
    assert rows[0].status == "RUNNING"


def test_aws_lists_the_instance_it_just_created(tmp_path, sandbox_bin):
    """Same round trip on EC2, where the name only ever existed inside
    `--tag-specifications` at create time and has to come back off the tag."""
    fake_cloud.install(sandbox_bin, name="aws")
    fake_cloud.install_unreachable_ssh(sandbox_bin)
    app, fn = _app(
        tmp_path,
        aws_config=AwsConfig(
            region="us-east-1", key_name="k", on_finish="leave", ssh_ready_wait_seconds=1
        ),
    )
    _run_until_dispatch_fails(aws, app, fn)

    rows = aws.list_jobs(region="us-east-1")
    assert [(r.backend, r.app, r.function) for r in rows] == [("aws", "vision", "train")]
    assert rows[0].name.startswith("runplz-vision-train-")
    assert rows[0].status == "running"


def test_listing_argv_is_what_the_cli_actually_receives(tmp_path, sandbox_bin):
    """The filter is the whole point of the command. gcloud joins its flags
    with `=`, so this asserts the flag whole -- a listing that dropped the
    label filter would report every instance in the project as a runplz job."""
    log = fake_cloud.install(sandbox_bin, name="gcloud")
    gcp.list_jobs(project="proj", zone="us-central1-a")

    lists = fake_cloud.calls_matching(log, "list")
    assert len(lists) == 1, fake_cloud.calls(log)
    assert "--filter=labels.runplz=1" in lists[0]
    assert "--project=proj" in lists[0]
    assert "--zones=us-central1-a" in lists[0]
    assert "--format=json" in lists[0]


def test_aws_listing_argv_is_what_the_cli_actually_receives(sandbox_bin):
    """aws splits its flags, so the filter values are separate argv elements."""
    log = fake_cloud.install(sandbox_bin, name="aws")
    aws.list_jobs(region="us-east-1")

    describes = fake_cloud.calls_matching(log, "describe-instances")
    assert len(describes) == 1, fake_cloud.calls(log)
    assert "Name=tag:runplz,Values=1" in describes[0]
    assert "--filters" in describes[0]


def test_nothing_created_lists_as_no_jobs(sandbox_bin):
    fake_cloud.install(sandbox_bin, name="gcloud")
    fake_cloud.install(sandbox_bin, name="aws")
    assert gcp.list_jobs(project="p") == []
    assert aws.list_jobs(region="us-east-1") == []


@pytest.mark.parametrize("on_finish", ["delete", "stop"])
def test_a_torn_down_gcp_instance_stops_being_listed(tmp_path, sandbox_bin, on_finish):
    """The other half of the round trip. A run that tore itself down must not
    keep showing up as a live job."""
    fake_cloud.install(sandbox_bin, name="gcloud")
    fake_cloud.install_unreachable_ssh(sandbox_bin)
    app, fn = _app(
        tmp_path,
        gcp_config=GcpConfig(project="p", zone="z", on_finish=on_finish, ssh_ready_wait_seconds=1),
    )
    _run_until_dispatch_fails(gcp, app, fn)

    assert gcp.list_jobs(project="p") == []


@pytest.mark.parametrize(
    "on_finish, still_listed, expected_status",
    [
        # `delete` terminates the box, and a terminated instance is outside the
        # state filter production asks for. `stop` is inside it on purpose: a
        # stopped box is still costing money and still the user's to clean up,
        # so `runplz ps` is supposed to keep showing it.
        ("delete", False, None),
        ("stop", True, "stopped"),
    ],
)
def test_aws_listing_follows_the_state_filter_production_asks_for(
    tmp_path, sandbox_bin, on_finish, still_listed, expected_status
):
    fake_cloud.install(sandbox_bin, name="aws")
    fake_cloud.install_unreachable_ssh(sandbox_bin)
    app, fn = _app(
        tmp_path,
        aws_config=AwsConfig(
            region="us-east-1", key_name="k", on_finish=on_finish, ssh_ready_wait_seconds=1
        ),
    )
    _run_until_dispatch_fails(aws, app, fn)

    rows = aws.list_jobs(region="us-east-1")
    assert [r.status for r in rows] == ([expected_status] if still_listed else [])


@pytest.mark.parametrize(
    "name, route, call",
    [
        ("gcloud", "compute/instances/list", lambda: gcp.list_jobs(project="p")),
        (
            "aws",
            "ec2/describe-instances#by-filter",
            lambda: aws.list_jobs(region="us-east-1"),
        ),
    ],
)
def test_a_malformed_listing_response_is_an_error_not_an_empty_list(sandbox_bin, name, route, call):
    """The failure this whole issue exists to prevent: a degraded provider
    must not be read as "your jobs are gone". The aws case also pins that
    `malformed` can target one call shape of a subcommand -- the by-id form
    that fetches an instance's IP is left alone."""
    fake_cloud.install(sandbox_bin, name=name, malformed={route: "{not-json"})
    with pytest.raises(RuntimeError, match="malformed JSON"):
        call()


@pytest.mark.parametrize(
    "name, route, call",
    [
        ("gcloud", "compute/instances/list", lambda: gcp.list_jobs(project="p")),
        ("aws", "ec2/describe-instances", lambda: aws.list_jobs(region="us-east-1")),
    ],
)
def test_a_failing_provider_cli_surfaces_rather_than_listing_nothing(
    sandbox_bin, name, route, call
):
    fake_cloud.install(
        sandbox_bin, name=name, fail_times={route: 1}, fail_message="ServiceUnavailable"
    )
    with pytest.raises(RuntimeError, match="failed"):
        call()


def test_stub_rejects_a_listing_that_dropped_its_filter(sandbox_bin):
    """The harness has to be able to fail the production argv, or these tests
    prove nothing. Both halves: a listing missing its filter is refused, and
    `ec2 describe-instances` with neither discriminator is ambiguous."""
    fake_cloud.install(sandbox_bin, name="gcloud")
    fake_cloud.install(sandbox_bin, name="aws")

    unfiltered = subprocess.run(
        ["gcloud", "compute", "instances", "list", "--project=p", "--format=json"],
        capture_output=True,
        text=True,
    )
    assert unfiltered.returncode == 2
    assert "missing required option" in unfiltered.stderr

    ambiguous = subprocess.run(
        ["aws", "ec2", "describe-instances", "--region", "us-east-1", "--output", "json"],
        capture_output=True,
        text=True,
    )
    assert ambiguous.returncode == 2
    assert "needs exactly one of" in ambiguous.stderr
