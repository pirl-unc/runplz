"""SshOptions: how runplz reaches a box (issue #79).

`port` used to be the only thing callers could pin, threaded through ~50
call sites. A second scalar beside it would have doubled that and made a
third worse, so the plumbing carries one object instead.
"""

import json
from pathlib import Path
from unittest import mock

import pytest

from runplz import AwsConfig, SshConfig, cli, runs
from runplz.backends import aws, ssh
from runplz.backends.ssh_common import SSH_OPTS, SshOptions, rsync_ssh_transport, ssh_cmd_opts


def test_default_options_change_nothing():
    assert ssh_cmd_opts() == list(SSH_OPTS)
    assert ssh_cmd_opts(SshOptions()) == list(SSH_OPTS)


def test_port_and_identity_reach_the_ssh_argv():
    argv = ssh_cmd_opts(SshOptions(port=2222, identity_file="/keys/k.pem"))
    assert argv[-6:] == ["-p", "2222", "-i", "/keys/k.pem", "-o", "IdentitiesOnly=yes"]


def test_identity_file_is_user_expanded():
    argv = ssh_cmd_opts(SshOptions(identity_file="~/.ssh/k.pem"))
    assert str(Path.home()) in argv[argv.index("-i") + 1]
    assert "~" not in argv[argv.index("-i") + 1]


def test_identities_only_accompanies_an_explicit_key():
    """Otherwise a loaded agent offers its other keys first and can trip
    the server's MaxAuthTries before ours is ever tried."""
    assert "IdentitiesOnly=yes" in ssh_cmd_opts(SshOptions(identity_file="/k.pem"))
    assert "IdentitiesOnly=yes" not in ssh_cmd_opts(SshOptions())


@pytest.mark.parametrize(
    "value,expect_port",
    [(None, None), (2222, 2222), (SshOptions(port=22), 22)],
)
def test_coerce_accepts_none_a_bare_port_and_itself(value, expect_port):
    assert SshOptions.coerce(value).port == expect_port


def test_coerce_rejects_nonsense():
    with pytest.raises(TypeError):
        SshOptions.coerce("2222")
    with pytest.raises(TypeError):
        SshOptions.coerce(True)


def test_rsync_transport_only_overrides_when_it_has_something_to_say():
    """A bare `-e ssh ...` for every rsync would be churn with no effect."""
    default = rsync_ssh_transport(None)
    assert rsync_ssh_transport(SshOptions()) == default
    assert rsync_ssh_transport(SshOptions(identity_file="/k.pem")) != default
    assert rsync_ssh_transport(SshOptions(port=2222)) != default


def test_rsync_transport_carries_the_identity():
    transport = rsync_ssh_transport(SshOptions(identity_file="/keys/k.pem"))
    assert "/keys/k.pem" in transport
    assert transport.startswith("ssh ")


# ---------------------------------------------------------------------------
# configs


def test_ssh_config_key_path_reaches_the_dispatch(tmp_path):
    key = tmp_path / "box.pem"
    key.write_text("x")
    cfg = SshConfig(port=2222, ssh_key_path=str(key))
    seen = {}

    class App:
        name = "t"
        ssh_config = cfg
        repo_root = tmp_path

    class Fn:
        name = "train"

    def fake_dispatch(**kw):
        seen.update(kw)

    with mock.patch.object(ssh, "dispatch_to_target", fake_dispatch):
        with mock.patch.object(ssh, "wait_until_ssh_reachable"):
            with mock.patch.object(ssh, "_warn_on_spec_mismatch"):
                ssh.run(App(), Fn(), [], {}, host="box.example.com")

    assert seen["ssh_opts"].port == 2222
    assert seen["ssh_opts"].identity_file == str(key)


def test_aws_hands_its_key_back_with_the_target(tmp_path):
    """The EC2 target is only knowable after the box exists, so the key has
    to travel back from provision() alongside it — the earlier version of
    this test asserted nothing and would not have caught that regressing."""
    key = tmp_path / "ec2.pem"
    key.write_text("x")
    cfg = AwsConfig(region="us-east-1", key_name="k", ssh_key_path=str(key))

    captured = {}

    def fake_run_on_provisioned_vm(**kw):
        captured["target"], captured["opts"] = kw["provision"]()

    class App:
        name = "vision"
        aws_config = cfg
        repo_root = tmp_path

    class Fn:
        name = "train"
        gpu = None
        min_gpus = 1
        min_cpu = None
        min_memory = None
        min_gpu_memory = None
        min_disk = None

    def fake_run_cli(cmd, **kw):
        if "run-instances" in cmd:
            return {"Instances": [{"InstanceId": "i-123"}]}
        if "get-parameter" in cmd:
            return mock.Mock(returncode=0, stdout="ami-123\n", stderr="")
        return mock.Mock(returncode=0, stdout="1.2.3.4\n", stderr="")

    with mock.patch.object(aws, "run_on_provisioned_vm", fake_run_on_provisioned_vm):
        with mock.patch.object(aws, "run_cli", fake_run_cli):
            aws.run(App(), Fn(), [], {})

    assert captured["target"] == "ubuntu@1.2.3.4"
    assert captured["opts"] is not None, "provision() dropped its ssh options"
    assert captured["opts"].identity_file == str(key)


# ---------------------------------------------------------------------------
# following a run afterwards


def _manifest(tmp_path, **ssh_options):
    """A dispatched run: the manifest (which is uploaded) plus the ssh
    sidecar (which never leaves this machine)."""
    meta = tmp_path / ".runplz"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "run.json").write_text(
        json.dumps(
            {
                "run_id": "r1",
                "target": "ubuntu@1.2.3.4",
                "remote_paths": {"meta": "~/runplz-runs/r1/out/.runplz"},
            }
        )
    )
    if ssh_options:
        (meta / "ssh.json").write_text(json.dumps(ssh_options))


@pytest.mark.parametrize("command", ["tail", "status", "kill"])
def test_follow_up_commands_reuse_the_recorded_options(tmp_path, command):
    """Otherwise `runplz tail` on an EC2 box fails to authenticate even
    though the dispatch that created it worked."""
    _manifest(tmp_path, identity_file="/keys/ec2.pem", port=2200)
    stdout = "---SUMMARY---\nalive_after=0\n---END---\n"
    fake = mock.Mock(returncode=0, stdout=stdout, stderr="")
    kwargs = {"lines": 5, "follow": False} if command == "tail" else {}
    with mock.patch("runplz.runs.subprocess.run", return_value=fake) as run_mock:
        getattr(runs, command)(
            outputs_dir=tmp_path, host_override=None, run_id_override=None, **kwargs
        )
    argv = run_mock.call_args.args[0]
    assert "/keys/ec2.pem" in argv
    assert "2200" in argv


def test_explicit_ssh_key_flag_wins_over_the_manifest(tmp_path):
    _manifest(tmp_path, identity_file="/keys/from-manifest.pem")
    fake = mock.Mock(returncode=0, stdout="---SUMMARY---\nalive_after=0\n---END---\n", stderr="")
    with mock.patch("runplz.runs.subprocess.run", return_value=fake) as run_mock:
        runs.kill(
            outputs_dir=tmp_path,
            host_override=None,
            run_id_override=None,
            ssh_overrides={"identity_file": "/keys/override.pem"},
        )
    argv = run_mock.call_args.args[0]
    assert "/keys/override.pem" in argv
    assert "/keys/from-manifest.pem" not in argv


def test_cli_exposes_ssh_key_and_port(tmp_path):
    _manifest(tmp_path)
    with mock.patch.object(runs, "kill", return_value=0) as kill_mock:
        cli.main(
            ["kill", "--outputs-dir", str(tmp_path), "--ssh-key", "/k.pem", "--ssh-port", "2200"]
        )
    overrides = kill_mock.call_args.kwargs["ssh_overrides"]
    assert overrides == {"identity_file": "/k.pem", "port": 2200}


def test_cli_without_ssh_flags_defers_to_the_manifest(tmp_path):
    _manifest(tmp_path)
    with mock.patch.object(runs, "kill", return_value=0) as kill_mock:
        cli.main(["kill", "--outputs-dir", str(tmp_path)])
    assert kill_mock.call_args.kwargs["ssh_overrides"] == {}


# ---------------------------------------------------------------------------
# the manifest records a path, never a key


def test_manifest_records_only_what_is_needed_to_reconnect():
    opts = SshOptions(port=2200, identity_file="~/.ssh/k.pem")
    recorded = opts.to_dict()
    assert recorded == {"port": 2200, "identity_file": "~/.ssh/k.pem"}
    assert SshOptions.from_dict(recorded) == opts


def test_default_options_record_nothing():
    assert SshOptions().to_dict() == {}
    assert SshOptions.from_dict({}) == SshOptions()
    assert SshOptions.from_dict(None) == SshOptions()


# ---------------------------------------------------------------------------
# what the review turned up


def test_new_host_keys_are_accepted_so_a_fresh_cloud_ip_can_connect():
    """Every probe runs BatchMode=yes, which cannot answer OpenSSH's default
    prompt — a brand-new EC2 IP would fail verification for the whole
    ssh_ready_wait_seconds while the instance bills."""
    assert "StrictHostKeyChecking=accept-new" in ssh_cmd_opts()


def test_a_changed_host_key_still_fails():
    """accept-new trusts a first sighting only; `no` would trust anything."""
    argv = ssh_cmd_opts()
    assert "StrictHostKeyChecking=no" not in argv


@pytest.mark.parametrize("bad", ["", "   ", "/nonexistent/key.pem"])
def test_configs_reject_a_key_path_that_will_not_work(bad):
    """IdentitiesOnly=yes means a typo'd path also disables the agent
    fallback, so this surfaces as an unexplained Permission denied — or on
    aws, a 30-minute wait on a billed box."""
    with pytest.raises(ValueError, match="ssh_key_path"):
        SshConfig(ssh_key_path=bad)
    with pytest.raises(ValueError, match="ssh_key_path"):
        AwsConfig(region="us-east-1", key_name="k", ssh_key_path=bad)


def test_one_cli_override_does_not_discard_the_other_field(tmp_path):
    """`--ssh-key` alone must not drop the port the dispatch recorded."""
    _manifest(tmp_path, identity_file="/keys/old.pem", port=2200)
    fake = mock.Mock(returncode=0, stdout="---SUMMARY---\nalive_after=0\n---END---\n", stderr="")
    with mock.patch("runplz.runs.subprocess.run", return_value=fake) as run_mock:
        runs.kill(
            outputs_dir=tmp_path,
            host_override=None,
            run_id_override=None,
            ssh_overrides={"identity_file": "/keys/new.pem"},
        )
    argv = run_mock.call_args.args[0]
    assert "/keys/new.pem" in argv
    assert "2200" in argv, "the recorded port was dropped"


def test_cli_rejects_an_out_of_range_ssh_port(tmp_path):
    _manifest(tmp_path)
    for bad in ("0", "70000"):
        with pytest.raises(SystemExit):
            cli.main(["kill", "--outputs-dir", str(tmp_path), "--ssh-port", bad])


def test_ps_can_authenticate_to_a_host_that_needs_a_key(tmp_path):
    """`runplz ps --host` against the EC2 box this PR enables."""
    from runplz.backends import ssh as ssh_backend

    with mock.patch.object(ssh_backend, "list_jobs", return_value=[]) as jobs:
        cli.main(["ps", "--host", "ubuntu@1.2.3.4", "--ssh-key", "/k.pem", "--ssh-port", "2200"])
    kwargs = jobs.call_args.kwargs
    assert kwargs["ssh_key_path"] == "/k.pem"
    assert kwargs["port"] == 2200


# ---------------------------------------------------------------------------
# the key path never leaves this machine


def test_the_uploaded_manifest_carries_no_ssh_details(tmp_path):
    """It is heredoc'd onto a rented, possibly multi-tenant box; a key path
    there discloses the local username and key filename."""
    from runplz.backends.ssh_common import build_remote_run_manifest, make_remote_run_context

    remote_run = make_remote_run_context(backend="aws", target="ubuntu@1.2.3.4", function_name="t")
    manifest = build_remote_run_manifest(
        remote_run=remote_run,
        repo=tmp_path,
        outputs_dir="out",
        args=[],
        kwargs={},
        env={},
    )
    blob = json.dumps(manifest)
    assert "ssh_options" not in manifest
    assert ".pem" not in blob
    assert "identity_file" not in blob


def test_ssh_details_are_written_beside_the_manifest_locally(tmp_path):
    from runplz.backends.ssh_common import read_local_ssh_options, write_local_ssh_options

    write_local_ssh_options(tmp_path, SshOptions(port=2200, identity_file="/k.pem"))
    assert (tmp_path / ".runplz" / "ssh.json").is_file()
    assert read_local_ssh_options(tmp_path) == SshOptions(port=2200, identity_file="/k.pem")


def test_default_options_leave_no_sidecar_behind(tmp_path):
    """And a later default-options run clears a stale one."""
    from runplz.backends.ssh_common import read_local_ssh_options, write_local_ssh_options

    write_local_ssh_options(tmp_path, SshOptions(identity_file="/k.pem"))
    write_local_ssh_options(tmp_path, None)
    assert not (tmp_path / ".runplz" / "ssh.json").exists()
    assert read_local_ssh_options(tmp_path) == SshOptions()


def test_missing_sidecar_reads_as_defaults(tmp_path):
    from runplz.backends.ssh_common import read_local_ssh_options

    assert read_local_ssh_options(tmp_path) == SshOptions()
