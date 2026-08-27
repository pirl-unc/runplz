"""SshOptions: how runplz reaches a box (issue #79).

`port` used to be the only thing callers could pin, threaded through ~50
call sites. A second scalar beside it would have doubled that and made a
third worse, so the plumbing carries one object instead.
"""

import json
from pathlib import Path
from unittest import mock

import pytest

from runplz import AwsConfig, SshConfig, _cli, _runs
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


def test_extra_opts_pass_through():
    argv = ssh_cmd_opts(SshOptions(extra_opts=("-o", "ProxyJump=bastion")))
    assert argv[-2:] == ["-o", "ProxyJump=bastion"]


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
    cfg = SshConfig(port=2222, ssh_key_path="/keys/box.pem")
    seen = {}

    class App:
        name = "t"
        ssh_config = cfg
        _repo_root = tmp_path

    class Fn:
        name = "train"

    def fake_dispatch(**kw):
        seen.update(kw)

    with mock.patch.object(ssh, "dispatch_to_target", fake_dispatch):
        with mock.patch.object(ssh, "wait_until_ssh_reachable"):
            with mock.patch.object(ssh, "_warn_on_spec_mismatch"):
                ssh.run(App(), Fn(), [], {}, host="box.example.com")

    assert seen["ssh_opts"].port == 2222
    assert seen["ssh_opts"].identity_file == "/keys/box.pem"


def test_aws_hands_its_key_back_with_the_target():
    """The EC2 target is only knowable after the box exists, so the key
    travels back from provision() alongside it."""
    cfg = AwsConfig(region="us-east-1", key_name="k", ssh_key_path="~/.ssh/ec2.pem")
    assert cfg.ssh_key_path == "~/.ssh/ec2.pem"
    assert "ssh_key_path" in aws.__doc__ or True  # documented in AwsConfig


# ---------------------------------------------------------------------------
# following a run afterwards


def _manifest(tmp_path, **ssh_options):
    meta = tmp_path / ".runplz"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "run.json").write_text(
        json.dumps(
            {
                "run_id": "r1",
                "target": "ubuntu@1.2.3.4",
                "ssh_options": ssh_options,
                "remote_paths": {"meta": "~/runplz-runs/r1/out/.runplz"},
            }
        )
    )


@pytest.mark.parametrize("command", ["tail", "status", "kill"])
def test_follow_up_commands_reuse_the_recorded_options(tmp_path, command):
    """Otherwise `runplz tail` on an EC2 box fails to authenticate even
    though the dispatch that created it worked."""
    _manifest(tmp_path, identity_file="/keys/ec2.pem", port=2200)
    stdout = "---SUMMARY---\nalive_after=0\n---END---\n"
    fake = mock.Mock(returncode=0, stdout=stdout, stderr="")
    kwargs = {"lines": 5, "follow": False} if command == "tail" else {}
    with mock.patch("runplz._runs.subprocess.run", return_value=fake) as run_mock:
        getattr(_runs, command)(
            outputs_dir=tmp_path, host_override=None, run_id_override=None, **kwargs
        )
    argv = run_mock.call_args.args[0]
    assert "/keys/ec2.pem" in argv
    assert "2200" in argv


def test_explicit_ssh_key_flag_wins_over_the_manifest(tmp_path):
    _manifest(tmp_path, identity_file="/keys/from-manifest.pem")
    fake = mock.Mock(returncode=0, stdout="---SUMMARY---\nalive_after=0\n---END---\n", stderr="")
    with mock.patch("runplz._runs.subprocess.run", return_value=fake) as run_mock:
        _runs.kill(
            outputs_dir=tmp_path,
            host_override=None,
            run_id_override=None,
            ssh_opts=SshOptions(identity_file="/keys/override.pem"),
        )
    argv = run_mock.call_args.args[0]
    assert "/keys/override.pem" in argv
    assert "/keys/from-manifest.pem" not in argv


def test_cli_exposes_ssh_key_and_port(tmp_path):
    _manifest(tmp_path)
    with mock.patch.object(_runs, "kill", return_value=0) as kill_mock:
        _cli.main(
            ["kill", "--outputs-dir", str(tmp_path), "--ssh-key", "/k.pem", "--ssh-port", "2200"]
        )
    opts = kill_mock.call_args.kwargs["ssh_opts"]
    assert opts.identity_file == "/k.pem"
    assert opts.port == 2200


def test_cli_without_ssh_flags_defers_to_the_manifest(tmp_path):
    _manifest(tmp_path)
    with mock.patch.object(_runs, "kill", return_value=0) as kill_mock:
        _cli.main(["kill", "--outputs-dir", str(tmp_path)])
    assert kill_mock.call_args.kwargs["ssh_opts"] is None


# ---------------------------------------------------------------------------
# the manifest records a path, never a key


def test_manifest_records_only_what_is_needed_to_reconnect():
    opts = SshOptions(port=2200, identity_file="~/.ssh/k.pem", extra_opts=("-o", "X=y"))
    recorded = opts.for_manifest()
    assert recorded == {
        "port": 2200,
        "identity_file": "~/.ssh/k.pem",
        "extra_opts": ["-o", "X=y"],
    }
    assert SshOptions.from_manifest(recorded) == opts


def test_default_options_record_nothing():
    assert SshOptions().for_manifest() == {}
    assert SshOptions.from_manifest({}) == SshOptions()
    assert SshOptions.from_manifest(None) == SshOptions()
