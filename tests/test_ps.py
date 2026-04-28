"""Coverage for `runplz ps` — label-filtered job listing across backends."""

import json
from unittest import mock

import pytest

from runplz import _cli
from runplz.backends import brev, local, modal, ssh

# ---------------------------------------------------------------------------
# local


def test_local_list_jobs_parses_docker_ps_json_lines():
    stdout = "\n".join(
        [
            json.dumps(
                {
                    "ID": "abc123",
                    "Names": "runplz-demo-train",
                    "CreatedAt": "2026-04-23 10:00:00 +0000 UTC",
                    "Status": "Up 5 minutes",
                    "Labels": "runplz=1,runplz-app=demo,runplz-function=train",
                }
            ),
            json.dumps(
                {
                    "ID": "def456",
                    "Names": "runplz-demo-eval",
                    "CreatedAt": "2026-04-23 10:05:00 +0000 UTC",
                    "Status": "Up 1 minute",
                    "Labels": "runplz=1,runplz-app=demo,runplz-function=eval",
                }
            ),
        ]
    )
    fake = mock.Mock(returncode=0, stdout=stdout, stderr="")
    with mock.patch("runplz.backends.local.subprocess.run", return_value=fake) as run_mock:
        rows = local.list_jobs()

    cmd = run_mock.call_args.args[0]
    assert cmd[:2] == ["docker", "ps"]
    assert "label=runplz=1" in cmd
    assert len(rows) == 2
    assert rows[0]["backend"] == "local"
    assert rows[0]["app"] == "demo"
    assert rows[0]["function"] == "train"
    assert rows[0]["name"] == "runplz-demo-train"
    assert rows[1]["function"] == "eval"


def test_local_list_jobs_silent_when_docker_daemon_down():
    """3.15.2: 'docker daemon not running' is informational on dev machines —
    if the daemon isn't running there can't be any local runplz containers.
    Treat as zero jobs, not a warning."""
    fake = mock.Mock(
        returncode=1,
        stdout="",
        stderr=(
            "Cannot connect to the Docker daemon at unix:///.../docker.sock. "
            "Is the docker daemon running?"
        ),
    )
    with mock.patch("runplz.backends.local.subprocess.run", return_value=fake):
        rows = local.list_jobs()
    assert rows == []


def test_local_list_jobs_silent_when_docker_binary_missing():
    """No `docker` on PATH at all is the same kind of "no local containers
    possible" condition — return empty rather than raising."""
    with mock.patch(
        "runplz.backends.local.subprocess.run", side_effect=FileNotFoundError("docker")
    ):
        rows = local.list_jobs()
    assert rows == []


def test_local_list_jobs_still_raises_on_other_docker_failures():
    """A non-daemon-down docker failure (permissions, etc.) is a real error."""
    fake = mock.Mock(returncode=1, stdout="", stderr="permission denied: /var/run/docker.sock")
    with mock.patch("runplz.backends.local.subprocess.run", return_value=fake):
        with pytest.raises(RuntimeError, match="docker ps"):
            local.list_jobs()


def test_local_list_jobs_ignores_malformed_json_lines():
    stdout = "not-json\n" + json.dumps(
        {
            "ID": "abc",
            "Names": "runplz-x-y",
            "CreatedAt": "t",
            "Status": "Up",
            "Labels": "runplz=1,runplz-app=x,runplz-function=y",
        }
    )
    fake = mock.Mock(returncode=0, stdout=stdout, stderr="")
    with mock.patch("runplz.backends.local.subprocess.run", return_value=fake):
        rows = local.list_jobs()
    assert len(rows) == 1


def test_local_run_stamps_labels(tmp_path):
    """The run path must actually add the labels that list_jobs filters on."""
    from runplz import App, Image

    app = App("demo")
    app._repo_root = tmp_path
    jobdir = tmp_path / "jobs"
    jobdir.mkdir()
    job = jobdir / "job.py"
    job.write_text("# fake\n")

    @app.function(image=Image.from_registry("ubuntu:22.04"))
    def train():  # pragma: no cover
        return "ok"

    fn = app.functions["train"]
    fn.module_file = str(job)

    calls = []

    def fake_run(cmd, *args, **kwargs):
        calls.append(cmd)
        return mock.Mock(returncode=0, stdout="{}", stderr="")

    with mock.patch("runplz.backends.local.subprocess.run", fake_run):
        with mock.patch("runplz.backends.local._nvidia_available", return_value=False):
            local.run(app, fn, [], {})

    run_cmd = calls[1]
    labels = [run_cmd[i + 1] for i, t in enumerate(run_cmd) if t == "--label"]
    assert "runplz=1" in labels
    assert "runplz-app=demo" in labels
    assert "runplz-function=train" in labels


# ---------------------------------------------------------------------------
# brev


def test_brev_list_jobs_filters_on_runplz_prefix():
    rows_json = json.dumps(
        [
            {"name": "runplz-demo-train-abcd1234", "status": "RUNNING"},
            {"name": "my-own-box", "status": "RUNNING"},
            {"name": "runplz-demo-eval-deadbeef", "status": "DEPLOYING"},
            # User-named --instance box that happens to start with runplz-.
            # Must NOT be treated as a live ephemeral run (no uuid suffix).
            {"name": "runplz-mygpu", "status": "RUNNING"},
        ]
    )
    fake = mock.Mock(returncode=0, stdout=rows_json, stderr="")
    with mock.patch("runplz.backends.brev._brev_capture", return_value=fake):
        jobs = brev.list_jobs()
    assert [j["name"] for j in jobs] == [
        "runplz-demo-train-abcd1234",
        "runplz-demo-eval-deadbeef",
    ]
    assert jobs[0]["app"] == "demo"
    assert jobs[0]["function"] == "train"
    assert jobs[0]["status"] == "RUNNING"


def test_brev_split_ephemeral_name_handles_multi_segment_app():
    assert brev._split_ephemeral_name("runplz-my-cool-app-train-abcd1234") == (
        "my-cool-app",
        "train",
    )
    assert brev._split_ephemeral_name("not-ours") == ("", "")
    assert brev._split_ephemeral_name("runplz-short") == ("", "")


def test_brev_list_jobs_raises_on_cli_failure():
    fake = mock.Mock(returncode=1, stdout="", stderr="bad auth")
    with mock.patch("runplz.backends.brev._brev_capture", return_value=fake):
        with pytest.raises(RuntimeError, match="brev ls"):
            brev.list_jobs()


# ---------------------------------------------------------------------------
# modal


def test_modal_list_jobs_parses_json():
    payload = json.dumps(
        [
            {"name": "runplz-demo-train", "state": "running", "created_at": "t1"},
            {"name": "runplz-demo-eval", "state": "stopped"},  # dropped
            {"name": "user-other-app", "state": "running"},  # dropped
        ]
    )
    fake = mock.Mock(returncode=0, stdout=payload, stderr="")
    with mock.patch("runplz.backends.modal.subprocess.run", return_value=fake):
        jobs = modal.list_jobs()
    assert [j["name"] for j in jobs] == ["runplz-demo-train"]
    assert jobs[0]["app"] == "demo"
    assert jobs[0]["function"] == "train"
    assert jobs[0]["backend"] == "modal"


def test_modal_list_jobs_falls_back_to_text_parse():
    # First call (--json) produces text that isn't a JSON array/object.
    text_table = (
        "+----------------+----------------------+----------+\n"
        "| App ID         | Name                 | State    |\n"
        "+----------------+----------------------+----------+\n"
        "| ap_abc         | runplz-demo-train    | running  |\n"
        "| ap_def         | user-random          | running  |\n"
        "+----------------+----------------------+----------+\n"
    )
    returns = [
        mock.Mock(returncode=0, stdout="not json at all", stderr=""),
        mock.Mock(returncode=0, stdout=text_table, stderr=""),
    ]
    with mock.patch("runplz.backends.modal.subprocess.run", side_effect=returns):
        jobs = modal.list_jobs()
    names = [j["name"] for j in jobs]
    assert "runplz-demo-train" in names


def test_modal_split_app_name():
    assert modal._split_modal_app_name("runplz-my-long-app-train") == ("my-long-app", "train")
    assert modal._split_modal_app_name("other-app") == ("", "")


# ---------------------------------------------------------------------------
# ssh


def test_ssh_list_jobs_requires_host_and_parses_remote_docker_ps():
    stdout = json.dumps(
        {
            "ID": "zz",
            "Names": "runplz-demo-train-abcd1234",
            "CreatedAt": "2026-04-23 10:00:00 +0000 UTC",
            "Status": "Up 3 minutes",
            "Labels": "runplz=1,runplz-app=demo,runplz-function=train",
        }
    )
    fake = mock.Mock(returncode=0, stdout=stdout, stderr="")
    with mock.patch("runplz.backends.ssh.subprocess.run", return_value=fake) as run_mock:
        jobs = ssh.list_jobs(host="my.box")
    cmd = run_mock.call_args.args[0]
    assert cmd[0] == "ssh"
    assert "my.box" in cmd
    assert len(jobs) == 1
    assert jobs[0]["backend"] == "ssh"
    assert jobs[0]["app"] == "demo"
    assert jobs[0]["function"] == "train"
    # host prefix is stamped onto the name so ps output is unambiguous across hosts.
    assert jobs[0]["name"].startswith("my.box:")


# ---------------------------------------------------------------------------
# CLI


def test_ps_cli_fans_out_and_prints_rows(capsys):
    rows_local = [
        {
            "backend": "local",
            "name": "runplz-demo-train",
            "app": "demo",
            "function": "train",
            "started": "t",
            "status": "Up 5m",
        }
    ]
    with mock.patch.object(local, "list_jobs", return_value=rows_local):
        with mock.patch.object(brev, "list_jobs", return_value=[]):
            with mock.patch.object(modal, "list_jobs", return_value=[]):
                rc = _cli.main(["ps"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "BACKEND" in out
    assert "runplz-demo-train" in out


def test_ps_cli_reports_empty_when_no_jobs(capsys):
    with mock.patch.object(local, "list_jobs", return_value=[]):
        with mock.patch.object(brev, "list_jobs", return_value=[]):
            with mock.patch.object(modal, "list_jobs", return_value=[]):
                rc = _cli.main(["ps"])
    assert rc == 0
    assert "no runplz jobs running" in capsys.readouterr().out


def test_ps_cli_single_backend_filter(capsys):
    with mock.patch.object(local, "list_jobs", return_value=[]) as local_mock:
        with mock.patch.object(brev, "list_jobs") as brev_mock:
            with mock.patch.object(modal, "list_jobs") as modal_mock:
                _cli.main(["ps", "local"])
    local_mock.assert_called_once()
    brev_mock.assert_not_called()
    modal_mock.assert_not_called()


def test_ps_cli_surfaces_errors_as_warnings(capsys):
    """Errors from one backend show up as warnings on stderr while the
    others' results still print. 3.15.2: rc is 0 as long as at least one
    backend succeeded (errors are partial, not total)."""
    with mock.patch.object(local, "list_jobs", side_effect=RuntimeError("no daemon")):
        with mock.patch.object(brev, "list_jobs", return_value=[]):
            with mock.patch.object(modal, "list_jobs", return_value=[]):
                rc = _cli.main(["ps"])
    err = capsys.readouterr().err
    assert "local listing failed" in err
    # Two backends succeeded (with empty results) so rc=0 — partial failures
    # don't make the whole command fail.
    assert rc == 0


def test_ps_cli_accepts_host_flag_3_15_1():
    """3.15.1: `--host` is the canonical name (matches `runplz tail`/`status`/`ssh`).
    `--ssh` is kept as a back-compat alias since pre-3.15.1 used it here."""
    from runplz.backends import ssh as ssh_backend

    with mock.patch.object(local, "list_jobs", return_value=[]):
        with mock.patch.object(brev, "list_jobs", return_value=[]):
            with mock.patch.object(modal, "list_jobs", return_value=[]):
                with mock.patch.object(ssh_backend, "list_jobs", return_value=[]) as ssh_mock:
                    _cli.main(["ps", "--host", "my.gpu.box"])
    ssh_mock.assert_called_once_with(host="my.gpu.box")


def test_ps_cli_prints_no_jobs_when_some_backend_succeeds(capsys):
    """3.15.2: don't suppress the table just because ONE backend errored.
    If at least one backend was queryable, an empty result is real
    information ('no jobs running anywhere reachable')."""
    with mock.patch.object(local, "list_jobs", side_effect=RuntimeError("docker missing")):
        with mock.patch.object(brev, "list_jobs", return_value=[]):
            with mock.patch.object(modal, "list_jobs", return_value=[]):
                rc = _cli.main(["ps"])
    out = capsys.readouterr()
    assert "no runplz jobs running" in out.out
    assert "local listing failed" in out.err
    # rc=0 because at least one backend succeeded.
    assert rc == 0


def test_ps_cli_returns_1_only_when_all_backends_fail(capsys):
    with mock.patch.object(local, "list_jobs", side_effect=RuntimeError("a")):
        with mock.patch.object(brev, "list_jobs", side_effect=RuntimeError("b")):
            with mock.patch.object(modal, "list_jobs", side_effect=RuntimeError("c")):
                rc = _cli.main(["ps"])
    out = capsys.readouterr()
    assert "no runplz jobs running" not in out.out
    assert rc == 1


def test_ps_cli_back_compat_ssh_flag_still_works():
    from runplz.backends import ssh as ssh_backend

    with mock.patch.object(local, "list_jobs", return_value=[]):
        with mock.patch.object(brev, "list_jobs", return_value=[]):
            with mock.patch.object(modal, "list_jobs", return_value=[]):
                with mock.patch.object(ssh_backend, "list_jobs", return_value=[]) as ssh_mock:
                    _cli.main(["ps", "--ssh", "my.gpu.box"])
    ssh_mock.assert_called_once_with(host="my.gpu.box")
