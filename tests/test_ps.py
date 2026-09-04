"""Coverage for `runplz ps` — label-filtered job listing across backends."""

import json
from unittest import mock

import pytest

from runplz import cli
from runplz.backends import brev, local, modal, ssh
from runplz.backends.listing import JobRecord

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
    assert rows[0].backend == "local"
    assert rows[0].app == "demo"
    assert rows[0].function == "train"
    assert rows[0].name == "runplz-demo-train"
    assert rows[1].function == "eval"


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
    app.repo_root = tmp_path
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
    assert [j.name for j in jobs] == [
        "runplz-demo-train-abcd1234",
        "runplz-demo-eval-deadbeef",
    ]
    assert jobs[0].app == "demo"
    assert jobs[0].function == "train"
    assert jobs[0].status == "RUNNING"


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
    assert [j.name for j in jobs] == ["runplz-demo-train"]
    assert jobs[0].app == "demo"
    assert jobs[0].function == "train"
    assert jobs[0].backend == "modal"


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
    names = [j.name for j in jobs]
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
    assert jobs[0].backend == "ssh"
    assert jobs[0].app == "demo"
    assert jobs[0].function == "train"
    # host prefix is stamped onto the name so ps output is unambiguous across hosts.
    assert jobs[0].name.startswith("my.box:")


# ---------------------------------------------------------------------------
# CLI


@pytest.fixture
def quiet_fan_out(monkeypatch):
    """Silence every fan-out backend so a test can assert on one of them.

    gcp/aws are in the default fan-out, so a test that patches only
    local/brev/modal is really asserting against whatever cloud credentials
    the developer's shell happens to export.
    """
    from runplz.backends import aws, gcp

    for module in (local, brev, modal, gcp, aws):
        monkeypatch.setattr(module, "list_jobs", lambda **kw: [], raising=False)


def test_ps_cli_fans_out_and_prints_rows(capsys, quiet_fan_out):
    rows_local = [
        JobRecord(
            backend="local",
            name="runplz-demo-train",
            app="demo",
            function="train",
            started="t",
            status="Up 5m",
        )
    ]
    with mock.patch.object(local, "list_jobs", return_value=rows_local):
        with mock.patch.object(brev, "list_jobs", return_value=[]):
            with mock.patch.object(modal, "list_jobs", return_value=[]):
                rc = cli.main(["ps"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "BACKEND" in out
    assert "runplz-demo-train" in out


def test_ps_cli_reports_empty_when_no_jobs(capsys, quiet_fan_out):
    with mock.patch.object(local, "list_jobs", return_value=[]):
        with mock.patch.object(brev, "list_jobs", return_value=[]):
            with mock.patch.object(modal, "list_jobs", return_value=[]):
                rc = cli.main(["ps"])
    assert rc == 0
    assert "no runplz jobs running" in capsys.readouterr().out


def test_ps_cli_single_backend_filter(capsys, quiet_fan_out):
    with mock.patch.object(local, "list_jobs", return_value=[]) as local_mock:
        with mock.patch.object(brev, "list_jobs") as brev_mock:
            with mock.patch.object(modal, "list_jobs") as modal_mock:
                cli.main(["ps", "local"])
    local_mock.assert_called_once()
    brev_mock.assert_not_called()
    modal_mock.assert_not_called()


def test_ps_cli_surfaces_errors_as_warnings(capsys, quiet_fan_out):
    """Errors from one backend show up as warnings on stderr while the
    others' results still print. 3.15.2: rc is 0 as long as at least one
    backend succeeded (errors are partial, not total)."""
    with mock.patch.object(local, "list_jobs", side_effect=RuntimeError("no daemon")):
        with mock.patch.object(brev, "list_jobs", return_value=[]):
            with mock.patch.object(modal, "list_jobs", return_value=[]):
                rc = cli.main(["ps"])
    err = capsys.readouterr().err
    assert "local listing failed" in err
    # Two backends succeeded (with empty results) so rc=0 — partial failures
    # don't make the whole command fail.
    assert rc == 0


def test_ps_cli_accepts_host_flag_3_15_1(quiet_fan_out):
    """3.15.1: `--host` is the canonical name (matches `runplz tail`/`status`/`ssh`).
    `--ssh` is kept as a back-compat alias since pre-3.15.1 used it here."""
    from runplz.backends import ssh as ssh_backend

    with mock.patch.object(local, "list_jobs", return_value=[]):
        with mock.patch.object(brev, "list_jobs", return_value=[]):
            with mock.patch.object(modal, "list_jobs", return_value=[]):
                with mock.patch.object(ssh_backend, "list_jobs", return_value=[]) as ssh_mock:
                    cli.main(["ps", "--host", "my.gpu.box"])
    ssh_mock.assert_called_once_with(host="my.gpu.box", port=None, ssh_key_path=None)


def test_ps_cli_prints_no_jobs_when_some_backend_succeeds(capsys, quiet_fan_out):
    """3.15.2: don't suppress the table just because ONE backend errored.
    If at least one backend was queryable, an empty result is real
    information ('no jobs running anywhere reachable')."""
    with mock.patch.object(local, "list_jobs", side_effect=RuntimeError("docker missing")):
        with mock.patch.object(brev, "list_jobs", return_value=[]):
            with mock.patch.object(modal, "list_jobs", return_value=[]):
                rc = cli.main(["ps"])
    out = capsys.readouterr()
    assert "no runplz jobs running" in out.out
    assert "local listing failed" in out.err
    # rc=0 because at least one backend succeeded.
    assert rc == 0


def test_ps_cli_returns_1_only_when_all_backends_fail(capsys, monkeypatch):
    """Every fan-out backend is failed explicitly, not left to fail by luck.

    Patching only local/brev/modal reached rc=1 because gcp/aws happened to
    raise MissingScope on a machine with no cloud env — which makes the test
    pass for a reason it does not state, and stop testing anything the day
    that changes.
    """
    from runplz.backends import aws, gcp

    def boom(**kw):
        raise RuntimeError("unreachable")

    for module in (local, brev, modal, gcp, aws):
        monkeypatch.setattr(module, "list_jobs", boom, raising=False)
    rc = cli.main(["ps"])
    out = capsys.readouterr()
    assert "no runplz jobs running" not in out.out
    # The table was suppressed, so the note explaining an empty table would
    # only bury the five warnings that say what actually went wrong.
    assert "was not listed" not in out.err
    assert out.err.count("listing failed") == 5
    assert rc == 1


def test_ps_cli_back_compat_ssh_flag_still_works(quiet_fan_out):
    from runplz.backends import ssh as ssh_backend

    with mock.patch.object(local, "list_jobs", return_value=[]):
        with mock.patch.object(brev, "list_jobs", return_value=[]):
            with mock.patch.object(modal, "list_jobs", return_value=[]):
                with mock.patch.object(ssh_backend, "list_jobs", return_value=[]) as ssh_mock:
                    cli.main(["ps", "--ssh", "my.gpu.box"])
    ssh_mock.assert_called_once_with(host="my.gpu.box", port=None, ssh_key_path=None)


# ---------------------------------------------------------------------------
# label producers and consumers share one vocabulary


def test_local_and_remote_label_flags_agree():
    """`runplz ps` finds containers by these labels; a drift breaks it."""
    from runplz.backends import docker

    argv = docker.label_args("myapp", "train")
    flags = docker.label_flags("myapp", "train")
    assert argv == [
        "--label",
        "runplz=1",
        "--label",
        "runplz-app=myapp",
        "--label",
        "runplz-function=train",
    ]
    for token in ("runplz=1", "runplz-app=myapp", "runplz-function=train"):
        assert token in flags


def test_ps_filter_matches_the_label_that_gets_stamped():
    from runplz.backends import docker

    assert docker.PS_FILTER == f"label={docker.RUNPLZ_LABEL}"
    assert docker.RUNPLZ_LABEL in " ".join(docker.label_args("a", "b"))


def test_label_args_omit_app_when_unknown():
    from runplz.backends import docker

    assert "runplz-app=" not in " ".join(docker.label_args(None, "train"))


def test_ps_row_parsing_is_shared_by_local_and_ssh():
    from runplz.backends import docker

    line = (
        '{"Names":"runplz-app-train","Labels":"runplz=1,runplz-app=myapp,'
        'runplz-function=train","Status":"Up 2 minutes","CreatedAt":"2026-08-26"}'
    )
    local_rows = docker.parse_ps_rows(line, backend="local")
    ssh_rows = docker.parse_ps_rows(line, backend="ssh", name_prefix="gpu.box:")
    assert local_rows[0].app == ssh_rows[0].app == "myapp"
    assert local_rows[0].function == ssh_rows[0].function == "train"
    assert local_rows[0].name == "runplz-app-train"
    assert ssh_rows[0].name == "gpu.box:runplz-app-train"


def test_ps_row_parsing_skips_garbage_rather_than_failing():
    from runplz.backends import docker

    rows = docker.parse_ps_rows("not json\n\n{}\n", backend="local")
    assert len(rows) == 1  # the empty object still yields a (blank) row


# ---------------------------------------------------------------------------
# selection: who gets asked, and who says they weren't


def test_ps_probes_an_ssh_host_even_when_a_positional_narrows_the_rest(quiet_fan_out):
    """`--host` has always been additive, independent of the positional.

    A "positional means only that backend" rule reads cleanly and loses half
    the answer: `runplz ps local --host box` is a request for both, and the
    box is the half the user cannot get any other way.
    """
    from runplz.backends import ssh as ssh_backend

    with mock.patch.object(local, "list_jobs", return_value=[]) as local_mock:
        with mock.patch.object(ssh_backend, "list_jobs", return_value=[]) as ssh_mock:
            rc = cli.main(["ps", "local", "--host", "my.box"])
    assert rc == 0
    local_mock.assert_called_once()
    ssh_mock.assert_called_once_with(host="my.box", ssh_key_path=None, port=None)


def test_ps_queries_each_comma_separated_host_separately(quiet_fan_out):
    """One call per host, so one unreachable box costs that box's rows rather
    than every box's, and the warning can name which one failed."""
    from runplz.backends import ssh as ssh_backend

    def fake(*, host, **kw):
        if host == "bad.box":
            raise RuntimeError("connection refused")
        return [JobRecord(backend="ssh", name=f"{host}:job")]

    with mock.patch.object(ssh_backend, "list_jobs", side_effect=fake):
        rc = cli.main(["ps", "--host", "good.box, bad.box"])
    assert rc == 0


def test_ps_names_the_failing_host_not_just_the_backend(capsys, quiet_fan_out):
    from runplz.backends import ssh as ssh_backend

    with mock.patch.object(ssh_backend, "list_jobs", side_effect=RuntimeError("refused")):
        cli.main(["ps", "--host", "bad.box"])
    assert "ssh:bad.box listing failed" in capsys.readouterr().err


def test_scope_for_a_fan_out_backend_does_not_recruit_it(quiet_fan_out, capsys):
    """`--region` scopes AWS, which was already going to be asked. It must not
    also be read as "and please add AWS", which would make
    `runplz ps local --region r` start querying a cloud the user narrowed away
    from.

    Nor is it silently dropped. The user narrowed the listing *and* scoped a
    backend that is not in it — two mistakes that used to cancel into silence,
    which is the failure class #20 / #143 / #153 were each about."""
    from runplz.backends import aws

    with mock.patch.object(aws, "list_jobs", return_value=[]) as aws_mock:
        with pytest.raises(SystemExit) as ei:
            cli.main(["ps", "local", "--region", "us-east-1"])
    aws_mock.assert_not_called()
    assert ei.value.code == 2
    err = capsys.readouterr().err
    assert "--region only applies to aws" in err
    assert "Listing: local" in err


def test_a_scope_flag_for_a_backend_being_listed_is_fine(quiet_fan_out):
    """The whole fan-out is listed, so `--region` reaches the AWS in it."""
    from runplz.backends import aws

    with mock.patch.object(aws, "list_jobs", return_value=[]) as aws_mock:
        cli.main(["ps", "--region", "us-east-1"])
    aws_mock.assert_called_once()


def test_a_scope_flag_that_invites_its_own_backend_is_fine(quiet_fan_out):
    """`--host` recruits ssh, so it always reaches something — including
    alongside a positional, which has always listed both."""
    from runplz.backends import ssh as ssh_backend

    with mock.patch.object(ssh_backend, "list_jobs", return_value=[]) as ssh_mock:
        cli.main(["ps", "local", "--host", "box"])
    ssh_mock.assert_called_once()


def test_an_ssh_option_without_a_host_is_refused_rather_than_dropped(quiet_fan_out, capsys):
    """`--ssh-port` without `--host` never recruits ssh, so nothing would have
    read it. Silently accepting an ssh option on a listing with no ssh in it
    tells the operator their port was applied."""
    with pytest.raises(SystemExit) as ei:
        cli.main(["ps", "local", "--ssh-port", "2222"])
    assert ei.value.code == 2
    assert "--ssh-port only applies to ssh" in capsys.readouterr().err


def test_ps_ssh_without_a_host_reports_the_missing_scope(capsys):
    """Previously `argparse: invalid choice: 'ssh'`, which said the backend
    could not be listed at all rather than naming the one thing it needed."""
    rc = cli.main(["ps", "ssh"])
    err = capsys.readouterr().err
    assert "ssh listing failed: MissingScope" in err
    assert "--host" in err
    assert rc == 1


def test_an_empty_table_admits_which_backends_went_unasked(capsys, quiet_fan_out):
    """An empty table is the one moment "nothing is running" and "nobody
    asked" look the same, and ssh jobs are invisible to a bare `runplz ps`."""
    rc = cli.main(["ps"])
    out = capsys.readouterr()
    assert "(no runplz jobs running)" in out.out
    assert "note: ssh was not listed; pass --host to include it" in out.err
    assert rc == 0


def test_the_unasked_note_is_silent_once_there_are_rows_to_show(capsys, quiet_fan_out):
    with mock.patch.object(
        local, "list_jobs", return_value=[JobRecord(backend="local", name="runplz-a-b")]
    ):
        cli.main(["ps"])
    assert "was not listed" not in capsys.readouterr().err


def test_the_unasked_note_is_silent_when_the_user_narrowed_deliberately(capsys, quiet_fan_out):
    """`runplz ps local` already says which backend the user wants; listing
    everything it isn't would be noise."""
    cli.main(["ps", "local"])
    assert "was not listed" not in capsys.readouterr().err


# ---------------------------------------------------------------------------
# table rendering


def test_ps_table_columns_come_from_the_record(capsys):
    """Headers are derived from JobRecord's fields, so this pins the rendered
    contract that derivation has to keep producing."""
    cli._print_ps_table(
        [
            JobRecord(
                backend="local",
                name="runplz-demo-train",
                app="demo",
                function="train",
                started="t",
                status="Up 5m",
            )
        ]
    )
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].split() == ["BACKEND", "NAME", "APP", "FUNCTION", "STARTED", "STATUS"]
    assert lines[1].split() == ["local", "runplz-demo-train", "demo", "train", "t", "Up", "5m"]


def test_ps_table_prints_a_null_field_as_blank_not_the_word_none(capsys):
    """Providers hand back nulls: `docker ps` omits keys and a half-created
    instance has no launch time. A column reading "None" is worse than an
    empty one — it looks like a value."""
    cli._print_ps_table([JobRecord(backend="aws", name="i-1", started=None, status=None)])
    body = capsys.readouterr().out.splitlines()[1]
    assert "None" not in body
    assert body.split() == ["aws", "i-1"]


# ---------------------------------------------------------------------------
# scope that resolves to nothing is scope the user did not supply


@pytest.mark.parametrize("raw", ["", "   ", ",", " , "])
def test_a_host_flag_naming_no_host_does_not_reach_ssh(raw, quiet_fan_out):
    """`--host ''` and `--host ,` name zero hosts.

    The pre-3.25 CLI filtered these out with `if h.strip()` and never called
    ssh. Splitting them into "one target" instead sends the empty string to
    `ssh`, which answers `Could not resolve hostname` — an error about the
    user's network rather than about their command.
    """
    from runplz.backends import ssh as ssh_backend

    with mock.patch.object(ssh_backend, "list_jobs") as ssh_mock:
        rc = cli.main(["ps", "--host", raw])
    ssh_mock.assert_not_called()
    assert rc == 0


def test_naming_ssh_with_an_empty_host_reports_the_missing_scope(capsys):
    """Asked for explicitly, ssh still has to say what it needs rather than
    quietly succeeding against a hostname of ''."""
    rc = cli.main(["ps", "ssh", "--host", ""])
    assert "ssh host is required" in capsys.readouterr().err
    assert rc == 1


def test_blank_scope_is_refused_from_the_flag_as_well_as_the_environment(monkeypatch):
    """`--region ''` and `AWS_DEFAULT_REGION=` are the same statement.

    Only the environment half was treated as unset, so the flag half reached
    the provider as `aws ec2 describe-instances --region ''`.
    """
    from runplz.backends import aws, registry
    from runplz.backends.listing import MissingScope

    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    monkeypatch.delenv("AWS_REGION", raising=False)
    with mock.patch.object(aws.subprocess, "run") as run:
        with pytest.raises(MissingScope, match="aws region is required"):
            registry.list_jobs("aws", region="")
    run.assert_not_called()


def test_surrounding_whitespace_is_stripped_from_a_host(quiet_fan_out):
    from runplz.backends import ssh as ssh_backend

    with mock.patch.object(ssh_backend, "list_jobs", return_value=[]) as ssh_mock:
        cli.main(["ps", "--host", " a.box , b.box "])
    assert [c.kwargs["host"] for c in ssh_mock.call_args_list] == ["a.box", "b.box"]


def test_ps_rejects_an_out_of_range_ssh_port(capsys):
    """The range check is declared on the field now, so this pins that the
    CLI still runs it — and still says it the same way."""
    for bad in ("0", "70000"):
        with pytest.raises(SystemExit):
            cli.main(["ps", "--ssh-port", bad])
        assert f"--ssh-port must be a valid TCP port (1-65535); got {bad}." in (
            capsys.readouterr().err
        )


def test_an_exported_region_does_not_break_a_narrowed_listing(quiet_fan_out, monkeypatch):
    """`AWS_DEFAULT_REGION` in the shell is not a claim about this command.
    Reading the environment here would make `runplz ps local` fail for anyone
    who has the variable exported, which is most AWS users."""
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    assert cli.main(["ps", "local"]) == 0


@pytest.mark.parametrize("raw", ["", "   ", ",", " , "])
def test_a_host_flag_naming_no_host_is_not_an_unreachable_scope(raw, quiet_fan_out):
    """`--host ,` names zero hosts, and "scope that resolves to nothing is
    scope the user did not supply" has to hold for this check too — otherwise
    it would reject a command the rest of the CLI treats as unscoped."""
    assert cli.main(["ps", "local", "--host", raw]) == 0
