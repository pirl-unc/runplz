"""Retrying idempotent SSH preparation after a transport blip (issue #84).

A real run reached SSH readiness and then died inside `_prepare_remote_run`
with `Can't assign requested address` / `client_loop: send disconnect: Broken
pipe`. The remote held only `run.json` and a `launch_prepared` event — nothing
staged, no bootstrap — and the identical command succeeded on the next try.

The transport blinked. The run did not have to die. But the retry that fixes
that must never be able to start a second training job, so everything here
comes in pairs: recover when nothing landed, refuse when anything did.
"""

import subprocess
from unittest import mock

import pytest

from runplz.backends import ssh_common as sc
from runplz.backends.ssh_common import DetachedProcessState as P
from runplz.backends.ssh_common import DetachedRunStatus as S

# Verbatim from the issue.
TRANSPORT_STDERR = (
    "Read from remote host: Can't assign requested address\n"
    "client_loop: send disconnect: Broken pipe"
)


def _blip():
    return subprocess.CalledProcessError(255, "ssh", stderr=TRANSPORT_STDERR)


def _remote_failure(code=1):
    return subprocess.CalledProcessError(code, "ssh", stderr="No space left on device")


@pytest.fixture
def no_sleep():
    with mock.patch.object(sc.time, "sleep", lambda _s: None):
        yield


@pytest.fixture
def remote_run():
    return sc.make_remote_run_context(backend="brev", target="box", function_name="train")


# ---------------------------------------------------------------------------
# classification


def test_ssh_reserves_255_for_its_own_failures():
    """Which is what makes it a precise 'nothing ran remotely' signal."""
    assert sc.is_ssh_transport_failure(255)
    assert not sc.is_ssh_transport_failure(0)
    assert not sc.is_ssh_transport_failure(1)
    assert not sc.is_ssh_transport_failure(None)


def test_rsync_transport_codes_include_ssh_and_rsyncs_own():
    assert sc.SSH_TRANSPORT_EXIT in sc.RSYNC_TRANSPORT_EXITS
    for rsync_code in (12, 30, 35):
        assert rsync_code in sc.RSYNC_TRANSPORT_EXITS


# ---------------------------------------------------------------------------
# the loop


def test_a_blip_is_retried_and_the_operation_completes(no_sleep):
    attempts = {"n": 0}

    def flaky():
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise _blip()
        return "done"

    assert sc.retry_on_transport_failure(flaky, label="x") == "done"
    assert attempts["n"] == 2


def test_a_genuine_remote_failure_is_surfaced_immediately(no_sleep):
    """Retrying a full disk just delays the error the user needs to see."""
    attempts = {"n": 0}

    def failing():
        attempts["n"] += 1
        raise _remote_failure()

    with pytest.raises(subprocess.CalledProcessError):
        sc.retry_on_transport_failure(failing, label="x")
    assert attempts["n"] == 1


def test_the_budget_is_bounded(no_sleep):
    attempts = {"n": 0}

    def always_blips():
        attempts["n"] += 1
        raise _blip()

    with pytest.raises(subprocess.CalledProcessError):
        sc.retry_on_transport_failure(always_blips, label="x")
    assert attempts["n"] == len(sc.SSH_PREP_RETRY_WAITS)


def test_the_original_failure_is_what_surfaces(no_sleep):
    def always_blips():
        raise _blip()

    with pytest.raises(subprocess.CalledProcessError) as ei:
        sc.retry_on_transport_failure(always_blips, label="x")
    assert ei.value.returncode == 255
    assert "Broken pipe" in ei.value.stderr


def test_backoff_is_honoured():
    slept = []

    def always_blips():
        raise _blip()

    with pytest.raises(subprocess.CalledProcessError):
        sc.retry_on_transport_failure(always_blips, label="x", sleep=slept.append)
    assert slept == [w for w in sc.SSH_PREP_RETRY_WAITS if w]


def test_rsync_transport_codes_are_retriable_when_asked(no_sleep):
    attempts = {"n": 0}

    def flaky():
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise subprocess.CalledProcessError(12, "rsync")
        return "synced"

    assert (
        sc.retry_on_transport_failure(flaky, label="x", retriable_exits=sc.RSYNC_TRANSPORT_EXITS)
        == "synced"
    )


def test_an_rsync_code_is_not_retriable_for_a_plain_ssh_call(no_sleep):
    """rsync's 12 means nothing to ssh; only the caller knows which is which."""
    attempts = {"n": 0}

    def failing():
        attempts["n"] += 1
        raise subprocess.CalledProcessError(12, "ssh")

    with pytest.raises(subprocess.CalledProcessError):
        sc.retry_on_transport_failure(failing, label="x")
    assert attempts["n"] == 1


# ---------------------------------------------------------------------------
# the call that actually failed


def test_prepare_remote_run_survives_the_blip_from_the_issue(no_sleep, remote_run):
    calls = {"n": 0}

    def flaky_ssh(target, cmd, *, ssh_opts=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _blip()

    with mock.patch.object(sc, "ssh_exec", flaky_ssh):
        sc._prepare_remote_run("box", remote_run, manifest={"run_id": remote_run.run_id})
    assert calls["n"] == 2, "the run died on a transport blip it could have survived"


def test_prepare_remote_run_still_fails_on_a_real_error(no_sleep, remote_run):
    with mock.patch.object(sc, "ssh_exec", side_effect=_remote_failure()):
        with pytest.raises(subprocess.CalledProcessError):
            sc._prepare_remote_run("box", remote_run, manifest={})


def test_ensure_remote_rsync_survives_a_blip(no_sleep):
    calls = {"n": 0}

    def flaky_ssh(target, cmd, *, ssh_opts=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _blip()

    with mock.patch.object(sc, "ssh_exec", flaky_ssh):
        sc._ensure_remote_rsync("box")
    assert calls["n"] == 2


def test_rsync_up_retries_a_half_transferred_tree(no_sleep, tmp_path, remote_run):
    (tmp_path / "f.py").write_text("x")
    calls = {"n": 0}

    def flaky_rsync(cmd, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise subprocess.CalledProcessError(12, "rsync")

    with mock.patch.object(sc, "run_local", flaky_rsync):
        with mock.patch.object(sc, "select_source_paths", return_value=None):
            with mock.patch.object(sc, "_record_remote_event", lambda *a, **k: None):
                sc.rsync_up(tmp_path, "box", remote_run=remote_run)
    assert calls["n"] == 2


# ---------------------------------------------------------------------------
# the no-duplicate guarantee
#
# A launch is the one operation that must never run twice. Its transport
# failure is ambiguous — the launcher may have detached before the connection
# dropped — so the retry asks the remote first.


def _attempt_launch(marker_status, probe_raises=None):
    """Return how many launch attempts were made against this remote state."""
    launches = {"n": 0}

    def always_blips(target, cmd, *, ssh_opts=None):
        launches["n"] += 1
        raise _blip()

    probe = (
        mock.patch.object(sc, "inspect_detached_run", side_effect=probe_raises)
        if probe_raises
        else mock.patch.object(sc, "inspect_detached_run", return_value=marker_status)
    )
    with mock.patch.object(sc, "ssh_exec", always_blips), probe:
        with mock.patch.object(sc.time, "sleep", lambda _s: None):
            with pytest.raises(subprocess.CalledProcessError):
                sc.retry_on_transport_failure(
                    lambda: sc.ssh_exec("box", "launcher"),
                    label="launch",
                    can_retry=lambda: not sc.detached_run_started("box", "p.pid", "e.ndjson"),
                )
    return launches["n"]


def test_a_launch_is_retried_when_the_remote_shows_nothing_landed():
    assert _attempt_launch(S(P.MISSING, False, None)) == len(sc.SSH_PREP_RETRY_WAITS)


@pytest.mark.parametrize(
    "status,why",
    [
        (S(P.RUNNING, False, 4242), "a bootstrap pid exists"),
        (S(P.MISSING, True, None), "a start event was recorded"),
        (S(P.RUNNING, True, 99), "the process is alive"),
        (S(P.ZOMBIE, False, 7), "a pid was recorded even if it since died"),
        (S(P.UNKNOWN, False, None), "the state cannot be established"),
    ],
)
def test_a_launch_is_never_retried_once_anything_landed(status, why):
    """A second launch is a second training job on the same GPU."""
    assert _attempt_launch(status) == 1, why


def test_an_unreachable_probe_counts_as_landed():
    """Declining costs one failed run; a wrong retry costs a duplicate job."""
    assert _attempt_launch(None, probe_raises=OSError("network gone")) == 1


def test_detached_run_started_reads_the_markers():
    with mock.patch.object(sc, "inspect_detached_run", return_value=S(P.MISSING, False, None)):
        assert sc.detached_run_started("box", "p", "e") is False
    with mock.patch.object(sc, "inspect_detached_run", return_value=S(P.MISSING, True, None)):
        assert sc.detached_run_started("box", "p", "e") is True
    with mock.patch.object(sc, "inspect_detached_run", return_value=S(P.RUNNING, False, 1)):
        assert sc.detached_run_started("box", "p", "e") is True


def test_container_exists_guards_the_container_start():
    with mock.patch.object(sc, "ssh_capture", return_value="/runplz-train-abc\n"):
        assert sc.container_exists("box", "runplz-train-abc") is True
    with mock.patch.object(sc, "ssh_capture", return_value="\n"):
        assert sc.container_exists("box", "runplz-train-abc") is False
    with mock.patch.object(sc, "ssh_capture", side_effect=OSError("network gone")):
        assert sc.container_exists("box", "runplz-train-abc") is True


def test_container_start_refuses_to_start_a_second_container(no_sleep):
    starts = {"n": 0}

    def always_blips(target, cmd, *, ssh_opts=None):
        starts["n"] += 1
        raise _blip()

    with mock.patch.object(sc, "ssh_exec", always_blips):
        with mock.patch.object(sc, "container_exists", return_value=True):
            with pytest.raises(subprocess.CalledProcessError):
                sc.retry_on_transport_failure(
                    lambda: sc.ssh_exec("box", "docker run ..."),
                    label="start container",
                    can_retry=lambda: not sc.container_exists("box", "c"),
                )
    assert starts["n"] == 1


# ---------------------------------------------------------------------------
# brev's existing-instance path, which is where this was hit


def test_brev_existing_instance_dispatch_survives_a_prepare_blip(tmp_path, no_sleep):
    """The reported shape: an existing running box, auto-create disabled."""
    from runplz import App, BrevConfig, Image
    from runplz.backends import brev

    app = App("vision", brev_config=BrevConfig(mode="vm", on_finish="leave"))
    app._repo_root = tmp_path
    (tmp_path / "job.py").write_text("# job\n")

    @app.function(image=Image.from_registry("ubuntu:22.04"))
    def train():
        pass

    train.module_file = str(tmp_path / "job.py")

    # Fail the ssh call *inside* _prepare_remote_run, not the function
    # itself — the retry being tested lives in there.
    ssh_calls = {"n": 0}

    def flaky_ssh(target, cmd, *, ssh_opts=None):
        ssh_calls["n"] += 1
        if ssh_calls["n"] == 1:
            raise _blip()

    with mock.patch.multiple(
        "runplz.backends.brev",
        _require_brev_cli=mock.DEFAULT,
        _skip_onboarding=mock.DEFAULT,
        _instance_exists=mock.Mock(return_value=True),
        _start_instance_if_stopped=mock.DEFAULT,
        _refresh_ssh=mock.DEFAULT,
        _apply_on_finish=mock.DEFAULT,
    ):
        with mock.patch.multiple(
            "runplz.backends.ssh_common",
            wait_until_ssh_reachable=mock.DEFAULT,
            ssh_exec=flaky_ssh,
            _ensure_remote_rsync=mock.DEFAULT,
            rsync_up=mock.DEFAULT,
            _check_preconditions=mock.DEFAULT,
            _ensure_docker=mock.DEFAULT,
            _remote_has_nvidia=mock.Mock(return_value=False),
            _build_image=mock.DEFAULT,
            _run_container_detached=mock.DEFAULT,
            _stream_and_wait=mock.Mock(return_value=0),
            ssh_capture=mock.DEFAULT,
            rsync_down=mock.DEFAULT,
        ):
            with mock.patch.object(sc.time, "sleep", lambda _s: None):
                brev.run(app, train, [], {}, instance="my-box")

    assert ssh_calls["n"] >= 2, "the blip should have been retried, not fatal"
