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

# Verbatim from the issue.
TRANSPORT_STDERR = (
    "Read from remote host: Can't assign requested address\n"
    "client_loop: send disconnect: Broken pipe"
)


def _blip():
    return subprocess.CalledProcessError(255, "ssh", stderr=TRANSPORT_STDERR)


def _remote_failure(code=1):
    return subprocess.CalledProcessError(code, "ssh", stderr="No space left on device")


# Direct tests of the loop use its `sleep=` seam. Tests that go through a
# production caller cannot reach that seam, so they take `fast_sleep` — kept
# narrow deliberately, because patching time.sleep process-wide turns any
# wall-clock loop reached during the test into a busy-loop.
def _nap(_s):
    return None


class _FakeClock:
    """Stands in for the `time` module inside ssh_common only.

    Patching `sc.time.sleep` set the attribute on the *global* time module —
    `ssh_common.time is time` — so every wall-clock loop reached during a
    test became a busy-loop. Replacing ssh_common's reference instead keeps
    the patch where it was advertised, and lets a test advance the clock.
    """

    def __init__(self):
        self.now = 0.0
        self.slept = []

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds

    def monotonic(self):
        return self.now

    def strftime(self, *a, **kw):
        import time as _time

        return _time.strftime(*a, **kw)

    def gmtime(self, *a, **kw):
        import time as _time

        return _time.gmtime(*a, **kw)


@pytest.fixture
def fast_sleep():
    """For tests that go through a production caller and cannot reach the
    loop's `sleep=` seam."""
    clock = _FakeClock()
    with mock.patch.object(sc, "time", clock):
        yield clock


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


def test_rsync_transport_codes_include_the_one_a_drop_produces():
    """12 is what a mid-transfer connection drop reports. 30 needs a
    --timeout runplz never passes and 35 is daemon-mode only, so without 12
    the rsync retry would cover nothing it exists for."""
    assert sc.SSH_TRANSPORT_EXIT in sc.RSYNC_TRANSPORT_EXITS
    assert 12 in sc.RSYNC_TRANSPORT_EXITS


# ---------------------------------------------------------------------------
# the loop


def test_a_blip_is_retried_and_the_operation_completes():
    attempts = {"n": 0}

    def flaky():
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise _blip()
        return "done"

    assert sc.retry_on_transport_failure(flaky, label="x", sleep=_nap) == "done"
    assert attempts["n"] == 2


def test_a_genuine_remote_failure_is_surfaced_immediately():
    """Retrying a full disk just delays the error the user needs to see."""
    attempts = {"n": 0}

    def failing():
        attempts["n"] += 1
        raise _remote_failure()

    with pytest.raises(subprocess.CalledProcessError):
        sc.retry_on_transport_failure(failing, label="x", sleep=_nap)
    assert attempts["n"] == 1


def test_the_budget_is_bounded():
    attempts = {"n": 0}

    def always_blips():
        attempts["n"] += 1
        raise _blip()

    with pytest.raises(subprocess.CalledProcessError):
        sc.retry_on_transport_failure(always_blips, label="x", sleep=_nap)
    assert attempts["n"] == len(sc.SSH_PREP_POLICY.waits)


def test_the_original_failure_is_what_surfaces():
    def always_blips():
        raise _blip()

    with pytest.raises(subprocess.CalledProcessError) as ei:
        sc.retry_on_transport_failure(always_blips, label="x", sleep=_nap)
    assert ei.value.returncode == 255
    assert "Broken pipe" in ei.value.stderr


def test_backoff_is_honoured():
    slept = []

    def always_blips():
        raise _blip()

    with pytest.raises(subprocess.CalledProcessError):
        sc.retry_on_transport_failure(always_blips, label="x", sleep=slept.append)
    assert slept == [w for w in sc.SSH_PREP_POLICY.waits if w]


def test_rsync_transport_codes_are_retriable_when_asked():
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


def test_an_rsync_code_is_not_retriable_for_a_plain_ssh_call():
    """rsync's 12 means nothing to ssh; only the caller knows which is which."""
    attempts = {"n": 0}

    def failing():
        attempts["n"] += 1
        raise subprocess.CalledProcessError(12, "ssh")

    with pytest.raises(subprocess.CalledProcessError):
        sc.retry_on_transport_failure(failing, label="x", sleep=_nap)
    assert attempts["n"] == 1


# ---------------------------------------------------------------------------
# the call that actually failed


def test_prepare_remote_run_survives_the_blip_from_the_issue(fast_sleep, remote_run):
    calls = {"n": 0}

    def flaky_ssh(target, cmd, *, ssh_opts=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _blip()

    with mock.patch.object(sc, "ssh_exec", flaky_ssh):
        sc._prepare_remote_run("box", remote_run, manifest={"run_id": remote_run.run_id})
    assert calls["n"] == 2, "the run died on a transport blip it could have survived"


def test_prepare_remote_run_still_fails_on_a_real_error(fast_sleep, remote_run):
    with mock.patch.object(sc, "ssh_exec", side_effect=_remote_failure()):
        with pytest.raises(subprocess.CalledProcessError):
            sc._prepare_remote_run("box", remote_run, manifest={})


def test_ensure_remote_rsync_survives_a_blip(fast_sleep):
    calls = {"n": 0}

    def flaky_ssh(target, cmd, *, ssh_opts=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _blip()

    with mock.patch.object(sc, "ssh_exec", flaky_ssh):
        sc._ensure_remote_rsync("box")
    assert calls["n"] == 2


def test_rsync_up_retries_a_half_transferred_tree(fast_sleep, tmp_path, remote_run):
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


def _launch_attempts(probe_stdout, *, remote_run, probe_returncode=0, probe_raises=None):
    """Launch attempts made by the *production* path against this remote state.

    Drives launch_detached_and_wait rather than rebuilding its
    retry_on_transport_failure call: a test that reconstructs the guard
    inline still passes when the guard in the source is inverted.
    """
    launches = {"n": 0}

    def always_blips(target, cmd, *, ssh_opts=None):
        launches["n"] += 1
        raise _blip()

    probe = mock.Mock(returncode=probe_returncode, stdout=probe_stdout, stderr="")
    run_patch = (
        mock.patch.object(sc.subprocess, "run", side_effect=probe_raises)
        if probe_raises
        else mock.patch.object(sc.subprocess, "run", return_value=probe)
    )
    with mock.patch.object(sc, "ssh_exec", always_blips), run_patch:
        with mock.patch.object(sc, "time", _FakeClock()):
            with pytest.raises(subprocess.CalledProcessError):
                sc.launch_detached_and_wait(
                    target="box", wrapped_command="train", remote_run=remote_run
                )
    return launches["n"]


def test_a_launch_is_retried_when_the_remote_shows_nothing_landed(remote_run):
    assert _launch_attempts("clear\n", remote_run=remote_run) == len(sc.SSH_PREP_POLICY.waits)


def test_a_launch_is_never_retried_once_anything_landed(remote_run):
    """A second launch is a second training job on the same GPU. The probe
    answers 'landed' for the claim marker, the pid file, or a start event."""
    assert _launch_attempts("landed\n", remote_run=remote_run) == 1


@pytest.mark.parametrize(
    "kwargs,why",
    [
        ({"probe_returncode": 255, "probe_stdout": ""}, "the box is unreachable"),
        ({"probe_returncode": 1, "probe_stdout": ""}, "the probe itself failed"),
        ({"probe_stdout": "", "probe_raises": subprocess.TimeoutExpired("ssh", 30)}, "it hung"),
    ],
)
def test_a_launch_is_never_retried_when_the_probe_cannot_answer(kwargs, why, remote_run):
    """Declining costs one failed run; a wrong retry costs a duplicate job."""
    assert _launch_attempts(remote_run=remote_run, **kwargs) == 1, why


@pytest.mark.parametrize(
    "stdout,returncode,expect",
    [
        ("clear\n", 0, False),
        ("landed\n", 0, True),
        ("", 1, True),
        ("", 255, True),
    ],
)
def test_detached_run_started_reads_the_probe(stdout, returncode, expect):
    probe = mock.Mock(returncode=returncode, stdout=stdout, stderr="")
    with mock.patch.object(sc.subprocess, "run", return_value=probe):
        assert sc.detached_run_started("box", "$HOME/m/bootstrap.pid", "$HOME/m/e") is expect


def test_detached_run_started_asks_about_the_claim_marker():
    """The window the marker exists to close."""
    probe = mock.Mock(returncode=0, stdout="clear\n", stderr="")
    with mock.patch.object(sc.subprocess, "run", return_value=probe) as run_mock:
        sc.detached_run_started("box", "$HOME/m/bootstrap.pid", "$HOME/m/events.ndjson")
    sent = run_mock.call_args.args[0][-1]
    assert sc.LAUNCH_CLAIM_FILENAME in sent
    assert "bootstrap.pid" in sent
    assert "remote_command_start" in sent


def _container_start_attempts(probe_returncode, *, remote_run, stderr=""):
    """Start attempts made by the production container path."""
    starts = {"n": 0}

    def always_blips(target, cmd, *, ssh_opts=None):
        starts["n"] += 1
        raise _blip()

    function = mock.Mock(env={})
    function.name = "train"
    stdout = "/c\n" if probe_returncode == 0 else ""
    probe = mock.Mock(returncode=probe_returncode, stdout=stdout, stderr=stderr)
    with mock.patch.object(sc, "ssh_exec", always_blips):
        with mock.patch.object(sc.subprocess, "run", return_value=probe):
            with mock.patch.object(sc, "time", _FakeClock()):
                with pytest.raises(subprocess.CalledProcessError):
                    sc._run_container_detached(
                        target="box",
                        container_name="runplz-train-abc",
                        function=function,
                        rel_script="job.py",
                        args=[],
                        kwargs={},
                        gpu_flag="",
                        app_name="app",
                        remote_run=remote_run,
                    )
    return starts["n"]


def test_container_start_is_retried_when_no_container_landed(remote_run):
    attempts = _container_start_attempts(
        1, remote_run=remote_run, stderr="Error: No such object: c"
    )
    assert attempts == len(sc.SSH_PREP_POLICY.waits)


def test_container_start_refuses_when_docker_itself_is_unhappy(remote_run):
    """A dockerd hiccup is not evidence that nothing landed."""
    attempts = _container_start_attempts(
        1, remote_run=remote_run, stderr="Cannot connect to the Docker daemon"
    )
    assert attempts == 1


def test_container_start_never_starts_a_second_container(remote_run):
    assert _container_start_attempts(0, remote_run=remote_run) == 1


def test_container_start_refuses_when_the_probe_is_unreachable(remote_run):
    assert _container_start_attempts(255, remote_run=remote_run) == 1


# ---------------------------------------------------------------------------
# brev's existing-instance path, which is where this was hit


def test_brev_existing_instance_dispatch_survives_a_prepare_blip(fast_sleep, tmp_path):
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
    # Count only the calls carrying the prepare payload: dispatch issues many
    # ssh_exec calls, and _record_remote_event swallows exceptions, so a bare
    # ">= 2" would stay green even if the blip were absorbed elsewhere.
    prepare_calls = {"n": 0}

    def flaky_ssh(target, cmd, *, ssh_opts=None):
        if "__RUNPLZ_MANIFEST__" not in cmd:
            return
        prepare_calls["n"] += 1
        if prepare_calls["n"] == 1:
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
            brev.run(app, train, [], {}, instance="my-box")

    assert prepare_calls["n"] == 2, "the blip should have been retried exactly once"


# ---------------------------------------------------------------------------
# the wall-clock budget
#
# Every other test fakes sleep, so time.monotonic never advances and the
# deadline is never reached — it needs a clock that actually moves.


def test_an_exhausted_budget_stops_further_attempts():
    """Otherwise a 4-attempt policy quadruples the worst case of a slow step."""
    clock = _FakeClock()
    attempts = {"n": 0}

    def always_blips():
        attempts["n"] += 1
        clock.now += 200  # each attempt is slow
        raise _blip()

    with mock.patch.object(sc, "time", clock):
        with pytest.raises(subprocess.CalledProcessError):
            sc.retry_on_transport_failure(
                always_blips, label="x", policy=sc.SSH_PREP_POLICY, sleep=clock.sleep
            )
    # deadline_s=300, so the third attempt is never started.
    assert attempts["n"] == 2
    assert attempts["n"] < len(sc.SSH_PREP_POLICY.waits)


def test_the_budget_is_checked_before_the_backoff_is_burned():
    """Sleeping 10s only to discover the budget was already spent is waste."""
    clock = _FakeClock()

    def always_blips():
        clock.now += 400
        raise _blip()

    with mock.patch.object(sc, "time", clock):
        with pytest.raises(subprocess.CalledProcessError):
            sc.retry_on_transport_failure(
                always_blips, label="x", policy=sc.SSH_PREP_POLICY, sleep=clock.sleep
            )
    assert clock.slept == [], "a spent budget should not pay for a wait first"


def test_the_results_pull_uses_a_shorter_budget_than_staging():
    """It runs after the job, often just before a paid box goes away."""
    assert sc.SSH_RESULTS_POLICY.deadline_s < sc.SSH_PREP_POLICY.deadline_s
    assert sum(sc.SSH_RESULTS_POLICY.waits) < sum(sc.SSH_PREP_POLICY.waits)


def test_the_launch_claim_is_written_before_the_spawn(remote_run):
    """The pid can only be recorded after spawning, which leaves a window in
    which an already-running job looks like one that never started."""
    lines = sc.build_detached_launcher(remote_run, "train").splitlines()
    claim = next(i for i, ln in enumerate(lines) if sc.LAUNCH_CLAIM_FILENAME in ln)
    spawn = next(i for i, ln in enumerate(lines) if "nohup" in ln)
    pid = next(i for i, ln in enumerate(lines) if sc.BOOTSTRAP_PID_FILENAME in ln)
    assert claim < spawn < pid


def test_a_claimed_launch_is_never_retried(remote_run):
    """Even with no pid file and no start event — the window this closes."""
    launches = {"n": 0}

    def always_blips(target, cmd, *, ssh_opts=None):
        launches["n"] += 1
        raise _blip()

    landed = mock.Mock(returncode=0, stdout="landed\n", stderr="")
    with mock.patch.object(sc, "ssh_exec", always_blips):
        with mock.patch.object(sc.subprocess, "run", return_value=landed):
            with mock.patch.object(sc, "time", _FakeClock()):
                with pytest.raises(subprocess.CalledProcessError):
                    sc.launch_detached_and_wait(
                        target="box", wrapped_command="train", remote_run=remote_run
                    )
    assert launches["n"] == 1
