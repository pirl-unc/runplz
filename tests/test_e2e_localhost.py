"""End-to-end over a real ssh connection. Costs nothing, needs no setup.

Every other test in this suite mocks `subprocess`, so it verifies that
runplz *builds* the commands its author expected -- never that those
commands do the right thing on a machine. That gap is where the expensive
bugs have lived: a NUL byte in generated shell, `container_exists` failing
open, a pgid-based kill that could not work because bash disables job
control in non-interactive shells, an env key that aborts the run under
`set -euo pipefail`. A string assertion cannot see any of them.

These drive the real staging, launch, kill and fetch-back paths against a
private sshd the fixture starts on loopback (see sshd_harness.py), so they
run on a laptop with Remote Login off and in CI alike.

Docker and the native path are deliberately out of scope: the native path
apt-gets a toolchain and installs torch, which is not something a test tier
should do. What is covered here is the shell-level plumbing -- reachable
directly because those stages became public API in 3.20.0.
"""

import uuid
from pathlib import Path

import pytest

from runplz.backends import ssh_common as sc

pytestmark = pytest.mark.live_ssh


@pytest.fixture
def target(sshd_server):
    """Address the box exactly as the code under test will.

    Includes the user when the backend needs one — the container's only
    account is root, and `SshOptions` has no user field.
    """
    return sshd_server.target


@pytest.fixture
def opts(sshd_server):
    return sshd_server.ssh_options()


@pytest.fixture
def remote_is_linux(target, opts):
    """Whether the box under test detaches the way production boxes do.

    macOS `nohup` refuses to detach in a non-interactive ssh session
    ("can't detach from console: Inappropriate ioctl for device"), and
    macOS has no `setsid`, so detached launch does not work there at all --
    issue #92. Every documented remote is Linux, so the detached tests run
    where it matters and skip rather than reporting a runplz bug that only
    exists on the harness's platform.

    Start Docker and this stops skipping anywhere: the container backend
    makes the remote Linux regardless of the host.
    """
    return "Linux" in sc.ssh_capture(target, "uname -s", ssh_opts=opts)


@pytest.fixture
def remote_run(target, opts):
    """A per-test run context, removed from the remote afterwards."""
    ctx = sc.make_remote_run_context(
        backend="ssh", target=target, function_name=f"t{uuid.uuid4().hex[:6]}"
    )
    yield ctx
    sc.ssh_capture(target, f"rm -rf {ctx.run_root_shell}", ssh_opts=opts)


def _wrapped(command: str, remote_run) -> str:
    """Wrap a command the way every production caller does.

    `launch_detached_and_wait`'s parameter is literally named
    `wrapped_command`: the caller is expected to have run the command
    through the logging wrapper, which is what records the start event and
    the exit code. Passing a raw command produces a run that starts, exits,
    and is then correctly reported as "never started" -- there is no start
    event and no exit-code file for the reader to find.
    """
    return sc._wrap_remote_command_for_logging(command, remote_run)


def _run_files(remote_run):
    """The same three paths `launch_detached_and_wait` derives internally."""
    meta = remote_run.meta_shell
    return {
        "pid_file": f"{meta}/{sc.BOOTSTRAP_PID_FILENAME}",
        "log_file": remote_run.last_log_shell,
        "events_file": remote_run.events_shell,
    }


def _local_repo(tmp_path) -> Path:
    repo = tmp_path / "repo"
    (repo / "sub").mkdir(parents=True)
    (repo / "train.py").write_text("print('hi')\n")
    (repo / "sub" / "data.txt").write_text("payload\n")
    (repo / ".env").write_text("SECRET=leaked\n")
    (repo / "id_rsa").write_text("PRIVATE KEY\n")
    return repo


def test_the_harness_really_is_a_live_connection(target, opts):
    """Guard the harness itself, so a broken fixture cannot fake a pass."""
    assert "E2E_OK" in sc.ssh_capture(target, "echo E2E_OK", ssh_opts=opts)


def test_prepare_remote_run_creates_the_layout_it_documents(target, opts, remote_run):
    sc.prepare_remote_run(target, remote_run, manifest={}, ssh_opts=opts)
    for path in (remote_run.run_root_shell, remote_run.out_shell, remote_run.meta_shell):
        probe = sc.ssh_capture(target, f"test -d {path} && echo YES", ssh_opts=opts)
        assert "YES" in probe, f"{path} was not created"


def test_rsync_up_transfers_the_repo_and_honors_secret_excludes(tmp_path, target, opts, remote_run):
    """The exclude list is a security control, checked against real rsync.

    Asserting on the rsync argv cannot tell you whether rsync honored the
    pattern -- only running it can.
    """
    repo = _local_repo(tmp_path)
    sc.prepare_remote_run(target, remote_run, manifest={}, ssh_opts=opts)
    sc.rsync_up(repo, target, remote_run=remote_run, ssh_opts=opts)

    listing = sc.ssh_capture(target, f"ls -a {remote_run.repo_shell}", ssh_opts=opts)
    assert "train.py" in listing
    assert ".env" not in listing, "a secret-shaped file reached the remote"
    assert "id_rsa" not in listing, "a private key reached the remote"

    nested = sc.ssh_capture(
        target,
        f"cat {remote_run.repo_shell}/sub/data.txt",
        ssh_opts=opts,
    )
    assert "payload" in nested


def test_a_detached_run_reports_its_real_exit_code(
    tmp_path, target, opts, remote_run, remote_is_linux
):
    """The detachment contract: the job outlives the ssh that started it."""
    if not remote_is_linux:
        pytest.skip(
            "ENVIRONMENT_UNAVAILABLE: detached launch is broken on a macOS remote — issue #92"
        )
    sc.prepare_remote_run(target, remote_run, manifest={}, ssh_opts=opts)

    code = sc.launch_detached_and_wait(
        target=target,
        wrapped_command=_wrapped("sleep 0.5; echo done; exit 7", remote_run),
        remote_run=remote_run,
        ssh_opts=opts,
    )
    assert code == 7, "the remote exit code must come back verbatim"


def test_kill_actually_stops_a_running_remote_process(target, opts, remote_run, remote_is_linux):
    """`runplz kill` is pure generated shell and has broken twice.

    The pgid approach shipped broken because bash disables job control in
    non-interactive shells, so the recorded pgid was the launching shell's;
    a later revision carried a literal NUL byte. Neither is visible without
    running the script against a real process.
    """
    if not remote_is_linux:
        pytest.skip(
            "ENVIRONMENT_UNAVAILABLE: detached launch is broken on a macOS remote — issue #92"
        )
    sc.prepare_remote_run(target, remote_run, manifest={}, ssh_opts=opts)
    files = _run_files(remote_run)
    # Launch, but do not wait: this test is about killing a live process.
    launcher = sc.build_detached_launcher(remote_run, _wrapped("sleep 300", remote_run))
    sc.ssh_exec(target, launcher, ssh_opts=opts)
    assert sc.wait_for_detached_start(
        target, files["pid_file"], files["events_file"], ssh_opts=opts
    )

    before = sc.inspect_detached_run(target, files["pid_file"], ssh_opts=opts)
    assert before.process_state is sc.DetachedProcessState.RUNNING, (
        f"expected a live process before the kill, got {before.process_state}"
    )

    kill_script = sc.build_kill_command(
        remote_run.meta_shell,
        run_id=remote_run.run_id,
        timeout_s=10,
    )
    sc.ssh_capture(target, kill_script, ssh_opts=opts)

    after = sc.inspect_detached_run(target, files["pid_file"], ssh_opts=opts)
    assert after.process_state is not sc.DetachedProcessState.RUNNING, (
        f"the process survived the kill: {after.process_state}"
    )


def test_rsync_down_brings_outputs_back(tmp_path, target, opts, remote_run):
    repo = _local_repo(tmp_path)
    host_out = tmp_path / "out"
    host_out.mkdir()
    sc.prepare_remote_run(target, remote_run, manifest={}, ssh_opts=opts)
    sc.rsync_up(repo, target, remote_run=remote_run, ssh_opts=opts)

    sc.ssh_capture(
        target,
        f"echo result > {remote_run.out_shell}/weights.txt",
        ssh_opts=opts,
    )
    sc.rsync_down(target, host_out, remote_run=remote_run, ssh_opts=opts)

    landed = host_out / "weights.txt"
    assert landed.exists(), f"outputs did not come back: {list(host_out.rglob('*'))}"
    assert landed.read_text().strip() == "result"


def test_preconditions_probe_parses_a_real_machine(target, opts):
    """`parse_probe_sections` against real uname/df output, not a fixture."""
    sc.check_preconditions(target, {"disk_free_gb": 0.001}, ssh_opts=opts)


def test_wait_until_ssh_reachable_succeeds_against_a_live_host(target, opts):
    sc.wait_until_ssh_reachable(target, ssh_opts=opts, max_wait_s=30)
