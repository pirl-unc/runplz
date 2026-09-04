"""Failure tests against the real disposable sshd, not mocked subprocesses.

The `live_ssh` marker is load-bearing, not decoration. Without it the billing
guard in conftest intercepts every `ssh` call and raises
`RuntimeError("... tried to run ssh for real")` — which satisfied a bare
`pytest.raises(Exception)` in all four tests below, so they passed for years
without the daemon this module exists to talk to (issue #96). Each assertion
here is now specific enough that the guard's error cannot stand in for the
failure under test.
"""

import subprocess
import threading

import pytest

from runplz.backends import ssh_common as sc

pytestmark = pytest.mark.live_ssh


def test_connection_refusal_is_reported(sshd_server):
    """255 is ssh's own "I could not connect", as distinct from any exit code
    the remote command might have produced."""
    opts = sshd_server.ssh_options()
    sshd_server.refuse_connections()
    try:
        with pytest.raises(subprocess.CalledProcessError) as caught:
            sc.ssh_exec(sshd_server.target, "true", ssh_opts=opts)
        assert caught.value.returncode == 255
    finally:
        sshd_server.start()


def test_readiness_timeout_is_reported(sshd_server):
    """The budget has to be non-zero or the loop never probes at all.

    With `max_wait_s=0` the deadline has already passed on entry, so this
    raised its timeout without touching the network — reporting
    `last error: ''` and passing whether or not ssh worked. Asserting on the
    recorded last error is what makes a real probe necessary.
    """
    sshd_server.refuse_connections()
    try:
        with pytest.raises(RuntimeError, match="never became reachable") as caught:
            sc.wait_until_ssh_reachable(
                sshd_server.target,
                ssh_opts=sshd_server.ssh_options(),
                max_wait_s=1,
                probe_interval_s=1,
            )
        assert "last error: ''" not in str(caught.value), (
            "the readiness loop reported a timeout without ever probing"
        )
    finally:
        sshd_server.start()


def test_remote_nonzero_exit_is_real(sshd_server):
    """The remote's exit code has to survive the transport verbatim — the
    strongest evidence that a real command ran on a real daemon, since no
    guard or mock produces a 23."""
    with pytest.raises(subprocess.CalledProcessError) as caught:
        sc.ssh_exec(sshd_server.target, "exit 23", ssh_opts=sshd_server.ssh_options())
    assert caught.value.returncode == 23


def test_mid_command_transport_drop_is_reported(sshd_server):
    """A transport that dies mid-command must surface as a transport failure
    (255), not as the command's own result.

    The drop has to kill the established session, not just the listener: an
    in-flight session is a forked child and outlives its parent, so stopping
    the listener alone let `sleep 5` run to completion.
    """
    timer = threading.Timer(0.4, sshd_server.drop_connection)
    timer.start()
    try:
        with pytest.raises(subprocess.CalledProcessError) as caught:
            sc.ssh_exec(sshd_server.target, "sleep 30", ssh_opts=sshd_server.ssh_options())
        assert caught.value.returncode == 255
    finally:
        timer.cancel()
        sshd_server.start()


def test_both_backends_offer_the_same_fault_surface():
    """The tier is only evidence for the backend it actually runs against.

    `DockerSshd` had neither fault method, so this file could only ever run
    against `LocalSshd` — and it silently did not run against the container
    backend a macOS developer gets by default, or the one `e2e-container`
    exercises in CI (#141). A missing method here is an AttributeError at
    fault-injection time, which reads as a broken test rather than a gap in
    the harness.
    """
    from sshd_harness import DockerSshd, LocalSshd

    for backend in (LocalSshd, DockerSshd):
        for primitive in ("start", "stop", "refuse_connections", "drop_connection"):
            assert callable(getattr(backend, primitive, None)), f"{backend.__name__}.{primitive}"
