"""Failure tests against the real disposable sshd, not mocked subprocesses."""

import pytest

from runplz.backends import ssh_common as sc


def test_connection_refusal_is_reported(sshd_server):
    opts = sshd_server.ssh_options()
    sshd_server.refuse_connections()
    try:
        with pytest.raises(Exception):
            sc.ssh_exec(sshd_server.target, "true", ssh_opts=opts)
    finally:
        sshd_server.start()


def test_readiness_timeout_is_reported(sshd_server):
    sshd_server.refuse_connections()
    try:
        with pytest.raises(RuntimeError):
            sc.wait_until_ssh_reachable(
                sshd_server.target,
                ssh_opts=sshd_server.ssh_options(),
                max_wait_s=0,
                probe_interval_s=1,
            )
    finally:
        sshd_server.start()


def test_remote_nonzero_exit_is_real(sshd_server):
    with pytest.raises(Exception):
        sc.ssh_exec(sshd_server.target, "exit 23", ssh_opts=sshd_server.ssh_options())


def test_mid_command_transport_drop_is_reported(sshd_server):
    import threading

    timer = threading.Timer(0.2, sshd_server.drop_connection)
    timer.start()
    try:
        with pytest.raises(Exception):
            sc.ssh_exec(sshd_server.target, "sleep 5", ssh_opts=sshd_server.ssh_options())
    finally:
        timer.cancel()
        sshd_server.start()
