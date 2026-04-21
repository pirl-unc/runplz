"""Proves the autouse guard in tests/conftest.py refuses real `brev`
invocations. Issue #35.
"""

import pytest

from runplz.backends import _ssh_common, brev


def test_guard_blocks_real_brev_ls():
    with pytest.raises(RuntimeError, match="tried to run `brev`"):
        brev.subprocess.run(["brev", "ls", "--json"], capture_output=True, text=True)


def test_guard_blocks_real_brev_create():
    with pytest.raises(RuntimeError, match="tried to run `brev`"):
        brev.subprocess.run(["brev", "create", "box", "--type", "gpu"], check=True)


def test_guard_blocks_real_rsync_via_ssh_common():
    # The guard also covers the shared SSH plumbing — that's where every
    # real rsync/ssh call lives in the 3.5+ architecture.
    with pytest.raises(RuntimeError, match="tried to run `rsync`"):
        _ssh_common.subprocess.run(["rsync", "-az", "src/", "box:dest/"], check=True)


def test_guard_blocks_real_ssh_via_ssh_common():
    with pytest.raises(RuntimeError, match="tried to run `ssh`"):
        _ssh_common.subprocess.run(["ssh", "box", "echo hi"], check=True)


def test_guard_lets_unrelated_commands_through():
    # `which` isn't in the billed set — should run (and normally succeed).
    r = brev.subprocess.run(["which", "true"], capture_output=True, text=True)
    assert r.returncode in (0, 1)


def test_guard_lets_docker_commands_through():
    # docker is intentionally not guarded — the local backend uses it for
    # real builds on the developer's machine, and the hit is bounded
    # (local resource, not billed).
    # We only verify the call reaches subprocess.run (it'll either succeed
    # or fail on `docker` being missing; either outcome is fine for this
    # test since the guard didn't trip).
    import subprocess as real

    assert brev.subprocess.run is not real.run  # guard wrapper installed
    # This just needs to not raise our RuntimeError.
    try:
        brev.subprocess.run(
            ["docker", "image", "inspect", "nonexistent:tag"],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        pass  # no docker on this host — still proves the guard didn't block it


def test_guard_subprocess_module_proxies_non_run_attributes():
    # Code that uses subprocess.TimeoutExpired / CalledProcessError /
    # DEVNULL through a guarded backend module must still work.
    assert brev.subprocess.TimeoutExpired is __import__("subprocess").TimeoutExpired
    assert brev.subprocess.CalledProcessError is __import__("subprocess").CalledProcessError
    assert brev.subprocess.DEVNULL == __import__("subprocess").DEVNULL


@pytest.mark.live_brev
def test_guard_allows_real_brev_when_opted_in():
    # The marker lets a test call brev for real. We don't actually issue
    # a billed call here; we just assert the guard's escape hatch works by
    # running `brev --version` (harmless, local-only, but still hits the
    # binary). If `brev` isn't installed on this machine the test's
    # FileNotFoundError is fine — the guard didn't raise, which is the
    # property we're verifying.
    try:
        r = brev.subprocess.run(["brev", "--version"], capture_output=True, text=True)
        assert r.returncode in (0, 1, 127)  # any non-guard result is fine
    except FileNotFoundError:
        pass
