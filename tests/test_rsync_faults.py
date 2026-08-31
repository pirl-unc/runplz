"""Rsync failures exercised through the real subprocess boundary."""

import stat

import pytest

from runplz.backends import ssh_common as sc


def _fake_rsync(bin_dir, *, mode):
    path = bin_dir / "rsync"
    path.write_text(
        "#!/bin/sh\n"
        f"mode={mode!r}\n"
        'if [ "$mode" = fail ]; then exit 12; fi\n'
        'if [ "$mode" = partial ]; then mkdir -p "$PWD/out"; '
        'printf partial > "$PWD/out/file"; exit 12; fi\n'
        "exit 0\n"
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


@pytest.mark.parametrize("mode", ["fail", "partial"])
def test_rsync_transport_failures_are_retried_and_reported(tmp_path, monkeypatch, mode):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _fake_rsync(bin_dir, mode=mode)
    monkeypatch.setenv("PATH", f"{bin_dir}:{__import__('os').environ['PATH']}")
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "input").write_text("input")
    with pytest.raises(Exception):
        sc.rsync_up(repo, "localhost")
