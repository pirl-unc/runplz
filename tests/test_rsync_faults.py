"""Rsync failures exercised through the real subprocess boundary."""

import stat

import pytest

from runplz.backends import ssh_common as sc


def _fake_rsync(bin_dir, *, mode, partial_dir):
    path = bin_dir / "rsync"
    path.write_text(
        "#!/bin/sh\n"
        f"mode={mode!r}\n"
        'if [ "$mode" = fail ]; then exit 12; fi\n'
        f'if [ "$mode" = partial ]; then mkdir -p {partial_dir!s}; '
        f"printf partial > {partial_dir!s}/file; exit 12; fi\n"
        "exit 0\n"
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


@pytest.mark.parametrize("mode", ["fail", "partial"])
def test_rsync_transport_failures_are_retried_and_reported(tmp_path, sandbox_bin, mode):
    partial_dir = tmp_path / "partial"
    _fake_rsync(sandbox_bin, mode=mode, partial_dir=partial_dir)
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "input").write_text("input")
    with pytest.raises(Exception):
        sc.rsync_up(repo, "localhost")
    if mode == "partial":
        assert (partial_dir / "file").read_text() == "partial"
