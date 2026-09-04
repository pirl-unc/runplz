"""Outputs survive the failure paths that were meant to salvage them (#150).

`max_runtime_seconds` is documented as a kill-switch for a wedged job — the
case where whatever the run managed to write is often the only evidence of
what went wrong. It discarded exactly that: `rsync_down` sat inside the
`try` *after* the runner, so a cap raise skipped the sync entirely, and
`exit_code` was still None so the `finally` did not even fetch the failure
tail. On a provisioning backend the box was then torn down with the results
still on it.

These drive the real `dispatch_to_target`, stubbing only the boundary calls,
so the ordering under test is the ordering that ships.
"""

from unittest import mock

import pytest

from runplz import App, Image
from runplz.backends import ssh_common as sc


def _app(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "job.py").write_text("# job\n")
    app = App("vision")

    @app.function(image=Image.from_registry("ubuntu:22.04"))
    def train():  # pragma: no cover - never executed
        pass

    fn = app.functions["train"]
    fn.module_file = str(repo / "job.py")
    app.repo_root = repo
    return app, fn


def _dispatch(tmp_path, *, runner, mode="container"):
    """Run dispatch_to_target with everything before the run stubbed out."""
    app, fn = _app(tmp_path)
    patches = {
        "prepare_remote_run": mock.DEFAULT,
        "ensure_remote_rsync": mock.DEFAULT,
        "rsync_up": mock.DEFAULT,
        "check_preconditions": mock.DEFAULT,
        "rsync_down": mock.DEFAULT,
        "fetch_failure_tail": mock.DEFAULT,
        "build_remote_run_manifest": mock.DEFAULT,
    }
    with mock.patch.multiple(sc, **patches) as mocks:
        mocks["fetch_failure_tail"].return_value = "Traceback: boom"
        with mock.patch.object(sc, "run_container_mode", side_effect=runner):
            try:
                sc.dispatch_to_target(
                    app=app,
                    function=fn,
                    args=[],
                    kwargs={},
                    target="box",
                    backend="ssh",
                    mode=mode,
                )
            except BaseException as exc:  # noqa: BLE001
                return mocks, exc
    return mocks, None


def test_a_capped_run_still_collects_its_partial_outputs(tmp_path):
    """The bug: `rsync_down` never ran when the cap fired, so a long training
    run terminated at its deadline lost every checkpoint it had written."""

    def capped(**kwargs):
        raise RuntimeError("Remote run exceeded max_runtime_seconds=5")

    mocks, raised = _dispatch(tmp_path, runner=capped)

    assert isinstance(raised, RuntimeError)
    assert "max_runtime_seconds" in str(raised)
    mocks["rsync_down"].assert_called_once()


def test_the_original_failure_is_not_replaced_by_a_sync_error(tmp_path):
    """A box reclaimed mid-run cannot sync. The user must still be told why
    the run died, not handed an rsync error naming the wrong problem."""

    def capped(**kwargs):
        raise RuntimeError("Remote run exceeded max_runtime_seconds=5")

    app, fn = _app(tmp_path)
    with mock.patch.multiple(
        sc,
        prepare_remote_run=mock.DEFAULT,
        ensure_remote_rsync=mock.DEFAULT,
        rsync_up=mock.DEFAULT,
        check_preconditions=mock.DEFAULT,
        build_remote_run_manifest=mock.DEFAULT,
        fetch_failure_tail=mock.DEFAULT,
    ):
        with (
            mock.patch.object(sc, "rsync_down", side_effect=OSError("host unreachable")),
            mock.patch.object(sc, "run_container_mode", side_effect=capped),
            pytest.raises(RuntimeError, match="max_runtime_seconds"),
        ):
            sc.dispatch_to_target(
                app=app,
                function=fn,
                args=[],
                kwargs={},
                target="box",
                backend="ssh",
                mode="container",
            )


def test_the_failure_tail_is_fetched_when_the_run_raised(tmp_path):
    """`exit_code` is still None on a raise, and the old gate was
    `exit_code is not None and exit_code != 0` — so the one path that most
    needed the remote log tail was the one that never fetched it."""

    def capped(**kwargs):
        raise RuntimeError("Remote run exceeded max_runtime_seconds=5")

    mocks, _ = _dispatch(tmp_path, runner=capped)
    mocks["fetch_failure_tail"].assert_called_once()


def test_the_tail_is_surfaced_rather_than_computed_and_dropped(tmp_path, capsys):
    """Fetching it and discarding it would be worse than not fetching: the
    raised error carries no tail, so it has to reach stdout."""

    def capped(**kwargs):
        raise RuntimeError("Remote run exceeded max_runtime_seconds=5")

    _dispatch(tmp_path, runner=capped)
    assert "Traceback: boom" in capsys.readouterr().out


def test_a_successful_run_syncs_exactly_once(tmp_path):
    """The salvage must not double-sync a healthy run."""
    mocks, raised = _dispatch(tmp_path, runner=lambda **kw: 0)
    assert raised is None
    mocks["rsync_down"].assert_called_once()


def test_a_failed_sync_on_a_successful_run_is_still_an_error(tmp_path):
    """The success path stays strict. A run that produced results and could
    not deliver them has failed, and swallowing that would lose outputs
    silently — the failure mode this whole issue is about."""
    app, fn = _app(tmp_path)
    with mock.patch.multiple(
        sc,
        prepare_remote_run=mock.DEFAULT,
        ensure_remote_rsync=mock.DEFAULT,
        rsync_up=mock.DEFAULT,
        check_preconditions=mock.DEFAULT,
        build_remote_run_manifest=mock.DEFAULT,
        fetch_failure_tail=mock.DEFAULT,
    ):
        with (
            mock.patch.object(sc, "rsync_down", side_effect=OSError("disk full")),
            mock.patch.object(sc, "run_container_mode", return_value=0),
            pytest.raises(OSError, match="disk full"),
        ):
            sc.dispatch_to_target(
                app=app,
                function=fn,
                args=[],
                kwargs={},
                target="box",
                backend="ssh",
                mode="container",
            )


def test_a_nonzero_exit_syncs_once_not_twice(tmp_path):
    """A run that fails normally already synced on the success path; the
    salvage branch must not run again for it."""
    mocks, raised = _dispatch(tmp_path, runner=lambda **kw: 3)
    assert isinstance(raised, RuntimeError)
    mocks["rsync_down"].assert_called_once()
