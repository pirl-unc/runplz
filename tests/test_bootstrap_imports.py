"""Import semantics for a dispatched job script (issue #89).

The script is loaded by path, so nothing lands on `sys.path` automatically.
What is reachable from it is a deliberate choice, not an accident, and these
tests pin all three parts of it.
"""

import subprocess
import sys
import textwrap

import pytest


def _job_tree(tmp_path):
    """A repo with a module at the root and a module beside the job script."""
    (tmp_path / "jobs").mkdir()
    (tmp_path / "helper_root.py").write_text("VALUE = 'repo-root'\n")
    (tmp_path / "jobs" / "helper_sibling.py").write_text("VALUE = 'sibling'\n")
    return tmp_path


def _run_job(tmp_path, body, out_dir="out"):
    """Dispatch a job through the real bootstrap, as a backend would."""
    (tmp_path / out_dir).mkdir(exist_ok=True)
    script = tmp_path / "jobs" / "train.py"
    script.write_text(
        textwrap.dedent(
            """
            from runplz import App, Image

            app = App("t")

            @app.function(image=Image.from_registry("ubuntu:22.04"))
            def go():
            """
        )
        + textwrap.indent(textwrap.dedent(body), "    ")
    )
    return subprocess.run(
        [sys.executable, "-m", "runplz.bootstrap"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin",
            "HOME": str(tmp_path),
            "RUNPLZ_SCRIPT": str(script),
            "RUNPLZ_FUNCTION": "go",
            "RUNPLZ_OUT": str(tmp_path / out_dir),
            "PYTHONPATH": ":".join(sys.path),
        },
    )


def test_job_can_import_a_sibling_module(tmp_path):
    """The bug: a job laid out as a directory of modules failed on dispatch."""
    _job_tree(tmp_path)
    proc = _run_job(tmp_path, "import helper_sibling\nprint(helper_sibling.VALUE)\n")
    assert proc.returncode == 0, proc.stderr
    assert "sibling" in proc.stdout


def test_job_can_still_import_from_the_repo_root(tmp_path):
    """The fix is additive. Adopting plain Python's order would break this.

    `python jobs/train.py` puts the script's directory first and the repo
    root nowhere, so inserting rather than appending would have regressed
    every job that imports from the repo root today.
    """
    _job_tree(tmp_path)
    proc = _run_job(tmp_path, "import helper_root\nprint(helper_root.VALUE)\n")
    assert proc.returncode == 0, proc.stderr
    assert "repo-root" in proc.stdout


def test_a_sibling_does_not_shadow_the_stdlib(tmp_path):
    """Appending puts the script dir after the stdlib, so the sibling is inert.

    Plain Python would let `jobs/colorsys.py` shadow `colorsys` for the whole
    run. `colorsys` specifically because it is *not* already in `sys.modules`
    when the job body executes — an already-imported module would resolve
    from the cache no matter what the path said, so the test would pass
    without exercising the ordering at all.
    """
    _job_tree(tmp_path)
    (tmp_path / "jobs" / "colorsys.py").write_text("SHOULD_NOT_SHADOW = True\n")
    proc = _run_job(
        tmp_path,
        "import sys\n"
        "assert 'colorsys' not in sys.modules, 'pick a module that is not preloaded'\n"
        "import colorsys\n"
        "print('shadowed' if hasattr(colorsys, 'SHOULD_NOT_SHADOW') else 'stdlib')\n",
    )
    assert proc.returncode == 0, proc.stderr
    assert "stdlib" in proc.stdout


def test_repo_root_wins_over_a_sibling_of_the_same_name(tmp_path):
    """Order is documented: repo root first, script dir appended."""
    _job_tree(tmp_path)
    (tmp_path / "same.py").write_text("WHERE = 'repo-root'\n")
    (tmp_path / "jobs" / "same.py").write_text("WHERE = 'sibling'\n")
    proc = _run_job(tmp_path, "import same\nprint(same.WHERE)\n")
    assert proc.returncode == 0, proc.stderr
    assert "repo-root" in proc.stdout


def test_script_dir_is_not_added_twice(tmp_path):
    """A job dispatched from its own directory must not duplicate the entry."""
    from runplz.bootstrap import _add_script_dir_to_path

    script = tmp_path / "job.py"
    script.write_text("# job\n")
    before = list(sys.path)
    try:
        _add_script_dir_to_path(str(script))
        _add_script_dir_to_path(str(script))
        assert sys.path.count(str(tmp_path.resolve())) == 1
    finally:
        sys.path[:] = before


@pytest.mark.parametrize("missing", ["no_such_module"])
def test_a_genuinely_missing_import_still_fails(tmp_path, missing):
    """The path change must not make unrelated imports mysteriously resolve."""
    _job_tree(tmp_path)
    proc = _run_job(tmp_path, f"import {missing}\n")
    assert proc.returncode != 0
    assert "No module named" in proc.stderr
