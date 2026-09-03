from pathlib import Path


def test_test_sh_runs_pytest_as_a_module_against_local_package():
    script = (Path(__file__).resolve().parents[1] / "test.sh").read_text()
    assert 'PYTHON_BIN="${PYTHON:-python}"' in script
    assert 'exec "$PYTHON_BIN" -m pytest' in script
    assert "--cov=runplz" in script


def test_deploy_sh_clears_stale_build_output_before_building():
    """Issue #136: setuptools copies sources into `build/lib` without pruning,
    so a module deleted from the tree lingers there across releases — this
    repo's held three renamed away in 3.20.0. Inert while `python -m build`
    goes via the sdist, and a shipped artifact if that ever changes."""
    script = (Path(__file__).resolve().parents[1] / "deploy.sh").read_text()
    cleanup = next(ln for ln in script.splitlines() if ln.startswith("rm -rf"))
    assert "build" in cleanup.split() and "dist" in cleanup.split()


def test_deploy_sh_leaves_the_egg_info_a_local_editable_install_needs():
    """The other half of the same decision. A legacy `pip install -e` — what
    develop.sh does — resolves through `runplz.egg-info/`, so clearing it
    during a release would break the tree of whoever is cutting it."""
    script = (Path(__file__).resolve().parents[1] / "deploy.sh").read_text()
    assert "rm -rf" in script
    assert "egg-info" not in next(ln for ln in script.splitlines() if ln.startswith("rm -rf"))
