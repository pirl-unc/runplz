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


# ---------------------------------------------------------------------------
# the e2e remote is a pytest option, not an environment variable


def test_e2e_remote_is_declared_as_a_pytest_option():
    """An env var is invisible: absent from `pytest --help`, easy to leave
    exported in a shell, and it was parsed independently in two places that
    could disagree. `--e2e-remote` is declared once and shows up in help."""
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, "-m", "pytest", "--help"], capture_output=True, text=True
    ).stdout
    assert "--e2e-remote={auto,local,docker}" in out


def test_the_retired_env_var_is_refused_rather_than_ignored():
    """A stale `RUNPLZ_E2E_REMOTE=docker` in a shell profile must not quietly
    downgrade the run to `auto` — that would test a different backend than the
    operator asked for without saying so."""
    import os
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_scripts.py", "-q"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
        env=dict(os.environ, RUNPLZ_E2E_REMOTE="docker"),
    )
    assert result.returncode != 0
    assert "no longer read" in result.stdout + result.stderr
    assert "--e2e-remote=docker" in result.stdout + result.stderr


def test_ci_selects_its_e2e_remote_explicitly():
    """Both jobs pin a backend rather than relying on auto-detection, so a
    runner that loses its sshd fails instead of silently skipping the tier."""
    workflow = (Path(__file__).resolve().parents[1] / ".github/workflows/tests.yml").read_text()
    assert "--e2e-remote=local" in workflow
    assert "--e2e-remote=docker" in workflow
    assert "RUNPLZ_E2E_REMOTE" not in workflow
