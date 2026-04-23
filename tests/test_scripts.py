from pathlib import Path


def test_test_sh_runs_pytest_as_a_module_against_local_package():
    script = (Path(__file__).resolve().parents[1] / "test.sh").read_text()
    assert "python -m pytest" in script
    assert "--cov=runplz" in script
