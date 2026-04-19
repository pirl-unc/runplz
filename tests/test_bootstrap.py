"""In-container bootstrap tests — simulate what a backend runs inside
a container: set RUNPLZ_* env vars, then invoke runplz._bootstrap.main().
"""

import json
import textwrap
from pathlib import Path
from unittest import mock

import pytest

from runplz import _bootstrap


def _user_job(tmp_path: Path) -> Path:
    script = tmp_path / "job.py"
    script.write_text(
        textwrap.dedent("""
        from runplz import App, Image
        app = App("t")
        image = Image.from_registry("ubuntu:22.04")

        CALLS = []

        @app.function(image=image)
        def echo(x, mult=1):
            CALLS.append((x, mult))
            return x * mult

        @app.local_entrypoint()
        def main():
            echo.remote(21)
    """)
    )
    return script


def _env_for(script, function, args=None, kwargs=None):
    return {
        "RUNPLZ_SCRIPT": str(script),
        "RUNPLZ_FUNCTION": function,
        "RUNPLZ_ARGS": json.dumps(args or []),
        "RUNPLZ_KWARGS": json.dumps(kwargs or {}),
    }


def test_bootstrap_loads_and_runs_user_function(tmp_path, capsys):
    script = _user_job(tmp_path)
    env = _env_for(script, "echo", args=[7], kwargs={"mult": 6})
    with mock.patch.dict("os.environ", env, clear=False):
        _bootstrap.main()
    # Function returned 42, bootstrap emits a sentinel.
    assert "[runplz] result: 42" in capsys.readouterr().out


def test_bootstrap_missing_script_raises(tmp_path):
    env = _env_for(tmp_path / "nope.py", "echo")
    with mock.patch.dict("os.environ", env, clear=False):
        # spec_from_file_location returns None for a nonexistent path in
        # some configurations; either RuntimeError or FileNotFoundError
        # is acceptable.
        with pytest.raises((RuntimeError, FileNotFoundError)):
            _bootstrap.main()


def test_bootstrap_missing_function_raises(tmp_path):
    script = _user_job(tmp_path)
    env = _env_for(script, "no_such_function")
    with mock.patch.dict("os.environ", env, clear=False):
        with pytest.raises(RuntimeError, match="no_such_function"):
            _bootstrap.main()


def test_bootstrap_default_args_and_kwargs(tmp_path, capsys):
    script = tmp_path / "job.py"
    script.write_text(
        textwrap.dedent("""
        from runplz import App, Image
        app = App("t")

        @app.function(image=Image.from_registry("ubuntu:22.04"))
        def zero_arg():
            return "hello"

        @app.local_entrypoint()
        def main():
            zero_arg.remote()
    """)
    )
    env = {
        "RUNPLZ_SCRIPT": str(script),
        "RUNPLZ_FUNCTION": "zero_arg",
        # Intentionally omit RUNPLZ_ARGS / RUNPLZ_KWARGS — defaults apply.
    }
    with mock.patch.dict("os.environ", env, clear=False):
        _bootstrap.main()
    assert "[runplz] result: 'hello'" in capsys.readouterr().out
