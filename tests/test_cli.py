"""CLI entry-point coverage.

Exercises argparse parsing, script discovery, App-loading, backend
selection plumbing, and the entrypoint dispatch — without actually
touching any real backend.
"""

import textwrap
from pathlib import Path
from unittest import mock

import pytest

from runplz import _cli


def _write_job(tmp_path: Path, body: str) -> Path:
    script = tmp_path / "job.py"
    script.write_text(textwrap.dedent(body))
    return script


def test_cli_script_not_found_errors(tmp_path, capsys):
    with pytest.raises(SystemExit):
        _cli.main(["local", str(tmp_path / "does-not-exist.py")])
    err = capsys.readouterr().err
    assert "script not found" in err


def test_cli_brev_without_instance_threads_none_for_ephemeral(tmp_path):
    """3.6: `runplz brev script.py` without --instance is valid now; the
    backend interprets instance=None as ephemeral mode."""
    script = _write_job(
        tmp_path,
        """
        from runplz import App, Image

        app = App("t")
        image = Image.from_registry("ubuntu:22.04")

        @app.function(image=image)
        def fn(): pass

        @app.local_entrypoint()
        def main(): fn.remote()
        """,
    )
    captured = {}
    with mock.patch(
        "runplz.backends.brev.run",
        lambda app, function, args, kwargs, **kw: captured.update({"kw": kw}),
    ):
        _cli.main(["brev", str(script)])
    assert captured["kw"]["instance"] is None


def test_cli_rejects_script_with_no_app(tmp_path):
    script = _write_job(tmp_path, "x = 1\n")
    with pytest.raises(SystemExit, match="No App found"):
        _cli.main(["local", str(script)])


def test_cli_rejects_script_with_multiple_apps(tmp_path):
    script = _write_job(
        tmp_path,
        """
        from runplz import App
        a = App("one")
        b = App("two")
        """,
    )
    with pytest.raises(SystemExit, match="Multiple Apps found"):
        _cli.main(["local", str(script)])


def test_cli_rejects_script_with_multiple_functions_and_no_entrypoint(tmp_path, capsys):
    """A single @app.function auto-runs as the entrypoint (3.14.0+), but
    multiple functions remain ambiguous and must be explicit."""
    script = _write_job(
        tmp_path,
        """
        from runplz import App, Image
        app = App("t")
        image = Image.from_registry("ubuntu:22.04")
        @app.function(image=image)
        def first(): pass
        @app.function(image=image)
        def second(): pass
        """,
    )
    with pytest.raises(SystemExit):
        _cli.main(["local", str(script)])
    err = capsys.readouterr().err
    assert "@app.local_entrypoint" in err
    assert "first" in err and "second" in err


def test_cli_rejects_script_with_no_function_and_no_entrypoint(tmp_path, capsys):
    script = _write_job(
        tmp_path,
        """
        from runplz import App
        app = App("t")
        """,
    )
    with pytest.raises(SystemExit):
        _cli.main(["local", str(script)])
    assert "declares no @app.function" in capsys.readouterr().err


def test_cli_local_no_build_sets_flag(tmp_path):
    script = _write_job(
        tmp_path,
        """
        from runplz import App, Image

        app = App("t")
        image = Image.from_registry("ubuntu:22.04")

        @app.function(image=image)
        def fn():
            return "ok"

        @app.local_entrypoint()
        def main():
            fn.remote()
        """,
    )
    # Mock out the local backend so we only inspect what kwargs it got.
    captured = {}

    def fake_local_run(app, function, args, kwargs, **backend_kwargs):
        captured["backend_kwargs"] = backend_kwargs
        return None

    with mock.patch("runplz.backends.local.run", fake_local_run):
        _cli.main(["local", "--no-build", str(script)])

    assert captured["backend_kwargs"]["build"] is False
    assert captured["backend_kwargs"]["outputs_dir"] == "out"


def test_cli_outputs_dir_flag_is_forwarded(tmp_path):
    script = _write_job(
        tmp_path,
        """
        from runplz import App, Image

        app = App("t")
        image = Image.from_registry("ubuntu:22.04")

        @app.function(image=image)
        def fn(): pass

        @app.local_entrypoint()
        def main(): fn.remote()
        """,
    )
    captured = {}
    with mock.patch(
        "runplz.backends.local.run",
        lambda app, function, args, kwargs, **kw: captured.update({"kw": kw}),
    ):
        _cli.main(["local", "--outputs-dir", "artifacts", str(script)])
    assert captured["kw"]["outputs_dir"] == "artifacts"


def test_cli_ssh_requires_host(tmp_path, capsys):
    script = _write_job(
        tmp_path,
        """
        from runplz import App, Image

        app = App("t")
        image = Image.from_registry("ubuntu:22.04")

        @app.function(image=image)
        def fn(): pass

        @app.local_entrypoint()
        def main(): fn.remote()
        """,
    )
    with pytest.raises(SystemExit):
        _cli.main(["ssh", str(script)])
    assert "--host is required" in capsys.readouterr().err


def test_cli_ssh_threads_host_into_backend_kwargs(tmp_path):
    script = _write_job(
        tmp_path,
        """
        from runplz import App, Image

        app = App("t")
        image = Image.from_registry("ubuntu:22.04")

        @app.function(image=image)
        def fn(): pass

        @app.local_entrypoint()
        def main(): fn.remote()
        """,
    )
    captured = {}
    with mock.patch(
        "runplz.backends.ssh.run",
        lambda app, function, args, kwargs, **kw: captured.update({"kw": kw}),
    ):
        _cli.main(["ssh", "--host", "gpu.example.com", str(script)])
    assert captured["kw"]["host"] == "gpu.example.com"


def test_cli_host_rejected_on_non_ssh_backend(tmp_path, capsys):
    script = _write_job(
        tmp_path,
        """
        from runplz import App, Image

        app = App("t")
        image = Image.from_registry("ubuntu:22.04")

        @app.function(image=image)
        def fn(): pass

        @app.local_entrypoint()
        def main(): fn.remote()
        """,
    )
    with pytest.raises(SystemExit):
        _cli.main(["local", "--host", "x", str(script)])
    assert "--host only applies" in capsys.readouterr().err


def test_cli_modal_dispatches_to_modal_backend(tmp_path):
    script = _write_job(
        tmp_path,
        """
        from runplz import App, Image

        app = App("t")
        image = Image.from_registry("ubuntu:22.04")

        @app.function(image=image)
        def fn(): pass

        @app.local_entrypoint()
        def main(): fn.remote()
        """,
    )
    called = []
    with mock.patch("runplz.backends.modal.run", lambda *a, **kw: called.append(True)):
        _cli.main(["modal", str(script)])
    assert called == [True]


# -- entrypoint pass-through args (issue #31) ----------------------------


def test_cli_passes_typed_kwargs_to_entrypoint(tmp_path):
    """@local_entrypoint def main(steps: int = 100, dataset: str = 'small'):
    → CLI `--steps=1000 --dataset=big` maps to main(steps=1000, dataset='big')."""
    script = _write_job(
        tmp_path,
        """
        from runplz import App, Image

        app = App("t")
        image = Image.from_registry("ubuntu:22.04")

        @app.function(image=image)
        def fn(): pass

        received = {}

        @app.local_entrypoint()
        def main(steps: int = 100, dataset: str = "small", lr: float = 1e-3):
            received["steps"] = steps
            received["dataset"] = dataset
            received["lr"] = lr
        """,
    )
    called = {}
    # Stash `received` into our test scope by mocking local.run (which never
    # runs here since main() doesn't touch fn). Instead read it from the
    # loaded module after the CLI call.
    import sys as _sys

    with mock.patch("runplz.backends.local.run", lambda *a, **kw: None):
        _cli.main(["local", str(script), "--steps=1000", "--dataset=big", "--lr=0.01"])
    mod = _sys.modules["_runplz_user_job"]
    called.update(mod.received)
    assert called == {"steps": 1000, "dataset": "big", "lr": 0.01}


def test_cli_entrypoint_uses_defaults_when_args_omitted(tmp_path):
    script = _write_job(
        tmp_path,
        """
        from runplz import App, Image

        app = App("t")
        image = Image.from_registry("ubuntu:22.04")

        @app.function(image=image)
        def fn(): pass

        seen = {}

        @app.local_entrypoint()
        def main(steps: int = 42):
            seen["steps"] = steps
        """,
    )
    import sys as _sys

    with mock.patch("runplz.backends.local.run", lambda *a, **kw: None):
        _cli.main(["local", str(script)])
    assert _sys.modules["_runplz_user_job"].seen == {"steps": 42}


def test_cli_entrypoint_missing_required_arg_errors(tmp_path, capsys):
    script = _write_job(
        tmp_path,
        """
        from runplz import App, Image

        app = App("t")
        image = Image.from_registry("ubuntu:22.04")

        @app.function(image=image)
        def fn(): pass

        @app.local_entrypoint()
        def main(dataset: str):
            pass
        """,
    )
    with pytest.raises(SystemExit):
        _cli.main(["local", str(script)])
    err = capsys.readouterr().err
    assert "--dataset" in err


def test_cli_entrypoint_type_mismatch_errors(tmp_path, capsys):
    script = _write_job(
        tmp_path,
        """
        from runplz import App, Image

        app = App("t")
        image = Image.from_registry("ubuntu:22.04")

        @app.function(image=image)
        def fn(): pass

        @app.local_entrypoint()
        def main(steps: int = 1):
            pass
        """,
    )
    with pytest.raises(SystemExit):
        _cli.main(["local", str(script), "--steps=not-a-number"])
    err = capsys.readouterr().err
    assert "--steps" in err
    assert "int" in err


def test_cli_entrypoint_bool_flag_forms(tmp_path):
    """--flag = True, --no-flag = False, --flag=true/false explicit."""
    script = _write_job(
        tmp_path,
        """
        from runplz import App, Image

        app = App("t")
        image = Image.from_registry("ubuntu:22.04")

        @app.function(image=image)
        def fn(): pass

        seen = {}

        @app.local_entrypoint()
        def main(verbose: bool = False):
            seen["verbose"] = verbose
        """,
    )
    import sys as _sys

    for argv, expected in [
        (["local", str(script)], False),
        (["local", str(script), "--verbose"], True),
        (["local", str(script), "--no-verbose"], False),
        (["local", str(script), "--verbose=true"], True),
        (["local", str(script), "--verbose=no"], False),
    ]:
        with mock.patch("runplz.backends.local.run", lambda *a, **kw: None):
            _cli.main(argv)
        assert _sys.modules["_runplz_user_job"].seen["verbose"] is expected, argv


def test_cli_entrypoint_optional_annotation_unwrapped(tmp_path):
    """Optional[int] should coerce to int, not choke on None default."""
    script = _write_job(
        tmp_path,
        """
        from typing import Optional
        from runplz import App, Image

        app = App("t")
        image = Image.from_registry("ubuntu:22.04")

        @app.function(image=image)
        def fn(): pass

        seen = {}

        @app.local_entrypoint()
        def main(steps: Optional[int] = None):
            seen["steps"] = steps
        """,
    )
    import sys as _sys

    with mock.patch("runplz.backends.local.run", lambda *a, **kw: None):
        _cli.main(["local", str(script), "--steps=7"])
    assert _sys.modules["_runplz_user_job"].seen == {"steps": 7}

    with mock.patch("runplz.backends.local.run", lambda *a, **kw: None):
        _cli.main(["local", str(script)])
    assert _sys.modules["_runplz_user_job"].seen == {"steps": None}


def test_cli_zero_arg_entrypoint_rejects_extras(tmp_path, capsys):
    """An entrypoint with no params shouldn't silently eat extra flags —
    it's almost always a typo."""
    script = _write_job(
        tmp_path,
        """
        from runplz import App, Image

        app = App("t")
        image = Image.from_registry("ubuntu:22.04")

        @app.function(image=image)
        def fn(): pass

        @app.local_entrypoint()
        def main():
            pass
        """,
    )
    with pytest.raises(SystemExit):
        _cli.main(["local", str(script), "--foo=bar"])
    err = capsys.readouterr().err
    assert "takes no arguments" in err


def test_repo_root_walks_up_to_git(tmp_path):
    (tmp_path / ".git").mkdir()
    sub = tmp_path / "nested" / "deeper"
    sub.mkdir(parents=True)
    script = sub / "job.py"
    script.write_text("x = 1\n")
    assert _cli._repo_root_for(script.resolve()) == tmp_path.resolve()


def test_repo_root_falls_back_to_script_parent_when_no_git(tmp_path):
    script = tmp_path / "job.py"
    script.write_text("x = 1\n")
    assert _cli._repo_root_for(script.resolve()) == tmp_path.resolve()
