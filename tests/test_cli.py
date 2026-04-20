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


def test_cli_brev_requires_instance(tmp_path, capsys):
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
        _cli.main(["brev", str(script)])
    assert "--instance is required" in capsys.readouterr().err


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


def test_cli_rejects_script_without_local_entrypoint(tmp_path, capsys):
    script = _write_job(
        tmp_path,
        """
        from runplz import App, Image
        app = App("t")
        image = Image.from_registry("ubuntu:22.04")
        @app.function(image=image)
        def fn(): pass
        """,
    )
    with pytest.raises(SystemExit):
        _cli.main(["local", str(script)])
    assert "has no @app.local_entrypoint" in capsys.readouterr().err


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
