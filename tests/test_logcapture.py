"""Coverage for the #23 log-file tee in runplz/_logcapture.py."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

from runplz._logcapture import default_log_path, resolve_log_path, tee_stdio_to

# --- default_log_path --------------------------------------------------


def test_default_log_path_shape(tmp_path):
    p = default_log_path(tmp_path, "my-app")
    assert p.parent == tmp_path
    assert p.name.startswith("runplz-my-app-")
    assert p.suffix == ".log"


def test_default_log_path_sanitizes_app_name(tmp_path):
    p = default_log_path(tmp_path, "openvax/runplz demo")
    assert "/" not in p.name
    assert " " not in p.name
    assert "runplz-" in p.name


def test_default_log_path_falls_back_when_app_name_is_empty_after_sanitize(tmp_path):
    # "///" sanitizes to "---" then strips to "" → fallback "app".
    p = default_log_path(tmp_path, "///")
    assert p.name.startswith("runplz-app-")


# --- resolve_log_path --------------------------------------------------


def test_resolve_no_log_file_wins(tmp_path):
    assert (
        resolve_log_path(
            log_file_flag="/tmp/x.log",
            no_log_file_flag=True,
            outputs_dir=tmp_path,
            app_name="x",
        )
        is None
    )


def test_resolve_explicit_log_file_takes_precedence(tmp_path):
    p = resolve_log_path(
        log_file_flag=str(tmp_path / "explicit.log"),
        no_log_file_flag=False,
        outputs_dir=tmp_path / "out",
        app_name="x",
    )
    assert p == (tmp_path / "explicit.log").resolve()


def test_resolve_default_under_outputs_dir(tmp_path):
    out = tmp_path / "out"
    p = resolve_log_path(
        log_file_flag=None,
        no_log_file_flag=False,
        outputs_dir=out,
        app_name="my-app",
    )
    assert p is not None
    assert p.parent == out.resolve()
    assert p.name.startswith("runplz-my-app-")


def test_resolve_expands_tilde(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    p = resolve_log_path(
        log_file_flag="~/my.log",
        no_log_file_flag=False,
        outputs_dir=tmp_path / "out",
        app_name="x",
    )
    assert p == (tmp_path / "my.log").resolve()


# --- tee_stdio_to (behavioural) ---------------------------------------


def test_tee_writes_python_prints_to_log(tmp_path):
    log = tmp_path / "out.log"
    with tee_stdio_to(log):
        print("hello-from-runplz", flush=True)
    contents = log.read_text()
    assert "hello-from-runplz" in contents
    assert "# runplz log — started" in contents
    assert "# argv:" in contents


def test_tee_writes_stderr_to_log(tmp_path):
    import sys

    log = tmp_path / "err.log"
    with tee_stdio_to(log):
        print("a warning", file=sys.stderr, flush=True)
    assert "a warning" in log.read_text()


def test_tee_creates_parent_dirs(tmp_path):
    log = tmp_path / "nested" / "dir" / "run.log"
    with tee_stdio_to(log):
        print("ok", flush=True)
    assert log.exists()
    assert "ok" in log.read_text()


def test_tee_returns_log_path_to_caller(tmp_path):
    log = tmp_path / "returned.log"
    with tee_stdio_to(log) as yielded:
        pass
    assert yielded == log


def test_tee_appends_on_repeated_runs(tmp_path):
    log = tmp_path / "append.log"
    with tee_stdio_to(log):
        print("first-run", flush=True)
    with tee_stdio_to(log):
        print("second-run", flush=True)
    text = log.read_text()
    assert "first-run" in text
    assert "second-run" in text


def test_tee_restores_sys_streams_after_exit(tmp_path):
    """After the context exits, sys.stdout/stderr must be the originals
    again — subsequent prints don't accidentally keep hitting the log."""
    import sys

    pre_stdout = sys.stdout
    log = tmp_path / "restore.log"
    with tee_stdio_to(log):
        print("inside", flush=True)
    assert sys.stdout is pre_stdout
    print("outside-after-tee", flush=True)
    assert "outside-after-tee" not in log.read_text()


def test_tee_delegates_non_write_attrs_to_primary(tmp_path):
    """``sys.stdout.isatty()``, ``.encoding``, etc. must still work while
    the tee is installed."""
    import sys

    log = tmp_path / "attrs.log"
    with tee_stdio_to(log):
        _ = sys.stdout.isatty()
        _ = sys.stdout.encoding  # noqa: B018


def test_tee_survives_log_write_failure():
    """If the log filehandle chokes on a write, the primary write must
    still complete — a broken logfile shouldn't swallow terminal output."""
    from io import StringIO

    from runplz._logcapture import _TeeStream

    class _BrokenLog:
        def write(self, _s):
            raise OSError("disk full")

        def flush(self):
            raise OSError("disk full")

        def close(self):
            pass

    primary = StringIO()
    tee = _TeeStream(primary, _BrokenLog())
    tee.write("made it\n")
    tee.flush()
    assert primary.getvalue() == "made it\n"


def test_cli_passes_log_file_flag_through(tmp_path):
    """CLI-level smoke: `runplz local script.py --log-file PATH` writes
    the header and Python prints to PATH."""
    import textwrap

    from runplz import _cli

    script = tmp_path / "job.py"
    script.write_text(
        textwrap.dedent(
            """
            from runplz import App, Image

            app = App("smoke")
            image = Image.from_registry("ubuntu:22.04")

            @app.function(image=image)
            def fn(): pass

            @app.local_entrypoint()
            def main():
                print("entrypoint-was-called")
            """
        )
    )
    log = tmp_path / "custom.log"
    with mock.patch("runplz.backends.local.run", lambda *a, **kw: None):
        _cli.main(["local", str(script), "--log-file", str(log)])

    body = log.read_text()
    assert "# runplz log" in body
    assert "# argv:" in body
    assert "entrypoint-was-called" in body


def test_cli_no_log_file_flag_skips_capture(tmp_path):
    """--no-log-file → no runplz-*.log under outputs-dir."""
    import textwrap

    from runplz import _cli

    script = tmp_path / "job.py"
    script.write_text(
        textwrap.dedent(
            """
            from runplz import App, Image
            app = App("silent")
            image = Image.from_registry("ubuntu:22.04")
            @app.function(image=image)
            def fn(): pass
            @app.local_entrypoint()
            def main(): pass
            """
        )
    )
    with mock.patch("runplz.backends.local.run", lambda *a, **kw: None):
        _cli.main(
            [
                "local",
                str(script),
                "--outputs-dir",
                str(tmp_path / "out"),
                "--no-log-file",
            ]
        )
    logs = list((tmp_path / "out").glob("runplz-*.log")) if (tmp_path / "out").exists() else []
    assert logs == []


def test_cli_default_creates_log_under_outputs_dir(tmp_path, monkeypatch):
    """No flags → runplz writes a timestamped log under <outputs-dir>."""
    import textwrap

    from runplz import _cli

    monkeypatch.chdir(tmp_path)
    script = tmp_path / "job.py"
    script.write_text(
        textwrap.dedent(
            """
            from runplz import App, Image
            app = App("defaulted")
            image = Image.from_registry("ubuntu:22.04")
            @app.function(image=image)
            def fn(): pass
            @app.local_entrypoint()
            def main(): pass
            """
        )
    )
    with mock.patch("runplz.backends.local.run", lambda *a, **kw: None):
        _cli.main(["local", str(script), "--outputs-dir", "out"])

    logs = list((tmp_path / "out").glob("runplz-defaulted-*.log"))
    assert len(logs) == 1, f"expected 1 default log, got {logs}"
    body = Path(logs[0]).read_text()
    assert "# argv:" in body


def test_cli_default_log_follows_repo_outputs_dir_when_invoked_outside_repo(tmp_path, monkeypatch):
    """The default log path should track the backend outputs dir rooted at
    the repo, not the caller's current working directory."""
    import textwrap

    from runplz import _cli

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    script = repo / "job.py"
    script.write_text(
        textwrap.dedent(
            """
            from runplz import App, Image
            app = App("outside")
            image = Image.from_registry("ubuntu:22.04")
            @app.function(image=image)
            def fn(): pass
            @app.local_entrypoint()
            def main(): pass
            """
        )
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.chdir(outside)

    with mock.patch("runplz.backends.local.run", lambda *a, **kw: None):
        _cli.main(["local", str(script), "--outputs-dir", "out"])

    repo_logs = list((repo / "out").glob("runplz-outside-*.log"))
    cwd_logs = list((outside / "out").glob("runplz-outside-*.log"))
    assert len(repo_logs) == 1, f"expected 1 repo-rooted log, got {repo_logs}"
    assert cwd_logs == []
