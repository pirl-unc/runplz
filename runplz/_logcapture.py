"""Tee the runplz driver's stdout + stderr to a log file.

Issue #23: when the local driver crashes or the tty dies, there's no
persisted record of what went wrong. runplz creates a billed Brev box,
the terminal closes, and the only diagnostic trail is gone with the
scrollback. Users had to remember to wrap invocations as:

    runplz brev --instance X jobs/foo.py 2>&1 | tee out/foo.log

This module does it for them.

Implementation: we wrap ``sys.stdout`` and ``sys.stderr`` with a tee
that writes to both the original stream and an append-mode log file.
Every Python-level ``print``, exception traceback, ``logging``
message, and all runplz driver output ends up in both places.

Scope: Python-level output only. Subprocesses inherit the original
fd 1 / fd 2 and write directly to the terminal — they *don't* hit our
tee. That's an intentional simplification: the motivating failure
mode from #23 was "the driver exited before ever reaching docker run,"
which is a Python-level event; capturing subprocess output would
require fd-level redirection that doesn't play nicely with pytest's
own capture machinery. We can revisit if subprocess capture becomes
necessary.
"""

from __future__ import annotations

import contextlib
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def default_log_path(outputs_dir: Path, app_name: str) -> Path:
    """Shape: ``<outputs-dir>/runplz-<app>-<YYYYmmdd-HHMMSS>.log``. Local
    time, not UTC — matches the user's shell history."""
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in app_name)
    safe = safe.strip("-") or "app"
    return outputs_dir / f"runplz-{safe}-{ts}.log"


class _TeeStream:
    """File-like wrapper that mirrors writes to a primary stream (keeps
    the terminal working) and a log file. Non-``write``/``flush`` attrs
    proxy to the primary so ``sys.stdout.isatty()``, ``.fileno()``, etc.
    stay honest."""

    def __init__(self, primary, log_fh):
        self._primary = primary
        self._log_fh = log_fh

    def write(self, s):
        n = self._primary.write(s)
        try:
            self._log_fh.write(s)
        except (OSError, ValueError):
            pass
        return n

    def flush(self):
        try:
            self._primary.flush()
        except (OSError, ValueError):
            pass
        try:
            self._log_fh.flush()
        except (OSError, ValueError):
            pass

    def __getattr__(self, name):
        return getattr(self._primary, name)


@contextlib.contextmanager
def tee_stdio_to(log_path: Path):
    """Wrap ``sys.stdout`` / ``sys.stderr`` with a tee for the duration
    of the context.

    Writes a breadcrumb header at the top of the log file so someone
    grepping an old file can tell which runplz invocation it was.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(log_path, "a", buffering=1)

    # Breadcrumb header — useful when a log file accumulates across
    # several runs (the default path includes a timestamp, so this is
    # mostly for `--log-file <fixed-path>` users who append to one file).
    log_fh.write(
        f"# runplz log — started {datetime.now().isoformat(timespec='seconds')}\n"
        f"# argv: {' '.join(sys.argv)}\n"
        f"# cwd:  {os.getcwd()}\n"
    )
    log_fh.flush()

    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    sys.stdout = _TeeStream(orig_stdout, log_fh)
    sys.stderr = _TeeStream(orig_stderr, log_fh)
    try:
        yield log_path
    finally:
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        try:
            log_fh.close()
        except OSError:
            pass


def resolve_log_path(
    *,
    log_file_flag: Optional[str],
    no_log_file_flag: bool,
    outputs_dir: Path,
    app_name: str,
) -> Optional[Path]:
    """Turn CLI flags into a final log-path decision.

    - ``--no-log-file`` wins outright.
    - ``--log-file PATH`` uses PATH verbatim.
    - Otherwise default under ``outputs_dir``.

    Returns ``None`` when no capture should be installed.
    """
    if no_log_file_flag:
        return None
    if log_file_flag:
        return Path(log_file_flag).expanduser().resolve()
    return default_log_path(outputs_dir, app_name).resolve()
