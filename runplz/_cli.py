"""`runplz` CLI.

Usage:
    runplz local <script.py>
    runplz brev --instance my-gpu-box <script.py>
    runplz ssh  --host gpu.example.com <script.py>
    runplz modal <script.py>

Extra arguments after <script.py> are passed through to the script's
@app.local_entrypoint — modal-style:

    @app.local_entrypoint()
    def main(steps: int = 100, dataset: str = "small"): ...

    runplz local script.py --steps=1000 --dataset=big

Supported types: str, int, float, bool (--flag for True, --no-flag for
False, --flag=true/false/yes/no/1/0 for explicit). Optional[T] is
treated as T; default values on the entrypoint params become optional.

Loads the user's script, finds its @app.local_entrypoint, sets the backend,
and invokes it.
"""

import argparse
import importlib.util
import inspect
import io
import sys
import typing
from pathlib import Path

from runplz.app import _repo_root_for


def main(argv=None):
    # Line-buffer stdout/stderr so subprocess output and Python prints interleave
    # correctly when the user is tailing a log file or piping through tee.
    # Without this, block buffering kicks in under `| tee` or `> file`, and
    # prints like "+ rsync ..." land minutes late, making logs unreadable.
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except (AttributeError, io.UnsupportedOperation):
        pass
    p = argparse.ArgumentParser(
        prog="runplz", description="Run a Python @app.function on a chosen backend."
    )
    p.add_argument("backend", choices=["local", "brev", "modal", "ssh"])
    p.add_argument("script", help="Path to a job script defining an App with @local_entrypoint.")
    p.add_argument(
        "--outputs-dir",
        default="out",
        help="Host directory to collect outputs into (default: out/).",
    )
    p.add_argument(
        "--instance",
        help=(
            "[brev] Brev instance name. Omit for ephemeral mode: runplz "
            "auto-creates a box sized to your function's specs and deletes "
            "it on exit."
        ),
    )
    p.add_argument(
        "--host",
        help="[ssh] SSH endpoint (hostname, user@host, or ~/.ssh/config alias).",
    )
    p.add_argument(
        "--no-build", action="store_true", help="[local] Skip docker build (reuse tagged image)."
    )
    p.add_argument(
        "--log-file",
        help=(
            "Tee stdout+stderr to this log file (overrides the default path). "
            "Captures the full driver trail — rsync, ssh, docker build, "
            "remote streamed logs, failure traces — so a closed terminal "
            "can't strand you without a diagnostic record."
        ),
    )
    p.add_argument(
        "--no-log-file",
        action="store_true",
        help="Disable the default log-file capture. See --log-file.",
    )
    args, entrypoint_argv = p.parse_known_args(argv)

    script_path = Path(args.script).resolve()
    if not script_path.is_file():
        p.error(f"script not found: {script_path}")

    app = _load_app(script_path)

    # repo_root = git root or script's parent
    app._repo_root = _repo_root_for(script_path)
    app._backend = args.backend
    app._backend_kwargs = {"outputs_dir": args.outputs_dir}

    if args.backend == "brev":
        # args.instance can legitimately be None now — triggers ephemeral mode.
        app._backend_kwargs["instance"] = args.instance
    elif args.instance:
        p.error(f"--instance only applies to the brev backend (got {args.backend!r}).")
    if args.backend == "ssh":
        if not args.host:
            p.error("--host is required for the ssh backend")
        app._backend_kwargs["host"] = args.host
    elif args.host:
        p.error(f"--host only applies to the ssh backend (got {args.backend!r}).")
    if args.no_build:
        if args.backend != "local":
            p.error(f"--no-build only applies to the local backend (got {args.backend!r}).")
        app._backend_kwargs["build"] = False

    if app._entrypoint is None:
        p.error(f"{script_path} has no @app.local_entrypoint() function")
    entrypoint_kwargs = _parse_entrypoint_args(app._entrypoint, entrypoint_argv, p.error)

    # Resolve the log path relative to the same outputs-dir we'll hand to
    # the backend. The log-capture has to wrap everything after this point,
    # including the entrypoint and the backend dispatch.
    from runplz._logcapture import resolve_log_path, tee_stdio_to

    outputs_dir_abs = (app._repo_root / args.outputs_dir).resolve()
    log_path = resolve_log_path(
        log_file_flag=args.log_file,
        no_log_file_flag=args.no_log_file,
        outputs_dir=outputs_dir_abs,
        app_name=app.name,
    )
    if log_path is None:
        app._entrypoint(**entrypoint_kwargs)
        return
    print(f"+ logging driver output to {log_path}", flush=True)
    with tee_stdio_to(log_path):
        app._entrypoint(**entrypoint_kwargs)


_TRUTHY = {"true", "yes", "1", "on"}
_FALSY = {"false", "no", "0", "off"}


def _parse_entrypoint_args(entrypoint, extra_argv, fail):
    """Map leftover CLI argv onto the entrypoint's typed signature.

    Each `@app.local_entrypoint()` parameter becomes a `--<name>` flag with
    coercion driven by the param's annotation. Anything with a default is
    optional; anything without one is required.
    """
    sig = inspect.signature(entrypoint)
    if not sig.parameters:
        if extra_argv:
            fail(
                f"entrypoint {entrypoint.__name__}() takes no arguments, "
                f"but extra CLI args were given: {' '.join(extra_argv)}"
            )
        return {}

    ep = argparse.ArgumentParser(
        prog=f"runplz ... {entrypoint.__name__}",
        description=f"Arguments for @app.local_entrypoint def {entrypoint.__name__}(...):",
        add_help=False,  # --help is already owned by the outer parser
    )
    for name, param in sig.parameters.items():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            fail(
                f"entrypoint {entrypoint.__name__}() uses *args/**kwargs which "
                f"is not supported by the CLI — use explicit keyword params "
                f"with type annotations instead."
            )
        declared_type = _effective_type(param.annotation)
        has_default = param.default is not inspect.Parameter.empty
        flag = f"--{name.replace('_', '-')}"
        if declared_type is bool:
            _add_bool_flag(ep, flag, name, has_default, param.default)
        else:
            ep.add_argument(
                flag,
                dest=name,
                required=not has_default,
                default=param.default if has_default else None,
                type=_coercer_for(declared_type, param_name=name, fail=fail),
            )
    try:
        ns = ep.parse_args(extra_argv)
    except SystemExit as exc:
        # argparse already printed the error to stderr; re-raise as our own
        # SystemExit so the outer main() exits with argparse's code.
        raise SystemExit(exc.code)
    return {k: v for k, v in vars(ns).items() if v is not None or k in sig.parameters}


def _effective_type(annotation):
    """Unwrap Optional[T] / `T | None` to T. Return `str` for Parameter.empty."""
    if annotation is inspect.Parameter.empty:
        return str
    origin = typing.get_origin(annotation)
    if origin is typing.Union or origin is getattr(__import__("types"), "UnionType", None):
        non_none = [a for a in typing.get_args(annotation) if a is not type(None)]
        if len(non_none) == 1:
            return non_none[0]
    return annotation


def _coercer_for(declared_type, *, param_name, fail):
    """Return a callable that argparse uses to coerce the raw string
    CLI value to the annotated type. Surfaces a clean error on mismatch."""
    if declared_type is str:
        return str

    def coerce(value):
        try:
            return declared_type(value)
        except (TypeError, ValueError) as exc:
            raise argparse.ArgumentTypeError(
                f"entrypoint param --{param_name.replace('_', '-')} "
                f"expected {declared_type.__name__}, got {value!r}: {exc}"
            ) from exc

    return coerce


def _add_bool_flag(parser, flag, dest, has_default, default_value):
    """Add a boolean entrypoint flag with three accepted forms:

    - `--flag` → True
    - `--no-flag` → False
    - `--flag=true|yes|1` / `--flag=false|no|0` → explicit

    We use a small action class so the same `dest` can be driven by
    either `--flag` or `--flag=<value>` without conflict.
    """

    def str_to_bool(s):
        low = s.strip().lower()
        if low in _TRUTHY:
            return True
        if low in _FALSY:
            return False
        raise argparse.ArgumentTypeError(
            f"expected true/false (or yes/no, 1/0) for {flag}; got {s!r}"
        )

    # --flag (no value → True), --flag=<bool>
    parser.add_argument(
        flag,
        dest=dest,
        nargs="?",
        const=True,
        default=(default_value if has_default else None),
        type=str_to_bool,
        required=not has_default,
    )
    # --no-flag → False
    no_flag = flag.replace("--", "--no-", 1)
    parser.add_argument(
        no_flag,
        dest=dest,
        action="store_false",
    )


def _load_app(script_path: Path):
    spec = importlib.util.spec_from_file_location("_runplz_user_job", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_runplz_user_job"] = module
    spec.loader.exec_module(module)

    from runplz.app import App

    apps = [v for v in vars(module).values() if isinstance(v, App)]
    if not apps:
        raise SystemExit(f"No App found in {script_path}")
    if len(apps) > 1:
        raise SystemExit(f"Multiple Apps found in {script_path}; expected exactly one.")
    return apps[0]


if __name__ == "__main__":
    main()
