"""`runplz` CLI.

Usage:
    runplz local <script.py>
    runplz brev --instance my-gpu-box <script.py>
    runplz ssh  --host gpu.example.com <script.py>
    runplz modal <script.py>
    runplz ps [local|brev|modal|ssh|gcp|aws] [--host <host>] [--region <r>]
    runplz tail [--outputs-dir <path>] [--host <host>] [--run-id <id>] [-n N] [-f]
    runplz status [--outputs-dir <path>] [--host <host>] [--run-id <id>]

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

__all__ = ["main"]


import argparse
import importlib.util
import inspect
import io
import sys
import typing
import warnings
from pathlib import Path

from runplz.app import repo_root_for
from runplz.backends import listing, registry
from runplz.backends.listing import JobRecord


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

    argv_list = list(sys.argv[1:] if argv is None else argv)
    if argv_list and argv_list[0] == "ps":
        return _ps_main(argv_list[1:])
    if argv_list and argv_list[0] == "tail":
        return _tail_main(argv_list[1:])
    if argv_list and argv_list[0] == "status":
        return _status_main(argv_list[1:])
    if argv_list and argv_list[0] in ("kill", "cancel"):
        return _kill_main(argv_list[1:], prog=argv_list[0])

    p = argparse.ArgumentParser(
        prog="runplz", description="Run a Python @app.function on a chosen backend."
    )
    p.add_argument("backend", choices=list(registry.names()))
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

    # Before binding: bind() would reject a function-less script with a
    # generic message, and this one names the script and says what to add.
    if app.entrypoint is None:
        _install_default_entrypoint_or_error(app, script_path, p.error)

    # The CLI binds through App.bind() rather than reimplementing it. It used
    # to set app._backend by hand with its own copy of the validation, and
    # that copy drifted: it never learned that gcp/aws require a config, so
    # `runplz gcp job.py` with no GcpConfig died on an AttributeError deep in
    # the driver instead of saying what was missing.
    try:
        app.bind(
            args.backend,
            instance=args.instance,
            host=args.host,
            outputs_dir=args.outputs_dir,
            build=not args.no_build,
            # The CLI knows the script being run, which beats the module a
            # function happens to be defined in — and hands bind() the answer
            # rather than making it shell out to git a second time.
            #
            # But only as a fallback: a bind(repo_root=...) argument outranks
            # a standing `app.repo_root = X`, so passing it unconditionally
            # would make that documented override a no-op on the CLI path,
            # which is the path almost everyone uses. `app.repo_root` is None
            # here unless the script assigned one, because bind() has not run.
            repo_root=None if app.repo_root is not None else repo_root_for(script_path),
        )
    except (ValueError, RuntimeError) as exc:
        p.error(str(exc))

    entrypoint_kwargs = _parse_entrypoint_args(app.entrypoint, entrypoint_argv, p.error)

    # Resolve the log path relative to the same outputs-dir we'll hand to
    # the backend. The log-capture has to wrap everything after this point,
    # including the entrypoint and the backend dispatch.
    from runplz.logcapture import resolve_log_path, tee_stdio_to

    outputs_dir_abs = (app.repo_root / args.outputs_dir).resolve()
    log_path = resolve_log_path(
        log_file_flag=args.log_file,
        no_log_file_flag=args.no_log_file,
        outputs_dir=outputs_dir_abs,
        app_name=app.name,
    )
    if log_path is None:
        app.entrypoint(**entrypoint_kwargs)
        return
    print(f"+ logging driver output to {log_path}", flush=True)
    with tee_stdio_to(log_path):
        app.entrypoint(**entrypoint_kwargs)


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


def _install_default_entrypoint_or_error(app, script_path, fail):
    """Synthesize a one-line entrypoint when the user didn't write one.

    The default behavior is unambiguous *only* when the script declares
    exactly one ``@app.function``: we wire that function's ``.remote(**kwargs)``
    as the entrypoint, with the function's own signature driving CLI argument
    parsing. With zero or multiple functions there's no good default — error
    out with guidance on what to do next.
    """
    fns = list(app.functions.values())
    if len(fns) == 0:
        fail(
            f"{script_path} declares no @app.function and no @app.local_entrypoint. "
            f"Add at least one @app.function so runplz has something to run."
        )
    if len(fns) > 1:
        names = ", ".join(sorted(app.functions))
        fail(
            f"{script_path} declares {len(fns)} @app.function "
            f"({names}) but no @app.local_entrypoint. "
            f"runplz only auto-runs a default function when exactly one exists. "
            f"Either remove the extras, or add an @app.local_entrypoint that picks one explicitly."
        )
    only_fn = fns[0]

    def _default_entrypoint(**kwargs):
        return only_fn.remote(**kwargs)

    # Mirror the function's signature so _parse_entrypoint_args can build the
    # right --flag set, and copy the name so the generated --help and error
    # messages still point at something meaningful.
    _default_entrypoint.__signature__ = inspect.signature(only_fn.fn)
    _default_entrypoint.__name__ = only_fn.name
    app.entrypoint = _default_entrypoint


def _load_app(script_path: Path):
    # Show runplz's own DeprecationWarnings while executing the user's script.
    # Python's default filters only surface them in `__main__`, and this loads
    # the script as `_runplz_user_job` — so a script using a deprecated alias
    # would get no notice at all before the alias is removed in 4.0.
    warnings.filterwarnings("default", category=DeprecationWarning, module=r"runplz\..*")
    warnings.filterwarnings("default", category=DeprecationWarning, module="_runplz_user_job")
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


def _ps_main(argv):
    """Dispatch ``runplz ps`` — list runplz jobs across backends.

    Every provider-specific detail here comes from the registry: the flags,
    which backends each flag reaches, and what scope a backend needs before
    it can be asked. What is left is the part that is genuinely the CLI's —
    who to ask, and what to do when one of them fails.
    """
    p = argparse.ArgumentParser(
        prog="runplz ps",
        description="List runplz jobs currently running across backends.",
    )
    p.add_argument(
        "backend",
        nargs="?",
        choices=registry.listable_names(),
        help=(
            f"Limit listing to one backend. Default: fan out to {', '.join(registry.ps_names())}."
        ),
    )
    fields = registry.scope_fields()
    for field, owners in fields:
        help_text = f"[{'|'.join(owners)}] {field.help}"
        if field.env:
            help_text += f" Or set {' / '.join(field.env)}."
        p.add_argument(
            field.flag,
            *field.aliases,
            dest=field.name,
            type=field.type,
            # Name the placeholder after the flag, not the driver keyword it
            # feeds: `--ssh-key SSH_KEY` is what the user typed and what the
            # help has always shown.
            metavar=field.flag.lstrip("-").replace("-", "_").upper(),
            help=help_text,
        )
    args = p.parse_args(argv)
    scope = {field.name: getattr(args, field.name) for field, _ in fields}
    for field, _ in fields:
        if field.validate is None or scope[field.name] is None:
            continue
        try:
            field.validate(scope[field.name])
        except ValueError as exc:
            p.error(f"{field.flag} {exc}")

    backends = _ps_selection(args.backend, scope)
    _reject_unreachable_scope(p, fields, scope, backends)

    rows = []
    errors = []
    successes = 0
    for backend, target_scope, label in _ps_targets(backends, scope):
        try:
            rows.extend(registry.list_jobs(backend, **target_scope))
            successes += 1
        except Exception as exc:  # noqa: BLE001
            errors.append((label, exc))

    # Print the table whenever at least one backend was reachable. Suppress
    # the "(no runplz jobs running)" line ONLY when every backend errored —
    # otherwise an empty result from the working backends is real
    # information the user asked for.
    if rows or successes > 0:
        _print_ps_table(rows)
    if successes > 0 and not rows and args.backend is None:
        _note_backends_not_listed(backends)
    for name, exc in errors:
        print(f"warning: {name} listing failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    return 1 if errors and not rows and successes == 0 else 0


def _reject_unreachable_scope(parser, fields, scope: dict, backends: list) -> None:
    """Refuse a scope flag that no backend being listed can use.

    `runplz ps local --region us-east-1` used to run and quietly ignore the
    region, because AWS is in the default fan-out and so is never *invited* by
    a flag -- and a positional had already excluded it. The user narrowed the
    listing and scoped a backend that was not in it, which is two mistakes
    that cancel into silence.

    A flag that is accepted, validated and then dropped is the failure class
    this repo has spent several releases removing (#20, #143, #153): the
    operator believes the listing was scoped and it was not. Erroring names
    both halves -- the flag, and who would have had to be listed for it to
    mean anything.
    """
    listed = set(backends)
    for field, owners in fields:
        raw = scope[field.name]
        # Only what the user typed. `resolve_all` falls back to the
        # environment, and an `AWS_DEFAULT_REGION` that happens to be exported
        # is not a claim about this command — erroring on it would break
        # `runplz ps local` for anyone with the variable set.
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            continue
        # Supplied but naming nothing — `--host ,` — is already "scope the
        # user did not supply" everywhere else, and must stay that here.
        # `raw` is non-blank, so this never reaches the environment either.
        if not field.resolve_all(raw):
            continue
        if listed & set(owners):
            continue
        parser.error(
            f"{field.flag} only applies to {'/'.join(owners)}, which "
            f"{'is' if len(owners) == 1 else 'are'} not being listed. "
            f"Listing: {', '.join(backends)}."
        )


def _ps_selection(chosen: str, scope: dict) -> list:
    """Which backends this `runplz ps` invocation should ask.

    A backend outside the default fan-out joins as soon as the user supplies
    the scope it needs, and it does so *whether or not* a positional narrowed
    the rest: `runplz ps local --host box` has always listed local jobs and
    the box's, and dropping that would quietly lose half the answer.

    The converse is deliberately not true. Supplying `--region` does not pull
    AWS into `runplz ps local` — AWS is already in the fan-out, so the flag
    scopes a backend that was going to be asked anyway rather than adding one.
    """
    fan_out = registry.ps_names()
    selected = [chosen] if chosen else list(fan_out)
    for name in registry.listable_names():
        if name in selected or name in fan_out:
            continue
        if registry.get(name).listing.invited_by(scope):
            selected.append(name)
    return selected


def _ps_targets(backends: list, scope: dict):
    """Expand a backend selection into `(backend, scope, error_label)` calls.

    Most backends are one call. A field marked `multiple` — ssh hosts, the
    only case — is comma-split into one call per value, so an unreachable box
    costs the user that box's row rather than every box's, and the warning
    names which one failed.
    """
    for backend in backends:
        spec = registry.get(backend).listing
        base = {f.name: scope.get(f.name) for f in spec.scope}
        fan = next((f for f in spec.scope if f.multiple), None)
        values = fan.resolve_all(base.get(fan.name)) if fan else []
        if not values:
            # No fan-out field, or nothing supplied for it. Ask once and let
            # scope resolution report a required field the user left out.
            yield backend, base, backend
            continue
        for value in values:
            yield backend, {**base, fan.name: value}, f"{backend}:{value}"


def _note_backends_not_listed(selected: list) -> None:
    """Say which backends went unasked when the table came back empty.

    An empty table is the one moment "nothing is running" and "nobody asked"
    look identical, and jobs on an ssh box are invisible to a bare
    `runplz ps`. Only said when there is nothing to show — once rows exist,
    the user can see what was covered.
    """
    for name in registry.listable_names():
        if name in selected:
            continue
        required = registry.get(name).listing.required_fields()
        # A backend kept out of the fan-out without required scope cannot be
        # invited by a flag, so "pass  to include it" would name nothing.
        how = (
            f"pass {' '.join(f.flag for f in required)}" if required else f"run `runplz ps {name}`"
        )
        print(f"note: {name} was not listed; {how} to include it", file=sys.stderr)


def _print_ps_table(rows: list[JobRecord]) -> None:
    if not rows:
        print("(no runplz jobs running)")
        return
    # Columns come from the record itself, so renaming or reordering a field
    # moves its column rather than silently printing a blank one.
    keys = JobRecord.field_names()
    cells = [[str(getattr(row, key) or "") for key in keys] for row in rows]
    widths = [len(key) for key in keys]
    for row in cells:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(value))
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*(key.upper() for key in keys)))
    for row in cells:
        print(fmt.format(*row))


def _tail_main(argv):
    from runplz import runs

    p = argparse.ArgumentParser(
        prog="runplz tail",
        description="Tail the remote driver log of the most recent run.",
    )
    _add_run_lookup_args(p)
    p.add_argument(
        "-n",
        dest="lines",
        type=int,
        default=120,
        help="Number of lines to show (default: 120).",
    )
    p.add_argument(
        "-f",
        "--follow",
        action="store_true",
        help="Stream new lines as they arrive (passes -F to remote tail).",
    )
    args = p.parse_args(argv)
    try:
        return runs.tail(
            outputs_dir=Path(args.outputs_dir).resolve(),
            host_override=args.host,
            run_id_override=args.run_id,
            ssh_overrides=_ssh_overrides_from_args(args, p.error),
            lines=args.lines,
            follow=args.follow,
        )
    except runs.ManifestNotFound as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except (RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


def _status_main(argv):
    from runplz import runs

    p = argparse.ArgumentParser(
        prog="runplz status",
        description="Print a one-screen summary of the most recent run's state.",
    )
    _add_run_lookup_args(p)
    args = p.parse_args(argv)
    try:
        return runs.status(
            outputs_dir=Path(args.outputs_dir).resolve(),
            host_override=args.host,
            run_id_override=args.run_id,
            ssh_overrides=_ssh_overrides_from_args(args, p.error),
        )
    except runs.ManifestNotFound as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except (RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


def _kill_main(argv, *, prog="kill"):
    from runplz import runs
    from runplz.backends.ssh_common import DEFAULT_KILL_TIMEOUT_S

    p = argparse.ArgumentParser(
        prog=f"runplz {prog}",
        description=(
            "Stop a running detached job: signals the bootstrap's process "
            "group (workers and forked children included) plus the docker "
            "container in VM+docker mode, then records killed_by_user."
        ),
    )
    _add_run_lookup_args(p)
    p.add_argument(
        "--signal",
        default="TERM",
        choices=["TERM", "INT", "KILL"],
        help=(
            "First signal to send (default: TERM). TERM lets the job unwind — "
            "flush checkpoints, close writers — before the SIGKILL escalation."
        ),
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_KILL_TIMEOUT_S,
        help=(
            f"Seconds to wait for a clean exit before escalating to SIGKILL "
            f"(default: {DEFAULT_KILL_TIMEOUT_S})."
        ),
    )
    p.add_argument(
        "--no-escalate",
        action="store_true",
        help="Never send SIGKILL — report what survived the first signal instead.",
    )
    args = p.parse_args(argv)
    if args.timeout < 0:
        p.error("--timeout must be >= 0")
    try:
        return runs.kill(
            outputs_dir=Path(args.outputs_dir).resolve(),
            host_override=args.host,
            run_id_override=args.run_id,
            ssh_overrides=_ssh_overrides_from_args(args, p.error),
            timeout_s=args.timeout,
            escalate=not args.no_escalate,
            first_signal=args.signal,
        )
    except runs.ManifestNotFound as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except (RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


def _ssh_overrides_from_args(args, fail):
    """The ssh fields the user pinned on the command line, if any.

    Returned per field rather than as a whole object: `--ssh-key` alone must
    not discard the port the dispatch recorded, which is what "override"
    means to anyone reading the help text.
    """
    key = getattr(args, "ssh_key", None)
    port = getattr(args, "ssh_port", None)
    if port is not None:
        try:
            listing.tcp_port_range(port)
        except ValueError as exc:
            fail(f"--ssh-port {exc}")
    overrides = {}
    if key:
        overrides["identity_file"] = key
    if port is not None:
        overrides["port"] = port
    return overrides


def _add_run_lookup_args(parser):
    parser.add_argument(
        "--outputs-dir",
        default="out",
        help="Local outputs dir to read run.json from (default: out/).",
    )
    parser.add_argument(
        "--host",
        help="Override the SSH target. Required when --run-id is given without a local manifest.",
    )
    parser.add_argument(
        "--run-id",
        help="Pin to a specific remote run_id instead of the most recent. Requires --host.",
    )
    parser.add_argument(
        "--ssh-key",
        help=(
            "Private key to authenticate with. Defaults to whatever the run's "
            "manifest recorded, so this is only needed when following a run "
            "by --host/--run-id with no local manifest."
        ),
    )
    parser.add_argument(
        "--ssh-port",
        type=int,
        help="SSH port. Defaults to the run manifest's, then ssh's own default.",
    )


if __name__ == "__main__":
    raise SystemExit(main())
