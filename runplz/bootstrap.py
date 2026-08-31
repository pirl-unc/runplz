"""In-container bootstrap: the orchestrator/container wire contract.

This module is the far side of every dispatch. A backend starts a container
or a remote process, sets the environment below, and runs::

    python -m runplz._bootstrap

`bootstrap` then imports the user's job script by file path and calls the
named Function's `.local(*args, **kwargs)`.

Environment contract — backends MUST set these:

===================  =====================================================
`RUNPLZ_SCRIPT`      absolute path to the user's script inside the container
`RUNPLZ_FUNCTION`    function name, matching an `@app.function` in it
`RUNPLZ_OUT`         outputs directory; **user code reads this** and it is
                     documented in the README as public API
`RUNPLZ_ARGS`        JSON list of positional args (optional, default `[]`)
`RUNPLZ_KWARGS`      JSON dict of keyword args (optional, default `{}`)
===================  =====================================================

This is a **cross-version contract, not an internal call.** runplz is not
staged to the remote — only the user's repo is (see "What runplz does NOT
ship to the remote" in the README) — so the container's runplz comes from
PyPI or the base image and its version is independent of the orchestrator's.
A 3.20 orchestrator can and does talk to a 3.19 container.

Two consequences:

- **The invoked module path is part of the wire format.** Backends still emit
  `python -m runplz._bootstrap`, which older containers understand. That
  legacy path is kept working by `runplz/_bootstrap.py`, a thin entry point
  that delegates here. Emitting `runplz.bootstrap` instead would break a new
  orchestrator against an older container, so the switch waits until 3.20+
  is broadly installed.
- **The env-var names may only be added to, never renamed**, for the same
  reason.

Import semantics: the script is loaded by path under the module name
`_runplz_user_job`, so it does not need to be installed or importable as a
package. Two directories are importable from it:

- the **repo root**, because `sys.path[0]` is the process CWD and runplz
  runs from the staged repo. This is first, so it wins on a name clash.
- the **script's own directory**, appended at dispatch (see
  `_add_script_dir_to_path`), so a job laid out as a directory of modules
  can import its siblings.

Note this is *not* what plain `python path/to/job.py` does: that puts the
script's directory first and does not put the repo root on the path at all.
The order here is deliberate — see `_add_script_dir_to_path` for why.
"""

__all__ = ["main"]


import importlib.util
import json
import os
import sys
from pathlib import Path


def _add_script_dir_to_path(script_path: str) -> None:
    """Let a job script import modules sitting next to it.

    The script is loaded by path, which puts nothing on `sys.path`, so
    `sys.path[0]` stays the process CWD -- the repo root. A job could import
    from the repo root but not from its own directory, so a job laid out as a
    directory of modules (a trainer plus `data.py`, `model.py` beside it)
    failed on dispatch. Issue #89.

    Appended, not inserted, and this is the whole design:

    * Plain `python jobs/train.py` puts the script's directory at
      `sys.path[0]` and does *not* have the repo root on the path at all --
      the exact opposite of what runplz does. Adopting Python's order would
      break every job that imports from the repo root today, which is the
      working behavior and the one that matches what runplz stages.
    * Appending is a strict superset: nothing that resolves today changes,
      and siblings newly resolve.
    * Landing after the stdlib and site-packages means a `jobs/types.py`
      does not shadow the stdlib for the rest of the run. Plain Python would
      let it. The cost is that a sibling whose name collides with a stdlib
      module stays unimportable -- a deliberate trade.
    """
    script_dir = str(Path(script_path).resolve().parent)
    if script_dir not in sys.path:
        sys.path.append(script_dir)


def main():
    script_path = os.environ["RUNPLZ_SCRIPT"]
    function_name = os.environ["RUNPLZ_FUNCTION"]
    args = json.loads(os.environ.get("RUNPLZ_ARGS", "[]"))
    kwargs = json.loads(os.environ.get("RUNPLZ_KWARGS", "{}"))

    script_path = str(Path(script_path).resolve())
    _add_script_dir_to_path(script_path)
    spec = importlib.util.spec_from_file_location("_runplz_user_job", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load user job from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["_runplz_user_job"] = module
    spec.loader.exec_module(module)

    fn = getattr(module, function_name, None)
    if fn is None:
        raise RuntimeError(f"Function {function_name!r} not found in {script_path}")
    # fn is a Function wrapper; call the underlying callable directly.
    result = fn.local(*args, **kwargs)
    if result is not None:
        # Emit a sentinel for CLI consumers, but keep stdout human-readable.
        print(f"[runplz] result: {result!r}")


if __name__ == "__main__":
    main()
