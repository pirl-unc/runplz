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
package. `sys.path[0]` remains the process CWD (the repo root), *not* the
script's own directory — so a job script can import modules from the repo
root but not siblings sitting next to it. This diverges from plain
`python path/to/job.py`; see issue tracking before relying on either.
"""

import importlib.util
import json
import os
import sys
from pathlib import Path


def main():
    script_path = os.environ["RUNPLZ_SCRIPT"]
    function_name = os.environ["RUNPLZ_FUNCTION"]
    args = json.loads(os.environ.get("RUNPLZ_ARGS", "[]"))
    kwargs = json.loads(os.environ.get("RUNPLZ_KWARGS", "{}"))

    script_path = str(Path(script_path).resolve())
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
