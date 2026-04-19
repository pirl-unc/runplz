"""In-container bootstrap.

Invoked by the runplz backend inside the container/VM. Imports the
user's job script by file path, then calls the target Function's
.local(*args, **kwargs).

Backends must set these env vars:
  RUNPLZ_SCRIPT      — absolute path to user's script file
  RUNPLZ_FUNCTION    — function name (matches @app.function)
  RUNPLZ_ARGS        — JSON list of args (default: [])
  RUNPLZ_KWARGS      — JSON dict of kwargs (default: {})
  RUNPLZ_OUT         — outputs directory (user code reads this)

The user's @app.function is imported by path (not by package), so the
script does not need to be installed.
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
        raise RuntimeError(
            f"Function {function_name!r} not found in {script_path}"
        )
    # fn is a Function wrapper; call the underlying callable directly.
    result = fn.local(*args, **kwargs)
    if result is not None:
        # Emit a sentinel for CLI consumers, but keep stdout human-readable.
        print(f"[runplz] result: {result!r}")


if __name__ == "__main__":
    main()
