"""Receipts and bounded observation/collection for detached Modal runs.

Only the generated *local* entrypoint imports this module. The remote image
need not contain this version of runplz. SDK operations run in a subprocess:
`FunctionCall.get(timeout=0)` bounds result waiting, not every network request.
"""

import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

__all__ = ["prepare_run", "record_launch", "status", "collect"]

_PROBE_TIMEOUT = 20
_STATES = {"succeeded", "failed", "pending", "expired", "unconfirmed"}


def _receipt_path(outputs_dir):
    return Path(outputs_dir) / ".runplz" / "run.json"


@contextmanager
def _atomic_file(path, *, mode="wb"):
    temporary = tempfile.NamedTemporaryFile(mode=mode, dir=path.parent, delete=False)
    try:
        with temporary as f:
            yield f
            f.flush()
            os.fsync(f.fileno())
        # Close before replacing, including on platforms that prohibit renaming
        # open files. A failed transfer never truncates the previous good copy.
        os.replace(temporary.name, path)
    finally:
        Path(temporary.name).unlink(missing_ok=True)


def _write(path, data):
    """Replace a receipt atomically, keeping the last readable copy on failure."""
    with _atomic_file(path, mode="w") as f:
        json.dump(data, f, indent=2)


def prepare_run(outputs_dir, app_name, function_name, volume_name):
    """Reserve an output directory before submitting any billable work."""
    path = _receipt_path(outputs_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    receipt = {
        "backend": "modal",
        "mode": "detached",
        "schema_version": 1,
        "run_id": run_id,
        "app_name": app_name,
        "function_name": function_name,
        "volume_name": volume_name,
        "remote_path": f"/runplz/{run_id}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "launch_state": "prepared",
    }
    try:
        with path.open("x") as f:
            json.dump(receipt, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
    except FileExistsError:
        raise ValueError(
            f"A run receipt already exists at {path}. Use a new --outputs-dir for each "
            "detached launch; collect or inspect the existing run before launching another."
        ) from None
    return path, receipt


def _read(outputs_dir):
    path = _receipt_path(outputs_dir)
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Cannot read Modal launch receipt {path}: {exc}") from exc
    if not isinstance(data, dict) or data.get("backend") != "modal":
        raise ValueError("runplz collect requires a detached Modal run receipt.")
    if data.get("mode") != "detached" or data.get("schema_version") != 1:
        raise ValueError("Unsupported Modal receipt format.")
    run_id = data.get("run_id")
    if not isinstance(run_id, str) or not re.fullmatch(r"[a-f0-9]{32}", run_id):
        raise ValueError("Invalid Modal run ID in receipt.")
    if data.get("remote_path") != f"/runplz/{run_id}":
        raise ValueError("Invalid Modal output path in receipt.")
    if not isinstance(data.get("volume_name"), str) or not data["volume_name"].strip():
        raise ValueError("Missing Modal volume name in receipt.")
    for field, prefix in (
        ("app_id", "ap"),
        ("function_id", "fu"),
        ("call_id", "fc"),
        ("volume_id", "vo"),
    ):
        if field in data and (
            not isinstance(data[field], str)
            or not re.fullmatch(rf"{prefix}-[A-Za-z0-9]+", data[field])
        ):
            raise ValueError(f"Invalid {field} in Modal receipt.")
    if "volume_id" in data and not isinstance(data.get("environment"), str):
        raise ValueError("Missing Modal environment in receipt.")
    return data


def record_launch(receipt_path, app, runner, volume):
    """Called inside `modal run --detach`; never wait for the spawned result."""
    from modal.config import config

    path = Path(receipt_path)
    receipt = _read(path.parent.parent)
    volume.hydrate()
    receipt.update(
        app_id=app.app_id,
        function_id=runner.object_id,
        volume_id=volume.object_id,
        environment=config.get("environment") or "",
        launch_state="submitting",
    )
    # Persist the app BEFORE the ambiguous network boundary. No retry here:
    # an interrupted spawn may already have accepted the work remotely.
    _write(path, receipt)
    call = runner.spawn()
    receipt.update(call_id=call.object_id, launch_state="submitted")
    _write(path, receipt)


def _volume(receipt):
    # An empty string means the workspace default, not the caller's newly
    # activated profile environment. Modal checks this variable at lookup time.
    os.environ["MODAL_ENVIRONMENT"] = receipt["environment"]
    import modal

    volume = modal.Volume.from_name(receipt["volume_name"], environment_name=receipt["environment"])
    volume.hydrate()
    if volume.object_id != receipt["volume_id"]:
        raise RuntimeError(
            "Modal volume identity changed. Restore the original Modal credentials/environment; "
            "refusing to read a different or recreated volume."
        )
    return volume


def _outcome(code):
    if type(code) is not int or not -255 <= code <= 255:
        raise ValueError("Invalid exit code in Modal result.")
    return {"state": "succeeded" if code == 0 else "failed", "exit_code": code}


def _probe(receipt):
    import modal

    if "volume_id" not in receipt:
        return {"state": "unconfirmed"}
    volume = _volume(receipt)
    try:
        content = bytearray()
        for chunk in volume.read_file(receipt["remote_path"] + "/.runplz/modal-result.json"):
            content.extend(chunk)
            if len(content) > 16384:
                raise ValueError("Oversized Modal completion record.")
    except FileNotFoundError:
        pass
    else:
        result = json.loads(content)
        if not isinstance(result, dict) or result.get("run_id") != receipt["run_id"]:
            raise ValueError("Modal completion record belongs to a different run.")
        return _outcome(result.get("exit_code"))
    if "call_id" not in receipt:
        return {"state": "unconfirmed"}
    call = modal.FunctionCall.from_id(receipt["call_id"])
    try:
        code = call.get(timeout=0)
    except modal.exception.FunctionTimeoutError:
        return {"state": "failed", "detail": "Modal function timeout"}
    except modal.exception.OutputExpiredError:
        return {"state": "expired", "detail": "Result expired; completion unknown"}
    except TimeoutError:
        return {"state": "pending"}
    return _outcome(code)


def _download(receipt, outputs_dir):
    from modal.volume import FileEntryType

    volume = _volume(receipt)
    root = PurePosixPath(receipt["remote_path"].lstrip("/"))
    local_root = Path(outputs_dir).resolve()
    count = 0
    for entry in volume.iterdir(str(root), recursive=True):
        remote = PurePosixPath(entry.path.lstrip("/"))
        if ".." in remote.parts:
            raise ValueError("Unsafe path in Modal volume listing.")
        try:
            relative = remote.relative_to(root)
        except ValueError:
            raise ValueError("Modal volume listing escaped the run's output directory.") from None
        if not relative.parts or relative.parts[0].casefold() == ".runplz":
            continue  # local control metadata belongs to the launcher, not the workload
        destination = local_root.joinpath(*relative.parts)
        if (
            not destination.resolve().is_relative_to(local_root)
            or destination.resolve() != destination
        ):
            raise ValueError(f"Refusing symlink destination: {destination}")
        if entry.type == FileEntryType.DIRECTORY:
            destination.mkdir(parents=True, exist_ok=True)
        elif entry.type == FileEntryType.FILE:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with _atomic_file(destination) as f:
                volume.read_file_into_fileobj(entry.path, f)
            count += 1
        else:
            raise ValueError(f"Unsupported Modal output file type: {entry.path}")
    return {"files": count}


def _worker(operation, outputs_dir):
    receipt = _read(outputs_dir)
    if operation == "probe":
        return _probe(receipt)
    if operation == "download":
        return _download(receipt, outputs_dir)
    raise ValueError("Unknown Modal worker operation.")


def _bounded_worker(operation, outputs_dir, *, timeout):
    try:
        result = subprocess.run(
            [sys.executable, "-m", "runplz.backends.modal_runs", operation, str(outputs_dir)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(
            f"Modal {operation} unavailable ({exc}); retry without relaunching."
        ) from exc
    if result.returncode:
        raise RuntimeError(
            f"Modal {operation} failed: {result.stderr.strip()[-1000:]}. "
            "Receipt and remote outputs retained; retry without relaunching."
        )
    try:
        response = json.loads(result.stdout)
    except ValueError:
        raise RuntimeError(f"Invalid response from Modal {operation}.") from None
    if not isinstance(response, dict):
        raise RuntimeError(f"Invalid response from Modal {operation}.")
    if operation == "probe":
        if not isinstance(response.get("state"), str) or response["state"] not in _STATES:
            raise RuntimeError("Invalid Modal status response.")
    elif type(response.get("files")) is not int or response["files"] < 0:
        raise RuntimeError("Invalid Modal download response.")
    return response


def _print_identity(receipt):
    print(f"run: {receipt['run_id']} (modal)")
    print(f"launch: {receipt.get('launch_state', 'unknown')}")
    print(f"app: {receipt.get('app_id', 'unconfirmed')}")
    print(f"call: {receipt.get('call_id', 'unconfirmed')}")
    if "app_id" in receipt:
        print(f"logs: modal app logs {receipt['app_id']}")
        print(f"stop: modal app stop {receipt['app_id']}")


def status(outputs_dir):
    """Print a bounded, nonblocking status check of a saved Modal run."""
    receipt = _read(outputs_dir)
    _print_identity(receipt)
    try:
        result = _bounded_worker("probe", outputs_dir, timeout=_PROBE_TIMEOUT)
    except RuntimeError as exc:
        print(f"state: unknown\n{exc}")
        return 2
    print(f"state: {result['state']}")
    if "exit_code" in result:
        print(f"exit code: {result['exit_code']}")
    if "detail" in result:
        print(result["detail"])
    print(f"collect: runplz collect --outputs-dir {shlex.quote(str(outputs_dir))}")
    return 2 if result["state"] in {"expired", "unconfirmed"} else 0


def collect(outputs_dir, *, timeout=600):
    """Collect a finished run, or salvage an expired result without claiming success.

    Returns 0 for successful collection of a successful job, 1 for a failed job,
    2 for unknown outcome, and 3 for a job which is still pending.
    """
    if type(timeout) is not int or timeout <= 0:
        raise ValueError("Collection timeout must be a positive integer.")
    receipt = _read(outputs_dir)
    result = _bounded_worker("probe", outputs_dir, timeout=_PROBE_TIMEOUT)
    state = result["state"]
    if state in {"pending", "unconfirmed"}:
        print(f"Modal run is {state}; no outputs downloaded and no job relaunched.")
        return 3 if state == "pending" else 2
    print(f"Collecting Modal outputs; job {state}, download limit {timeout}s...", flush=True)
    download = _bounded_worker("download", outputs_dir, timeout=timeout)
    # Collection can race a fast job whose spawn acknowledgment is still
    # arriving. Never overwrite the launcher's receipt with an earlier copy.
    _write(
        _receipt_path(outputs_dir).with_name("modal-collection.json"),
        {
            "run_id": receipt["run_id"],
            "collected_at": datetime.now(timezone.utc).isoformat(),
            "outcome": result,
        },
    )
    print(f"Collected {download.get('files', '?')} files into {outputs_dir}; job {state}.")
    return {"succeeded": 0, "failed": 1, "expired": 2}[state]


if __name__ == "__main__":
    try:
        print(json.dumps(_worker(sys.argv[1], Path(sys.argv[2]))))
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(2)
