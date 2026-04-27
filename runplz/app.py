"""App and Function — Modal-shaped surface.

Intentionally minimal:
  - @app.function(image, gpu, timeout, env) decorates a module-level function
  - fn.local(...) calls it in the current process
  - fn.remote(...) dispatches to the selected backend
  - @app.local_entrypoint() marks the function `runplz` invokes

Args passed to .remote(...) must be JSON-serializable. No closures, no locals.
"""

import inspect
import json
from pathlib import Path
from typing import Callable, Optional

from runplz.config import BrevConfig, ModalConfig, SshConfig
from runplz.image import Image


class Function:
    def __init__(
        self,
        app: "App",
        fn: Callable,
        *,
        image: Image,
        gpu: Optional[str],
        timeout: int,
        env: dict,
        min_cpu: Optional[float] = None,
        min_memory: Optional[float] = None,
        min_gpu_memory: Optional[float] = None,
        min_disk: Optional[float] = None,
        num_gpus: int = 1,
        preconditions: Optional[dict] = None,
    ):
        _validate_resources(
            fn_name=fn.__name__,
            gpu=gpu,
            min_cpu=min_cpu,
            min_memory=min_memory,
            min_gpu_memory=min_gpu_memory,
            min_disk=min_disk,
            num_gpus=num_gpus,
            timeout=timeout,
        )
        self.app = app
        self.fn = fn
        self.image = image
        # Resource requests — all minimums. Units: vCPUs (float OK), GB for
        # everything memory/disk-related. Each backend picks a matching
        # instance (Modal: direct; Brev: via `brev search`).
        #
        # `gpu`: exact GPU name (one of Modal's accepted labels). Common:
        #   - "T4"            Turing,   16 GB,   sm_75
        #   - "L4"            Ada,      24 GB,   sm_89
        #   - "L40S"          Ada,      48 GB,   sm_89
        #   - "A10" / "A10G"  Ampere,   24 GB,   sm_86
        #   - "A100-40GB"     Ampere,   40 GB,   sm_80
        #   - "A100-80GB"     Ampere,   80 GB,   sm_80
        #   - "H100"          Hopper,   80 GB,   sm_90
        #   - "H200"          Hopper,  141 GB,   sm_90
        #   - "V100"          Volta,    16 GB,   sm_70
        self.gpu = gpu
        self.min_cpu = min_cpu  # vCPUs (float for fractional on Modal)
        self.min_memory = min_memory  # GB of RAM
        self.min_gpu_memory = min_gpu_memory  # GB of VRAM per GPU
        self.min_disk = min_disk  # GB of disk
        # Number of GPUs. Maps to `brev search --min-gpus N`, Modal's
        # `gpu="A100-80GB:4"` count-suffix syntax, and the SSH backend's
        # spec-mismatch probe. Default 1 means "give me one of whatever
        # `gpu=...` asks for" when `gpu` is set. With gpu=None, this is
        # ignored (we don't allocate GPU-less multi-GPU boxes).
        self.num_gpus = num_gpus
        self.timeout = timeout
        self.env = dict(env or {})
        # Preconditions: declarative remote-state requirements probed *after*
        # rsync_up but *before* bootstrap, so a misprovisioned box (small
        # /dev/shm, full disk, missing GPU) fails fast instead of wasting
        # paid GPU minutes on a doomed run. See runplz/backends/_ssh_common.py
        # `_check_preconditions`. v1 keys: shm_gb, disk_free_gb, gpu_count,
        # gpu_memory_gb. Issue #56.
        self.preconditions = _normalize_preconditions(fn.__name__, preconditions)
        self.name = fn.__name__
        self.module_file = str(Path(inspect.getfile(fn)).resolve())

    def local(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

    def remote(self, *args, **kwargs):
        _ensure_json_safe(args, kwargs)
        return self.app._dispatch(self, list(args), dict(kwargs))

    def __call__(self, *args, **kwargs):
        raise RuntimeError(
            f"Plain-calling a runplz Function is intentionally disabled. Use "
            f"`{self.name}.local(*args, **kwargs)` to run it in this process, "
            f"or `{self.name}.remote(*args, **kwargs)` to dispatch it to the "
            f"backend selected on the command line "
            f"(e.g. `runplz brev --instance <box> {self.module_file}`)."
        )


class App:
    def __init__(
        self,
        name: str,
        *,
        brev_config: Optional[BrevConfig] = None,
        modal_config: Optional[ModalConfig] = None,
        ssh_config: Optional[SshConfig] = None,
    ):
        self.name = name
        self.brev_config = brev_config or BrevConfig()
        self.modal_config = modal_config or ModalConfig()
        self.ssh_config = ssh_config or SshConfig()
        self.functions: dict[str, Function] = {}
        self._entrypoint: Optional[Callable] = None

        # Runtime-populated by the CLI before local_entrypoint fires.
        self._backend: Optional[str] = None
        self._backend_kwargs: dict = {}
        self._repo_root: Optional[Path] = None

    def function(
        self,
        *,
        image: Image,
        gpu: Optional[str] = None,
        min_cpu: Optional[float] = None,
        min_memory: Optional[float] = None,
        min_gpu_memory: Optional[float] = None,
        min_disk: Optional[float] = None,
        num_gpus: int = 1,
        timeout: int = 60 * 60,
        env: Optional[dict] = None,
        preconditions: Optional[dict] = None,
    ):
        def decorator(fn: Callable) -> Function:
            f = Function(
                self,
                fn,
                image=image,
                gpu=gpu,
                min_cpu=min_cpu,
                min_memory=min_memory,
                min_gpu_memory=min_gpu_memory,
                min_disk=min_disk,
                num_gpus=num_gpus,
                timeout=timeout,
                env=env or {},
                preconditions=preconditions,
            )
            self.functions[f.name] = f
            return f

        return decorator

    def local_entrypoint(self):
        def decorator(fn: Callable) -> Callable:
            self._entrypoint = fn
            return fn

        return decorator

    def bind(
        self,
        backend: str,
        *,
        instance: Optional[str] = None,
        host: Optional[str] = None,
        outputs_dir: str = "out",
        build: bool = True,
    ) -> "App":
        """Attach a backend to this App from pure Python, no CLI needed.

        Args:
          backend: "local", "brev", "ssh", or "modal".
          instance: required for `backend="brev"`; rejected for others.
          host: required for `backend="ssh"`; rejected for others. The
            ssh endpoint (hostname, user@host, or an ssh config alias).
          outputs_dir: host dir to collect `/out` into. Applies to all backends.
          build: local-only. `False` skips `docker build` and reuses the last
            tagged image. Rejected for non-local backends (Brev rebuilds on
            the remote; Modal manages its own layer cache).

        Use from a `if __name__ == "__main__":` guard in your script:

            app.bind("local")
            app.bind("brev", instance="my-gpu-box")
            app.bind("ssh",  host="gpu.example.com")
            train.remote()

        …which is what `runplz <backend> jobs/train.py` does under the hood.
        The CLI is preferred for CI/shared scripts; this is for notebooks
        and one-off runs where you already have `app` in scope.
        """
        if backend not in ("local", "brev", "modal", "ssh"):
            raise ValueError(f"backend must be 'local', 'brev', 'modal', or 'ssh'; got {backend!r}")
        # brev accepts instance=None → ephemeral mode (runplz auto-creates
        # a box sized to the function and deletes it on exit).
        if backend != "brev" and instance is not None:
            raise ValueError(
                f"instance={instance!r} only applies to backend='brev'; got backend={backend!r}."
            )
        if backend == "ssh" and not host:
            raise ValueError("host=... is required when backend='ssh'")
        if backend != "ssh" and host is not None:
            raise ValueError(
                f"host={host!r} only applies to backend='ssh'; got backend={backend!r}."
            )
        if backend != "local" and not build:
            raise ValueError(
                f"build=False only applies to backend='local' (it skips `docker "
                f"build`). On backend={backend!r} it would be silently ignored."
            )
        if not outputs_dir or not str(outputs_dir).strip():
            raise ValueError("outputs_dir must be a non-empty path string.")
        if not self.functions:
            raise RuntimeError(
                "App.bind() needs at least one @app.function() declared so we "
                "can locate the script's repo root."
            )
        any_fn = next(iter(self.functions.values()))
        self._repo_root = _repo_root_for(Path(any_fn.module_file))
        self._backend = backend
        self._backend_kwargs = {"outputs_dir": outputs_dir}
        if backend == "brev":
            self._backend_kwargs["instance"] = instance
        if backend == "ssh":
            self._backend_kwargs["host"] = host
        if backend == "local" and not build:
            self._backend_kwargs["build"] = False
        return self

    def _dispatch(self, function: Function, args: list, kwargs: dict):
        if self._backend is None:
            raise RuntimeError(
                f"{function.name}.remote(...) was called but no backend is "
                "selected. runplz Functions dispatch via the `runplz` CLI, "
                "which binds a backend before invoking @local_entrypoint. "
                f"Run: `runplz <local|brev|modal> {function.module_file}`. "
                f"(For in-process execution without a backend, use "
                f"{function.name}.local(...) instead.)"
            )
        backend = self._backend
        if backend == "local":
            from runplz.backends import local

            return local.run(self, function, args, kwargs, **self._backend_kwargs)
        if backend == "brev":
            from runplz.backends import brev

            return brev.run(self, function, args, kwargs, **self._backend_kwargs)
        if backend == "modal":
            from runplz.backends import modal

            return modal.run(self, function, args, kwargs, **self._backend_kwargs)
        if backend == "ssh":
            from runplz.backends import ssh

            return ssh.run(self, function, args, kwargs, **self._backend_kwargs)
        raise ValueError(f"Unknown backend: {backend!r}")


def _ensure_json_safe(args, kwargs):
    try:
        json.dumps([list(args), dict(kwargs)])
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "Function.remote(...) args must be JSON-serializable. "
            "Use primitives/lists/dicts, not closures or custom objects."
        ) from exc


def _repo_root_for(script_path: Path) -> Path:
    for parent in [script_path.parent, *script_path.parents]:
        if (parent / ".git").exists():
            return parent
    return script_path.parent


def validate_image_vs_brev_mode(*, fn_name: str, image: Image, brev_config: BrevConfig):
    """Catch image/Brev-mode mismatches before we ssh anywhere.

    Called from the Brev backend's `run()` entrypoint — not at function
    decoration — because local/modal users shouldn't be constrained by
    the Brev config on a shared App. A Dockerfile image is fine with
    Modal and local regardless of what `brev_config.mode` says.
    """
    if image.dockerfile is None:
        return  # registry-based images work with every mode
    if brev_config.mode == "container":
        raise ValueError(
            f"@app.function({fn_name}): BrevConfig(mode='container') requires "
            f"Image.from_registry(...). Image.from_dockerfile(...) can't "
            f"translate to inline installs on a container-mode Brev box. "
            f"Either switch the image to Image.from_registry(...) + DSL ops, "
            f"or set brev_config=BrevConfig(mode='vm')."
        )
    if brev_config.mode == "vm" and not brev_config.use_docker:
        raise ValueError(
            f"@app.function({fn_name}): BrevConfig(mode='vm', use_docker=False) "
            f"runs the function natively over ssh and ignores any Dockerfile. "
            f"Use Image.from_registry(...) or flip use_docker=True."
        )


def _validate_resources(
    *,
    fn_name: str,
    gpu: Optional[str],
    min_cpu: Optional[float],
    min_memory: Optional[float],
    min_gpu_memory: Optional[float],
    min_disk: Optional[float],
    num_gpus: int,
    timeout: int,
):
    if gpu is not None and (not isinstance(gpu, str) or not gpu.strip()):
        raise ValueError(f"@app.function({fn_name}): gpu must be a non-empty string or None.")
    positive = {
        "min_cpu": min_cpu,
        "min_memory": min_memory,
        "min_gpu_memory": min_gpu_memory,
        "min_disk": min_disk,
    }
    for label, value in positive.items():
        if value is not None and value <= 0:
            raise ValueError(
                f"@app.function({fn_name}): {label} must be > 0 when set; got {value!r}."
            )
    if min_gpu_memory is not None and gpu is None:
        raise ValueError(
            f"@app.function({fn_name}): min_gpu_memory={min_gpu_memory} requires gpu=... "
            "(can't filter VRAM without asking for a GPU)."
        )
    if not isinstance(num_gpus, int) or num_gpus < 1:
        raise ValueError(
            f"@app.function({fn_name}): num_gpus must be a positive int; got {num_gpus!r}."
        )
    if num_gpus > 1 and gpu is None:
        raise ValueError(
            f"@app.function({fn_name}): num_gpus={num_gpus} requires gpu=... "
            "(can't request multiple GPUs without a model)."
        )
    if not isinstance(timeout, int) or timeout <= 0:
        raise ValueError(
            f"@app.function({fn_name}): timeout must be a positive int (seconds); got {timeout!r}."
        )


# Precondition keys we know how to probe. Adding a new key is a two-place
# change: list it here so user-supplied values are validated, then teach
# `_check_preconditions` how to probe and compare it.
PRECONDITION_KEYS = ("shm_gb", "disk_free_gb", "gpu_count", "gpu_memory_gb")


def _normalize_preconditions(fn_name: str, raw: Optional[dict]) -> dict:
    """Validate the user's `preconditions=` dict and return a clean copy.

    Rejects unknown keys (typos like ``shm_gib`` would silently no-op
    otherwise) and non-positive values.
    """
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"@app.function({fn_name}): preconditions must be a dict; got {type(raw).__name__}."
        )
    cleaned: dict = {}
    for key, value in raw.items():
        if key not in PRECONDITION_KEYS:
            raise ValueError(
                f"@app.function({fn_name}): unknown precondition key {key!r}. "
                f"Supported: {', '.join(PRECONDITION_KEYS)}."
            )
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(
                f"@app.function({fn_name}): precondition {key}={value!r} must be a positive number."
            )
        cleaned[key] = float(value)
    return cleaned
