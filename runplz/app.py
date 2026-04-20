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

from runplz.config import BrevConfig, ModalConfig
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
    ):
        _validate_resources(
            fn_name=fn.__name__,
            gpu=gpu,
            min_cpu=min_cpu,
            min_memory=min_memory,
            min_gpu_memory=min_gpu_memory,
            min_disk=min_disk,
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
        self.timeout = timeout
        self.env = dict(env or {})
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
    ):
        self.name = name
        self.brev_config = brev_config or BrevConfig()
        self.modal_config = modal_config or ModalConfig()
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
        timeout: int = 60 * 60,
        env: Optional[dict] = None,
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
                timeout=timeout,
                env=env or {},
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
        outputs_dir: str = "out",
        build: bool = True,
    ) -> "App":
        """Attach a backend to this App from pure Python, no CLI needed.

        Args:
          backend: "local", "brev", or "modal".
          instance: required for `backend="brev"`; rejected for others.
          outputs_dir: host dir to collect `/out` into. Applies to all backends.
          build: local-only. `False` skips `docker build` and reuses the last
            tagged image. Rejected for non-local backends (Brev rebuilds on
            the remote; Modal manages its own layer cache).

        Use from a `if __name__ == "__main__":` guard in your script:

            app.bind("local")
            train.remote()

        …which is what `runplz local jobs/train.py` does under the hood.
        The CLI is still the preferred invocation for CI/shared scripts —
        this is for notebooks and one-off runs where you already have
        `app` in scope.
        """
        if backend not in ("local", "brev", "modal"):
            raise ValueError(f"backend must be 'local', 'brev', or 'modal'; got {backend!r}")
        if backend == "brev" and not instance:
            raise ValueError("instance=... is required when backend='brev'")
        if backend != "brev" and instance is not None:
            raise ValueError(
                f"instance={instance!r} only applies to backend='brev'; got backend={backend!r}."
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
    if not isinstance(timeout, int) or timeout <= 0:
        raise ValueError(
            f"@app.function({fn_name}): timeout must be a positive int (seconds); got {timeout!r}."
        )
