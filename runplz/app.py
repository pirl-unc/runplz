"""App and Function — Modal-shaped surface.

Intentionally minimal:
  - @app.function(image, gpu, timeout, env) decorates a module-level function
  - fn.local(...) calls it in the current process
  - fn.remote(...) dispatches to the selected backend
  - @app.local_entrypoint() marks the function `runplz` invokes

Args passed to .remote(...) must be JSON-serializable. No closures, no locals.
"""

__all__ = [
    "App",
    "Function",
    "repo_root_for",
    "validate_image_vs_brev_mode",
    "PRECONDITION_KEYS",
]


import inspect
import json
import os
import re
import warnings
from pathlib import Path
from typing import Callable, Optional

from runplz.backends import registry
from runplz.config import AwsConfig, BrevConfig, GcpConfig, ModalConfig, SshConfig
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
        min_gpus: Optional[int] = None,
        preconditions: Optional[dict] = None,
    ):
        # min_gpus is the canonical name going forward (parallels min_cpu /
        # min_memory / min_gpu_memory). num_gpus is the legacy alias — kept
        # so existing scripts don't break. Setting both with conflicting
        # values is rejected so we never silently drop one.
        num_gpus = _coalesce_min_gpus(fn.__name__, min_gpus=min_gpus, num_gpus=num_gpus)
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
        # Minimum number of GPUs. Maps to `brev search --min-gpus N`, Modal's
        # `gpu="A100-80GB:4"` count-suffix syntax, and the SSH backend's
        # spec-mismatch probe. Default 1 means "give me one of whatever
        # `gpu=...` asks for" when `gpu` is set. With gpu=None and no
        # min_gpu_memory, this is ignored (we don't allocate GPU-less
        # multi-GPU boxes). The legacy `num_gpus` attribute is kept as
        # an alias so existing dispatch code (brev / modal) keeps working.
        self.num_gpus = num_gpus
        self.min_gpus = num_gpus
        self.timeout = timeout
        self.env = _normalize_env(fn.__name__, env)
        # Preconditions: declarative remote-state requirements probed *after*
        # rsync_up but *before* bootstrap, so a misprovisioned box (small
        # /dev/shm, full disk, missing GPU) fails fast instead of wasting
        # paid GPU minutes on a doomed run. See runplz/backends/ssh_common.py
        # `check_preconditions`. v1 keys: shm_gb, disk_free_gb, gpu_count,
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
        # Show a CWD-relative path when possible — bare absolute paths are
        # noisy (often /Users/.../code/proj/jobs/long_name.py) and obscure
        # the actually-actionable suggestion. Falls back to basename when
        # the file lives outside the cwd (rare; e.g. installed examples).
        try:
            display_path = str(Path(self.module_file).relative_to(Path.cwd()))
        except ValueError:
            display_path = Path(self.module_file).name
        raise RuntimeError(
            f"Plain-calling a runplz Function is intentionally disabled. Use "
            f"`{self.name}.local(*args, **kwargs)` to run it in this process, "
            f"or `{self.name}.remote(*args, **kwargs)` to dispatch it to the "
            f"backend selected on the command line "
            f"(e.g. `runplz brev --instance <box> {display_path}`)."
        )


class App:
    def __init__(
        self,
        name: str,
        *,
        brev_config: Optional[BrevConfig] = None,
        modal_config: Optional[ModalConfig] = None,
        ssh_config: Optional[SshConfig] = None,
        gcp_config: Optional[GcpConfig] = None,
        aws_config: Optional[AwsConfig] = None,
    ):
        self.name = name
        self.brev_config = brev_config or BrevConfig()
        self.modal_config = modal_config or ModalConfig()
        self.ssh_config = ssh_config or SshConfig()
        # Provisioning clouds have required fields, so they stay None
        # until the user supplies one. bind() raises a pointed error
        # rather than constructing an invalid default.
        self.gcp_config = gcp_config
        self.aws_config = aws_config
        self.functions: dict[str, Function] = {}
        self.entrypoint: Optional[Callable] = None

        # Runtime-populated by the CLI before local_entrypoint fires.
        self._backend: Optional[str] = None
        self._backend_kwargs: dict = {}
        # Two values, because three things can set a repo root and they do
        # not have the same lifetime:
        #   _repo_root_value    what dispatch uses (per-bind)
        #   _repo_root_assigned a standing choice made via `app.repo_root = X`
        # A bind(repo_root=...) argument overrides for that call only; it must
        # neither erase a standing choice nor survive into the next bind.
        self._repo_root_value: Optional[Path] = None
        self._repo_root_assigned: Optional[Path] = None

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
        min_gpus: Optional[int] = None,
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
                min_gpus=min_gpus,
                timeout=timeout,
                env=env or {},
                preconditions=preconditions,
            )
            self.functions[f.name] = f
            return f

        return decorator

    def local_entrypoint(self):
        def decorator(fn: Callable) -> Callable:
            if self.entrypoint is not None:
                # Last-wins would leave the first driver unreachable with no
                # output at all — the CLI runs one and never mentions the
                # other. Every comparable ambiguity here is an error: multiple
                # Apps in a script, or multiple @app.function with no
                # entrypoint to disambiguate. Issue #87.
                raise ValueError(
                    f"{self.name} already has an @app.local_entrypoint "
                    f"({self.entrypoint.__name__}); only one is allowed. "
                    f"Remove one, or have it call the other."
                )
            self.entrypoint = fn
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
        repo_root: Optional[Path] = None,
    ) -> "App":
        """Attach a backend to this App from pure Python, no CLI needed.

        Args:
          backend: one of `runplz.backends.registry.names()` — currently
            local, brev, modal, ssh, gcp, aws.
          instance: required for `backend="brev"`; rejected for others.
          host: required for `backend="ssh"`; rejected for others. The
            ssh endpoint (hostname, user@host, or an ssh config alias).
          outputs_dir: host dir to collect `/out` into. Applies to all backends.
          build: local-only. `False` skips `docker build` and reuses the last
            tagged image. Rejected for non-local backends (Brev rebuilds on
            the remote; Modal manages its own layer cache).
          repo_root: skip the git lookup and use this. The CLI knows the
            script being run, which is more authoritative than the module a
            function happens to be defined in — and it saves a second
            `git rev-parse`.

        Use from a `if __name__ == "__main__":` guard in your script:

            app.bind("local")
            app.bind("brev", instance="my-gpu-box")
            app.bind("ssh",  host="gpu.example.com")
            train.remote()

        …which is what `runplz <backend> jobs/train.py` does under the hood.
        The CLI is preferred for CI/shared scripts; this is for notebooks
        and one-off runs where you already have `app` in scope.
        """
        spec = registry.get(backend)
        if spec.required_config_attr and getattr(self, spec.required_config_attr) is None:
            raise ValueError(
                f"backend={backend!r} needs App(..., {spec.required_config_attr}=...). "
                f"It provisions a box for you, so it needs to know where: "
                f"project/zone for gcp, region/key_name for aws."
            )
        if instance is not None and not spec.accepts_instance:
            raise ValueError(
                f"--instance / instance=... only applies to the brev backend "
                f"(got backend={backend!r})."
            )
        if spec.accepts_host and not host:
            raise ValueError(
                "the ssh backend needs a host: pass --host <target> on the "
                "command line, or host=... to App.bind()."
            )
        if host is not None and not spec.accepts_host:
            raise ValueError(
                f"--host / host=... only applies to the ssh backend (got backend={backend!r})."
            )
        if not build and not spec.accepts_no_build:
            raise ValueError(
                f"--no-build / build=False only applies to the local backend "
                f"(it skips `docker build`). On backend={backend!r} it would be "
                f"silently ignored."
            )
        if not outputs_dir or not str(outputs_dir).strip():
            raise ValueError("outputs_dir must be a non-empty path string.")
        # Only needed to *infer* the repo root — with one handed in, or a
        # standing assignment, there is nothing to locate.
        if not self.functions and repo_root is None and self._repo_root_assigned is None:
            raise RuntimeError(
                "App.bind() needs at least one @app.function() declared so we "
                "can locate the script's repo root, or an explicit repo_root."
            )
        # Recomputed from scratch on every bind, in precedence order, so no
        # branch can leave a stale value behind from a previous call.
        if repo_root is not None:
            self._repo_root_value = _coerce_repo_root(repo_root, "repo_root")
        elif self._repo_root_assigned is not None:
            self._repo_root_value = self._repo_root_assigned
        else:
            any_fn = next(iter(self.functions.values()))
            self._repo_root_value = repo_root_for(Path(any_fn.module_file))
        self._backend = backend
        self._backend_kwargs = {"outputs_dir": outputs_dir}
        # Which selector each backend takes comes from the registry too, so
        # this stays right when a backend is added.
        if spec.accepts_instance:
            self._backend_kwargs["instance"] = instance
        if spec.accepts_host:
            self._backend_kwargs["host"] = host
        if spec.accepts_no_build and not build:
            self._backend_kwargs["build"] = False
        return self

    @property
    def repo_root(self) -> Optional[Path]:
        """Local directory staged to the remote.

        Normally inferred from the job script's location. Assign it to
        override; the value is resolved to an absolute path, so a string or a
        relative path is accepted and fails here rather than deep inside
        dispatch after a box has been paid for.
        """
        return self._repo_root_value

    @repo_root.setter
    def repo_root(self, value) -> None:
        if value is None:
            self._repo_root_value = None
            self._repo_root_assigned = None
            return
        resolved = _coerce_repo_root(value, "App.repo_root")
        self._repo_root_value = resolved
        # A standing choice: it outlives any single bind(), unlike a
        # bind(repo_root=...) argument.
        self._repo_root_assigned = resolved

    @property
    def _entrypoint(self):
        """Deprecated alias for :attr:`entrypoint` (renamed in 3.20.0).

        Kept because the failure without it is silent, not loud: a script
        doing `app._entrypoint = driver` would leave `entrypoint` unset, and
        the CLI would synthesize a default from the single @app.function and
        dispatch *that* instead — a different job, no error.
        """
        return self.entrypoint

    @_entrypoint.setter
    def _entrypoint(self, value) -> None:
        warnings.warn(
            "App._entrypoint was renamed to App.entrypoint in 3.20.0; "
            "assign that instead. This alias goes away in 4.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.entrypoint = value

    @property
    def _repo_root(self) -> Optional[Path]:
        """Deprecated alias for :attr:`repo_root` (renamed in 3.20.0).

        Assigning the old name used to write the field directly, which now
        also means skipping the validation the public setter added.
        """
        return self._repo_root_value

    @_repo_root.setter
    def _repo_root(self, value) -> None:
        warnings.warn(
            "App._repo_root was renamed to App.repo_root in 3.20.0; assign "
            "that instead. This alias goes away in 4.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.repo_root = value

    def require_repo_root(self, *, context: str = "dispatch") -> Path:
        """Return `repo_root`, or raise if it is unset.

        One definition of the invariant, called from `_dispatch` and from each
        backend's `run()`. It used to be re-derived and re-worded in four
        places, so the same state produced different guidance depending on
        which entry point you came through.
        """
        if self._repo_root_value is None:
            raise RuntimeError(
                f"{context} needs App.repo_root, which is not set. Dispatch "
                "through the `runplz` CLI or call App.bind(...), either of "
                "which sets it."
            )
        return self._repo_root_value

    def _dispatch(self, function: Function, args: list, kwargs: dict):
        if self._backend is None:
            raise RuntimeError(
                f"{function.name}.remote(...) was called but no backend is "
                "selected. runplz Functions dispatch via the `runplz` CLI, "
                "which binds a backend before invoking @local_entrypoint. "
                f"Run: `runplz <{'|'.join(registry.names())}> {function.module_file}`. "
                f"(For in-process execution without a backend, use "
                f"{function.name}.local(...) instead.)"
            )
        # registry.load() first: an unknown backend is the more specific
        # error, and load() only imports a module -- nothing is provisioned.
        module = registry.load(self._backend)
        # Checked here, not in a backend: every backend funnels through this
        # method, and the provisioning ones would otherwise create and pay for
        # a box before noticing that no repo could be staged to it.
        self.require_repo_root(context=f"{function.name}.remote(...)")
        # The backends compute the script's path *relative to* repo_root to
        # find it on the remote. If it is not underneath, `relative_to` raises
        # -- but only after a provisioning backend has created a paid box,
        # waited for ssh and rsynced the whole tree up. Check it here instead.
        script = Path(function.module_file).resolve()
        if not script.is_relative_to(self.repo_root):
            raise ValueError(
                f"App.repo_root ({self.repo_root}) does not contain "
                f"{function.name}'s script ({script}). runplz stages repo_root "
                f"to the remote and locates the script inside it, so the script "
                f"must live under it."
            )
        return module.run(self, function, args, kwargs, **self._backend_kwargs)


def _ensure_json_safe(args, kwargs):
    try:
        json.dumps([list(args), dict(kwargs)])
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "Function.remote(...) args must be JSON-serializable. "
            "Use primitives/lists/dicts, not closures or custom objects."
        ) from exc


def _coerce_repo_root(value, label: str) -> Path:
    """Resolve a repo root to an absolute directory, or fail here.

    `bind()` already validates `outputs_dir` this way. Without it,
    `repo_root = ""` resolves to the process CWD and silently rsyncs whatever
    the caller happened to be sitting in — a home directory, or `/` — up to a
    remote box, and a typo'd path is only noticed after provisioning.
    """
    # os.fspath so a PathLike is checked too, not only `str`. Note the limit:
    # `Path("")` is already `Path(".")` at construction, so it is
    # indistinguishable from a deliberate `"."` here and resolves to the CWD.
    # Only the raw empty/whitespace string can be caught.
    if not os.fspath(value).strip():
        raise ValueError(f"{label} must be a non-empty path")
    path = Path(value).resolve()
    if not path.is_dir():
        raise ValueError(f"{label} must be an existing directory, got {path}")
    return path


def repo_root_for(script_path: Path) -> Path:
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
    if not isinstance(num_gpus, int) or num_gpus < 1:
        raise ValueError(
            f"@app.function({fn_name}): min_gpus / num_gpus must be a positive int; "
            f"got {num_gpus!r}."
        )
    # Multi-GPU without an explicit model is fine *if* the user has at least
    # told the selector what kind of GPU to look for (via min_gpu_memory).
    # Otherwise the request is too vague — refuse to provision random GPUs.
    if num_gpus > 1 and gpu is None and min_gpu_memory is None:
        raise ValueError(
            f"@app.function({fn_name}): min_gpus={num_gpus} without gpu=... "
            "needs at least min_gpu_memory= so the selector can filter sensibly."
        )
    if not isinstance(timeout, int) or timeout <= 0:
        raise ValueError(
            f"@app.function({fn_name}): timeout must be a positive int (seconds); got {timeout!r}."
        )


# Precondition keys we know how to probe. Adding a new key is a two-place
# change: list it here so user-supplied values are validated, then teach
# `check_preconditions` how to probe and compare it.
PRECONDITION_KEYS = ("shm_gb", "disk_free_gb", "gpu_count", "gpu_memory_gb")


def _coalesce_min_gpus(fn_name: str, *, min_gpus: Optional[int], num_gpus: int) -> int:
    """Reconcile the new ``min_gpus`` and the legacy ``num_gpus`` kwargs.

    `num_gpus` defaults to 1; treat that default as "unset" so a user passing
    only `min_gpus=4` doesn't trip a spurious conflict. If both are passed
    explicitly with different values, refuse silently picking one — the user
    is asking for two different things.
    """
    if min_gpus is None:
        return num_gpus
    if not isinstance(min_gpus, int) or min_gpus < 1:
        raise ValueError(
            f"@app.function({fn_name}): min_gpus must be a positive int; got {min_gpus!r}."
        )
    if num_gpus != 1 and num_gpus != min_gpus:
        raise ValueError(
            f"@app.function({fn_name}): conflicting min_gpus={min_gpus} and "
            f"num_gpus={num_gpus}. Pass only one (prefer min_gpus going forward)."
        )
    return min_gpus


_ENV_KEY_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _normalize_env(fn_name: str, raw: Optional[dict]) -> dict:
    """Validate the user's `env=` dict and return a clean copy.

    Every entry is rendered into remote shell as ``export KEY=<quoted value>``.
    Values are quoted; keys cannot be, because a shell variable name is not a
    word that quoting applies to. A key that is not a valid identifier
    therefore produces shell that *parses* — so `sh -n` does not catch it —
    and fails at runtime::

        $ bash -c 'set -euo pipefail; export MY-VAR=x; echo reached'
        bash: export: `MY-VAR=x': not a valid identifier   # exit 1

    The remote command runs under ``set -euo pipefail``, so that aborts the
    whole job, after the box has been provisioned and the repo rsynced, with
    an error naming neither runplz nor the offending key. Checked here
    instead: the laptop knows everything it needs to. Issue #88.
    """
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"@app.function({fn_name}): env must be a dict; got {type(raw).__name__}.")
    cleaned: dict = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not _ENV_KEY_RE.fullmatch(key):
            raise ValueError(
                f"@app.function({fn_name}): env key {key!r} is not a valid shell "
                f"identifier. Use letters, digits and underscores, not starting "
                f"with a digit — the remote `export` would fail mid-run otherwise."
            )
        cleaned[key] = value
    return cleaned


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
