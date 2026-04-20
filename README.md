# runplz

[![PyPI](https://img.shields.io/pypi/v/runplz.svg)](https://pypi.org/project/runplz/)

Tiny Modal-shaped job harness — one Python decoration, multiple backends.

```python
# jobs/train.py
from runplz import App, BrevConfig, Image

app = App("my-job", brev=BrevConfig())  # auto-creates the Brev box if missing

image = (
    Image.from_registry("pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime")
    .apt_install("rsync", "build-essential")
    .pip_install("pandas>=2.0", "scikit-learn")
    .pip_install_local_dir(".", editable=True)
)

@app.function(
    image=image,
    gpu="T4",
    min_cpu=4, min_memory=26, min_gpu_memory=16, min_disk=100,
    timeout=60 * 60,
)
def train():
    import subprocess
    subprocess.run(["bash", "scripts/train.sh"], check=True)

@app.local_entrypoint()
def main():
    train.remote()
```

Invoke via the CLI:

```bash
runplz local  jobs/train.py
runplz brev   --instance my-box jobs/train.py
runplz modal  jobs/train.py
```

…or from pure Python (notebook, REPL, `python jobs/train.py`):

```python
# at the bottom of jobs/train.py
if __name__ == "__main__":
    app.bind("brev", instance="my-box")   # or "local" / "modal"
    train.remote()
```

`app.bind(...)` is the programmatic equivalent of the CLI — it attaches a
backend (plus the same flags: `instance=`, `outputs_dir=`, `build=`) so
`.remote()` knows where to dispatch.

## How it's structured

**The CLI is the only entry point.** `runplz <backend> <script>` does three
things:

1. Imports your script (finds the `App` instance at module scope).
2. Binds the chosen backend to that `App` (the reason `python script.py`
   won't work — nothing has told the `App` where to dispatch).
3. Calls whatever you've decorated with `@app.local_entrypoint()`.

Inside that entrypoint you call `train.remote()`, which serializes a
minimal dispatch (env vars + a path to your script) and runs on the
selected backend. Args and kwargs must be JSON-serializable.

### Decorators you'll use

- **`@app.function(image=..., gpu=..., ...)`** — marks a function as
  running *on the backend*. Its body never executes locally (unless you
  call `.local()`; see below).
- **`@app.local_entrypoint()`** — marks the driver that runs *inside the
  CLI process*, on your machine. Typical body: build args, call
  `fn.remote(...)` once, maybe inspect the result. There can be at most
  one per script.

### Ways to invoke a function

- `train.remote(...)` → dispatch on the currently-selected backend (what
  the CLI set). This is the normal case.
- `train.local(...)` → run the body *in this Python process*. No
  container, no remote. Useful for `pytest` or a quick REPL sanity check
  where you don't want to shell out to docker/brev/modal.
- `train(...)` → raises. Always go through `.remote()` or `.local()` so
  the dispatch is explicit.

### What the CLI flags do

- `--instance <name>` — **required** for `brev`; the Brev box to attach
  to. If it doesn't exist and `BrevConfig(auto_create_instances=True)`
  (the default), runplz provisions it for you (cheapest match for your
  resource constraints, or an explicit `BrevConfig(instance_type=...)`
  if you pinned one).
- `--no-build` — **local only**; reuse the last tagged docker image
  instead of rebuilding.
- `--outputs-dir <path>` — where to collect `/out` back to on the host
  (default `./out/`).

All three have `app.bind(...)` equivalents (`instance=`, `build=False`,
`outputs_dir=`) for the pure-Python invocation path.

## Backend config

`App(..., brev=BrevConfig(...), modal=ModalConfig(...))`. Both default
to instances of their respective config class, so you only pass one when
you need to override something.

### BrevConfig

All fields are validated at construction time — an invalid config raises
`ValueError` immediately, not later during dispatch.

| field                    | default | what it does                                                                                                                                      |
| ------------------------ | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `auto_create_instances`  | `True`  | If `--instance` points at a non-existent box, `brev create` it. Set `False` to hard-fail instead of auto-provisioning.                            |
| `instance_type`          | `None`  | Pin a specific Brev instance type string (e.g. `"n1-standard-4:nvidia-tesla-t4:1"`). Skips the constraint-based picker.                           |
| `mode`                   | `"vm"`  | `"vm"` = full Brev VM + docker-in-VM. `"container"` = the box IS the base image; runplz applies Image DSL ops inline over ssh (lighter, no DinD). |
| `use_docker`             | `True`  | VM-mode only. `False` skips docker and installs a native venv on the box. Legacy escape hatch for providers where container mode isn't available. |

Invalid combinations (rejected at construction):

- `mode` not in `{"vm", "container"}`
- `mode="container"` with `use_docker=False` (contradictory — the box *is* the image)
- `instance_type=""` (must be a non-empty string or `None`)

### ModalConfig

`ModalConfig()` is a no-op today. Modal reads auth from `~/.modal.toml`
and schedules resources from `@app.function(gpu=..., cpu=..., memory=...)`;
we don't expose Modal-specific knobs. The class exists as a slot in
`App(modal=...)` so the signature doesn't break when fields are added.

### Why not one unified config?

Surveyed the fields — there is no genuine overlap today. Brev has real
provisioning knobs (mode, instance type, docker-or-native); Modal has
nothing we expose. A shared base class would be empty. If/when a
genuinely cross-backend concept shows up (e.g. per-App secrets, a shared
retry policy), we'll factor it into a `BaseConfig` then. Until then, the
split is the honest API.

## Image DSL

Declared once, translated per backend:

```python
Image.from_registry("pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime")
    .apt_install("bzip2", "rsync")
    .pip_install("pandas>=2.0", index_url="https://...")
    .pip_install_local_dir(".", editable=True)
    .run_commands("echo hi")
```

- **Modal** — rendered as a `modal.Image.from_registry(...)` chain; layers
  build on Modal's cluster and cache per-hash.
- **local** — synthesized into a Dockerfile passed to `docker build -f -`
  with the repo as context (so `pip_install_local_dir` can `COPY` your
  source).
- **Brev (mode=vm)** — same Dockerfile synthesis, shipped over rsync and
  built on the remote box.
- **Brev (mode=container)** — the box IS the base image; the layer ops
  run inline over ssh. Lighter, and sidesteps a historical Brev GPU+docker
  flakiness (see [`docs/brev-ssh-bug-report.md`](docs/brev-ssh-bug-report.md)).

You can also use `Image.from_dockerfile("path/to/Dockerfile")` to point at
an existing Dockerfile you maintain; runplz just runs it.

## Resource constraints

All memory/disk fields in **GB**:

```python
@app.function(
    image=image,
    gpu="T4",            # modal-style label; "A100", "H100", "L4", ...
    min_cpu=4,
    min_memory=26,       # RAM
    min_gpu_memory=16,   # VRAM
    min_disk=100,
    timeout=60 * 60,
)
```

How they're honored per backend:

| constraint        | local | brev                        | modal                         |
| ----------------- | ----- | --------------------------- | ----------------------------- |
| `gpu`             |  —    | `brev search --gpu-name`    | `@app.function(gpu=...)`      |
| `min_cpu`         |  —    | `--min-vcpu`                | `cpu=`                        |
| `min_memory`      |  —    | `--min-ram`                 | `memory=` (converted to MB)   |
| `min_gpu_memory`  |  —    | `--min-vram`                | baked into gpu string: `A100-80GB` |
| `min_disk`        |  —    | `--min-disk` (filter + provision) | warned, dropped (no modal kwarg) |

`local` ignores these — it uses whatever your machine has and auto-detects
NVIDIA runtime via `docker info`.

On brev, the constraints drive `brev search --sort price` and runplz picks
the cheapest match. Override with `BrevConfig(instance_type="...")` when
you need a specific shape.

## Install

```bash
pip install runplz                 # core (local + brev)
pip install 'runplz[modal]'        # add Modal support
```

The core dependency set is empty. Backends shell out to system CLIs:

- `local` → `docker`
- `brev`  → `brev`, `docker` (or skipped in `mode="container"`), `ssh`, `rsync`
- `modal` → `modal>=1.1,<2` Python package

## Outputs

Write to `$RUNPLZ_OUT` inside your function. runplz collects that directory
back to `./out/` on the host (rsync on brev, tar-return on modal, bind-mount
on local). On modal, returns are capped at ~256 MB — if you're writing
more, switch to `modal.Volume` for now (a runplz-native volume abstraction
is TODO).

## Caveats

- `.remote()` args must be JSON-serializable. No closures, no custom
  objects. Deliberate: the remote dispatch is env vars + a path.
- Your job script is imported by path at runtime (not installed as a
  package), so it can live anywhere in the repo.
- One `App` per script. Multiple `App`s in one file is ambiguous for the
  CLI loader and errors.

## Tests

```bash
pytest tests/
```

~120 offline tests — DSL rendering, BrevConfig validation, Modal GPU-label
translation, instance picker with mocked subprocess, CLI guards.

## License

Apache 2.0 — see [`LICENSE`](LICENSE).
