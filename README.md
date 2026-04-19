# runplz

[![PyPI](https://img.shields.io/pypi/v/runplz.svg)](https://pypi.org/project/runplz/)

Tiny Modal-shaped job harness — one Python decoration, multiple backends.

```python
# jobs/train.py
from runplz import App, BrevConfig, Image

app = App("my-job", brev=BrevConfig(auto_create=True))

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

```bash
runplz local  jobs/train.py
runplz brev   --instance my-box jobs/train.py
runplz modal  jobs/train.py
```

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
  to. If it doesn't exist and `BrevConfig(auto_create=True)`, runplz
  provisions it (using the cheapest match for your resource constraints,
  or an explicit `BrevConfig(instance_type=...)` if you pinned one).
- `--no-build` — **local only**; reuse the last tagged docker image
  instead of rebuilding.
- `--outputs-dir <path>` — where to collect `/out` back to on the host
  (default `./out/`).

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
