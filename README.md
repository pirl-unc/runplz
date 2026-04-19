# runplz

[![PyPI](https://img.shields.io/pypi/v/runplz.svg)](https://pypi.org/project/runplz/)

Tiny Modal-shaped job harness — one Python decoration, multiple backends.

```python
from runplz import App, BrevConfig, Image

app = App("my-job", brev=BrevConfig(instance_type="g2-standard-4:nvidia-l4:1"))

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

Run on whichever backend you like:

```bash
runplz local  path/to/job.py
runplz brev   --instance my-box path/to/job.py
runplz modal  path/to/job.py
```

## What it does

- **User-facing API** mirrors `modal.Image` + `@app.function` + `.remote()`.
  Learn it once; it works on all backends.
- **Image DSL**: `from_registry(...)` → `apt_install(...)` → `pip_install(...)`
  → `pip_install_local_dir(...)` → `run_commands(...)`. Translates to Modal's
  `Image` chain, a synthesized Dockerfile, or inline install commands over
  ssh depending on the backend.
- **Resource requests** in GB: `gpu="T4"`, `min_cpu=4`, `min_memory=26`,
  `min_gpu_memory=16`, `min_disk=100`. Forwarded directly to Modal; on Brev,
  drives `brev search --gpu-name X --min-vcpu Y ... --sort price` to pick
  the cheapest matching instance.
- **Three backends**:
  - **`local`** — `docker build` + `docker run`. Auto-detects NVIDIA runtime
    via `docker info`.
  - **`brev`** — `brev create` + rsync + ssh'd `docker build`/`docker run`
    OR inline install ("container mode"). Supports both `mode="vm"` and
    `mode="container"` on `BrevConfig`.
  - **`modal`** — generates a module-scope `modal.App` file and shells to
    `modal run`. Skips `serialized=True` so local/remote Python versions
    don't have to match.

## Install

```bash
pip install runplz                 # core only (local + brev backends)
pip install 'runplz[modal]'        # add Modal support
```

The core dependency set is empty. Backends shell out to system CLIs:

- `local`  → `docker`
- `brev`   → `brev`, `docker`, `ssh`, `rsync`
- `modal`  → `modal>=1.1,<2` Python package

## Design notes

- **`.remote()` args must be JSON-serializable.** No closures, no custom
  objects. Deliberate: it keeps the remote dispatch mechanism small and
  portable (env vars + a path to the user's script).
- **Function bodies are imported by path, not installed.** The in-container
  `runplz._bootstrap` reads `RUNPLZ_SCRIPT`/`RUNPLZ_FUNCTION` from env and
  imports the user's file with `importlib.util.spec_from_file_location`. So
  your job files can live anywhere in the repo and don't need to be a
  Python package.
- **Backend-agnostic output convention.** Write to `$RUNPLZ_OUT` inside the
  container; the runner collects that directory back to `./out/` on the
  host.

## Common gotchas

- **Brev GPU + docker `--gpus all` has been flaky** in the past (see
  [`docs/brev-notes.md`](docs/brev-notes.md)). If a run stalls, try
  `BrevConfig(mode="container", ...)` — that path bypasses Brev's VM
  docker runtime entirely.
- **Modal function return values max at ~256 MB.** runplz's modal backend
  tar-returns `/out`; if your run writes more than that you'll hit the cap.
  (TODO: switch to `modal.Volume`.)

## Example

See `examples/simple_job.py` for the smallest working thing.

## Tests

```bash
pytest tests/
```

25 offline tests — DSL rendering, BrevConfig validation, Modal → Brev GPU
label translation, instance picker with mocked subprocess, CLI guards.

## License

Apache 2.0 — see [`LICENSE`](LICENSE).
