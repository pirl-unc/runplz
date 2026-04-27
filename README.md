# runplz

[![PyPI](https://img.shields.io/pypi/v/runplz.svg)](https://pypi.org/project/runplz/)

Tiny Modal-shaped job harness — one Python decoration, multiple backends.

### Smallest working example

A single `@app.function` is enough — no `@app.local_entrypoint` needed.
runplz auto-runs the function as the entrypoint when there's exactly one.

```python
# jobs/train.py
from runplz import App, Image

app = App("my-job")

@app.function(
    image=Image.from_registry("pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime"),
    gpu="T4",
)
def train():
    import torch
    print("cuda available:", torch.cuda.is_available())
```

```bash
runplz brev jobs/train.py     # ephemeral GPU box, runs train(), tears down
runplz local jobs/train.py    # docker on your machine
runplz modal jobs/train.py    # Modal serverless
```

### Adding constraints + outputs

Resource minimums (`min_cpu`, `min_memory`, `min_gpu_memory`, `min_gpus`,
`min_disk`) shape what the brev / modal selector picks. Anything written
under `$RUNPLZ_OUT` rsyncs back to `./out/` on your machine.

```python
# jobs/train.py
from runplz import App, Image

app = App("my-job")

image = (
    Image.from_registry("pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime")
    .pip_install("pandas>=2.0", "scikit-learn")
    .pip_install_local_dir(".", editable=True)
)

# "Any GPU with at least 24 GB VRAM" — selector picks the cheapest match.
# On Brev: searches across all matching models. On Modal: maps to the
# smallest standard model that meets the bar (here: L4).
@app.function(image=image, min_gpu_memory=24, min_cpu=4, min_memory=16)
def train(steps: int = 1000):
    import os
    out = os.environ["RUNPLZ_OUT"]
    os.makedirs(out, exist_ok=True)
    with open(f"{out}/result.txt", "w") as f:
        f.write(f"trained {steps} steps\n")
```

```bash
runplz brev jobs/train.py --steps=5000   # entrypoint args parse from the tail of argv
```

### Custom driver

When you need to do more than fire-and-forget — multiple `.remote()`
calls, post-processing, picking which function to run — declare an
explicit `@app.local_entrypoint`. It runs *in the local CLI process*;
`.remote()` dispatches to the chosen backend.

```python
@app.function(image=image, gpu="A100", num_gpus=4)
def train(fold: int): ...

@app.function(image=image, min_cpu=8)
def aggregate(): ...

@app.local_entrypoint()
def main(folds: int = 4):
    for i in range(folds):
        train.remote(fold=i)
    aggregate.remote()
```

`.remote()` doesn't bring the function's return value back — the remote
body runs in a separate process, possibly on a separate machine.
Communicate via files (see ["Data in and out"](#data-in-and-out) below)
or stdout (captured to the driver log).

Invoke via the CLI:

```bash
runplz local  jobs/train.py
runplz brev   jobs/train.py                       # ephemeral: runplz picks a box, runs, deletes
runplz brev   --instance my-box jobs/train.py     # attach to an existing named Brev box
runplz ssh    --host gpu.example.com jobs/train.py
runplz modal  jobs/train.py
```

Entrypoint params are parsed from the tail of argv, modal-style:

```bash
runplz local jobs/train.py --steps=1000 --dataset=big
# calls main(steps=1000, dataset="big")
```

…or from pure Python (notebook, REPL, `python jobs/train.py`):

```python
# at the bottom of jobs/train.py
if __name__ == "__main__":
    app.bind("brev", instance="my-box")   # or "local" / "modal" / ssh with host=
    train.remote()
```

`app.bind(...)` is the programmatic equivalent of the CLI — it attaches a
backend (plus the same flags: `instance=`, `host=`, `outputs_dir=`, `build=`)
so `.remote()` knows where to dispatch.

## How it's structured

Two entry points, same dispatch underneath. `python script.py` won't
work on its own — the `App` doesn't know which backend to target until
something binds one.

**CLI (preferred for CI / shared scripts).** `runplz <backend> <script>`:

1. Imports your script (finds the `App` instance at module scope).
2. Binds the chosen backend to that `App`.
3. Calls whatever you've decorated with `@app.local_entrypoint()`.

**`App.bind(...)` (for notebooks, one-off scripts, tests).** Bind the
backend yourself, then call `.remote()` directly — no CLI, no
`@local_entrypoint` required:

```python
app.bind("local")                         # or "modal"
app.bind("brev")                          # ephemeral: runplz creates + deletes
app.bind("brev", instance="my-gpu-box")   # an existing Brev instance
app.bind("ssh",  host="gpu.example.com")  # user-owned remote box
train.remote()
```

Brev has three instance paths:

1. **Omit `instance=`** (ephemeral) — runplz auto-names a box sized to
   your function, creates it, runs, and `brev delete`s it on exit.
2. **`instance="my-box"`, box exists** — attach and run. If a previous
   run's `on_finish="stop"` paused it, runplz `brev start`s it first.
3. **`instance="my-box"`, box doesn't exist** — runplz **fails** by
   default (typo guard). Opt in to auto-create with
   `BrevConfig(auto_create_instances=True)`.

Brev's managed SSH config adds a `Host <name>` alias so `ssh <name>`
works without further setup. The SSH backend (`host=`) works with any
ssh endpoint reachable from your shell — an alias, `user@host`, or a
bare hostname.

Either way, `.remote()` serializes a minimal dispatch (env vars + a
path to your script) and runs on the selected backend. Args and
kwargs must be JSON-serializable.

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

- `--instance <name>` — **optional** for `brev`. Omit it for
  **ephemeral mode**: runplz auto-names a box sized to your function
  (cheapest match from the selector, or `BrevConfig(instance_type=...)`
  if you pinned one), creates it, runs, and **deletes** it on exit.
  With a named `--instance`, runplz attaches to an existing box
  (auto-starting it if a previous run's `on_finish="stop"` paused it);
  if the name doesn't exist, runplz **fails** by default so a typo
  can't silently provision a new billed box — opt in to auto-create
  with `BrevConfig(auto_create_instances=True)`.
- `--host <name>` — **required** for `ssh`; any ssh endpoint reachable
  from your shell (bare hostname, `user@host`, or a `~/.ssh/config`
  alias). No provisioning — you own the box.
- `--no-build` — **local only**; reuse the last tagged docker image
  instead of rebuilding.
- `--outputs-dir <path>` — where to collect `/out` back to on the host
  (default `./out/`).

All four have `app.bind(...)` equivalents (`instance=`, `host=`,
`build=False`, `outputs_dir=`) for the pure-Python invocation path.

## Backend config

`App(..., brev_config=BrevConfig(...), modal_config=ModalConfig(...),
ssh_config=SshConfig(...))`. Each defaults to an instance of its
respective config class, so you only pass one when you need to override
something — the headline example above omits all three and relies on
defaults.

### BrevConfig

All fields are validated at construction time — an invalid config raises
`ValueError` immediately, not later during dispatch.

| field                    | default | what it does                                                                                                                                      |
| ------------------------ | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `auto_create_instances`  | `False` | When `--instance` points at a non-existent box, hard-fail rather than silently `brev create` it (typo-safe default). Set `True` to opt into auto-provisioning.                            |
| `instance_type`          | `None`  | Pin a specific Brev instance type string (e.g. `"n1-standard-4:nvidia-tesla-t4:1"`). Skips the constraint-based picker.                           |
| `mode`                   | `"container"` | `"container"` (default) = the Brev box IS the base image; runplz applies Image DSL ops inline over ssh. Lighter, no DinD, sidesteps a known GPU+docker SSH-wedging bug. Requires `Image.from_registry(...)`. `"vm"` = full Brev VM + docker-in-VM; use when you need a user Dockerfile or the legacy native path. |
| `use_docker`             | `True`  | VM-mode only. `False` skips docker and installs a native venv on the box. Legacy escape hatch for providers where container mode isn't available. |
| `on_finish`              | `"stop"` | What runplz does to the Brev box when the App exits (success **or** failure). `"stop"` → `brev stop` (disk cached, small ongoing charge). `"delete"` → `brev delete` (zero ongoing cost, cold rebuild). `"leave"` → never touch the box (opt-in for interactive dev workflows). |
| `max_runtime_seconds`    | `None`  | Wall-clock kill-switch. When set, runplz kills the remote container/process and raises `RuntimeError` after this many seconds so a wedged job can't keep billing forever. `None` = unlimited.                                                                       |
| `ssh_ready_wait_seconds` | `1800` (30 min) | How long to wait for the freshly-provisioned Brev box to become SSH-reachable. Default covers 8×A100/H100 cold boots on Denvr / OCI (15-18 min in practice). Bump for slower provider / shape combos. |

Invalid combinations (raised eagerly):

- `mode` not in `{"vm", "container"}` — at config construction
- `mode="container"` with `use_docker=False` — at config construction (contradictory; the box *is* the image)
- `instance_type=""` — at config construction
- `on_finish` not in `{"stop", "delete", "leave"}` — at config construction
- `max_runtime_seconds <= 0` — at config construction (use `None` for unlimited)
- `mode="container"` with `Image.from_dockerfile(...)` — at Brev dispatch (container mode has no Dockerfile step)
- `mode="vm", use_docker=False` with `Image.from_dockerfile(...)` — at Brev dispatch (native path ignores the Dockerfile)

Image/mode checks fire at **Brev dispatch**, not at function decoration,
so local/Modal users aren't constrained by the default Brev config on a
shared `App`.

### SshConfig

`App(..., ssh_config=SshConfig(...))` plus `--host <target>` (CLI) or
`app.bind("ssh", host=...)` (Python). runplz never provisions or tears
down — you own the box. The backend rsyncs your repo up, optionally
warns about spec mismatches, dispatches the bootstrap (docker or
native), and rsyncs outputs back.

| field                    | default | what it does                                                                                                                     |
| ------------------------ | ------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `user`                   | `None`  | Ssh login user. `None` → whatever's in the host URL or your `~/.ssh/config`.                                                     |
| `port`                   | `None`  | Ssh port. `None` → the default (22 or whatever your `~/.ssh/config` says). When set, threaded into both the `ssh` command (`-p N`) and the `rsync` transport (`-e "ssh -p N ..."`). Also accepted inline on `host="user@example.com:2222"`. |
| `use_docker`             | `True`  | Build + `docker run` the image on the remote. `False` = native venv install (mirrors `BrevConfig(mode="vm", use_docker=False)`). |
| `on_finish`              | `"leave"` | Pinned to `"leave"`; runplz doesn't touch the lifecycle of a user-owned box. Setting `"stop"` / `"delete"` raises at config construction. |
| `max_runtime_seconds`    | `None`  | Wall-clock kill-switch — same semantics as `BrevConfig.max_runtime_seconds`.                                                     |
| `ssh_ready_wait_seconds` | `1800` (30 min) | How long to wait for the SSH box to become reachable before giving up. Mostly useful when the user is booting the box just before the runplz invocation. |

**Spec-mismatch warnings.** Because the SSH box is fixed (no selector
chooses it for you), runplz probes the remote at dispatch and warns when
your function's declared constraints aren't met — e.g. a function with
`min_memory=32` against a 16GB remote, or `gpu="A100"` against a box
where `nvidia-smi` reports a T4 (or no GPUs at all). Warnings only — the
job still runs; the user may know something we don't (MIG slicing,
overcommit, etc.).

### What runplz does NOT ship to the remote

To keep local secrets local, runplz excludes these patterns by default from every host → remote transfer (Brev's and SSH's `rsync_up`, plus Modal's image build context):

`.env`, `.env.local`, `.env.*.local`, `.env.production`, `.env.development`, `*.pem`, `*.key`, `id_rsa`, `id_rsa.*`, `id_ed25519`, `id_ed25519.*`, `credentials.json`, `.aws`, `.ssh`, `.netrc`, `.git-credentials`

If you *need* a secret inside the remote environment, inject it via `@app.function(env={"X": ...})` or Modal Secrets rather than by relaxing this list.

### ModalConfig

`ModalConfig()` is a no-op today. Modal reads auth from `~/.modal.toml`
and schedules resources from `@app.function(gpu=..., cpu=..., memory=...)`;
we don't expose Modal-specific knobs. The class exists as a slot in
`App(modal_config=...)` so the signature doesn't break when fields are added.

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

| constraint        | local | brev                              | modal                                   | ssh                              |
| ----------------- | ----- | --------------------------------- | --------------------------------------- | -------------------------------- |
| `gpu`             |  —    | `brev search --gpu-name`          | `@app.function(gpu=...)`                | spec-probe warn on model/absent  |
| `num_gpus`        |  —    | `--min-gpus` (when N > 1)         | `:N` suffix on gpu string (`A100:4`)    | spec-probe warn on count         |
| `min_cpu`         |  —    | `--min-vcpu`                      | `cpu=`                                  | spec-probe warn on nproc         |
| `min_memory`      |  —    | `--min-ram`                       | `memory=` (converted to MB)             | spec-probe warn on meminfo       |
| `min_gpu_memory`  |  —    | `--min-vram`                      | `-NGB` suffix on gpu string             | spec-probe warn on VRAM          |
| `min_disk`        |  —    | `--min-disk` (filter + provision) | **raises** `ValueError` (use a Volume)  | —                                |

`local` ignores these — it uses whatever your machine has and auto-detects
NVIDIA runtime via `docker info`.

On brev, the constraints drive `brev search --sort price`. runplz picks
the cheapest match, with one refinement: when the top few candidates
are within **5% on price**, preference goes to whichever has the lowest
availability/start-latency hint (when `brev search` exposes one — field
names tried: `estimated_start_seconds`, `eta_seconds`, `eta_s`,
`queue_wait_seconds`, `availability_rank`). A $0.01/hr difference isn't
worth a job sitting 5 minutes in a queue. If no candidate has a hint,
plain cheapest wins. Override the whole picker with
`BrevConfig(instance_type="...")` when you need a specific shape.

### Multi-GPU (`num_gpus=N`)

`@app.function(gpu="A100-80GB", num_gpus=4)` requests 4 GPUs of that
model. Defaults to `1`. Maps to:

- **brev**: `brev search --min-gpus N` filters the instance-type catalog.
- **Modal**: appended as `:N` to the gpu string (Modal's native syntax —
  `A100-80GB:4`).
- **ssh**: the spec-mismatch probe warns if `nvidia-smi` returns fewer
  than `N` GPUs on the remote.
- **local**: ignored, like other specs — your machine is your machine.

Requires `gpu=...` (can't ask for multiple GPUs without a model).

### Multiple functions, multiple shapes?

Resources live on the `@app.function` (Modal-shaped), not on the `App`.
Can different functions land on different hardware within one `App`?
Depends on the backend:

- **Modal**: yes — each `.remote()` schedules independently against Modal's
  pool. A `cpu_prep()` and a `gpu_train()` on the same `App` can land on
  completely different boxes.
- **Brev (ephemeral, `instance=None`)**: yes — each `.remote()` call spins
  up its own auto-named box sized to that function's specs and deletes it
  on exit. The cost of that isolation is per-function provisioning
  overhead (minutes of cold-start each).
- **Brev (named `--instance my-box`)**: no. One named box serves the whole
  invocation, so all functions share its shape. When
  `auto_create_instances=True` and the box doesn't exist, the **first
  function that dispatches** pins the provisioned shape — subsequent
  functions reuse it even if their specs would demand something bigger.
  Workaround: separate invocations with different names, pre-create the
  box, or drop `--instance` to go ephemeral.
- **SSH**: no. The box is fixed at dispatch (you own it). Spec mismatches
  surface as warnings from the probe.
- **Local**: specs are ignored; your machine is your machine.

## Install

```bash
pip install runplz                 # core (local + brev + ssh)
pip install 'runplz[modal]'        # add Modal support
```

The core dependency set is empty. Backends shell out to system CLIs:

- `local` → `docker`
- `brev`  → `brev`, `docker` (or skipped in `mode="container"`), `ssh`, `rsync`
- `ssh`   → `ssh`, `rsync` (docker on the remote if `use_docker=True`)
- `modal` → `modal>=1.1,<2` Python package

## Data in and out

runplz doesn't serialize args/returns — you move data via files. The
remote function sees your repo under `/workspace/` and writes results
to `$RUNPLZ_OUT`, which comes back to `./out/` on your machine.

### Inputs — your repo goes up

The entire repo (minus `.env` / secrets / `.git` / caches — see
["What runplz does NOT ship"](#what-runplz-does-not-ship-to-the-remote)) is rsynced to
the remote before dispatch. Read input files by relative path the same
way you would locally:

```python
@app.function(image=image)
def train():
    import pandas as pd
    df = pd.read_csv("data/train.csv")   # from /workspace/data/train.csv
    ...
```

Large datasets that you don't want to rsync every run: host them on
S3 / GCS / Modal Volume and have the remote function pull them at
start-up. runplz's `.env` exclusion means you can ship `boto3`
credentials via `@app.function(env=...)` without leaking them into the
image layer.

### Outputs — write to `$RUNPLZ_OUT`

`RUNPLZ_OUT` is set to the remote's output directory (`/out` inside
docker, or `$HOME/runplz-out` on ssh/native paths). Anything you drop
there is collected back to `./out/` on the host:

```python
@app.function(image=image, gpu="T4")
def train():
    import os, torch
    model = ...
    torch.save(model.state_dict(), f"{os.environ['RUNPLZ_OUT']}/weights.pt")
```

Transport per backend:

- **local** — bind-mount. No size cap.
- **brev / ssh** — `rsync` from the remote after dispatch. No size cap
  beyond remote disk.
- **modal** — the remote returns `/out` as a tar.gz blob, subject to
  Modal's ~256 MB return-value cap. runplz measures the blob before
  extracting: **warns at 200 MB**, **raises `RuntimeError` at 256 MB**
  (the archive may already be truncated, and silently unpacking a
  truncated tar would lose data).

### Large / persistent outputs on Modal — use a Volume

When your results are bigger than 256 MB — or when you want them to
persist across runs without being re-rsynced — mount a Modal Volume
at `/out`:

```python
import modal

volume = modal.Volume.from_name("training-outputs", create_if_missing=True)

@app.function(image=image, gpu="T4", volumes={"/out": volume})
def train():
    import os, torch
    model = ...
    torch.save(model.state_dict(), "/out/weights.pt")
    # volume.commit() not needed — Modal auto-commits on function exit
```

Then download locally after the run:

```python
# jobs/download.py — a separate script, or a follow-up local_entrypoint
import modal
vol = modal.Volume.from_name("training-outputs")
for entry in vol.iterdir("/"):
    with open(f"./out/{entry.path.lstrip('/')}", "wb") as f:
        for chunk in vol.read_file(entry.path):
            f.write(chunk)
```

Brev / ssh don't have a direct volume equivalent — for durable output
on those backends, write to a mounted network drive the box already
has, or push to S3 at the end of the function.

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

~250 offline tests — DSL rendering, config validation across all four
backends, Modal GPU-label / count-suffix translation, instance picker
with cost-tolerance + availability tiebreaker, CLI guards + entrypoint
arg pass-through, SSH spec-mismatch probe, Brev lifecycle (auto-start,
ephemeral, on_finish). All backends are mocked — no cloud calls.
CI runs on Python 3.10 / 3.11 / 3.12 via GitHub Actions.

## License

Apache 2.0 — see [`LICENSE`](LICENSE).
