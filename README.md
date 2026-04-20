# runplz

[![PyPI](https://img.shields.io/pypi/v/runplz.svg)](https://pypi.org/project/runplz/)

Tiny Modal-shaped job harness — one Python decoration, multiple backends.

```python
# jobs/train.py
from runplz import App, Image

app = App("my-job")  # default BrevConfig auto-creates the Brev box on first run

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
app.bind("brev", instance="my-gpu-box")   # an existing Brev instance
train.remote()                            #   (same name `brev ls` shows)
```

The Brev `instance=` string is whatever you named the box in Brev — it
must already exist in your Brev account, or `auto_create_instances=True`
must be set on your `BrevConfig`. Brev's managed SSH config adds a
`Host <name>` alias so `ssh <name>` works without further setup.

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

- `--instance <name>` — **required** for `brev`; the Brev box to attach
  to. If it doesn't exist, runplz **fails** by default so a typoed name
  can't silently provision a new billed box. Opt in with
  `BrevConfig(auto_create_instances=True)` to have runplz `brev create`
  missing boxes (cheapest match for your resource constraints, or an
  explicit `BrevConfig(instance_type=...)` if you pinned one).
- `--no-build` — **local only**; reuse the last tagged docker image
  instead of rebuilding.
- `--outputs-dir <path>` — where to collect `/out` back to on the host
  (default `./out/`).

All three have `app.bind(...)` equivalents (`instance=`, `build=False`,
`outputs_dir=`) for the pure-Python invocation path.

## Backend config

`App(..., brev_config=BrevConfig(...), modal_config=ModalConfig(...))`.
Both default to instances of their respective config class, so you only
pass one when you need to override something — the headline example
above omits both and relies on defaults.

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

Invalid combinations (raised eagerly):

- `mode` not in `{"vm", "container"}` — at config construction
- `mode="container"` with `use_docker=False` — at config construction (contradictory; the box *is* the image)
- `instance_type=""` — at config construction
- `on_finish` not in `{"stop", "delete", "leave"}` — at config construction
- `max_runtime_seconds <= 0` — at config construction (use `None` for unlimited)
- `mode="container"` with `Image.from_dockerfile(...)` — at Brev dispatch (container mode has no Dockerfile step)
- `mode="vm", use_docker=False` with `Image.from_dockerfile(...)` — at Brev dispatch (native path ignores the Dockerfile)

### What runplz does NOT ship to the remote

To keep local secrets local, runplz excludes these patterns by default from every host → remote transfer (both Brev's `rsync_up` and Modal's image build context):

`.env`, `.env.local`, `.env.*.local`, `.env.production`, `.env.development`, `*.pem`, `*.key`, `id_rsa`, `id_rsa.*`, `id_ed25519`, `id_ed25519.*`, `credentials.json`, `.aws`, `.ssh`, `.netrc`, `.git-credentials`

If you *need* a secret inside the remote environment, inject it via `@app.function(env={"X": ...})` or Modal Secrets rather than by relaxing this list.

Image/mode checks fire at **Brev dispatch**, not at function decoration, so local/Modal users aren't constrained by the default Brev config on a shared `App`.

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

| constraint        | local | brev                        | modal                         |
| ----------------- | ----- | --------------------------- | ----------------------------- |
| `gpu`             |  —    | `brev search --gpu-name`    | `@app.function(gpu=...)`      |
| `min_cpu`         |  —    | `--min-vcpu`                | `cpu=`                        |
| `min_memory`      |  —    | `--min-ram`                 | `memory=` (converted to MB)   |
| `min_gpu_memory`  |  —    | `--min-vram`                | baked into gpu string: `A100-80GB` |
| `min_disk`        |  —    | `--min-disk` (filter + provision) | warned, dropped (no modal kwarg) |

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

### Multiple functions, multiple shapes?

Resources live on the `@app.function` (Modal-shaped), not on the `App`.
Can different functions land on different hardware within one `App`?
Depends on the backend:

- **Modal**: yes — each `.remote()` schedules independently against Modal's
  pool. A `cpu_prep()` and a `gpu_train()` on the same `App` can land on
  completely different boxes.
- **Brev**: no. One `runplz brev --instance my-box <script>` invocation
  targets a single named Brev box. If you have multiple functions with
  different specs, they all share that box. When `auto_create_instances=True`
  and the box doesn't exist, the **first function that dispatches** determines
  the provisioned shape — subsequent functions reuse it, even if their
  specs would demand something bigger. Workaround: separate invocations
  with different `--instance` names, or pre-create the box yourself.
- **Local**: specs are ignored; your machine is your machine.

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
