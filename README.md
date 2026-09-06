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
`.remote()` dispatches to the chosen backend. Exactly one per `App` — a
second `@app.local_entrypoint` is an error, because last-wins would leave
the first driver unreachable with nothing printed to say so.

```python
@app.function(image=image, gpu="A100", min_gpus=4)
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

#### What a backend actually is

Very little. Everything a run does on a remote box — staging the repo,
probing preconditions, building/running the container, streaming output,
collecting outputs, capturing a failure tail, removing the container — is
one shared path in `ssh_common.dispatch_to_target`. Provisioning backends
wrap it in `run_on_provisioned_vm`, which handles the signal traps and the
always-runs teardown that stop a killed orchestrator from leaking a billed
box.

So a backend is only its own provider vocabulary:

| shared | per-backend |
|---|---|
| `backends.ssh_common` — dispatch, streaming, outputs, failure tails, remote-path validation | how to create a machine |
| `backends.docker` — container labels and `docker ps` parsing | how to reach it over ssh |
| `backends.provisioning` — instance naming, GPU-shape lookup, teardown contract | how to tear it down |
| `backends.registry` — what backends exist and what each accepts | which shapes that provider sells |
| `config` — field validation shared by every remote config | |

These are public modules with explicit `__all__`s, not private helpers: a
backend is *expected* to be written against them. `ssh_common.dispatch_to_target`
is the contract; the staging and streaming helpers it calls internally stay
private, because a backend should never need to reach past it.

That is why `gcp.py` and `aws.py` are a couple of hundred lines each, and
why adding another provider is mostly a table of machine types plus three
functions.

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

### Operations CLI: `ps`, `tail`, `status`, `kill`

Four subcommands let you check on — and stop — jobs without retyping ssh
aliases or remembering remote run IDs:

```bash
runplz ps                          # list runplz jobs across all backends
runplz ps brev                     # one backend
runplz ps --host my.gpu.box        # also probe an SSH host (alongside the rest)
runplz ps ssh --host my.gpu.box    # that host and nothing else

runplz tail                        # tail remote driver log of the most recent run
runplz tail -n 500                 # last N lines (default 120)
runplz tail -f                     # stream new lines as they arrive

runplz status                      # one-screen summary: target, last event, last heartbeat, event count

runplz kill                        # stop the most recent run (SIGTERM, then SIGKILL)
runplz kill --timeout 60           # give it longer to checkpoint before SIGKILL
runplz kill --no-escalate          # SIGTERM only; report whatever survives
runplz cancel                      # alias for kill
```

`runplz tail` / `status` / `kill` reuse whatever ssh port and key the
dispatch recorded in the run manifest, so following a run on a box that
needed one just works. `--ssh-key` / `--ssh-port` override that, which is
what you want when targeting `--host`/`--run-id` with no local manifest.

`tail`, `status` and `kill` default to "most recent run in `./out/`" by
reading the local `out/.runplz/run.json` manifest the dispatch path
writes. Pass `--outputs-dir <path>` to point at a different one, or
`--host <h> --run-id <id>` to target a specific run by ID. SSH has no
host registry, so a bare `runplz ps` skips it and says so; pass
`--host` to include it, alongside the other backends or on its own as
`runplz ps ssh --host <h>`.

If SSH is unavailable, `status` falls back to saved local metadata for the same
host and run after a 10-second probe timeout. It labels this as a **local snapshot**,
not live status—useful after an ephemeral host has been deleted. Output-sync state
is shown separately from the run outcome. Native/container environment setup also
records its start and outcome; a transport failure means completion is unconfirmed.

Downloads have a 60-second idle timeout. Failure salvage additionally has a
60-second transfer budget shared across retries, and individual lifecycle-event
writes time out after 10 seconds. A failed salvage remains an explicit local event;
it does not replace the original run failure. During cleanup, cancellation is
deferred until the remaining cleanup steps have been attempted.

A scope flag that no listed backend can use is an error, not a no-op:
`runplz ps local --region us-east-1` narrows the listing to `local` and then
scopes AWS, which is not in it. That used to run and quietly ignore the
region, leaving you to believe the listing was scoped when it was not.

#### What `kill` actually stops

`kill` reports success only after confirming no survivors. An unavailable Docker
daemon or unreadable stop result is **unconfirmed**, not a stopped container, and
returns a nonzero exit code. Lifecycle events distinguish delivered signals from
failed attempts.

A remote run is a tree, not a process: a bash supervisor, the bootstrap,
the worker(s) it spawns, and any DataLoader children those fork. runplz
launches the bootstrap with its run id in the environment, so every
descendant inherits it, and `kill` stops exactly the processes carrying
that id. That still finds workers orphaned to init after the supervisor
died — the case where `pkill -P` has nothing left to match — without
touching anything else on the box.

A process group would have been the obvious handle, but bash disables job
control in non-interactive shells, so the bootstrap never becomes a group
leader and its pgid is the *launching shell's* — not unique to the run,
and unsafe to signal wholesale. A run id is unique, survives reparenting,
and cannot be recycled by PID wraparound.

In VM+docker mode the container is signalled separately: it is a child of
dockerd, so it carries no marker of ours.

`kill` sends `SIGTERM` first so the job can flush checkpoints and close
writers, waits `--timeout` seconds (default 10), then escalates to
`SIGKILL` unless you passed `--no-escalate`. On exit it prints the
before/after state, any surviving pids, per-GPU memory still in use (when
`nvidia-smi` is present), the last heartbeat and a short log tail, and
appends `event=killed_by_user` to the run's `events.ndjson`.

Exit codes make it safe to chain:

| Code | Meaning |
|---|---|
| `0` | the run is stopped — including "it had already finished", so `kill` is idempotent |
| `2` | couldn't reach the host, or couldn't read a result — assume nothing was stopped |
| `3` | signalled, but something is **still alive** (pids or the container) |

So `runplz kill && runplz brev job.py` will not start a second job on a
GPU the first one still holds.

Runs launched before 3.16 carry no marker; `kill` falls back to the
recorded bootstrap pid alone and says so in its output.

## Public API

Everything below is importable and covered by semantic versioning. Each
module declares an `__all__`; if a name is not in one, treat it as an
internal that can move in a patch release.

| Module | What it is |
|---|---|
| `runplz` | Re-exports everything below that a job script needs: `App`, `Function`, `Image`, `ImageOp`, and the five backend configs. The only import most scripts need. |
| `runplz.app` | `App`, `Function`, `repo_root_for`, `validate_image_vs_brev_mode`, `PRECONDITION_KEYS`. |
| `runplz.config` | `BrevConfig`, `SshConfig`, `ModalConfig`, `GcpConfig`, `AwsConfig`. |
| `runplz.image` | `Image` and the `ImageOp` DSL. |
| `runplz.cli` | The `runplz` console script (`main`). Also reachable as `python -m runplz.cli`. |
| `runplz.runs` | The `tail` / `status` / `kill` verbs, plus the reader for the `run.json` manifest that `rsync_down` leaves in your outputs dir. |
| `runplz.bootstrap` | The in-container loader and its **environment contract** — `RUNPLZ_SCRIPT`, `RUNPLZ_FUNCTION`, `RUNPLZ_OUT`, `RUNPLZ_ARGS`, `RUNPLZ_KWARGS`. |
| `runplz.backends.registry` | What backends exist and what each accepts — the single source of truth behind the CLI's choices. |
| `runplz.backends.listing` | The shape of a listed job and the scope a backend needs to produce one: `JobRecord`, `ScopeField`, `ListingSpec`, `MissingScope`, `InvalidScope`, `ListingUnsupported`. |
| `runplz.backends.ssh_common` | The shared layer every ssh-reachable backend runs on: `dispatch_to_target`, `run_on_provisioned_vm`, and the individual pipeline stages. |
| `runplz.backends.provisioning` | Retry policy, GPU shape tables, instance naming, and teardown shared by the cloud drivers. |
| `runplz.backends.local`, `runplz.backends.ssh`, `runplz.backends.brev`, `runplz.backends.modal`, `runplz.backends.gcp`, `runplz.backends.aws` | The backend drivers. Each exports `run` and — where the backend declares a `ListingSpec` — `list_jobs`, returning `JobRecord`s. The contract `registry.load()` calls. Normally reached through the CLI or `App.bind()`, not imported directly. |
| `runplz.backends.docker` | Container labels and `docker ps` parsing, shared by the local and ssh backends. |
| `runplz.selector` | `pick_machine` / `pick_machines` — cost-tolerance shape selection with an availability tiebreak. |
| `runplz.excludes` | `DEFAULT_TRANSFER_EXCLUDES`, the secret-shaped patterns kept off every host -> remote transfer. |
| `runplz.logcapture` | Tees the driver's stdout/stderr to a log file (what `--log-file` drives). |

### 4.0.0: `list_jobs` takes its scope, it no longer finds it

`aws.list_jobs()` and `gcp.list_jobs()` used to read `AWS_DEFAULT_REGION` /
`GOOGLE_CLOUD_PROJECT` themselves. That fallback now lives once, in the
registry's `ListingSpec`, and both take their scope as a required keyword:

```python
# before (3.x)                      # after (4.0)
aws.list_jobs()                     registry.list_jobs("aws")
gcp.list_jobs()                     registry.list_jobs("gcp")
```

`registry.list_jobs(name, **scope)` is the entry point to prefer: it reads
the same environment variables, validates before spawning a provider CLI,
and works the same way for every backend. Calling a driver directly still
works — `aws.list_jobs(region="us-east-1")` — but you supply the region.

Writing a new backend means answering three questions — how do I create a
box, what ssh target do I hand over, how do I tear it down — and passing
the answers to `run_on_provisioned_vm`. See `runplz/backends/gcp.py`; it
is about 150 lines end to end.

### Two module paths are deliberately underscore-named

`runplz._bootstrap` and `runplz._cli` are legacy entry points kept for
compatibility, not internals.

`runplz._bootstrap` is the load-bearing one. runplz does **not** ship
itself to the remote — only your repo goes over the wire — so the
container's runplz comes from PyPI or your base image and its version is
independent of the orchestrator's. A 3.20 orchestrator routinely talks to
a 3.19 container. That makes the invoked module path part of the wire
format, so backends still emit `python -m runplz._bootstrap`, which older
containers understand. For the same reason the `RUNPLZ_*` variables may
only be added to, never renamed.

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
| `max_runtime_seconds`    | `None`  | Wall-clock kill-switch. When set, runplz stops the remote run and raises `RuntimeError` after this many seconds so a wedged job can't keep billing forever. Still a hard stop, but not an instant one: the run is sent `SIGTERM` first and `SIGKILL` ~5s later if it has not gone, so a job that traps `SIGTERM` gets to flush a partial checkpoint — which, since outputs are collected from a capped run, is often the only evidence of what went wrong. Records a `killed_by_runtime_cap` event. `None` = unlimited.                                                                       |
| `max_inactivity_seconds` | `None`  | Opt-in watchdog on *application silence*, independent of `max_runtime_seconds`. When set, runplz checks how long it has been since the job last produced output — its driver log, its container's log in docker mode, or its outputs dir; past this many seconds it records a `remote_command_stalled` event and captures bounded diagnostics (this run's processes and their states, including zombies, plus `nvidia-smi`). Deliberately not the heartbeat — that ticks on a timer and proves only that the process exists. `None` = no watchdog. |
| `inactivity_action`      | `"diagnose"` | What to do on expiry. `"diagnose"` warns once per stall and keeps monitoring; `"terminate"` stops exactly this run. Outputs are synced back either way. |
| `ssh_ready_wait_seconds` | `1800` (30 min) | How long to wait for the freshly-provisioned Brev box to become SSH-reachable. Default covers 8×A100/H100 cold boots on Denvr / OCI (15-18 min in practice). Bump for slower provider / shape combos. |
| `instance_type_fallback_count` | `3` | When auto-picking, how many ranked candidate types to pass to `brev create` for transparent fallback (Brev's CLI tries them in order if A fails on Nebius, B on OCI, etc.). Set to 1 for single-type behavior. Ignored when `instance_type=` is pinned. |
| `exclude_providers` | `("oci",)` | Provider prefixes to drop from `brev search` results before the auto-pick selector ranks them. Match is case-insensitive and segment-aware (`"oci"` matches `oci.a100x8.sxm.brev-dgxc` but not `ocifoo`). Default blocks OCI because Brev support has confirmed the OCI launchpad path fails server-side on most orgs (issue #62). Set to `()` to disable, or extend (`("oci", "shadeform")`) for orgs where another provider is also broken. **A pinned `instance_type=` bypasses this filter** — your pin, your consequences. |

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
| `max_inactivity_seconds` | `None`  | Opt-in watchdog on *application silence*, independent of `max_runtime_seconds`. When set, runplz checks how long it has been since the job last produced output — its driver log, its container's log in docker mode, or its outputs dir; past this many seconds it records a `remote_command_stalled` event and captures bounded diagnostics (this run's processes and their states, including zombies, plus `nvidia-smi`). Deliberately not the heartbeat — that ticks on a timer and proves only that the process exists. `None` = no watchdog. |
| `inactivity_action`      | `"diagnose"` | What to do on expiry. `"diagnose"` warns once per stall and keeps monitoring; `"terminate"` stops exactly this run. Outputs are synced back either way. |
| `ssh_ready_wait_seconds` | `1800` (30 min) | How long to wait for the SSH box to become reachable before giving up. Mostly useful when the user is booting the box just before the runplz invocation. |

**Spec-mismatch warnings.** Because the SSH box is fixed (no selector
chooses it for you), runplz probes the remote at dispatch and warns when
your function's declared constraints aren't met — e.g. a function with
`min_memory=32` against a 16GB remote, or `gpu="A100"` against a box
where `nvidia-smi` reports a T4 (or no GPUs at all). Warnings only — the
job still runs; the user may know something we don't (MIG slicing,
overcommit, etc.).

`SshConfig(ssh_key_path=...)` pins the private key for a box whose key
ssh would not otherwise offer — the same plumbing the AWS backend uses.

### What runplz does NOT ship to the remote

For Brev and SSH launches from a Git worktree, runplz stages the files Git knows about plus
untracked files that are not ignored. Repository, `.git/info/exclude`, and global Git ignore rules
are honored, so ignored run artifacts and sibling output directories are not copied to the remote
source snapshot. Directories outside a Git worktree retain full-tree staging with the exclusions
below.

To keep local secrets local, runplz also excludes these patterns by default from every host →
remote transfer (Brev's and SSH's `rsync_up`, plus Modal's image build context), even if Git would
otherwise select them:

`.env`, `.env.local`, `.env.*.local`, `.env.production`, `.env.development`, `*.pem`, `*.key`, `id_rsa`, `id_rsa.*`, `id_ed25519`, `id_ed25519.*`, `credentials.json`, `.aws`, `.ssh`, `.netrc`, `.git-credentials`

If you *need* a secret inside the remote environment, inject it via `@app.function(env={"X": ...})` or Modal Secrets rather than by relaxing this list.

`env` keys must be valid shell identifiers (letters, digits and
underscores, not starting with a digit) — they are rendered as
`export KEY=...` on the remote, where a name like `MY-VAR` aborts the run
*after* the box is provisioned. runplz rejects it at decoration time
instead. Values need no such care; they are always quoted.

### GcpConfig

`App(..., gcp_config=GcpConfig(...))`, then `runplz gcp jobs/train.py` or
`app.bind("gcp")`. runplz creates a GCE VM sized to your function, runs the
job on it, and deletes it — there is no instance name to pass, because it
makes one.

```python
from runplz import App, GcpConfig, Image

app = App("vision", gcp_config=GcpConfig(project="my-proj", zone="us-central1-a"))

@app.function(image=Image.from_registry("pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime"),
              gpu="A100-80GB", min_gpus=8, min_disk=500)
def train():
    ...
```

`gpu=` / `min_gpus` / `min_gpu_memory` pick the machine type
(`a2-ultragpu-8g` above); pass `machine_type=` to override. Auth is
whatever `gcloud` already has. SSH rides on `gcloud compute config-ssh`,
which publishes a `NAME.ZONE.PROJECT` alias — so plain `ssh`/`rsync` work
and you can ssh to a `on_finish="leave"` box yourself afterwards.

| field | default | notes |
|---|---|---|
| `project`, `zone` | **required** | no guessing from your gcloud config: a box billed to the wrong project is worse than an error |
| `machine_type`, `accelerator` | derived | escape hatch for shapes runplz has no table entry for |
| `image_family`, `image_project` | Deep Learning VM | ships NVIDIA drivers + docker, so runs don't pay a driver install first |
| `network`, `subnet` | project default | pinned, never created — runplz opens no firewall rules |
| `spot` | `False` | cheaper, reclaimable. No retry-on-reclaim yet |
| `on_finish` | `"delete"` | `"stop"` / `"leave"` also valid |
| `max_runtime_seconds` | `None` | wall-clock kill-switch, same semantics as `BrevConfig.max_runtime_seconds` |
| `max_inactivity_seconds`, `inactivity_action` | `None`, `"diagnose"` | application-silence watchdog, same semantics as `BrevConfig` |
| `dry_run` | `False` | print every gcloud command, execute none |

### AwsConfig

`App(..., aws_config=AwsConfig(...))`, then `runplz aws jobs/train.py` or
`app.bind("aws")`.

```python
from runplz import App, AwsConfig

app = App("vision", aws_config=AwsConfig(region="us-east-1", key_name="my-key"))
```

`key_name` is **required** — EC2 gives no way to reach a box without a key
pair, so runplz refuses to provision one it cannot ssh to.

Point `ssh_key_path` at the private half of that key pair and runplz passes
it through as `-i` — to ssh *and* to rsync's transport — along with
`IdentitiesOnly=yes`, so a loaded agent can't offer its other keys first and
trip the server's `MaxAuthTries`. Leave it `None` if the key is already
agent-loaded or named in your ssh config.

The security group in play must allow inbound TCP 22 from wherever you run
runplz, or the run will sit in the ssh wait until it times out.

| field | default | notes |
|---|---|---|
| `region`, `key_name` | **required** | |
| `instance_type` | derived | from `gpu=` / `min_gpus`; `g5.12xlarge`, `p5.48xlarge`, … |
| `ami` | resolved from SSM | Deep Learning AMI. Ids are region-specific and roll monthly, so runplz looks the current one up rather than pinning a stale id |
| `ssh_user` | `"ubuntu"` | DLAMIs are Ubuntu-based |
| `ssh_key_path` | `None` | private half of `key_name`, e.g. `~/.ssh/my-key.pem` |
| `subnet_id`, `security_group_id` | account default VPC | pinned, never created |
| `volume_gb` | `Function.min_disk` | always sent with `DeleteOnTermination`, so `on_finish="delete"` takes the disk too |
| `spot` | `False` | cheaper, reclaimable. No retry-on-reclaim yet |
| `on_finish` | `"delete"` | `"stop"` / `"leave"` also valid |
| `max_runtime_seconds` | `None` | wall-clock kill-switch, same semantics as `BrevConfig.max_runtime_seconds` |
| `max_inactivity_seconds`, `inactivity_action` | `None`, `"diagnose"` | application-silence watchdog, same semantics as `BrevConfig` |
| `dry_run` | `False` | print every aws command, execute none |

Both drivers shell out to the vendor CLI rather than an SDK: it keeps
runplz's core dependency-free, and the test suite's billed-command guard
already covers `gcloud` and `aws`, so a test that forgets to mock cannot
quietly launch a paid instance.

All four remote backends — `brev`, `ssh`, `gcp`, `aws` — run the same
dispatch core (`ssh_common.dispatch_to_target`), so staging, preconditions,
streaming, output collection, failure tails and container cleanup behave
identically wherever you run. A backend is only its own provisioning and
teardown.

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

All memory/disk fields in **GB**, GPU counts unitless:

```python
@app.function(
    image=image,
    gpu="T4",            # exact model label; "A100", "H100", "L4", ...
    min_cpu=4,
    min_memory=16,       # RAM
    min_gpu_memory=24,   # VRAM (works without gpu= — see below)
    min_gpus=1,          # canonical name; num_gpus accepted as legacy alias
    min_disk=100,
    timeout=60 * 60,
)
```

How they're honored per backend:

| constraint        | local | brev                              | modal                                   | ssh                              |
| ----------------- | ----- | --------------------------------- | --------------------------------------- | -------------------------------- |
| `gpu`             |  —    | `brev search --gpu-name`          | `@app.function(gpu=...)`                | spec-probe warn on model/absent  |
| `min_gpus`        |  —    | `--min-gpus` when > 1             | `:N` suffix on gpu string (`A100:4`)    | spec-probe warn on count         |
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

### `min_gpu_memory` without `gpu=`: pick any model that fits

Setting only `min_gpu_memory=` with no `gpu=` means "any GPU with at
least N GB VRAM, you pick the model" — the cheapest match wins:

```python
@app.function(image=image, min_gpu_memory=24)
def train(): ...
```

- **Brev**: `brev search` runs in GPU mode with `--min-vram N`; the
  selector ranks by price across all matching models.
- **Modal**: maps to a default model from a small VRAM ladder
  (`T4`/`L4`/`A100-40GB`/`A100-80GB`/`H200`) — cheapest model that meets
  the bar.

Pin `gpu="L4"` (or whichever) to opt out of auto-selection.

### Multi-GPU (`min_gpus=N`)

`@app.function(gpu="A100-80GB", min_gpus=4)` requests at least 4 GPUs of
that model. The legacy `num_gpus=N` is accepted as an alias for
backwards compatibility (parallels `min_cpu`/`min_memory`/`min_gpu_memory`
naming). Defaults to `1`. Maps to:

- **brev**: `brev search --min-gpus N` filters the instance-type catalog.
- **Modal**: appended as `:N` to the gpu string (`A100-80GB:4`).
- **ssh**: spec-probe warns if `nvidia-smi` returns fewer than `N`.
- **local**: ignored, like other specs.

Setting `min_gpus > 1` without `gpu=` requires at least `min_gpu_memory=`
so the selector knows what kind of GPU to look for.

### Remote preconditions: fail fast on a misprovisioned box

For long, expensive jobs, declare remote-state minimums runplz will probe
*before* bootstrap so a misconfigured box (small `/dev/shm`, full disk,
missing GPU) fails immediately instead of crashing 10 minutes into a
training run:

```python
@app.function(
    image=image, gpu="A100", min_gpus=8,
    preconditions={
        "shm_gb": 4 * NUM_WORKERS,   # PyTorch DataLoader pain point
        "disk_free_gb": 50,
        "gpu_count": 8,
        "gpu_memory_gb": 80,
    },
)
def train(): ...
```

Below the declared minimum prints a warning; below 50% of it raises
`PreconditionFailed` and bails before the bootstrap dispatches. Probe
runs in a single ssh round-trip on brev/ssh; Modal manages disk/shm
itself so preconditions are silently no-op there.

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
pip install runplz                 # includes the Modal Python SDK
```

The Python package includes the Modal SDK. Backends also shell out to
system-installed CLIs:

- `local` → `docker`
- `brev`  → `brev`, `docker` (or skipped in `mode="container"`), `ssh`, `rsync`
- `ssh`   → `ssh`, `rsync` (docker on the remote if `use_docker=True`)
- `modal` → `modal>=1.1,<2` Python package (included automatically)

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

### Imports — repo root first, then the script's own directory

Your job script is loaded by path, and two directories are importable from it:

1. **the repo root** (what runplz stages), searched first
2. **the script's own directory**, so a job laid out as a package of modules
   can import its siblings

```
myrepo/
  common.py          # import common          -> works
  jobs/
    train.py         # the job script
    data.py          # import data            -> works
```

This is deliberately *not* what plain `python jobs/train.py` does: that puts
the script's directory first and the repo root nowhere at all. Two
consequences worth knowing:

- On a name clash the **repo root wins**, unlike plain Python.
- A sibling module is searched *after* the standard library, so a
  `jobs/types.py` will **not** shadow `types` for your run — where plain
  Python would let it. The trade-off is that a sibling named after a stdlib
  module is not importable.

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
@app.function(image=image, gpu="T4", volumes={"/out": "training-outputs"})
def train():
    import torch
    model = ...
    torch.save(model.state_dict(), "/out/weights.pt")
```

Pass the volume's **name**, not a `modal.Volume` object. runplz generates a
standalone Modal entrypoint and runs it in a separate process, so an object
built in your script cannot reach it — runplz calls
`modal.Volume.from_name(name, create_if_missing=True)` for you inside the
generated file.

With `/out` on a volume, runplz stops tarring the outputs into the function
return — which is the point, since that return is what the ~256 MB cap
applies to. Modal commits the volume when the function exits, and runplz
then downloads it into your local outputs dir, so `runplz modal job.py`
leaves results in `./out/` exactly as it does without a volume. Mount a
volume somewhere other than `/out` (say `/data` for a dataset) and outputs
still come back the normal way.

Backends that cannot mount a volume reject `volumes=` at bind time rather
than ignoring it — a dropped mount would write results to disk that
disappears with the box, and you would find out hours later.

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
- `runplz ps` lists AWS/GCP instances carrying runplz's tag/label. Pass
  `--region` (or set `AWS_DEFAULT_REGION`) and `--project`/`--zone` (or set
  the corresponding gcloud environment variables) when querying those
  providers; each backend declares what it needs, and missing scope is
  reported before the provider CLI is called rather than after. Scope that
  reaches no listed backend is refused rather than dropped.
  `runplz tail` / `status` / `kill` work as usual once a run has written its
  manifest.
- Spot capacity (`spot=True`) is a plain passthrough on both clouds: if the
  provider reclaims the box mid-run, the run fails. No retry loop yet.
- Provisioning calls retry transient control-plane failures (a 503, a
  throttle) and give up immediately on the ones that never clear — an
  exhausted quota, a missing key pair. An error runplz doesn't recognise is
  treated as final rather than retried on a guess.
- SSH transport blips during run *preparation* (staging, image build, rsync)
  are retried with bounded backoff. ssh reserves exit 255 for its own
  failures, which tells you the command did not *complete* — not that it
  never started — so only steps that converge on the same state when
  repeated are retried. Launching is different: a dropped connection there
  is ambiguous, so runplz asks the box whether a bootstrap already exists
  and refuses the retry if one does, or if it cannot get an answer. It will
  never start a second training job on the same GPU.

## Tests

```bash
pytest tests/
```

1,062 tests, all offline and all free. CI runs Python 3.10 / 3.11 / 3.12
via GitHub Actions. Three tiers:

**Unit / mocked.** The bulk of it: DSL rendering, config validation across
every backend, GPU-label translation, the instance picker's
cost-tolerance and availability tiebreak, CLI guards and entrypoint
argument pass-through, Brev lifecycle. `subprocess` is mocked.

**Cloud lifecycle against stub CLIs** (`test_e2e_fake_cloud.py`). Real
`subprocess` calls to stub `gcloud` / `aws` executables the test installs
on `PATH`, so argv, JSON parsing, exit-code handling, retry classification
and teardown are checked against something that can disagree with the
author — the mocked tier can only confirm what the author already believed.

**End-to-end over real ssh** (`test_e2e_localhost.py`). The test starts its
own unprivileged `sshd` on loopback, so no Remote Login, no CI-only
service, no configuration. Real staging, rsync, detached launch, kill and
fetch-back. This is the tier that catches generated shell that parses but
does not work.

Pass `pytest --e2e-remote=docker` (or just have Docker running on a
non-Linux host) and the same tests run against a Debian container instead,
so the remote matches production. `--e2e-remote=local` forces the real
sshd, and `auto` — the default — reaches for Docker only where a local
sshd would be the wrong platform. `pytest --help` lists it.

**Shape catalogue** (`test_cloud_catalogue.py`). `botocore` bundles the EC2
instance-type list as a static enum, so every shape runplz can emit is
checked against the real catalogue with no account and no network. This
repo shipped `p3.xlarge`, which does not exist. Note that an API emulator
would not have caught it — `moto` accepts `--instance-type not-a-real-type`
quite happily. GCE has no offline equivalent (`gcloud emulators` covers
only firestore and spanner), so its shapes get format and self-consistency
checks only.

Nothing in any tier can spend money: `conftest.py` intercepts
`subprocess.run` and refuses `brev` / `gcloud` / `aws` / `ssh` / `rsync`
unless the test carries the matching `live_*` marker, or the binary
resolves inside a stub directory the test created itself.

The test names and markers make the runtime layer explicit. Tests using
`mock.patch` exercise Python-level control flow only; tests using the
`sandbox_bin` fixture exercise the real CLI subprocess boundary against an
input-derived stub; tests marked `live_ssh` exercise a real SSH transport
and are the only tier that can be skipped when the host cannot provide a
local/container SSH service. A skip reports that environmental limitation
and is never counted as a passing integration assertion. Failure scenarios
also assert the command or probe that was observed, so a mock cannot pass
solely because the production call was accidentally bypassed.

## License

Apache 2.0 — see [`LICENSE`](LICENSE).
