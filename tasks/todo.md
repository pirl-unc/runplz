## 2026-09-04 PR Plan — Stop the container gracefully on the cap (#158)

Branch: `fix/graceful-container-stop-on-cap` (off `main` @ 4.3.2)

- [x] `build_kill_command` takes the container name instead of only reading it
- [x] `build_kill_command` takes the event it records, instead of always
      claiming a user did it
- [x] The cap uses one stop path whenever the run context is known
- [x] The watchdog's terminate stops claiming `killed_by_user` too
- [x] Bump `runplz/version.py` to 4.4.0
- [x] `./format.sh`, `./lint.sh`, `./test.sh`
- [ ] Review, merge, deploy

### Also in this PR: every deferral this session left open

Asked to stop deferring, so the three loose ends recorded above are closed
here rather than carried:

1. **`--image-id` / `--instance-ids` value validation** (#154's "Not done").
   `ami-` and `i-` prefixes, not the real id widths -- the stubs mint short
   ids of their own, and the bug worth catching is a value that is not an id
   at all: an ARN, a name, or the empty string from a failed `ssm
   get-parameter`, which would otherwise launch with no image.

2. **`runs._parse_status_sections`** (#155's "Not deduped") was a
   byte-for-byte copy of `ssh_common.parse_probe_sections`. #155 moved the
   kv-block parser to `ssh_common` and left its sibling behind, which was the
   worse of the two states: one parser shared, one duplicated, side by side.

3. **A `runplz ps` scope flag that reaches no listed backend** was silently
   ignored -- deferred back in the listing-registry PR as "a new failure mode
   the issue does not ask for". It is the same failure class as #153, #143 and
   #20, and this PR is already about not letting a stop path lie, so leaving
   it would have been inconsistent.

   Two things that check had to get right, both pinned by tests: it reads only
   what the user *typed*, because `resolve_all` falls back to the environment
   and an exported `AWS_DEFAULT_REGION` would otherwise break `runplz ps
   local` for most AWS users; and `--host ,` still counts as unsupplied, since
   "scope that resolves to nothing is scope the user did not supply" holds
   everywhere else.

### The bug

`raise_for_runtime_cap` branches on shape, and the docker branch comes first:

    if container_name is not None:   cleanup = f"sudo docker kill {container_name}"
    elif remote_run is not None:     cleanup = build_kill_command(...)
    else:                            cleanup = "pkill -f 'runplz._bootstrap'"

`docker kill` defaults to **SIGKILL**. The job gets no chance to flush a
partial checkpoint or close a writer. The other branch sends TERM, waits 5s,
and only then escalates. So the mode that is the default for ssh/gcp/aws is
the one that stops least gracefully -- on the exact path `max_runtime_seconds`
exists for, which is a wedged job whose partial output is the only evidence of
what went wrong. #150 made those outputs survive collection; this makes there
be more of them to collect.

The special case existed because `stream_and_wait` had no run context to scope
a kill to. #153 gave it one.

### The second bug, which blocks the first

`build_kill_command` unconditionally appends `killed_by_user` when it signals.
It is not only used by `runplz kill`: `raise_for_runtime_cap`'s native branch
and the watchdog's `_terminate_stalled_run` both use it, so a capped run and a
stalled run each already record that a user killed them. Nobody did.

Unifying the cap onto `build_kill_command` without fixing that would spread
the false attribution to docker mode as well -- and #155 has just added a
truthful `killed_by_runtime_cap` beside it, so the stream would carry both a
correct event and a contradictory one.

So the recorded event becomes a parameter. `runplz kill` keeps
`killed_by_user`; the cap and the watchdog pass `event=None` and record their
own, which they already do, with fields the shell does not have
(`threshold_seconds`, `idle_seconds`).

### Why the container name is passed, not just read

In docker mode the run's processes live in the container's PID namespace, so
the host `/proc` marker scan finds nothing, and `run_container_detached`
writes no `bootstrap.pid`. The script's only handle on the run is
`{meta}/container`.

Reading that file is fine today -- the same shell that starts the container
writes it, under `set -euo pipefail`. But the orchestrator already holds the
name in memory, and `docker kill <name>` could not miss. Preserving that
exactly means passing the name rather than trading a guarantee for elegance:
an explicit container wins, and the file remains the fallback for `runplz
kill`, which learns the name no other way.

### What `max_runtime_seconds` now promises

Still a hard stop, no longer an instant one: TERM, then KILL after 5s. Worth
saying in the README, since "kills the remote container/process" now takes up
to the escalation budget.


## 2026-09-04 PR Plan — Fake cloud CLIs validate option values (#154)

Branch: `test/fake-cloud-value-validation` (off `main` @ 4.3.1)

- [x] Declare per-option value specs beside the existing per-route vocabularies
- [x] Machine/instance types and accelerators from the live catalogues
- [x] Numeric and enumerated options
- [x] Reject before any state mutation, like the missing-option check
- [x] Update hand-written argv that used a shape runplz never generates
- [x] Bump `runplz/version.py` to 4.3.2
- [x] `./format.sh`, `./lint.sh`, `./test.sh`
- [ ] Review, merge, deploy

### The gap

`tests/fake_cloud.py` validated option *names* and never option *values*, so
both of these exited 0:

    gcloud compute instances create x ... --machine-type=totally-invented-9000
    aws ec2 run-instances ... --count abc

The module's own docstring names the bug class it exists to catch -- "invented
machine types, retry strings written from memory" -- so the harness still
could not catch the first of the two failures it was built for.

### Why a syntactic check would not have worked

The obvious cheap fix is a shape check: machine types look like
`family-class-number` on GCP and `family.size` on AWS. But #154's own example,
`totally-invented-9000`, is *exactly* that shape. A grammar check passes it.

Only membership in runplz's own catalogue rejects it -- which is also the
check that catches the thing actually worth catching, a catalogue entry that
stops existing upstream.

### The asymmetry, stated rather than hidden

That makes the stubs deliberately stricter than the real CLIs: `aws` sells
`t3.micro` and this one refuses it, because the vocabulary is "what runplz
generates", not "what the provider offers". Recorded in the module docstring,
because it is a trap otherwise -- a future test exercising a user's
`instance_type=` override has to extend `_option_values`, and that should be a
deliberate act.

Five tests hand-wrote `--instance-type t3.micro` in argv that was otherwise
pretending to be runplz's. They now use `m6i.large`, which runplz does
generate. The sixth use stays: it is a cross-CLI *name* rejection test whose
value is never reached.

### Not done *(closed later in #160)*

`--image-id` / `--instance-ids` have obvious vocabularies (`ami-`, `i-`)
but are not in #154's three categories and no bug motivates them, so they are
left alone rather than folded in.


## 2026-09-04 PR Plan — A lifecycle event for the runtime cap (#155)

Branch: `feat/runtime-cap-lifecycle-event` (off `main` @ 4.3.0)

- [x] Record `killed_by_runtime_cap` before raising, from what cleanup measured
- [x] Leave the raised error unchanged in type and message
- [x] Assert the event by observation, not by mocking the recorder
- [x] Bump `runplz/version.py` to 4.3.1
- [x] `./format.sh`, `./lint.sh`, `./test.sh`
- [ ] Review, merge, deploy

### The gap

`raise_for_runtime_cap` killed the run and raised, but recorded nothing. Every
neighbouring path already leaves a terminal event -- `bootstrap_launch_failed`,
`killed_by_user`, `remote_command_stalled` -- so a capped run was the one stop
reason `events.ndjson` could not explain. Its last entry was whatever the job
wrote before it was killed, which reads like an abrupt crash.

That matters more since #150: a capped run's outputs now survive, so the
artefacts a user gets include the partial results with no record of why they
are partial, and telling "hit the cap" from "crashed" needed the driver log
rather than the event stream `runplz status` and `runplz tail` actually read.

### What the event carries

The cleanup `raise_for_runtime_cap` already runs was throwing away its own
answer. `build_kill_command` emits a SUMMARY block -- final process state,
survivors, container state -- and the function captured it only to discard it.
So `process_state` is measured, not guessed.

`docker kill` and the `pkill` fallback report nothing, so on those paths the
field is *absent* rather than filled with an assumption about what the signal
probably did; the docker path names the container it stopped instead.

`_record_remote_event` drops None fields and warns rather than raising, so it
cannot mask the cap error on its way out -- pinned by a test that breaks the
event write and asserts the RuntimeError is unchanged.

### Also in this PR

`_parse_kv_block` moved from `runs.py` to `ssh_common.parse_kv_block`. Both
the `kill` CLI and the cap path read the same SUMMARY block, and the command
that emits it is built in `ssh_common`, so that is where its parser belongs --
`runs.py` already imports from there. Not deduped in that PR:
`runs._parse_status_sections` was still a copy of
`ssh_common.parse_probe_sections`. Folded in later, in the #158 batch --
and the note above was wrong to say it had been filed, because it had not.

### Deliberately not in this PR

In docker mode the cap stops the container with a bare `docker kill`, which is
an immediate SIGKILL. Now that `stream_and_wait` carries a `remote_run` (#153),
`build_kill_command` could stop it instead -- TERM first, then escalate, and it
signals the container anyway via the container file. That would give the job a
chance to flush partial output, which is the thing #150 was about, and would
make the event uniform across modes. It changes what the cap does, not what it
records, so it is its own issue.


## 2026-09-04 PR Plan — Watchdog in docker mode (#153)

Branch: `fix/watchdog-in-docker-mode` (off `main` @ 4.2.2)

- [x] Give the probe a docker-log activity leg, so an actively-printing
      container is never mistaken for a stalled one
- [x] Thread `remote_run` + the watchdog params into `stream_and_wait`
- [x] Keep the watchdog's wake-ups off the reconnect budget
- [x] Terminate stops the run and returns, so outputs are still collected
- [x] Declare the fields on `GcpConfig` / `AwsConfig` and thread them through
- [x] Add `stream_and_wait` to the signature test that should have caught this
- [x] Bump `runplz/version.py` to 4.3.0
- [x] `./format.sh`, `./lint.sh`, `./test.sh`
- [ ] Review, merge, deploy

### The bug

`use_docker` defaults to `True` on `SshConfig`, `GcpConfig` and `AwsConfig`,
so all three dispatch with `mode="docker"`. That branch calls
`stream_and_wait`, which never took the watchdog parameters -- only
`run_container_mode` and `run_native` did. So `max_inactivity_seconds`
validates, is documented, is threaded the whole way through dispatch, and
then does nothing on the default configuration. No warning, no error.

A watchdog that is believed but inert is worse than no watchdog: the operator
who sets it stops watching manually.

### Why "support" rather than "refuse"

Both were open in #153. Refusing would leave a README-advertised option
working only on brev container mode and the native path -- never on the
default -- and this repo has spent 4.1.0 and #142/#143/#20 removing exactly
that shape of limitation. The mechanics port cleanly: same reconnect shape,
and `build_kill_command` already signals the container (it reads the
container file `run_container_detached` writes), so terminate needs nothing
new.

### The part the issue missed

`build_inactivity_probe` watches two things that move only when the
application moves: `run_driver.log` and the outputs directory. **Docker mode
writes no `run_driver.log`** -- the container's output goes to `docker logs`,
and the detached driver log belongs to the native/container launcher.

Wiring the watchdog in as-is would therefore leave only the outputs-dir leg,
and a job that prints steadily but writes no files would be declared stalled
and, under `terminate`, killed. That is a worse failure than the silent no-op
being fixed, and it is the "fake faithfulness" FIDELITY.md just named: a
probe that looks like it measures activity while measuring almost none of it.

So the probe gains a third leg for docker mode -- the mtime of the container's
own log file, via `docker inspect --format {{.LogPath}}`. An empty LogPath (a
non-`json-file` log driver) stats to 0, which reads as "never written" and is
dropped by the `max` over the legs, so an unusual log driver degrades to the
outputs-dir signal rather than inventing a stall.

### Two details in `stream_and_wait`

- The final `docker wait` must be bounded by the *runtime* budget only. Reusing
  the loop's bound -- which is the min of the cap and the poll interval -- would
  turn a 60s watchdog tick into a spurious runtime-cap raise on a container the
  loop is merely waiting on.
- A watchdog wake-up reattaches with `--tail 0`, like a reconnect, but must not
  spend a reconnect. Reattaching with `--tail all` would reprint the whole log
  every poll; counting it as a reconnect would burn the budget and silently
  drop the live stream on any legitimately quiet job.

After the loop gives up the live stream, only the runtime cap applies -- the
detached path already behaves that way, and the watchdog needs a stream to
wake up from.

### Why it was missed

`test_the_config_reaches_the_monitor` walks a list of functions and asserts
each takes the watchdog parameters. `stream_and_wait` was not in the list.
It is now.


## 2026-09-04 PR Plan — Preserve outputs when a run is cut short (#150)

Branch: `fix/preserve-outputs-on-runtime-cap` (off `main` @ 4.2.0)

- [x] Salvage outputs on the failure path without masking the original error
- [x] Fetch and surface the failure tail when the run raised
- [x] Keep the success path strict about a failed sync
- [x] Bump `runplz/version.py` to 4.2.1
- [x] `./format.sh`, `./lint.sh`, `./test.sh`
- [ ] Review, merge, deploy

### The bug

`rsync_down` sat inside the `try` and after the runner, so any raising path
skipped it. `max_runtime_seconds` -- documented as a kill-switch for a wedged
job, which is exactly the case where partial output is the only evidence --
therefore discarded everything the run had written. `exit_code` was also still
None on that path, so the `finally` did not even fetch the failure tail, and
on a provisioning backend `teardown()` then deleted the box.

Found while implementing the #122 watchdog, whose terminate mode had to avoid
copying this shape in order to meet its own "preserve outputs" requirement.

### Design

Collection moves into the `finally`, guarded by a `synced` flag:

- success path keeps calling `rsync_down` inside the `try`, so a genuine sync
  failure on a healthy run is still a hard error -- swallowing it would lose
  outputs silently, which is the failure mode this issue is about;
- the `finally` retries only when the success path did not run, and is
  best-effort: an exception is already unwinding and must not be replaced by
  an rsync error naming the wrong problem.

Safe because everything inside that `try` happens after `rsync_up` proved the
box reachable -- `prepare_remote_run`, `ensure_remote_rsync`, `rsync_up` and
`check_preconditions` are all outside it.

The failure-tail gate widens from `exit_code is not None and exit_code != 0`
to include the raising path, and the tail is printed: the raised error is
unchanged in type and message, so computing a tail and dropping it would be
worse than not fetching one.

### Also in this PR

Two lessons from my own mistakes while building #122, recorded in
`tasks/lessons.md` rather than left in a chat log: a scripted edit anchored on
text that was not unique broke `ssh_common.py` mid-function, and a test fake
raised `TimeoutExpired` for a call that had been handed `timeout=None`, which
the real `subprocess.run` cannot do.

Audited the suite's other `TimeoutExpired` fakes afterwards. Each raises on a
call that genuinely carries a timeout, so they are faithful and nothing else
needed changing -- reported rather than turned into busywork.


## 2026-09-03 PR Plan — Container fault injection (#141)

Branch: `test/container-fault-injection` (off `main` @ 4.1.1)

- [x] `DockerSshd.refuse_connections` / `drop_connection`
- [x] Keep PID 1 off sshd so killing the daemon does not kill the endpoint
- [x] Make `DockerSshd.start` idempotent, like `LocalSshd.start`
- [x] Run `test_ssh_faults.py` in the container CI job
- [x] Contract test that both backends expose the same fault surface
- [ ] Verify in CI (no Docker daemon on this machine)
- [ ] Review, merge, deploy

### Why the tier could not run on containers

`test_ssh_faults.py` is the repo's real-ssh-against-real-sshd tier. It only
ever worked against `LocalSshd`: `DockerSshd` implemented neither fault
method, so the tier raised AttributeError against the container backend --
which is what a macOS developer gets by default and what `e2e-container`
runs in CI.

It did not surface because the file had no `live_ssh` marker until #140. The
billing guard intercepted every ssh call before the harness methods were
reached, so the tests passed on the guard's own exception and never got far
enough to notice the missing methods.

### The PID 1 problem

The container ran `sshd -D` as PID 1, so killing the daemon would kill the
container and take the published port with it. A "refused connection" would
then be indistinguishable from an endpoint that vanished. PID 1 is now
`sleep infinity` and sshd runs beside it, so the port survives the fault and
`start()` can restore the daemon in place -- the same endpoint, the way
`LocalSshd.start()` already behaves.

`drop_connection` mirrors the LocalSshd fix from #140: kill the session
children before the listener, because an in-flight session is a forked child
that outlives its parent.

### Verification limit, stated plainly

There is no Docker daemon on this machine, so the container path is verified
by CI, not locally. What is verified here: the LocalSshd path still passes,
and a contract test asserts both backends expose the same four primitives --
which is the specific gap that let this go unnoticed.


## 2026-09-03 PR Plan — Pin the cloud selector's minimum contract (#95)

Branch: `test/pin-selector-minimums` (off `main` @ 4.1.0)

- [x] Reproduce #95's four examples against main before writing anything
- [x] Property test: every selection satisfies every declared minimum, or raises
- [x] Pin that the raise happens before any billed CLI call
- [x] Bump `runplz/version.py` to 4.1.1
- [x] `./format.sh`, `./lint.sh`, `./test.sh`
- [ ] Review, merge, deploy

### The behaviour is already correct

#95 says AWS/GCP selection clamps past the largest known CPU shape instead of
failing, and that GPU selection ignores `min_cpu` / `min_memory`. Checked all
four of its literal examples on main first, the same way #96 turned out to be
mostly stale:

| example | on main |
|---|---|
| `min_cpu=200` (aws, gcp) | raises `CloudCliError` |
| `min_memory=2048` (aws, gcp) | raises `CloudCliError` |
| `gpu=T4, min_cpu=100` | raises `CloudCliError` |

`select_machine` carries the contract, and its docstring names it: "count
validation, CPU/RAM validation and the fail-instead-of-clamp contract live
here once". That landed in `e861671` -- the same commit that closed most of
#96, both filed while reviewing PR #94.

### What is actually missing

Nothing pins it. **No test in the suite calls `select_machine` directly.** It
reaches 98% line coverage entirely through `gcp.resolve_shape` and
`aws.resolve_instance_type`, in example-based tests.

Coverage is not the same as the property. Every one of those examples would
still pass if the selector went back to clamping, because a clamp returns a
*finite smaller shape* rather than an error -- and no example probes past the
largest offering. That is exactly the bug #95 describes, and it is currently
invisible to the suite.

So the work is the property, not a fix: for every offering catalogue, GPU
label and GPU count, a selection either satisfies every declared minimum or
raises. Swept ~5000 combinations by hand first -- 2974 satisfied, 1987 raised,
zero under-provisioned -- which is what the test now asserts permanently.

### The other half of the requirement

"...or raise **before a billed CLI runs**". Verified separately by driving
`aws.run` / `gcp.run` with an impossible minimum and counting subprocess
calls: both raise with zero. Pinned too, since the cost of regressing that is
a provisioned box rather than a failed test.

### Review

The mutation check is the whole argument. Reintroducing the clamp -- one
`return largest` where the raise used to be -- fails all nine new tests and
leaves **all 87 pre-existing cloud-backend tests passing**. The suite could
not see this bug before; it can now.

That asymmetry is the reason to write the property rather than more examples.
A clamp returns a plausible smaller machine, not an error, so it is invisible
to any test that does not deliberately ask for more than the catalogue holds.

One guard on the guard: `test_the_sweep_actually_exercises_the_refusal_path`
asserts the sweep really does reach the raising branch. Without it, a
catalogue change that made everything satisfiable would leave the main
property holding vacuously.

1182 passing, coverage 95.36%.


## 2026-09-03 PR Plan — Implement the documented Modal Volume contract (#143)

Branch: `feat/modal-volumes` (off `main` @ 4.0.6)

- [x] `@app.function(volumes={"/out": "name"})` on `Function` / `App.function`
- [x] Backends that cannot mount a volume raise instead of ignoring it
- [x] Render the mount on the generated Modal function
- [x] Skip the tar when the outputs dir is volume-backed
- [x] Download the volume into the local outputs dir after the run
- [x] README: replace the example that never worked
- [x] Bump `runplz/version.py` to 4.1.0
- [x] `./format.sh`, `./lint.sh`, `./test.sh`
- [ ] Review, merge, deploy

### The gap

The README has documented a Volume mount as *the* answer to Modal's ~256 MB
return-value cap since 3.24.31:

    @app.function(image=image, gpu="T4", volumes={"/out": volume})

`App.function()` has never accepted `volumes`, so that example raises
TypeError. The backend has no volume field to render and still tars `/out`
into a single function return. `modal.py`'s own module TODO says as much.

So the only documented escape from the cap did not exist, and #19 -- which
added the size detection that tells users to go use a Volume -- was closed
before the Volume half was built.

### API decision

Per-function `volumes={mount_path: volume_name}` on `@app.function`, taking a
volume **name**, not a live `modal.Volume`.

The object cannot work: the Modal backend generates a Python file at module
scope and shells to `modal run`, so nothing constructed in the user's process
crosses into the generated entrypoint. Only a name can, and the generated file
calls `modal.Volume.from_name(name, create_if_missing=True)` itself. The
README example therefore changes regardless of where the kwarg lives.

`ModalConfig` was the other candidate -- its docstring says it exists as the
slot for Modal fields -- but it is per-App, so two functions in one App could
not mount different volumes. Volumes are a per-function property, so they go
on the per-function decorator.

### Silence is the failure mode

A backend that ignored `volumes=` would drop a durability request without
saying so, which is the same class of bug as #142 (Modal listing) and the
`min_disk` handling this repo already fixed loudly in #20. Backends declare
whether they can mount, and the ones that cannot raise at bind time -- before
anything is provisioned or billed.

### Verification honesty

There is no Modal account in the test environment, so the download and mount
are asserted at the argv/rendered-source level, the same fidelity the fake
cloud tier gives aws/gcloud. What *is* executable: the README example is
accepted by the real decorator, and a volume-backed run does not put the
outputs directory through the function return.

### Review

The README example is now extracted from README.md and `exec`'d, rather than
paraphrased into a test. #143 existed because a documented example had never
been run against the code; a test that re-types the example could drift the
same way, and this one cannot.

Mutation-checked: dropping the `volumes=` passthrough fails three tests,
including the README one.

Only a mount at the outputs directory diverts outputs. A volume at `/data`
is durable scratch and leaves the return path alone, which keeps the existing
small-output behaviour exactly as it was.

If the download fails the error says the outputs are still in the volume and
gives the command to fetch them. The run succeeded and the data is durable --
the one conclusion a user must not reach there is that their results are gone.

Minor bump, not patch: `@app.function(volumes=...)` is new public surface, and
`min_disk`'s error message on Modal now points at a feature that exists.

1173 passing, coverage 95.36%.


## 2026-09-03 PR Plan — Close the executable-test fidelity gaps (#96)

Branch: `test/close-fidelity-gaps` (off `main` @ `c47b504`)

- [x] Audit all six claims against HEAD before writing anything
- [x] Per-route option vocabulary, replacing the flat set shared by both stubs
- [x] Validate argv before mutating stub state
- [x] `live_ssh` marker + guard-proof assertions in `test_ssh_faults.py`
- [x] Make `drop_connection` actually drop an established session
- [x] Restore the CI fail-on-skip guard
- [x] Bump `runplz/version.py` to 4.0.4
- [x] `./format.sh`, `./lint.sh`, `./test.sh`
- [ ] Review, merge, deploy

### Audit first

#96 was filed against PR #94 and lists six claims. Most were fixed since by
`e861671`, so the first job was finding out which were still true rather than
"fixing" what was already fixed:

| # | Claim | Verdict |
|---|---|---|
| 1 | fake CLIs accept missing/unknown args | partly live |
| 2 | lifecycles always install unreachable SSH | fixed |
| 3 | precondition test passes an empty requirement | fixed |
| 4 | an AWS describe assertion passes on zero calls | fixed |
| 5 | SSH harness does not isolate known_hosts | fixed |
| 6 | container CI must fail on setup/skip | partly live -- regressed |

### What was actually live

**Per-route vocabulary.** `_KNOWN_OPTIONS` was one flat set handed to both
stubs, so gcloud accepted `--instance-type` and aws accepted `--zone`, and
`instances list --zone=z` passed -- the exact confusion the file's own comment
warned about while accepting it. Options are now per route, with required
folded into allowed at install time so a required option can never also be an
unknown one.

**Validate before mutating.** Required-option checking ran *after* the stateful
block, so `aws ec2 run-instances --region us-east-1` allocated a client token
and an instance entry and *then* exited 2. A test asserting that rejection was
seeding the next call's state.

**The CI skip guard.** `e861671` deleted the "fail if the tier silently
skipped" step, so a skip in the container tier reported success again. Restored
for `e2e-container`, and the main job now pins `RUNPLZ_E2E_REMOTE=local` so a
runner without sshd fails instead of silently dropping 14 live-ssh tests.
Checked against a real CI run first: the current job reports `1151 passed, 1
skipped` and that one skip is a parametrized "this count is sold", not an
environment skip -- so a blanket skip-fail belongs only on the container job.

### Not in the issue, found while auditing

`test_ssh_faults.py` had no `live_ssh` marker. The marker is a permission, not
a label: without it the billing guard raises
`RuntimeError("tried to run ssh for real")`, which satisfied every
`pytest.raises(Exception)` in the file. Four tests, none of which had ever
reached the daemon their own docstring named.

Fixing the marker exposed two more:

- `test_mid_command_transport_drop_is_reported` would have *failed* once real,
  because `drop_connection` stopped the listener and an established session is
  a forked child that survives it. `sleep 5` ran to completion. The harness now
  kills the session children, and the drop produces a 255 in ~0.6s.
- `test_readiness_timeout_is_reported` passed `max_wait_s=0`, so the deadline
  had already passed on entry and the loop never probed -- it reported
  `last error: ''` whether or not ssh worked. It now uses a real budget and
  asserts the recorded last error is non-empty.

Every assertion in that file is now specific enough that the guard cannot stand
in for it: verified by removing the marker, which fails all four.


## 2026-09-03 PR Plan — Detached launch on a macOS remote (#92)

Branch: `fix/nohup-optional-detach` (off `main` @ `ed0ce7a`)

- [x] Probe `nohup` in `build_detached_launcher` and drop it when it refuses
- [x] Executable regression test with a failing `nohup` stub on PATH
- [x] Re-point the five tests that identify the spawn line by the word "nohup"
- [x] Drop the `remote_is_linux` skips once the detached tests pass on Darwin
- [x] Bump `runplz/version.py` to 4.0.3
- [x] `./format.sh`, `./lint.sh`, `./test.sh`
- [ ] Review, merge, deploy

### The failure

Against a macOS remote every detached run dies immediately. `run_driver.log`
holds only `nohup: can't detach from console: Inappropriate ioctl for device`,
and runplz reports `detached bootstrap failed to start (state=dead)`.

Specific to the non-interactive ssh session: `ssh_exec` sends `bash -lc <cmd>`
with no pty, and macOS `nohup` refuses to detach there. Bare `nohup` in a local
shell on the same machine is fine. `setsid` is not an alternative -- macOS does
not have it.

### Why nohup turns out to be droppable

The question was whether `nohup` carries the SIGHUP guarantee #74 established.
It does not. Measured on this machine (Darwin) before touching the code:

| probe | result |
|---|---|
| `trap '' HUP` + background + `</dev/null`, then a real SIGHUP | survived |
| identical, trap removed (control) | died -- `Hangup: 1` |

The control matters: without it "survived" could just mean no signal was
delivered. #74's diff (`102cc11`) confirms the shape -- its entire production
change was two `trap '' HUP` lines, one in the launching parent before the fork
and one as the first line of the heredoc'd `run.sh`. `nohup` was untouched.

So `nohup`'s only residual role is the one its docstring claims: it `exec`s in
place, so `$!` stays the bootstrap pid, which is why it was chosen over
`setsid` (which may fork for a process-group leader). A plain backgrounded
`bash "$run_script" &` is equally PID-stable.

### Design

Probe once, drop it if it refuses:

    runplz_nohup=nohup
    nohup true >/dev/null 2>&1 || runplz_nohup=
    RUNPLZ_RUN_ID=<id> ${runplz_nohup} bash "<run.sh>" </dev/null >> "<log>" 2>&1 &

A runtime probe rather than a `uname` test: production has no remote-OS
detection anywhere, `build_detached_launcher` receives only a
`RemoteRunContext` and a command string so it could not learn the OS without
new plumbing, and a probe is right for any platform whose `nohup` refuses
rather than for Darwin specifically.

One spawn line rather than an `if/else` around two, so the env assignment and
the redirections cannot drift. The env assignment has to stay on the exec line:
`/proc/<pid>/environ` is fixed at exec time.

### Why the bug is invisible

bash forks the job *before* `nohup` execs and fails, so `echo $!` still writes a
pid -- of a process that is already dead. Indistinguishable from "the payload
crashed instantly". Only `run_driver.log` carries the real message, and nothing
parses it.

Docker mode is unaffected: `run_container_detached` backgrounds with
`sudo docker run -d` and never calls `build_detached_launcher`.

### Testing a platform CI does not have

All three CI jobs are `ubuntu-latest`, so a test needing a Mac would never run.
Instead the *failure* is made reproducible anywhere: the launcher is plain bash
and the existing #74 test already executes it with `bash -c`, so running it with
a stub `nohup` on PATH that exits non-zero reproduces the production symptom
exactly -- pid file written, payload never runs.

### Review

Verified on the platform, not only on the simulation. This machine is a Mac,
and `RUNPLZ_E2E_REMOTE=local` forces the harness to a local sshd instead of the
Docker backend it picks by default on non-Linux -- which makes the "remote" an
actual macOS box under an actual non-interactive ssh session.

| launcher | `tests/test_e2e_localhost.py` on a macOS remote |
|---|---|
| pre-fix | **2 failed**, 6 passed -- the two detached tests |
| fixed | **8 passed** |

So the two tests that had been skipping since #92 was filed were skipping a
real, reproducible bug, and they now hold it down. The stubbed-`nohup` unit
test covers the same thing on Linux CI: it fails on the pre-fix launcher for
the production reason (no marker -- the payload never ran) and passes after.

Five tests identified "the spawn line" as the first line containing `nohup`,
which after this change matches the probe rather than the spawn. All five now
find it by the run script it launches, which is what actually identifies it and
does not care which tool prefixes the command.

1149 passing, coverage 95.46%.


## 2026-09-03 PR Plan — Typed JobRecord + backend listing capabilities (#133)

Branch: `feat/typed-job-records` (off `main` @ `917d1e6`)

- [x] New public `runplz/backends/listing.py`: `JobRecord`, `ScopeField`,
      `ListingSpec`, `MissingScope`, `ListingUnsupported`
- [x] `registry.BackendSpec.lists_jobs` -> `listing: ListingSpec | None`;
      add `listable_names()`, `scope_fields()`, `list_jobs()` dispatch
- [x] Every driver returns `JobRecord`; aws/gcp stop resolving their own scope
- [x] `cli._ps_main` generates flags + selection from the registry
- [x] Guard `runplz.backends.aws` / `.gcp` in `tests/conftest.py`
- [x] Bump `runplz/version.py` to 4.0.0 (breaking: see below)
- [x] `./format.sh`, `./lint.sh`, `./test.sh`
- [ ] Review, merge, deploy

### The problem

`runplz ps` grew per-provider by accretion:

- Six independent dict literals (`local`, `brev` x1, `modal` x2, `aws`, `gcp`,
  `docker.parse_ps_rows`) all had to agree with `_print_ps_table`'s
  `row.get(key, "")`. Nothing checked that they did.
- `_collect_backend_jobs` mapped CLI args to driver kwargs through an
  `if backend == ...` chain, and the flags themselves were hardcoded.
- `aws.list_jobs` / `gcp.list_jobs` each resolved their own env fallbacks and
  raised after dispatch, so "what scope does this backend need" was written
  twice, in two shapes, in the wrong layer.
- `ssh` was excluded from `lists_jobs`, so `_ps_main` carried a second loop
  just for it and `runplz ps ssh` was an "invalid choice".

### Design

`listing.py` holds pure data with no provider knowledge and no argparse
import, so the dependency graph stays a DAG:
`listing` <- `docker` / `registry` / every driver.

- `JobRecord` — the one row shape. `_print_ps_table` derives its headers from
  `dataclasses.fields()`, so a renamed field can't drift from the table.
- `ScopeField` — `name` (the driver kwarg) / `flag` + `aliases` (CLI spelling)
  / `help` / `env` / `required` / `multiple` / `type`. `name` and `flag`
  decouple on purpose: `--ssh-key` feeds `ssh_key_path`.
- `ListingSpec` — `scope` plus an explicit `default_fan_out`. Explicit, not
  inferred from "every required field has an env fallback": those two
  correlate today by coincidence, and inferring would have put `ssh` in the
  bare fan-out and added a third warning line to the most common invocation.
- `resolve()` applies env fallbacks and raises `MissingScope` **before** any
  provider CLI is spawned — the acceptance criterion that scope is validated
  ahead of dispatch, not inside the driver.
- `BackendSpec.listing = None` is the explicit "cannot enumerate jobs";
  `registry.list_jobs` raises `ListingUnsupported` rather than returning `[]`.

### Compatibility notes

Four behaviours that a naive unification would have broken, all now pinned by
tests:

- The ssh probe runs **independently of the positional**: `runplz ps local
  --host box` lists local jobs *and* the box's. Selection is therefore
  "positional (or the fan-out set), plus any non-fan-out backend whose
  required scope the user supplied" — not "scope implies backend", which
  would make `runplz ps local --region r` start querying AWS.
- Missing scope stays a per-backend warning with rc 1, not a parser error
  (which would return 2 with different stderr).
- `registry.list_jobs` does not wrap provider errors, so the
  `warning: <backend> listing failed: <ExcName>: ...` line is unchanged for
  every real provider failure.
- Unset optional scope is passed explicitly as `None`, matching the ssh call
  shape the existing tests assert.

Deliberately not done *(closed later in #160)*: rejecting a scope flag that no
selected backend accepts (today it is silently ignored; erroring is a new
failure mode the issue does not ask for).

### Sequencing

`listing.py` first (nothing depends on it), then `registry`, then the six
drivers, then `cli`, then tests and docs. Tests stay green at each step
except for the intentional record-access churn.

### Review

Landed as designed. 1104 passing, coverage 95.39% (floor 95.0), with
`listing.py`, `registry.py` and `docker.py` each at 100%.

What the change actually removed: `_collect_backend_jobs`'s if-chain, the
second dispatch loop in `_ps_main`, six hardcoded flag declarations, two
copies of the env-fallback logic, and one of the two duplicated
`--ssh-port` range checks. `runplz ps --help` renders the same text as
before, generated — including the `[aws]` tags and the `Or set
AWS_DEFAULT_REGION / AWS_REGION` hints, both derived from the registry.

Behaviour added, both small:

- `runplz ps ssh` is now a valid target. It used to be `invalid choice:
  'ssh'`, which said the backend could not be listed when it only needed a
  host; it now reports the host.
- An empty table names the backends that went unasked. That is the moment
  "nothing is running" and "nobody asked" are indistinguishable, and ssh
  jobs are invisible to a bare `runplz ps`. Suppressed once rows exist or
  once a positional narrowed the selection.

Fixed in passing: `runplz.backends.aws` and `.gcp` were missing from
`tests/conftest.py`'s `_MODULES_TO_GUARD`, so `list_jobs` reached the real
`aws` / `gcloud` binaries whenever a test drove `runplz ps` on a machine
with the provider env vars set — the billed-CLI call that guard exists to
stop. Pre-existing, and this PR's tests would have widened it.

### Second review pass

The fix commits were themselves reviewed, and the first fix turned out to be
half a fix. `--host ,` was still reaching ssh with a hostname of "," on the
*positional* path: `invited_by` asked `resolve_all` ("how many targets?")
while the required-field check asked `resolve` ("is it supplied?"), and for
"," those two disagreed. The fan-out path was guarded, `runplz ps ssh
--host ,` was not, and neither was `registry.list_jobs("ssh", host=",")`.

`resolve` is now defined in terms of `resolve_all`, so the two cannot
disagree by construction, and a test asserts that invariant directly across
every blank/separator input rather than only at the CLI.

Also from that pass:

- The 4.0.0 migration block had been inserted *between* table rows, orphaning
  four of them — they would have rendered as literal pipe-delimited text on
  the PyPI project page for this release. Moved below the table.
- `resolve` applied the field's `type` to environment values but not explicit
  ones, so `registry.list_jobs("ssh", port="2200")` handed the driver a
  string while the same field from an env var gave an int. Applied to both.
- The `validate` hook only ran in the CLI's argparse loop, so it was a check
  on one spelling of the input rather than on the value. It runs in
  `ListingSpec.resolve` now, which is the path every caller takes.
- `gcp.list_jobs` silently returned `[]` for valid-JSON-but-not-a-list (`{}`,
  `null`) — reporting "no jobs", the one answer a listing must never invent,
  and the exact failure this issue exists to prevent. It raises now, matching
  aws. The aws comment claiming parity with gcp was written before that was
  true; corrected.
- `scope_fields` guarded `flag` and `name` but not `aliases`. All three keys
  are one guard now, which also removed a branch the flag/name guards had
  made unreachable.

### Version: 4.0.0, not 3.25.0

`aws.list_jobs()` and `gcp.list_jobs()` are in their modules' `__all__`, and
the README's Public API table — which it says is semver-covered — lists both
modules. Moving their env fallbacks up into the registry made `region` and
`project` required keyword arguments, so a downstream `aws.list_jobs()` with
`AWS_DEFAULT_REGION` set now raises TypeError. That is a major break by the
repo's own stated rule, whatever the usual patch-bump habit, so it ships as
one, with a migration note in the README pointing at
`registry.list_jobs(name, **scope)`.

### Code review follow-up

Twelve findings addressed. The one real regression I had introduced:
`--host ''` and `--host ,` reached ssh with a garbage hostname, because the
pre-3.25 CLI filtered them with `if h.strip()` and my `_ps_targets` split
turned "no targets" into "one empty target". The user saw
`Could not resolve hostname` — an error about their network rather than
their command.

Root cause was that resolution and splitting had been separated: `resolve`
treated a blank *explicit* value as supplied while correctly treating a
blank *environment* value as unset, and the comma split lived in the CLI
where an environment-supplied value never reached it. Both now live on
`ScopeField` (`resolve` / `resolve_all`), so a value is treated the same
way whichever source it came from. That also fixed `--region ''` reaching
the provider as `--region ''`.

The rest, briefly:

- `scope_fields` guarded flag collisions but not `name` collisions, so two
  backends could claim one driver keyword under different flags — argparse
  accepts that and cross-feeds them. Both halves are guarded now.
- `has_required_scope` answered True by vacuous `all([])`, so a backend
  declaring `default_fan_out=False` with no required scope rejoined the
  fan-out it opted out of. Renamed to `invited_by` and made explicit.
- The `--ssh-port` range check reached into the scope dict by the literal
  key `"port"`, so renaming the field would have silently disabled it. It
  is a `validate` hook on the field now, one definition shared with
  `tail`/`status`/`kill`.
- `registry.list_jobs` silently dropped undeclared keywords, turning a typo
  into "aws region is required". It raises TypeError naming what it accepts.
- Environment values skipped the field's `type`, so a typed field with an
  env fallback would have handed the driver a string.
- The "not listed" note printed even when every backend errored and no
  table was drawn, burying the warnings that explained the failure. Gated
  on the table having been printed.
- `aws.list_jobs` died with AttributeError on valid-JSON-but-not-an-object
  (`[]`, `null`), escaping its own malformed-JSON handler. Pre-existing;
  fixed to match the guard gcp already had.
- The seven pre-existing `runplz ps` CLI tests were passing only because
  gcp/aws happen to error on a machine with no cloud env. Migrated onto the
  `quiet_fan_out` fixture that was added for exactly that hazard.

Two near-misses worth keeping, both caught in design review rather than by
a test, because no test pinned either:

- The ssh probe has always run *independently of the positional*, so
  `runplz ps local --host box` lists both. The obvious "positional means
  only that backend" selection rule silently drops half the answer. Now
  pinned by `test_ps_probes_an_ssh_host_even_when_a_positional_narrows_the_rest`.
- Deriving "is in the default fan-out" from "every required field has an
  env fallback" gives the right answer for all six backends today purely by
  coincidence, and would have added a third warning line to the most common
  invocation. `default_fan_out` is stated instead.

## 2026-08-27 PR Plan — Retry idempotent SSH preparation (#84)

Branch: `fix/retry-ssh-preparation` (off `main` @ `1182715`)

- [x] Classify ssh transport failure (exit 255) and rsync's transport codes
- [x] Retry the idempotent pre-bootstrap steps with bounded backoff
- [x] Guard both launch paths with a remote marker check
- [x] Bump `runplz/version.py` to 3.19.1
- [x] `./format.sh`, `./lint.sh`, `./test.sh`
- [ ] Review, merge, deploy

### The failure

Reported against 3.17.0, Brev backend, existing running instance with auto-create
disabled. SSH readiness succeeded; the next call died:

    Read from remote host ...: Can't assign requested address
    client_loop: send disconnect: Broken pipe

Traceback: `brev.run -> run_on_provisioned_vm -> dispatch_to_target ->
_prepare_remote_run -> ssh_exec -> run_local`. The remote afterwards held only
`run.json` and a `launch_prepared` event — nothing staged, no bootstrap, no training
process. The identical command then succeeded.

### Scope / design

Not the same problem as #81. That is the vendor CLI attempt loop, before the box is
reachable. This is the transport dropping *after* readiness, inside shared SSH
preparation.

- **What makes retrying safe:** ssh reserves exit 255 for its *own* failures, as
  distinct from the remote command's exit code. A 255 means nothing ran remotely,
  so repeating an idempotent step converges on the same state. A remote command
  that genuinely failed (exit 1, no disk) is surfaced immediately — retrying it
  would only delay the error the user needs to see.
- **Retried:** `_prepare_remote_run` (writes/overwrites the run dir),
  `_ensure_remote_rsync`, `_build_image`, `rsync_up`, `rsync_down`. rsync also gets
  its own transport codes (12, 30, 35).
- **Guarded, not retried:** the detached launcher and `docker run -d`. A transport
  failure there is ambiguous — the launcher may have detached before the connection
  dropped — so the retry asks the remote whether a bootstrap pid, a start event or a
  container already exists, and declines if so. An unreachable probe counts as
  "exists": declining costs one failed run, a wrong retry costs a second training
  job on the same GPU.

### Code review — 14 findings, all addressed

Two broke the guarantee this PR is *about*:

- **`container_exists` failed open.** `ssh_capture` does not raise on ssh's exit 255,
  so my try/except was dead and an unreachable probe returned empty stdout — which
  read as "no container" and would have let a retry start a second one. It reads the
  return code now.
- **The probe fired before the backoff.** Order was `attempt, PROBE, sleep` — so
  during a real blip the probe hit the same broken link, answered "can't tell", and
  (failing closed) vetoed every retry. The guarded launch paths therefore never
  retried at all. The two bugs pointed opposite ways: container fail-open, detached
  fail-closed-always.

And the tests certified a guarantee they never exercised: they rebuilt the
`retry_on_transport_failure(..., can_retry=...)` call inline, so inverting the guard
in the source left every test green. They drive `launch_detached_and_wait` and
`_run_container_detached` now, and four deliberate mutations of the guards are each
caught by a failing test.

Also: the "255 means nothing ran remotely" claim was wrong — ssh returns 255 for a
session that drops *mid-command* too, so `_ensure_remote_rsync` now waits out an
interrupted dpkg lock rather than retrying into it; rsync's exit 12 was dropped from
the retriable set (it also means "no remote rsync binary", a deterministic failure
that would burn the budget before showing the real error); the loop now uses
`provisioning.RetryPolicy` and its shared budget helper, so it has a wall-clock
deadline and the empty-schedule guard it was missing; `rsync_down` takes the short
teardown-side budget; container-mode's image ops and native setup were left
unretried while their siblings were retried; and the `no_sleep` fixture patched
`time.sleep` process-wide when the loop already exposes a `sleep=` seam.

### Code review round 2 — 15 findings, all addressed

The round-1 fixes were themselves incomplete:

- **`container_exists` could hang out of the retry loop.** Its `subprocess.run(timeout=60)`
  raises `TimeoutExpired` from inside `can_retry`, escaping the loop and destroying the
  transport error the user needed. `container_running` — the near-identical probe 850
  lines away — already handled that. They share one `_docker_inspect` helper now.
- **"any non-zero docker exit means absent"** was wrong: a dockerd hiccup or a sudo
  refusal read as "nothing landed" and unblocked the retry. Only docker's own
  *no such object* counts as absence.
- **The launcher had a race the guard could not see.** The pid file is written *after*
  the spawn, so a probe in that window reads "nothing here" and permits a second
  bootstrap. The launcher now claims the run *before* spawning, and the probe asks
  about the claim marker as well as the pid and the start event.
- **rsync exit 12 was wrong to exclude.** Round 1 dropped it as "too broad", but it is
  the only code a mid-transfer drop produces — 30 needs a `--timeout` we never pass and
  35 is daemon-mode only. Without it the rsync retry covered nothing it existed for.
- **Two probes read a transport failure as data.** `_remote_has_nvidia` returned "no
  GPU" on a blip, silently dropping `--gpus all` — a multi-hour job on CPU on a paid GPU
  box. `_ensure_docker` read it as "daemon unreachable" and piped `get.docker.com | sudo
  sh` onto a box with working docker.
- **The dpkg repair reached one of four apt call sites**, and the `fuser` it waits with
  is absent from exactly the slim images the code exists for. One `apt_lock_wait_shell`
  helper now, with a fuser-free fallback.
- **The loop diverged from the `RetryPolicy` contract** it claimed to share: it skipped
  `waits[0]` and checked the budget after burning the backoff.
- **`fast_sleep` still patched `time.sleep` process-wide** — `ssh_common.time` *is* the
  time module — and `deadline_s` had no test at all, because every test faked sleep so
  the clock never advanced. That also surfaced a real bug: the budget was measured on
  ssh_common's clock but compared against provisioning's.

### Review section

- `./format.sh`, `./lint.sh` pass. `./test.sh` passes (`738 passed`), 33 new, and
  no existing test needed changing.
- Reproduced the issue's exact stderr against the exact call that failed, and
  proved the no-duplicate guarantee across every marker state including the
  ambiguous ones.
- No live cloud calls.

## 2026-08-27 PR Plan — Share the CLI retry loop (#81)

Branch: `feat/shared-cli-retries` (off `main` @ `053ba33`)

- [x] `run_with_retries` + `RetryPolicy`, used by brev, gcp and aws
- [x] Per-provider transient/non-retriable tables stay per-provider
- [x] gcp/aws gain retries on create, AMI lookup, describe and teardown
- [x] Bump `runplz/version.py` to 3.19.0 (3.18.0 went to #82, which merged first)
- [x] `./format.sh`, `./lint.sh`, `./test.sh`
- [ ] Review, merge, deploy

### Scope / design

The loop is common; the classification is not. An unclassified failure is neither
transient nor final — it stops immediately, because retrying an error we don't
understand is a guess made at the user's expense.

### Code review — 15 findings, all addressed

Two could have cost real money:

- **`run-instances` was retried with no `--client-token`.** A retry after a launch
  whose *response* was lost starts a second instance that nothing ever tears down.
  RunInstances is only idempotent with a client token; the run name is now sent as
  one.
- **A retried gcp create that already landed leaked the VM.** Attempt 2 returns
  "already exists", which matched neither table, so `created["ok"]` stayed False and
  teardown printed "was never created" while the box billed. Handled explicitly now.

And the tables themselves were wrong — written from memory rather than from real CLI
output. Verified against the actual strings: gcloud prints `Internal error. Please
try again`, not `internalerror`, and `Quota 'NVIDIA_T4_GPUS' exceeded`, where the two
words are never adjacent. **The GCP half retried essentially nothing.** Also missing:
`InvalidInstanceID.NotFound`, the EC2 eventual-consistency error that is the single
most likely `describe-instances` failure. And a bare `"503"` substring matched an
instance name containing `503`.

Also: no overall deadline (a hung create went 900s -> 2700s, a hung teardown held the
process for 30 minutes in a `finally`); teardown backoff widened the window where a
Ctrl-C abandons the delete (short teardown policy now); `run_cli` stopped printing the
argv whenever a policy was passed, so the machine type and disk size vanished from the
log of *successful* runs and appeared only on failure; an empty `waits` fell through to
a bare `assert`; `retry_waits` was dead configuration; and the tests covered neither
create path nor the non-retriable branch — the two things the issue's acceptance
criteria actually named.

### Review section

- `./format.sh`, `./lint.sh` pass. `./test.sh` passes (`679 passed`), 27 in the new file.
- Classification verified against real gcloud/aws error strings, not invented ones.
- Suite runtime unchanged.
- No live cloud calls.

## 2026-08-27 PR Plan — Thread ssh options, not just a port (#79)

Branch: `feat/ssh-identity-file` (off `main` @ `053ba33`)

- [x] `SshOptions` replaces the bare `port` threaded through ~50 call sites
- [x] `SshConfig.ssh_key_path` / `AwsConfig.ssh_key_path`
- [x] Follow-up commands (`tail`/`status`/`kill`/`ps`) can reach a keyed box
- [x] Bump `runplz/version.py` to 3.18.0
- [x] `./format.sh`, `./lint.sh`, `./test.sh`
- [ ] Review, merge, deploy

### Scope / design

- One object, not a second scalar. `port` alone was already threaded through ~50
  sites; adding `identity_file` beside it would have doubled that and made a third
  worse. `ssh_cmd_opts` / `rsync_ssh_transport` coerce None / int / SshOptions.
- `provision()` returns `(target, ssh_opts)` now — aws only learns its target after
  the box exists, so the options travel back with it.
- The key reaches rsync too, via its `-e` transport, and only when the options
  actually differ from ssh's defaults.

### Code review — 15 findings, all addressed

Two were expensive:

- **No host-key policy in `SSH_OPTS`.** Every probe runs `BatchMode=yes`, which
  can't answer OpenSSH's default prompt, so a brand-new EC2 IP would fail
  verification on every attempt for the full 1800s wait *on a billed GPU box*.
  Only `_ensure_docker` had been passing `accept-new`; it is shared now.
- **The key path was written into the manifest uploaded to the remote box** — the
  same document where the codebase already masks anything env-shaped containing
  "KEY". It disclosed the local username and key filename to a rented, possibly
  multi-tenant host. Moved to a local-only sidecar (`.runplz/ssh.json`), which
  survives `rsync_down` because that has no `--delete`.

Also: `--ssh-key` alone silently discarded the recorded port (merged per field now);
`IdentitiesOnly=yes` plus a typo'd path disabled the agent fallback with no
diagnostic (both configs validate the path exists); `runplz ps` had no way to
authenticate to the box this PR enables; `run_on_provisioned_vm` replaced rather
than merged caller options; `extra_opts` was unreachable from every public surface
(dropped rather than surfaced on a guess); `--ssh-port` had no range check; the
rsync default-transport check built and string-compared two full command lines,
twice; README contradicted itself and `aws.py` still steered users to `ssh-add`.

And one embarrassing one: `test_aws_hands_its_key_back_with_the_target` asserted
`... or True` and could never fail — the exact regression the PR describes catching
would have shipped green. It drives `provision()` now.

**Breaking change, called out rather than papered over:** the keyword-only `port=`
parameter on the public ssh helpers is now `ssh_opts=`. `ssh_cmd_opts(2222)` still
works positionally, but `ssh_exec(target, cmd, port=2222)` does not. Those names went
public one day ago in 3.17.0; carrying permanent aliases for a one-day-old surface
costs more than it saves.

### Review section

- `./format.sh`, `./lint.sh` pass. `./test.sh` passes (`678 passed`), 32 new.
- Verified the identity reaches the ssh argv, the rsync transport, and a follow-up
  `runplz kill` reconstructed from the sidecar.
- No live cloud calls.

## 2026-08-26 PR Plan — Direct GCP / AWS provisioning backends (#25)

Branch: `feat/gcp-aws-backends` (off `main` @ `27fd9c5`)

- [x] Commit 1 — extract the shared VM dispatch + lifecycle core into `ssh_common`
- [x] Commit 2/3 — `gcp.py` + `GcpConfig`, `aws.py` + `AwsConfig`, both CLI-based
- [x] Bump `runplz/version.py` to 3.17.0
- [x] Run `./format.sh`, `./lint.sh`, `./test.sh`
- [ ] Review the final diff, push, open a PR closing #25, merge, deploy 3.17.0

### The "is there a library for this" question

No third-party one worth taking. Apache Libcloud adds a dependency to a repo whose
stated scope is stdlib-only, and its GCE accelerator support is weak. SkyPilot/dstack
sit at runplz's own layer and would subsume it rather than serve it. The abstraction
worth having was *internal* and already half-present: `ssh.py::run()` and
`brev.py::run()` duplicated the dispatch core almost line for line. Extracting it once
means a new cloud is ~150 lines of provision → hand over a target → tear down.

### CLI, not SDK

The issue text suggested `boto3 ec2.run_instances`. Went the other way, because
`tests/conftest.py` already lists `gcloud` and `aws` in `_BILLED_COMMANDS` behind
`live_gcp`/`live_aws` markers — an SDK call is invisible to that guard, so a test that
forgot to mock could launch a real p5.48xlarge. CLI keeps the new code inside the
existing safety net and the core dependency-free. Added `test_conftest_guard` cases
proving the guard covers `runplz.backends._cloud`.

### Scope / design

- `dispatch_to_target()` owns its own container cleanup and failure-tail capture,
  because both must happen before a provisioning caller deletes the box — afterwards
  the logs are gone.
- `run_on_provisioned_vm()` opens its try *before* provision so a mid-provision failure
  still reaches teardown (issue #29), and only tears down if something was allocated.
- `orchestrator_signal_cleanup` moved up from brev so every provisioning backend gets
  the SIGTERM/SIGHUP billing-leak protection from #38, not just brev.
- GCP SSH rides on `gcloud compute config-ssh` — a direct analogue of `brev refresh`,
  so it plugs into the existing `_wait_until_ssh_reachable` refresh callback with no
  new ssh plumbing.
- AWS always sends an explicit block-device-mapping even at the AMI's default size:
  that is what pins `DeleteOnTermination`, without which `on_finish="delete"` leaves a
  billed EBS volume and fails the issue's own "deletes VM + disks" criterion.
- brev.py migrated onto the shared core as well. First cut deferred it (filed #78) on
  risk grounds; that was wrong — brev is the backend in daily use, so leaving it on a
  private copy means shared-core fixes never reach the code that matters most, and its
  2800 lines of tests are the safety net that makes the migration checkable, not a
  reason to avoid it. -93 lines. Doing it surfaced a real bug in my own extraction:
  `run_on_provisioned_vm` skipped teardown when provision() raised, which is exactly
  the billing leak #29 fixed. Teardown is now unconditional.

### Unification pass

After brev moved onto the shared core, kept going through the rest of the
per-backend duplication:

- **`_docker.py`** — container labels were string literals in four files (written in
  `local.py` + `ssh_common.py`, read back in `local.py` + `ssh.py`), so a rename would
  have silently broken `runplz ps` rather than failing anything. Producer and consumer
  now share constants; the two near-identical `docker ps` row parsers and the
  verbatim-duplicated label parser collapse into one.
- **`config.py`** — one `_validate_remote_common` backs all four remote configs instead
  of three copies of the same three checks.
- **`_cloud.py`** — owns instance naming for every provisioning backend, both halves.
  `make_instance_name` / `split_instance_name` are inverses that must agree (ps reads
  app/function back out of a name) and were 200 lines apart in a brev-only file.
- **`apply_teardown()`** — brev, gcp and aws each restated the same billing-safety
  contract (leave = nothing; never raise, it runs in a finally; never fail quietly,
  a silent teardown is a box that bills). One implementation owns the rules; each
  backend supplies only its provider's command. brev keeps its retries and
  post-action verification inside its own action callable.
- **`_registry.py`** — adding a backend meant editing five places that had to agree
  (bind validator, argparse choices, `_dispatch` if-chain, ps tuple, ps if-chain).
  Now one entry.

Net: ssh/gcp/aws are 7-9 functions each. brev's remaining 30 are all genuinely
`brev`-CLI vocabulary — ls/create/start/refresh, search-row parsing, onboarding,
terminal states, instance-type picking.

**Stopped short of** merging `brev._brev_capture` into `_cloud.run_cli`. The retry
*loop* is common but the *classification* is not — brev's transient/non-retriable
patterns are Brev-API-specific (issue #62's org/config gaps), and GCP quota errors
and AWS throttling look nothing like them. The contracts differ too: `_brev_capture`
returns a CompletedProcess for the caller to inspect, `run_cli` raises. Extracting
just the loop would be a thin win wrapped around the most heavily-tested code path in
the repo. Filed as a follow-up rather than forced.

### Making the shared layer public

The unified modules were private (`_cloud`, `_docker`, `_registry`) and much of the
shared contract sat behind underscores, which was the wrong signal: a backend is
*expected* to be written against this layer.

- Modules renamed with semantic names: `_docker` -> `backends.docker`,
  `_cloud` -> `backends.provisioning` (brev provisions too, so "cloud" was wrong),
  `_registry` -> `backends.registry`.
- Promoted the 13 ssh_common names other modules actually call:
  `wait_until_ssh_reachable`, `ssh_exec`, `ssh_capture`, `ssh_cmd_opts`, `run_local`,
  `rsync_down`, `rsync_ssh_transport`, `parse_probe_sections`, `container_running`,
  `raise_for_runtime_cap`, `render_image_ops_script`, `CLEANUP_SIGNALS`,
  `validate_remote_path`. No compat aliases — every reference was updated.
- `remote_shell_path` moved from `_runs` into `ssh_common` alongside
  `validate_remote_path` / `validate_run_id` / `is_safe_run_id`. `_runs` was reaching
  across a module line for two regexes; now it calls functions.
- Every shared module declares an explicit `__all__` grouped by purpose.

**What stayed private, deliberately.** The staging/streaming helpers
(`_prepare_remote_run`, `_build_image`, `_run_container_detached`, `_stream_and_wait`,
`_run_native`, `_check_preconditions`, ...) are internals of `dispatch_to_target`.
`dispatch_to_target` *is* the contract; a backend should never need to reach past it.
`tests/test_public_api.py` pins both halves: nothing exported may be private, and
those internals may not appear in `__all__`.

One behavior change fell out: rejecting a tampered manifest path now raises
`ValueError` (the accurate type) rather than `RuntimeError`, so the three CLI
handlers catch both. The message got better in the process — it names where the bad
value came from.

### Review section

- All flags were validated offline against the installed CLIs (`gcloud compute
  instances create --help`, `aws ec2 run-instances help`) before being emitted —
  caught `--subnet` vs `--subnetwork` without touching either cloud account.
- Running `dry_run=True` end to end caught a real bug: gcp's teardown was missing the
  flag and **actually invoked `gcloud compute instances delete`** during a dry run. It
  failed only because the local auth token happened to be stale. Fixed, plus a test per
  backend asserting `subprocess.run` is never called in dry-run.
- `./format.sh` and `./lint.sh` pass. `./test.sh` passes (`577 passed`, 93% coverage),
  55 of them new.
- No live cloud calls at any point. Nothing was provisioned and nothing was billed.

## 2026-08-26 PR Plan — Sweep of #75 / #72 / #67 (kill + packaging)

Branch: `feat/runplz-kill-and-packaging-fixes` (off `main` @ `102cc11`)

### Sweep verdict

| Issue | Status before this PR | Evidence |
|---|---|---|
| #75 deploy.sh PEP 668 | Still broken | `deploy.sh:10` still ran `python3 -m pip install --upgrade build twine`, and only *after* `./lint.sh` + `./test.sh` |
| #72 SPDX license | Still broken | `pyproject.toml:10` still used the `license = {file = ...}` table; `requires = ["setuptools>=61.0"]` predates PEP 639 |
| #67 `runplz kill` | Partially landed | `runplz ps` / `tail` / `status` shipped in 3.15.x — those were the issue's *bonus* asks. The core `kill` did not exist; its plumbing (run manifest, pid probe, zombie-aware process state) now does. |

Nothing was already fully fixed; #67 had shrunk to just the kill itself.

- [x] #75 — provision an isolated build venv in `deploy.sh` *before* lint/test, so it both
  stops mutating an externally managed interpreter and fails fast; gitignore `.deploy-venv/`
- [x] #72 — `license = "Apache-2.0"` + `license-files = ["LICENSE"]`, `setuptools>=77.0.3`;
  verified `License-Expression` / `License-File` in both wheel and sdist metadata
- [x] #67 — `runplz kill` / `runplz cancel`
- [x] #76 (found en route) — manifest paths are `~/`-prefixed and tilde expands before
  quoting, so `tail`/`status`/`kill` were all reading a path the remote shell cannot resolve;
  rewrite the leading `~/` to `$HOME/` and validate `--run-id` before it enters a shell command
- [x] Bump `runplz/version.py` to 3.16.0
- [x] Run `./format.sh`, `./lint.sh`, and `./test.sh`
- [ ] Review the final diff, push, open a PR closing #75/#72/#67, wait for green CI, merge,
  and deploy 3.16.0 to PyPI

### Scope / design

> The first cut here keyed on the process *group*; code review showed that was unsound.
> See "Code review round 2" below for what actually shipped. Bullets updated to match.

- **Identify a run's processes by a per-run environment marker.** The painful case in #67 is
  the one where the bash supervisor is already dead and only orphaned workers survive,
  reparented to init — `pkill -P` has nothing left to match. The bootstrap is exec'd with
  `RUNPLZ_RUN_ID=<id>`, every descendant inherits it, and kill scans `/proc/*/environ` for it.
  Unique to the run, survives reparenting, immune to PID wraparound.
- Runs launched before the marker existed fall back to the recorded bootstrap pid — but only
  while no terminal exit event has been recorded, since a finished run's pid can be recycled.
- VM+docker mode records the container name in `<meta>/container`. The container is a child of
  dockerd, so it carries no marker of ours and must be signalled separately.
- The whole signal → poll → escalate dance runs remotely in one ssh hop, so the escalation
  clock measures the job rather than ssh latency and a flaky link can't strand a half-signalled
  run.
- Guards: non-numeric pids are discarded and pid 0/1 refused numerically — `kill -TERM 0`
  signals every process in the caller's own group. Zombies are excluded so they can't pin the
  wait loop into a pointless SIGKILL.
- Everything interpolated into the remote shell is validated first, including the meta path,
  which arrives from a manifest rsynced down off the remote box.
- Idempotent by design: killing an already-finished run prints `nothing to kill` and exits 0,
  so it is safe in a relaunch script.
- Reused `_add_run_lookup_args` and `_parse_status_sections` rather than inventing a second run
  lookup or output format.

### Code review round 2 — 15 findings, all addressed

The first cut identified a run's processes by **process group**. That was
wrong: bash disables job control in non-interactive shells, so the bootstrap
never becomes a group leader and the recorded pgid is the *launching shell's*.
Blast radius on anything sharing it, and no protection against PID wraparound
reusing a stale pgid/pid from a finished run.

Replaced with a per-run **environment marker**: the bootstrap is exec'd with
`RUNPLZ_RUN_ID=<id>`, every descendant inherits it, and kill scans
`/proc/*/environ` for it. Unique, survives reparenting, cannot be recycled.
That one change resolved the group-blast-radius, PID-wraparound, zero-padded-
pgid and duplicated-procfs-parsing findings together.

Also fixed:
- `alive_after` / `survivors` are now reported, so a kill that left processes
  running can no longer print "stopped with SIGTERM"
- `kill` exits 3 when anything survives and 2 when the remote script produced
  no readable result — it used to return 0 in both cases
- pid `0`/`1` rejected numerically (`kill -TERM 0` signals our own group)
- `remote_shell_path` validates the manifest path: dropping `shlex.quote` for
  `$HOME` expansion had opened `$(...)` execution from a manifest that is
  rsynced down off the remote box
- `run_id`, `first_signal`, `timeout_s`, `proc_root` validated in
  `build_kill_command`, not just at the argparse layer
- the reported signal is the one actually sent (`--signal INT` no longer
  claims SIGTERM)
- log tail lines are prefixed so a `--- ... ---` line in a log can't truncate
  the section parser
- one shared heartbeat renderer for `status` and `kill`
- the runtime-cap timeout path no longer `pkill`s every runplz bootstrap on
  the box; it stops the specific run
- `BOOTSTRAP_PID_FILENAME` now actually centralizes the filename
- a real NUL byte was reaching the generated script instead of the `\0`
  escape `tr` needs — caught by executing the script, not by reading it

### Review section

- Verified the generated kill shell by executing it, not just by string-matching: it takes down
  an orphaned bootstrap + 2 workers in a real process group, is a clean no-op on a run that never
  existed, and refuses pgid 1. Parses under both `sh -n` and `bash -n`.
- Also fixed two latent problems found on the way:
  - **#76**: `runplz status` was silently reporting `(none recorded)` for healthy runs because
    the tilde path never resolved and its errors are swallowed by `2>/dev/null || true`. `kill`
    would have shipped with the same bug, so this had to be fixed here rather than deferred.
  - `runplz._runs` shells out to `ssh` but was missing from `conftest._MODULES_TO_GUARD`, so a
    test that forgot to mock could have hit real infra — exactly what that guard exists to
    prevent.
- `./format.sh` and `./lint.sh` pass. `./test.sh` passes (`522 passed`, 93% total coverage),
  with the new tests clean under `-W error::DeprecationWarning` across 5 consecutive runs.

## 2026-08-26 PR Plan — HUP-Safe Detached Bootstrap (#73)

Branch: `fix-detached-bootstrap-hup` (off `main` @ `dc5cd38`)

- [x] Ignore SIGHUP in the remote launcher shell before spawning `nohup`, closing the pre-exec
  race where SSH teardown can kill the child before `nohup` installs its signal disposition
- [x] Retain SIGHUP ignore inside the generated `run.sh` wrapper as defense in depth
- [x] Add regression coverage for launcher ordering and a real detached child surviving SIGHUP
- [x] Bump `runplz/version.py` to 3.15.4
- [x] Run `./format.sh`, `./lint.sh`, and `./test.sh`
- [ ] Review the final diff, commit, push, open a PR closing #73, wait for green CI, merge, and
  deploy 3.15.4 to PyPI

### Scope / design

- Keep the existing PID-stable `nohup bash` launcher, lifecycle events, startup handshake, and
  reconnect monitor unchanged.
- Install the ignored HUP disposition in the parent shell before `nohup` is forked so the child
  inherits safety during the fork-to-exec window; install the same disposition at the top of the
  generated child script so later shell behavior cannot reintroduce the hazard.
- Lock both properties with a structural ordering assertion plus an executable signal test that
  sends HUP to the recorded detached PID and observes successful completion.

### Review section

- The launcher now installs the ignored HUP disposition before any child can be forked and repeats
  it as the first line of `run.sh`; PID tracking, event semantics, diagnostics, and reconnect logic
  are unchanged.
- The regression test executes the generated launcher, signals the recorded child PID, and proves
  the payload survives and completes.
- Focused SSH/Brev tests pass (`144 passed`).
- `./format.sh` and `./lint.sh` pass.
- `./test.sh` passes (`452 passed`, 93% total coverage).

## 2026-08-20 PR Plan — Git-Aware Staging + Reliable Detached Bootstrap (#68, #69)

Branch: `fix-ignored-staging-detached-bootstrap` (off `main` @ `9fd9a41`)

- [x] Move shared SSH behavior from private `_ssh_common` to public
  `runplz.backends.ssh_common`, retaining a compatibility import for the old module path
- [x] Add a public, Git-aware source-staging contract that selects tracked files plus intentional
  untracked files while omitting ignored artifacts and deleted tracked paths
- [x] Preserve the existing full-tree rsync behavior for directories that are not Git worktrees
- [x] Report tracked dirtiness, intentional untracked inputs, and ignored artifacts separately in
  remote run manifests
- [x] Replace the non-portable `nohup setsid bash` launcher with a portable, PID-stable detached
  `nohup bash` launcher whose stdin/stdout/stderr are fully redirected
- [x] Add a bounded startup handshake that recognizes missing, dead, and zombie bootstrap PIDs
  before entering log streaming
- [x] Make detached log streaming terminate when the bootstrap exits or becomes a zombie, while
  retaining SSH reconnect and runtime-cap behavior
- [x] Include detached driver-log context in failure diagnostics and continue through normal output
  download so the generated run directory is recoverable locally
- [x] Add public-contract regression tests for ignored/untracked/tracked/deleted staging, manifest
  state, portable launch construction, startup failure, zombie detection, and stream termination
- [x] File the stale `test.sh` regression test discovered by the full suite as `#70` and update its
  assertion to cover the configurable `PYTHON_BIN` module invocation
- [x] Bump `runplz/version.py` for the PR
- [x] Run `./format.sh`
- [x] Run `./lint.sh`
- [x] Run `./test.sh`
- [x] Review the final diff for minimal scope and compatibility
- [x] Commit confirmed paths, push the branch, and open draft PR `#71` closing `#68`, `#69`,
  and `#70`

### 2026-08-20 review-fix plan

- [x] Keep `UNKNOWN` and live-but-not-started bootstrap states in detached reconnect monitoring;
  fail startup only for explicit missing, dead, or zombie states
- [x] Replace GNU `ps -o stat=` lifecycle checks with a shared `/proc/<pid>/stat` shell contract
  that distinguishes zombies using only shell builtins and `kill -0`
- [x] Exclude every selected Git path absent from the working tree, including sparse-checkout
  skip-worktree entries, while preserving broken symlinks with `lexists`
- [x] Parse superproject gitlinks and recursively prefix each initialized submodule's own
  tracked-plus-nonignored-untracked selection; never pass an opaque gitlink directory to rsync
- [x] Add regression tests for transient-SSH startup uncertainty, procps-free process probes,
  sparse checkouts, initialized submodule ignores, and uninitialized submodules
- [x] Run `./format.sh`, `./lint.sh`, and `./test.sh`
- [x] Re-review the complete PR diff and repeat the fix/review cycle until no actionable findings
  remain
- [ ] Push fixes, wait for all PR CI jobs, mark PR #71 ready, merge it, update clean `main`, and run
  `./deploy.sh` through PyPI upload plus tag push

#### Review-fix design

- Detached lifecycle:
  - startup probing returns terminal failure only for `MISSING`, `DEAD`, or `ZOMBIE`; `RUNNING`
    without an event and `UNKNOWN` both flow into `tail_and_wait_for_detached`
  - generate one shared remote shell fragment that reads `/proc/$pid/stat`, extracts the state after
    the final process-name parenthesis, and falls back to `kill -0` when `/proc` is unavailable
  - the log follower stops only for explicit terminal state; an unavailable state source remains
    conservative and the existing reconnect/runtime-cap logic stays authoritative

- Git source selection:
  - retain Git's cached-plus-nonignored-untracked selection, but require `os.path.lexists` before a
    path reaches rsync so deleted and sparse-absent entries cannot produce stat failures
  - query staged modes to identify `160000` gitlinks, remove those directory entries from the flat
    selection, and recursively select initialized submodule contents with path prefixes
  - omit uninitialized/absent submodules instead of recursively copying an opaque directory or
    falling back to full-tree behavior that would bypass submodule ignores

- Verification:
  - build real temporary sparse worktrees and local submodules in public-contract tests
  - assert generated lifecycle shell has `/proc` state parsing and no `ps` dependency
  - prove unknown startup calls the reconnect monitor and never records launch failure

#### Review-fix results

- All four supplied review findings are addressed with regression coverage.
- A second complete diff review found no additional actionable staging or lifecycle defects.
- `./format.sh` and `./lint.sh` pass.
- `./test.sh` passes (`451 passed`, 93% total coverage).

### Scope / design

- Public SSH surface:
  - make `runplz.backends.ssh_common` the canonical home of backend-agnostic SSH staging,
    lifecycle, and transport behavior
  - expose the new source-selection, repository-state, launcher-construction, and detached-state
    contracts under public names so callers and tests do not need private imports
  - leave a small `_ssh_common` compatibility module so existing imports do not break abruptly

- Git-aware staging:
  - ask Git for the union of cached/tracked paths and untracked, non-ignored paths
  - subtract tracked paths deleted from the working tree, then pass the NUL-delimited selection to
    rsync with `--files-from`, `--from0`, and explicit recursion
  - keep the current noise, secret, and configured-output exclusions as defense in depth
  - fall back to the current full-tree rsync command when the source is not a Git worktree or Git
    cannot produce a selection
  - parse porcelain Git status into separate `repo_dirty`, `repo_untracked`, and `repo_ignored`
    manifest fields; `repo_dirty` will mean tracked modifications rather than ignored artifacts

- Detached bootstrap:
  - construct the launcher as `nohup bash <run.sh> </dev/null >>driver.log 2>&1 &`; `nohup` execs
    bash without `setsid`'s fork/session-leader ambiguity, keeping `$!` tied to the bootstrap
  - poll briefly for the first `remote_command_start` event; fail with process/event/driver-log
    diagnostics if the PID is absent, dead, zombie, or never reaches the startup event
  - make the remote log-tail command watch the PID state and stop its `tail -F` child when the job
    is no longer live, so a dead launcher cannot block the client forever
  - return a normal nonzero run status for launch failures so callers still rsync the per-run output
    and metadata before applying Brev `on_finish`

- Tests:
  - initialize small temporary Git repositories to verify the exact public source-selection
    semantics, including tracked files later covered by ignore rules
  - exercise public launcher/process-state contracts with captured commands and representative
    alive/dead/zombie/start-event responses
  - retain backend orchestration tests for port propagation, rsync destinations, reconnect limits,
    runtime caps, output download, and failure-tail behavior
  - update the stale test-script assertion from issue `#70`; current `main` changed the runner to
    `"$PYTHON_BIN" -m pytest` without updating the old hardcoded-string assertion

### Review section

- Implemented:
  - public `runplz.backends.ssh_common` module with explicit supported exports and a compatibility
    import at the former `_ssh_common` path
  - Git-selected rsync input using tracked plus non-ignored untracked paths, minus working-tree
    deletions; non-Git directories retain full-tree staging
  - separate `repo_dirty`, `repo_untracked`, and `repo_ignored` manifest state
  - portable `nohup bash` launcher without `setsid`, plus bounded startup-event verification
  - zombie-aware process probes and a remote tail wrapper that stops when the bootstrap is no
    longer live
  - launch-failure diagnostics covering PID state, recent lifecycle events/heartbeats, driver log,
    and bootstrap log while preserving the normal output download path
  - user-facing staging documentation and version bump to `3.15.3`
  - issue `#70` plus its narrow stale test-script assertion fix

- Validation:
  - `./format.sh` passed
  - `./lint.sh` passed
  - `./test.sh` passed (`447 passed`, 93% total coverage)
  - targeted SSH/Brev/public-contract tests passed before the final full-suite cycle
  - final `git diff --check` passed
  - draft PR opened: `pirl-unc/runplz#71`

## 2026-04-23 PR Plan — Remote Run Forensics + Brev Lifecycle Diagnostics

- [x] Introduce a shared per-launch remote run context for SSH/Brev
- [x] Replace fixed `~/runplz-repo` / `~/runplz-out` usage with unique per-run paths
- [x] Emit structured remote lifecycle metadata (`run.json`, `events.ndjson`, `heartbeat.ndjson`)
- [x] Sync lifecycle metadata back through the normal outputs download path
- [x] Preserve richer per-attempt Brev lifecycle diagnostics in the driver log
- [x] Verify post-action Brev state after `create`, `start`, `stop`, and `delete`
- [x] Add regression coverage for the new remote pathing, lifecycle logging, and Brev verification
- [x] Bump `runplz/version.py` for the PR
- [x] Run `./format.sh`
- [x] Run `./lint.sh`
- [x] Run `./test.sh`
- [x] Commit, push branch, and open a draft PR linked to `#48`, `#49`, and `#50`

### Scope / design

- Shared SSH layer:
  - add a `RemoteRunContext` that generates a stable `run_id` plus unique
    remote directories under `~/runplz-runs/<run_id>/`
  - use per-run `repo/`, `out/`, and metadata paths instead of fixed
    `~/runplz-repo` / `~/runplz-out`
  - print the chosen remote paths in the local driver log
  - write `run.json`, `events.ndjson`, and `heartbeat.ndjson` under the
    remote out tree so `rsync_down` brings them back automatically
  - wrap native/container-mode remote commands with event/heartbeat/trap
    logging so parent-process lifecycle is auditable even when the user
    payload fails
  - record host-side events around rsync/build/cleanup transitions

- Brev backend:
  - preserve fuller stdout/stderr context for each retry attempt, including
    timestamps and elapsed time in `_brev_capture`
  - add an instance snapshot helper over `brev ls --json`
  - verify lifecycle post-state after create/start/stop/delete with a short
    poll window and loud warnings when the CLI says success but the box still
    exists / stays running
  - include a final instance snapshot in create-failure paths

- Tests:
  - assert rsync/build/run helpers use the new per-run paths
  - assert lifecycle files/paths are threaded into run helpers
  - assert Brev lifecycle verification and diagnostics fire on the expected paths
  - keep SSH/Brev end-to-end happy-path tests patched against the new helper API

### Review section

- Implemented:
  - shared `RemoteRunContext` plumbing for SSH/Brev with unique remote
    `~/runplz-runs/<run_id>/repo` and `out` directories plus a stable
    `~/runplz-latest` symlink
  - per-run remote metadata under `out/.runplz/` with `run.json`,
    `events.ndjson`, `heartbeat.ndjson`, and per-run `last.log`
  - remote event/heartbeat/trap logging for native and container-mode
    execution, plus detached-container monitoring for the VM+docker path
  - richer Brev lifecycle diagnostics with attempt timestamps, elapsed
    time, full retry-attempt stdout/stderr, and post-action `brev ls`
    verification
  - create-failure snapshots and loud warnings when `stop` / `delete`
    report success but the instance still looks alive afterward
  - version bump to `3.9.2`

- Validation:
  - `./format.sh` passed
  - `./lint.sh` passed
  - `./test.sh` passed (`342 passed`, `95%` total coverage)
  - draft PR opened: `pirl-unc/runplz#51`

## 2026-04-23 PR Plan — Fix Review Issue #46

- [x] Fix unsafe Modal tar extraction and add regression coverage
- [x] Fix CLI default log-path rooting so logs follow the repo outputs dir
- [x] Fix SSH/Brev reconnect handling so reconnect fallback does not bypass runtime caps
- [x] Fix remote Dockerfile builds to honor `Image.from_dockerfile(..., context=...)`
- [x] Fix `test.sh` so pytest-cov measures the local checkout, not an installed package
- [x] Bump `runplz/version.py` for the PR
- [x] Run `./format.sh`
- [x] Run `./lint.sh`
- [x] Run `./test.sh`
- [x] Commit, push branch, and open a draft PR linked to `#46`

### Review section

- Implemented:
  - safe Modal tar extraction with member validation and Python-3.14-safe
    `filter="data"` extraction when available
  - repo-rooted default CLI log placement
  - bounded `docker wait` after SSH log-stream reconnect exhaustion
  - remote Dockerfile build-context support for SSH/Brev VM builds
  - `test.sh` switched to `python -m pytest --cov=runplz ...`
  - version bump to `3.9.1`

- Validation:
  - `./format.sh` passed
  - `./lint.sh` passed
  - `./test.sh` passed (`336 passed`)
  - coverage now reports real local-checkout data (`95%` total) instead of
    `CoverageWarning: No data was collected`
  - draft PR opened: `pirl-unc/runplz#47`

## 2026-04-23 Bug / Error Review

- [x] Inspect repository state and identify the main code paths to review
- [x] Run `./format.sh`
- [x] Run `./lint.sh`
- [x] Run `./test.sh`
- [x] Review likely failure-prone backend and CLI paths for bugs not covered by the checks
- [x] Document findings and residual risks

### Review section

- Confirmed findings:
  1. `runplz/backends/modal.py` extracts untrusted tar output with
     `tar.extractall(dest)` and allows path traversal outside `dest`.
  2. `runplz/_cli.py` resolves the default log path relative to
     `Path.cwd()`, not the backend outputs directory rooted at the repo.
  3. `runplz/backends/_ssh_common.py::_stream_and_wait()` says it is
     giving up after max reconnects, then immediately does an unbounded
     `docker wait` anyway.
  4. `runplz/backends/_ssh_common.py::_build_image()` ignores
     `Image.from_dockerfile(..., context=...)` on SSH/Brev remote builds.
  5. `./test.sh` reports green tests, but the coverage signal is broken:
     `pytest --cov ...` emits `CoverageWarning: No data was collected`
     and reports 0% coverage for the whole package.

- Validation run:
  - `./format.sh` passed
  - `./lint.sh` passed
  - `./test.sh` passed (`331 passed`), with the broken coverage warning
    and a Python 3.14 tar-extraction deprecation warning from
    `runplz/backends/modal.py`

- Tracking issue:
  - `pirl-unc/runplz#46`

# runplz 3.3 — seven footguns in one PR

Branch: `3.3-seven-footguns` (off main @ v3.2.0)

Bundles seven open footgun issues into a single minor release. Grouped
this way because several fixes touch the same code paths (brev.run,
modal tar roundtrip, cross-backend transfer excludes) and splitting
would mean three PRs racing on the same lines.

## Issues closed

- [ ] **#14 — Brev `--instance` typo auto-creates a billed box.**
  Flip `BrevConfig.auto_create_instances` default `True → False`.
  Improve the "not found" RuntimeError to surface the exact override
  (`auto_create_instances=True`) rather than make the user grep the
  docs. Breaking default; acceptable because the incorrect behavior
  costs real money.
- [ ] **#17 — Brev failures raise "exited with status N" with no log
  context.** Capture the last ~50 lines of remote output and include
  in the RuntimeError. For VM+docker: `docker logs --tail 50` before
  `docker rm -f`. For container-mode / native: ring buffer during
  streaming.
- [ ] **#16 — `docker wait` has no wall-clock cap.** Add
  `BrevConfig.max_runtime_seconds: Optional[int] = None`. On trip,
  issue `docker kill` on the remote and raise.
- [ ] **#19 — Modal output tarball silently truncates at ~256MB.**
  Measure blob size; warn > 200MB with a pointer to Modal Volumes;
  raise > 256MB because data may already be lost.
- [ ] **#20 — Modal `min_disk` silently dropped.** Convert the existing
  `print()` into a `ValueError` at dispatch.
- [ ] **#18 — rsync_up has no default `.env` / secret excludes.**
  Centralize `DEFAULT_EXCLUDES` (covers `.env*`, ssh keys, `.aws/`,
  `credentials.json`, `*.pem`, `*.key`). Plumb through brev `_rsync_up`
  and modal `add_local_dir`.
- [ ] **#21 — Local `--no-build` reuses last tag without telling you.**
  Print the reused image tag so the user can confirm the intended
  image is about to run.

## Implementation order

Each landed + tested before moving on, so a failure mid-PR leaves
a clean partial branch.

1. **tasks/todo.md** (this file)
2. **#14** — config + brev.py error message (smallest)
3. **#21** — local.py one-liner
4. **#20** — modal.py ValueError (replaces print)
5. **#18** — shared DEFAULT_EXCLUDES constant, wire into brev + modal
6. **#17** — brev log-tail capture
7. **#19** — modal output size guard
8. **#16** — max_runtime_seconds plumbing (largest)
9. Version bump to 3.3.0 + README
10. format.sh / lint.sh / test.sh → commit → PR → merge → deploy.sh

## Test plan

New tests per issue, landed alongside code:

- #14: BrevConfig default asserts False; `run()` raises with override
  mentioned when instance missing + auto_create=False.
- #21: `build=False` path prints a line containing the tag.
- #20: `min_disk=1` on modal dispatch raises ValueError.
- #18: rsync cmd includes each DEFAULT_EXCLUDES entry; modal image
  builder filters them from the context.
- #17: non-zero exit → RuntimeError message contains log-tail text.
- #19: blob > 200MB warns; blob > 256MB raises.
- #16: cap None = unchanged; cap exceeded = docker kill + raise.

## Out of scope

- Breaking field renames. `auto_create_instances` keeps its name.
- Wiring Modal `min_disk` through to the Modal API — issue is the
  silent drop, not the missing feature. If Modal adds disk-size later
  we can wire it; not this PR.
- Adding a `.dockerignore` generator for local. Docker already honors
  `.dockerignore` if present; #18's shared excludes do not apply to
  the local docker-build context (different transport).

## Review section

_Filled in after implementation._

---

## Public API surface: minimize private names (3.20.0)

**Rule applied:** a module or name typed by anything outside its own file is
public. Underscores are reserved for genuinely file-local helpers.

**Constraint:** no compat aliases for names that get patched. `_old = new` and
`__getattr__` module shims are read-safe but patch-unsafe — patching the alias
sets an attribute nothing reads, so the test passes without testing. Rename
outright, update every call site.

### Phase 1 — `ssh_common.py` dispatch pipeline goes public
The module was made public at 3.15.3 but its 11 pipeline stages stayed
underscore-private while being imported by `brev.py`. Promote them and give the
module an `__all__`.

- [x] `_prepare_remote_run` → `prepare_remote_run`
- [x] `_ensure_remote_rsync` → `ensure_remote_rsync`
- [x] `_check_preconditions` → `check_preconditions`
- [x] `_ensure_docker` → `ensure_docker`
- [x] `_build_image` → `build_image`
- [x] `_run_container_detached` → `run_container_detached`
- [x] `_run_container_mode` → `run_container_mode`
- [x] `_run_native` → `run_native`
- [x] `_stream_and_wait` → `stream_and_wait`
- [x] `_fetch_failure_tail` → `fetch_failure_tail`
- [x] `_remote_has_nvidia` → `remote_has_nvidia`
- [x] update `brev.py` re-export block, `ssh.py`, `local.py`, ~176 test refs

### Phase 2 — private modules that are already public interfaces
- [x] `_cli.py` → `cli.py` — named in `pyproject.toml:43` entry points
- [x] `_bootstrap.py` → `bootstrap.py` — `python -m runplz._bootstrap` is baked
      into generated remote shell at 3 sites; its env-var protocol is a wire
      contract and `$RUNPLZ_OUT` is documented user API
- [x] `_runs.py` → `runs.py` — owns the on-disk manifest format read back by
      `ps`/`kill` across versions
- [x] write real contract docstrings on each (this is the "clear semantics" half)
- [x] ~~stays private: `_excludes.py`, `_logcapture.py`, `_selector.py`~~
      **Reversed in round 2** — all three are imported across a module line
      by a public module, so they were promoted to `excludes.py`,
      `logcapture.py`, `selector.py`. The original call was wrong because
      the audit behind it only detected private *names*, not private
      *modules*.

### Phase 3 — `app.py` backend-facing contract
- [x] `_repo_root_for` → `repo_root_for`
- [x] `App._repo_root` → `App.repo_root` (read by 4 modules)
- [x] `App._entrypoint` → `App.entrypoint`
- [x] `_VALID_BACKENDS` — fold into `registry.names()`, single source of truth

### Verification
- [x] `./format.sh`, `./lint.sh`, `./test.sh` all green
- [x] assert no test patches a name that no longer exists (a rename that leaves
      a dead patch target is the exact failure this plan is designed to avoid)
- [x] version bump + PR + deploy

**Not in this PR (as planned):** the 3 correctness-sweep findings
(`sys.exit(main())`, bootstrap `sys.path`, env-key validation).

**Superseded — read this instead.** `sys.exit(main())` *did* ship, in round 2:
this PR newly advertised `python -m runplz.cli` in the README while that path
silently discarded exit codes, so promoting it and leaving it broken was not a
defensible split. Rounds 2-5 added further behavior changes — `SSH_OPTS`
list->tuple, `App.repo_root` validation and precedence, `App._dispatch`
preconditions, brev's deprecating `__getattr__`, the CLI no longer overriding a
standing `repo_root`. The "mechanical rename only" framing stopped being
accurate after round 2; this is a rename **plus** the API hardening the rename
exposed. Still deferred: the bootstrap `sys.path` divergence and env-key
validation.

### Review

**Result: zero cross-module private references remain.** Before this change
`brev.py` imported 11 underscore names from `ssh_common`, and `cli.py`
reached into `app.py` for four more. An AST sweep now finds none.

- 11 dispatch stages promoted to public and declared in `__all__`
- `_cli` → `cli`, `_runs` → `runs`, `_bootstrap` → `bootstrap`
- `_VALID_BACKENDS` and `_PS_BACKENDS` deleted — they were pure aliases of
  `registry.names()` / `ps_names()`, so the CLI now reads the registry
  directly and there is nothing left to drift
- `App._repo_root`/`._entrypoint` → `App.repo_root`/`.entrypoint`;
  `_repo_root_for` → `repo_root_for`
- `__all__` added to `runs`, `cli`, `app`, `config`, `image`; ssh_common's
  expanded from 56 to 87 entries (31 public names were undeclared)
- README gained a "Public API" section; AGENTS.md scope list updated

**Underscores that stayed, and why.** `_excludes`, `_logcapture`,
`_selector` are genuinely file-local (**superseded — see round 2, all three
were promoted**). `_bootstrap`, `_cli`, `_ssh_common`
are compat entry points, documented as such in their own docstrings.

**The one thing that changed the plan.** `_bootstrap` looked like the
clearest rename candidate — until it turned out runplz does not ship
itself to the remote, so the container's runplz version is independent of
the orchestrator's. The invoked module path is therefore a cross-version
wire format: emitting `runplz.bootstrap` would break a new orchestrator
against a container pinned to an older runplz. The implementation moved to
the public `bootstrap.py`, but backends still emit the legacy path, and
`test_emitted_bootstrap_path_is_the_legacy_one` fails if someone
"finishes" the rename without a deprecation window.

**Policy reversal, recorded deliberately.** `test_dispatch_internals_stay_private`
asserted the opposite of this change — that the pipeline stages must stay
out of `__all__`. It was replaced by `test_dispatch_pipeline_is_public`,
which states the reasoning: a seam that another module imports and five
test modules patch is an interface.

**Verification.** format/lint/test green, 742 passed (up 4). Beyond the
suite: a clean-venv install confirmed the new `runplz.cli:main` entry
point resolves and backend choices still come from the registry; both
`python -m runplz._bootstrap` and `python -m runplz.bootstrap` were
executed end to end; and two mutations of renamed stages
(`check_preconditions` no-op, `stream_and_wait` always 0) each failed the
suite, proving the renamed seams still intercept rather than silently
passing.

### Review round 2 (code-review findings)

The review caught a claim I had made and got wrong: I reported "zero
cross-module private references" after the first pass, but my audit only
detected private *names*, not private *modules*. In
`from runplz._excludes import DEFAULT_TRANSFER_EXCLUDES` the name is public
and the module is not, so five such imports were live while the check
reported clean. `test_nothing_imports_a_private_name_or_module_across_a_module_line`
now walks the AST and checks both halves; the corrected audit reports zero
for real, with three underscore modules left, all documented compat shims.

Fixed:

- `runplz/cli.py` discarded `main()`'s return code while the shim this PR
  added propagated it — and this PR newly advertised `python -m runplz.cli`
  in the README. All three entry points now agree (verified: exit 1).
- promoted `_excludes`/`_selector`/`_logcapture`; each was imported across
  a module line by a public module
- `provisioning.__all__` omitted five public constants, including
  `ALREADY_EXISTS`, which `gcp.py` imports — a name the README's own rule
  called droppable in a patch release
- the `__all__` drift guard covered only `ssh_common`; now parametrized
  over all 13 public modules, and it immediately found three more gaps
  (`selector.MachineChoice`, `DEFAULT_COST_TOLERANCE`,
  `logcapture.default_log_path`). Also handles `AnnAssign`.
- the wire-format guard omitted `modal.py`, a third emit site, and its
  `>= 3` total was satisfiable by a docstring and a `pkill` string; now
  asserts per-file
- `SSH_OPTS` was a mutable list in `__all__` -> tuple
- `dispatch_to_target` dereferenced `app.repo_root` unguarded; now raises
  the same actionable error `local.py`/`modal.py` do
- `bind()` silently overwrote a caller-set `repo_root`, newly invited by
  making the attribute public; now only infers when unset
- deleted brev's 24 dead re-exports and repointed 52 test call sites at
  `ssh_common`. This also retired an overstated justification: the policy
  test had cited "brev has to re-export all eleven" as evidence the stages
  are API. brev called none of them.
- the registry test spawned an interpreter to read one argparse message;
  now in-process

Both new guards were mutation-tested: switching modal.py to the new
bootstrap path, and reintroducing a private-module import, each fail.

### Review round 3

Two findings were regressions I introduced in round 2:

- `bind()`'s `elif self.repo_root is None` made repo_root **sticky across
  binds** — `bind(repo_root=X)` then `bind("local")` kept X. Verified by
  running it. `repo_root` is now a property: assignment coerces to an
  absolute Path and records caller intent; a `bind(repo_root=...)` argument
  is per-call and sets the value directly, so it no longer leaks. Four
  behaviors pinned by tests.
- The `repo is None` guard I added to `dispatch_to_target` fires **after**
  provisioning — gcp/aws/brev would create a box and wait out the ssh
  timeout first. Moved to `App._dispatch`, above every backend, after
  `registry.load()` so "unknown backend" stays the more specific error.

Also fixed:

- both legacy shims bound `main` by value, so patching `runplz.cli.main`
  did not reach `runplz._cli.main` — the exact aliasing trap this PR's own
  policy warns about. Both now forward via module `__getattr__`, matching
  `backends/_ssh_common.py`, and both `raise SystemExit(main())`.
- brev's deleted names were plain public names on a public module since
  3.5. Deleting them outright would ImportError on a minor bump, so brev
  now forwards them to ssh_common with a DeprecationWarning (drop in 4.0).
  `brev.__all__` stays `["run"]`.
- the wire-format guard's `>= 3` floor was satisfiable by a docstring and a
  `pkill` string; it now counts `python -m runplz._bootstrap` invocations
  exactly. Mutation-tested: removing one real emit now fails.
- the private-reference guard skipped every relative import
  (`from . import _x` has `module=None`); it now resolves `node.level`.
- the backend-choices test only asserted registry ⊆ choices, losing the
  deleted constant's two-directional guarantee; it now compares the
  parser's actual `choices` for both `run` and `ps`.
- README's Public API table was missing seven of the thirteen public
  modules, so the promotions were invisible to their audience.
  `test_readme_documents_every_public_module` pins the table to
  `PUBLIC_MODULES` — it found three of those gaps itself.
- orphan comment describing the deleted `_PS_BACKENDS`; SSH_OPTS tuple
  rationale recorded at the definition.

Filed rather than fixed: #87 (two `@local_entrypoint` decorators silently
last-wins). Real, but a behavior change, and this PR is a rename.

### Review round 4

`repo_root` broke a third time, in a new way: `app.repo_root = X` then
`bind(repo_root=Y)` then `bind()` yielded Y — the per-call argument both
destroyed the standing assignment and leaked forward, violating both
invariants the round-3 comments claimed. Each earlier fix passed its own
single-branch test, so the model was replaced rather than patched again:
two fields (`_repo_root` effective, `_repo_root_assigned` standing), and
bind() recomputes from scratch in precedence order every call, so no branch
can leave a stale value. The whole matrix is now one test.

- the public setter validated nothing: `repo_root = ""` resolved to the
  process CWD, so rsync_up would stage whatever directory the caller
  happened to be in — a home dir, or `/` — onto a remote box. Empty,
  whitespace, nonexistent and non-directory values now raise, matching the
  validation `bind()` already did for `outputs_dir`.
- `App._entrypoint` -> `App.entrypoint` had no alias, and the failure was
  *silent*: `app._entrypoint = driver` left `entrypoint` unset, so the CLI
  synthesized a default from the lone @app.function and dispatched a
  different job. Deprecating property alias added — a silent wrong-job run
  is worse than the ImportError the brev shim exists to prevent.
- the brev forwarder had zero coverage; a typo in `_MOVED_TO_SSH_COMMON`
  would have shipped green. Now asserted name-by-name against ssh_common,
  including the warning and `dir()`.
- brev's `__getattr__` had no `__dir__`, so the 24 compat names resolved
  but were invisible to introspection.
- the private-import guard's allowlist exempted nothing today but would
  have exempted the three highest-risk files forever. Removed — the shims
  alias public modules and pass on their own merits.
- `text.count("-m", 0) and ...` in the wire guard read as two checks and
  was one. Deleted.
- **tasks/todo.md contradicted the diff**: a checked-off "stays private:
  _excludes/_logcapture/_selector" survived round 2's reversal. Marked.
- README's `runplz.app` row omitted two `__all__` names; a new test pins
  rows against each module's exports.
- the removal of `_excludes`/`_selector`/`_logcapture` (no shim, unlike
  `_cli`/`_bootstrap`) is now pinned by a test so the asymmetry reads as a
  decision: those were never an invocation path.

809 passed.

### Review round 5

**Process failure worth recording.** Two "fixed" claims in round 4 were
false: the dead `text.count("-m", 0)` expression was never deleted (the
formatter had reflowed my anchor line, so `str.replace` matched nothing and
I never re-grepped), and the todo.md scope note was edited with an anchor
that likewise did not match. Both were then written up as done. Every fix
this round was verified by a re-grep/execution pass afterwards, and that
pass caught nothing outstanding.

Round-5 findings, all confirmed by running them first:

- **the CLI discarded a standing `app.repo_root`.** It passed
  `repo_root=repo_root_for(script_path)` unconditionally, and a bind
  argument outranks a standing assignment — so the override this PR
  documented was a no-op on the path almost everyone uses. Now passed only
  as a fallback. Mutation-tested.
- `repo_root` was validated for existence but not for *containing the
  function's script*, so `relative_to()` still raised inside
  `dispatch_to_target` — after the box was created, ssh waited out and the
  tree rsynced. Checked in `_dispatch` now, before any provisioning.
- `_coerce_repo_root`'s empty check was `isinstance(value, str)`, so a
  PathLike bypassed it. Uses `os.fspath` now. Documented limit: `Path("")`
  is already `Path(".")` at construction and cannot be distinguished from a
  deliberate `"."`.
- `App._repo_root` got no alias while `App._entrypoint` did, so the old
  spelling still wrote the field and skipped all the new validation.
- **brev's compat set was exactly inverted.** It forwarded the eleven
  *public* stage names, which never existed on brev, and dropped the eleven
  *underscore* spellings that did — verified against
  `git show origin/main:runplz/backends/brev.py`. Both spellings now map to
  ssh_common.
- both DeprecationWarnings were invisible where they matter: the CLI
  executes user scripts as `_runplz_user_job`, and Python only surfaces
  DeprecationWarning in `__main__`. The CLI now enables them for runplz and
  for the loaded script; verified the warning appears against the user's
  own line number.
- `test_readme_lists_every_exported_name` ended in `or len(row) > 40`,
  true of every row — it asserted nothing. Replaced with a check that no
  row names a symbol absent from every public `__all__`; mutation-tested.
- `PUBLIC_MODULES` omitted the six backend drivers even though
  `registry.load()` calls `run`/`list_jobs` across a module line. All six
  now declare `__all__` (brev's `["run"]` had omitted `list_jobs`) and have
  a README row.
- the private-import guard derived a module's own name from `path.stem`
  without its package, and had no non-vacuity assertion — the test it
  replaced asserted the directory existed for exactly that reason.
- the `repo_root is None` invariant existed in four places with three
  wordings; now one `App.require_repo_root()` with four call sites.
- `bind()`'s "declare a function so we can locate the repo root" fired even
  when a repo_root was handed in, where the reason is false.

Not addressed: the three legacy shims still implement attribute forwarding
with two idioms. They behave identically and are pinned by tests; a shared
helper would need its own public home for an eight-line idiom.

843 passed.

---

## Fail before you pay: validate the job spec locally (3.21.0)

Closes #88 and #87. Both are the same defect shape and it is the shape that
costs money: a mistake that is fully checkable on the laptop, but which
currently surfaces only after a box has been provisioned — or not at all.

- [x] **#88 env keys.** Rendered as `export KEY=<quoted value>`. Values are
      quoted; keys cannot be, because a variable name is not a word quoting
      applies to. A non-identifier key produces shell that *parses* — so
      `sh -n` never catches it — and fails at runtime. The remote runs under
      `set -euo pipefail`, so it aborts the job after provisioning and rsync,
      with an error naming neither runplz nor the key. Validated in
      `_normalize_env`, alongside the existing `_normalize_preconditions`.
- [x] **#87 duplicate `@app.local_entrypoint`.** Was last-wins, so the first
      driver became unreachable and nothing said so. Now an error, matching
      every comparable ambiguity (multiple `App`s, multiple `@app.function`
      with no entrypoint).
- [x] README documents both rules
- [x] version bump, PR, deploy

### Verification

- both mutation-tested: reverting either validation fails the new tests
- `test_every_accepted_env_key_survives_a_real_shell` renders the same
  `export` line the backends emit and runs it under `bash -euo pipefail`,
  so the regex and bash cannot silently diverge on what an identifier is
- end-to-end through the real console script: both errors exit 1 and the
  traceback points at the user's own decorator line

Left alone: the CLI shows a 23-line traceback for these, because they are
raised while executing the user's script. Suppressing that would hide
genuine user-code errors, and the user's own line is already in the trace.

---

## Job script import semantics (3.22.0)

Closes #89.

**The issue's premise was wrong and I corrected it publicly first.** I had
filed it as "`python jobs/train.py` works, dispatch doesn't". Measured, the
two are exact opposites:

|  | `sys.path[0]` | sibling import | repo-root import |
|---|---|---|---|
| `python jobs/train.py` | script's directory | works | **fails** |
| runplz bootstrap | `''` (CWD = repo root) | **fails** | works |

My original check put the imports inside the function body, so running the
script never executed them — "no output" meant "never ran".

That inverted the fix. The issue's own suggested option 1 (insert the script
dir at `sys.path[0]`, matching Python) would have **broken every job that
imports from the repo root today** — the working, staged-by-design behavior.

- [x] **Append**, don't insert. A strict superset: nothing that resolves
      today changes, siblings newly resolve, and landing after the stdlib
      means `jobs/types.py` cannot shadow `types` for the run. Plain Python
      would let it. Cost: a sibling named after a stdlib module stays
      unimportable — a deliberate trade.
- [x] `bootstrap`'s contract docstring rewritten; it documented the old
      behavior
- [x] README gains an "Imports" section stating the order and both
      consequences
- [x] version bump, PR, deploy

### Verification

Mutation-tested twice, and the second mutation exposed a weak test of mine:

- removing the fix fails the sibling test
- switching `append` -> `insert(0)` (plain-Python order) fails the ordering
  test **and** the shadowing test

The shadowing test originally used `types`, which is already in
`sys.modules` when the job body runs — so it resolved from cache regardless
of path order and passed under the `insert(0)` mutation. Rewritten to use
`colorsys`, which is not preloaded, with an in-test assertion that it is not
preloaded so the test cannot silently rot back into vacuity.

---

## Two new test tiers (3.23.0)

The suite was 11.6k lines of test against 9.4k of source, 427 `mock.patch`
sites, 138 patched `subprocess` calls — and 4 places that executed anything
real. It verified that runplz *builds* the strings its author expected, so
it could only ever be as correct as the author's model of the remote. Every
expensive bug this session lived in that gap.

- [x] **Cloud lifecycle against stub CLIs** (`test_e2e_fake_cloud.py`).
      Stub `gcloud`/`aws` executables (`fake_cloud.py`) on `PATH`: real
      subprocess, real argv, real JSON, real exit codes, scripted transient
      failures so the retry loop is *exercised* rather than described.
      11 tests covering create argv, all three `on_finish` modes,
      teardown-after-dispatch-failure, retry vs non-retry classification,
      and instance-id round-tripping.
- [x] **End-to-end over real ssh** (`test_e2e_localhost.py`). The tier
      starts its own unprivileged `sshd` on loopback (`sshd_harness.py`),
      so it needs no Remote Login, no CI-only service and no config —
      it runs everywhere and is not skipped by default.
- [x] The billing guard got *stronger*, not weaker: a billed name is
      allowed only when it resolves inside a directory the test created
      (`sandbox_bin`). A real `gcloud` stays blocked while a stub is
      installed, and a test that forgets to install its stub still trips it.
- [x] `FakeClock` lifted out of `test_transport_retries.py` into
      `tests/clock.py`; the new `fast_clock` fixture patches the `time`
      reference in both sleeping modules. Cloud tier: 128s -> 4s.

### It found a real bug immediately

`nohup` on macOS cannot detach in a non-interactive ssh session
("can't detach from console: Inappropriate ioctl for device"), and macOS
has no `setsid`, so **detached launch does not work against a macOS
remote** — filed as #92. Linux remotes, which is every documented target
and CI, are unaffected. The two detached tests skip when the remote reports
Darwin, linking the issue, so they exercise production behavior in CI
rather than reporting a platform artifact as a runplz failure.

### Verification

Both tiers mutation-tested rather than assumed:

- dropping `.env` from the secret-exclude list fails the rsync test — that
  exclude list is a security control and was previously only ever checked
  by asserting on rsync's argv, which cannot tell you whether rsync obeyed
- breaking `rsync_down` fails the outputs test
- a test asserting the harness is a live connection guards the fixture, so
  a broken sshd cannot produce a green run

882 passed, 3 skipped, 32s.

---

## 2026-08-31 PR #94 follow-up — capability selection + executable contracts

Branch: `catalogue-and-container-tier` (off `main` @ `2414202`)

### Acceptance contract

- [ ] A cloud-selected machine either satisfies every declared CPU, RAM, GPU model and GPU-count
      minimum or fails locally before invoking a billed CLI. No selector silently clamps an
      unsatisfiable request to its largest known shape.
- [ ] AWS and GCP resolve through one capability-aware selector and one offering model. Tests are
      generated from that model and call the production resolver; they do not rebuild production
      names or enumerate hand-picked boundary inputs.
- [ ] Every AWS-emitted instance type is checked against botocore's offline EC2 catalogue. The
      catalogue is required in the dev suite and cannot silently disappear behind a module skip.
- [ ] Fake cloud executables reject missing, duplicate and unknown arguments, derive responses from
      inputs, and enforce lifecycle state. One schema/state-machine implementation backs both
      providers; per-provider data contains only vocabulary and response shapes.
- [ ] At least one AWS and one GCP lifecycle reaches the real SSH endpoint supplied by the shared
      harness, proving provider output -> target/user/options -> SSH readiness -> teardown.
- [ ] The live precondition test sends a real requirement and proves the remote probe/parser ran.
      Cloud assertions cannot pass vacuously when an expected command was never called.
- [ ] Live SSH uses an isolated temporary known-hosts file; the tier does not mutate `~/.ssh`.
      Explicit Docker mode fails rather than skips, invalid backend modes are rejected, and CI runs
      the container tier once.
- [ ] Coverage measures branches and has a ratcheted floor. Remaining exclusions are limited to
      compatibility entrypoints or genuinely platform-only defensive paths and are documented.

### Implementation plan

- [x] Resolve resource under-provisioning and false-positive integration coverage in this branch.
- [x] Introduce immutable machine offerings with CPU/RAM/GPU capabilities and a shared selector.
- [x] Migrate AWS/GCP CPU, bundled-GPU and attachable-GPU resolution to the shared selector.
- [x] Replace catalogue spot cases with registry-generated existence/capacity/monotonicity tests.
- [x] Replace permissive fake routes with strict declarative command schemas and duplicate/unknown argument checks.
- [x] Add a narrow shared lifecycle dispatch seam and drive both providers into a real sshd.
- [x] Repair vacuous assertions and isolate SSH host-key state.
- [x] Enable branch coverage/fail-under and make the explicit container tier fail on setup failure.
- [x] Mutation-probe invalid family names, undersized fallback, missing provider handoff, missing
      precondition probe, and skipped container execution.
- [ ] Run `./format.sh`, `./lint.sh`, `./test.sh`; inspect branch coverage and CI.

### Review

- Pending.
## 2026-08-31 PR #94 follow-up — test contracts and resource selection

- [x] Add capability-aware shape selection; reject requests larger than known offerings
- [x] Drive catalogue tests through public resolvers and assert resource minima
- [x] Tighten fake cloud executable validation and connect provisioning to live SSH
- [x] Make the live precondition test actually probe and isolate SSH known-hosts state
- [x] Make optional test dependencies and CI skip behavior fail loudly
- [x] Bump version and run format/lint/test/branch-coverage checks

### Review

Implemented; verification is recorded in the final review below.
# Coverage follow-up — shared runtime failure paths

- [x] Inventory uncovered `ssh_common`/CLI branches and map each to public behavior.
- [x] Add behavior-driven tests for reusable SSH wait/command and staging failure semantics.
- [x] Run format, lint, and full test/coverage gates; record measured result (1025 passed, 3 skips, 93.92% branch coverage).
- [ ] Open, review, merge, and deploy a versioned PR if the change is validated.
# Issue #122 — Detached inactivity watchdog

- [x] Add opt-in `max_inactivity_seconds` and `inactivity_action` to SSH/Brev
      configuration with validation; default disabled. *(landed earlier in
      #124 — the fields and validation existed, nothing read them.)*
- [x] Thread the options through dispatch and detached/container monitoring.
- [x] Use heartbeat/progress timestamps (not stdout silence alone), bounded
      diagnostics, and exact-run termination when requested.
- [x] Add tests for healthy silence, diagnose, terminate, reconnect, and output
      preservation; verify event and command observation.
- [x] Run format, lint, full tests (1197 passing, 95.30%).
- [ ] Review, merge, and deploy as its own PR.

### Notes from implementing it

**The heartbeat is not a progress signal.** `runplz_heartbeat_loop` runs on a
timer as a background job of the wrapper shell, independent of the user's
command, so it keeps ticking while the job is wedged. It proves the process
exists — which is exactly the signal that failed in the reported incident. The
watchdog reads the driver log and the outputs directory instead, both of which
move only when the application moves.

**Where the wake-up came from.** `tail_and_wait_for_detached` blocks in
`tail -F` with `timeout=None` unless a runtime cap is set, so there was no
moment at which silence could be noticed. The watchdog bounds that timeout;
a `TimeoutExpired` is then either the cap or a watchdog tick, told apart by
whether the runtime budget is actually spent.

**The reconnect trap.** `reconnects` is cumulative for the run and capped at
20, so a tick that fell through to `reconnects += 1` would burn the budget on
any legitimately quiet job and silently drop its live log stream. Ticks
`continue` before that line. Mutation-checked: removing the `continue` fails
three tests.

**Terminate cannot raise.** `rsync_down` sits inside the `try` and before the
`finally` in `dispatch_to_target`, so `raise_for_runtime_cap` skips the
outputs sync entirely — a capped run loses what it produced. The watchdog
stops the run and returns normally instead, leaving the completion path and
the sync intact. Worth filing that the runtime cap has this bug.

**Free diagnostics.** `build_kill_command` already emits a bounded SUMMARY /
HEARTBEAT / LOGTAIL block including `gpu_mem_used`, which
`raise_for_runtime_cap` captures and discards. Terminate mode prints it.
# PR 0 — Test-fidelity audit
- [ ] Add shared command-observation assertions to every subprocess fake.
- [ ] Add mutation probes for each failure scenario and document exercised runtime layers.
- [ ] Distinguish sandbox-unavailable skips from passing integration tests.
- [ ] Run format, lint, tests, review, merge, and deploy.
# PR E — Coverage ratchet
- [ ] Raise the coverage floor to 95%+ and verify CI across supported Python versions.
- [ ] Add/retain parameterized lifecycle coverage where shared logic has branches.
- [ ] Run format, lint, tests, review, merge, and deploy.
# PR 0 follow-up — runtime-layer fidelity documentation
- [ ] Document test harness layers and mutation-probe interpretation.
- [ ] Run format, lint, tests, review, merge, and deploy.

## 2026-09-04 — Truthful remote lifecycle status

### Problem statement

`runplz status` currently renders the literal tail of `events.ndjson`. Output salvage always
appends `rsync_down_start` after the run's outcome, so the user cannot see
`killed_by_runtime_cap`, `remote_command_stalled`, or even a normal `remote_command_exit`.
The sync phase also has no completion/failure event, making a finished download
indistinguishable from a hung one. Several earlier failure paths likewise leave a progress event
at the tail or assert a cleanup action before it has been observed.

### Behavioral contract

- [x] `status` reports the latest non-output-sync lifecycle event as `last event`, preserving
      already-shipped terminal events even though salvage runs later, and reports the latest
      output-sync event separately.
- [x] `rsync_down` records start, done, and failed outcomes. Failure recording is best-effort and
      never replaces the original transfer error.
- [x] A hard precondition failure records `precondition_failed` and enters the existing salvage
      path so the remote manifest/event stream is copied locally and `status` can resolve it.
- [x] A remote image-build failure records `build_image_failed`; a `docker run -d` failure records
      `container_launch_failed`. Both preserve the original exception.
- [x] An orchestrator termination signal reaching an active dispatch records
      `orchestrator_signalled` after container cleanup, so a monitor-written SIGKILL
      `remote_command_exit` cannot misreport the causal outcome.
- [x] Watchdog termination records `action="terminate"` only after the cleanup summary confirms a
      signal was sent and nothing survived; no-op, failed, and incomplete cleanup get truthful
      action values in `remote_command_stalled`.
- [x] Runtime-cap cleanup records `killed_by_runtime_cap` only when the cleanup summary confirms
      something was stopped; a last-second natural exit does not get a false kill event.
- [x] Regression tests drive production ordering and failure edges, including malformed/missing
      cleanup summaries, without weakening existing success/failure semantics.
- [x] Bump the patch version, run `./format.sh`, `./lint.sh`, and `./test.sh`, and review the diff.
- [x] Push and open PR #164, closing issue #163.

### Implementation notes

Keep event writes best-effort: lifecycle reporting must not mask the operational exception it is
describing. Do not require `jq` or another remote dependency for `status`; use the existing
newline-delimited JSON shape and POSIX tools already assumed by the CLI. Keep the scope in the
backend-agnostic SSH dispatch/event machinery and CLI status rendering.

### Review

Implemented causal status selection without adding a remote dependency: one probe retrieves the
small non-heartbeat event stream, then Python separates output-sync events and applies typed event
semantics. Operator/runtime control events take precedence over the secondary
`remote_command_exit` they can cause only when the event's measured outcome warrants it. Phase
failures preserve their original exceptions, transport ambiguity is named `*_unconfirmed`, and
signal exceptions pass through best-effort reporting instead of being swallowed.

Verification: `./format.sh`, `./lint.sh`, and `./test.sh` pass; 1278 tests passed, 1 platform test
skipped, and total coverage is 95.07%.

### Review follow-up — durable, evidence-based outcomes across backends

- [x] Record `orchestrator_signalled` before salvage so Brev, GCP, and AWS deletion cannot erase
      the only copy; retain causal precedence over a later cleanup-induced remote exit.
- [x] Give the direct SSH backend the same signal translation used by provisioned backends, while
      keeping user-owned-host teardown semantics unchanged.
- [x] Treat every `runtime_cap_reached` result as causal: confirmed no-op, failed stop, and
      unconfirmed cleanup all still explain why orchestration ended.
- [x] Make user kill events carry measured completion state; emit a distinct attempt event for
      survivors and never let legacy/unconfirmed kills permanently override a later natural exit.
- [x] Render start-only output-sync history as `completion unknown`, including pre-4.4.3 streams
      and interrupted transfers.
- [x] Add regression coverage for persistent and ephemeral dispatch, all four SSH-derived
      backends, current/legacy kill records, cap cleanup outcomes, and sync history.
- [x] Re-run `./format.sh`, `./lint.sh`, and `./test.sh`; update PR #164 and confirm CI.

### Review follow-up results

Signal events are now appended before failure salvage, so the final transfer persists them before
an ephemeral host is deleted; direct SSH installs the same signal translator without adding host
teardown. Runtime caps remain causal even when cleanup is unconfirmed, while watchdog and user
kills override later exits only with explicit no-survivor evidence. A failed `--no-escalate` kill
is named `kill_attempted_by_user`, and old fieldless kill/stall records yield to a later exit.
Start-only sync records now say `started (completion unknown)`.

Verification: `./format.sh`, `./lint.sh`, and `./test.sh` pass; 1290 tests passed, 1 platform test
skipped, and total coverage is 95.14%. PR #164 is updated and its lint, Python 3.10–3.12, and
container end-to-end checks all pass.

### Second review follow-up — cancellation and locally durable sync outcomes

#### Specification

- [x] Make `OrchestratorKilled` bypass every ordinary `except Exception` boundary by construction,
      while retaining the explicit dispatch catch that records the signal and the outer lifecycle
      `finally` that tears provisioned hosts down.
- [x] Record `rsync_down_done` and `rsync_down_failed` directly in the local downloaded metadata as
      well as remotely; local reporting is best-effort and must never replace a transfer error.
- [x] Treat non-string `event` fields as malformed/non-authoritative without performing a set
      lookup, so arrays, objects, and other externally edited values cannot crash status.
- [x] Add regressions through the existing broad catches, successful and failed transfer paths,
      ephemeral-survival semantics, and malformed event objects.
- [x] Run `./format.sh`, `./lint.sh`, and `./test.sh`; push PR #164 and confirm every CI job.

#### Review

`OrchestratorKilled` now inherits directly from `BaseException`, so routine best-effort
`except Exception` blocks cannot turn operator cancellation into continued staging or launch.
Successful and failed output transfers append their terminal outcome to the local surviving event
stream as well as the remote live stream; a local reporting error remains a warning and cannot
replace the transfer exception. Status validates event-name types before selection and rendering,
so externally edited arrays/objects remain reportable rather than crashing set lookup.

Local verification: `./format.sh`, `./lint.sh`, and `./test.sh` pass; 1293 tests passed, 1 platform
test skipped, and total coverage is 95.14%.

## Lifecycle reliability follow-up (4.4.4)

### Specification

Keep lifecycle evidence conservative and durable across the shared SSH dispatch used by SSH,
Brev, GCP, and AWS. No backend-specific orchestration rewrite or cloud resources are needed.

- [x] Distinguish unknown Docker state from a confirmed stop, count only delivered signals, and
      make CLI/cap/watchdog consumers reject unconfirmed kill summaries.
- [x] Preserve cancellation during every cleanup stage without skipping later salvage/removal;
      persist the signal locally even when the final transfer has already happened.
- [x] Put real subprocess deadlines on event writes and failure salvage, plus an idle timeout on
      ordinary downloads. Successful large transfers must not inherit a short total deadline.
- [x] Fall back to matching local metadata when the bounded remote status probe fails; explicitly
      label it as a snapshot and never use it for a different host/run override.
- [x] Record start/done/failed/unconfirmed environment-setup outcomes in native/container modes.
- [x] Tolerate non-string and invalid-calendar timestamps without hiding the rest of status.
- [x] Add regression tests for each failure, file/link a tracking issue, and bump the version.
- [x] Run format, lint, and the complete test suite; review the diff and prepare the PR.

### Review

Tracked by https://github.com/pirl-unc/runplz/issues/166. All six fixes share the existing SSH
dispatch/CLI machinery; no per-provider orchestration fork was added. Stop evidence is tri-state,
signal delivery is measured, cleanup defers operator signals until its bounded steps finish, and
late cancellation is written to the surviving local stream. Status labels matching offline evidence
as a snapshot. Native/container setup distinguishes a failure from transport uncertainty.

Verification: `./format.sh`, `./lint.sh`, and `./test.sh` pass: 1368 passed, 1 platform test skipped,
95.49% coverage. Regressions execute the kill shell under sh/bash, exercise actual signal handlers,
verify timeout enforcement, and cover shared SSH/Brev/GCP/AWS cleanup paths. The full suite also
exercises local loopback SSH and stub cloud CLIs; no paid cloud workloads were launched.

PR #167 is open. This request ends at an open PR; no merge or deployment is authorized for
this follow-up yet.

### PR #167 review follow-up: match the snapshot's SSH endpoint

The hostname and run path are insufficient when forwarded ports select different machines.
Reuse the status probe's effective SSH options and compare its port with the locally recorded
`ssh.json` port before accepting an offline snapshot. An unspecified port remains unspecified:
SSH config may choose a nonstandard port, so do not equate it with an explicit port 22.
Credential-only overrides must not invalidate an otherwise matching endpoint.

- [x] Add CLI regressions for different, equal, inherited, and unspecified SSH ports.
- [x] Require a matching recorded port before returning a successful offline status snapshot.
- [x] Record the review lesson and tracking issue; keep the PR's existing 4.4.4 version bump.
- [x] Run `./format.sh`, `./lint.sh`, and `./test.sh`; prepare the update for PR #167.

Review results: tracked by https://github.com/pirl-unc/runplz/issues/168. The new CLI tests
reproduced six unsafe fallback cases before the fix; all 16 port-matching cases now pass,
including both manifest-selected and explicitly selected run IDs. A credential-only override
still permits the same snapshot. Formatting and lint pass; the full suite passes with 1384 tests,
1 platform test skipped, and 95.49% coverage. The PR update and CI results are tracked on GitHub.
