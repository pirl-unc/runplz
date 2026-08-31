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
