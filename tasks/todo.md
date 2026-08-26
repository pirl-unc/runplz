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
- brev.py deliberately NOT migrated onto the shared core — 1200 lines behind 2800 lines
  of tests, wrong risk trade in a PR adding two backends. Filed as #78.

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
