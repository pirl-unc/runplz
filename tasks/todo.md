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
- [ ] Commit, push branch, and open a draft PR linked to `#48`, `#49`, and `#50`

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
