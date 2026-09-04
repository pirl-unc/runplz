# Project Lessons

- Wall-clock timestamps serialized at second precision make boundary tests
  flaky under parallel CI; assert the bounded rounding representation rather
  than coupling correctness to scheduler timing.

- `nohup` alone does not close the fork-to-exec SIGHUP race: install the ignored disposition in
  the launching parent before spawning, and retain it in long-lived child wrappers as defense in
  depth. Test detachment with the real wrapper shape and an actual signal, not only string checks
  or a trivial execing shell.
- A module whose name is typed by anyone outside this package is public, underscore or not.
  Apply the test mechanically: is the name in `pyproject.toml` entry points, in generated shell
  we emit, in the README, or in another module's docstring? If yes it is an API and needs a
  documented contract, not a leading underscore. Shared behavior with a stable, independently
  useful contract belongs in a public module tested through public names. Do not default
  reusable staging or process-lifecycle logic to underscore-prefixed modules merely because its
  first callers are internal — and when this rule is applied, apply it to the whole repo, not
  just the subpackage that prompted it.
- Treat an unreachable remote as unknown, never as failed. Unknown lifecycle state must remain in
  reconnect/runtime-cap enforcement unless the remote job is explicitly cleaned up.
- Remote process tracking must work in minimal container images. Prefer Linux `/proc` state and
  shell builtins over assuming GNU `ps`/procps is installed.
- A Git index is not a flat list of copyable files: sparse entries may be absent and submodules are
  gitlinks. Validate working-tree presence and recurse through initialized submodule Git selections
  before handing paths to rsync.
- Do not derive a capability flag from a property that merely correlates with it today.
  "Is this backend in the default fan-out" and "does every required field have an env
  fallback" agreed for all six backends by coincidence, and inferring one from the other
  reads as a rule while being an accident. State the capability; let the correlation be
  a coincidence.
- When a refactor moves a filter, check what the old filter was *rejecting*, not just what
  it accepted. `[h for h in raw.split(",") if h.strip()]` silently encoded "blank and
  separator-only input means no targets at all"; re-expressing it as a split lost that and
  sent an empty hostname to ssh. Behaviour living in a comprehension's `if` is still
  behaviour, and usually has no test.
- Resolution and normalization of one value belong in one place. Splitting them — blank
  handling in one layer, comma-splitting in another — makes the two disagree the moment a
  value arrives from a source only one of them sees (an env var rather than a flag).
- A test that passes because unrelated code happens to fail is not passing for its stated
  reason. `runplz ps` CLI tests patched three of five fan-out backends and relied on the
  other two erroring on a dev machine with no cloud credentials. Patch the whole surface a
  test claims to control.
- A platform CI cannot reach is still testable: stub the tool that fails there. macOS
  `nohup` refuses to detach under a non-interactive ssh session, and all CI runners are
  Linux -- but the launcher is plain bash, so a `nohup` on PATH that exits the way macOS
  exits reproduces the production symptom on any runner. Reserve the real platform for
  confirming the fix, not for carrying the regression test.
- Before removing a belt-and-braces layer, find out what it is actually holding. `nohup`
  looked load-bearing for SIGHUP safety; the traps installed by #74 were, and nohup's real
  job was PID stability, which plain backgrounding also gives. Measure with a control --
  "survived the signal" means nothing unless the same test without the guard dies.
- A test that identifies a line of generated script by a keyword pins the wrong thing.
  Five tests found the spawn by grepping for `nohup`; when nohup became conditional they
  matched the probe instead. Identify it by what it does -- the script it launches.
- A fake may only produce outcomes the real thing can produce. A `subprocess.run` stub that
  raises `TimeoutExpired` regardless of the `timeout=` it was handed is describing an event
  that cannot happen — `timeout=None` never expires — so the test it supports proves nothing
  about the branch it claims to cover. When a fake and an assertion disagree, check which one
  is lying before changing either; here the fake was, and fixing it exposed a real branch.
  (Audited the other `TimeoutExpired` fakes afterwards: each raises on a call that genuinely
  carries a timeout, so they are faithful.)
- Anchor a scripted edit on text that is unique in the file, and prove the file still imports
  before moving on. A `str.replace` of `driver_log = f"{remote_run.meta_shell}/run_driver.log"`
  matched a second, unrelated site in a 3000-line module and produced a mid-function syntax
  error. Lint caught it immediately, which is the point: run the check between edits, not at
  the end of a batch, so the failure names the edit that caused it.
- Where cleanup lives decides whether evidence survives. `rsync_down` sat inside the `try`
  after the runner, so every raising path -- including the `max_runtime_seconds` kill-switch
  whose entire purpose is salvaging a wedged run -- discarded the outputs. Put collection in
  the `finally`, best-effort so it cannot replace the original exception, and keep the success
  path strict so a genuine sync failure is still an error.
- A remote lifecycle event is not durable evidence when teardown deletes the remote. Record every
  causal outcome before the final salvage transfer, then let semantic status selection handle any
  lower-level exit event appended afterward. Ordering for display and ordering for persistence are
  separate concerns.
- An attempted action is not an outcome. Only prioritize `killed`/`terminated` events when measured
  cleanup state confirms no survivors; failed and legacy-unconfirmed attempts must yield to a later
  natural exit. Likewise, a start event without a matching finish proves only "completion unknown,"
  never that the operation is still active.
