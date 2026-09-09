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
- A control-flow sentinel implemented as an ordinary `Exception` is only as reliable as the weakest
  broad catch in its call tree. Signal cancellation must bypass routine best-effort handlers by
  construction; do not depend on auditing every present and future `except Exception` boundary.
- A remote event written after a download is not part of the downloaded evidence. For terminal
  transfer outcomes, write the local surviving stream explicitly as well as the remote live stream;
  otherwise ephemeral teardown makes every successful transfer look start-only forever.
- Tolerating malformed JSON means validating field types too. Parsing an object is insufficient
  before hash/set operations: externally edited values can be arrays or objects and therefore
  unhashable.
- Cleanup has its own cancellation boundaries: a sentinel escaping one finally action must not
  skip subsequent salvage or removal. Test signals during cleanup, not only during the workload.
- Failed observation is not a negative observation. Docker inspection errors mean unknown;
  successful signal delivery and an explicit no-survivor probe are separate facts.
- A retry deadline bounds retries, not a blocked subprocess. Give best-effort cleanup operations
  their own subprocess timeouts, and distinguish transfer inactivity from total transfer duration.
- Match offline evidence against the full recorded endpoint, not just its hostname: forwarded
  SSH ports can select different machines. Use the same effective options as the live probe, and
  test overrides that change endpoint identity separately from overrides that only change credentials.
- Exercise provider result decoding, not only invented exception stubs: a provider-confirmed
  terminal result can arrive as an SDK exception. Classify it at the result boundary so ordinary
  authentication, connection, and lookup errors are still unknown, while terminated jobs can
  salvage already-committed artifacts.
- A worker's finally block cannot clean up after the parent kills that worker. Give temporary
  downloads a parent-owned, per-attempt directory and test actual process termination after a
  partial write. Atomic destination replacement protects old files, but does not by itself
  prevent temporary-file leaks.
- Test SDK integrations against both the minimum supported and currently resolved versions.
  A stub that works with an eager loader may fail when a newer SDK hydrates lazily. Model the
  real client/lookup protocol and keep hydration errors outside terminal-result classification.
- A CLI denylist does not protect an SDK-backed provider. Guard every execution boundary the
  project exposes, including synchronous and `.aio` descriptors, while leaving read-only APIs
  usable. The live marker must delegate to the captured original, and ordinary mocks must be
  able to replace the guard without reaching provider infrastructure.
- A command guard must classify the execution target, not merely hide `argv[0]` behind an opaque
  first-token helper. Normalize argv at the guarded call site and recognize supported module
  execution (`python -m provider`) explicitly. When guarding an SDK, inventory all public methods
  that can submit work or provision capacity; common-looking methods are not necessarily funnels.
- For a default-deny SDK safety guard, prefer the narrow transport/control-plane choke point over
  enumerating public methods. If the policy can require an explicit marker for live reads as well
  as writes, intercept every real RPC and let offline fake clients bypass that boundary naturally;
  public API inventories are incomplete by construction and age badly as an SDK evolves.
- A subprocess safety guard must follow execution semantics, not the visual shape of `args`.
  `shell=True` delegates to a language, `env -S` delegates to another tokenizer, `executable=`
  replaces the launched program, and `env` assignments are not shell identifiers. Fail closed on
  opaque layers; normalize explicit replacements and wrapper operands before granting any marker
  or sandbox exemption.
- A safety guard's parser must fail in the *allowing* direction never. Identifying "the
  program" in an argv required modelling `env`'s operand rules and CPython's flag arity,
  and each gap in those tables (`-um`, `modal.cli.entry_point`, `uv run modal`) silently
  permitted a billed launch. Scanning every token for a billed name is shorter, needs no
  table per wrapper, and its failure mode is a false block that a test fixes by mocking.
  Where an allowlist is unavoidable, prefer one whose entries can only *narrow* blocking
  and justify each ("`which` never execs its operand").
- Read subprocess options the way the child receives them, not the way you expect them to
  be written. `run(*popenargs, **kwargs)` forwards positionals to `Popen`, so `shell=` and
  `executable=` read from `kwargs` alone are invisible when passed positionally. Bind
  against `inspect.signature(subprocess.Popen)` instead of hand-maintaining an index table.
- An in-process patch does not survive `fork`+`exec`. Guarding an SDK at its channel
  boundary protects only this interpreter; a module that shells out to `python -m itself`
  needs the *spawn* refused as well. And check the whole class: assert the invariant that
  every module importing `subprocess` is registered, rather than adding the one that was
  missed.
- When a review reports a bypass, reproduce it before trusting the repro. One of five
  "confirmed" cases was already blocked, for an incidental reason — the hole was real but
  the given command did not demonstrate it. Load the old code in isolation with execution
  stubbed and diff old-vs-new classification; never establish a control by running the
  bypass for real, because that is exactly the billed launch under test.
- When a guard needs a special case for one wrapper, that is the signal the altitude is
  wrong, not that the case needs writing. Modelling `env` by name cost two bypasses (a
  string-form `env -S` the name check never saw, and marked tests refused because an
  *inner* program's flag looked like env's). Reading every word of every argument covers
  `env`, `sh -c`, `--opt=value` and the next wrapper with no name to miss, and deleted four
  state variables on the way.
- Exempt only what you can actually see being launched. "This token resolves into the test
  sandbox" is not a safety property when the token is an argument to an unclassified
  wrapper -- `env -u PATH modal` resolves `modal` from a PATH the guard does not control.
  Narrowing the exemption to the program position removed the entire question of whether
  our PATH reading was still valid.
- A shared mutable test double is a coupling nobody declares. One `_GuardedSubprocessModule`
  instance handed to every module meant patching *any* module's `subprocess.run` silently
  unguarded all of them -- and 51 tests had come to depend on it, patching
  `brev.subprocess.run` for a call that `provisioning` issues. Give each seam its own
  double; the tests that break are the ones that were passing for the wrong reason.
- Fixing a review finding can introduce a worse one. The one-line passthrough added to keep
  an offline worker test running turned off the guard for every backend at once. Re-review
  the fix, not just the bug -- and prefer the narrowest escape hatch the harness already
  offers over a new one.
