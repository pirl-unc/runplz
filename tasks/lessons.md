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
