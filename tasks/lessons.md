# Project Lessons

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
