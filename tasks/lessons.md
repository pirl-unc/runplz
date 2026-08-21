# Project Lessons

- Shared backend behavior with a stable, independently useful contract should live in a public
  module and be tested through public names. Do not default reusable staging or process-lifecycle
  logic to underscore-prefixed modules/helpers merely because its first callers are internal.
- Treat an unreachable remote as unknown, never as failed. Unknown lifecycle state must remain in
  reconnect/runtime-cap enforcement unless the remote job is explicitly cleaned up.
- Remote process tracking must work in minimal container images. Prefer Linux `/proc` state and
  shell builtins over assuming GNU `ps`/procps is installed.
- A Git index is not a flat list of copyable files: sparse entries may be absent and submodules are
  gitlinks. Validate working-tree presence and recurse through initialized submodule Git selections
  before handing paths to rsync.
