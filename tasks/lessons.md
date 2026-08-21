# Project Lessons

- Shared backend behavior with a stable, independently useful contract should live in a public
  module and be tested through public names. Do not default reusable staging or process-lifecycle
  logic to underscore-prefixed modules/helpers merely because its first callers are internal.
