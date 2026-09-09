# Test fidelity map

This suite deliberately mixes runtime layers. A test is only evidence for the
layer named here; a unit mock is not evidence that a vendor CLI accepted the
same command.

| Harness / tests | Runtime layer | Failure signal | Mutation it catches |
| --- | --- | --- | --- |
| `test_ssh_faults.py` | real `ssh` client + disposable `sshd` | refused socket, timeout, remote exit, dropped transport | removing exception/return-code handling |
| `test_rsync_faults.py` | real `rsync` subprocess + temp filesystem | nonzero transfer / partial marker | ignoring rsync status or trusting a complete file |
| `test_cloud_consistency.py`, `test_e2e_fake_cloud.py` | real Python subprocess invoking stateful fake `aws`/`gcloud` | input-derived JSON, state transitions, observed argv | changing options, retry count, or idempotency semantics |
| `test_e2e_localhost.py` | real detached SSH/container path when available | remote lifecycle and reconnect behavior | bypassing the detached monitor |
| backend unit tests | mocked provider/SSH calls | deterministic parser and branch contracts | parser/branch regressions only; not CLI fidelity |

The `live_ssh` marker is what makes the top row of that table true, not
decoration. It is a *permission*: without it the billing guard intercepts every
`ssh` call and raises, and a test asserting `pytest.raises(Exception)` passes on
the guard's own error without ever reaching the daemon. That is how
`test_ssh_faults.py` spent four tests proving nothing (#96). Assertions in that
tier must be specific enough that the guard cannot stand in for the failure
under test -- a remote exit code, a 255 transport failure, a message only the
real code path produces.

The same default-deny rule covers Modal through `live_modal`, over both the CLI
and the SDK.

The command guard does not try to work out which token is *the* program. It
scans every token in argv and refuses any whose basename is a billed CLI, or
that names `modal`/`modal.*`/`runplz.backends.modal_runs` after a `-m`-family
flag (`-m`, `-mmodal`, and clusters like `-um`). That is why it needs no table
of wrappers: `uv run modal`, `timeout 600 modal`, `nohup modal` and `env FOO=1
modal` are all caught by the same rule, and a wrapper nobody has thought of yet
is caught too. Matching is strict basename equality, so `gcloud compute
config-ssh`, `aws ssm --name /aws/service/...` and `rsync --exclude=.ssh` stay
runnable. All five process-starting entry points are guarded — `run`, `Popen`,
`call`, `check_call`, `check_output` — and options are read from Popen's
signature, so they are seen whether passed positionally or by keyword.

It fails closed on anything it cannot read: `shell=True`, a string command
`shlex` cannot split, arguments that do not bind to `Popen`, and `env`'s own
options (`-S` re-tokenizes its argument into a command that is never visible).
Use explicit argv or a direct mock in those tests.

The SDK guard sits at the shared channel boundary, so it covers Function,
Cls/Obj, autoscaler, warm-container, App, Sandbox, read-only, synchronous,
asynchronous, and future SDK operations without maintaining a public-method
inventory. It is a hard failure if that boundary cannot be found: `modal` is a
runtime dependency, so a missing `modal.client._Client` means the SDK moved it,
and continuing would leave the suite unguarded and still green. The boundary is
per-process, which is why the worker child (`python -m
runplz.backends.modal_runs`) is refused as a CLI launch — it is a fresh
interpreter that builds a real client where no in-process patch applies. Every
`runplz` module that imports `subprocess` must appear in `_MODULES_TO_GUARD`;
a test asserts it, because a module missing from that list bypasses the guard
entirely rather than noisily.

Offline fake clients and explicit mocks remain usable without the marker. The
marker is consulted per billed name, so a `live_ssh` test may run `rsync -e
'ssh ...'`. A test-installed CLI in `sandbox_bin` is allowed only when the token
naming it resolves into the sandbox under the effective `executable=`,
`env=`/PATH and `cwd=` context, and only when it exists and is executable; a
PATH fake cannot exempt an explicit outside path, a `python -m modal`, or a
lookup whose PATH the command itself rewrites.

Tests that need a local/container SSH service are marked as environmental
integration tests. An unavailable service produces an explicit `SKIPPED`
result with the reason; it is never counted as a passing integration assertion.
The fake cloud and rsync tests are self-contained and must not skip for missing
vendor tools because their executables are created inside `sandbox_bin`.

The fake CLI logs every invocation. Use `fake_cloud.assert_observed(...)` (and
an exact `count` for retries) so a test cannot pass after the production path
silently stops invoking the command it claims to exercise.

A fake may only produce outcomes the real call can produce. Simulating an
impossible one makes the test pass without exercising the branch it names, and
that is indistinguishable from coverage. Two that have actually bitten:

- Raising `subprocess.TimeoutExpired` regardless of the `timeout=` the fake was
  handed. `timeout=None` never expires, so a fake that raises anyway invents an
  event the branch under test can never see.
- Returning a nonzero `returncode` for a command production runs with
  `check=True`. Real `subprocess.run` raises `CalledProcessError` there, so the
  fake skips the error handling the test claims to cover.

When a fake and an assertion disagree, work out which one is lying before
changing either. Fixing the fake is usually what exposes the real branch;
loosening the assertion only hides it.
