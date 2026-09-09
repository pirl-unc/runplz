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

The command guard does not try to work out which token is *the* program, and it
models no wrapper by name. It reads every word the command is built from --
including the words *inside* an argument, so `sh -c "modal run job"`,
`env -S "modal run job"` and `--split-string=modal run job` are read like any
other argv — and refuses any word whose basename is a billed CLI or whose root
package is one (`modal.cli.entry_point`, and `runplz.*`, since running this
package as a child interpreter can reach any backend). That is why `uv run
modal`, `timeout 600 modal`, `nohup modal` and every `env` spelling are caught
by one rule, and a wrapper nobody has thought of yet is caught too. Matching is
strict basename equality, so `gcloud compute config-ssh`, `aws ssm --name
/aws/service/...` and `rsync --exclude=.ssh` stay runnable, and an interpreter
flag is only read as `-m` when a module name actually follows it, so `rsync
-avzm` is not a Modal launch. All seven process-starting entry points are
guarded — `run`, `Popen`, `call`, `check_call`, `check_output`, `getoutput` and
`getstatusoutput`, the last two because they run a command line through a shell.
Options are read by binding against `Popen`'s signature, so `shell=`,
`executable=`, `env=` and `cwd=` are seen whether passed positionally or by
keyword, and `Popen(args=[...])` classifies like `Popen([...])`.

It fails closed on what it cannot read: `shell=True`, arguments that do not bind
to `Popen`, and a command shape that is neither argv nor a command line. Use
explicit argv or a direct mock in those tests.

The SDK guard sits at the shared channel boundary, so it covers Function,
Cls/Obj, autoscaler, warm-container, App, Sandbox, read-only, synchronous,
asynchronous, and future SDK operations without maintaining a public-method
inventory. It is a hard failure if that boundary cannot be found: `modal` is a
runtime dependency, so a missing `modal.client._Client` means the SDK moved it,
and continuing would leave the suite unguarded and still green. The boundary is
per-process, which is why a child interpreter running this package — the Modal
worker at `python -m runplz.backends.modal_runs` — is refused as a launch: it
builds a real client where no in-process patch applies.

Two invariants keep the module list honest, because a module missing from it
bypasses the guard silently rather than noisily. Every `runplz` module that
imports `subprocess` must be listed, and must bind it as plain `import
subprocess`: `from subprocess import run` and `import subprocess as sp` leave
nothing for the fixture to replace. And every listed module must actually be
patched, so a dead entry cannot read as coverage.

Offline fake clients and explicit mocks remain usable without the marker. Each
guarded module gets its **own** wrapper, so patching one module's
`subprocess.run` opts out that module alone — patch the module that issues the
call, which for a provider CLI is `provisioning`, not the backend that builds
the argv. The marker is consulted per billed name, so a `live_ssh` test may run
`rsync -e 'ssh ...'`. A `sandbox_bin` stub is exempt only when it is the program
actually launched, under the effective `executable=`, `env=`/PATH and `cwd=`,
and only when it exists and is executable. A billed name anywhere else in the
command is an argument to something the guard did not classify, so no PATH of
ours can vouch for it: `env modal ...` is refused even with a stub installed.

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
