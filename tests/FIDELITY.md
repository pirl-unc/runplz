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

The command guard hooks `subprocess.Popen` — the one seam every spawn in the
process goes through, whether spelled `run`, `check_output`, `getoutput`,
`from subprocess import run`, an asyncio subprocess, or a call from a test
helper — so there is no list of modules to keep in sync and no way for a new
module to shell out unguarded. Options are read by binding against
`Popen.__init__`'s signature, so `shell=`, `executable=`, `env=` and `cwd=` are
seen whether passed positionally or by keyword, and `getoutput` is refused for
what it is: `shell=True`.

It does not try to work out which token is *the* program, and it models no
wrapper by name. It reads every word the command is built from — inside an
argument too, split on whitespace, the shell's own separators and `=`, with a
version specifier stripped (`modal[aws]`, `modal~=1.5`) and every suffix of a
short-option cluster considered (`-Smodal`, `-mmodal`) — and refuses any word
whose casefolded basename is a billed CLI, or that names the `modal` package or
its worker in this package (`runplz.backends.modal_runs`, by module name or by
file path). So `uv run modal`, `timeout 600 modal`, `sh -c "cd /tmp;modal run
job"`, `uvx modal==1.5` and every `env` spelling are caught by one rule, and a
wrapper nobody has thought of yet is caught too. Matching is strict basename
equality, so `gcloud compute config-ssh`, `aws ssm --name /aws/service/...`,
`rsync --exclude=.ssh`, `cat aws.json` and `rsync -avzm` stay runnable, and
runplz's own bootstrap module in an ssh remote-command string is not a launch.

A child running this package's CLI (`runplz`, `python -m runplz.cli`, or the
file by path) can dispatch to any backend, so no single marker can vouch for
it: it is refused outright, like `shell=True`. The bare console-script name is
billed only as the launched program, because `runplz` is ordinary data
everywhere else in this repo's own argv. The guard fails closed on what it
cannot read — `shell=True`, arguments that do not bind to `Popen`, a command
shape that is neither argv nor a command line. It does not read stdin
(`input=`) or `env=` values; a launch delivered that way is outside the
contract, and no product path carries one.

The SDK guard sits at the shared channel boundary, so it covers Function,
Cls/Obj, autoscaler, warm-container, App, Sandbox, read-only, synchronous,
asynchronous, and future SDK operations without maintaining a public-method
inventory. It is a hard failure if that boundary cannot be found: `modal` is a
runtime dependency, so a missing `modal.client._Client` means the SDK moved it,
and continuing would leave the suite unguarded and still green. The boundary is
per-process, which is why the worker child is refused as a launch: it builds a
real client where no in-process patch applies. The three tests that must run
that real child and can show it stays offline use the `real_child_processes`
fixture, with a comment saying why.

Offline fake clients and explicit mocks remain usable without the marker.
`mock.patch("<module>.subprocess.run")` takes the boundary over for the whole
process — `subprocess` is one module — so patch the function that issues the
call; for a provider CLI that is `provisioning.run_with_retries`'s
`subprocess.run`, not the backend that builds the argv. The marker is consulted
per billed name, so a `live_ssh` test may run `rsync -e 'ssh ...'`. A
`sandbox_bin` stub is exempt only when it is the program actually launched,
under the effective `executable=`, `env=`/PATH and `cwd=`, and only when it
exists and is executable. A billed name anywhere else in the command is an
argument to something the guard did not classify, so no PATH of ours can vouch
for it: `env modal ...` is refused even with a stub installed.

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
