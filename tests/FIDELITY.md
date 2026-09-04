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
