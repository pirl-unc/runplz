"""Stub `gcloud` / `aws` executables that provision localhost.

The cloud backends shell out to the real CLIs, so everything between
`run()` and the CLI boundary -- argv construction, JSON parsing, retry
classification, teardown, the ssh handover -- is only ever exercised
against `mock.patch`. That means those tests assert what the author
*believed* the CLI does. Two bugs this session came from exactly that gap
(invented machine types, retry strings written from memory).

These stubs close it from the other side: real `subprocess` calls, real
argv, real JSON on stdout, real exit codes -- and an "instance" that is
just localhost, so the ssh half can run for real too when sshd is up.

Every invocation is appended to a JSONL log the test can assert on, and
`fail_times` scripts transient failures so the retry loop is exercised
rather than described.
"""

import json
import os
import stat
from pathlib import Path

# Stdout the stub returns for each subcommand it understands. Keyed by the
# argv slice that identifies the call.
_GCLOUD_ROUTES = {
    ("compute", "instances", "create"): json.dumps(
        [{"name": "fake-instance", "status": "RUNNING"}]
    ),
    ("compute", "config-ssh"): "",
    ("compute", "instances", "delete"): "",
    ("compute", "instances", "stop"): "",
    ("compute", "instances", "list"): "[]",
}

_AWS_ROUTES = {
    ("ec2", "run-instances"): json.dumps({"Instances": [{"InstanceId": "i-fake0123456789"}]}),
    ("ec2", "describe-instances"): "127.0.0.1",
    ("ec2", "wait", "instance-running"): "",
    ("ec2", "terminate-instances"): json.dumps({"TerminatingInstances": []}),
    ("ec2", "stop-instances"): json.dumps({"StoppingInstances": []}),
    ("ssm", "get-parameter"): "ami-0fake00000000000",
}

_TEMPLATE = '''#!/usr/bin/env python3
"""Stub {name} installed by tests/fake_cloud.py. Talks to nothing."""
import json, os, sys

LOG = {log!r}
ROUTES = {routes!r}
FAIL_TIMES = {fail_times!r}
FAIL_MESSAGE = {fail_message!r}

argv = sys.argv[1:]
positional = [a for a in argv if not a.startswith("-")]

with open(LOG, "a") as fh:
    fh.write(json.dumps({{"argv": argv}}) + "\\n")

# Scripted transient failures, so the retry loop is exercised for real.
key = None
for route in sorted(ROUTES, key=len, reverse=True):
    if list(route) == positional[: len(route)]:
        key = route
        break

if key is not None and FAIL_TIMES.get("/".join(key), 0) > 0:
    seen = 0
    with open(LOG) as fh:
        for line in fh:
            entry = json.loads(line)
            p = [a for a in entry["argv"] if not a.startswith("-")]
            if list(key) == p[: len(key)]:
                seen += 1
    if seen <= FAIL_TIMES["/".join(key)]:
        sys.stderr.write(FAIL_MESSAGE + "\\n")
        sys.exit(1)

if key is None:
    sys.stderr.write("stub {name}: unhandled command: " + " ".join(argv) + "\\n")
    sys.exit(2)

sys.stdout.write(ROUTES[key])
sys.exit(0)
'''


def install(bin_dir: Path, *, name: str, fail_times=None, fail_message="") -> Path:
    """Write a stub `gcloud` or `aws` into `bin_dir`; return its call log."""
    routes = {"gcloud": _GCLOUD_ROUTES, "aws": _AWS_ROUTES}[name]
    log = bin_dir / f"{name}-calls.jsonl"
    script = bin_dir / name
    script.write_text(
        _TEMPLATE.format(
            name=name,
            log=str(log),
            routes=routes,
            fail_times=fail_times or {},
            fail_message=fail_message,
        )
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    log.write_text("")
    return log


def calls(log: Path) -> list:
    """Every argv the stub was invoked with, in order."""
    if not log.exists():
        return []
    return [json.loads(line)["argv"] for line in log.read_text().splitlines() if line.strip()]


def calls_matching(log: Path, *tokens: str) -> list:
    """Invocations whose argv contains all of `tokens`."""
    return [argv for argv in calls(log) if all(t in argv for t in tokens)]


def install_unreachable_ssh(bin_dir: Path) -> Path:
    """An `ssh` that always fails, standing in for a box that never came up.

    Used to drive the "dispatch died, teardown must still run" path with a
    realistic failure rather than a mocked exception -- the provisioning
    half stays completely real.
    """
    script = bin_dir / "ssh"
    script.write_text(
        "#!/bin/sh\necho 'ssh: connect to host port 22: Connection refused' >&2\nexit 255\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return script


def ssh_alias_to_localhost(ssh_config: Path, alias: str) -> None:
    """Point a gcloud-style ssh alias at localhost, as config-ssh would."""
    ssh_config.parent.mkdir(parents=True, exist_ok=True)
    with open(ssh_config, "a") as fh:
        fh.write(
            f"\nHost {alias}\n"
            f"  HostName 127.0.0.1\n"
            f"  User {os.environ.get('USER', 'runner')}\n"
            f"  StrictHostKeyChecking no\n"
            f"  UserKnownHostsFile /dev/null\n"
        )
