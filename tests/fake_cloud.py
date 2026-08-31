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

# The stub is intentionally small, but it must reject command lines that the
# real CLIs would reject. Keeping the option vocabulary here means a typo in a
# production command fails the test instead of being accepted by a permissive
# fake.
_KNOWN_OPTIONS = {
    "--project",
    "--zone",
    "--machine-type",
    "--image-family",
    "--image-project",
    "--labels",
    "--format",
    "--quiet",
    "--accelerator",
    "--maintenance-policy",
    "--boot-disk-size",
    "--boot-disk-type",
    "--network",
    "--subnet",
    "--service-account",
    "--scopes",
    "--provisioning-model",
    "--no-restart-on-failure",
    "--delete-disks",
    "--region",
    "--image-id",
    "--instance-type",
    "--key-name",
    "--count",
    "--client-token",
    "--tag-specifications",
    "--output",
    "--subnet-id",
    "--security-group-ids",
    "--iam-instance-profile",
    "--block-device-mappings",
    "--instance-market-options",
    "--query",
    "--name",
    "--instance-ids",
}

_REQUIRED_OPTIONS = {
    "gcloud": {
        "compute/instances/create": {
            "--project",
            "--zone",
            "--machine-type",
            "--image-family",
            "--image-project",
        },
        "compute/config-ssh": {"--project"},
        "compute/instances/delete": {"--project", "--zone"},
        "compute/instances/stop": {"--project", "--zone"},
    },
    "aws": {
        "ec2/run-instances": {
            "--region",
            "--image-id",
            "--instance-type",
            "--key-name",
            "--count",
            "--client-token",
            "--tag-specifications",
            "--output",
        },
        "ec2/describe-instances": {"--region", "--instance-ids", "--query", "--output"},
        "ec2/wait/instance-running": {"--region", "--instance-ids"},
        "ssm/get-parameter": {"--region", "--name", "--query", "--output"},
        "ec2/terminate-instances": {"--region", "--instance-ids", "--output"},
        "ec2/stop-instances": {"--region", "--instance-ids", "--output"},
    },
}

_TEMPLATE = '''#!/usr/bin/env python3
"""Stub {name} installed by tests/fake_cloud.py. Talks to nothing."""
import json, os, sys

LOG = {log!r}
STATE = LOG + ".state"
ROUTES = {routes!r}
FAIL_TIMES = {fail_times!r}
FAIL_MESSAGE = {fail_message!r}
KNOWN_OPTIONS = {_KNOWN_OPTIONS!r}
REQUIRED_OPTIONS = {_REQUIRED_OPTIONS!r}

argv = sys.argv[1:]
positional = [a for a in argv if not a.startswith("-")]

try:
    with open(STATE) as fh:
        state = json.load(fh)
except FileNotFoundError:
    state = {{}}

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

option_names = {{a.split("=", 1)[0] for a in argv if a.startswith("--")}}
unknown = sorted(option_names - KNOWN_OPTIONS)
if unknown:
    sys.stderr.write("stub {name}: unknown option(s): " + " ".join(unknown) + "\\n")
    sys.exit(2)
option_counts = {{a.split("=", 1)[0]: 0 for a in argv if a.startswith("--")}}
for option in (a.split("=", 1)[0] for a in argv if a.startswith("--")):
    option_counts[option] += 1
duplicates = sorted(option for option, count in option_counts.items() if count > 1)
if duplicates:
    sys.stderr.write("stub {name}: duplicate option(s): " + " ".join(duplicates) + "\\n")
    sys.exit(2)

def option_value(name):
    prefix = name + "="
    for value in argv:
        if value.startswith(prefix):
            return value[len(prefix):]
    try:
        return argv[argv.index(name) + 1]
    except (ValueError, IndexError):
        return None

if key == ("ec2", "run-instances"):
    instance_id = "i-fake0123456789"
    state[instance_id] = "running"
elif key in (("ec2", "describe-instances"), ("ec2", "wait", "instance-running"),
             ("ec2", "terminate-instances"), ("ec2", "stop-instances")):
    instance_id = option_value("--instance-ids")
    if instance_id not in state:
        sys.stderr.write("InvalidInstanceID.NotFound: instance does not exist\\n")
        sys.exit(1)
    if key == ("ec2", "terminate-instances"):
        state[instance_id] = "terminated"
    elif key == ("ec2", "stop-instances"):
        state[instance_id] = "stopped"
elif key == ("compute", "instances", "create") and len(positional) > 3:
    state[positional[3]] = "RUNNING"
elif (
    key in (("compute", "instances", "delete"), ("compute", "instances", "stop"))
    and len(positional) > 3
):
    instance_name = positional[3]
    if instance_name not in state:
        sys.stderr.write("instance does not exist\\n")
        sys.exit(1)
    state[instance_name] = "DELETED" if key[-1] == "delete" else "TERMINATED"

with open(STATE, "w") as fh:
    json.dump(state, fh)
route_name = "/".join(key)
missing = sorted(REQUIRED_OPTIONS.get("{name}", {{}}).get(route_name, set()) - option_names)
if missing:
    sys.stderr.write("stub {name}: missing required option(s): " + " ".join(missing) + "\\n")
    sys.exit(2)
if key[0:3] == ("compute", "instances", "create") and len(positional) <= 3:
    sys.stderr.write("stub gcloud: create requires an instance name\\n")
    sys.exit(2)
if key[0:3] in (("compute", "instances", "delete"), ("compute", "instances", "stop")):
    if len(positional) <= 3:
        sys.stderr.write("stub gcloud: lifecycle command requires an instance name\\n")
        sys.exit(2)

if "{name}" == "gcloud" and route_name == "compute/config-ssh":
    config = os.environ.get("RUNPLZ_FAKE_SSH_CONFIG")
    if config:
        parent = os.path.dirname(config)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(config, "w") as fh:
            fh.write(
                "Host *\\n"
                f"  HostName {{os.environ.get('RUNPLZ_FAKE_SSH_HOST', '127.0.0.1')}}\\n"
                f"  User {{os.environ.get('RUNPLZ_FAKE_SSH_USER', 'runner')}}\\n"
                f"  Port {{os.environ.get('RUNPLZ_FAKE_SSH_PORT', '')}}\\n"
                f"  IdentityFile {{os.environ.get('RUNPLZ_FAKE_SSH_IDENTITY', '')}}\\n"
                "  StrictHostKeyChecking no\\n"
                "  UserKnownHostsFile /dev/null\\n"
            )

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
            _KNOWN_OPTIONS=_KNOWN_OPTIONS,
            _REQUIRED_OPTIONS=_REQUIRED_OPTIONS,
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
