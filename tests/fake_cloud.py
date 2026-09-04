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

Option *values* are validated as well as option names (#154), from the live
provisioning catalogues. That makes the stubs deliberately *stricter* than the
real CLIs about machine shapes: `aws` sells `t3.micro`, and this one refuses
it, because the vocabulary is "what runplz generates" rather than "what the
provider offers". The point is to fail when runplz's catalogue drifts out of
the names the provider actually sells -- which a permissive syntactic check
could not do, since `totally-invented-9000` is shaped exactly like a real GCP
machine type. A test that legitimately needs a shape runplz never picks -- a
user's `instance_type=` override -- has to add it to `_option_values`, and
that should be a deliberate act.
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
# real CLIs would reject. The vocabulary is per route, not one flat set shared
# by both stubs: a single set let `gcloud compute instances create` accept
# `--instance-type` and `--region`, and `aws ec2 run-instances` accept `--zone`
# and `--format`, none of which those CLIs have. It also let `--zone` stand in
# for `--zones` on `instances list` -- the exact confusion this file's own
# comment warned about while accepting it.
#
# Each route lists what it *also* accepts; the required options below are
# folded in automatically, so a required option never has to be repeated here.
_OPTIONAL_OPTIONS = {
    "gcloud": {
        "compute/instances/create": {
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
        },
        "compute/config-ssh": {"--quiet"},
        "compute/instances/delete": {"--quiet", "--delete-disks"},
        "compute/instances/stop": {"--quiet"},
        "compute/instances/list": {"--zones"},
    },
    "aws": {
        "ec2/run-instances": {
            "--subnet-id",
            "--security-group-ids",
            "--iam-instance-profile",
            "--block-device-mappings",
            "--instance-market-options",
            "--query",
        },
        "ec2/describe-instances#by-id": set(),
        "ec2/describe-instances#by-filter": set(),
        "ec2/wait/instance-running": set(),
        "ssm/get-parameter": set(),
        "ec2/terminate-instances": set(),
        "ec2/stop-instances": set(),
    },
}

# One subcommand can be two different calls. `aws ec2 describe-instances` is
# both "what is this instance's IP" (--instance-ids, --output text, a bare
# address) and "what runplz instances exist" (--filters, --output json, a
# Reservations document). They take different options and return different
# shapes, so the stub resolves a variant and lets validation and response
# follow the call shape rather than the subcommand name.
_VARIANTS = {
    "ec2/describe-instances": {"--instance-ids": "by-id", "--filters": "by-filter"},
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
        # --filter is required, not incidental: a listing that dropped its
        # label filter would report every instance in the project as a runplz
        # job, and the stub should refuse that argv rather than answer it.
        "compute/instances/list": {"--project", "--filter", "--format"},
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
        "ec2/describe-instances#by-id": {"--region", "--instance-ids", "--query", "--output"},
        "ec2/describe-instances#by-filter": {"--region", "--filters", "--output"},
        "ec2/wait/instance-running": {"--region", "--instance-ids"},
        "ssm/get-parameter": {"--region", "--name", "--query", "--output"},
        "ec2/terminate-instances": {"--region", "--instance-ids", "--output"},
        "ec2/stop-instances": {"--region", "--instance-ids", "--output"},
    },
}


def _machine_names(cpu_shapes, gpu_tables) -> list:
    """Every machine name one provider's catalogue can select."""
    names = {offering.name for offering in cpu_shapes}
    for shapes in gpu_tables.values():
        names.update(offering.name for offering in shapes.offerings)
    return sorted(names)


def _accelerator_names(gpu_tables) -> list:
    return sorted({s.accelerator for s in gpu_tables.values() if s.accelerator})


def _option_values() -> dict:
    """What each option may *contain*, per route.

    Validating names alone still let the stub take
    `--machine-type=totally-invented-9000` and `--count abc` with rc=0 -- the
    first of which is the bug class this module exists to catch.

    Built from the live provisioning catalogues rather than copied beside
    them, so a shape that is removed upstream and dropped from the table stops
    being accepted here too. See the module docstring for why that vocabulary
    is deliberately narrower than the real CLIs'.

    The "*" route applies to every route of that CLI; a per-route entry wins.
    """
    from runplz.backends.provisioning import (
        AWS_CPU_SHAPES,
        AWS_GPUS,
        GCP_CPU_SHAPES,
        GCP_GPUS,
    )

    return {
        "gcloud": {
            "*": {
                # Parametrized projections (`value(name)`) are not modelled;
                # runplz only ever asks for json.
                "--format": {"choices": ["json", "yaml", "text", "csv", "none"]},
            },
            "compute/instances/create": {
                "--machine-type": {"choices": _machine_names(GCP_CPU_SHAPES, GCP_GPUS)},
                "--accelerator": {"accelerator": _accelerator_names(GCP_GPUS)},
                "--maintenance-policy": {"choices": ["MIGRATE", "TERMINATE"]},
                "--provisioning-model": {"choices": ["SPOT", "STANDARD"]},
                "--boot-disk-size": {"int": True, "units": ["GB", "MB", "TB", "GiB", "TiB"]},
                "--boot-disk-type": {
                    "choices": ["pd-standard", "pd-balanced", "pd-ssd", "pd-extreme"]
                },
            },
        },
        "aws": {
            "*": {
                "--output": {"choices": ["json", "text", "table", "yaml", "yaml-stream"]},
                # EC2 resource ids are `<type>-<hex>`. Not pinned to the real
                # width, because the stubs mint short ids of their own and the
                # bug worth catching is a value that is not an id at all -- an
                # ARN, a name, an empty string from a failed lookup.
                "--instance-ids": {"prefix": "i-"},
            },
            "ec2/run-instances": {
                "--instance-type": {"choices": _machine_names(AWS_CPU_SHAPES, AWS_GPUS)},
                "--count": {"int": True},
                "--image-id": {"prefix": "ami-"},
            },
        },
    }


def _values_for(name: str) -> dict:
    """Per-route value specs for one CLI, with the "*" entries folded in."""
    table = _option_values()[name]
    shared = table.get("*", {})
    routes = set(table) | set(_allowed_options(name))
    return {route: {**shared, **table.get(route, {})} for route in routes if route != "*"}


def _allowed_options(name: str) -> dict:
    """Every option each route of one CLI accepts, required ones folded in.

    Keeping the two tables separate and merging here means a required option
    is written once: it cannot be required-but-not-allowed, which would reject
    the very argv the stub demands.
    """
    optional = _OPTIONAL_OPTIONS[name]
    required = _REQUIRED_OPTIONS[name]
    return {route: set(optional[route]) | set(required.get(route, set())) for route in optional}


_TEMPLATE = '''#!/usr/bin/env python3
"""Stub {name} installed by tests/fake_cloud.py. Talks to nothing."""
import json, os, sys

LOG = {log!r}
STATE = LOG + ".state"
TOKENS = LOG + ".tokens"
NAMES = LOG + ".names"
ROUTES = {routes!r}
FAIL_TIMES = {fail_times!r}
FAIL_MESSAGE = {fail_message!r}
MALFORMED = {malformed!r}
ALLOWED_OPTIONS = {_ALLOWED_OPTIONS!r}
REQUIRED_OPTIONS = {_REQUIRED_OPTIONS!r}
VARIANTS = {_VARIANTS!r}
OPTION_VALUES = {_OPTION_VALUES!r}

argv = sys.argv[1:]
positional = [a for a in argv if not a.startswith("-")]

try:
    with open(STATE) as fh:
        state = json.load(fh)
except FileNotFoundError:
    state = {{}}
try:
    with open(TOKENS) as fh:
        tokens = json.load(fh)
except FileNotFoundError:
    tokens = {{}}
try:
    with open(NAMES) as fh:
        names = json.load(fh)
except FileNotFoundError:
    names = {{}}

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

# Resolve the call shape before anything reads it. A route with variants is
# ambiguous until we know which discriminating option was passed, and both the
# stateful block below and the required-option check need the answer -- so an
# argv that names none of them (or several) is rejected here rather than
# falling through to whichever branch happens to be written first.
route_name = "/".join(key)
variants = VARIANTS.get(route_name)
if variants:
    matched = sorted({{name for option, name in variants.items() if option in option_names}})
    if len(matched) != 1:
        sys.stderr.write(
            "stub {name}: " + route_name + " needs exactly one of "
            + " ".join(sorted(variants)) + "\\n"
        )
        sys.exit(2)
    route_name = route_name + "#" + matched[0]

# Per route, not one vocabulary shared by both CLIs -- so gcloud rejects
# `--instance-type` and aws rejects `--zone`, and `instances list` rejects the
# singular `--zone` it does not take.
unknown = sorted(option_names - ALLOWED_OPTIONS.get(route_name, set()))
if unknown:
    sys.stderr.write("stub {name}: unknown option(s): " + " ".join(unknown) + "\\n")
    sys.exit(2)

# Argv is fully validated before anything is mutated. Validating after the
# fact still let a rejected command line leave state behind: `aws ec2
# run-instances --region us-east-1` allocated a client token and an
# instance entry and *then* exited 2 for its missing options, so a test
# asserting the rejection was silently seeding the next call's state.
missing = sorted(REQUIRED_OPTIONS.get("{name}", {{}}).get(route_name, set()) - option_names)
if missing:
    sys.stderr.write("stub {name}: missing required option(s): " + " ".join(missing) + "\\n")
    sys.exit(2)
# Option values, not just option names. `--machine-type=totally-invented-9000`
# and `--count abc` both exited 0 before this: the stub asserted the shape of a
# command line while accepting anything at all inside it.
def bad_value(spec, value):
    if value is None:
        return "has no value"
    if "choices" in spec:
        if value not in spec["choices"]:
            return "expected one of " + " ".join(spec["choices"])
        return None
    if "accelerator" in spec:
        # gcloud's own shape: type=<name>,count=<n>.
        fields = dict(
            piece.split("=", 1) for piece in value.split(",") if "=" in piece
        )
        if set(fields) != {{"type", "count"}}:
            return "expected type=<accelerator>,count=<n>"
        if fields["type"] not in spec["accelerator"]:
            return "unknown accelerator " + fields["type"]
        if not fields["count"].isdigit():
            return "count must be an integer"
        return None
    if "prefix" in spec:
        if not value.startswith(spec["prefix"]) or value == spec["prefix"]:
            return "expected a value starting with " + spec["prefix"]
        return None
    if spec.get("int"):
        digits = value
        for unit in spec.get("units", []):
            if digits.endswith(unit):
                digits = digits[: -len(unit)]
                break
        if not digits.isdigit():
            units = spec.get("units")
            suffix = (" with an optional " + "/".join(units) + " suffix") if units else ""
            return "expected an integer" + suffix
        return None
    return None

for option, spec in sorted(OPTION_VALUES.get(route_name, {{}}).items()):
    if option not in option_names:
        continue
    problem = bad_value(spec, option_value(option))
    if problem is not None:
        sys.stderr.write(
            "stub {name}: invalid value for " + option + ": "
            + repr(option_value(option)) + " (" + problem + ")\\n"
        )
        sys.exit(2)

if key[0:3] == ("compute", "instances", "create") and len(positional) <= 3:
    sys.stderr.write("stub gcloud: create requires an instance name\\n")
    sys.exit(2)
if key[0:3] in (("compute", "instances", "delete"), ("compute", "instances", "stop")):
    if len(positional) <= 3:
        sys.stderr.write("stub gcloud: lifecycle command requires an instance name\\n")
        sys.exit(2)

if key == ("ec2", "run-instances"):
    token = option_value("--client-token")
    instance_id = tokens.get(token)
    if instance_id is None:
        instance_id = "i-fake" + str(len(tokens) + 1).zfill(10)
        tokens[token] = instance_id
    state[instance_id] = "running"
    spec = option_value("--tag-specifications") or ""
    for chunk in "[]{{}}":
        spec = spec.replace(chunk, ",")
    parts = [piece for piece in spec.split(",") if piece]
    for index, piece in enumerate(parts[:-1]):
        if piece == "Key=Name" and parts[index + 1].startswith("Value="):
            names[instance_id] = parts[index + 1][len("Value="):]
            break
elif route_name in ("ec2/describe-instances#by-id", "ec2/wait/instance-running",
                    "ec2/terminate-instances", "ec2/stop-instances"):
    instance_id = option_value("--instance-ids")
    if instance_id not in state or state[instance_id] in {{"terminated", "DELETED"}}:
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
with open(TOKENS, "w") as fh:
    json.dump(tokens, fh)
with open(NAMES, "w") as fh:
    json.dump(names, fh)
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

# A variant key wins, so one call shape of a subcommand can be made malformed
# without the other; the bare route name still works and still covers both.
malformed_body = MALFORMED.get(route_name, MALFORMED.get("/".join(key)))
if malformed_body is not None:
    sys.stdout.write(malformed_body)
else:
    if key == ("ec2", "run-instances"):
        sys.stdout.write(json.dumps({{"Instances": [{{"InstanceId": instance_id}}]}}))
    elif route_name == "ec2/describe-instances#by-id":
        # Keep the fake response tied to the requested resource.  A
        # production bug that asks about the wrong instance must not still
        # receive a universal localhost address.
        requested = option_value("--instance-ids") or ""
        digits = "".join(ch for ch in requested if ch.isdigit())
        octet = (int(digits[-3:]) if digits else 1) % 254 or 1
        sys.stdout.write("127.0.0." + str(octet))
    elif route_name == "ec2/describe-instances#by-filter":
        # Report what this stub actually created rather than a fixture, so a
        # create-then-list test proves the round trip. Terminated instances
        # are dropped because production asks for
        # `Name=instance-state-name,Values=pending,running,stopping,stopped`;
        # the tag filter needs no work because the stub only ever creates
        # runplz-tagged instances.
        reservations = []
        for iid in sorted(state):
            if not iid.startswith("i-") or state[iid] == "terminated":
                continue
            reservations.append({{"Instances": [{{
                "InstanceId": iid,
                "LaunchTime": "2026-01-01T00:00:00+00:00",
                "State": {{"Name": state[iid]}},
                "Tags": [
                    {{"Key": "Name", "Value": names.get(iid, iid)}},
                    {{"Key": "runplz", "Value": "1"}},
                ],
            }}]}})
        sys.stdout.write(json.dumps({{"Reservations": reservations}}))
    elif route_name == "compute/instances/list":
        # Same, for gcloud -- which already keys its state by instance name,
        # so nothing extra had to be recorded. DELETED and TERMINATED drop out
        # for the same reason terminated does on the aws side.
        sys.stdout.write(json.dumps([
            {{
                "name": iname,
                "creationTimestamp": "2026-01-01T00:00:00.000-08:00",
                "status": state[iname],
                "labels": {{"runplz": "1"}},
            }}
            for iname in sorted(state)
            if state[iname] not in ("DELETED", "TERMINATED")
        ]))
    else:
        sys.stdout.write(ROUTES[key])
sys.exit(0)
'''


def install(bin_dir: Path, *, name: str, fail_times=None, fail_message="", malformed=None) -> Path:
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
            malformed=malformed or {},
            _ALLOWED_OPTIONS=_allowed_options(name),
            _REQUIRED_OPTIONS=_REQUIRED_OPTIONS,
            _VARIANTS=_VARIANTS,
            _OPTION_VALUES=_values_for(name),
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


def assert_observed(log: Path, *tokens: str, count: int | None = None) -> list:
    """Assert that the real stub observed a command containing ``tokens``.

    Keeping this assertion beside the subprocess fake makes tests fail when a
    production path is accidentally replaced by a mock or silently skipped.
    ``count`` optionally pins the number of observations for retry tests.
    """
    observed = calls_matching(log, *tokens)
    assert observed, (
        f"fake {log.stem.removesuffix('-calls')} observed no command containing {tokens!r}"
    )
    if count is not None:
        assert len(observed) == count, (tokens, observed)
    return observed


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
