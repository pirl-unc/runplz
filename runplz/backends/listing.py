"""One shape for a listed job, and one way to say what listing it costs.

`runplz ps` asks every backend the same question — what is running? — but
the backends need very different things before they can answer. Local
docker needs nothing. AWS needs a region. SSH needs a host, and there is no
registry to get one from.

That variation used to live in two wrong places at once: the CLI hardcoded a
flag per provider and an ``if backend == ...`` chain to turn those flags back
into keyword arguments, while each cloud driver separately re-read its own
environment variables and raised its own error *after* dispatch had already
begun. Adding a backend meant editing both.

This module holds the two types that replace all of it:

- :class:`JobRecord` is what every driver returns. One shape, so the table
  formatter reads fields instead of guessing at dictionary keys.
- :class:`ListingSpec` is what a backend declares in the registry: the
  :class:`ScopeField` values it needs, where each may come from, and whether
  a bare ``runplz ps`` should query it at all. Scope is resolved and checked
  here, before any provider CLI is spawned.

Deliberately free of provider knowledge and of argparse, so the dependency
graph stays a DAG: this module imports nothing from runplz, and the drivers,
`runplz.backends.docker` and `runplz.backends.registry` all import it.
"""

import os
from dataclasses import dataclass, fields
from typing import Callable, Optional

__all__ = [
    "JobRecord",
    "ScopeField",
    "ListingSpec",
    "MissingScope",
    "ListingUnsupported",
    "tcp_port_range",
]


def tcp_port_range(port) -> None:
    """Reject a port outside 1-65535.

    Phrased to read as a continuation of whichever flag carries it, so one
    definition serves `runplz ps --ssh-port` and the `tail`/`status`/`kill`
    override that used to keep its own copy of these three lines.
    """
    if not 0 < port < 65536:
        raise ValueError(f"must be a valid TCP port (1-65535); got {port}.")


class MissingScope(RuntimeError):
    """A backend was asked to list jobs without the scope it requires.

    Raised before the provider CLI is invoked — the point is that runplz
    never spends a round trip (or an API call) to be told something it could
    have known from the arguments.
    """


class ListingUnsupported(RuntimeError):
    """This backend cannot enumerate jobs at all.

    Distinct from "no jobs are running", which is an empty list. A backend
    with no listing at all must say so rather than let an empty table imply
    the user's jobs are gone.
    """


@dataclass(frozen=True)
class JobRecord:
    """One running job, as `runplz ps` reports it.

    Every field is a string because every one of them is provider-formatted
    passthrough: `started` is whatever timestamp the provider prints and
    `status` is its own state vocabulary (`running`, `RUNNING`, `Up 5
    minutes`, `DEPLOYING`). Normalizing either would mean inventing a
    mapping that hides what the provider actually said.

    Field order is the column order of the `runplz ps` table.
    """

    backend: str
    name: str
    app: str = ""
    function: str = ""
    started: str = ""
    status: str = ""

    @classmethod
    def field_names(cls) -> tuple:
        """The fields in table-column order.

        The formatter reads this rather than repeating the names, so a
        renamed field moves the column with it instead of silently printing
        blanks.
        """
        return tuple(f.name for f in fields(cls))


@dataclass(frozen=True)
class ScopeField:
    """One value a backend needs before it can list jobs.

    `name` is the keyword the driver's `list_jobs` takes; `flag` is how the
    user spells it. They are separate on purpose — ssh's `--ssh-key` feeds a
    parameter called `ssh_key_path`, and neither name should have to bend to
    the other.

    Carrying `flag` here is what lets the missing-scope message say "pass
    --region" without the driver knowing that a CLI exists.
    """

    name: str
    flag: str
    help: str
    # Additional CLI spellings. `--host` keeps `--ssh` from before 3.15.1.
    aliases: tuple = ()
    # Environment variables consulted, in order, when the flag is absent.
    env: tuple = ()
    required: bool = False
    # Comma-separated, one listing call per value. SSH hosts are the case:
    # there is no registry, so the user names the targets, and each one is
    # queried (and can fail) on its own.
    multiple: bool = False
    # Value parser, handed to argparse as `type` and applied to environment
    # values, which arrive as strings however the field is declared.
    type: Callable = str
    # Optional constraint, raising ValueError with a message that reads as a
    # continuation of the flag: "must be a valid TCP port (1-65535)". Lives
    # with the declaration so a renamed field cannot leave its check behind.
    validate: Optional[Callable] = None

    def resolve(self, value=None):
        """The explicit value, else the first environment variable set, else
        None.

        Blank counts as unset from *either* source. `--region ''` and
        `AWS_DEFAULT_REGION=` are equally "a region the user did not supply",
        and forwarding either one sends `--region ''` to the provider instead
        of saying what is missing.
        """
        if isinstance(value, str):
            value = value.strip()
        if value is not None and value != "":
            return value
        for name in self.env:
            found = (os.environ.get(name) or "").strip()
            if found:
                # Environment values are always strings; a typed field has to
                # apply its parser here or the driver gets "2222", not 2222.
                return self.type(found)
        return None

    def resolve_all(self, value=None) -> list:
        """Every target this field names, which is at most one unless it is
        `multiple`.

        Splitting lives here rather than in the CLI so a value is split the
        same way whichever source it came from, and so a field that resolves
        to nothing but separators — `--host ,` — is correctly empty rather
        than one target named ",".
        """
        resolved = self.resolve(value)
        if resolved is None:
            return []
        if not self.multiple:
            return [resolved]
        return [part.strip() for part in str(resolved).split(",") if part.strip()]

    def missing_message(self, backend: str) -> str:
        """Why this backend cannot be listed, and what would fix it."""
        fix = f"pass {self.flag}"
        if self.env:
            fix += f" or set {'/'.join(self.env)}"
        return f"{backend} {self.name} is required; {fix}"


@dataclass(frozen=True)
class ListingSpec:
    """How `runplz ps` can enumerate one backend's jobs.

    `default_fan_out` is stated rather than derived from "every required
    field has an environment fallback". Those two happen to agree for the
    backends that exist today, which is exactly why deriving it is a trap:
    it reads as a rule while really being a coincidence, and the first
    backend that breaks the correlation gets silently added to — or dropped
    from — the output of a bare `runplz ps`.
    """

    scope: tuple = ()
    # True when a bare `runplz ps` should query this backend unprompted.
    default_fan_out: bool = True

    def required_fields(self) -> tuple:
        return tuple(f for f in self.scope if f.required)

    def resolve(self, backend: str, values: dict) -> dict:
        """Turn user-supplied scope into driver keyword arguments.

        Every declared field is returned, with None for the ones left unset,
        so a driver sees the same call shape whether or not the optional
        scope was given. Raises :class:`MissingScope` for a required field
        that no flag and no environment variable supplied.
        """
        resolved = {}
        for f in self.scope:
            value = f.resolve(values.get(f.name))
            if value is None and f.required:
                raise MissingScope(f.missing_message(backend))
            resolved[f.name] = value
        return resolved

    def invited_by(self, values: dict) -> bool:
        """True when the user's flags amount to asking for this backend.

        Asked of backends outside the default fan-out. One with no required
        scope can never be invited this way — there is no flag to give — so
        it answers False rather than True-by-vacuous-`all()`, which would
        quietly return it to the fan-out it opted out of. Such a backend
        joins only when named positionally.
        """
        required = self.required_fields()
        return bool(required) and all(f.resolve_all(values.get(f.name)) for f in required)
