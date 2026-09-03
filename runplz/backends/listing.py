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
]


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
    # Value parser, handed to argparse as `type`.
    type: Callable = str

    def resolve(self, value: Optional[str] = None) -> Optional[str]:
        """The explicit value, else the first environment variable that is
        set, else None. An empty environment variable counts as unset —
        `AWS_DEFAULT_REGION=` is not a region."""
        if value is not None:
            return value
        for name in self.env:
            found = os.environ.get(name)
            if found:
                return found
        return None

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

    def has_required_scope(self, values: dict) -> bool:
        """True when the user supplied enough to list this backend.

        Asked of backends outside the default fan-out, to decide whether the
        user's flags invited one in.
        """
        return all(f.resolve(values.get(f.name)) is not None for f in self.required_fields())
