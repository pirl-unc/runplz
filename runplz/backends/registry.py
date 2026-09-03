"""One description of what backends exist and what each one accepts.

Adding a backend used to mean editing five places that had to agree: the
`bind()` validator, the CLI's `choices`, `App._dispatch`'s if-chain, the
`runplz ps` backend tuple, and its own if-chain in `_collect_backend_jobs`.
Any one of them missed and the backend was half-wired — importable but
absent from `--help`, or dispatchable but invisible to `ps`.

Now it is one entry here. Everything else reads from this.

That includes listing. A backend declares a `ListingSpec` saying what scope
`runplz ps` needs before it can ask — and a backend that cannot enumerate
jobs at all declares `listing=None`, which :func:`list_jobs` refuses rather
than answering with an empty list. The CLI builds its flags, its fan-out set
and its validation from those declarations, so `_collect_backend_jobs`'s
per-provider if-chain is gone too.

Backends are imported lazily by name, so a missing optional dependency
(`modal`) only surfaces when that backend is actually used, and importing
`runplz` never drags in every cloud driver.
"""

import importlib
from dataclasses import dataclass
from typing import Optional

from runplz.backends.listing import (
    JobRecord,
    ListingSpec,
    ListingUnsupported,
    ScopeField,
    tcp_port_range,
)

__all__ = [
    "BackendSpec",
    "BACKENDS",
    "names",
    "get",
    "load",
    "ps_names",
    "listable_names",
    "scope_fields",
    "list_jobs",
    "provisioning_names",
]


@dataclass(frozen=True)
class BackendSpec:
    """What runplz needs to know about one backend."""

    name: str
    module: str
    # App attribute holding this backend's config, when one is *required*
    # before dispatch. Backends with usable defaults leave this None.
    required_config_attr: Optional[str] = None
    # How `runplz ps` can enumerate this backend's jobs. None means it
    # cannot — an explicit incapacity, not an empty result.
    listing: Optional[ListingSpec] = None
    # Which target selector this backend takes, if any.
    accepts_instance: bool = False
    accepts_host: bool = False
    # Only `local` can skip the image build and reuse a tagged image.
    accepts_no_build: bool = False


BACKENDS = {
    spec.name: spec
    for spec in (
        BackendSpec(
            name="local",
            module="runplz.backends.local",
            listing=ListingSpec(),
            accepts_no_build=True,
        ),
        BackendSpec(
            name="brev",
            module="runplz.backends.brev",
            listing=ListingSpec(),
            # instance=None is meaningful: ephemeral mode, where runplz
            # creates a box sized to the function and deletes it on exit.
            accepts_instance=True,
        ),
        BackendSpec(
            name="modal",
            module="runplz.backends.modal",
            listing=ListingSpec(),
        ),
        BackendSpec(
            name="ssh",
            module="runplz.backends.ssh",
            # SSH has no host registry, so there is nothing for a bare
            # `runplz ps` to query — it joins only when given a host.
            listing=ListingSpec(
                default_fan_out=False,
                scope=(
                    ScopeField(
                        name="host",
                        flag="--host",
                        aliases=("--ssh",),  # pre-3.15.1 spelling, kept indefinitely
                        required=True,
                        multiple=True,
                        help=(
                            "Also probe this SSH host. SSH has no job registry, so "
                            "the host must be supplied. May be given multiple times "
                            "(comma-separated)."
                        ),
                    ),
                    ScopeField(
                        name="ssh_key_path",
                        flag="--ssh-key",
                        help=(
                            "Private key for --host. Needed for a box that is not "
                            "in your ssh config — an EC2 instance, say."
                        ),
                    ),
                    ScopeField(
                        name="port",
                        flag="--ssh-port",
                        type=int,
                        validate=tcp_port_range,
                        help="Port for --host.",
                    ),
                ),
            ),
            accepts_host=True,
        ),
        BackendSpec(
            name="gcp",
            module="runplz.backends.gcp",
            required_config_attr="gcp_config",
            listing=ListingSpec(
                scope=(
                    ScopeField(
                        name="project",
                        flag="--project",
                        env=("GOOGLE_CLOUD_PROJECT", "GCLOUD_PROJECT"),
                        required=True,
                        help="Project to query.",
                    ),
                    ScopeField(
                        name="zone",
                        flag="--zone",
                        env=("CLOUDSDK_COMPUTE_ZONE",),
                        help="Zone to query.",
                    ),
                )
            ),
        ),
        BackendSpec(
            name="aws",
            module="runplz.backends.aws",
            required_config_attr="aws_config",
            listing=ListingSpec(
                scope=(
                    ScopeField(
                        name="region",
                        flag="--region",
                        env=("AWS_DEFAULT_REGION", "AWS_REGION"),
                        required=True,
                        help="Region to query.",
                    ),
                )
            ),
        ),
    )
}


def names() -> tuple:
    """Every backend name, in the order they should be offered."""
    return tuple(BACKENDS)


def get(name: str) -> BackendSpec:
    try:
        return BACKENDS[name]
    except KeyError:
        raise ValueError(f"backend must be one of {names()}; got {name!r}") from None


def ps_names() -> tuple:
    """Backends a bare `runplz ps` queries without extra input from the user."""
    return tuple(n for n, spec in BACKENDS.items() if spec.listing and spec.listing.default_fan_out)


def listable_names() -> tuple:
    """Every backend `runplz ps` can enumerate, given the scope it asks for.

    Wider than :func:`ps_names` — ssh belongs here but not in the fan-out,
    since it can be listed only once the user names a host.
    """
    return tuple(n for n, spec in BACKENDS.items() if spec.listing)


def scope_fields() -> list:
    """Every listing scope field, with the backends that accept it.

    Deduplicated by flag so the CLI offers one `--region`, not one per
    backend, and can tag its help text with who reads it. Returns
    `(ScopeField, backend_names)` pairs in registry order.
    """
    seen = {}
    owner_of_name = {}
    for name in listable_names():
        for field in BACKENDS[name].listing.scope:
            # Both halves of a field's identity have to be unique, because the
            # CLI keys argparse's `dest` — and the scope dict — by `name` while
            # keying the option itself by `flag`. Guarding only the flag lets
            # two backends declare the same `name` under different flags, which
            # argparse accepts and then silently cross-feeds.
            claimed = owner_of_name.setdefault(field.name, (field, name))
            if claimed[0] != field:
                raise ValueError(
                    f"scope {field.name!r} is declared differently by "
                    f"{claimed[1]} ({claimed[0].flag}) and {name} ({field.flag}); "
                    "they must agree"
                )
            existing = seen.get(field.flag)
            if existing is None:
                seen[field.flag] = (field, [name])
                continue
            if existing[0] != field:
                raise ValueError(
                    f"{field.flag} is declared differently by "
                    f"{existing[1][0]} and {name}; they must agree"
                )
            existing[1].append(name)
    return [(field, tuple(backends)) for field, backends in seen.values()]


def list_jobs(backend: str, **scope) -> list[JobRecord]:
    """Enumerate one backend's running jobs as :class:`JobRecord` values.

    The single place scope is validated, and it happens before the driver —
    and therefore before any provider CLI — is reached. Provider failures
    from the driver propagate untouched: wrapping them would only relabel
    the error the user needs to read.

    The parameter is `backend`, not `name`, so that a future `ScopeField`
    called `name` cannot collide with it.
    """
    spec = get(backend)
    if spec.listing is None:
        raise ListingUnsupported(
            f"the {backend} backend cannot list jobs, so `runplz ps` has nothing to report "
            "for it — this is not the same as having none running."
        )
    accepted = {f.name for f in spec.listing.scope}
    unknown = sorted(set(scope) - accepted)
    if unknown:
        # Silently dropping these would turn a typo into "aws region is
        # required", an error that names the right fix while hiding that a
        # value was passed and thrown away.
        raise TypeError(
            f"{backend} listing does not take {', '.join(unknown)}; "
            f"it accepts {', '.join(sorted(accepted)) or 'no scope'}"
        )
    return load(backend).list_jobs(**spec.listing.resolve(backend, scope))


def provisioning_names() -> tuple:
    """Backends that create the machine themselves."""
    return tuple(n for n, spec in BACKENDS.items() if spec.required_config_attr)


def load(name: str):
    """Import and return a backend module."""
    return importlib.import_module(get(name).module)
