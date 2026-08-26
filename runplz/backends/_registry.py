"""One description of what backends exist and what each one accepts.

Adding a backend used to mean editing five places that had to agree: the
`bind()` validator, the CLI's `choices`, `App._dispatch`'s if-chain, the
`runplz ps` backend tuple, and its own if-chain in `_collect_backend_jobs`.
Any one of them missed and the backend was half-wired — importable but
absent from `--help`, or dispatchable but invisible to `ps`.

Now it is one entry here. Everything else reads from this.

Backends are imported lazily by name, so a missing optional dependency
(`modal`) only surfaces when that backend is actually used, and importing
`runplz` never drags in every cloud driver.
"""

import importlib
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class BackendSpec:
    """What runplz needs to know about one backend."""

    name: str
    module: str
    # App attribute holding this backend's config, when one is *required*
    # before dispatch. Backends with usable defaults leave this None.
    required_config_attr: Optional[str] = None
    # True when `runplz ps` can enumerate this backend's jobs unprompted.
    # SSH can't (no host registry); gcp/aws don't yet — both need a
    # per-target query rather than a global list.
    lists_jobs: bool = False
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
            lists_jobs=True,
            accepts_no_build=True,
        ),
        BackendSpec(
            name="brev",
            module="runplz.backends.brev",
            lists_jobs=True,
            # instance=None is meaningful: ephemeral mode, where runplz
            # creates a box sized to the function and deletes it on exit.
            accepts_instance=True,
        ),
        BackendSpec(name="modal", module="runplz.backends.modal", lists_jobs=True),
        BackendSpec(name="ssh", module="runplz.backends.ssh", accepts_host=True),
        BackendSpec(
            name="gcp",
            module="runplz.backends.gcp",
            required_config_attr="gcp_config",
        ),
        BackendSpec(
            name="aws",
            module="runplz.backends.aws",
            required_config_attr="aws_config",
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
    """Backends `runplz ps` can list without extra input from the user."""
    return tuple(n for n, spec in BACKENDS.items() if spec.lists_jobs)


def provisioning_names() -> tuple:
    """Backends that create the machine themselves."""
    return tuple(n for n, spec in BACKENDS.items() if spec.required_config_attr)


def load(name: str):
    """Import and return a backend module."""
    return importlib.import_module(get(name).module)
