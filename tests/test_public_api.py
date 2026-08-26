"""The shared backend layer is public API, not private helpers.

A backend is expected to be written against these modules, so the contract
is stated in `__all__` rather than inferred from what happens to be
importable — and pinned here so it can't drift back behind underscores.
"""

import importlib

import pytest

SHARED_MODULES = (
    "runplz.backends.ssh_common",
    "runplz.backends.docker",
    "runplz.backends.provisioning",
    "runplz.backends.registry",
)


@pytest.mark.parametrize("name", SHARED_MODULES)
def test_shared_module_declares_its_contract(name):
    module = importlib.import_module(name)
    assert getattr(module, "__all__", None), f"{name} must declare __all__"
    assert module.__doc__, f"{name} must say what it is for"


@pytest.mark.parametrize("name", SHARED_MODULES)
def test_nothing_exported_is_private_or_missing(name):
    module = importlib.import_module(name)
    for export in module.__all__:
        assert not export.startswith("_"), f"{name}.{export} is exported but private"
        assert hasattr(module, export), f"{name}.{export} is exported but absent"


@pytest.mark.parametrize("name", SHARED_MODULES)
def test_exports_are_unique(name):
    module = importlib.import_module(name)
    assert len(module.__all__) == len(set(module.__all__)), name


def test_backends_reach_the_shared_layer_by_its_public_names():
    """No backend should be importing an underscore name across a module line."""
    import re
    from pathlib import Path

    offenders = []
    backends = Path("runplz/backends")
    for path in sorted(backends.glob("*.py")):
        if path.name in ("ssh_common.py", "_ssh_common.py"):
            continue  # ssh_common's own internals; the shim delegates wholesale
        text = path.read_text()
        for match in re.finditer(
            r"from runplz\.backends(?:\.\w+)? import \(([^)]*)\)|"
            r"from runplz\.backends(?:\.\w+)? import ([^\n(]+)",
            text,
        ):
            imported = match.group(1) or match.group(2) or ""
            if "noqa: F401" in match.group(0):
                continue  # deliberate re-export of internals for test patching
            for token in imported.replace("\n", " ").split(","):
                token = token.strip().split(" as ")[0].strip()
                if token.startswith("_"):
                    offenders.append(f"{path.name}: {token}")
    assert not offenders, "private cross-module imports: " + ", ".join(offenders)


def test_dispatch_internals_stay_private():
    """dispatch_to_target is the contract; its helpers are not."""
    from runplz.backends import ssh_common

    for internal in (
        "_prepare_remote_run",
        "_build_image",
        "_run_container_detached",
        "_stream_and_wait",
        "_run_native",
        "_check_preconditions",
    ):
        assert hasattr(ssh_common, internal), internal
        assert internal not in ssh_common.__all__, (
            f"{internal} is an implementation detail of dispatch_to_target; "
            f"a backend should never need it"
        )


def test_old_private_module_paths_are_gone():
    """The unified layer is public — the underscore names should not resolve."""
    for gone in (
        "runplz.backends._cloud",
        "runplz.backends._docker",
        "runplz.backends._registry",
    ):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(gone)


def test_ssh_common_compat_shim_still_resolves_public_names():
    """`_ssh_common` predates 3.15.3 and must keep working."""
    from runplz.backends import _ssh_common

    assert _ssh_common.dispatch_to_target is not None
    assert _ssh_common.wait_until_ssh_reachable is not None
