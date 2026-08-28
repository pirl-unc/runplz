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
    # Anchored on this file, not the CWD: a CWD-relative glob finds nothing
    # when pytest runs from anywhere else and the check passes vacuously.
    backends = Path(__file__).resolve().parents[1] / "runplz" / "backends"
    assert backends.is_dir(), backends
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


def test_dispatch_pipeline_is_public():
    """The stages dispatch_to_target runs are API, not internals.

    This reverses an earlier call that kept them underscore-private on the
    grounds that `dispatch_to_target` was the only contract. Two things
    disproved it: `brev.py` has to re-export all eleven across a module
    boundary, and five test modules patch them by name. A seam that other
    modules import and tests patch is an interface; the underscore only hid
    that. They are individually useful too — a backend that wants just
    `build_image` should not have to reach past a private name to get it.
    """
    from runplz.backends import ssh_common

    for stage in (
        "prepare_remote_run",
        "ensure_remote_rsync",
        "check_preconditions",
        "ensure_docker",
        "remote_has_nvidia",
        "build_image",
        "run_container_detached",
        "run_container_mode",
        "run_native",
        "stream_and_wait",
        "fetch_failure_tail",
    ):
        assert hasattr(ssh_common, stage), stage
        assert stage in ssh_common.__all__, f"{stage} is public; declare it in __all__"


def test_all_declares_every_public_name():
    """__all__ must not drift from what the module actually exposes.

    A public name missing from __all__ is an undocumented API; an __all__
    entry with no definition is an ImportError waiting for a `*` import.
    """
    import ast
    from pathlib import Path

    from runplz.backends import ssh_common

    path = Path(ssh_common.__file__)
    tree = ast.parse(path.read_text())
    defined = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not node.name.startswith("_"):
                defined.append(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and not target.id.startswith("_"):
                    defined.append(target.id)

    declared = set(ssh_common.__all__)
    assert not [n for n in defined if n not in declared], "public but undeclared: " + ", ".join(
        n for n in defined if n not in declared
    )
    assert not [n for n in declared if not hasattr(ssh_common, n)], (
        "declared but undefined: " + ", ".join(n for n in declared if not hasattr(ssh_common, n))
    )


def test_old_private_module_paths_are_gone():
    """The unified layer is public — the underscore names should not resolve."""
    for gone in (
        "runplz.backends._cloud",
        "runplz.backends._docker",
        "runplz.backends._registry",
        # _runs was never a wire name or an entry point, so it moved outright.
        "runplz._runs",
    ):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(gone)


def test_legacy_entry_points_still_dispatch():
    """Two underscore paths are kept on purpose, and must keep working.

    `runplz._bootstrap` is a cross-machine wire name: backends emit
    `python -m runplz._bootstrap`, and the container's runplz version is
    independent of the orchestrator's, so the emitted path must stay
    understood by older installs. `runplz._cli` is cheaper insurance for
    anyone who wired up `python -m runplz._cli` directly.
    """
    import runplz._bootstrap
    import runplz._cli
    import runplz.bootstrap
    import runplz.cli

    assert runplz._bootstrap.main is runplz.bootstrap.main
    assert runplz._cli.main is runplz.cli.main


def test_emitted_bootstrap_path_is_the_legacy_one():
    """Guard the wire format itself, not just that both modules import.

    Emitting `runplz.bootstrap` would break a new orchestrator dispatching
    into a container pinned to an older runplz. This test fails if someone
    "finishes" the rename without a deprecation window.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "runplz"
    emitters = [root / "backends" / "local.py", root / "backends" / "ssh_common.py"]
    found = 0
    for path in emitters:
        text = path.read_text()
        assert "runplz.bootstrap" not in text, f"{path.name} emits the new bootstrap path"
        found += text.count("runplz._bootstrap")
    assert found >= 3, f"expected the legacy bootstrap path at 3+ emit sites, found {found}"


def test_ssh_common_compat_shim_still_resolves_public_names():
    """`_ssh_common` predates 3.15.3 and must keep working."""
    from runplz.backends import _ssh_common

    assert _ssh_common.dispatch_to_target is not None
    assert _ssh_common.wait_until_ssh_reachable is not None
