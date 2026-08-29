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

# Every module that is public API, which must match the README's "Public API"
# table (test_readme_documents_every_public_module below pins that). The
# __all__ guards run over all of them: the README states "not in __all__ means
# internal", so a public module with a stale __all__ mislabels its own surface.
PUBLIC_MODULES = SHARED_MODULES + (
    "runplz.app",
    "runplz.config",
    "runplz.image",
    "runplz.cli",
    "runplz.runs",
    "runplz.bootstrap",
    "runplz.excludes",
    "runplz.selector",
    "runplz.logcapture",
)


@pytest.mark.parametrize("name", PUBLIC_MODULES)
def test_shared_module_declares_its_contract(name):
    module = importlib.import_module(name)
    assert getattr(module, "__all__", None), f"{name} must declare __all__"
    assert module.__doc__, f"{name} must say what it is for"


@pytest.mark.parametrize("name", PUBLIC_MODULES)
def test_nothing_exported_is_private_or_missing(name):
    module = importlib.import_module(name)
    for export in module.__all__:
        assert not export.startswith("_"), f"{name}.{export} is exported but private"
        assert hasattr(module, export), f"{name}.{export} is exported but absent"


@pytest.mark.parametrize("name", PUBLIC_MODULES)
def test_exports_are_unique(name):
    module = importlib.import_module(name)
    assert len(module.__all__) == len(set(module.__all__)), name


def test_nothing_imports_a_private_name_or_module_across_a_module_line():
    """No module should reach into another module's underscores.

    Both halves matter, and an earlier version of this check only had the
    first: `from runplz._excludes import DEFAULT_TRANSFER_EXCLUDES` imports a
    *public name* from a *private module*, so a name-only check reports a
    clean tree while three such imports are live. Walk the AST rather than
    regex the source so import style cannot hide one.
    """
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "runplz"
    # Deliberate legacy entry points; their whole job is to alias a public
    # module, and the reasons are in their docstrings.
    allowed = {"_cli.py", "_bootstrap.py", "_ssh_common.py"}

    offenders = []
    for path in sorted(root.rglob("*.py")):
        if path.name in allowed:
            continue
        tree = ast.parse(path.read_text())
        package = ".".join(path.relative_to(root.parent).with_suffix("").parts[:-1])
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                # Resolve relative imports first: `from . import _x` has
                # module=None, and an earlier version treated that as
                # "not a runplz import" and skipped the alias check too.
                module = node.module or ""
                if node.level:
                    module = f"{package}.{module}" if module else package
                if not module.startswith("runplz"):
                    continue
                own = {f"runplz.backends.{path.stem}", f"runplz.{path.stem}"}
                if module not in own and any(part.startswith("_") for part in module.split(".")):
                    offenders.append(f"{path.name}: from {module} import ...")
                if module in own:
                    continue  # a module importing from itself is not a boundary
                for alias in node.names:
                    if alias.name.startswith("_") and not alias.name.startswith("__"):
                        offenders.append(f"{path.name}: from {module} import {alias.name}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.name
                    if name.startswith("runplz") and any(
                        part.startswith("_") for part in name.split(".")
                    ):
                        offenders.append(f"{path.name}: import {name}")

    assert not offenders, "private cross-module references:\n  " + "\n  ".join(offenders)


def test_dispatch_pipeline_is_public():
    """The stages dispatch_to_target runs are API, not internals.

    This reverses an earlier call that kept them underscore-private on the
    grounds that `dispatch_to_target` was the only contract. What disproved
    it: five test modules patch these stages by name, and they are
    individually useful — a backend that wants just `build_image`, or to
    drive `stream_and_wait` itself, should not have to reach past a private
    name to get it. A seam that much code addresses directly is an
    interface; the underscore only hid that.

    (An earlier version of this docstring also cited brev.py re-exporting
    all eleven. That block was dead weight — brev called none of them — so
    it was deleted and the tests now address ssh_common directly.)
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


@pytest.mark.parametrize("name", PUBLIC_MODULES)
def test_all_declares_every_public_name(name):
    """__all__ must not drift from what the module actually exposes.

    A public name missing from __all__ is an undocumented API, and the README
    tells readers to treat it as droppable in a patch release — which is wrong
    and dangerous when another module imports it (as gcp.py does with
    provisioning.ALREADY_EXISTS).
    """
    import ast
    from pathlib import Path

    module = importlib.import_module(name)
    tree = ast.parse(Path(module.__file__).read_text())

    defined = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not node.name.startswith("_"):
                defined.append(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and not target.id.startswith("_"):
                    defined.append(target.id)
        elif isinstance(node, ast.AnnAssign):
            # `X: Final = ...` is just as public as `X = ...`
            if isinstance(node.target, ast.Name) and not node.target.id.startswith("_"):
                defined.append(node.target.id)

    declared = set(module.__all__)
    undeclared = [n for n in defined if n not in declared]
    assert not undeclared, f"{name}: public but undeclared: {', '.join(undeclared)}"
    dangling = [n for n in declared if not hasattr(module, n)]
    assert not dangling, f"{name}: declared but undefined: {', '.join(dangling)}"


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
    """Guard the wire format itself, across every backend that emits it.

    Emitting `runplz.bootstrap` would break a new orchestrator dispatching
    into a container pinned to an older runplz, because runplz is not staged
    to the remote and the container's copy comes from PyPI or the base image.

    Counting bare occurrences is not enough: ssh_common also mentions the
    path in a docstring and in a `pkill -f` fallback, so a loose total stays
    satisfied even after a real emit site is switched. Assert per-file
    instead, and include modal.py — it is a third emit site and an earlier
    version of this test omitted it, which is exactly the hole that lets
    someone "finish" the rename and ship a break.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "runplz"
    # Exact counts of the *invocation* specifically. A `>=` floor on bare
    # occurrences is not enough: ssh_common also names the path in a docstring
    # and in a `pkill -f` fallback, so deleting a real emit still left the
    # total above the floor and shipped a broken wire contract green.
    emitters = {
        "backends/local.py": 1,
        "backends/modal.py": 1,
        "backends/ssh_common.py": 3,
    }
    for rel, expected in emitters.items():
        text = (root / rel).read_text()
        emits = text.count("-m", 0) and text.count("runplz._bootstrap")
        invocations = text.count('"-m", "runplz._bootstrap"') + text.count("-m runplz._bootstrap")
        assert invocations == expected, (
            f"{rel}: expected {expected} `python -m runplz._bootstrap` "
            f"invocation(s), found {invocations}"
        )
        assert emits  # the path must still appear at all
        # Substring-free check that no emit switched to the new path.
        assert '"runplz.bootstrap"' not in text and "-m runplz.bootstrap" not in text, (
            f"{rel} emits the new bootstrap path"
        )


def test_ssh_common_compat_shim_still_resolves_public_names():
    """`_ssh_common` predates 3.15.3 and must keep working."""
    from runplz.backends import _ssh_common

    assert _ssh_common.dispatch_to_target is not None
    assert _ssh_common.wait_until_ssh_reachable is not None


def test_readme_documents_every_public_module():
    """The README's Public API table is the user-facing half of PUBLIC_MODULES.

    Promoting a module out of an underscore is invisible to the people it was
    done for unless the table lists it — and the table is where the README
    says semver coverage begins.
    """
    from pathlib import Path

    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()
    table = readme.split("## Public API", 1)[1].split("\n## ", 1)[0]
    missing = [name for name in PUBLIC_MODULES if f"`{name}`" not in table]
    assert not missing, "not in the README Public API table: " + ", ".join(missing)
