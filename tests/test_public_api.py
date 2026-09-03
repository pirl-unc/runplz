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
PUBLIC_MODULES = (
    SHARED_MODULES
    + (
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
    + ("runplz.backends.listing",)
    + tuple(f"runplz.backends.{name}" for name in ("local", "ssh", "brev", "modal", "gcp", "aws"))
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
    # No allowlist. The legacy shims (_cli, _bootstrap, _ssh_common) alias
    # *public* modules, so they pass this check on their own merits —
    # exempting them by name would only hide a future regression in the three
    # files most likely to reach for a private module.
    offenders = []
    scanned = 0
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text())
        scanned += 1
        dotted = ".".join(path.relative_to(root.parent).with_suffix("").parts)
        package = dotted.rsplit(".", 1)[0]
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
                # The module's real dotted name, not a stem-based guess:
                # `{f"runplz.backends.{stem}", f"runplz.{stem}"}` would have
                # exempted `backends/local.py` importing privates from a
                # hypothetical top-level `runplz.local`.
                own = {dotted}
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

    # The predecessor asserted the directory existed so the check could not
    # pass on an empty glob; keep an equivalent here.
    assert scanned > 15, f"only scanned {scanned} files — the walk is not finding the package"
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
        invocations = text.count('"-m", "runplz._bootstrap"') + text.count("-m runplz._bootstrap")
        assert invocations == expected, (
            f"{rel}: expected {expected} `python -m runplz._bootstrap` "
            f"invocation(s), found {invocations}"
        )
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


def test_brev_forwards_its_moved_names_to_ssh_common():
    """The back-compat shim must be exercised, not just written.

    Nothing else imports a moved name from brev, so a typo in
    `_MOVED_TO_SSH_COMMON`, or a name that never existed on ssh_common,
    would ship green — and the ImportError this shim exists to prevent
    would surface only in a downstream repo.
    """
    import warnings

    from runplz.backends import brev, ssh_common

    assert brev._MOVED_TO_SSH_COMMON, "the compat set should not be empty"
    for old_name, new_name in sorted(brev._MOVED_TO_SSH_COMMON.items()):
        assert hasattr(ssh_common, new_name), (
            f"brev forwards {old_name} to ssh_common.{new_name}, which does not exist"
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert getattr(brev, old_name) is getattr(ssh_common, new_name), old_name
        assert caught, f"{old_name} forwarded without a DeprecationWarning"
        assert issubclass(caught[0].category, DeprecationWarning), old_name

    # The underscore spellings are the ones that were actually importable from
    # brev before 3.20.0; forwarding only the new public names would have been
    # backwards -- covering names nobody could have used while dropping the
    # ones they could.
    for stage in brev._STAGES_RENAMED_IN_3_20:
        assert f"_{stage}" in brev._MOVED_TO_SSH_COMMON, stage

    # dir() must agree with what actually resolves.
    assert set(brev._MOVED_TO_SSH_COMMON) <= set(dir(brev))

    with pytest.raises(AttributeError):
        brev.definitely_not_a_real_name


def test_moved_private_modules_are_gone_on_purpose():
    """`_excludes`/`_selector`/`_logcapture` moved with no compat shim.

    Unlike `_cli`/`_bootstrap`, these were never an invocation path — the
    leading underscore is Python's own "do not import this" marker, so a
    minor bump may drop them. Pinned so the asymmetry reads as a decision
    rather than an oversight.
    """
    for gone in ("runplz._excludes", "runplz._selector", "runplz._logcapture"):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(gone)

    # ...while the two that ARE invocation paths keep working.
    for kept in ("runplz._cli", "runplz._bootstrap"):
        assert importlib.import_module(kept) is not None


def test_legacy_shim_dir_lists_the_target_module():
    """__dir__ on the shims is otherwise never executed by the suite."""
    import runplz._bootstrap
    import runplz._cli

    assert "main" in dir(runplz._cli)
    assert "main" in dir(runplz._bootstrap)


@pytest.mark.parametrize("name", PUBLIC_MODULES)
def test_readme_rows_name_no_dead_exports(name):
    """A backticked symbol in a module's row must still be in its __all__.

    Scoped deliberately: rows are allowed to summarize rather than enumerate,
    so this does not demand every export appear. What it does catch is the
    README naming a symbol that was renamed or removed — the drift that makes
    the table actively wrong rather than merely incomplete.

    (The previous version of this test ended in `or len(row) > 40`, which is
    true of every row in the table, so it asserted nothing at all.)
    """
    import importlib
    from pathlib import Path

    importlib.import_module(name)  # a row for an unimportable module is worse
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()
    table = readme.split("## Public API", 1)[1].split("\n## ", 1)[0]
    row = next((ln for ln in table.splitlines() if f"`{name}`" in ln), None)
    assert row, f"{name} has no row in the Public API table"

    # A row may legitimately backtick: a symbol from another public module
    # (cross-reference), a filename, or one of the documented RUNPLZ_* env
    # vars, which are contract but not Python exports. Anything else that
    # looks like an identifier should resolve somewhere.
    every_export = set()
    for other in PUBLIC_MODULES:
        every_export |= set(importlib.import_module(other).__all__)
    known_modules = set(PUBLIC_MODULES) | {"runplz"}

    stale = [
        token
        for token in row.split("`")[1::2]
        if token not in known_modules
        and token not in every_export
        and not token.startswith("RUNPLZ_")
        and "." not in token
        and "/" not in token
        and token.replace("_", "").isalnum()
    ]
    assert not stale, f"{name} row names {stale}, which no public __all__ exports"
