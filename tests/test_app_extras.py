"""App / Image edge-case coverage not already hit by test_runplz.py."""

from pathlib import Path

import pytest

from runplz import App, BrevConfig, Image, ImageOp, ModalConfig
from runplz.app import _ensure_json_safe


def test_app_defaults_build_own_configs():
    app = App("x")
    assert isinstance(app.brev_config, BrevConfig)
    assert isinstance(app.modal_config, ModalConfig)


def test_app_dispatch_rejects_unknown_backend():
    app = App("x")

    @app.function(image=Image.from_registry("ubuntu:22.04"))
    def fn():
        pass

    app._backend = "k8s"  # not a registered backend
    # Same message bind() gives — one registry, one wording.
    with pytest.raises(ValueError, match="backend must be one of"):
        fn.remote()


def test_ensure_json_safe_rejects_closures_and_objects():
    class Box:
        pass

    with pytest.raises(TypeError, match="JSON-serializable"):
        _ensure_json_safe((Box(),), {})
    with pytest.raises(TypeError, match="JSON-serializable"):
        _ensure_json_safe((), {"x": Box()})


def test_image_op_kwargs_dict_roundtrip():
    op = ImageOp(kind="pip_install", args=("x",), kwargs=(("index_url", "u"),))
    assert op.kwargs_dict() == {"index_url": "u"}


def test_image_run_commands_appends_raw_run_lines():
    img = Image.from_registry("ubuntu:22.04").run_commands(
        "echo hi", "pip install requests && echo done"
    )
    df = img.render_dockerfile()
    assert "RUN echo hi" in df
    assert "RUN pip install requests && echo done" in df


def test_image_pip_install_with_index_url_in_dockerfile():
    img = Image.from_registry("ubuntu:22.04").pip_install(
        "torch", index_url="https://download.pytorch.org/whl/cu121"
    )
    df = img.render_dockerfile()
    assert "--index-url" in df
    assert "https://download.pytorch.org/whl/cu121" in df


def test_image_render_dockerfile_empty_ops_is_ok():
    df = Image.from_registry("ubuntu:22.04").render_dockerfile()
    assert df.startswith("FROM ubuntu:22.04")


def test_image_render_dockerfile_with_context_still_works():
    # context is only honored by resolve(); render is base + ops only.
    img = Image.from_dockerfile("Dockerfile", context="subdir")
    assert img.context == "subdir"


def test_image_resolve_requires_dockerfile(tmp_path):
    img = Image.from_registry("ubuntu:22.04")
    with pytest.raises(ValueError, match="from_dockerfile"):
        img.resolve(tmp_path)


def test_image_resolve_raises_if_dockerfile_missing(tmp_path):
    img = Image.from_dockerfile("no-such-file")
    with pytest.raises(FileNotFoundError):
        img.resolve(tmp_path)


def test_image_pip_install_local_dir_non_editable():
    img = Image.from_registry("ubuntu:22.04").pip_install_local_dir(".", editable=False)
    df = img.render_dockerfile()
    # Non-editable install uses the argv-form without "-e".
    assert '"pip", "install", "--no-cache-dir", "/workspace"' in df
    assert '"-e"' not in df


# ---------------------------------------------------------------------------
# the backend registry is the single source of truth


def test_registry_is_the_only_backend_list():
    """These used to be three hand-maintained tuples that had to agree.

    `app._VALID_BACKENDS` and `cli._PS_BACKENDS` were pure aliases of the
    registry, so they are gone: the CLI reads `registry.names()` and
    `registry.ps_names()` directly and there is nothing left to drift.
    """
    import runplz.app
    import runplz.cli
    from runplz.backends import registry

    assert not hasattr(runplz.app, "_VALID_BACKENDS")
    assert not hasattr(runplz.cli, "_PS_BACKENDS")
    assert set(registry.ps_names()) <= set(registry.names())
    assert set(registry.provisioning_names()) <= set(registry.names())


def test_cli_backend_choices_equal_the_registry():
    """argparse `choices` must EQUAL the registry, not merely contain it.

    The deleted `_VALID_BACKENDS` assertion was two-directional. A one-way
    "every registry name appears in the error message" check passes for
    `choices=list(registry.names()) + ["bogus"]`, so assert on the parser's
    actual choices instead of a formatted message.
    """
    import argparse
    from unittest import mock

    from runplz import cli
    from runplz.backends import registry

    seen = {}
    real_add = argparse.ArgumentParser.add_argument

    def capture(self, *args, **kwargs):
        if args and args[0] in ("backend",) and "choices" in kwargs:
            seen[args[0]] = kwargs["choices"]
        return real_add(self, *args, **kwargs)

    with mock.patch.object(argparse.ArgumentParser, "add_argument", capture):
        with pytest.raises(SystemExit):
            cli.main(["not-a-backend", "job.py"])

    assert "backend" in seen, "the backend argument declared no choices"
    assert list(seen["backend"]) == list(registry.names())


def test_ps_backend_choices_equal_the_registry():
    """Same two-directional guarantee for `runplz ps`, which lost _PS_BACKENDS."""
    import argparse
    from unittest import mock

    from runplz import cli
    from runplz.backends import registry

    seen = {}
    real_add = argparse.ArgumentParser.add_argument

    def capture(self, *args, **kwargs):
        if args and args[0] == "backend" and "choices" in kwargs:
            seen["backend"] = kwargs["choices"]
        return real_add(self, *args, **kwargs)

    with mock.patch.object(argparse.ArgumentParser, "add_argument", capture):
        with pytest.raises(SystemExit):
            cli.main(["ps", "not-a-backend"])

    assert list(seen["backend"]) == list(registry.ps_names())


def test_every_registered_backend_is_importable_and_runnable():
    from runplz.backends import registry

    for name in registry.names():
        if name == "modal":
            continue  # optional extra; may not be installed
        module = registry.load(name)
        assert callable(module.run), name


def test_every_ps_backend_exposes_list_jobs():
    from runplz.backends import registry

    for name in registry.ps_names():
        if name == "modal":
            continue
        assert callable(registry.load(name).list_jobs), name


def test_registry_rejects_an_unknown_backend():
    from runplz.backends import registry

    with pytest.raises(ValueError, match="backend must be one of"):
        registry.get("k8s")


# ---------------------------------------------------------------------------
# repo_root became public in 3.20.0, which makes its assignment semantics API


def _app_with_fn(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "job.py").write_text("# job\n")
    app = App("v")

    @app.function(image=Image.from_registry("ubuntu:22.04"))
    def train():
        pass

    train.module_file = str(repo / "job.py")
    return app, repo.resolve()


def test_repo_root_assignment_is_coerced_to_an_absolute_path(tmp_path):
    """A str or relative path must fail here, not deep inside dispatch.

    Unresolved, `repo / outputs_dir` raises TypeError and `relative_to(repo)`
    raises ValueError — both after a paid box has been provisioned.
    """
    app, _ = _app_with_fn(tmp_path)
    app.repo_root = str(tmp_path)
    assert isinstance(app.repo_root, Path)
    assert app.repo_root.is_absolute()


def test_bind_repo_root_argument_does_not_leak_into_later_binds(tmp_path):
    """A per-call argument applies to that call only."""
    app, repo = _app_with_fn(tmp_path)
    other = (tmp_path / "other").resolve()
    other.mkdir()

    app.bind("ssh", host="h", repo_root=other)
    assert app.repo_root == other

    app.bind("local")
    assert app.repo_root == repo, "a bind()-scoped repo_root leaked into the next bind"


def test_assigned_repo_root_survives_bind(tmp_path):
    """Assigning the public attribute is a choice; bind() must not discard it."""
    app, _ = _app_with_fn(tmp_path)
    other = (tmp_path / "other").resolve()
    other.mkdir()

    app.repo_root = other
    app.bind("local")
    assert app.repo_root == other


def test_dispatch_refuses_before_provisioning_when_repo_root_is_unset(tmp_path):
    """The check belongs above the backends, not inside one.

    A provisioning backend would otherwise create the box and wait out the
    ssh timeout before discovering there is nothing to stage to it.
    """
    app, _ = _app_with_fn(tmp_path)
    app.bind("local")
    app._repo_root = None
    fn = next(iter(app.functions.values()))
    with pytest.raises(RuntimeError, match="repo_root"):
        app._dispatch(fn, [], {})


def test_repo_root_precedence_matrix(tmp_path):
    """All three sources, in every order that previously went wrong.

    This got fixed twice and broke twice: first a bind()-scoped repo_root
    leaked into the next bind, then a per-call argument silently destroyed a
    standing `app.repo_root = X` assignment. The whole matrix is pinned here
    rather than one branch at a time, since each earlier fix passed its own
    single-branch test.
    """
    app, repo = _app_with_fn(tmp_path)
    chosen = (tmp_path / "chosen").resolve()
    percall = (tmp_path / "percall").resolve()
    chosen.mkdir()
    percall.mkdir()

    # bind argument wins for its own call...
    app.repo_root = chosen
    app.bind("ssh", host="h", repo_root=percall)
    assert app.repo_root == percall

    # ...but does not outlive it, and does not erase the standing choice.
    app.bind("local")
    assert app.repo_root == chosen

    # with no standing choice, a bind argument still does not leak.
    app2, repo2 = _app_with_fn(tmp_path / "b")
    app2.bind("ssh", host="h", repo_root=percall)
    app2.bind("local")
    assert app2.repo_root == repo2


@pytest.mark.parametrize(
    "bad, reason",
    [("", "empty"), ("   ", "whitespace"), ("/definitely/not/here", "nonexistent")],
)
def test_repo_root_rejects_unusable_values(tmp_path, bad, reason):
    """`repo_root = ''` used to resolve to the CWD and stage it to a remote."""
    app, _ = _app_with_fn(tmp_path)
    with pytest.raises(ValueError, match="repo_root"):
        app.repo_root = bad


def test_repo_root_rejects_a_file(tmp_path):
    app, repo = _app_with_fn(tmp_path)
    with pytest.raises(ValueError, match="existing directory"):
        app.repo_root = repo / "job.py"


def test_legacy_entrypoint_attribute_still_installs_a_driver(tmp_path):
    """`app._entrypoint = fn` was the pre-3.20 programmatic path.

    Without the alias the assignment lands on a fresh attribute, `entrypoint`
    stays None, and the CLI synthesizes a default from the single
    @app.function and dispatches *that* — a different job, silently.
    """
    import warnings

    app, _ = _app_with_fn(tmp_path)

    def driver():
        pass

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        app._entrypoint = driver

    assert app.entrypoint is driver
    assert app._entrypoint is driver
    assert caught and issubclass(caught[0].category, DeprecationWarning)
