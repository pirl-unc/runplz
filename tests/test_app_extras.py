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
    """Same two-directional guarantee for `runplz ps`, which lost _PS_BACKENDS.

    The choices are `listable_names()`, not `ps_names()`: ssh can be listed,
    it just cannot be listed *unprompted*, and rejecting `runplz ps ssh` as an
    invalid choice hid a capability the backend has rather than reporting the
    host it still needs.
    """
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

    assert list(seen["backend"]) == list(registry.listable_names())


def test_every_registered_backend_is_importable_and_runnable():
    from runplz.backends import registry

    for name in registry.names():
        if name == "modal":
            continue  # optional extra; may not be installed
        module = registry.load(name)
        assert callable(module.run), name


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
    app.repo_root = None
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


def test_dispatch_rejects_a_repo_root_that_does_not_contain_the_script(tmp_path):
    """`relative_to` would otherwise raise after the box is paid for.

    The backends locate the script *inside* repo_root to find it on the
    remote. When it is not underneath, the failure used to land in
    dispatch_to_target — after provision(), the ssh wait, and rsync_up.
    """
    app, _ = _app_with_fn(tmp_path)
    elsewhere = (tmp_path / "elsewhere").resolve()
    elsewhere.mkdir()
    app.repo_root = elsewhere
    app.bind("local")

    fn = next(iter(app.functions.values()))
    with pytest.raises(ValueError, match="does not contain"):
        app._dispatch(fn, [], {})


def test_legacy_repo_root_attribute_validates_like_the_new_one(tmp_path):
    """`app._repo_root = X` used to write the field and skip every check."""
    import warnings

    app, repo = _app_with_fn(tmp_path)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        app._repo_root = repo
    assert app.repo_root == repo
    assert caught and issubclass(caught[0].category, DeprecationWarning)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="existing directory"):
            app._repo_root = "/definitely/not/here"


def test_bind_without_functions_is_fine_when_repo_root_is_supplied(tmp_path):
    """The 'declare a function so we can locate the repo root' error only
    applies when the repo root actually has to be inferred."""
    target = (tmp_path / "given").resolve()
    target.mkdir()

    app = App("x")
    app.bind("local", repo_root=target)
    assert app.repo_root == target

    with pytest.raises(RuntimeError, match="at least one"):
        App("y").bind("local")


# ---------------------------------------------------------------------------
# Fail on the laptop, not after the box is paid for (#87, #88)


@pytest.mark.parametrize(
    "key",
    ["MY-VAR", "2FAST", "HAS SPACE", "a;touch /tmp/x", "with.dot", "", "$HOME"],
)
def test_env_key_must_be_a_shell_identifier(key):
    """`export {key}=...` parses but fails at runtime for a non-identifier.

    `sh -n` does not catch it — the syntax is valid — and the remote runs
    under `set -euo pipefail`, so it aborts the job after provisioning with
    an error naming neither runplz nor the key.
    """
    app = App("t")
    with pytest.raises(ValueError, match="not a valid shell identifier"):

        @app.function(image=Image.from_registry("ubuntu:22.04"), env={key: "x"})
        def fn():
            pass


@pytest.mark.parametrize("key", ["OK", "OK_VAR", "_LEADING", "x2", "A1_b2"])
def test_valid_env_keys_are_accepted(key):
    app = App("t")

    @app.function(image=Image.from_registry("ubuntu:22.04"), env={key: "x"})
    def fn():
        pass

    assert fn.env == {key: "x"}


def test_env_must_be_a_dict():
    app = App("t")
    with pytest.raises(ValueError, match="env must be a dict"):

        @app.function(image=Image.from_registry("ubuntu:22.04"), env=[("A", "b")])
        def fn():
            pass


def test_every_accepted_env_key_survives_a_real_shell():
    """The regex is only right if the shell agrees with it.

    Renders the same `export` line the backends emit and runs it, so this
    fails if the accepted character set and bash's idea of an identifier
    ever diverge.
    """
    import shlex
    import subprocess

    app = App("t")

    @app.function(
        image=Image.from_registry("ubuntu:22.04"),
        env={"OK": "1", "OK_VAR": "a b", "_LEAD": "x'y", "n2": "$NOPE"},
    )
    def fn():
        pass

    exports = " ".join(f"export {k}={shlex.quote(str(v))};" for k, v in fn.env.items())
    proc = subprocess.run(
        ["bash", "-c", f'set -euo pipefail; {exports} echo OK; echo "$OK_VAR"'],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "a b" in proc.stdout


def test_second_local_entrypoint_is_rejected():
    """Last-wins left the first driver unreachable with no output at all."""
    app = App("t")

    @app.local_entrypoint()
    def first():
        pass

    with pytest.raises(ValueError, match="already has an @app.local_entrypoint"):

        @app.local_entrypoint()
        def second():
            pass

    assert app.entrypoint is first, "the first entrypoint must survive the rejection"


def test_one_local_entrypoint_still_works():
    app = App("t")

    @app.local_entrypoint()
    def only():
        pass

    assert app.entrypoint is only
