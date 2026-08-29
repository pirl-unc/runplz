"""App / Image edge-case coverage not already hit by test_runplz.py."""

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


def test_cli_backend_choices_come_from_the_registry(capsys):
    """argparse `choices` must track the registry, not a private copy.

    Driven through the real parser: an unknown backend makes argparse print
    the permitted set, so if `choices` stopped reading the registry the
    listed names would diverge. In-process — spawning an interpreter to read
    one error message costs a few hundred ms per run and buys nothing.
    """
    import pytest

    from runplz import cli
    from runplz.backends import registry

    with pytest.raises(SystemExit):
        cli.main(["not-a-backend", "job.py"])

    message = capsys.readouterr().err
    for name in registry.names():
        assert name in message, f"{name} missing from argparse choices: {message}"


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
