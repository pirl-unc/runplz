"""Coverage for the auto-default entrypoint, min_gpus alias, and
min_gpu_memory-without-gpu cross-backend behavior (3.14.0)."""

from unittest import mock

import pytest

from runplz import App, Image
from runplz._cli import _install_default_entrypoint_or_error
from runplz.backends import brev as brev_backend
from runplz.backends.modal import _modal_default_gpu_for_vram, _modal_gpu_string

# ---------------------------------------------------------------------------
# Default entrypoint synthesis


def _failer(msg):
    raise SystemExit(msg)


def test_default_entrypoint_synthesizes_for_single_function():
    app = App("demo")

    @app.function(image=Image.from_registry("ubuntu:22.04"))
    def train():  # pragma: no cover — body never invoked here
        pass

    _install_default_entrypoint_or_error(app, "/x.py", _failer)
    # The synthesized entrypoint forwards kwargs to .remote(); patch it
    # so we don't actually dispatch.
    with mock.patch.object(app.functions["train"], "remote") as remote_mock:
        app._entrypoint()
    remote_mock.assert_called_once_with()


def test_default_entrypoint_signature_mirrors_function():
    """The CLI parses --flags off the entrypoint's signature, so the
    synthesized one must look like the underlying @app.function."""
    import inspect as _inspect

    app = App("demo")

    @app.function(image=Image.from_registry("ubuntu:22.04"))
    def train(steps: int = 100, dataset: str = "small"):  # pragma: no cover
        pass

    _install_default_entrypoint_or_error(app, "/x.py", _failer)
    sig = _inspect.signature(app._entrypoint)
    assert list(sig.parameters) == ["steps", "dataset"]
    assert sig.parameters["steps"].annotation is int
    assert sig.parameters["dataset"].default == "small"


def test_default_entrypoint_errors_when_no_functions():
    app = App("demo")
    with pytest.raises(SystemExit, match="declares no @app.function"):
        _install_default_entrypoint_or_error(app, "/x.py", _failer)


def test_default_entrypoint_errors_with_helpful_message_on_multiple():
    app = App("demo")

    @app.function(image=Image.from_registry("ubuntu:22.04"))
    def train():  # pragma: no cover
        pass

    @app.function(image=Image.from_registry("ubuntu:22.04"))
    def evaluate():  # pragma: no cover
        pass

    with pytest.raises(SystemExit) as ei:
        _install_default_entrypoint_or_error(app, "/x.py", _failer)
    msg = str(ei.value)
    assert "evaluate" in msg
    assert "train" in msg
    assert "@app.local_entrypoint" in msg


# ---------------------------------------------------------------------------
# min_gpus / num_gpus alias


def test_min_gpus_sets_num_gpus():
    app = App("demo")

    @app.function(image=Image.from_registry("ubuntu:22.04"), gpu="A100", min_gpus=4)
    def f():  # pragma: no cover
        pass

    fn = app.functions["f"]
    assert fn.num_gpus == 4
    assert fn.min_gpus == 4


def test_num_gpus_still_works_for_back_compat():
    app = App("demo")

    @app.function(image=Image.from_registry("ubuntu:22.04"), gpu="A100", num_gpus=2)
    def f():  # pragma: no cover
        pass

    fn = app.functions["f"]
    assert fn.num_gpus == 2
    assert fn.min_gpus == 2


def test_min_gpus_and_num_gpus_conflict_raises():
    app = App("demo")
    with pytest.raises(ValueError, match="conflicting"):

        @app.function(
            image=Image.from_registry("ubuntu:22.04"),
            gpu="A100",
            min_gpus=4,
            num_gpus=8,
        )
        def f():  # pragma: no cover
            pass


# ---------------------------------------------------------------------------
# min_gpu_memory without gpu= now allowed


def test_min_gpu_memory_alone_no_longer_requires_gpu():
    app = App("demo")

    @app.function(image=Image.from_registry("ubuntu:22.04"), min_gpu_memory=24)
    def f():  # pragma: no cover
        pass

    fn = app.functions["f"]
    assert fn.gpu is None
    assert fn.min_gpu_memory == 24


def test_multi_gpu_without_gpu_still_requires_min_gpu_memory():
    """A bare `min_gpus=2` with no gpu= and no min_gpu_memory= is too vague."""
    app = App("demo")
    with pytest.raises(ValueError, match="needs at least min_gpu_memory"):

        @app.function(image=Image.from_registry("ubuntu:22.04"), min_gpus=2)
        def f():  # pragma: no cover
            pass


# ---------------------------------------------------------------------------
# Brev selector: gpu mode triggered by min_gpu_memory alone


def test_brev_searches_gpu_mode_when_only_min_gpu_memory_set():
    app = App("demo")

    @app.function(image=Image.from_registry("ubuntu:22.04"), min_gpu_memory=24)
    def f():  # pragma: no cover
        pass

    fake = mock.Mock(returncode=0, stdout="[]", stderr="")
    with mock.patch("runplz.backends.brev._brev_capture", return_value=fake) as ctx:
        brev_backend._pick_instance_types(app.functions["f"], n=1)
    cmd = ctx.call_args.args[0]
    assert "gpu" in cmd  # mode token
    assert "--min-vram" in cmd
    assert "--gpu-name" not in cmd  # no model pinned


def test_brev_still_searches_cpu_mode_when_no_gpu_constraints():
    app = App("demo")

    @app.function(image=Image.from_registry("ubuntu:22.04"), min_cpu=8)
    def f():  # pragma: no cover
        pass

    fake = mock.Mock(returncode=0, stdout="[]", stderr="")
    with mock.patch("runplz.backends.brev._brev_capture", return_value=fake) as ctx:
        brev_backend._pick_instance_types(app.functions["f"], n=1)
    cmd = ctx.call_args.args[0]
    assert "cpu" in cmd
    assert "--min-vram" not in cmd


# ---------------------------------------------------------------------------
# Modal: derives default GPU model when only min_gpu_memory set


def test_modal_default_gpu_ladder_picks_cheapest_above_threshold():
    assert _modal_default_gpu_for_vram(8) == "T4"
    assert _modal_default_gpu_for_vram(16) == "T4"
    assert _modal_default_gpu_for_vram(20) == "L4"
    assert _modal_default_gpu_for_vram(40) == "A100-40GB"
    assert _modal_default_gpu_for_vram(80) == "A100-80GB"
    assert _modal_default_gpu_for_vram(120) == "H200"


def test_modal_default_gpu_above_max_raises():
    with pytest.raises(ValueError, match="exceeds the largest"):
        _modal_default_gpu_for_vram(200)


def test_modal_gpu_string_returns_none_when_no_gpu_constraints():
    assert _modal_gpu_string(None, None) is None


def test_modal_gpu_string_synthesizes_when_only_min_gpu_memory_set():
    # min 24 GB → L4 base; VRAM suffix appended.
    s = _modal_gpu_string(None, 24)
    assert s.startswith("L4")
    assert "24GB" in s


def test_modal_gpu_string_passes_through_explicit_gpu():
    s = _modal_gpu_string("A100-80GB", None, num_gpus=2)
    assert s == "A100-80GB:2"
