"""Offline tests for runplz.

Covers the pieces we can test without actually provisioning cloud
resources: DSL rendering, config validation, instance-picker logic,
CLI arg parsing. Backends that drive real SSH / docker / modal are
exercised by examples/simple_job.py against the three backends — that's
the closest thing we have to an integration test.
"""

import json
import types
from unittest import mock

import pytest

from runplz import App, BrevConfig, Image, ModalConfig
from runplz.backends.brev import (
    _brev_gpu_name,
    _pick_instance_type,
    _render_ops_script,
)

# ---- Image DSL rendering --------------------------------------------------


def _sample_image():
    return (
        Image.from_registry("pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime")
        .apt_install("bzip2", "rsync")
        .pip_install("pandas>=2.0", "scikit-learn")
        .pip_install_local_dir(".", editable=True)
    )


def test_image_from_registry_records_base():
    img = Image.from_registry("ubuntu:22.04")
    assert img.base == "ubuntu:22.04"
    assert img.dockerfile is None
    assert img.ops == ()


def test_image_layer_ops_chain_is_immutable():
    a = Image.from_registry("ubuntu:22.04")
    b = a.apt_install("curl")
    assert a.ops == ()
    assert len(b.ops) == 1
    assert b.ops[0].kind == "apt_install"
    assert b.ops[0].args == ("curl",)


def test_image_cant_layer_on_from_dockerfile():
    img = Image.from_dockerfile("docker/Dockerfile.train")
    with pytest.raises(ValueError, match="from_dockerfile"):
        img.pip_install("requests")


def test_render_dockerfile_uses_exec_form_for_pip():
    # Exec form is what prevents `pandas>=2.0` from being parsed as a
    # shell redirect. Regression test for that bug.
    df = _sample_image().render_dockerfile()
    assert 'RUN ["pip", "install", "--no-cache-dir", "pandas>=2.0"' in df
    assert "COPY . /workspace" in df
    assert 'RUN ["pip", "install", "--no-cache-dir", "-e", "/workspace"]' in df
    assert "FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime" in df


def test_render_dockerfile_requires_from_registry():
    img = Image.from_dockerfile("docker/Dockerfile.train")
    with pytest.raises(ValueError, match="from_registry"):
        img.render_dockerfile()


# ---- Brev ops-script rendering --------------------------------------------


def test_render_ops_script_uses_shlex_quote_for_packages():
    script = _render_ops_script(_sample_image())
    # Each op becomes a semicolon-joined bash line. Version specifiers
    # with shell metacharacters must be single-quoted by shlex.quote.
    assert "'pandas>=2.0'" in script
    assert "bzip2 rsync" in script
    assert 'pip install --quiet -e "$HOME/runplz-repo"' in script


# `_render_ops_script` is never reached with a Dockerfile image — the
# Function-level validator in runplz.app rejects that combo at decoration
# time (see test_function_rejects_dockerfile_image_on_container_mode below).


# ---- BrevConfig validation (at construction time) ------------------------


def test_brev_default_constructs_ok_and_uses_container_mode():
    cfg = BrevConfig()  # does not raise
    assert cfg.mode == "container"


def test_brev_rejects_unknown_mode():
    with pytest.raises(ValueError, match="mode must be"):
        BrevConfig(mode="kubernetes")


def test_brev_rejects_container_plus_use_docker_false():
    with pytest.raises(ValueError, match="contradictory"):
        BrevConfig(mode="container", use_docker=False)


def test_brev_allows_vm_plus_use_docker_false():
    # Legacy escape hatch for mode=vm.
    BrevConfig(mode="vm", use_docker=False)


def test_brev_rejects_empty_instance_type():
    with pytest.raises(ValueError, match="non-empty"):
        BrevConfig(instance_type="   ")


def test_brev_default_auto_create_is_false():
    # 3.3: flipped from True. A typoed --instance name must not silently
    # provision a new billed box — users opt in to auto-create explicitly.
    assert BrevConfig().auto_create_instances is False


def test_brev_default_on_finish_is_stop():
    assert BrevConfig().on_finish == "stop"


def test_brev_default_max_runtime_seconds_is_none():
    # Off by default. Users opt in when they want a billing kill-switch.
    assert BrevConfig().max_runtime_seconds is None


def test_brev_rejects_non_positive_max_runtime_seconds():
    with pytest.raises(ValueError, match="max_runtime_seconds must be a positive int"):
        BrevConfig(max_runtime_seconds=0)
    with pytest.raises(ValueError, match="max_runtime_seconds must be a positive int"):
        BrevConfig(max_runtime_seconds=-1)


def test_brev_accepts_positive_max_runtime_seconds():
    BrevConfig(max_runtime_seconds=1)
    BrevConfig(max_runtime_seconds=3600)


def test_brev_default_ssh_ready_wait_seconds_is_30_minutes():
    """3.7.2: bumped from 1200s (20min) → 1800s (30min) to cover
    8×A100 Denvr/OCI cold boots."""
    assert BrevConfig().ssh_ready_wait_seconds == 1800


def test_brev_rejects_non_positive_ssh_ready_wait_seconds():
    with pytest.raises(ValueError, match="ssh_ready_wait_seconds must be a positive int"):
        BrevConfig(ssh_ready_wait_seconds=0)
    with pytest.raises(ValueError, match="ssh_ready_wait_seconds must be a positive int"):
        BrevConfig(ssh_ready_wait_seconds=-1)


def test_brev_accepts_custom_ssh_ready_wait_seconds():
    BrevConfig(ssh_ready_wait_seconds=2400)  # 40 min for exotic shapes


def test_brev_rejects_unknown_on_finish():
    with pytest.raises(ValueError, match="on_finish must be one of"):
        BrevConfig(on_finish="terminate")


def test_brev_accepts_all_documented_on_finish_values():
    for value in ("stop", "delete", "leave"):
        BrevConfig(on_finish=value)  # does not raise


def test_modal_config_constructs_as_noop():
    # ModalConfig has no fields today; just make sure the slot still exists.
    ModalConfig()


# ---- GPU label translation (Modal → Brev) ---------------------------------


@pytest.mark.parametrize(
    "modal_label, expected_brev_name",
    [
        ("T4", "T4"),
        ("L4", "L4"),
        ("A100-40GB", "A100"),
        ("A100-80GB", "A100"),
        ("H100", "H100"),
        ("a100-40gb", "A100"),  # accept lowercased too
    ],
)
def test_brev_gpu_name_strips_vram_suffix(modal_label, expected_brev_name):
    assert _brev_gpu_name(modal_label) == expected_brev_name


# ---- Instance picker → `brev search` -------------------------------------


def _fn_with(
    gpu=None,
    min_cpu=None,
    min_memory=None,
    min_gpu_memory=None,
    min_disk=None,
    num_gpus=1,
):
    return types.SimpleNamespace(
        gpu=gpu,
        min_cpu=min_cpu,
        min_memory=min_memory,
        min_gpu_memory=min_gpu_memory,
        min_disk=min_disk,
        num_gpus=num_gpus,
    )


def test_pick_instance_type_builds_correct_search_cmd():
    captured = {}

    def fake_run(cmd, *a, **kw):
        captured["cmd"] = cmd
        return mock.Mock(
            returncode=0,
            stdout=json.dumps([{"type": "n1-highmem-4:nvidia-tesla-t4:1"}]),
        )

    fn = _fn_with(gpu="T4", min_cpu=4, min_memory=26, min_gpu_memory=16, min_disk=100)
    with mock.patch("runplz.backends.brev.subprocess.run", fake_run):
        result = _pick_instance_type(fn)
    assert result == "n1-highmem-4:nvidia-tesla-t4:1"
    cmd = captured["cmd"]
    assert cmd[:3] == ["brev", "search", "gpu"]
    assert "--gpu-name" in cmd and "T4" in cmd
    assert "--min-vcpu" in cmd and "4" in cmd
    assert "--min-ram" in cmd and "26" in cmd
    assert "--min-vram" in cmd and "16" in cmd
    assert "--min-disk" in cmd and "100" in cmd


def test_pick_instance_type_threads_num_gpus_through():
    """3.6: Function.num_gpus > 1 maps to `brev search --min-gpus N`."""
    captured = {}

    def fake_run(cmd, *a, **kw):
        captured["cmd"] = cmd
        return mock.Mock(returncode=0, stdout=json.dumps([{"type": "a100-x4"}]))

    fn = _fn_with(gpu="A100", num_gpus=4)
    with mock.patch("runplz.backends.brev.subprocess.run", fake_run):
        _pick_instance_type(fn)
    cmd = captured["cmd"]
    assert "--min-gpus" in cmd and "4" in cmd


def test_pick_instance_type_omits_min_gpus_when_only_one():
    """num_gpus=1 is the default — don't noise up `brev search` with it."""
    captured = {}

    def fake_run(cmd, *a, **kw):
        captured["cmd"] = cmd
        return mock.Mock(returncode=0, stdout=json.dumps([{"type": "t4-x1"}]))

    fn = _fn_with(gpu="T4", num_gpus=1)
    with mock.patch("runplz.backends.brev.subprocess.run", fake_run):
        _pick_instance_type(fn)
    assert "--min-gpus" not in captured["cmd"]


def test_pick_instance_type_cpu_when_no_gpu():
    def fake_run(cmd, *a, **kw):
        # Verify we called `search cpu` not `search gpu` when gpu=None.
        assert cmd[:3] == ["brev", "search", "cpu"]
        return mock.Mock(returncode=0, stdout=json.dumps([{"type": "n2d-highmem-2"}]))

    with mock.patch("runplz.backends.brev.subprocess.run", fake_run):
        result = _pick_instance_type(_fn_with(min_memory=16))
    assert result == "n2d-highmem-2"


# ---- Selector tiebreaker integration with brev search -------------------


def test_pick_instance_type_prefers_faster_start_within_5pct():
    """Brev returns two candidates at $1.00 and $1.02 (both in the 5%
    tolerance band) with different eta_seconds. Selector should pick
    the faster-to-start one."""
    brev_rows = [
        # Brev sorts by price, so the "cheapest" row comes first. Our
        # selector sees both and uses eta_seconds as tiebreaker.
        {"type": "t4-slow-region", "hourly_price": 1.00, "eta_seconds": 300},
        {"type": "t4-fast-region", "hourly_price": 1.02, "eta_seconds": 20},
    ]

    def fake_run(cmd, *a, **kw):
        return mock.Mock(returncode=0, stdout=json.dumps(brev_rows))

    with mock.patch("runplz.backends.brev.subprocess.run", fake_run):
        result = _pick_instance_type(_fn_with(gpu="T4", min_memory=16))
    assert result == "t4-fast-region"


def test_pick_instance_type_cost_wins_outside_5pct():
    """$1.00 vs $1.15 is 15%, well outside the band — cheapest wins
    regardless of availability."""
    brev_rows = [
        {"type": "cheap-slow", "hourly_price": 1.00, "eta_seconds": 500},
        {"type": "expensive-fast", "hourly_price": 1.15, "eta_seconds": 5},
    ]

    def fake_run(cmd, *a, **kw):
        return mock.Mock(returncode=0, stdout=json.dumps(brev_rows))

    with mock.patch("runplz.backends.brev.subprocess.run", fake_run):
        result = _pick_instance_type(_fn_with(gpu="T4"))
    assert result == "cheap-slow"


def test_pick_instance_type_legacy_shape_no_price_field_falls_back_to_first():
    """Older/minimal `brev search --json` output: just a `type` and nothing
    else. Must still return the first row — never regress the pre-selector
    behavior just because the schema is thin."""
    brev_rows = [{"type": "minimal-row-1"}, {"type": "minimal-row-2"}]

    def fake_run(cmd, *a, **kw):
        return mock.Mock(returncode=0, stdout=json.dumps(brev_rows))

    with mock.patch("runplz.backends.brev.subprocess.run", fake_run):
        result = _pick_instance_type(_fn_with(gpu="T4"))
    assert result == "minimal-row-1"


def test_pick_instance_type_alternate_price_key_name():
    """Brev schema drift: accept `price`, `usd_per_hour`, etc. as price
    fields. Here we ship the rare `estimated_hourly` to confirm the fallback
    chain works end-to-end."""
    brev_rows = [
        {"type": "slower", "estimated_hourly": 0.60, "eta_seconds": 200},
        {"type": "faster", "estimated_hourly": 0.62, "eta_seconds": 10},
    ]

    def fake_run(cmd, *a, **kw):
        return mock.Mock(returncode=0, stdout=json.dumps(brev_rows))

    with mock.patch("runplz.backends.brev.subprocess.run", fake_run):
        result = _pick_instance_type(_fn_with(gpu="T4"))
    assert result == "faster"


def test_pick_instance_type_returns_none_on_no_match():
    with mock.patch(
        "runplz.backends.brev.subprocess.run",
        return_value=mock.Mock(returncode=0, stdout="[]"),
    ):
        assert _pick_instance_type(_fn_with(gpu="H100", min_memory=999999)) is None


# ---- App / @app.function --------------------------------------------------


def test_function_decorator_records_resource_requests():
    app = App("t")

    @app.function(
        image=_sample_image(),
        gpu="T4",
        min_cpu=4,
        min_memory=26,
        min_gpu_memory=16,
        min_disk=100,
        timeout=3600,
        env={"FOO": "bar"},
    )
    def train():
        pass

    assert train.gpu == "T4"
    assert train.min_cpu == 4
    assert train.min_memory == 26
    assert train.min_gpu_memory == 16
    assert train.min_disk == 100
    assert train.timeout == 3600
    assert train.env == {"FOO": "bar"}


def test_function_plain_call_raises_with_helpful_message():
    app = App("t")

    @app.function(image=_sample_image(), gpu="T4")
    def train():
        return "ran"

    with pytest.raises(RuntimeError, match="\\.local|\\.remote"):
        train()
    assert train.local() == "ran"  # local() still works


def test_remote_requires_backend_to_be_selected():
    app = App("t")

    @app.function(image=_sample_image(), gpu="T4")
    def train():
        pass

    with pytest.raises(RuntimeError, match="no backend is selected"):
        train.remote()


def test_remote_rejects_non_json_args():
    app = App("t")

    @app.function(image=_sample_image(), gpu="T4")
    def train(x):
        return x

    app._backend = "local"
    with pytest.raises(TypeError, match="JSON-serializable"):
        train.remote(object())


# ---- CLI entry ------------------------------------------------------------


def test_cli_allows_brev_without_instance_for_ephemeral_mode(tmp_path):
    """Used to require --instance. 3.6 flipped this: omitting --instance
    triggers ephemeral mode (runplz auto-creates a box and deletes it on
    exit). The CLI must stop rejecting the missing flag."""
    import textwrap

    from runplz import _cli

    script = tmp_path / "job.py"
    script.write_text(
        textwrap.dedent(
            """
            from runplz import App, Image

            app = App("ephemeral-test")
            image = Image.from_registry("ubuntu:22.04")

            @app.function(image=image, gpu="T4")
            def fn():
                pass

            @app.local_entrypoint()
            def main():
                fn.remote()
            """
        )
    )

    captured = {}
    with mock.patch(
        "runplz.backends.brev.run",
        lambda app, function, args, kwargs, **kw: captured.update({"kw": kw}),
    ):
        _cli.main(["brev", str(script)])
    # instance is threaded through as None; brev.run() expands it to an
    # ephemeral name on its own.
    assert captured["kw"]["instance"] is None


# ---- App.bind() — pure-Python invocation ---------------------------------


def test_bind_local_sets_backend_and_kwargs():
    app = App("t")

    @app.function(image=_sample_image())
    def train():
        pass

    app.bind("local")
    assert app._backend == "local"
    assert app._backend_kwargs["outputs_dir"] == "out"
    assert "build" not in app._backend_kwargs  # default build=True, no override


def test_bind_local_with_no_build_flag():
    app = App("t")

    @app.function(image=_sample_image())
    def train():
        pass

    app.bind("local", build=False)
    assert app._backend_kwargs["build"] is False


def test_bind_brev_accepts_none_instance_for_ephemeral_mode():
    """3.6: instance=None on brev → ephemeral mode. App.bind() threads it
    through as-is; brev.run() expands it into a generated name + forced
    auto_create_instances=True + on_finish="delete"."""
    app = App("t")

    @app.function(image=_sample_image(), gpu="T4")
    def train():
        pass

    app.bind("brev")  # no instance= — ephemeral
    assert app._backend == "brev"
    assert app._backend_kwargs["instance"] is None

    app.bind("brev", instance="my-box")
    assert app._backend_kwargs["instance"] == "my-box"


def test_bind_rejects_unknown_backend():
    app = App("t")

    @app.function(image=_sample_image())
    def train():
        pass

    with pytest.raises(ValueError, match="must be 'local', 'brev', 'modal', or 'ssh'"):
        app.bind("k8s")


def test_bind_requires_a_function_to_locate_repo_root():
    app = App("t")
    with pytest.raises(RuntimeError, match="at least one @app.function"):
        app.bind("local")


def test_bind_rejects_instance_on_non_brev_backend():
    app = App("t")

    @app.function(image=_sample_image())
    def train():
        pass

    with pytest.raises(ValueError, match="only applies to backend='brev'"):
        app.bind("local", instance="stray-box")


def test_bind_rejects_no_build_on_non_local_backend():
    app = App("t")

    @app.function(image=_sample_image(), gpu="T4")
    def train():
        pass

    with pytest.raises(ValueError, match="build=False only applies to backend='local'"):
        app.bind("brev", instance="b", build=False)


def test_bind_returns_self_for_chaining():
    app = App("t")

    @app.function(image=_sample_image())
    def train():
        pass

    assert app.bind("local") is app


def test_bind_rejects_empty_outputs_dir():
    app = App("t")

    @app.function(image=_sample_image())
    def train():
        pass

    with pytest.raises(ValueError, match="outputs_dir"):
        app.bind("local", outputs_dir="   ")


def test_bind_threads_outputs_dir():
    app = App("t")

    @app.function(image=_sample_image())
    def train():
        pass

    app.bind("local", outputs_dir="custom/out")
    assert app._backend_kwargs["outputs_dir"] == "custom/out"


# ---- @app.function() resource validation --------------------------------


def test_function_rejects_non_positive_min_cpu():
    app = App("t")
    with pytest.raises(ValueError, match="min_cpu must be > 0"):

        @app.function(image=_sample_image(), min_cpu=0)
        def train():
            pass


def test_function_rejects_non_positive_timeout():
    app = App("t")
    with pytest.raises(ValueError, match="timeout must be a positive int"):

        @app.function(image=_sample_image(), timeout=0)
        def train():
            pass


def test_function_rejects_min_gpu_memory_without_gpu():
    app = App("t")
    with pytest.raises(ValueError, match="min_gpu_memory=.* requires gpu"):

        @app.function(image=_sample_image(), min_gpu_memory=16)
        def train():
            pass


def test_function_rejects_empty_gpu_string():
    app = App("t")
    with pytest.raises(ValueError, match="gpu must be a non-empty"):

        @app.function(image=_sample_image(), gpu="")
        def train():
            pass


def test_function_rejects_num_gpus_zero():
    app = App("t")
    with pytest.raises(ValueError, match="num_gpus must be a positive int"):

        @app.function(image=_sample_image(), gpu="A100", num_gpus=0)
        def train():
            pass


def test_function_rejects_num_gpus_greater_than_one_without_gpu():
    app = App("t")
    with pytest.raises(ValueError, match="num_gpus=4 requires gpu="):

        @app.function(image=_sample_image(), num_gpus=4)
        def train():
            pass


def test_function_num_gpus_defaults_to_one():
    app = App("t")

    @app.function(image=_sample_image(), gpu="T4")
    def train():
        pass

    assert app.functions["train"].num_gpus == 1


def test_function_accepts_multi_gpu_with_model():
    app = App("t")

    @app.function(image=_sample_image(), gpu="A100", num_gpus=4)
    def train():
        pass

    assert app.functions["train"].num_gpus == 4


# ---- Brev dispatch-time image vs mode validation -------------------------
# These checks live in the Brev backend's run() (not at decoration) so that
# local/modal users aren't forced to set brev_config just to use a Dockerfile
# image. Local/modal dispatch ignores brev_config entirely.


def test_validate_image_vs_brev_mode_rejects_dockerfile_on_container():
    from runplz.app import validate_image_vs_brev_mode

    with pytest.raises(ValueError, match="mode='container'"):
        validate_image_vs_brev_mode(
            fn_name="train",
            image=Image.from_dockerfile("Dockerfile"),
            brev_config=BrevConfig(mode="container"),
        )


def test_validate_image_vs_brev_mode_rejects_dockerfile_on_vm_native():
    from runplz.app import validate_image_vs_brev_mode

    with pytest.raises(ValueError, match="use_docker=False"):
        validate_image_vs_brev_mode(
            fn_name="train",
            image=Image.from_dockerfile("Dockerfile"),
            brev_config=BrevConfig(mode="vm", use_docker=False),
        )


def test_validate_image_vs_brev_mode_allows_dockerfile_on_vm_docker():
    from runplz.app import validate_image_vs_brev_mode

    # Returns None, no raise.
    validate_image_vs_brev_mode(
        fn_name="train",
        image=Image.from_dockerfile("Dockerfile"),
        brev_config=BrevConfig(mode="vm", use_docker=True),
    )


def test_validate_image_vs_brev_mode_allows_registry_on_container():
    from runplz.app import validate_image_vs_brev_mode

    validate_image_vs_brev_mode(
        fn_name="train",
        image=Image.from_registry("ubuntu:22.04"),
        brev_config=BrevConfig(mode="container"),
    )


def test_local_backend_accepts_dockerfile_with_default_container_mode(tmp_path):
    """Regression: using Image.from_dockerfile with a local-only App must not
    be blocked just because the (unused) default brev_config is mode='container'.
    """
    app = App("t")  # defaults: brev_config.mode == "container"

    # Function decoration must succeed — the image/mode check only fires when
    # actually dispatching to Brev.
    @app.function(image=Image.from_dockerfile("Dockerfile"))
    def train():
        pass

    assert "train" in app.functions


# ---- CLI tightened flag/backend enforcement -----------------------------


def test_cli_errors_on_instance_with_non_brev_backend(tmp_path):
    from runplz import _cli

    script = tmp_path / "job.py"
    script.write_text(
        "from runplz import App, Image\n"
        "app = App('t')\n"
        "image = Image.from_registry('ubuntu:22.04')\n"
        "@app.function(image=image)\n"
        "def f():\n"
        "    pass\n"
        "@app.local_entrypoint()\n"
        "def main():\n"
        "    pass\n"
    )
    with pytest.raises(SystemExit):
        _cli.main(["local", str(script), "--instance", "stray"])


def test_cli_errors_on_no_build_with_non_local_backend(tmp_path):
    from runplz import _cli

    script = tmp_path / "job.py"
    script.write_text(
        "from runplz import App, Image\n"
        "app = App('t')\n"
        "image = Image.from_registry('ubuntu:22.04')\n"
        "@app.function(image=image)\n"
        "def f():\n"
        "    pass\n"
        "@app.local_entrypoint()\n"
        "def main():\n"
        "    pass\n"
    )
    with pytest.raises(SystemExit):
        _cli.main(["brev", str(script), "--instance", "b", "--no-build"])
