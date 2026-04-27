"""Brev backend coverage — exercises every branch of brev.py with
subprocess/ssh/rsync mocked out, plus a full happy-path `run()` for
each of the three modes (docker vm, native vm, container).
"""

import json
import subprocess
import types
from pathlib import Path
from unittest import mock

import pytest

from runplz import App, BrevConfig, Image
from runplz.backends import brev

# -- helpers ---------------------------------------------------------------


class FakeSH:
    """Record all subprocess.run() calls and reply per a pre-programmed
    sequence. Each call's return value can be (returncode, stdout,
    stderr) or a callable that inspects the cmd."""

    def __init__(self, default=(0, "", ""), scripted=None):
        self.default = default
        self.scripted = list(scripted or [])
        self.calls = []

    def __call__(self, cmd, *args, **kwargs):
        self.calls.append((list(cmd), dict(kwargs)))
        if self.scripted:
            entry = self.scripted.pop(0)
            if callable(entry):
                entry = entry(cmd, kwargs)
            rc, out, err = entry
        else:
            rc, out, err = self.default
        return mock.Mock(returncode=rc, stdout=out, stderr=err)


def _app(tmp_path, *, cfg=None):
    app = App("panalle", brev_config=cfg or BrevConfig())
    app._repo_root = tmp_path
    return app


def _function(app, image, *, module_file=None, **extra):
    @app.function(image=image, **extra)
    def train():  # pragma: no cover — called only under runner control
        return "ok"

    fn = app.functions["train"]
    if module_file is not None:
        fn.module_file = str(module_file)
    return fn


def _job_inside(tmp_path, name="jobs/train.py"):
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("# fake\n")
    return p


# -- BrevConfig validation lives in __post_init__; covered in test_runplz.py --


# -- _brev_gpu_name covered in test_runplz.py; skipping ------------------


# -- _render_ops_script already smoke-tested; add edges ------------------


def test_render_ops_script_run_commands_and_index_url(tmp_path):
    img = (
        Image.from_registry("ubuntu:22.04")
        .pip_install("torch", index_url="https://download.pytorch.org/whl/cu121")
        .run_commands("echo hi")
    )
    s = brev._render_ops_script(img)
    assert "--index-url https://download.pytorch.org/whl/cu121" in s
    assert "echo hi" in s


def test_render_ops_script_non_editable_local_dir(tmp_path):
    img = Image.from_registry("ubuntu:22.04").pip_install_local_dir(".", editable=False)
    s = brev._render_ops_script(img)
    # The quoted path appears literally with double quotes so $HOME expands.
    assert 'pip install --quiet "$HOME/runplz-repo"' in s
    assert "-e" not in s.split("pip install")[-1]


# -- _instance_exists -----------------------------------------------------


def test_instance_exists_true_when_name_present():
    with mock.patch(
        "runplz.backends.brev.subprocess.run",
        return_value=mock.Mock(returncode=0, stdout=json.dumps([{"name": "foo"}])),
    ):
        assert brev._instance_exists("foo") is True


def test_instance_exists_false_when_missing():
    with mock.patch(
        "runplz.backends.brev.subprocess.run",
        return_value=mock.Mock(returncode=0, stdout="[]"),
    ):
        assert brev._instance_exists("bar") is False


def test_instance_exists_handles_non_list_json():
    with mock.patch(
        "runplz.backends.brev.subprocess.run",
        return_value=mock.Mock(returncode=0, stdout=json.dumps({"instances": [{"name": "x"}]})),
    ):
        assert brev._instance_exists("x") is True


def test_instance_exists_raises_on_cli_failure():
    # Silently returning False here would let the caller auto-create a
    # duplicate billed box. See brev._instance_exists docstring.
    with mock.patch(
        "runplz.backends.brev.subprocess.run",
        return_value=mock.Mock(returncode=1, stdout="", stderr="auth expired"),
    ):
        with pytest.raises(RuntimeError, match="brev ls"):
            brev._instance_exists("x")


def test_instance_exists_raises_on_invalid_json():
    with mock.patch(
        "runplz.backends.brev.subprocess.run",
        return_value=mock.Mock(returncode=0, stdout="not-json", stderr=""),
    ):
        with pytest.raises(RuntimeError, match="unparseable JSON"):
            brev._instance_exists("x")


def test_instance_exists_raises_on_unexpected_shape():
    with mock.patch(
        "runplz.backends.brev.subprocess.run",
        return_value=mock.Mock(returncode=0, stdout='"a-bare-string"', stderr=""),
    ):
        with pytest.raises(RuntimeError, match="unexpected shape"):
            brev._instance_exists("x")


def test_instance_exists_false_when_json_null():
    # brev ls --json can return `null` when the org has zero instances.
    with mock.patch(
        "runplz.backends.brev.subprocess.run",
        return_value=mock.Mock(returncode=0, stdout="null"),
    ):
        assert brev._instance_exists("x") is False


def test_instance_exists_handles_dict_with_null_instances():
    with mock.patch(
        "runplz.backends.brev.subprocess.run",
        return_value=mock.Mock(returncode=0, stdout='{"instances": null}'),
    ):
        assert brev._instance_exists("x") is False


# -- _wait_until_ssh_reachable --------------------------------------------


def test_wait_until_ssh_reachable_returns_on_first_success(monkeypatch):
    """First probe succeeds → helper returns immediately, no sleep."""
    from runplz.backends import _ssh_common

    monkeypatch.setattr(_ssh_common, "SSH_OPTS", [])
    calls = []

    def fake_run(cmd, *a, **kw):
        calls.append(cmd)
        return mock.Mock(returncode=0, stdout="", stderr="")

    sleeps = []
    monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))
    with mock.patch("runplz.backends._ssh_common.subprocess.run", fake_run):
        brev._wait_until_ssh_reachable("box", max_wait_s=60, probe_interval_s=1)

    assert len(calls) == 1
    assert calls[0][:1] == ["ssh"]
    assert sleeps == []  # no retries needed


def test_wait_until_ssh_reachable_raises_after_deadline(monkeypatch):
    """SSH never reachable → raises with informative error."""
    from runplz.backends import _ssh_common

    monkeypatch.setattr(_ssh_common, "SSH_OPTS", [])
    clock = [0.0]
    monkeypatch.setattr("time.time", lambda: clock[0])
    monkeypatch.setattr(
        "time.sleep",
        lambda s: clock.__setitem__(0, clock[0] + s),
    )

    def fake_run(cmd, *a, **kw):
        return mock.Mock(returncode=255, stdout="", stderr="Connection refused")

    with mock.patch("runplz.backends._ssh_common.subprocess.run", fake_run):
        with pytest.raises(RuntimeError, match="never became reachable"):
            brev._wait_until_ssh_reachable("box", max_wait_s=5, probe_interval_s=1)


def test_wait_until_ssh_reachable_invokes_refresh_callback_periodically(monkeypatch):
    """Every 4 failed probes the helper should fire the caller-provided
    refresh_callback (Brev uses it to re-run `brev refresh` in case the
    SSH port changed when the box finished provisioning)."""
    from runplz.backends import _ssh_common

    monkeypatch.setattr(_ssh_common, "SSH_OPTS", [])
    clock = [0.0]
    monkeypatch.setattr("time.time", lambda: clock[0])
    monkeypatch.setattr("time.sleep", lambda s: clock.__setitem__(0, clock[0] + s))

    probes = []
    refresh_calls = {"n": 0}

    def fake_run(cmd, *a, **kw):
        probes.append(cmd[0])
        return mock.Mock(returncode=255, stdout="", stderr="refused")

    def bump():
        refresh_calls["n"] += 1

    with mock.patch("runplz.backends._ssh_common.subprocess.run", fake_run):
        with pytest.raises(RuntimeError):
            brev._wait_until_ssh_reachable(
                "box", max_wait_s=10, probe_interval_s=1, refresh_callback=bump
            )

    assert sum(1 for c in probes if c == "ssh") >= 4
    assert refresh_calls["n"] >= 1


# -- _require_brev_cli ----------------------------------------------------


def test_require_brev_cli_raises_when_missing():
    with mock.patch(
        "runplz.backends.brev.subprocess.run",
        return_value=mock.Mock(returncode=1),
    ):
        with pytest.raises(RuntimeError, match="brev` CLI not found"):
            brev._require_brev_cli()


def test_require_brev_cli_silent_when_present():
    with mock.patch(
        "runplz.backends.brev.subprocess.run",
        return_value=mock.Mock(returncode=0),
    ):
        brev._require_brev_cli()  # no raise


# -- _skip_onboarding -----------------------------------------------------


def test_skip_onboarding_writes_file_on_first_call(tmp_path, monkeypatch, capsys):
    fake_home = tmp_path / "home"
    monkeypatch.setattr(brev, "_BREV_ONBOARDING", fake_home / ".brev" / "onboarding_step.json")
    brev._skip_onboarding()
    text = (fake_home / ".brev" / "onboarding_step.json").read_text()
    parsed = json.loads(text)
    assert parsed["hasRunBrevShell"] is True
    assert "+ wrote" in capsys.readouterr().out


def test_skip_onboarding_tolerates_os_error(monkeypatch):
    monkeypatch.setattr(brev, "_BREV_ONBOARDING", Path("/nonexistent/root/not/allowed.json"))
    # Should not raise even if mkdir/write fails.
    brev._skip_onboarding()


# -- _pick_instance_type + _create_instance additional edges -------------


def test_pick_instance_type_handles_non_list_json():
    with mock.patch(
        "runplz.backends.brev.subprocess.run",
        return_value=mock.Mock(returncode=0, stdout=json.dumps({"type": "x"})),
    ):
        # `isinstance(results, list)` is False → returns None.
        assert (
            brev._pick_instance_type(
                types.SimpleNamespace(
                    gpu="T4",
                    min_cpu=None,
                    min_memory=None,
                    min_gpu_memory=None,
                    min_disk=None,
                )
            )
            is None
        )


def test_pick_instance_type_returns_none_on_non_zero_rc():
    with mock.patch(
        "runplz.backends.brev.subprocess.run",
        return_value=mock.Mock(returncode=1, stdout=""),
    ):
        assert (
            brev._pick_instance_type(
                types.SimpleNamespace(
                    gpu="T4",
                    min_cpu=None,
                    min_memory=None,
                    min_gpu_memory=None,
                    min_disk=None,
                )
            )
            is None
        )


def test_pick_instance_type_parses_Type_capital_key():
    with mock.patch(
        "runplz.backends.brev.subprocess.run",
        return_value=mock.Mock(returncode=0, stdout=json.dumps([{"Type": "alt-key"}])),
    ):
        result = brev._pick_instance_type(
            types.SimpleNamespace(
                gpu="T4",
                min_cpu=None,
                min_memory=None,
                min_gpu_memory=None,
                min_disk=None,
            )
        )
        assert result == "alt-key"


def test_create_instance_with_resource_request_uses_picker(tmp_path):
    app = _app(tmp_path)
    fn = _function(
        app,
        Image.from_registry("ubuntu:22.04"),
        module_file=_job_inside(tmp_path),
        gpu="T4",
        min_cpu=4,
        min_memory=26,
        min_disk=100,
    )

    recorded = {}

    def fake_capture(cmd, **kw):
        if cmd[:2] == ["brev", "create"]:
            recorded["create_cmd"] = list(cmd)
            return mock.Mock(returncode=0, stdout="", stderr="")
        return mock.Mock(
            returncode=0,
            stdout=json.dumps([{"name": "x", "status": "RUNNING"}]),
            stderr="",
        )

    with mock.patch("runplz.backends.brev._pick_instance_types", return_value=["picked-type"]):
        with mock.patch("runplz.backends.brev._brev_capture", fake_capture):
            brev._create_instance("x", cfg=app.brev_config, image=fn.image, function=fn)

    cmd = recorded["create_cmd"]
    assert cmd[:2] == ["brev", "create"]
    assert "picked-type" in cmd
    # min_disk forwarded to provisioning
    i = cmd.index("--min-disk")
    assert cmd[i + 1] == "100"


def test_create_instance_no_constraints_falls_through_to_picker(tmp_path):
    """When no instance_type is pinned and no constraints are declared, we
    still call the picker — it defaults to `brev search cpu --sort price`
    and returns the cheapest available box rather than raising."""
    cfg = BrevConfig(auto_create_instances=True)
    app = _app(tmp_path, cfg=cfg)
    fn = _function(app, Image.from_registry("ubuntu:22.04"), module_file=_job_inside(tmp_path))

    recorded = {}
    with mock.patch("runplz.backends.brev._pick_instance_types", return_value=["cheap-cpu-type"]):
        with mock.patch(
            "runplz.backends.brev._brev_capture",
            lambda cmd, **kw: (
                recorded.setdefault("c", cmd),
                mock.Mock(returncode=0, stdout="", stderr=""),
            )[1],
        ):
            brev._create_instance("x", cfg=cfg, image=fn.image, function=fn)

    assert "cheap-cpu-type" in recorded["c"]


def test_create_instance_passes_multiple_type_flags_for_fallback(tmp_path):
    """3.9.0: auto-picked types fan out into repeated --type flags so
    Brev's native multi-provider retry kicks in."""
    cfg = BrevConfig(auto_create_instances=True, mode="vm", instance_type_fallback_count=3)
    app = _app(tmp_path, cfg=cfg)
    fn = _function(
        app, Image.from_registry("ubuntu:22.04"), module_file=_job_inside(tmp_path), gpu="T4"
    )

    recorded = {}

    def fake_capture(cmd, **kw):
        if cmd[:2] == ["brev", "create"]:
            recorded["cmd"] = list(cmd)
            return mock.Mock(returncode=0, stdout="", stderr="")
        return mock.Mock(
            returncode=0,
            stdout=json.dumps([{"name": "multi", "status": "RUNNING"}]),
            stderr="",
        )

    with mock.patch(
        "runplz.backends.brev._pick_instance_types",
        return_value=["type-a", "type-b", "type-c"],
    ):
        with mock.patch("runplz.backends.brev._brev_capture", fake_capture):
            brev._create_instance("multi", cfg=cfg, image=fn.image, function=fn)

    cmd = recorded["cmd"]
    # Three --type flags in the same order the picker returned.
    type_positions = [i for i, tok in enumerate(cmd) if tok == "--type"]
    assert len(type_positions) == 3
    types_passed = [cmd[i + 1] for i in type_positions]
    assert types_passed == ["type-a", "type-b", "type-c"]


def test_create_instance_pinned_instance_type_skips_fallback(tmp_path):
    """If the user explicitly pinned an instance_type, we pass only
    that one — no fallback list, no picker call."""
    cfg = BrevConfig(
        auto_create_instances=True,
        mode="vm",
        instance_type="my-pinned-type",
        instance_type_fallback_count=5,  # should be ignored for pinned types
    )
    app = _app(tmp_path, cfg=cfg)
    fn = _function(app, Image.from_registry("ubuntu:22.04"), module_file=_job_inside(tmp_path))

    recorded = {}
    picker_called = {"n": 0}

    def fake_capture(cmd, **kw):
        if cmd[:2] == ["brev", "create"]:
            recorded["cmd"] = list(cmd)
            return mock.Mock(returncode=0, stdout="", stderr="")
        return mock.Mock(
            returncode=0,
            stdout=json.dumps([{"name": "pinned", "status": "RUNNING"}]),
            stderr="",
        )

    def fake_picker(fn, *, n):
        picker_called["n"] += 1
        return ["SHOULD-NEVER-BE-USED"]

    with mock.patch("runplz.backends.brev._pick_instance_types", fake_picker):
        with mock.patch("runplz.backends.brev._brev_capture", fake_capture):
            brev._create_instance("pinned", cfg=cfg, image=fn.image, function=fn)

    assert picker_called["n"] == 0, "picker must not run when instance_type is pinned"
    type_positions = [i for i, tok in enumerate(recorded["cmd"]) if tok == "--type"]
    assert len(type_positions) == 1
    assert recorded["cmd"][type_positions[0] + 1] == "my-pinned-type"


def test_create_instance_fallback_count_one_passes_single_type(tmp_path):
    """Setting instance_type_fallback_count=1 matches pre-3.9 behavior."""
    cfg = BrevConfig(auto_create_instances=True, mode="vm", instance_type_fallback_count=1)
    app = _app(tmp_path, cfg=cfg)
    fn = _function(
        app, Image.from_registry("ubuntu:22.04"), module_file=_job_inside(tmp_path), gpu="T4"
    )

    recorded = {}

    def fake_capture(cmd, **kw):
        if cmd[:2] == ["brev", "create"]:
            recorded["cmd"] = list(cmd)
            return mock.Mock(returncode=0, stdout="", stderr="")
        return mock.Mock(
            returncode=0,
            stdout=json.dumps([{"name": "solo", "status": "RUNNING"}]),
            stderr="",
        )

    seen_n = {}

    def fake_picker(fn, *, n):
        seen_n["n"] = n
        return ["only-type"]

    with mock.patch("runplz.backends.brev._pick_instance_types", fake_picker):
        with mock.patch("runplz.backends.brev._brev_capture", fake_capture):
            brev._create_instance("solo", cfg=cfg, image=fn.image, function=fn)

    assert seen_n["n"] == 1  # picker was asked for exactly 1 type
    type_positions = [i for i, tok in enumerate(recorded["cmd"]) if tok == "--type"]
    assert len(type_positions) == 1


def test_create_instance_raises_when_picker_returns_none(tmp_path):
    cfg = BrevConfig(auto_create_instances=True)
    app = _app(tmp_path, cfg=cfg)
    fn = _function(
        app, Image.from_registry("ubuntu:22.04"), module_file=_job_inside(tmp_path), gpu="T4"
    )
    with mock.patch("runplz.backends.brev._pick_instance_types", return_value=[]):
        with pytest.raises(RuntimeError, match="no matching instances"):
            brev._create_instance("x", cfg=cfg, image=fn.image, function=fn)


def test_create_instance_explicit_instance_type_bypasses_picker(tmp_path):
    cfg = BrevConfig(auto_create_instances=True, instance_type="my-explicit-type")
    app = _app(tmp_path, cfg=cfg)
    # No constraints AND no function-level picker call should happen.
    fn = _function(app, Image.from_registry("ubuntu:22.04"), module_file=_job_inside(tmp_path))

    recorded = {}
    picker_called = {"n": 0}

    def fake_picker(_fn):
        picker_called["n"] += 1
        return "SHOULD-NOT-BE-USED"

    with mock.patch("runplz.backends.brev._pick_instance_type", fake_picker):
        with mock.patch(
            "runplz.backends.brev._brev_capture",
            lambda cmd, **kw: (
                recorded.setdefault("c", cmd),
                mock.Mock(returncode=0, stdout="", stderr=""),
            )[1],
        ):
            brev._create_instance("x", cfg=cfg, image=fn.image, function=fn)

    assert picker_called["n"] == 0
    assert "my-explicit-type" in recorded["c"]


def test_create_instance_container_mode_adds_image_flag(tmp_path):
    cfg = BrevConfig(mode="container")
    app = _app(tmp_path, cfg=cfg)
    image = Image.from_registry("pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime")
    fn = _function(app, image, module_file=_job_inside(tmp_path), gpu="T4")

    recorded = {}
    with mock.patch("runplz.backends.brev._pick_instance_types", return_value=["picked-type"]):
        with mock.patch(
            "runplz.backends.brev._brev_capture",
            lambda cmd, **kw: (
                recorded.setdefault("c", cmd),
                mock.Mock(returncode=0, stdout="", stderr=""),
            )[1],
        ):
            brev._create_instance("x", cfg=cfg, image=image, function=fn)

    cmd = recorded["c"]
    assert "--mode" in cmd
    assert "container" in cmd
    assert "--container-image" in cmd
    assert "pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime" in cmd


# `_create_instance` used to guard against Dockerfile + container mode
# here; that's now rejected at function-decoration time by runplz.app
# (see tests/test_runplz.py::test_function_rejects_dockerfile_image_on_container_mode).


# -- _ensure_docker -------------------------------------------------------


def test_ensure_docker_happy_path():
    # subprocess.run returns 0 → wait script exit 0 → no fallback.
    with mock.patch(
        "runplz.backends._ssh_common.subprocess.run",
        return_value=mock.Mock(returncode=0),
    ):
        brev._ensure_docker("some-box")  # no raise


def test_ensure_docker_falls_back_to_get_docker_sh(capsys):
    # First call (wait script) fails → fallback calls _sh with curl | sh.
    fallback_called = {}

    def fake_sh(cmd):
        fallback_called["cmd"] = cmd

    with mock.patch(
        "runplz.backends._ssh_common.subprocess.run",
        return_value=mock.Mock(returncode=1),
    ):
        with mock.patch("runplz.backends._ssh_common._sh", fake_sh):
            brev._ensure_docker("some-box")

    assert "get.docker.com" in fallback_called["cmd"][-1]
    assert "falling back" in capsys.readouterr().out


# -- _remote_has_nvidia ---------------------------------------------------


def test_remote_has_nvidia_true():
    with mock.patch(
        "runplz.backends._ssh_common.subprocess.run",
        return_value=mock.Mock(returncode=0, stdout="y\n"),
    ):
        assert brev._remote_has_nvidia("box") is True


def test_remote_has_nvidia_false():
    with mock.patch(
        "runplz.backends._ssh_common.subprocess.run",
        return_value=mock.Mock(returncode=0, stdout="n\n"),
    ):
        assert brev._remote_has_nvidia("box") is False


# -- _rsync_up / _rsync_down / _ssh / _sh ---------------------------------


def test_rsync_up_excludes_and_no_delete(tmp_path):
    recorded = {}
    with mock.patch("runplz.backends._ssh_common._sh", lambda c: recorded.setdefault("c", c)):
        brev._rsync_up(tmp_path, "my-box")
    cmd = recorded["c"]
    assert cmd[:2] == ["rsync", "-az"]
    assert "--delete" not in cmd  # regression: --delete was removed
    assert "--exclude=.git" in cmd
    assert "--exclude=out" in cmd


def test_rsync_up_excludes_configured_outputs_dir(tmp_path):
    """Issue #55: when outputs_dir != "out", the local outputs tree was
    silently re-uploaded on every launch. Verify the configured outputs_dir
    is added as an --exclude pattern."""
    recorded = {}
    with mock.patch("runplz.backends._ssh_common._sh", lambda c: recorded.setdefault("c", c)):
        brev._rsync_up(tmp_path, "my-box", outputs_dir="brev_runs")
    cmd = recorded["c"]
    assert "--exclude=/brev_runs/" in cmd
    # Single-segment names also get the unanchored form, matching the
    # existing convention for "out".
    assert "--exclude=brev_runs" in cmd


def test_rsync_up_skips_outputs_dir_exclude_for_default(tmp_path):
    """outputs_dir="out" is already in _RSYNC_NOISE_EXCLUDES — don't double up."""
    recorded = {}
    with mock.patch("runplz.backends._ssh_common._sh", lambda c: recorded.setdefault("c", c)):
        brev._rsync_up(tmp_path, "my-box", outputs_dir="out")
    cmd = recorded["c"]
    assert cmd.count("--exclude=out") == 1
    assert "--exclude=/out/" not in cmd


def test_rsync_up_handles_nested_outputs_dir(tmp_path):
    """A multi-segment outputs_dir like 'data/runs' must be anchored so we
    don't accidentally exclude every directory named 'runs' in the tree."""
    recorded = {}
    with mock.patch("runplz.backends._ssh_common._sh", lambda c: recorded.setdefault("c", c)):
        brev._rsync_up(tmp_path, "my-box", outputs_dir="data/runs")
    cmd = recorded["c"]
    assert "--exclude=/data/runs/" in cmd
    # No unanchored form for multi-segment paths.
    assert "--exclude=runs" not in cmd
    assert "--exclude=data/runs" not in cmd


def test_rsync_up_skips_outputs_dir_exclude_when_outside_repo(tmp_path):
    """Absolute outputs_dir paths outside the repo can't be inside the rsync
    source, so emitting an exclude for them would be noise."""
    outside = tmp_path.parent / "elsewhere"
    recorded = {}
    with mock.patch("runplz.backends._ssh_common._sh", lambda c: recorded.setdefault("c", c)):
        brev._rsync_up(tmp_path, "my-box", outputs_dir=str(outside))
    cmd = recorded["c"]
    # No outputs-dir-derived exclude — only the standard noise + secrets.
    assert not any(x.startswith(f"--exclude={outside}") for x in cmd)


def test_rsync_up_excludes_default_secret_files(tmp_path):
    """Issue #18: a repo with .env / ssh keys / credentials.json must not
    ship those to the remote box by default."""
    from runplz._excludes import DEFAULT_TRANSFER_EXCLUDES

    recorded = {}
    with mock.patch("runplz.backends._ssh_common._sh", lambda c: recorded.setdefault("c", c)):
        brev._rsync_up(tmp_path, "my-box")

    cmd = recorded["c"]
    for pat in DEFAULT_TRANSFER_EXCLUDES:
        assert f"--exclude={pat}" in cmd, f"missing --exclude={pat}"


def test_rsync_down_runs_correct_cmd(tmp_path):
    recorded = {}
    with mock.patch("runplz.backends._ssh_common._sh", lambda c: recorded.setdefault("c", c)):
        brev._rsync_down("my-box", tmp_path)
    cmd = recorded["c"]
    assert cmd[:2] == ["rsync", "-az"]
    assert cmd[2].endswith(":runplz-out/")  # remote side
    assert cmd[3] == f"{tmp_path}/"


def test_make_remote_run_context_uses_unique_runplz_runs_layout():
    remote_run = brev.make_remote_run_context(
        backend="brev",
        target="gpu.example.com",
        function_name="train",
    )
    assert remote_run.run_root_rel.startswith("runplz-runs/")
    assert remote_run.repo_rel.endswith("/repo")
    assert remote_run.out_rel.endswith("/out")
    assert remote_run.meta_rel.endswith("/out/.runplz")
    assert "gpu-example-com" in remote_run.run_id


def test_prepare_remote_run_writes_manifest_and_latest_link(tmp_path):
    remote_run = brev.make_remote_run_context(backend="brev", target="box", function_name="train")
    manifest = brev.build_remote_run_manifest(
        remote_run=remote_run,
        repo=tmp_path,
        outputs_dir="out",
        args=["x"],
        kwargs={"epochs": 2},
        env={"API_TOKEN": "secret", "PLAIN": "ok"},
    )

    recorded = {}
    with mock.patch(
        "runplz.backends._ssh_common._ssh",
        lambda target, cmd, **kw: recorded.update({"target": target, "cmd": cmd}),
    ):
        brev._prepare_remote_run("box", remote_run, manifest=manifest)

    cmd = recorded["cmd"]
    assert recorded["target"] == "box"
    assert "run.json" in cmd
    assert "events.ndjson" in cmd
    assert "heartbeat.ndjson" in cmd
    assert "ln -sfn" in cmd
    assert "***" in cmd  # masked API_TOKEN value in run.json


def test_rsync_up_uses_remote_run_repo_when_provided(tmp_path):
    remote_run = brev.make_remote_run_context(backend="ssh", target="box", function_name="train")
    recorded = {}
    with mock.patch("runplz.backends._ssh_common._record_remote_event", lambda *a, **k: None):
        with mock.patch(
            "runplz.backends._ssh_common._sh",
            lambda c: recorded.setdefault("c", c),
        ):
            brev._rsync_up(tmp_path, "my-box", remote_run=remote_run)

    cmd = recorded["c"]
    assert cmd[-1].endswith(f":~/{remote_run.repo_rel}/")


def test_run_native_with_remote_run_uses_per_run_lifecycle_files(tmp_path):
    """The launcher script handed to ssh must reference the per-run meta
    files (out/, last.log, heartbeat.ndjson) and the wrapper's
    ``bootstrap_start`` event. Post-3.11 this is visible on the _ssh
    launcher call (not subprocess.run) because the bootstrap is now
    detached."""
    app = _app(tmp_path)
    fn = _function(
        app,
        Image.from_registry("ubuntu:22.04"),
        module_file=_job_inside(tmp_path),
        env={"EXTRA": "ok"},
    )
    remote_run = brev.make_remote_run_context(backend="ssh", target="box", function_name="train")
    recorded = {"ssh_cmds": []}

    def fake_ssh(target, cmd, **kw):
        recorded["ssh_cmds"].append(cmd)

    def fake_sub_run(cmd, *a, **kw):
        # Return a "pid is dead" / "exit_code 0" stdout so the poll loop
        # exits immediately and we can make assertions. stdout must be
        # a real string — the exit-code parser uses json.loads on it.
        cmd_str = " ".join(cmd)
        if "kill -0" in cmd_str:
            return mock.Mock(returncode=0, stdout="dead", stderr="")
        if "remote_command_exit" in cmd_str:
            return mock.Mock(
                returncode=0,
                stdout='{"ts":"x","event":"remote_command_exit","exit_code":0}\n',
                stderr="",
            )
        # tail -F: pretend we streamed then EOF.
        return mock.Mock(returncode=0, stdout="", stderr="")

    with mock.patch("runplz.backends._ssh_common._ssh", fake_ssh):
        with mock.patch("runplz.backends._ssh_common.subprocess.run", fake_sub_run):
            rc = brev._run_native(
                target="box",
                function=fn,
                rel_script="jobs/train.py",
                args=[],
                kwargs={},
                has_nvidia=False,
                remote_run=remote_run,
            )

    assert rc == 0
    # The launcher script (2nd _ssh call: 1st is the apt-get setup) must
    # embed the per-run paths through the heredoc-wrapped inner command.
    launcher = recorded["ssh_cmds"][-1]
    assert remote_run.out_rel in launcher
    assert remote_run.last_log_rel in launcher
    assert "heartbeat.ndjson" in launcher
    assert "bootstrap_start" in launcher
    # And the new detach plumbing: setsid + nohup + pid file.
    assert "setsid" in launcher
    assert "nohup" in launcher
    assert "bootstrap.pid" in launcher


def test_ssh_packages_cmd_in_bash_lc():
    recorded = {}
    with mock.patch("runplz.backends._ssh_common._sh", lambda c: recorded.setdefault("c", c)):
        brev._ssh("instance", "echo hi")
    assert recorded["c"][0] == "ssh"
    # Last arg is `bash -lc '<escaped cmd>'`.
    assert recorded["c"][-1].startswith("bash -lc ")
    assert "echo hi" in recorded["c"][-1]


def test_sh_prints_and_runs(capsys):
    with mock.patch(
        "runplz.backends._ssh_common.subprocess.run",
        return_value=mock.Mock(returncode=0),
    ) as patched:
        brev._sh(["echo", "hello world"])
    assert patched.called
    # shlex-quoted print with + prefix
    assert "+ echo" in capsys.readouterr().out


def test_ssh_capture_runs_with_capture():
    with mock.patch(
        "runplz.backends._ssh_common.subprocess.run",
        return_value=mock.Mock(returncode=0, stdout="stdout-content"),
    ) as patched:
        out = brev._ssh_capture("box", "echo hi")
    assert out == "stdout-content"
    assert patched.call_args.kwargs.get("capture_output") is True


# -- _ensure_remote_rsync -------------------------------------------------


def test_ensure_remote_rsync_sends_expected_cmd():
    recorded = {}
    with mock.patch(
        "runplz.backends._ssh_common._ssh",
        lambda target, cmd, **kw: recorded.update({"i": target, "c": cmd}),
    ):
        brev._ensure_remote_rsync("box")
    assert recorded["i"] == "box"
    assert "apt-get install" in recorded["c"]
    assert "rsync" in recorded["c"]


# -- _build_image ---------------------------------------------------------


def test_build_image_dockerfile_uses_docker_build_f(tmp_path):
    img = Image.from_dockerfile("docker/Dockerfile")
    recorded = {}
    with mock.patch(
        "runplz.backends._ssh_common._ssh",
        lambda i, c, **kw: recorded.setdefault("c", c),
    ):
        brev._build_image("box", img)
    assert "docker build -f docker/Dockerfile" in recorded["c"]
    assert "__EOF__" not in recorded["c"]  # no synthesized Dockerfile heredoc


def test_build_image_dockerfile_honors_context():
    img = Image.from_dockerfile("docker/Dockerfile", context="docker")
    recorded = {}
    with mock.patch(
        "runplz.backends._ssh_common._ssh",
        lambda i, c, **kw: recorded.setdefault("c", c),
    ):
        brev._build_image("box", img)
    assert "docker build -f docker/Dockerfile" in recorded["c"]
    assert recorded["c"].endswith(" runplz-train:remote docker")


def test_build_image_from_registry_pipes_synthesized_dockerfile():
    img = Image.from_registry("pytorch/pytorch:2.4.0").pip_install("numpy")
    recorded = {}
    with mock.patch(
        "runplz.backends._ssh_common._ssh",
        lambda i, c, **kw: recorded.setdefault("c", c),
    ):
        brev._build_image("box", img)
    c = recorded["c"]
    assert "cat <<'__EOF__' | sudo docker build -f - -t" in c
    assert "FROM pytorch/pytorch:2.4.0" in c


# -- _run_container_detached ----------------------------------------------


def test_run_container_detached_builds_full_docker_run(tmp_path):
    app = _app(tmp_path)
    image = Image.from_registry("ubuntu:22.04")
    fn = _function(app, image, module_file=_job_inside(tmp_path), env={"USER_VAR": "1"})

    recorded = {}
    with mock.patch(
        "runplz.backends._ssh_common._ssh",
        lambda i, c, **kw: recorded.setdefault("c", c),
    ):
        brev._run_container_detached(
            target="box",
            container_name="runplz-train-abc123",
            function=fn,
            rel_script="jobs/train.py",
            args=[],
            kwargs={},
            gpu_flag="--gpus all",
        )
    c = recorded["c"]
    assert "sudo docker run -d --name runplz-train-abc123" in c
    assert "--network=host" in c
    assert "--gpus all" in c
    assert "RUNPLZ_RUNNER_SCRIPT" not in c  # old env name — must be RUNPLZ_SCRIPT
    assert "RUNPLZ_SCRIPT" in c
    assert "USER_VAR=1" in c


def test_run_container_detached_with_remote_run_starts_monitor(tmp_path):
    app = _app(tmp_path)
    image = Image.from_registry("ubuntu:22.04")
    fn = _function(app, image, module_file=_job_inside(tmp_path))
    remote_run = brev.make_remote_run_context(backend="brev", target="box", function_name="train")

    recorded = {}
    with mock.patch(
        "runplz.backends._ssh_common._ssh",
        lambda i, c, **kw: recorded.setdefault("c", c),
    ):
        brev._run_container_detached(
            target="box",
            container_name="runplz-train-abc123",
            function=fn,
            rel_script="jobs/train.py",
            args=[],
            kwargs={},
            gpu_flag="",
            remote_run=remote_run,
        )

    c = recorded["c"]
    assert remote_run.out_rel in c
    assert "runplz_event container_started" in c
    assert "heartbeat.ndjson" in c
    assert "docker wait runplz-train-abc123" in c


# -- _stream_and_wait + _container_running -------------------------------


def test_stream_and_wait_exits_when_container_done():
    # First log-stream ssh returns rc=0; _container_running says not running.
    seq = iter(
        [
            mock.Mock(returncode=0),  # docker logs -f
            mock.Mock(returncode=0, stdout="false"),  # docker inspect (not running)
            mock.Mock(returncode=0, stdout="0\n"),  # docker wait → exit 0
        ]
    )

    def fake_run(cmd, *a, **kw):
        return next(seq)

    with mock.patch("runplz.backends._ssh_common.subprocess.run", fake_run):
        rc = brev._stream_and_wait("box", "c")
    assert rc == 0


def test_stream_and_wait_reconnects_then_gives_up(capsys):
    # docker logs -f keeps returning and container_running keeps saying
    # True. After max_reconnects we should bail out and try docker wait.
    ssh_returns = mock.Mock(returncode=255)  # logs -f dropped
    inspect_true = mock.Mock(returncode=0, stdout="true")  # still running
    wait_final = mock.Mock(returncode=0, stdout="137\n")

    order = []

    def fake_run(cmd, *a, **kw):
        # Classify which call this is based on cmd contents.
        joined = " ".join(str(x) for x in cmd)
        order.append(joined)
        if "docker logs" in joined:
            return ssh_returns
        if "docker inspect" in joined:
            return inspect_true
        if "docker wait" in joined:
            return wait_final
        return mock.Mock(returncode=0, stdout="")

    # Keep the retry ceiling small so the test finishes fast.
    with mock.patch("runplz.backends._ssh_common.subprocess.run", fake_run):
        with mock.patch("runplz.backends._ssh_common.time.sleep", lambda _s: None):
            rc = brev._stream_and_wait("box", "c", max_reconnects=3)
    assert rc == 137
    # Should have emitted the give-up message once max_reconnects hit.
    assert "giving up on log stream" in capsys.readouterr().out


def test_stream_and_wait_wait_path_still_honors_runtime_cap_after_give_up():
    wait_timeouts = []

    def fake_run(cmd, *a, **kw):
        joined = " ".join(str(x) for x in cmd)
        if "docker logs" in joined:
            return mock.Mock(returncode=255, stdout="", stderr="")
        if "docker inspect" in joined:
            return mock.Mock(returncode=0, stdout="true", stderr="")
        if "docker wait" in joined:
            wait_timeouts.append(kw.get("timeout"))
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=kw.get("timeout", 0))
        if "docker kill" in joined:
            return mock.Mock(returncode=0, stdout="", stderr="")
        return mock.Mock(returncode=0, stdout="", stderr="")

    with mock.patch("runplz.backends._ssh_common.subprocess.run", fake_run):
        with mock.patch("runplz.backends._ssh_common.time.sleep", lambda _s: None):
            with pytest.raises(RuntimeError, match="max_runtime_seconds=7"):
                brev._stream_and_wait("box", "c", max_reconnects=0, max_runtime_seconds=7)

    assert wait_timeouts
    assert 0 < wait_timeouts[0] <= 7


def test_container_running_treats_ssh_failure_as_still_running():
    # subprocess.run raises TimeoutExpired → conservative return True.
    with mock.patch(
        "runplz.backends._ssh_common.subprocess.run",
        side_effect=__import__("subprocess").TimeoutExpired(cmd="ssh", timeout=30),
    ):
        assert brev._container_running("box", "c") is True


def test_container_running_returns_false_on_inspect_false():
    with mock.patch(
        "runplz.backends._ssh_common.subprocess.run",
        return_value=mock.Mock(returncode=0, stdout="false\n"),
    ):
        assert brev._container_running("box", "c") is False


def test_container_running_true_on_running_output():
    with mock.patch(
        "runplz.backends._ssh_common.subprocess.run",
        return_value=mock.Mock(returncode=0, stdout="true\n"),
    ):
        assert brev._container_running("box", "c") is True


# -- run() full happy paths -----------------------------------------------


def test_run_fails_when_instance_missing_without_auto_create(tmp_path):
    cfg = BrevConfig(auto_create_instances=False, mode="vm")
    app = _app(tmp_path, cfg=cfg)
    fn = _function(app, Image.from_registry("ubuntu:22.04"), module_file=_job_inside(tmp_path))

    with mock.patch("runplz.backends.brev._require_brev_cli"):
        with mock.patch("runplz.backends.brev._skip_onboarding"):
            with mock.patch("runplz.backends.brev._instance_exists", return_value=False):
                with pytest.raises(RuntimeError) as ei:
                    brev.run(app, fn, [], {}, instance="gone")

    msg = str(ei.value)
    assert "not found" in msg
    # 3.3: surface the exact override so users don't grep the docs.
    assert "auto_create_instances=True" in msg
    assert "brev ls" in msg


def test_run_vm_docker_mode_end_to_end(tmp_path):
    cfg = BrevConfig(mode="vm", use_docker=True)
    app = _app(tmp_path, cfg=cfg)
    fn = _function(app, Image.from_registry("ubuntu:22.04"), module_file=_job_inside(tmp_path))

    with mock.patch.multiple(
        "runplz.backends.brev",
        _require_brev_cli=mock.DEFAULT,
        _skip_onboarding=mock.DEFAULT,
        _instance_exists=mock.Mock(return_value=True),
        _refresh_ssh=mock.DEFAULT,
        _wait_until_ssh_reachable=mock.DEFAULT,
        _prepare_remote_run=mock.DEFAULT,
        _ensure_docker=mock.DEFAULT,
        _rsync_up=mock.DEFAULT,
        _remote_has_nvidia=mock.Mock(return_value=True),
        _build_image=mock.DEFAULT,
        _run_container_detached=mock.DEFAULT,
        _stream_and_wait=mock.Mock(return_value=0),
        _ssh_capture=mock.DEFAULT,
        _rsync_down=mock.DEFAULT,
        _apply_on_finish=mock.DEFAULT,
    ):
        brev.run(app, fn, [], {}, instance="box")


def test_run_threads_ssh_ready_wait_seconds_into_helper(tmp_path):
    """3.7.2: BrevConfig.ssh_ready_wait_seconds must reach
    _wait_until_ssh_reachable (otherwise the knob is decorative)."""
    cfg = BrevConfig(mode="vm", use_docker=True, ssh_ready_wait_seconds=2400)
    app = _app(tmp_path, cfg=cfg)
    fn = _function(app, Image.from_registry("ubuntu:22.04"), module_file=_job_inside(tmp_path))

    seen = {}

    def capture_wait(target, **kw):
        seen["max_wait_s"] = kw.get("max_wait_s")

    with mock.patch.multiple(
        "runplz.backends.brev",
        _require_brev_cli=mock.DEFAULT,
        _skip_onboarding=mock.DEFAULT,
        _instance_exists=mock.Mock(return_value=True),
        _refresh_ssh=mock.DEFAULT,
        _wait_until_ssh_reachable=capture_wait,
        _prepare_remote_run=mock.DEFAULT,
        _ensure_docker=mock.DEFAULT,
        _rsync_up=mock.DEFAULT,
        _remote_has_nvidia=mock.Mock(return_value=False),
        _build_image=mock.DEFAULT,
        _run_container_detached=mock.DEFAULT,
        _stream_and_wait=mock.Mock(return_value=0),
        _ssh_capture=mock.DEFAULT,
        _rsync_down=mock.DEFAULT,
        _apply_on_finish=mock.DEFAULT,
    ):
        brev.run(app, fn, [], {}, instance="box")

    assert seen["max_wait_s"] == 2400


def test_run_vm_docker_mode_nonzero_exit_raises(tmp_path):
    """Issue #17: non-zero exit must surface the remote log tail in the
    RuntimeError so users don't have to ssh in to see what actually broke."""
    cfg = BrevConfig(mode="vm", use_docker=True)
    app = _app(tmp_path, cfg=cfg)
    fn = _function(app, Image.from_registry("ubuntu:22.04"), module_file=_job_inside(tmp_path))

    tail_output = "Traceback (most recent call last):\n  File 'x.py'\nRuntimeError: oops"

    def fake_ssh_capture(instance, cmd, **kw):
        # The _fetch_failure_tail helper issues `docker logs --tail N ...`
        # for VM+docker mode. Return our canned tail to that call.
        if "docker logs --tail" in cmd:
            return tail_output
        return ""

    with mock.patch.multiple(
        "runplz.backends.brev",
        _require_brev_cli=mock.DEFAULT,
        _skip_onboarding=mock.DEFAULT,
        _instance_exists=mock.Mock(return_value=True),
        _refresh_ssh=mock.DEFAULT,
        _wait_until_ssh_reachable=mock.DEFAULT,
        _prepare_remote_run=mock.DEFAULT,
        _ensure_docker=mock.DEFAULT,
        _rsync_up=mock.DEFAULT,
        _remote_has_nvidia=mock.Mock(return_value=False),
        _build_image=mock.DEFAULT,
        _run_container_detached=mock.DEFAULT,
        _stream_and_wait=mock.Mock(return_value=137),
        _ssh_capture=fake_ssh_capture,
        _rsync_down=mock.DEFAULT,
        _apply_on_finish=mock.DEFAULT,
    ):
        # _fetch_failure_tail lives in _ssh_common and calls _ssh_capture
        # via its own module namespace — patch it there too.
        with mock.patch("runplz.backends._ssh_common._ssh_capture", fake_ssh_capture):
            with pytest.raises(RuntimeError) as ei:
                brev.run(app, fn, [], {}, instance="box")

    msg = str(ei.value)
    assert "exited with status 137" in msg
    assert "RuntimeError: oops" in msg
    assert "--- last" in msg and "lines of remote output ---" in msg


def test_run_container_mode_nonzero_exit_includes_remote_log_tail(tmp_path):
    """Container-mode tees bootstrap output to $HOME/.runplz-last.log;
    _fetch_failure_tail tails that file when no docker container exists."""
    cfg = BrevConfig(mode="container", on_finish="leave")
    app = _app(tmp_path, cfg=cfg)
    fn = _function(
        app,
        Image.from_registry("pytorch/pytorch:2.4.0"),
        module_file=_job_inside(tmp_path),
    )

    tail_output = "ValueError: bad tensor shape"

    def fake_ssh_capture(instance, cmd, **kw):
        # Container-mode dispatch now tails the per-run metadata log.
        if "last.log" in cmd:
            return tail_output
        return ""

    with mock.patch.multiple(
        "runplz.backends.brev",
        _require_brev_cli=mock.DEFAULT,
        _skip_onboarding=mock.DEFAULT,
        _instance_exists=mock.Mock(return_value=True),
        _refresh_ssh=mock.DEFAULT,
        _wait_until_ssh_reachable=mock.DEFAULT,
        _prepare_remote_run=mock.DEFAULT,
        _ensure_remote_rsync=mock.DEFAULT,
        _rsync_up=mock.DEFAULT,
        _run_container_mode=mock.Mock(return_value=1),
        _ssh_capture=fake_ssh_capture,
        _rsync_down=mock.DEFAULT,
    ):
        with mock.patch("runplz.backends._ssh_common._ssh_capture", fake_ssh_capture):
            with mock.patch(
                "runplz.backends.brev.subprocess.run",
                return_value=mock.Mock(returncode=0, stdout="", stderr=""),
            ):
                with pytest.raises(RuntimeError) as ei:
                    brev.run(app, fn, [], {}, instance="box")

    msg = str(ei.value)
    assert "exited with status 1" in msg
    assert "ValueError: bad tensor shape" in msg


# -- _start_instance_if_stopped / _instance_status (issue: auto-start) ----


def test_instance_status_returns_none_when_instance_not_listed():
    with mock.patch(
        "runplz.backends.brev.subprocess.run",
        return_value=mock.Mock(
            returncode=0, stdout=json.dumps([{"name": "other-box", "status": "RUNNING"}])
        ),
    ):
        assert brev._instance_status("gone") is None


def test_instance_status_returns_status_field():
    with mock.patch(
        "runplz.backends.brev.subprocess.run",
        return_value=mock.Mock(
            returncode=0, stdout=json.dumps([{"name": "my-box", "status": "STOPPED"}])
        ),
    ):
        assert brev._instance_status("my-box") == "STOPPED"


def test_instance_status_tolerates_alternate_field_names():
    # Brev has used "state" and "power_state" in past schema versions.
    with mock.patch(
        "runplz.backends.brev.subprocess.run",
        return_value=mock.Mock(
            returncode=0, stdout=json.dumps([{"name": "my-box", "power_state": "paused"}])
        ),
    ):
        assert brev._instance_status("my-box") == "paused"


def test_start_instance_if_stopped_issues_brev_start():
    calls = []

    def fake_run(cmd, *a, **kw):
        calls.append(list(cmd))
        if cmd[:2] == ["brev", "ls"]:
            return mock.Mock(
                returncode=0, stdout=json.dumps([{"name": "my-box", "status": "STOPPED"}])
            )
        return mock.Mock(returncode=0, stdout="", stderr="")

    with mock.patch("runplz.backends.brev.subprocess.run", fake_run):
        brev._start_instance_if_stopped("my-box")

    start_calls = [c for c in calls if c[:2] == ["brev", "start"]]
    assert start_calls == [["brev", "start", "my-box"]]


def test_start_instance_if_stopped_noop_when_running():
    calls = []

    def fake_run(cmd, *a, **kw):
        calls.append(list(cmd))
        return mock.Mock(returncode=0, stdout=json.dumps([{"name": "my-box", "status": "RUNNING"}]))

    with mock.patch("runplz.backends.brev.subprocess.run", fake_run):
        brev._start_instance_if_stopped("my-box")

    # Only the ls probe; never a `brev start`.
    assert all(c[:2] != ["brev", "start"] for c in calls)


def test_start_instance_if_stopped_silent_when_status_unknown():
    # Missing status field → skip (SSH reachability loop will handle it).
    calls = []

    def fake_run(cmd, *a, **kw):
        calls.append(list(cmd))
        return mock.Mock(returncode=0, stdout=json.dumps([{"name": "my-box"}]))

    with mock.patch("runplz.backends.brev.subprocess.run", fake_run):
        brev._start_instance_if_stopped("my-box")
    assert all(c[:2] != ["brev", "start"] for c in calls)


# -- ephemeral mode (instance=None) --------------------------------------


def test_make_ephemeral_name_shape():
    name = brev._make_ephemeral_name("My App", "train_step")
    assert name.startswith("runplz-my-app-train-step-")
    # 8-char uuid suffix = 32 hex chars / 4
    suffix = name.rsplit("-", 1)[-1]
    assert len(suffix) == 8
    assert all(c in "0123456789abcdef" for c in suffix)


def test_make_ephemeral_name_sanitizes_slashes_and_dots():
    # App/function names can have characters Brev won't accept.
    name = brev._make_ephemeral_name("openvax/runplz", "train.v2")
    assert " " not in name
    assert "/" not in name
    assert "." not in name
    assert name.startswith("runplz-openvax-runplz-train-v2-")


def test_run_ephemeral_mode_generates_name_and_forces_delete(tmp_path):
    """instance=None → runplz generates a name, forces auto_create + delete."""
    cfg = BrevConfig(mode="vm", use_docker=True)  # default on_finish="stop"
    app = _app(tmp_path, cfg=cfg)
    fn = _function(app, Image.from_registry("ubuntu:22.04"), module_file=_job_inside(tmp_path))

    captured = {}

    def capture_create(name, *, cfg=None, image=None, function=None):
        captured["create_name"] = name
        captured["create_cfg"] = cfg

    captured_finish = {}

    def capture_finish(*, instance, cfg):
        captured_finish["instance"] = instance
        captured_finish["on_finish"] = cfg.on_finish

    with mock.patch.multiple(
        "runplz.backends.brev",
        _require_brev_cli=mock.DEFAULT,
        _skip_onboarding=mock.DEFAULT,
        _instance_exists=mock.Mock(return_value=False),
        _create_instance=capture_create,
        _refresh_ssh=mock.DEFAULT,
        _wait_until_ssh_reachable=mock.DEFAULT,
        _prepare_remote_run=mock.DEFAULT,
        _ensure_docker=mock.DEFAULT,
        _rsync_up=mock.DEFAULT,
        _remote_has_nvidia=mock.Mock(return_value=False),
        _build_image=mock.DEFAULT,
        _run_container_detached=mock.DEFAULT,
        _stream_and_wait=mock.Mock(return_value=0),
        _ssh_capture=mock.DEFAULT,
        _rsync_down=mock.DEFAULT,
        _apply_on_finish=capture_finish,
    ):
        brev.run(app, fn, [], {}, instance=None)

    # Name was generated — `_app` helper uses app name "panalle".
    assert captured["create_name"].startswith("runplz-panalle-train-")
    # Auto-create was forced on for the ephemeral path even though the App's
    # BrevConfig defaults it to False.
    assert captured["create_cfg"].auto_create_instances is True
    # And "stop" was upgraded to "delete" so we don't leak a billed box
    # that nothing will ever reuse.
    assert captured_finish["on_finish"] == "delete"


def test_run_ephemeral_preserves_explicit_on_finish_leave(tmp_path):
    """If the user pinned on_finish='leave' explicitly, respect it — they're
    probably debugging and want the box to stick around."""
    cfg = BrevConfig(mode="vm", use_docker=True, on_finish="leave")
    app = _app(tmp_path, cfg=cfg)
    fn = _function(app, Image.from_registry("ubuntu:22.04"), module_file=_job_inside(tmp_path))

    captured_finish = {}

    def capture_finish(*, instance, cfg):
        captured_finish["on_finish"] = cfg.on_finish

    with mock.patch.multiple(
        "runplz.backends.brev",
        _require_brev_cli=mock.DEFAULT,
        _skip_onboarding=mock.DEFAULT,
        _instance_exists=mock.Mock(return_value=False),
        _create_instance=mock.DEFAULT,
        _refresh_ssh=mock.DEFAULT,
        _wait_until_ssh_reachable=mock.DEFAULT,
        _prepare_remote_run=mock.DEFAULT,
        _ensure_docker=mock.DEFAULT,
        _rsync_up=mock.DEFAULT,
        _remote_has_nvidia=mock.Mock(return_value=False),
        _build_image=mock.DEFAULT,
        _run_container_detached=mock.DEFAULT,
        _stream_and_wait=mock.Mock(return_value=0),
        _ssh_capture=mock.DEFAULT,
        _rsync_down=mock.DEFAULT,
        _apply_on_finish=capture_finish,
    ):
        brev.run(app, fn, [], {}, instance=None)

    assert captured_finish["on_finish"] == "leave"


def test_run_named_instance_triggers_auto_start_when_stopped(tmp_path):
    """Existing instance that's stopped (from a prior on_finish='stop' run)
    must be `brev start`-ed before the dispatch tries to SSH."""
    cfg = BrevConfig(mode="vm", use_docker=True, on_finish="leave")
    app = _app(tmp_path, cfg=cfg)
    fn = _function(app, Image.from_registry("ubuntu:22.04"), module_file=_job_inside(tmp_path))

    start_calls = []

    def fake_start(name):
        start_calls.append(name)

    with mock.patch.multiple(
        "runplz.backends.brev",
        _require_brev_cli=mock.DEFAULT,
        _skip_onboarding=mock.DEFAULT,
        _instance_exists=mock.Mock(return_value=True),
        _start_instance_if_stopped=fake_start,
        _refresh_ssh=mock.DEFAULT,
        _wait_until_ssh_reachable=mock.DEFAULT,
        _prepare_remote_run=mock.DEFAULT,
        _ensure_docker=mock.DEFAULT,
        _rsync_up=mock.DEFAULT,
        _remote_has_nvidia=mock.Mock(return_value=False),
        _build_image=mock.DEFAULT,
        _run_container_detached=mock.DEFAULT,
        _stream_and_wait=mock.Mock(return_value=0),
        _ssh_capture=mock.DEFAULT,
        _rsync_down=mock.DEFAULT,
        _apply_on_finish=mock.DEFAULT,
    ):
        brev.run(app, fn, [], {}, instance="my-box")

    assert start_calls == ["my-box"], "existing-instance path must call _start_instance_if_stopped"


# -- #29: try/finally widens to cover post-create setup ----------------


def test_setup_failure_after_create_still_runs_on_finish(tmp_path):
    """Regression for #29: if _refresh_ssh / _wait_until_ssh_reachable /
    _rsync_up raises *after* _create_instance succeeds, the Brev box is
    already billed — runplz must still fire _apply_on_finish so the box
    doesn't leak."""
    cfg = BrevConfig(auto_create_instances=True, mode="vm", use_docker=True, on_finish="delete")
    app = _app(tmp_path, cfg=cfg)
    fn = _function(app, Image.from_registry("ubuntu:22.04"), module_file=_job_inside(tmp_path))

    finish_calls = []

    def capture_finish(*, instance, cfg):
        finish_calls.append((instance, cfg.on_finish))

    create_calls = []

    def fake_create(name, *, cfg=None, image=None, function=None):
        create_calls.append(name)

    with mock.patch.multiple(
        "runplz.backends.brev",
        _require_brev_cli=mock.DEFAULT,
        _skip_onboarding=mock.DEFAULT,
        _instance_exists=mock.Mock(return_value=False),
        _create_instance=fake_create,
        # _refresh_ssh raises — simulates the Brev API transient hitting
        # a post-create setup step. Before #29 this would leak the box.
        _refresh_ssh=mock.Mock(side_effect=RuntimeError("rpc error: context deadline exceeded")),
        _wait_until_ssh_reachable=mock.DEFAULT,
        _prepare_remote_run=mock.DEFAULT,
        _ensure_docker=mock.DEFAULT,
        _rsync_up=mock.DEFAULT,
        _remote_has_nvidia=mock.Mock(return_value=False),
        _build_image=mock.DEFAULT,
        _run_container_detached=mock.DEFAULT,
        _stream_and_wait=mock.Mock(return_value=0),
        _ssh_capture=mock.DEFAULT,
        _rsync_down=mock.DEFAULT,
        _apply_on_finish=capture_finish,
    ):
        with pytest.raises(RuntimeError, match="context deadline"):
            brev.run(app, fn, [], {}, instance="new-box")

    assert create_calls == ["new-box"]
    # Crucially: _apply_on_finish DID fire despite the post-create failure.
    assert finish_calls == [("new-box", "delete")]


def test_typo_with_auto_create_false_raises_without_calling_on_finish(tmp_path):
    """Negative: a typoed --instance name with auto_create_instances=False
    must raise BEFORE the try/finally — there's no box to clean up, and
    running `brev stop` on a nonexistent name would be noise."""
    cfg = BrevConfig(auto_create_instances=False)
    app = _app(tmp_path, cfg=cfg)
    fn = _function(app, Image.from_registry("ubuntu:22.04"), module_file=_job_inside(tmp_path))

    finish_calls = []

    def capture_finish(*, instance, cfg):  # pragma: no cover — should not fire
        finish_calls.append(instance)

    with mock.patch.multiple(
        "runplz.backends.brev",
        _require_brev_cli=mock.DEFAULT,
        _skip_onboarding=mock.DEFAULT,
        _instance_exists=mock.Mock(return_value=False),
        _apply_on_finish=capture_finish,
    ):
        with pytest.raises(RuntimeError, match="not found"):
            brev.run(app, fn, [], {}, instance="typoed")

    assert finish_calls == [], "on_finish must NOT run when typo-guard raises"


# -- #38: SIGTERM → on_finish still fires ------------------------------


def test_sigterm_during_dispatch_triggers_on_finish(tmp_path):
    """Regression for #38: kill -TERM on the orchestrator must run
    _apply_on_finish instead of silently exiting the process."""
    cfg = BrevConfig(auto_create_instances=True, mode="vm", use_docker=True, on_finish="delete")
    app = _app(tmp_path, cfg=cfg)
    fn = _function(app, Image.from_registry("ubuntu:22.04"), module_file=_job_inside(tmp_path))

    finish_calls = []

    def capture_finish(*, instance, cfg):
        finish_calls.append(instance)

    def fake_stream(instance, container_name, max_runtime_seconds=None):
        # Simulate the orchestrator receiving SIGTERM while the remote
        # training is streaming. Raise the context manager's translated
        # exception — which is what signal.signal's handler would do.
        raise brev._OrchestratorKilled("runplz orchestrator killed by SIGTERM")

    with mock.patch.multiple(
        "runplz.backends.brev",
        _require_brev_cli=mock.DEFAULT,
        _skip_onboarding=mock.DEFAULT,
        _instance_exists=mock.Mock(return_value=False),
        _create_instance=mock.DEFAULT,
        _refresh_ssh=mock.DEFAULT,
        _wait_until_ssh_reachable=mock.DEFAULT,
        _prepare_remote_run=mock.DEFAULT,
        _ensure_docker=mock.DEFAULT,
        _rsync_up=mock.DEFAULT,
        _remote_has_nvidia=mock.Mock(return_value=False),
        _build_image=mock.DEFAULT,
        _run_container_detached=mock.DEFAULT,
        _stream_and_wait=fake_stream,
        _ssh_capture=mock.DEFAULT,
        _rsync_down=mock.DEFAULT,
        _apply_on_finish=capture_finish,
    ):
        with pytest.raises(brev._OrchestratorKilled):
            brev.run(app, fn, [], {}, instance=None)

    assert len(finish_calls) == 1
    assert finish_calls[0].startswith("runplz-panalle-train-")


def test_orchestrator_signal_cleanup_installs_and_restores_handlers():
    """The context manager must install a handler while active and
    restore the original on exit. Verifies we don't leak a handler
    into the caller's process."""
    import signal

    original = signal.getsignal(signal.SIGTERM)
    with brev._orchestrator_signal_cleanup("x"):
        installed = signal.getsignal(signal.SIGTERM)
        assert installed is not original  # our handler is active
    # Restored.
    assert signal.getsignal(signal.SIGTERM) is original


# -- #28: _refresh_ssh retries on transient Brev errors ---------------


def test_refresh_ssh_retries_on_rpc_context_deadline_exceeded():
    """First call returns `rpc error: context deadline exceeded`; second
    succeeds. _refresh_ssh must ride through the transient."""
    attempts = []

    def fake_run(cmd, *a, **kw):
        attempts.append(list(cmd))
        if len(attempts) == 1:
            return mock.Mock(
                returncode=1,
                stdout="",
                stderr="rpc error: code = Internal desc = context deadline exceeded",
            )
        return mock.Mock(returncode=0, stdout="", stderr="")

    with mock.patch("runplz.backends.brev.subprocess.run", fake_run):
        with mock.patch("time.sleep", lambda _s: None):
            brev._refresh_ssh()  # must not raise
    assert len(attempts) == 2


def test_refresh_ssh_gives_up_after_all_attempts_fail():
    """If every retry hits a transient error, raise with context."""

    def fake_run(cmd, *a, **kw):
        return mock.Mock(returncode=1, stdout="", stderr="rpc error: eof")

    with mock.patch("runplz.backends.brev.subprocess.run", fake_run):
        with mock.patch("time.sleep", lambda _s: None):
            with pytest.raises(RuntimeError, match="failed after"):
                brev._refresh_ssh()


def test_refresh_ssh_does_not_retry_non_transient_errors():
    """`brev refresh` failing with an auth error isn't transient — retry
    would waste time. Raise on the first attempt."""
    attempts = []

    def fake_run(cmd, *a, **kw):
        attempts.append(list(cmd))
        return mock.Mock(
            returncode=1, stdout="", stderr="you are not authenticated — run `brev login`"
        )

    with mock.patch("runplz.backends.brev.subprocess.run", fake_run):
        with mock.patch("time.sleep", lambda _s: None):
            with pytest.raises(RuntimeError, match="failed after"):
                brev._refresh_ssh()
    assert len(attempts) == 1, "non-transient errors should not retry"


# -- 3.8.0: every brev CLI call goes through the retry wrapper --------


def test_brev_capture_retries_on_http_500():
    """HTTP 500 from Brev's control plane is transient — retry."""
    attempts = []

    def fake_run(cmd, *a, **kw):
        attempts.append(list(cmd))
        if len(attempts) == 1:
            return mock.Mock(returncode=1, stdout="", stderr="HTTP 500 Internal Server Error")
        return mock.Mock(returncode=0, stdout="", stderr="")

    with mock.patch("runplz.backends.brev.subprocess.run", fake_run):
        with mock.patch("time.sleep", lambda _s: None):
            r = brev._brev_capture(["brev", "ls", "--json"], label="brev ls")
    assert r.returncode == 0
    assert len(attempts) == 2


def test_brev_capture_retries_on_unexpected_eof():
    attempts = []

    def fake_run(cmd, *a, **kw):
        attempts.append(list(cmd))
        if len(attempts) < 3:
            return mock.Mock(returncode=1, stdout="", stderr="unexpected EOF")
        return mock.Mock(returncode=0, stdout="ok", stderr="")

    with mock.patch("runplz.backends.brev.subprocess.run", fake_run):
        with mock.patch("time.sleep", lambda _s: None):
            r = brev._brev_capture(["brev", "ls"], label="t")
    assert r.returncode == 0
    assert len(attempts) == 3


def test_brev_capture_retries_on_timeout():
    """subprocess.TimeoutExpired is always transient (slow API)."""
    attempts = []

    def fake_run(cmd, *a, **kw):
        attempts.append(list(cmd))
        if len(attempts) < 2:
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=1)
        return mock.Mock(returncode=0, stdout="", stderr="")

    with mock.patch("runplz.backends.brev.subprocess.run", fake_run):
        with mock.patch("time.sleep", lambda _s: None):
            r = brev._brev_capture(["brev", "refresh"], label="brev refresh")
    assert r.returncode == 0
    assert len(attempts) == 2


def test_brev_capture_returns_final_failure_on_non_transient():
    """Non-transient errors short-circuit — the caller gets the
    CompletedProcess to inspect rather than an exception."""
    attempts = []

    def fake_run(cmd, *a, **kw):
        attempts.append(list(cmd))
        return mock.Mock(returncode=1, stdout="", stderr="CREATE_FAILED: shadeform not_found")

    with mock.patch("runplz.backends.brev.subprocess.run", fake_run):
        with mock.patch("time.sleep", lambda _s: None):
            r = brev._brev_capture(["brev", "create", "x"], label="brev create")
    assert r.returncode == 1
    assert len(attempts) == 1  # broker said no — don't waste retries


def test_instance_exists_retries_context_deadline_exceeded():
    """Attempt 7 from the real report: `brev ls --json context deadline
    exceeded`. With 3.8.0 this must retry rather than raise."""
    attempts = []

    def fake_run(cmd, *a, **kw):
        attempts.append(list(cmd))
        if len(attempts) < 2:
            return mock.Mock(
                returncode=1,
                stdout="",
                stderr="context deadline exceeded",
            )
        return mock.Mock(returncode=0, stdout='[{"name": "my-box"}]', stderr="")

    with mock.patch("runplz.backends.brev.subprocess.run", fake_run):
        with mock.patch("time.sleep", lambda _s: None):
            assert brev._instance_exists("my-box") is True
    assert len(attempts) == 2


def test_create_instance_retries_http_500():
    """Attempt 6 from the real report: HTTP 500 from brevapi. With 3.8.0
    the retry loop rides through it."""
    attempts = []

    def fake_run(cmd, *a, **kw):
        attempts.append(list(cmd))
        if attempts[-1][:2] == ["brev", "create"]:
            if len(attempts) == 1:
                return mock.Mock(
                    returncode=1,
                    stdout="",
                    stderr="HTTP 500 Internal Server Error from brevapi",
                )
            return mock.Mock(returncode=0, stdout="", stderr="")
        return mock.Mock(returncode=0, stdout="", stderr="")

    cfg = BrevConfig(auto_create_instances=True, mode="vm", instance_type="some-type")
    fn = types.SimpleNamespace(
        name="t",
        gpu=None,
        min_cpu=None,
        min_memory=None,
        min_gpu_memory=None,
        min_disk=None,
        num_gpus=1,
    )
    img = Image.from_registry("ubuntu:22.04")
    with mock.patch("runplz.backends.brev.subprocess.run", fake_run):
        with mock.patch("time.sleep", lambda _s: None):
            brev._create_instance("my-new-box", cfg=cfg, image=img, function=fn)

    create_calls = [c for c in attempts if c[:2] == ["brev", "create"]]
    assert len(create_calls) == 2  # retried and succeeded


def test_create_instance_already_exists_treated_as_success():
    """The HTTP-500-after-create scenario: Brev actually *did* create the
    box but the response came back as 500. Our retry then gets 'already
    exists', and we verify via `brev ls` that it's really there."""
    attempts = []

    def fake_run(cmd, *a, **kw):
        attempts.append(list(cmd))
        if cmd[:2] == ["brev", "create"]:
            return mock.Mock(
                returncode=1,
                stdout="",
                stderr="workspace with this name already exists",
            )
        if cmd[:2] == ["brev", "ls"]:
            return mock.Mock(
                returncode=0,
                stdout='[{"name": "idempotent-box"}]',
                stderr="",
            )
        return mock.Mock(returncode=0, stdout="", stderr="")

    cfg = BrevConfig(auto_create_instances=True, mode="vm", instance_type="t")
    fn = types.SimpleNamespace(
        name="t",
        gpu=None,
        min_cpu=None,
        min_memory=None,
        min_gpu_memory=None,
        min_disk=None,
        num_gpus=1,
    )
    img = Image.from_registry("ubuntu:22.04")
    with mock.patch("runplz.backends.brev.subprocess.run", fake_run):
        with mock.patch("time.sleep", lambda _s: None):
            # Must not raise — "already exists" + confirmed-by-ls == success.
            brev._create_instance("idempotent-box", cfg=cfg, image=img, function=fn)


def test_create_instance_already_exists_but_not_listed_raises():
    """If `brev create` says 'already exists' but `brev ls` doesn't
    confirm, that's a confusing state — don't silently pretend it worked."""

    def fake_run(cmd, *a, **kw):
        if cmd[:2] == ["brev", "create"]:
            return mock.Mock(returncode=1, stdout="", stderr="already exists")
        if cmd[:2] == ["brev", "ls"]:
            return mock.Mock(returncode=0, stdout="[]", stderr="")
        return mock.Mock(returncode=0, stdout="", stderr="")

    cfg = BrevConfig(auto_create_instances=True, mode="vm", instance_type="t")
    fn = types.SimpleNamespace(
        name="t",
        gpu=None,
        min_cpu=None,
        min_memory=None,
        min_gpu_memory=None,
        min_disk=None,
        num_gpus=1,
    )
    img = Image.from_registry("ubuntu:22.04")
    with mock.patch("runplz.backends.brev.subprocess.run", fake_run):
        with mock.patch("time.sleep", lambda _s: None):
            with pytest.raises(RuntimeError, match="brev create"):
                brev._create_instance("ghost", cfg=cfg, image=img, function=fn)


def test_brev_capture_bails_early_on_missing_cloudcredid():
    """Issue #62: org/config-gap errors like missing OCI cloudCredId must
    NOT eat the full retry budget — they're guaranteed-fail until a human
    fixes them in the Brev console."""
    attempts = []

    def fake_run(cmd, *a, **kw):
        attempts.append(list(cmd))
        return mock.Mock(
            returncode=1,
            stdout="",
            stderr="cloudCredId or workspaceGroupId must be specified on request",
        )

    with mock.patch("runplz.backends.brev.subprocess.run", fake_run):
        with mock.patch("time.sleep", lambda _s: None):
            r = brev._brev_capture(["brev", "create", "x"], label="brev create")
    assert r.returncode == 1
    # 1 attempt, not the full 4 — non-retriable error.
    assert len(attempts) == 1


def test_brev_capture_bails_early_on_quota_exceeded():
    """Quota exhaustion is server-side capacity — retrying won't help."""
    attempts = []

    def fake_run(cmd, *a, **kw):
        attempts.append(list(cmd))
        return mock.Mock(
            returncode=1,
            stdout="",
            stderr="ERROR: quota exceeded for region us-west-2",
        )

    with mock.patch("runplz.backends.brev.subprocess.run", fake_run):
        with mock.patch("time.sleep", lambda _s: None):
            brev._brev_capture(["brev", "create", "x"], label="brev create")
    assert len(attempts) == 1


def test_brev_capture_still_retries_genuine_transient():
    """Sanity: the early-bail logic doesn't accidentally short-circuit the
    transient-retry path that 3.8.0 introduced."""
    attempts = []

    def fake_run(cmd, *a, **kw):
        attempts.append(list(cmd))
        if len(attempts) < 3:
            return mock.Mock(returncode=1, stdout="", stderr="HTTP 500")
        return mock.Mock(returncode=0, stdout="ok", stderr="")

    with mock.patch("runplz.backends.brev.subprocess.run", fake_run):
        with mock.patch("time.sleep", lambda _s: None):
            r = brev._brev_capture(["brev", "ls"], label="brev ls")
    assert r.returncode == 0
    assert len(attempts) == 3


def test_create_instance_reframes_cloudcredid_error():
    """The raw API string is opaque — runplz should translate it into
    actionable guidance (issue #62)."""

    def fake_run(cmd, *a, **kw):
        if cmd[:2] == ["brev", "create"]:
            return mock.Mock(
                returncode=1,
                stdout="",
                stderr="cloudCredId or workspaceGroupId must be specified on request",
            )
        # Any subsequent ls / snapshot — empty.
        return mock.Mock(returncode=0, stdout="[]", stderr="")

    cfg = BrevConfig(
        auto_create_instances=True, mode="vm", instance_type="oci.a100x8.sxm.brev-dgxc"
    )
    fn = types.SimpleNamespace(
        name="t",
        gpu=None,
        min_cpu=None,
        min_memory=None,
        min_gpu_memory=None,
        min_disk=None,
        num_gpus=1,
    )
    img = Image.from_registry("ubuntu:22.04")
    with mock.patch("runplz.backends.brev.subprocess.run", fake_run):
        with mock.patch("time.sleep", lambda _s: None):
            with pytest.raises(RuntimeError) as ei:
                brev._create_instance("release-exact", cfg=cfg, image=img, function=fn)
    msg = str(ei.value)
    assert "no cloud credential" in msg
    assert "oci.a100x8.sxm.brev-dgxc" in msg
    assert "Brev web UI" in msg
    # The raw API error is appended at the end so users / support can grep for it.
    assert "cloudCredId" in msg


def test_looks_non_retriable_recognizes_known_patterns():
    assert brev._looks_non_retriable("cloudCredId or workspaceGroupId must be specified on request")
    assert brev._looks_non_retriable("ERROR: quota exceeded for this region")
    assert brev._looks_non_retriable("provider not enabled in this org")
    assert brev._looks_non_retriable("401 Unauthorized")
    # Genuine transient — must NOT match.
    assert not brev._looks_non_retriable("HTTP 500 Internal Server Error")
    assert not brev._looks_non_retriable("context deadline exceeded")
    assert not brev._looks_non_retriable("")


def test_check_terminal_state_raises_for_failure_status():
    """3.8.0: if `brev ls` shows the instance in FAILURE (Nebius / shadeform
    provisioning died at the provider layer), we must bail early from
    the SSH-reachable poll instead of burning the full 30-min budget."""
    with mock.patch(
        "runplz.backends.brev.subprocess.run",
        return_value=mock.Mock(
            returncode=0,
            stdout=json.dumps([{"name": "doomed", "status": "FAILURE"}]),
            stderr="",
        ),
    ):
        with pytest.raises(brev.BrevInstanceFailed, match="FAILURE"):
            brev._check_terminal_state("doomed")


def test_check_terminal_state_noop_on_running_status():
    with mock.patch(
        "runplz.backends.brev.subprocess.run",
        return_value=mock.Mock(
            returncode=0,
            stdout=json.dumps([{"name": "doomed", "status": "STARTING"}]),
            stderr="",
        ),
    ):
        brev._check_terminal_state("doomed")  # must not raise


def test_check_terminal_state_raises_for_deploying_failed():
    with mock.patch(
        "runplz.backends.brev.subprocess.run",
        return_value=mock.Mock(
            returncode=0,
            stdout=json.dumps([{"name": "h100", "status": "DEPLOYING_FAILED"}]),
            stderr="",
        ),
    ):
        with pytest.raises(brev.BrevInstanceFailed, match="DEPLOYING_FAILED"):
            brev._check_terminal_state("h100")


def test_wait_until_ssh_reachable_bails_on_brev_instance_failed(monkeypatch):
    """When the refresh callback raises BrevInstanceFailed, the SSH-
    reachable loop must stop probing and propagate — not swallow it as
    a best-effort callback hiccup."""
    from runplz.backends import _ssh_common

    monkeypatch.setattr(_ssh_common, "SSH_OPTS", [])
    clock = [0.0]
    monkeypatch.setattr("time.time", lambda: clock[0])
    monkeypatch.setattr("time.sleep", lambda s: clock.__setitem__(0, clock[0] + s))

    def always_refused(cmd, *a, **kw):
        return mock.Mock(returncode=255, stderr="refused", stdout="")

    call_count = {"n": 0}

    def failing_callback():
        call_count["n"] += 1
        raise brev.BrevInstanceFailed("status=FAILURE")

    with mock.patch("runplz.backends._ssh_common.subprocess.run", always_refused):
        with pytest.raises(brev.BrevInstanceFailed, match="FAILURE"):
            brev._wait_until_ssh_reachable(
                "doomed",
                max_wait_s=120,
                probe_interval_s=1,
                refresh_callback=failing_callback,
            )

    # Should have bailed after the first refresh_callback invocation
    # (fires every 4 probes) — not the full 120s of probing.
    assert call_count["n"] == 1


def test_apply_on_finish_retries_transient_then_succeeds(capsys):
    """`brev stop` hitting a transient should retry, not loud-warn on
    the first blip. (Billing-leak relevance: the finally block must not
    give up on the first hiccup.)"""
    cfg = BrevConfig(mode="vm", use_docker=True)  # on_finish default = "stop"
    attempts = []

    def fake_run(cmd, *a, **kw):
        attempts.append(list(cmd))
        if (
            cmd[:2] == ["brev", "stop"]
            and len([c for c in attempts if c[:2] == ["brev", "stop"]]) < 2
        ):
            return mock.Mock(returncode=1, stdout="", stderr="context deadline exceeded")
        if cmd[:2] == ["brev", "ls"]:
            return mock.Mock(
                returncode=0,
                stdout=json.dumps([{"name": "box", "status": "STOPPED"}]),
                stderr="",
            )
        return mock.Mock(returncode=0, stdout="", stderr="")

    with mock.patch("runplz.backends.brev.subprocess.run", fake_run):
        with mock.patch("time.sleep", lambda _s: None):
            brev._apply_on_finish(instance="box", cfg=cfg)

    # Retried and succeeded. No "box may still be running" warning.
    out = capsys.readouterr().out
    assert "may still be running" not in out
    stop_calls = [c for c in attempts if c[:2] == ["brev", "stop"]]
    assert len(stop_calls) == 2


def test_fetch_failure_tail_returns_empty_on_ssh_error():
    """_fetch_failure_tail must never raise — the caller is already about
    to raise the real error and we don't want to mask it."""

    def boom(*a, **kw):
        raise RuntimeError("ssh went away")

    with mock.patch("runplz.backends._ssh_common._ssh_capture", boom):
        out = brev._fetch_failure_tail(target="box", container_name=None)
    # Helper swallows the error and returns a diagnostic string.
    assert "could not fetch remote log tail" in out
    assert "ssh went away" in out


# -- max_runtime_seconds wall-clock cap (issue #16) -----------------------


def test_raise_for_runtime_cap_kills_container_and_raises():
    """VM+docker path: trip ⇒ docker kill + RuntimeError."""
    calls = []

    def fake_sub(cmd, *a, **kw):
        calls.append(list(cmd))
        return mock.Mock(returncode=0, stdout="", stderr="")

    with mock.patch("runplz.backends._ssh_common.subprocess.run", fake_sub):
        with pytest.raises(RuntimeError) as ei:
            brev._raise_for_runtime_cap("box", 60, container_name="c-abc")

    kill_cmds = [c for c in calls if any("docker kill" in tok for tok in c)]
    assert kill_cmds, f"no docker kill issued; saw: {calls}"
    assert any("c-abc" in tok for tok in kill_cmds[0])
    assert "max_runtime_seconds=60" in str(ei.value)


def test_raise_for_runtime_cap_pkills_bootstrap_in_native_or_container_mode():
    """No container_name ⇒ pkill the bootstrap process tree."""
    calls = []

    def fake_sub(cmd, *a, **kw):
        calls.append(list(cmd))
        return mock.Mock(returncode=0, stdout="", stderr="")

    with mock.patch("runplz.backends._ssh_common.subprocess.run", fake_sub):
        with pytest.raises(RuntimeError):
            brev._raise_for_runtime_cap("box", 30, container_name=None)

    pkill_cmds = [c for c in calls if any("pkill" in tok for tok in c)]
    assert pkill_cmds, f"no pkill issued; saw: {calls}"
    assert any("runplz._bootstrap" in tok for tok in pkill_cmds[0])


def test_run_container_mode_timeout_triggers_cap():
    """A TimeoutExpired from the ssh subprocess in container-mode must route
    through _raise_for_runtime_cap → RuntimeError."""

    def fake_sub(cmd, *a, **kw):
        if kw.get("timeout") is not None:
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=kw["timeout"])
        return mock.Mock(returncode=0, stdout="", stderr="")

    fn = types.SimpleNamespace(
        name="train",
        image=Image.from_registry("ubuntu:22.04"),
        env={},
    )
    with mock.patch("runplz.backends._ssh_common.subprocess.run", fake_sub):
        with mock.patch("runplz.backends._ssh_common._render_ops_script", return_value=""):
            with pytest.raises(RuntimeError, match="max_runtime_seconds=5"):
                brev._run_container_mode(
                    target="box",
                    function=fn,
                    rel_script="jobs/j.py",
                    args=[],
                    kwargs={},
                    max_runtime_seconds=5,
                )


def test_stream_and_wait_raises_when_cap_exceeded():
    """_stream_and_wait should trip the cap when docker logs -f keeps
    streaming past the deadline."""

    def fake_sub(cmd, *a, **kw):
        if "docker logs -f" in " ".join(cmd):
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=kw.get("timeout", 0))
        return mock.Mock(returncode=0, stdout="", stderr="")

    with mock.patch("runplz.backends._ssh_common.subprocess.run", fake_sub):
        with pytest.raises(RuntimeError, match="max_runtime_seconds=2"):
            brev._stream_and_wait("box", "container-xyz", max_runtime_seconds=2)


def test_launch_detached_and_wait_falls_back_to_blocking_without_remote_run():
    """No ``remote_run`` means no meta/events files to poll, so the helper
    keeps the old synchronous ssh behavior. This preserves back-compat
    with any ad-hoc caller that doesn't construct a remote_run."""
    from runplz.backends._ssh_common import _launch_detached_and_wait

    recorded = []

    def fake_sub(cmd, *a, **kw):
        recorded.append((cmd, kw.get("timeout")))
        return mock.Mock(returncode=0, stdout="", stderr="")

    with mock.patch("runplz.backends._ssh_common.subprocess.run", fake_sub):
        rc = _launch_detached_and_wait(
            target="box",
            wrapped_command="echo hi",
            remote_run=None,
            max_runtime_seconds=42,
        )
    assert rc == 0
    assert len(recorded) == 1, "fallback path should make exactly one ssh call"
    assert recorded[0][1] == 42


def test_launch_detached_and_wait_double_quotes_home_paths_for_expansion():
    """The launcher script writes to ``$HOME``-relative paths; those must
    be double-quoted (allowing ``$HOME`` expansion) not single-quoted /
    shlex.quote'd (which would pass the literal string ``$HOME/…`` to the
    remote shell and fail under ``set -euo pipefail``).

    Regression lock for the 2026-04-24 launch bug: ``shlex.quote`` was
    wrapping paths like ``$HOME/runplz-runs/…/run.sh`` in single quotes,
    causing ``cat > '$HOME/…'`` to try to write to a literal path in
    cwd. Training never started — events log stopped at
    ``rsync_up_done`` with no ``remote_command_start``.
    """
    from runplz.backends._ssh_common import _launch_detached_and_wait

    remote_run = brev.make_remote_run_context(backend="ssh", target="box", function_name="train")
    captured = {}

    def fake_ssh(target, cmd, **kw):
        captured["cmd"] = cmd

    def fake_sub(cmd, *a, **kw):
        cmd_str = " ".join(cmd)
        if "kill -0" in cmd_str:
            return mock.Mock(returncode=0, stdout="dead", stderr="")
        if "remote_command_exit" in cmd_str:
            return mock.Mock(
                returncode=0,
                stdout='{"event":"remote_command_exit","exit_code":0}\n',
                stderr="",
            )
        return mock.Mock(returncode=0, stdout="", stderr="")

    with mock.patch("runplz.backends._ssh_common._ssh", fake_ssh):
        with mock.patch("runplz.backends._ssh_common.subprocess.run", fake_sub):
            _launch_detached_and_wait(
                target="box",
                wrapped_command="echo hi",
                remote_run=remote_run,
            )
    launcher = captured["cmd"]
    # ``cat >`` target must be double-quoted (so $HOME expands), not
    # single-quoted. Same for chmod / nohup / echo-pid targets.
    assert f'cat > "{remote_run.meta_shell}/run.sh"' in launcher
    assert f'chmod +x "{remote_run.meta_shell}/run.sh"' in launcher
    assert f'nohup setsid bash "{remote_run.meta_shell}/run.sh"' in launcher
    assert f'echo $! > "{remote_run.meta_shell}/bootstrap.pid"' in launcher
    # And explicitly NOT the shlex.quote form that broke the previous
    # release.
    assert f"'{remote_run.meta_shell}/run.sh'" not in launcher, (
        "shlex-quoted path would suppress $HOME expansion on the remote"
    )


def test_launch_detached_and_wait_writes_pid_and_uses_setsid_nohup(tmp_path):
    """The launcher script must include all three detach ingredients:
    setsid (new session), nohup (SIGHUP-proof + /dev/null stdin), and
    stdout+stderr redirected to a file (no pipe to the ssh socket).

    Missing any one of these leaves the process tethered to the ssh
    session — the bug we're fixing."""
    from runplz.backends._ssh_common import _launch_detached_and_wait

    remote_run = brev.make_remote_run_context(backend="ssh", target="box", function_name="train")
    launcher_seen = {}

    def fake_ssh(target, cmd, **kw):
        launcher_seen["cmd"] = cmd

    def fake_sub(cmd, *a, **kw):
        cmd_str = " ".join(cmd)
        if "kill -0" in cmd_str:
            return mock.Mock(returncode=0, stdout="dead", stderr="")
        if "remote_command_exit" in cmd_str:
            return mock.Mock(
                returncode=0,
                stdout='{"event":"remote_command_exit","exit_code":0}\n',
                stderr="",
            )
        return mock.Mock(returncode=0, stdout="", stderr="")

    with mock.patch("runplz.backends._ssh_common._ssh", fake_ssh):
        with mock.patch("runplz.backends._ssh_common.subprocess.run", fake_sub):
            rc = _launch_detached_and_wait(
                target="box",
                wrapped_command="echo hi",
                remote_run=remote_run,
            )
    assert rc == 0
    cmd = launcher_seen["cmd"]
    assert "setsid" in cmd
    assert "nohup" in cmd
    assert "</dev/null" in cmd
    assert "bootstrap.pid" in cmd
    # Heredoc carries the user command through verbatim.
    assert "echo hi" in cmd


def test_tail_and_wait_reconnects_when_pid_still_alive():
    """If ``tail -F`` exits while the remote pid is still live, the helper
    must reconnect. Direct test of the resilience we care about."""
    from runplz.backends._ssh_common import _tail_and_wait_for_detached

    call_log = []
    # Non-exhausting: first 2 pid checks say alive (forcing reconnects),
    # the rest say dead so both the main loop break AND the post-break
    # "wait until clear" loop exit cleanly. Using a counter so the
    # iterator can't run out mid-test.
    pid_check_count = {"n": 0}

    def fake_sub(cmd, *a, **kw):
        cmd_str = " ".join(cmd)
        call_log.append(cmd_str)
        if "tail -n +1 -F" in cmd_str:
            return mock.Mock(returncode=255, stdout="", stderr="")
        if "kill -0" in cmd_str:
            pid_check_count["n"] += 1
            state = "alive" if pid_check_count["n"] <= 2 else "dead"
            return mock.Mock(returncode=0, stdout=state, stderr="")
        if "remote_command_exit" in cmd_str:
            return mock.Mock(
                returncode=0,
                stdout='{"event":"remote_command_exit","exit_code":0}\n',
                stderr="",
            )
        return mock.Mock(returncode=0, stdout="", stderr="")

    with mock.patch("runplz.backends._ssh_common.subprocess.run", fake_sub):
        with mock.patch("runplz.backends._ssh_common.time.sleep"):
            rc = _tail_and_wait_for_detached(
                target="box",
                pid_file="/tmp/pid",
                log_file="/tmp/log",
                events_file="/tmp/events.ndjson",
            )
    assert rc == 0
    # Three tail invocations: initial + 2 reconnects before pid died.
    tail_calls = [c for c in call_log if "tail -n +1 -F" in c]
    assert len(tail_calls) == 3


def test_tail_and_wait_gives_up_streaming_after_max_reconnects_but_still_returns_exit_code():
    """After ``max_reconnects`` we stop re-tailing but MUST keep polling
    until the pid clears so the caller sees the real exit code."""
    from runplz.backends._ssh_common import _tail_and_wait_for_detached

    tail_calls = 0
    pid_calls = 0

    def fake_sub(cmd, *a, **kw):
        nonlocal tail_calls, pid_calls
        cmd_str = " ".join(cmd)
        if "tail -n +1 -F" in cmd_str:
            tail_calls += 1
            return mock.Mock(returncode=255, stdout="", stderr="")
        if "kill -0" in cmd_str:
            pid_calls += 1
            # First N probes report alive (forces reconnects), last probe
            # reports dead so the wait-after-giveup loop can exit.
            return mock.Mock(
                returncode=0,
                stdout="alive" if pid_calls < 4 else "dead",
                stderr="",
            )
        if "remote_command_exit" in cmd_str:
            return mock.Mock(
                returncode=0,
                stdout='{"event":"remote_command_exit","exit_code":7}\n',
                stderr="",
            )
        return mock.Mock(returncode=0, stdout="", stderr="")

    with mock.patch("runplz.backends._ssh_common.subprocess.run", fake_sub):
        with mock.patch("runplz.backends._ssh_common.time.sleep"):
            rc = _tail_and_wait_for_detached(
                target="box",
                pid_file="/tmp/pid",
                log_file="/tmp/log",
                events_file="/tmp/events.ndjson",
                max_reconnects=2,
            )
    assert rc == 7
    # After 2 reconnects (tail called 3 times: initial + 2 reconnects)
    # we stop tailing but continue polling the pid.
    assert tail_calls == 3
    assert pid_calls >= 3


def test_read_remote_exit_code_parses_last_entry():
    """The file can accumulate multiple ``remote_command_exit`` entries
    across retries; we want the last one."""
    from runplz.backends._ssh_common import _read_remote_exit_code

    def fake_sub(cmd, *a, **kw):
        return mock.Mock(
            returncode=0,
            stdout=(
                # Two lines; `tail -n 1` in the probe would strip the first.
                # We emulate that by returning only the last line.
                '{"event":"remote_command_exit","exit_code":13}\n'
            ),
            stderr="",
        )

    with mock.patch("runplz.backends._ssh_common.subprocess.run", fake_sub):
        assert _read_remote_exit_code("box", "/tmp/events") == 13


def test_read_remote_exit_code_defaults_to_1_on_missing_or_malformed():
    """Unknown exit code must NOT return 0 — a missing remote_command_exit
    entry means something went sideways before the trap fired, which is
    exactly a failure we want the caller to see as nonzero."""
    from runplz.backends._ssh_common import _read_remote_exit_code

    # Empty stdout (grep matched nothing).
    with mock.patch(
        "runplz.backends._ssh_common.subprocess.run",
        return_value=mock.Mock(returncode=0, stdout="", stderr=""),
    ):
        assert _read_remote_exit_code("box", "/tmp/events") == 1

    # Malformed JSON.
    with mock.patch(
        "runplz.backends._ssh_common.subprocess.run",
        return_value=mock.Mock(returncode=0, stdout="not-json\n", stderr=""),
    ):
        assert _read_remote_exit_code("box", "/tmp/events") == 1

    # Exit code missing from the entry.
    with mock.patch(
        "runplz.backends._ssh_common.subprocess.run",
        return_value=mock.Mock(
            returncode=0,
            stdout='{"event":"remote_command_exit"}\n',
            stderr="",
        ),
    ):
        assert _read_remote_exit_code("box", "/tmp/events") == 1


def test_remote_pid_alive_treats_ssh_timeout_as_alive():
    """If ssh itself hangs, we can't prove the remote job is dead, so the
    safe default is "still alive" — let the caller keep polling instead
    of prematurely returning a fake exit code."""
    from runplz.backends._ssh_common import _remote_pid_alive

    def fake_sub(cmd, *a, **kw):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=kw.get("timeout", 0))

    with mock.patch("runplz.backends._ssh_common.subprocess.run", fake_sub):
        assert _remote_pid_alive("box", "/tmp/pid") is True


def test_stream_and_wait_no_cap_passes_none_timeout():
    """Regression: when max_runtime_seconds is None, the ssh subprocess.run
    call must get timeout=None (not 0, not a negative number), so normal
    long jobs aren't killed."""
    seen_timeouts = []

    def fake_sub(cmd, *a, **kw):
        if "docker logs -f" in " ".join(cmd):
            seen_timeouts.append(kw.get("timeout"))
            # Pretend the container stopped immediately after this.
            return mock.Mock(returncode=0, stdout="", stderr="")
        if "docker inspect" in " ".join(cmd):
            # _container_running should see False → loop breaks.
            return mock.Mock(returncode=0, stdout="false", stderr="")
        if "docker wait" in " ".join(cmd):
            return mock.Mock(returncode=0, stdout="0", stderr="")
        return mock.Mock(returncode=0, stdout="", stderr="")

    with mock.patch("runplz.backends._ssh_common.subprocess.run", fake_sub):
        exit_code = brev._stream_and_wait("box", "c-1", max_runtime_seconds=None)

    assert exit_code == 0
    assert seen_timeouts == [None], f"expected None, got {seen_timeouts}"


def test_run_vm_native_mode_end_to_end(tmp_path):
    cfg = BrevConfig(mode="vm", use_docker=False)
    app = _app(tmp_path, cfg=cfg)
    fn = _function(app, Image.from_registry("ubuntu:22.04"), module_file=_job_inside(tmp_path))

    with mock.patch.multiple(
        "runplz.backends.brev",
        _require_brev_cli=mock.DEFAULT,
        _skip_onboarding=mock.DEFAULT,
        _instance_exists=mock.Mock(return_value=True),
        _refresh_ssh=mock.DEFAULT,
        _wait_until_ssh_reachable=mock.DEFAULT,
        _prepare_remote_run=mock.DEFAULT,
        _rsync_up=mock.DEFAULT,
        _remote_has_nvidia=mock.Mock(return_value=False),
        _run_native=mock.Mock(return_value=0),
        _rsync_down=mock.DEFAULT,
        _apply_on_finish=mock.DEFAULT,
    ):
        brev.run(app, fn, [], {}, instance="box")


def test_run_container_mode_end_to_end(tmp_path):
    cfg = BrevConfig(mode="container")
    app = _app(tmp_path, cfg=cfg)
    fn = _function(
        app,
        Image.from_registry("pytorch/pytorch:2.4.0").pip_install("numpy"),
        module_file=_job_inside(tmp_path),
    )

    with mock.patch.multiple(
        "runplz.backends.brev",
        _require_brev_cli=mock.DEFAULT,
        _skip_onboarding=mock.DEFAULT,
        _instance_exists=mock.Mock(return_value=True),
        _refresh_ssh=mock.DEFAULT,
        _wait_until_ssh_reachable=mock.DEFAULT,
        _prepare_remote_run=mock.DEFAULT,
        _ensure_remote_rsync=mock.DEFAULT,
        _rsync_up=mock.DEFAULT,
        _run_container_mode=mock.Mock(return_value=0),
        _rsync_down=mock.DEFAULT,
        _apply_on_finish=mock.DEFAULT,
    ):
        brev.run(app, fn, [], {}, instance="box")


def test_run_creates_instance_when_auto_create(tmp_path):
    cfg = BrevConfig(auto_create_instances=True, mode="vm")
    app = _app(tmp_path, cfg=cfg)
    fn = _function(app, Image.from_registry("ubuntu:22.04"), module_file=_job_inside(tmp_path))

    created = {}

    def fake_create(name, *, cfg=None, image=None, function=None):
        created["name"] = name
        created["function"] = function

    with mock.patch.multiple(
        "runplz.backends.brev",
        _require_brev_cli=mock.DEFAULT,
        _skip_onboarding=mock.DEFAULT,
        _instance_exists=mock.Mock(return_value=False),
        _create_instance=fake_create,
        _refresh_ssh=mock.DEFAULT,
        _wait_until_ssh_reachable=mock.DEFAULT,
        _prepare_remote_run=mock.DEFAULT,
        _ensure_docker=mock.DEFAULT,
        _rsync_up=mock.DEFAULT,
        _remote_has_nvidia=mock.Mock(return_value=False),
        _build_image=mock.DEFAULT,
        _run_container_detached=mock.DEFAULT,
        _stream_and_wait=mock.Mock(return_value=0),
        _ssh_capture=mock.DEFAULT,
        _rsync_down=mock.DEFAULT,
        _apply_on_finish=mock.DEFAULT,
    ):
        brev.run(app, fn, [], {}, instance="newbox")

    assert created["name"] == "newbox"
    assert created["function"] is fn


# -- on_finish lifecycle --------------------------------------------------


def _full_run_mocks():
    """Shared mock.multiple kwargs for brev.run() end-to-end with a clean VM-
    docker-mode path. Callers override `_stream_and_wait` to control exit code
    (and can wrap in another context to assert _apply_on_finish calls)."""
    return dict(
        _require_brev_cli=mock.DEFAULT,
        _skip_onboarding=mock.DEFAULT,
        _instance_exists=mock.Mock(return_value=True),
        _refresh_ssh=mock.DEFAULT,
        _wait_until_ssh_reachable=mock.DEFAULT,
        _prepare_remote_run=mock.DEFAULT,
        _ensure_docker=mock.DEFAULT,
        _rsync_up=mock.DEFAULT,
        _remote_has_nvidia=mock.Mock(return_value=False),
        _build_image=mock.DEFAULT,
        _run_container_detached=mock.DEFAULT,
        _stream_and_wait=mock.Mock(return_value=0),
        _ssh_capture=mock.DEFAULT,
        _rsync_down=mock.DEFAULT,
    )


def test_on_finish_default_stop_calls_brev_stop(tmp_path):
    cfg = BrevConfig(mode="vm", use_docker=True)  # on_finish default = "stop"
    app = _app(tmp_path, cfg=cfg)
    fn = _function(app, Image.from_registry("ubuntu:22.04"), module_file=_job_inside(tmp_path))

    captured = {"calls": []}

    def fake_sub(cmd, *a, **kw):
        captured["calls"].append(list(cmd))
        return mock.Mock(returncode=0, stdout="", stderr="")

    with mock.patch.multiple("runplz.backends.brev", **_full_run_mocks()):
        with mock.patch("runplz.backends.brev.subprocess.run", fake_sub):
            brev.run(app, fn, [], {}, instance="box")

    stop_calls = [c for c in captured["calls"] if c[:2] == ["brev", "stop"]]
    assert stop_calls == [["brev", "stop", "box"]]


def test_on_finish_delete_calls_brev_delete(tmp_path):
    cfg = BrevConfig(mode="vm", use_docker=True, on_finish="delete")
    app = _app(tmp_path, cfg=cfg)
    fn = _function(app, Image.from_registry("ubuntu:22.04"), module_file=_job_inside(tmp_path))

    captured = {"calls": []}

    def fake_sub(cmd, *a, **kw):
        captured["calls"].append(list(cmd))
        return mock.Mock(returncode=0, stdout="", stderr="")

    with mock.patch.multiple("runplz.backends.brev", **_full_run_mocks()):
        with mock.patch("runplz.backends.brev.subprocess.run", fake_sub):
            brev.run(app, fn, [], {}, instance="box")

    delete_calls = [c for c in captured["calls"] if c[:2] == ["brev", "delete"]]
    assert delete_calls == [["brev", "delete", "box"]]


def test_on_finish_leave_never_touches_box(tmp_path):
    cfg = BrevConfig(mode="vm", use_docker=True, on_finish="leave")
    app = _app(tmp_path, cfg=cfg)
    fn = _function(app, Image.from_registry("ubuntu:22.04"), module_file=_job_inside(tmp_path))

    captured = {"calls": []}

    def fake_sub(cmd, *a, **kw):
        captured["calls"].append(list(cmd))
        return mock.Mock(returncode=0, stdout="", stderr="")

    with mock.patch.multiple("runplz.backends.brev", **_full_run_mocks()):
        with mock.patch("runplz.backends.brev.subprocess.run", fake_sub):
            brev.run(app, fn, [], {}, instance="box")

    for c in captured["calls"]:
        assert c[:2] != ["brev", "stop"], f"unexpected stop: {c}"
        assert c[:2] != ["brev", "delete"], f"unexpected delete: {c}"


def test_on_finish_fires_even_when_remote_run_fails(tmp_path):
    """The whole point of putting cleanup in `finally` — a failed remote
    run must still trigger the configured lifecycle action."""
    cfg = BrevConfig(mode="vm", use_docker=True)  # default on_finish="stop"
    app = _app(tmp_path, cfg=cfg)
    fn = _function(app, Image.from_registry("ubuntu:22.04"), module_file=_job_inside(tmp_path))

    captured = {"calls": []}

    def fake_sub(cmd, *a, **kw):
        captured["calls"].append(list(cmd))
        return mock.Mock(returncode=0, stdout="", stderr="")

    # Simulate non-zero exit from the remote container.
    mocks = _full_run_mocks()
    mocks["_stream_and_wait"] = mock.Mock(return_value=137)

    with mock.patch.multiple("runplz.backends.brev", **mocks):
        with mock.patch("runplz.backends.brev.subprocess.run", fake_sub):
            with pytest.raises(RuntimeError, match="status 137"):
                brev.run(app, fn, [], {}, instance="box")

    stop_calls = [c for c in captured["calls"] if c[:2] == ["brev", "stop"]]
    assert stop_calls == [["brev", "stop", "box"]], "on_finish must fire even on failure"


def test_container_rm_force_called_in_finally_on_failure(tmp_path):
    """Orphaned-container guard: a failing `_stream_and_wait` used to leak the
    container on the remote box. With the finally wrap, `docker rm -f` fires."""
    cfg = BrevConfig(mode="vm", use_docker=True, on_finish="leave")
    app = _app(tmp_path, cfg=cfg)
    fn = _function(app, Image.from_registry("ubuntu:22.04"), module_file=_job_inside(tmp_path))

    ssh_calls = {"cmds": []}

    def fake_ssh_capture(instance, cmd, **kw):
        ssh_calls["cmds"].append(cmd)
        return ""

    mocks = _full_run_mocks()
    # Simulate an exception in the middle of the run (after container start).
    mocks["_rsync_down"] = mock.Mock(side_effect=RuntimeError("rsync exploded"))
    mocks["_ssh_capture"] = fake_ssh_capture

    with mock.patch.multiple("runplz.backends.brev", **mocks):
        with pytest.raises(RuntimeError, match="rsync exploded"):
            brev.run(app, fn, [], {}, instance="box")

    assert any("docker rm -f runplz-train-" in c for c in ssh_calls["cmds"]), ssh_calls["cmds"]


def test_on_finish_fires_in_container_mode(tmp_path):
    """`mode="container"` has no local docker container to rm, but the box
    itself still needs to be stopped/deleted. Guards against future
    refactors of the try/finally that might skip _apply_on_finish on the
    container-mode branch."""
    cfg = BrevConfig(mode="container", on_finish="delete")
    app = _app(tmp_path, cfg=cfg)
    fn = _function(app, Image.from_registry("ubuntu:22.04"), module_file=_job_inside(tmp_path))

    captured = {"calls": []}

    def fake_sub(cmd, *a, **kw):
        captured["calls"].append(list(cmd))
        return mock.Mock(returncode=0, stdout="", stderr="")

    with mock.patch.multiple(
        "runplz.backends.brev",
        _require_brev_cli=mock.DEFAULT,
        _skip_onboarding=mock.DEFAULT,
        _instance_exists=mock.Mock(return_value=True),
        _refresh_ssh=mock.DEFAULT,
        _wait_until_ssh_reachable=mock.DEFAULT,
        _prepare_remote_run=mock.DEFAULT,
        _ensure_remote_rsync=mock.DEFAULT,
        _rsync_up=mock.DEFAULT,
        _run_container_mode=mock.Mock(return_value=0),
        _rsync_down=mock.DEFAULT,
    ):
        with mock.patch("runplz.backends.brev.subprocess.run", fake_sub):
            brev.run(app, fn, [], {}, instance="box")

    delete_calls = [c for c in captured["calls"] if c[:2] == ["brev", "delete"]]
    assert delete_calls == [["brev", "delete", "box"]]


def test_apply_on_finish_warns_on_nonzero_exit(capsys):
    """A non-zero `brev stop/delete` is a billing-leak signal and must not
    be silent — that's the whole point of on_finish existing."""
    cfg = BrevConfig(mode="vm", use_docker=True)  # on_finish default = "stop"
    with mock.patch(
        "runplz.backends.brev.subprocess.run",
        return_value=mock.Mock(returncode=1, stdout="", stderr="brev api 503"),
    ):
        brev._apply_on_finish(instance="box", cfg=cfg)

    out = capsys.readouterr().out
    assert "exited 1" in out
    assert "check `brev ls`" in out
    assert "brev api 503" in out


def test_apply_on_finish_silent_on_success(capsys):
    cfg = BrevConfig(mode="vm", use_docker=True)

    def fake_run(cmd, *a, **kw):
        if cmd[:2] == ["brev", "ls"]:
            return mock.Mock(
                returncode=0,
                stdout=json.dumps([{"name": "box", "status": "STOPPED"}]),
                stderr="",
            )
        return mock.Mock(returncode=0, stdout="", stderr="")

    with mock.patch("runplz.backends.brev.subprocess.run", fake_run):
        brev._apply_on_finish(instance="box", cfg=cfg)

    # Only the "+ on_finish=stop: running ..." line; no warning.
    out = capsys.readouterr().out
    assert "warning" not in out.lower()


def test_apply_on_finish_warns_on_subprocess_exception(capsys):
    """If _brev_capture exhausts retries and raises on timeout, the
    caller's try/except still prints a warning (best-effort cleanup
    in a finally block must not propagate)."""
    cfg = BrevConfig(mode="vm", use_docker=True)

    def always_timeout(*a, **kw):
        raise RuntimeError("`brev stop box` timed out after 120s on all 4 attempts.")

    with mock.patch("runplz.backends.brev._brev_capture", side_effect=always_timeout):
        brev._apply_on_finish(instance="box", cfg=cfg)

    out = capsys.readouterr().out
    assert "RuntimeError" in out
    assert "check `brev ls`" in out


def test_verify_post_action_state_warns_when_delete_still_present(capsys):
    snapshot = {"name": "box", "status": "RUNNING"}
    with mock.patch("runplz.backends.brev._instance_snapshot", return_value=snapshot):
        brev._verify_post_action_state("delete", "box", timeout_s=0, poll_interval_s=0)

    out = capsys.readouterr().out
    assert "returned success but post-action state is still" in out
    assert "status=RUNNING" in out


# -- _run_native and _run_container_mode remote commands ------------------


def test_run_native_builds_expected_remote_cmd(tmp_path):
    app = _app(tmp_path)
    fn = _function(
        app,
        Image.from_registry("ubuntu:22.04"),
        module_file=_job_inside(tmp_path),
        env={"EXTRA": "ok"},
    )

    recorded = {}

    def fake_sh_ssh(instance, cmd, **kw):
        # First call is setup, second call (subprocess.run) is the actual run.
        recorded.setdefault("setup", cmd)

    def fake_sub_run(cmd, *a, **kw):
        recorded["run_cmd"] = cmd
        return mock.Mock(returncode=0)

    with mock.patch("runplz.backends._ssh_common._ssh", fake_sh_ssh):
        with mock.patch("runplz.backends._ssh_common.subprocess.run", fake_sub_run):
            rc = brev._run_native(
                target="box",
                function=fn,
                rel_script="jobs/train.py",
                args=[],
                kwargs={},
                has_nvidia=True,
            )
    assert rc == 0
    # Setup installs python3-venv & pip.
    assert "python3-venv" in recorded["setup"]
    # CUDA 12.1 wheel index when has_nvidia=True.
    assert "cu121" in recorded["setup"]
    # Run command has user env + bootstrap invocation.
    joined = " ".join(recorded["run_cmd"])
    assert "EXTRA=ok" in joined or "EXTRA" in recorded["run_cmd"][-1]
    assert "runplz.runners._bootstrap" in joined or "runplz._bootstrap" in joined


def test_run_native_cpu_index_url(tmp_path):
    app = _app(tmp_path)
    fn = _function(app, Image.from_registry("ubuntu:22.04"), module_file=_job_inside(tmp_path))

    recorded = {}

    def fake_ssh(i, c, **kw):
        recorded.setdefault("setup", c)

    with mock.patch("runplz.backends._ssh_common._ssh", fake_ssh):
        with mock.patch(
            "runplz.backends._ssh_common.subprocess.run",
            return_value=mock.Mock(returncode=0),
        ):
            brev._run_native(
                target="box",
                function=fn,
                rel_script="jobs/train.py",
                args=[],
                kwargs={},
                has_nvidia=False,
            )
    assert "cpu" in recorded["setup"]
    assert "cu121" not in recorded["setup"]


def test_run_container_mode_builds_expected_remote_cmd(tmp_path):
    app = _app(tmp_path, cfg=BrevConfig(mode="container"))
    fn = _function(
        app,
        Image.from_registry("pytorch/pytorch:2.4.0").pip_install("pandas"),
        module_file=_job_inside(tmp_path),
    )

    recorded = {}

    def fake_ssh(i, c, **kw):
        # First call = ops_script install; ignore.
        recorded["ops"] = c

    def fake_sub_run(cmd, *a, **kw):
        recorded["run"] = cmd
        return mock.Mock(returncode=0)

    with mock.patch("runplz.backends._ssh_common._ssh", fake_ssh):
        with mock.patch("runplz.backends._ssh_common.subprocess.run", fake_sub_run):
            rc = brev._run_container_mode(
                target="box",
                function=fn,
                rel_script="jobs/train.py",
                args=[],
                kwargs={},
            )
    assert rc == 0
    assert "pip install --quiet pandas" in recorded["ops"]
    assert "runplz._bootstrap" in recorded["run"][-1]
