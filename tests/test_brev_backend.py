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

    def fake_sh(cmd):
        recorded["create_cmd"] = cmd

    with mock.patch("runplz.backends.brev._pick_instance_type", return_value="picked-type"):
        with mock.patch("runplz.backends.brev._sh", fake_sh):
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
    with mock.patch("runplz.backends.brev._pick_instance_type", return_value="cheap-cpu-type"):
        with mock.patch("runplz.backends.brev._sh", lambda c: recorded.setdefault("c", c)):
            brev._create_instance("x", cfg=cfg, image=fn.image, function=fn)

    assert "cheap-cpu-type" in recorded["c"]


def test_create_instance_raises_when_picker_returns_none(tmp_path):
    cfg = BrevConfig(auto_create_instances=True)
    app = _app(tmp_path, cfg=cfg)
    fn = _function(
        app, Image.from_registry("ubuntu:22.04"), module_file=_job_inside(tmp_path), gpu="T4"
    )
    with mock.patch("runplz.backends.brev._pick_instance_type", return_value=None):
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
        with mock.patch("runplz.backends.brev._sh", lambda c: recorded.setdefault("c", c)):
            brev._create_instance("x", cfg=cfg, image=fn.image, function=fn)

    assert picker_called["n"] == 0
    assert "my-explicit-type" in recorded["c"]


def test_create_instance_container_mode_adds_image_flag(tmp_path):
    cfg = BrevConfig(mode="container")
    app = _app(tmp_path, cfg=cfg)
    image = Image.from_registry("pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime")
    fn = _function(app, image, module_file=_job_inside(tmp_path), gpu="T4")

    recorded = {}
    with mock.patch("runplz.backends.brev._pick_instance_type", return_value="picked-type"):
        with mock.patch("runplz.backends.brev._sh", lambda c: recorded.setdefault("c", c)):
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
        # Container-mode dispatch issues `tail -n N "$HOME/.runplz-last.log"`.
        if ".runplz-last.log" in cmd:
            return tail_output
        return ""

    with mock.patch.multiple(
        "runplz.backends.brev",
        _require_brev_cli=mock.DEFAULT,
        _skip_onboarding=mock.DEFAULT,
        _instance_exists=mock.Mock(return_value=True),
        _refresh_ssh=mock.DEFAULT,
        _wait_until_ssh_reachable=mock.DEFAULT,
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
    with mock.patch(
        "runplz.backends.brev.subprocess.run",
        return_value=mock.Mock(returncode=0, stdout="", stderr=""),
    ):
        brev._apply_on_finish(instance="box", cfg=cfg)

    # Only the "+ on_finish=stop: running ..." line; no warning.
    out = capsys.readouterr().out
    assert "warning" not in out.lower()


def test_apply_on_finish_warns_on_subprocess_exception(capsys):
    cfg = BrevConfig(mode="vm", use_docker=True)
    with mock.patch(
        "runplz.backends.brev.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="brev stop", timeout=120),
    ):
        brev._apply_on_finish(instance="box", cfg=cfg)

    out = capsys.readouterr().out
    assert "TimeoutExpired" in out
    assert "check `brev ls`" in out


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
