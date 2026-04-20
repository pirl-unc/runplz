"""Brev backend: provision (optional) → rsync repo → ssh docker build+run → rsync outputs.

Assumes `brev` CLI is installed and `brev login` has been run. Uses Brev's
managed SSH config (`brev refresh` populates ~/.brev/ssh_config, which
~/.ssh/config Includes).

Long training runs are resilient to SSH disconnects: the remote docker
container is started detached (`docker run -d`), logs are streamed via
`docker logs -f` in a reconnect loop, and we wait for the container to
exit via `docker wait` (which tolerates transient tunnel drops).
"""

import json
import shlex
import subprocess
import time
import uuid
from pathlib import Path
from typing import Optional

REMOTE_REPO_DIR = "runplz-repo"
REMOTE_OUT_DIR = "runplz-out"
REMOTE_IMAGE_TAG = "runplz-train:brev"

# Brev's ~/.brev/ssh_config sets `ControlMaster auto` (connection
# multiplexing). That's fast for short repeated calls but catastrophic
# for our workload: a long-lived `docker logs -f` ssh session holds the
# master, the underlying TCP goes stale (common on GCP N1/G2 GPU boxes),
# and every subsequent ssh call — including our `docker inspect` health
# probe — hangs for ~5 minutes waiting for the dead master to time out.
# Force a fresh TCP connection per ssh call to sidestep that entirely.
#
# ServerAliveInterval=30 + large ServerAliveCountMax keeps each
# individual session alive during idle stretches (docker image pulls,
# data downloads, between-epoch pauses).
_SSH_OPTS = [
    "-o",
    "ControlMaster=no",
    "-o",
    "ControlPath=none",
    "-o",
    "ServerAliveInterval=30",
    "-o",
    "ServerAliveCountMax=240",
    "-o",
    "TCPKeepAlive=yes",
]

# Brev's CLI hangs on an interactive walkthrough once an instance exists in the
# org, blocking `brev ls`/`brev refresh`. Overwriting this file with the
# completed state skips it. The file lives under the user's home — cheap and
# harmless.
_BREV_ONBOARDING = Path.home() / ".brev" / "onboarding_step.json"
_BREV_ONBOARDING_DONE = {
    "step": 999,
    "hasRunBrevShell": True,
    "hasRunBrevOpen": True,
}


def run(app, function, args, kwargs, *, instance: str, outputs_dir: str = "out"):
    _require_brev_cli()
    _skip_onboarding()

    cfg = app.brev_config
    from runplz.app import validate_image_vs_brev_mode

    validate_image_vs_brev_mode(fn_name=function.name, image=function.image, brev_config=cfg)
    if not _instance_exists(instance):
        if cfg.auto_create_instances:
            _create_instance(instance, cfg=cfg, image=function.image, function=function)
        else:
            raise RuntimeError(
                f"Brev instance {instance!r} not found and "
                f"BrevConfig(auto_create_instances=False). "
                f"Create it first (e.g. `brev create {instance} --type <TYPE>` for vm mode, "
                f"or `brev create {instance} --mode container --type <TYPE> "
                f"--container-image <IMAGE>` for container mode)."
            )
    _refresh_ssh()
    # `brev create` has its own internal wait-for-ready loop, but on some
    # providers (8-GPU boxes, slow pull of large container images) that
    # loop times out before SSH is actually reachable. Poll explicitly.
    _wait_until_ssh_reachable(instance)

    repo = app._repo_root
    host_out = (repo / outputs_dir).resolve()
    host_out.mkdir(parents=True, exist_ok=True)

    if cfg.mode == "container":
        # Pre-built container images (e.g. pytorch/pytorch) don't ship with
        # rsync. Install it before the first rsync call, otherwise we get
        # "rsync: command not found" on the remote end.
        _ensure_remote_rsync(instance)
    _rsync_up(repo, instance)
    rel_script = Path(function.module_file).resolve().relative_to(repo)

    container_name: Optional[str] = None
    exit_code: Optional[int] = None
    try:
        if cfg.mode == "container":
            # Brev `--mode container` box IS the user's container image. Apply
            # our Image DSL ops inline (apt_install, pip_install,
            # pip_install_local_dir, run_commands) via ssh, then invoke the
            # bootstrap. No docker-in-docker, no nvidia-container-toolkit,
            # no Brev VM sidecar stack.
            exit_code = _run_container_mode(
                instance=instance,
                function=function,
                rel_script=str(rel_script),
                args=args,
                kwargs=kwargs,
            )
        elif cfg.use_docker:
            _ensure_docker(instance)
            gpu_flag = "--gpus all" if _remote_has_nvidia(instance) else ""
            container_name = f"runplz-{function.name}-{uuid.uuid4().hex[:8]}"
            _build_image(instance, function.image)
            _run_container_detached(
                instance=instance,
                container_name=container_name,
                function=function,
                rel_script=str(rel_script),
                args=args,
                kwargs=kwargs,
                gpu_flag=gpu_flag,
            )
            exit_code = _stream_and_wait(instance, container_name)
        else:
            # Legacy native path. Skips docker; installs python + torch +
            # user code into a venv on a plain VM-mode Brev box. Use this only
            # if you can't use mode="container" (e.g. specific provider flow).
            exit_code = _run_native(
                instance=instance,
                function=function,
                rel_script=str(rel_script),
                args=args,
                kwargs=kwargs,
                has_nvidia=_remote_has_nvidia(instance),
            )
        _rsync_down(instance, host_out)
    finally:
        # Container cleanup first (while the box is definitely up), then
        # the box itself per cfg.on_finish. Both are best-effort so a
        # failure in cleanup doesn't mask the real error from the try block.
        if container_name is not None:
            try:
                _ssh_capture(
                    instance,
                    f"sudo docker rm -f {container_name} >/dev/null 2>&1 || true",
                )
            except Exception as exc:  # noqa: BLE001
                print(f"+ warning: failed to remove container {container_name}: {exc}", flush=True)
        _apply_on_finish(instance=instance, cfg=cfg)
    if exit_code != 0:
        raise RuntimeError(f"Remote run exited with status {exit_code}")


def _apply_on_finish(*, instance: str, cfg) -> None:
    """Stop / delete / leave the Brev box per `cfg.on_finish`.

    Always best-effort: we never want box-cleanup to swallow a real error
    from the try block. Failures here print a warning and move on.
    """
    if cfg.on_finish == "leave":
        return
    action = cfg.on_finish  # "stop" or "delete"
    print(f"+ on_finish={action}: running `brev {action} {instance}`", flush=True)
    try:
        subprocess.run(
            ["brev", action, instance],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except Exception as exc:  # noqa: BLE001
        print(
            f"+ warning: `brev {action} {instance}` raised {type(exc).__name__}: {exc}. "
            f"The box may still be running — check `brev ls`.",
            flush=True,
        )


_NATIVE_VENV = "$HOME/runplz-venv"
_NATIVE_OUT = f"$HOME/{REMOTE_OUT_DIR}"


def _ensure_remote_rsync(instance: str):
    """Install rsync on the remote if missing (pytorch/pytorch image and
    similar are slim and don't ship with rsync)."""
    cmd = (
        "command -v rsync >/dev/null 2>&1 && exit 0; "
        "export DEBIAN_FRONTEND=noninteractive; "
        "sudo apt-get update -qq && "
        "sudo apt-get install -y -qq --no-install-recommends rsync"
    )
    _ssh(instance, cmd)


def _run_container_mode(*, instance, function, rel_script, args, kwargs):
    """Brev `--mode container`: the SSH box IS the user's container image.

    Runs the user's Image DSL ops (apt_install, pip_install, ...) inline
    over ssh, then invokes the bootstrap. Because the box is already the
    user-selected pytorch/cuda image, there's no docker-in-docker, no
    `--gpus all` nvidia-container-toolkit path, no Brev VM-mode sidecars.
    """
    ops_script = _render_ops_script(function.image)
    if ops_script:
        _ssh(instance, ops_script)

    # Values that contain $HOME must use double-quoted exports so the
    # remote shell expands them. shlex.quote would wrap in single quotes,
    # preventing expansion. Non-path values (JSON, name) go through
    # shlex.quote since they may contain shell metacharacters.
    user_env_exports = " ".join(
        f"export {k}={shlex.quote(str(v))};" for k, v in function.env.items()
    )
    remote = (
        "set -euo pipefail; "
        "export PATH=/opt/conda/bin:$PATH; "
        f'export RUNPLZ_OUT="$HOME/{REMOTE_OUT_DIR}"; '
        f'export RUNPLZ_SCRIPT="$HOME/{REMOTE_REPO_DIR}/{rel_script}"; '
        f"export RUNPLZ_FUNCTION={shlex.quote(function.name)}; "
        f"export RUNPLZ_ARGS={shlex.quote(json.dumps(args))}; "
        f"export RUNPLZ_KWARGS={shlex.quote(json.dumps(kwargs))}; "
        f"{user_env_exports} "
        'mkdir -p "$RUNPLZ_OUT"; '
        # cd to the rsync'd repo so user-level `subprocess.run(["bash",
        # "scripts/..."])` calls with relative paths find their scripts.
        f'cd "$HOME/{REMOTE_REPO_DIR}"; '
        "python -m runplz._bootstrap"
    )
    r = subprocess.run(["ssh", *_SSH_OPTS, instance, f"bash -lc {shlex.quote(remote)}"])
    return r.returncode


def _render_ops_script(image) -> str:
    """Translate Image DSL ops into a bash script that runs them in
    sequence on the remote container-mode box. Idempotent: apt/pip on
    already-present packages is a cheap no-op.

    The image is guaranteed to be `Image.from_registry(...)` — the
    decoration-time validator in runplz.app rejects Dockerfile images
    when `brev_config.mode == "container"`.
    """
    lines = ["set -euo pipefail"]
    # Start on an apt-free note: wait for any apt from box bootstrap.
    lines.append(
        "for i in $(seq 1 60); do "
        "  sudo fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1 "
        "    && { echo waiting for apt; sleep 10; } "
        "    || break; "
        "done"
    )
    lines.append("export DEBIAN_FRONTEND=noninteractive")
    lines.append("export PATH=/opt/conda/bin:$PATH")

    apt_packages_seen: list[str] = []
    pip_packages_seen: list[str] = []
    for op in image.ops:
        kw = op.kwargs_dict()
        if op.kind == "apt_install" and op.args:
            apt_packages_seen.extend(op.args)
            pkgs = " ".join(shlex.quote(p) for p in op.args)
            lines.append(
                f"sudo apt-get update -qq && sudo apt-get install -y -qq "
                f"--no-install-recommends {pkgs}"
            )
        elif op.kind == "pip_install" and op.args:
            pip_packages_seen.extend(op.args)
            pkgs = " ".join(shlex.quote(p) for p in op.args)
            idx = ""
            if "index_url" in kw:
                idx = f" --index-url {shlex.quote(kw['index_url'])}"
            lines.append(f"pip install --quiet{idx} {pkgs}")
        elif op.kind == "pip_install_local_dir":
            path = kw.get("path", ".")
            editable = kw.get("editable", "1") == "1"
            flags = "-e " if editable else ""
            # Use the rsync'd repo (we rsync it before running). Keep $HOME
            # unquoted so the remote shell expands it; only quote `path`.
            rel = path.lstrip("./")
            sub = f"/{rel}" if rel else ""
            lines.append(f'pip install --quiet {flags}"$HOME/{REMOTE_REPO_DIR}{sub}"')
        elif op.kind == "run" and op.args:
            for cmd in op.args:
                lines.append(cmd)
    return "; ".join(lines)


def _run_native(*, instance, function, rel_script, args, kwargs, has_nvidia):
    """Install user code natively and run the job over ssh.

    Two ssh sessions: (1) idempotent setup — wait for Brev's own apt, then
    apt-get + venv + pip install; (2) actually run the user's function
    via the bootstrap, with env vars set for this specific invocation.
    """
    torch_index = (
        "https://download.pytorch.org/whl/cu121"
        if has_nvidia
        else "https://download.pytorch.org/whl/cpu"
    )
    setup = (
        "set -euo pipefail; "
        # Wait out Brev's own apt activity (installs docker + nvidia
        # runtime at first boot even when we don't want them).
        "for i in $(seq 1 120); do "
        "  sudo fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1 "
        "    && { echo waiting for apt; sleep 10; } "
        "    || break; "
        "done; "
        "sudo apt-get update -qq; "
        "sudo apt-get install -y -qq --no-install-recommends "
        "  python3 python3-venv python3-pip bzip2 wget rsync build-essential; "
        f"python3 -m venv {_NATIVE_VENV}; "
        f"source {_NATIVE_VENV}/bin/activate; "
        "pip install --quiet --upgrade pip; "
        f"pip install --quiet torch --index-url {torch_index}; "
        f"pip install --quiet -e $HOME/{REMOTE_REPO_DIR}"
    )
    _ssh(instance, setup)

    user_env_exports = " ".join(
        f"export {k}={shlex.quote(str(v))};" for k, v in function.env.items()
    )
    remote = (
        "set -euo pipefail; "
        f'source "$HOME/runplz-venv/bin/activate"; '
        f'export RUNPLZ_OUT="$HOME/{REMOTE_OUT_DIR}"; '
        f'export RUNPLZ_SCRIPT="$HOME/{REMOTE_REPO_DIR}/{rel_script}"; '
        f"export RUNPLZ_FUNCTION={shlex.quote(function.name)}; "
        f"export RUNPLZ_ARGS={shlex.quote(json.dumps(args))}; "
        f"export RUNPLZ_KWARGS={shlex.quote(json.dumps(kwargs))}; "
        f"{user_env_exports} "
        'mkdir -p "$RUNPLZ_OUT"; '
        f'cd "$HOME/{REMOTE_REPO_DIR}"; '
        "python -m runplz._bootstrap"
    )
    r = subprocess.run(["ssh", *_SSH_OPTS, instance, f"bash -lc {shlex.quote(remote)}"])
    return r.returncode


def _build_image(instance: str, image):
    """Build the image on the remote, either from a user Dockerfile or
    from our Image DSL (by synthesizing a Dockerfile on the fly)."""
    if image.dockerfile is not None:
        build = (
            f"set -euo pipefail; "
            f"cd ~/{REMOTE_REPO_DIR} && "
            f"sudo docker build -f {shlex.quote(image.dockerfile)} "
            f"-t {REMOTE_IMAGE_TAG} ."
        )
    else:
        df = image.render_dockerfile()
        # Pipe the synthesized Dockerfile to docker build via stdin,
        # using the repo as context so pip_install_local_dir can COPY it.
        build = (
            f"set -euo pipefail; "
            f"cd ~/{REMOTE_REPO_DIR} && "
            f"cat <<'__EOF__' | sudo docker build -f - -t {REMOTE_IMAGE_TAG} .\n"
            f"{df}\n"
            f"__EOF__"
        )
    _ssh(instance, build)


def _run_container_detached(
    *, instance, container_name, function, rel_script, args, kwargs, gpu_flag
):
    env_flags = " ".join(f"-e {shlex.quote(f'{k}={v}')}" for k, v in function.env.items())
    runner_env = (
        f"-e RUNPLZ_OUT=/out "
        f"-e RUNPLZ_SCRIPT={shlex.quote('/workspace/' + rel_script)} "
        f"-e RUNPLZ_FUNCTION={shlex.quote(function.name)} "
        f"-e RUNPLZ_ARGS={shlex.quote(json.dumps(args))} "
        f"-e RUNPLZ_KWARGS={shlex.quote(json.dumps(kwargs))}"
    )
    # --network=host bypasses docker's bridge + iptables NAT machinery for the
    # container. We tried this specifically to see if conntrack saturation
    # or nvidia-container-toolkit netfilter rules were behind Brev GPU's
    # SSH-goes-dark-after-a-few-minutes bug; it wasn't (probing showed the
    # port-22 SYN/ACK still completes but the banner bytes never arrive,
    # even with host networking). Keeping the flag anyway: simpler
    # networking, no NAT overhead, and it rules out a whole class of
    # failure modes if we ever see this again.
    start = (
        f"set -euo pipefail; "
        f"mkdir -p ~/{REMOTE_OUT_DIR} && "
        f"sudo docker run -d --name {container_name} --network=host {gpu_flag} "
        f"-v $HOME/{REMOTE_OUT_DIR}:/out "
        f"{runner_env} {env_flags} "
        f"{REMOTE_IMAGE_TAG} python -m runplz._bootstrap"
    )
    _ssh(instance, start)


def _stream_and_wait(instance: str, container_name: str, max_reconnects: int = 20) -> int:
    """Stream container logs and return its exit code.

    `docker logs -f` exits when the container stops, so we loop across SSH
    reconnects: if ssh drops mid-stream we re-attach with `--tail 0` to
    pick up where we left off, then call `docker wait` for the exit code.
    Gives up after `max_reconnects` consecutive reconnect attempts so we
    don't loop forever if the box is permanently unreachable.
    """
    print(f"+ streaming logs from {container_name} (resilient to reconnects)", flush=True)
    tail = "all"
    reconnects = 0
    while True:
        cmd = f"sudo docker logs -f --tail {tail} {container_name}"
        r = subprocess.run(
            ["ssh", *_SSH_OPTS, instance, cmd],
        )
        # Container may have exited cleanly (rc=0) or ssh may have dropped.
        # Either way, check whether the container is still running.
        running = _container_running(instance, container_name)
        if not running:
            break
        reconnects += 1
        if reconnects > max_reconnects:
            print(
                f"+ ssh reconnected {max_reconnects} times without finishing; "
                f"giving up on log stream. Container {container_name} may "
                f"still be running — check with `brev exec {instance} "
                f"'sudo docker logs {container_name}'`.",
                flush=True,
            )
            break
        print(
            f"+ ssh disconnected (rc={r.returncode}); container still "
            f"running, reconnecting log stream "
            f"({reconnects}/{max_reconnects})",
            flush=True,
        )
        tail = "0"
        time.sleep(2)
    # Container stopped. Get its exit code (docker wait returns immediately
    # for stopped containers and prints the exit code).
    r = subprocess.run(
        ["ssh", *_SSH_OPTS, instance, f"sudo docker wait {container_name}"],
        capture_output=True,
        text=True,
    )
    try:
        return int(r.stdout.strip() or "1")
    except ValueError:
        return 1


def _container_running(instance: str, container_name: str) -> bool:
    # Treat ssh hangs / errors as "assume still running" so the caller keeps
    # retrying the log stream instead of giving up. If the box really did die
    # mid-training, docker wait at the end will return its final exit code.
    try:
        r = subprocess.run(
            [
                "ssh",
                *_SSH_OPTS,
                instance,
                f"sudo docker inspect --format '{{{{.State.Running}}}}' {container_name}",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return True
    if r.returncode != 0:
        return True
    return r.stdout.strip() == "true"


def _ssh_capture(instance: str, remote_cmd: str) -> str:
    r = subprocess.run(
        ["ssh", *_SSH_OPTS, instance, remote_cmd],
        capture_output=True,
        text=True,
        timeout=60,
    )
    return r.stdout


def _require_brev_cli():
    if subprocess.run(["which", "brev"], capture_output=True).returncode != 0:
        raise RuntimeError(
            "`brev` CLI not found. Install via `brew install brev` (macOS) "
            "or the script at https://developer.nvidia.com/brev, then run "
            "`brev login`."
        )


def _skip_onboarding():
    try:
        _BREV_ONBOARDING.parent.mkdir(parents=True, exist_ok=True)
        existed = _BREV_ONBOARDING.exists()
        _BREV_ONBOARDING.write_text(json.dumps(_BREV_ONBOARDING_DONE))
        if not existed:
            print(
                f"+ wrote {_BREV_ONBOARDING} to skip Brev CLI walkthrough "
                f"(prevents `brev ls` from hanging once instances exist)",
                flush=True,
            )
    except OSError:
        pass


def _instance_exists(name: str) -> bool:
    """True iff `brev ls` lists an instance with this name.

    Raises RuntimeError if the `brev` CLI itself failed (bad auth, network,
    Brev API outage, malformed JSON) — we refuse to silently treat a degraded
    listing as "instance doesn't exist," because the caller's fallback is to
    auto-create a *new billed box*, which may duplicate one the user already
    has.

    Returns False only when `brev ls` succeeded but the target name is
    definitively not in the list (including the documented `null` / empty
    shapes Brev returns for empty orgs).
    """
    r = subprocess.run(["brev", "ls", "--json"], capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        raise RuntimeError(
            f"`brev ls --json` failed with exit code {r.returncode}. "
            f"Refusing to continue — if we assumed the instance was missing we'd "
            f"auto-create a duplicate billed box. Run `brev login` / check the "
            f"`brev` CLI and retry. stderr: {(r.stderr or '').strip()[:500]}"
        )
    try:
        data = json.loads(r.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"`brev ls --json` returned unparseable JSON. Refusing to continue "
            f"(see _instance_exists docstring for why). stdout head: "
            f"{(r.stdout or '').strip()[:200]!r}"
        ) from exc
    # `brev ls --json` shape varies: list of instances, dict with an
    # "instances" key, or null when the org has no instances at all.
    if data is None:
        return False
    if isinstance(data, list):
        instances = data
    elif isinstance(data, dict):
        instances = data.get("instances", []) or []
    else:
        # Unknown shape — treat as CLI failure rather than "empty."
        raise RuntimeError(
            f"`brev ls --json` returned an unexpected shape ({type(data).__name__}). "
            f"Refusing to continue (see _instance_exists docstring)."
        )
    return any(i.get("name") == name for i in instances)


def _create_instance(name: str, *, cfg=None, image=None, function=None):
    """Provision a Brev instance.

    Picks the instance type in this order:
    1. `cfg.instance_type` — explicit user override (if set).
    2. Cheapest match from `brev search` driven by `function`'s resource
       constraints. If no GPU is requested, this falls through to the
       cheapest CPU box; if a GPU is requested, the cheapest matching
       GPU box. With no constraints at all, we get the cheapest CPU box
       Brev offers — a sensible default.

    Raises only if the picker finds no matches (e.g. constraints too tight).
    """
    instance_type: Optional[str] = cfg.instance_type if cfg is not None else None
    if instance_type is None:
        instance_type = _pick_instance_type(function)
        if instance_type is None:
            raise RuntimeError(
                "`brev search` returned no matching instances. Loosen the "
                "function's resource constraints, pass an explicit "
                "`instance_type=...` on BrevConfig, or pre-create the instance."
            )

    cmd = ["brev", "create", name, "--type", instance_type]
    # Propagate --min-disk from function if set (also controls
    # actual provisioned disk size on Brev, not just filter).
    if function is not None and function.min_disk is not None:
        cmd += ["--min-disk", str(function.min_disk)]
    if cfg is not None and cfg.mode == "container":
        # `image.base` is guaranteed to be set here — Dockerfile images are
        # rejected at function-decoration time by runplz.app's validator.
        cmd += ["--mode", "container", "--container-image", image.base]
    _sh(cmd)


def _brev_gpu_name(modal_name: str) -> str:
    """Translate Modal-style GPU labels to Brev `--gpu-name` filter strings.

    Modal accepts things like "A100-40GB"; brev search wants a base name
    like "A100" and separately filters by VRAM. Stripping the suffix is
    good enough for matching.
    """
    n = modal_name.upper()
    # Strip any trailing "-40GB" / "-80GB" etc.
    for suffix in ("-40GB", "-80GB", "-16GB", "-24GB"):
        if n.endswith(suffix):
            return n[: -len(suffix)]
    return n


def _pick_instance_type(function) -> Optional[str]:
    """Run `brev search gpu` (or cpu) with filters from `function`'s
    resource requests; return the cheapest matching TYPE string.
    Returns None if no match."""
    mode = "gpu" if function.gpu is not None else "cpu"
    cmd = ["brev", "search", mode, "--json", "--sort", "price"]
    if function.gpu:
        cmd += ["--gpu-name", _brev_gpu_name(function.gpu)]
    if function.min_gpu_memory is not None:
        cmd += ["--min-vram", str(function.min_gpu_memory)]
    if function.min_cpu is not None:
        cmd += ["--min-vcpu", str(int(function.min_cpu))]
    if function.min_memory is not None:
        cmd += ["--min-ram", str(function.min_memory)]
    if function.min_disk is not None:
        cmd += ["--min-disk", str(function.min_disk)]
    print("+ " + " ".join(shlex.quote(c) for c in cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        return None
    try:
        results = json.loads(r.stdout)
    except json.JSONDecodeError:
        return None
    # `brev search --json` returns a list of matches sorted by $/hr.
    if not isinstance(results, list) or not results:
        return None
    first = results[0]
    return first.get("type") or first.get("Type") or first.get("name")


def _refresh_ssh():
    _sh(["brev", "refresh"])


def _wait_until_ssh_reachable(
    instance: str, *, max_wait_s: int = 1200, probe_interval_s: int = 15
) -> None:
    """Block until an SSH session to `instance` succeeds, or raise.

    `brev create` has its own internal readiness check, but on slow-boot
    providers (8-GPU boxes, large container image pulls) it frequently
    times out before the SSH daemon inside the instance is listening.
    This polls with short-timeout SSH probes and refreshes Brev's SSH
    config mid-loop in case the port changes (which it does when Brev
    promotes an instance from the bootstrap-shim port to the real one).
    """
    import time

    print(
        f"+ waiting for {instance} SSH to become reachable (up to {max_wait_s}s)...",
        flush=True,
    )
    deadline = time.time() + max_wait_s
    last_err = ""
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        probe = subprocess.run(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                f"ConnectTimeout={probe_interval_s}",
                *_SSH_OPTS,
                instance,
                "true",
            ],
            capture_output=True,
            text=True,
        )
        if probe.returncode == 0:
            print(f"+ {instance} SSH ready (attempt {attempt})", flush=True)
            return
        last_err = (probe.stderr or probe.stdout or "").strip().splitlines()[-1:] or [""]
        last_err = last_err[0]
        # Every ~minute, re-run `brev refresh` in case the SSH config
        # needs updating (new port, new hostname).
        if attempt % 4 == 0:
            print(
                f"+ {instance} still unreachable after {attempt} probes "
                f"(last: {last_err}); refreshing brev config...",
                flush=True,
            )
            try:
                subprocess.run(["brev", "refresh"], check=False, capture_output=True, timeout=60)
            except subprocess.TimeoutExpired:
                pass
        time.sleep(probe_interval_s)
    raise RuntimeError(
        f"SSH to {instance} never became reachable within {max_wait_s}s "
        f"(last error: {last_err!r}). The instance may still be booting; "
        f"try again, or check `brev ls` for its status."
    )


def _ensure_docker(instance: str, timeout_s: int = 420):
    # Brev's own bootstrap installs docker shortly after the instance comes up.
    # On GPU boxes the binary appears before dockerd is listening, so poll
    # `docker info` (daemon reachable) rather than just presence of the binary.
    print(f"+ waiting for docker daemon on {instance} (up to {timeout_s}s)", flush=True)
    wait_script = (
        "for i in $(seq 1 60); do "
        "if command -v docker >/dev/null 2>&1 && "
        "   sudo docker info >/dev/null 2>&1; then exit 0; fi; "
        "sudo fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1 "
        "  && echo 'apt busy, waiting' || echo 'waiting for docker daemon'; "
        "sleep 7; "
        "done; exit 1"
    )
    r = subprocess.run(
        [
            "ssh",
            *_SSH_OPTS,
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=accept-new",
            instance,
            wait_script,
        ],
        timeout=timeout_s,
    )
    if r.returncode != 0:
        print(
            f"+ docker daemon not reachable on {instance} after {timeout_s}s "
            f"(Brev's bootstrap may be stuck — check `brev ls` / "
            f"`brev exec {instance} 'systemctl status docker'`); "
            f"falling back to get-docker.sh",
            flush=True,
        )
        _sh(["ssh", instance, "curl -fsSL https://get.docker.com | sudo sh"])


def _remote_has_nvidia(instance: str) -> bool:
    # nvidia-smi is often pre-installed on Brev boxes even without a GPU;
    # the reliable signal is /proc/driver/nvidia, which only exists when the
    # kernel module is loaded against real hardware.
    r = subprocess.run(
        ["ssh", *_SSH_OPTS, instance, "test -d /proc/driver/nvidia && echo y || echo n"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return r.returncode == 0 and r.stdout.strip() == "y"


def _rsync_up(repo: Path, instance: str):
    # Intentionally no --delete: a user who sshes in and leaves files under
    # ~/runplz-repo/ (logs, probe scripts, local edits) shouldn't have those
    # wiped by the next run. Stale files on the remote are cheap; accidental
    # user-data loss is not.
    _sh(
        [
            "rsync",
            "-az",
            "--exclude=.git",
            "--exclude=.venv",
            "--exclude=__pycache__",
            "--exclude=*.egg-info",
            "--exclude=build",
            "--exclude=dist",
            "--exclude=out",
            f"{repo}/",
            f"{instance}:{REMOTE_REPO_DIR}/",
        ]
    )


def _rsync_down(instance: str, local_out: Path):
    _sh(["rsync", "-az", f"{instance}:{REMOTE_OUT_DIR}/", f"{local_out}/"])


def _ssh(instance: str, remote_cmd: str):
    # Pass the whole pipeline as a SINGLE arg to ssh. If we pass
    # ["ssh", host, "bash", "-lc", cmd] instead, ssh space-joins the trailing
    # argv before sending to the remote shell, which then re-parses — turning
    # `bash -lc 'set -euo pipefail; X'` into `bash -lc set -euo pipefail; X`
    # (i.e. `set` runs with no args as the -c command, X runs in the outer
    # shell without errexit). Quoting with shlex.quote around the whole
    # command string avoids that.
    _sh(["ssh", *_SSH_OPTS, instance, f"bash -lc {shlex.quote(remote_cmd)}"])


def _sh(cmd):
    print("+ " + " ".join(shlex.quote(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True)
