"""SSH backend: dispatch to a user-owned remote machine.

Zero provisioning, zero lifecycle — the user manages the box. runplz
just rsyncs the repo up, optionally warns about spec mismatches,
dispatches the bootstrap (docker or native), and rsyncs outputs back.

Target resolution: the `host` string passed to `App.bind("ssh", host=...)`
or `runplz ssh --host <name>` is whatever ssh/rsync treat as a reachable
endpoint — a bare hostname, a `user@host[:port]` URL, or an alias from
your ~/.ssh/config. SshConfig.user / .port, when set, override the URL
via ssh's -l / -p flags.

This backend shares all the SSH plumbing with the brev backend via
`runplz.backends._ssh_common`.
"""

import re
from pathlib import Path
from typing import Optional

from runplz.backends._ssh_common import (
    FAILURE_TAIL_LINES,
    _build_image,
    _ensure_docker,
    _ensure_remote_rsync,
    _fetch_failure_tail,
    _remote_has_nvidia,
    _rsync_down,
    _rsync_up,
    _run_container_detached,
    _run_native,
    _ssh_capture,
    _stream_and_wait,
    _wait_until_ssh_reachable,
    make_container_name,
)

__all__ = ["run"]


def run(app, function, args, kwargs, *, host: str, outputs_dir: str = "out"):
    cfg = app.ssh_config
    target = _build_ssh_target(host, user=cfg.user, port=cfg.port)

    _wait_until_ssh_reachable(target)
    _warn_on_spec_mismatch(target, function)

    repo = app._repo_root
    host_out = (repo / outputs_dir).resolve()
    host_out.mkdir(parents=True, exist_ok=True)

    # Make sure rsync is present before we try to upload.
    _ensure_remote_rsync(target)
    _rsync_up(repo, target)

    rel_script = Path(function.module_file).resolve().relative_to(repo)

    container_name: Optional[str] = None
    exit_code: Optional[int] = None
    try:
        if cfg.use_docker:
            _ensure_docker(target)
            gpu_flag = "--gpus all" if _remote_has_nvidia(target) else ""
            container_name = make_container_name(function.name)
            _build_image(target, function.image)
            _run_container_detached(
                target=target,
                container_name=container_name,
                function=function,
                rel_script=str(rel_script),
                args=args,
                kwargs=kwargs,
                gpu_flag=gpu_flag,
            )
            exit_code = _stream_and_wait(
                target, container_name, max_runtime_seconds=cfg.max_runtime_seconds
            )
        else:
            # Native: can't use Image.from_registry's container env — fall back
            # to installing python+torch in a venv on the remote. Mirrors
            # brev's use_docker=False path.
            exit_code = _run_native(
                target=target,
                function=function,
                rel_script=str(rel_script),
                args=args,
                kwargs=kwargs,
                has_nvidia=_remote_has_nvidia(target),
                max_runtime_seconds=cfg.max_runtime_seconds,
            )
        _rsync_down(target, host_out)
    finally:
        failure_tail = ""
        if exit_code is not None and exit_code != 0:
            failure_tail = _fetch_failure_tail(target=target, container_name=container_name)
        if container_name is not None:
            try:
                _ssh_capture(
                    target,
                    f"sudo docker rm -f {container_name} >/dev/null 2>&1 || true",
                )
            except Exception as exc:  # noqa: BLE001
                print(f"+ warning: failed to remove container {container_name}: {exc}", flush=True)
        # No on_finish for SSH backend: the user owns the box lifecycle.

    if exit_code != 0:
        msg = f"Remote run exited with status {exit_code}"
        if failure_tail:
            msg += (
                f"\n--- last {FAILURE_TAIL_LINES} lines of remote output ---\n"
                f"{failure_tail}\n"
                f"--- end remote output ---"
            )
        raise RuntimeError(msg)


def _build_ssh_target(host: str, *, user: Optional[str], port: Optional[int]) -> str:
    """Build the ssh/rsync endpoint string from host + optional user/port.

    If the host string already includes a user ("alex@...") or port (":22"),
    SshConfig.user / .port still win — we rewrite rather than concatenate,
    so conflicting values fail loudly instead of silently.
    """
    bare = host
    if "@" in bare:
        existing_user, bare = bare.split("@", 1)
        if user is None:
            user = existing_user
    if ":" in bare and "]" not in bare:
        # IPv6 bracketed URLs have colons; bail out on those, let ssh parse.
        bare, existing_port = bare.rsplit(":", 1)
        if port is None:
            try:
                port = int(existing_port)
            except ValueError:
                bare = f"{bare}:{existing_port}"  # wasn't a port after all
    target = f"{user}@{bare}" if user else bare
    if port:
        # rsync needs `-e "ssh -p <port>"` but _ssh_common helpers don't
        # thread that kwarg through. Instead, advise ~/.ssh/config or put
        # the port in the alias. As a fallback, emit a warning: pinning
        # ports inline is not yet supported.
        # TODO: plumb port through SSH_OPTS everywhere.
        print(
            f"+ warning: SshConfig.port={port!r} not yet wired into rsync/ssh "
            f"invocations. Set the port in your ~/.ssh/config instead, or "
            f"bake it into the host alias. Proceeding without -p.",
            flush=True,
        )
    return target


# --- Spec-mismatch warnings ----------------------------------------------


_MEMINFO_LINE = re.compile(r"^MemTotal:\s+(\d+)\s+kB", re.MULTILINE)
_NVIDIA_LINE = re.compile(r"^([^,]+),\s*(\d+)\s*MiB$", re.MULTILINE)


def _warn_on_spec_mismatch(target: str, function) -> None:
    """Probe the remote box and warn when its specs don't meet the function's
    constraints. Best-effort — never raises. The user may know something we
    don't (overcommitting a dev box, MIG-partitioned GPUs, etc.).

    Probes nproc, /proc/meminfo, nvidia-smi in a single ssh call so we don't
    add latency for every dimension.
    """
    try:
        probe = _ssh_capture(
            target,
            "echo '---NPROC---'; nproc; "
            "echo '---MEMINFO---'; cat /proc/meminfo 2>/dev/null | head -1; "
            "echo '---NVIDIA---'; "
            "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null; "
            "echo '---END---'",
        )
    except Exception as exc:  # noqa: BLE001
        print(
            f"+ warning: could not probe remote specs on {target}: {type(exc).__name__}: {exc}",
            flush=True,
        )
        return

    sections = _parse_probe_sections(probe)
    warnings = []
    warnings.extend(_check_cpu(sections.get("NPROC", ""), function))
    warnings.extend(_check_memory(sections.get("MEMINFO", ""), function))
    warnings.extend(_check_gpu(sections.get("NVIDIA", ""), function))
    for w in warnings:
        print(f"+ spec-mismatch warning: {w}", flush=True)


def _parse_probe_sections(probe: str) -> dict:
    sections: dict[str, str] = {}
    current = None
    buf: list[str] = []
    for line in (probe or "").splitlines():
        line = line.rstrip()
        if line.startswith("---") and line.endswith("---"):
            if current:
                sections[current] = "\n".join(buf).strip()
                buf = []
            current = line.strip("-").strip()
        else:
            buf.append(line)
    if current and current != "END":
        sections[current] = "\n".join(buf).strip()
    return sections


def _check_cpu(nproc_out: str, function) -> list[str]:
    if function.min_cpu is None:
        return []
    try:
        remote_cpus = int(nproc_out.strip().splitlines()[0])
    except (ValueError, IndexError):
        return []
    if remote_cpus < function.min_cpu:
        return [
            f"Function declares min_cpu={function.min_cpu!r} but the remote "
            f"reports {remote_cpus} vCPUs."
        ]
    return []


def _check_memory(meminfo_out: str, function) -> list[str]:
    if function.min_memory is None:
        return []
    m = _MEMINFO_LINE.search(meminfo_out)
    if not m:
        return []
    remote_gb = int(m.group(1)) / (1024 * 1024)
    if remote_gb < function.min_memory:
        return [
            f"Function declares min_memory={function.min_memory!r} GB but "
            f"the remote reports {remote_gb:.1f} GB of RAM."
        ]
    return []


def _check_gpu(nvidia_out: str, function) -> list[str]:
    if function.gpu is None and function.min_gpu_memory is None:
        return []
    gpus: list[tuple[str, int]] = []
    for m in _NVIDIA_LINE.finditer(nvidia_out or ""):
        name, mib = m.group(1).strip(), int(m.group(2))
        gpus.append((name, mib))
    warnings = []
    if function.gpu is not None and not gpus:
        warnings.append(
            f"Function declares gpu={function.gpu!r} but `nvidia-smi` on "
            f"the remote returned no GPUs (is this the right box?)."
        )
    # Count check (num_gpus > 1).
    num_gpus = getattr(function, "num_gpus", 1) or 1
    if num_gpus > 1 and len(gpus) < num_gpus:
        warnings.append(
            f"Function declares num_gpus={num_gpus} but the remote has {len(gpus)} GPU(s)."
        )
    if function.min_gpu_memory is not None and gpus:
        need_mib = int(function.min_gpu_memory * 1024)
        best_mib = max(mib for _, mib in gpus)
        if best_mib < need_mib:
            best_gb = best_mib / 1024
            warnings.append(
                f"Function declares min_gpu_memory={function.min_gpu_memory!r} "
                f"GB but the largest remote GPU has {best_gb:.1f} GB VRAM."
            )
    if function.gpu is not None and gpus:
        want = function.gpu.upper().split("-", 1)[0]  # strip "-40GB"
        names = [n.upper() for n, _ in gpus]
        if not any(want in n for n in names):
            warnings.append(
                f"Function declares gpu={function.gpu!r} but remote GPUs "
                f"are: {', '.join(n for n, _ in gpus)}."
            )
    return warnings
