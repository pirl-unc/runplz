"""SSH backend: dispatch to a user-owned remote machine.

Zero provisioning, zero lifecycle — the user manages the box. runplz
just rsyncs the repo up, optionally warns about spec mismatches,
dispatches the bootstrap (docker or native), and rsyncs outputs back.

Target resolution: the `host` string passed to `App.bind("ssh", host=...)`
or `runplz ssh --host <name>` is whatever ssh/rsync treat as a reachable
endpoint — a bare hostname, a `user@host[:port]` URL, or an alias from
your ~/.ssh/config. SshConfig.user / .port, when set, override the URL
via ssh's -l / -p flags.

This backend shares all the SSH plumbing with the brev backend via the
public `runplz.backends.ssh_common` module.
"""

import re
import subprocess
from typing import Optional

from runplz.backends import docker
from runplz.backends.listing import JobRecord
from runplz.backends.ssh_common import (
    SshOptions,
    dispatch_to_target,
    orchestrator_signal_cleanup,
    parse_probe_sections,
    ssh_capture,
    ssh_cmd_opts,
    wait_until_ssh_reachable,
)

# `run` and `list_jobs` are the driver contract the registry calls; the
# rest is this driver's own testable surface.
__all__ = [
    "run",
    "list_jobs",
]


def run(app, function, args, kwargs, *, host: str, outputs_dir: str = "out"):
    cfg = app.ssh_config
    target, port = _build_ssh_target(host, user=cfg.user, port=cfg.port)
    ssh_opts = SshOptions(port=port, identity_file=cfg.ssh_key_path)

    wait_until_ssh_reachable(target, ssh_opts=ssh_opts, max_wait_s=cfg.ssh_ready_wait_seconds)
    _warn_on_spec_mismatch(target, function, ssh_opts=ssh_opts)

    # No teardown: the user owns this box's lifecycle, which is the whole
    # point of the ssh backend. Everything else is the shared dispatch.
    with orchestrator_signal_cleanup(target):
        dispatch_to_target(
            app=app,
            function=function,
            args=args,
            kwargs=kwargs,
            target=target,
            backend="ssh",
            outputs_dir=outputs_dir,
            mode="docker" if cfg.use_docker else "native",
            max_runtime_seconds=cfg.max_runtime_seconds,
            max_inactivity_seconds=cfg.max_inactivity_seconds,
            inactivity_action=cfg.inactivity_action,
            ssh_opts=ssh_opts,
        )


def list_jobs(
    *,
    host: str,
    user: Optional[str] = None,
    port: Optional[int] = None,
    ssh_key_path: Optional[str] = None,
) -> list[JobRecord]:
    """Return runplz jobs currently running on the SSH target ``host``.

    SSH has no registry of hosts, so the caller must supply one. Filters on
    the ``runplz=1`` label — the same label stamped by :func:`run_container_detached`.
    """
    target, resolved_port = _build_ssh_target(host, user=user, port=port)
    ssh_opts = SshOptions(port=resolved_port, identity_file=ssh_key_path)
    cmd = [
        "ssh",
        *ssh_cmd_opts(ssh_opts),
        target,
        f"sudo docker ps --filter {docker.PS_FILTER} "
        f"--format '{docker.PS_FORMAT}' 2>/dev/null || true",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        raise RuntimeError(
            f"ssh to {target!r} failed (rc={r.returncode}). "
            f"stderr: {(r.stderr or '').strip()[:300]}"
        )
    return docker.parse_ps_rows(r.stdout, backend="ssh", name_prefix=f"{target}:")


def _build_ssh_target(
    host: str, *, user: Optional[str], port: Optional[int]
) -> tuple[str, Optional[int]]:
    """Split a user-supplied host string into an ssh target + port.

    Accepts ``hostname``, ``user@hostname``, ``hostname:port``, and
    ``user@hostname:port``. IPv6 bracketed literals (``[::1]:22``) are
    passed to ssh as-is.

    Precedence: explicit ``SshConfig.user`` / ``SshConfig.port`` always
    win over values inlined into the host URL — explicit config should
    be obvious at the call site, not quietly overridden by user@host
    parsing.
    """
    bare = host
    if "@" in bare:
        existing_user, bare = bare.split("@", 1)
        if user is None:
            user = existing_user
    if ":" in bare and "]" not in bare:
        bare_candidate, existing_port = bare.rsplit(":", 1)
        try:
            parsed_port = int(existing_port)
        except ValueError:
            parsed_port = None
        if parsed_port is not None:
            bare = bare_candidate
            if port is None:
                port = parsed_port
    target = f"{user}@{bare}" if user else bare
    return target, port


# --- Spec-mismatch warnings ----------------------------------------------


_MEMINFO_LINE = re.compile(r"^MemTotal:\s+(\d+)\s+kB", re.MULTILINE)
_NVIDIA_LINE = re.compile(r"^([^,]+),\s*(\d+)\s*MiB$", re.MULTILINE)


def _warn_on_spec_mismatch(target: str, function, *, ssh_opts: Optional[SshOptions] = None) -> None:
    """Probe the remote box and warn when its specs don't meet the function's
    constraints. Best-effort — never raises. The user may know something we
    don't (overcommitting a dev box, MIG-partitioned GPUs, etc.).

    Probes nproc, /proc/meminfo, nvidia-smi in a single ssh call so we don't
    add latency for every dimension.
    """
    try:
        probe = ssh_capture(
            target,
            "echo '---NPROC---'; nproc; "
            "echo '---MEMINFO---'; cat /proc/meminfo 2>/dev/null | head -1; "
            "echo '---NVIDIA---'; "
            "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null; "
            "echo '---END---'",
            ssh_opts=ssh_opts,
        )
    except Exception as exc:  # noqa: BLE001
        print(
            f"+ warning: could not probe remote specs on {target}: {type(exc).__name__}: {exc}",
            flush=True,
        )
        return

    sections = parse_probe_sections(probe)
    warnings = []
    warnings.extend(_check_cpu(sections.get("NPROC", ""), function))
    warnings.extend(_check_memory(sections.get("MEMINFO", ""), function))
    warnings.extend(_check_gpu(sections.get("NVIDIA", ""), function))
    for w in warnings:
        print(f"+ spec-mismatch warning: {w}", flush=True)


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
