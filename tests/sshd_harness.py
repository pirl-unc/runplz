"""A throwaway sshd, so the end-to-end tier needs no machine setup.

`sshd` runs fine unprivileged on a high port as long as it only ever
authenticates the user that started it -- exactly our case. So the ssh
end-to-end tests do not depend on Remote Login being enabled, on a
CI-only service, or on any state outside a temp directory.

Everything lives in that temp dir: host key, client key, authorized_keys,
known_hosts, config, pid, log. Nothing touches ~/.ssh, and the daemon binds to
127.0.0.1 on an ephemeral port so it is not reachable off the machine.
"""

import platform
import shutil
import socket
import subprocess
import time
from pathlib import Path

SSHD_CANDIDATES = ("/usr/sbin/sshd", "/usr/local/sbin/sshd", "sshd")


def find_sshd():
    """Path to an sshd binary, or None if this machine has none."""
    for candidate in SSHD_CANDIDATES:
        found = candidate if "/" in candidate else shutil.which(candidate)
        if found and Path(found).exists():
            return found
    return None


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _keygen(path: Path) -> None:
    subprocess.run(
        ["ssh-keygen", "-q", "-t", "ed25519", "-f", str(path), "-N", ""],
        check=True,
        capture_output=True,
    )


class LocalSshd:
    """A private sshd on 127.0.0.1. Address it with `.port` / `.identity`."""

    #: How to address this box. `SshOptions` carries the port and identity
    #: but not the user, so any user must live in the target string --
    #: otherwise the code under test connects as whoever is running pytest.
    target = "127.0.0.1"

    def __init__(self, root: Path):
        self.root = root
        self.port = _free_port()
        self.identity = root / "id"
        self.known_hosts = root / "known_hosts"

    def ssh_options(self):
        """Production options pointed entirely at the harness temp directory."""
        from runplz.backends.ssh_common import SshOptions

        return SshOptions(
            port=self.port,
            identity_file=str(self.identity),
            known_hosts_file=str(self.known_hosts),
        )

    def start(self) -> "LocalSshd":
        if not (self.root / "hostkey").exists():
            _keygen(self.root / "hostkey")
        if not self.identity.exists():
            _keygen(self.identity)
        (self.root / "authorized_keys").write_bytes((self.root / "id.pub").read_bytes())
        for name in ("authorized_keys", "id", "hostkey"):
            (self.root / name).chmod(0o600)

        config = self.root / "sshd_config"
        config.write_text(
            f"Port {self.port}\n"
            "ListenAddress 127.0.0.1\n"
            f"HostKey {self.root / 'hostkey'}\n"
            f"PidFile {self.root / 'sshd.pid'}\n"
            f"AuthorizedKeysFile {self.root / 'authorized_keys'}\n"
            # A pytest tmp dir is not owned or moded the way sshd expects a
            # home directory to be. Safe here: the only key that can
            # authenticate is the one generated two lines above, and the
            # daemon is bound to loopback.
            "StrictModes no\n"
            "UsePAM no\n"
            "PasswordAuthentication no\n"
            "PubkeyAuthentication yes\n"
            "LogLevel QUIET\n"
        )
        subprocess.run(
            [find_sshd(), "-f", str(config), "-E", str(self.root / "sshd.log")],
            check=True,
            capture_output=True,
        )
        self._await_ready()
        return self

    def _await_ready(self, attempts: int = 40) -> None:
        for _ in range(attempts):
            if self.probe():
                return
            time.sleep(0.25)
        log_path = self.root / "sshd.log"
        log = log_path.read_text() if log_path.exists() else "(no log)"
        raise RuntimeError(f"sshd on port {self.port} never became reachable:\n{log}")

    def probe(self) -> bool:
        try:
            proc = subprocess.run(
                ["ssh", *self.ssh_args(), self.target, "true"],
                capture_output=True,
                timeout=20,
            )
        except Exception:
            return False
        return proc.returncode == 0

    def ssh_args(self) -> list:
        return [
            "-i",
            str(self.identity),
            "-p",
            str(self.port),
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            "IdentitiesOnly=yes",
        ]

    def stop(self) -> None:
        pid_file = self.root / "sshd.pid"
        if not pid_file.exists():
            return
        try:
            subprocess.run(["kill", pid_file.read_text().strip()], capture_output=True, timeout=10)
        except Exception:
            pass

    def refuse_connections(self) -> None:
        """Stop accepting connections while preserving the endpoint options."""
        self.stop()

    def drop_connection(self) -> None:
        """Drop connections that are already established, not just the listener.

        `stop()` kills the pid in sshd.pid, which is the *listener*; an
        in-flight session is a forked child and survives it. So a test that
        called this mid-command watched its command finish normally and then
        asserted an exception that was really coming from somewhere else.
        Killing the children first is what actually severs the transport.
        """
        pid_file = self.root / "sshd.pid"
        if pid_file.exists():
            try:
                subprocess.run(
                    ["pkill", "-P", pid_file.read_text().strip()],
                    capture_output=True,
                    timeout=10,
                )
            except Exception:
                pass
        self.stop()


# ---------------------------------------------------------------------------
# Optional container backend.
#
# The local sshd runs the tests against *this* machine, so on a Mac the
# "remote" is macOS -- which is not what anyone deploys to, and where
# detached launch is broken outright (issue #92). A Linux container makes
# the remote match production on any host, so the detached tests run on a
# Mac instead of skipping. It costs a Docker daemon, so it is not the
# default: `--e2e-remote` selects, and `auto` only reaches for Docker
# where the local sshd would be the wrong platform.

DOCKERFILE = """\
FROM debian:stable-slim
RUN apt-get update \
 && apt-get install -y --no-install-recommends openssh-server rsync procps \
 && apt-get clean \
 && mkdir -p /run/sshd /root/.ssh \
 && chmod 700 /root/.ssh
COPY authorized_keys /root/.ssh/authorized_keys
RUN chmod 600 /root/.ssh/authorized_keys \
 && sed -i 's/^#*PermitRootLogin.*/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config
# PID 1 is deliberately not sshd: the fault tests kill the daemon, and
# killing PID 1 would take the container -- and its published port -- with
# it, so a "refused connection" would be indistinguishable from a vanished
# endpoint. sshd is started beside it by the harness.
CMD ["sleep", "infinity"]
"""


def docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=30).returncode == 0
    except Exception:
        return False


class DockerSshd:
    """A Debian container running sshd, addressed like LocalSshd."""

    def __init__(self, root: Path):
        self.root = root
        self.identity = root / "id"
        self.known_hosts = root / "known_hosts"
        self._image = None
        self._container = None
        self.host = "127.0.0.1"
        self.port = None
        self.target = "root@127.0.0.1"

    def start(self) -> "DockerSshd":
        from testcontainers.core.container import DockerContainer
        from testcontainers.core.image import DockerImage

        if self._container is None:
            # Guarded like LocalSshd.start: `ssh-keygen -f <existing>` prompts
            # before overwriting, and with no tty that is a non-zero exit — so
            # an unguarded keygen turns the second start() into a hard failure.
            # Only reachable now that fault tests restore the daemon by calling
            # start() again.
            if not self.identity.exists():
                _keygen(self.identity)
            context = self.root / "ctx"
            context.mkdir(parents=True, exist_ok=True)
            (context / "Dockerfile").write_text(DOCKERFILE)
            (context / "authorized_keys").write_bytes((self.root / "id.pub").read_bytes())
            self.identity.chmod(0o600)

            self._image = DockerImage(path=str(context), tag="runplz-e2e-sshd:test").build()
            self._container = DockerContainer(str(self._image)).with_exposed_ports(22).start()
            self.host = self._container.get_container_host_ip()
            self.port = int(self._container.get_exposed_port(22))
            # The container's only account is root. `SshOptions` cannot express
            # a user, so it has to ride in the target -- and the harness probe
            # must use the same address, or it will report a healthy box that
            # the code under test cannot reach.
            self.target = f"root@{self.host}"
        # Idempotent like LocalSshd.start: a fault test kills the daemon and
        # calls start() to restore it, and must get the same endpoint back.
        self._start_sshd()
        self._await_ready()
        return self

    _SSHD_PID = "/run/sshd.pid"

    def _sh(self, script: str):
        """Run a shell snippet inside the container."""
        return self._container.exec(["sh", "-c", script])

    def _start_sshd(self) -> None:
        # No -D: sshd daemonizes, so this exec returns once it is listening.
        self._sh(f"/usr/sbin/sshd -e -o PidFile={self._SSHD_PID}")

    def refuse_connections(self) -> None:
        """Stop accepting connections while keeping the endpoint addressable.

        The container and its published port stay up, so a client gets a
        refusal rather than the endpoint disappearing underneath it.
        """
        self._sh(f"[ -f {self._SSHD_PID} ] && kill $(cat {self._SSHD_PID}) || true")

    def drop_connection(self) -> None:
        """Sever sessions that are already established, not just the listener.

        Same shape as the LocalSshd fix: an in-flight session is a forked child
        of the daemon and outlives it, so killing the listener alone lets a
        running command finish normally.
        """
        self._sh(
            f"[ -f {self._SSHD_PID} ] && pkill -P $(cat {self._SSHD_PID}) || true; "
            f"[ -f {self._SSHD_PID} ] && kill $(cat {self._SSHD_PID}) || true"
        )

    _await_ready = LocalSshd._await_ready
    probe = LocalSshd.probe
    ssh_options = LocalSshd.ssh_options

    def ssh_args(self) -> list:
        return [
            "-i",
            str(self.identity),
            "-p",
            str(self.port),
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            "IdentitiesOnly=yes",
        ]

    def stop(self) -> None:
        for resource in (self._container, self._image):
            try:
                if resource is not None:
                    resource.stop()
            except Exception:
                pass


MODES = ("auto", "local", "docker")


def select_backend(root: Path, mode: str = "auto"):
    """Pick an ssh backend for the requested mode.

    `local` (a real sshd on this machine, no daemon needed), `docker` (a
    Linux container, matching production), or `auto` -- which uses Docker
    only where the local sshd would give the wrong platform, so Linux CI
    keeps the faster path and a Mac gets Linux fidelity when Docker is up.

    The mode is passed in rather than read from the environment here: it is
    one `--e2e-remote` option, parsed once by conftest, so this module and the
    fixture cannot disagree about what was asked for.

    Returns (backend, reason_if_unavailable).
    """
    if mode not in MODES:
        raise ValueError(f"--e2e-remote must be one of {MODES}; got {mode!r}")
    if mode == "local":
        return (LocalSshd(root), None) if find_sshd() else (None, "no sshd binary")
    if mode == "docker":
        if not docker_available():
            return None, "--e2e-remote=docker but no Docker daemon is reachable"
        return DockerSshd(root), None
    if platform.system() != "Linux" and docker_available():
        return DockerSshd(root), None
    if find_sshd():
        return LocalSshd(root), None
    if docker_available():
        return DockerSshd(root), None
    return None, "neither an sshd binary nor a Docker daemon is available"
