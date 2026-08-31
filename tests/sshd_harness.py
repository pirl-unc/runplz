"""A throwaway sshd, so the end-to-end tier needs no machine setup.

`sshd` runs fine unprivileged on a high port as long as it only ever
authenticates the user that started it -- exactly our case. So the ssh
end-to-end tests do not depend on Remote Login being enabled, on a
CI-only service, or on any state outside a temp directory.

Everything lives in that temp dir: host key, client key, authorized_keys,
config, pid, log. Nothing touches ~/.ssh, and the daemon binds to
127.0.0.1 on an ephemeral port so it is not reachable off the machine.
"""

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

    def __init__(self, root: Path):
        self.root = root
        self.port = _free_port()
        self.identity = root / "id"

    def start(self) -> "LocalSshd":
        _keygen(self.root / "hostkey")
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
                ["ssh", *self.ssh_args(), "127.0.0.1", "true"],
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
