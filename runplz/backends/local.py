"""Local backend: docker build + docker run.

Auto-detects NVIDIA runtime via `docker info`; passes `--gpus all` when
available, omits otherwise. The training library handles CPU/MPS/CUDA
selection on its own.

Accepts both `Image.from_dockerfile(...)` (build from a user Dockerfile)
and `Image.from_registry(...).apt_install/pip_install/...` (synthesize
a Dockerfile on the fly, build from repo root as context).
"""

import json
import subprocess
from pathlib import Path

IMAGE_TAG_DEFAULT = "runplz-local"


def run(
    app,
    function,
    args,
    kwargs,
    *,
    image_tag: str = IMAGE_TAG_DEFAULT,
    build: bool = True,
    outputs_dir: str = "out",
):
    repo = app._repo_root
    if repo is None:
        raise RuntimeError("App repo_root not set (CLI should have set this).")

    host_out = (repo / outputs_dir).resolve()
    host_out.mkdir(parents=True, exist_ok=True)

    if build:
        _build_image(function.image, repo, image_tag)

    script_in_container = _container_path_for(function.module_file, repo)

    cmd = [
        "docker",
        "run",
        "--rm",
        "--name",
        f"runplz-{app.name}-{function.name}",
        "-v",
        f"{host_out}:/out",
        "-w",
        "/workspace",
        "-e",
        "RUNPLZ_OUT=/out",
        "-e",
        f"RUNPLZ_SCRIPT={script_in_container}",
        "-e",
        f"RUNPLZ_FUNCTION={function.name}",
        "-e",
        f"RUNPLZ_ARGS={json.dumps(args)}",
        "-e",
        f"RUNPLZ_KWARGS={json.dumps(kwargs)}",
    ]
    if _nvidia_available():
        cmd += ["--gpus", "all"]
    for k, v in function.env.items():
        cmd += ["-e", f"{k}={v}"]
    cmd += [image_tag, "python", "-m", "runplz._bootstrap"]

    _print_cmd(cmd)
    subprocess.run(cmd, check=True)


def _build_image(image, repo: Path, tag: str):
    """Build the Docker image. Handles both image shapes:
    - `Image.from_dockerfile(...)`: `docker build -f <path> -t <tag> <ctx>`
    - `Image.from_registry(...)` + DSL ops: pipe the synthesized
      Dockerfile into `docker build -f - -t <tag> <repo>` so
      `pip_install_local_dir`'s `COPY` can see the repo.
    """
    if image.dockerfile is not None:
        df, ctx = image.resolve(repo)
        cmd = ["docker", "build", "-f", str(df), "-t", tag, str(ctx)]
        _print_cmd(cmd)
        subprocess.run(cmd, check=True)
        return
    dockerfile_text = image.render_dockerfile()
    cmd = ["docker", "build", "-f", "-", "-t", tag, str(repo)]
    _print_cmd(cmd)
    subprocess.run(cmd, check=True, input=dockerfile_text, text=True)


def _nvidia_available() -> bool:
    r = subprocess.run(
        ["docker", "info", "--format", "{{json .Runtimes}}"],
        capture_output=True,
        text=True,
    )
    return r.returncode == 0 and "nvidia" in r.stdout


def _container_path_for(host_path: str, repo: Path) -> str:
    # Assumes the image bakes the repo at /workspace.
    rel = Path(host_path).resolve().relative_to(repo)
    return str(Path("/workspace") / rel)


def _print_cmd(cmd):
    # flush=True matches the convention in brev/modal backends — lets
    # users tailing a log file see status prints land before subprocess
    # output.
    print("+ " + " ".join(cmd), flush=True)
