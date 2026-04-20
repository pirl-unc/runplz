"""runplz — tiny Modal-shaped job harness.

One Python decoration, three backends (local Docker, Brev, Modal).

Write a job script:

    # jobs/train.py
    from runplz import App, Image

    # auto_create_instances=False by default — opt in to brev create
    app = App("my-job")
    image = (
        Image.from_registry("pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime")
        .pip_install("pandas", "scikit-learn")
        .pip_install_local_dir(".", editable=True)
    )

    @app.function(image=image, gpu="T4", min_cpu=4, min_memory=26)
    def train():
        import subprocess
        subprocess.run(["bash", "scripts/train.sh"], check=True)

    @app.local_entrypoint()
    def main():
        train.remote()

Then run it from the CLI:

    runplz local jobs/train.py
    runplz brev --instance my-box jobs/train.py
    runplz ssh  --host gpu.example.com jobs/train.py
    runplz modal jobs/train.py

Two entry points: the CLI (above) and ``App.bind(backend, ...)`` from
within any Python script / notebook. Both wire a backend to the App;
inside the entrypoint call ``fn.remote(...)`` to dispatch on the chosen
backend, or ``fn.local(...)`` to run the body in the current process.
"""

from runplz.app import App, Function
from runplz.config import BrevConfig, ModalConfig, SshConfig
from runplz.image import Image, ImageOp
from runplz.version import __version__

__all__ = [
    "App",
    "Function",
    "Image",
    "ImageOp",
    "BrevConfig",
    "ModalConfig",
    "SshConfig",
    "__version__",
]
