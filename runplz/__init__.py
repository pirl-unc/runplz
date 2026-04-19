"""runplz — tiny Modal-shaped job harness.

One Python decoration, three backends (local Docker, Brev, Modal).

Minimal example:

    from runplz import App, BrevConfig, Image

    app = App("my-job", brev=BrevConfig(instance_type="g2-standard-4:nvidia-l4:1"))
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

Run via the CLI:

    runplz local path/to/job.py
    runplz brev --instance my-box path/to/job.py
    runplz modal path/to/job.py
"""

from runplz.app import App, Function
from runplz.config import BrevConfig, ModalConfig
from runplz.image import Image, ImageOp

__version__ = "1.0.0"
__all__ = [
    "App",
    "Function",
    "Image",
    "ImageOp",
    "BrevConfig",
    "ModalConfig",
    "__version__",
]
