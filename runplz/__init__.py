"""runplz — tiny Modal-shaped job harness.

One Python decoration, six backends (local Docker, Brev, SSH, GCP, AWS, Modal).
Smallest working script — a single `@app.function` is enough; runplz
auto-runs it as the entrypoint when there's exactly one:

    # jobs/train.py
    from runplz import App, Image

    app = App("my-job")

    @app.function(
        image=Image.from_registry("pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime"),
        gpu="T4",
    )
    def train():
        import torch
        print("cuda available:", torch.cuda.is_available())

Run it from the CLI:

    runplz local jobs/train.py                       # docker on your machine
    runplz brev  jobs/train.py                       # ephemeral GPU box
    runplz brev  --instance my-box jobs/train.py     # attach to existing brev box
    runplz ssh   --host gpu.example.com jobs/train.py
    runplz gcp   jobs/train.py                       # ephemeral GCE VM
    runplz aws   jobs/train.py                       # ephemeral EC2 instance
    runplz modal jobs/train.py

Multi-step driver? Add an explicit ``@app.local_entrypoint()`` and call
``.remote()`` yourself. The two entry points are the CLI (above) and
``App.bind(backend, ...)`` from any Python script / notebook. See
``runplz ps`` / ``runplz tail`` / ``runplz status`` for follow-along
operations on long-running runs.
"""

from runplz.app import App, Function
from runplz.config import AwsConfig, BrevConfig, GcpConfig, ModalConfig, SshConfig
from runplz.image import Image, ImageOp
from runplz.version import __version__

__all__ = [
    "App",
    "Function",
    "Image",
    "ImageOp",
    "AwsConfig",
    "BrevConfig",
    "GcpConfig",
    "ModalConfig",
    "SshConfig",
    "__version__",
]
