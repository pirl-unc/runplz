"""Minimal runplz example — trains a trivial PyTorch model.

Invoke via the CLI (never `python examples/simple_job.py` directly):
    runplz local  examples/simple_job.py
    runplz brev   --instance my-gpu-box examples/simple_job.py
    runplz modal  examples/simple_job.py

The CLI imports this file, attaches the selected backend to `app`, and
calls `main()` (the @local_entrypoint). Inside main(), `train.remote()`
dispatches to the backend.
"""

from runplz import App, BrevConfig, Image

app = App(
    "runplz-simple",
    brev_config=BrevConfig(auto_create_instances=False, mode="vm"),
)

image = (
    Image.from_registry("pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime")
    .apt_install("rsync")
    .pip_install("numpy>=1.22")
)


@app.function(image=image, gpu="T4", min_cpu=2, min_memory=16, timeout=10 * 60)
def train():
    import os

    import torch

    print("cuda available:", torch.cuda.is_available(), flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.nn.Linear(128, 10).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for step in range(200):
        x = torch.randn(64, 128, device=device)
        y = model(x).sum()
        opt.zero_grad(); y.backward(); opt.step()
        if step % 50 == 0:
            print(f"step {step}: loss={y.item():.4f}", flush=True)
    out = os.environ.get("RUNPLZ_OUT", "/out")
    os.makedirs(out, exist_ok=True)
    torch.save(model.state_dict(), f"{out}/weights.pt")
    print(f"wrote {out}/weights.pt", flush=True)


@app.local_entrypoint()
def main():
    train.remote()
