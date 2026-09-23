# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.2",
#   "transformers>=4.44",
#   "numpy>=1.26",
#   "scikit-learn>=1.4",
#   "huggingface_hub>=0.25",
# ]
# ///
"""Fine-tune one encoder on Hugging Face Jobs and push the result to the Hub.

Submitted with `hf jobs uv run`. The job clones this repository rather than
vendoring the pipeline, so it trains exactly the code the results were measured
on, and pushes the finished model straight to the Hub so nothing has to be
copied back down to a laptop.

    hf jobs uv run deploy/jobs/train_on_hub.py --flavor t4-small \
        --secrets HF_TOKEN --env MODEL=legal-bert --env REPO_ID=user/name
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = os.environ.get("SOURCE_REPO", "https://github.com/ManasDasri/NNDL.git")
MODEL = os.environ.get("MODEL", "legal-bert")
REPO_ID = os.environ.get("REPO_ID", "")
EPOCHS = os.environ.get("EPOCHS", "3")
BATCH_SIZE = os.environ.get("BATCH_SIZE", "16")
CHECKOUT = Path("/tmp/nndl")


def run(*command: str) -> None:
    print(f"\n$ {' '.join(command)}", flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    import torch

    print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"gpu: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        # Worth failing loudly: a CPU run here would cost money and take days.
        raise SystemExit("no GPU visible; refusing to train on CPU")

    if not CHECKOUT.exists():
        run("git", "clone", "--depth", "1", REPO, str(CHECKOUT))
    os.chdir(CHECKOUT)
    sys.path.insert(0, str(CHECKOUT / "src"))

    output_dir = CHECKOUT / "outputs" / MODEL.replace("-", "_")

    run(sys.executable, "-m", "legal_risk_classifier.prepare",
        "--cuad_json", "data/CUAD_v1.json", "--out_dir", "data/processed")

    run(sys.executable, "-m", "legal_risk_classifier.train_transformer",
        "--model", MODEL, "--epochs", EPOCHS, "--batch_size", BATCH_SIZE,
        "--grad_accumulation", "1", "--output_dir", str(output_dir))

    run(sys.executable, "-m", "legal_risk_classifier.evaluate",
        "--run_dir", str(output_dir), "--split", "test", "--pooling", "max")

    if REPO_ID:
        run(sys.executable, "-m", "legal_risk_classifier.export_hf",
            "--run_dir", str(output_dir), "--repo_id", REPO_ID, "--push")
    else:
        print("REPO_ID unset; trained but not pushed", flush=True)

    print("\n=== JOB COMPLETE ===", flush=True)


if __name__ == "__main__":
    main()
