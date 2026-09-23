"""HuggingFace Space entry point.

The Space holds only a checkpoint and this file; the model code is installed
from the repository so the demo cannot drift from the evaluated pipeline.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/AnushreeKasturi/NNDL.git"
CHECKOUT = Path("nndl")

if not CHECKOUT.exists():
    subprocess.run(["git", "clone", "--depth", "1", REPO, str(CHECKOUT)], check=True)
sys.path.insert(0, str(CHECKOUT / "src"))

from legal_risk_classifier.demo import build_interface  # noqa: E402
from legal_risk_classifier.runtime import CNNRuntime  # noqa: E402

RUN_DIR = Path(os.environ.get("RUN_DIR", "run"))

runtime = CNNRuntime(RUN_DIR)
build_interface(runtime).launch()
