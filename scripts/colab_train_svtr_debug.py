from __future__ import annotations

import os
import subprocess
import sys

subprocess.run([
    "bash", "-lc",
    "mkdir -p /content/itda_ocr_lab && tar -xzf /content/itda_ocr_bundle.tar.gz -C /content/itda_ocr_lab",
], check=True)
os.chdir("/content/itda_ocr_lab")
subprocess.run([
    sys.executable, "scripts/train_synthetic_svtr_tiny.py",
    "--output", "/content/attn_runs/svtr_tiny_ctc_debug",
    "--train-count", "10000",
    "--val-count", "1000",
    "--epochs", "24",
    "--batch-size", "128",
    "--device", "cuda",
], check=True)
