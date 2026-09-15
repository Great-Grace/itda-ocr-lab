"""Train the first attention-suite models on a GPU Colab runtime."""
from __future__ import annotations

import subprocess
import sys
import os
from pathlib import Path


def run(*args: str) -> None:
    subprocess.run(args, check=True)


run("bash", "-lc", "mkdir -p /content/itda_ocr_lab && tar -xzf /content/itda_ocr_bundle.tar.gz -C /content/itda_ocr_lab")
root = Path("/content/itda_ocr_lab")
os.chdir(root)
for name, script_args in [
    ("svtr_tiny_ctc", ["scripts/train_synthetic_svtr_tiny.py", "--output", "/content/attn_runs/svtr_tiny_ctc"]),
    ("crnn_ctc", ["scripts/train_synthetic_attention.py", "--architecture", "crnn_ctc", "--output", "/content/attn_runs/crnn_ctc"]),
    ("crnn_attn", ["scripts/train_synthetic_attention.py", "--architecture", "crnn_attn", "--output", "/content/attn_runs/crnn_attn"]),
]:
    print(f"START {name}", flush=True)
    run(sys.executable, *script_args, "--train-count", "10000", "--val-count", "1000", "--epochs", "8", "--batch-size", "128", "--device", "cuda")
    print(f"DONE {name}", flush=True)
