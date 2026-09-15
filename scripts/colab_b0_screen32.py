"""Colab bootstrap for the CPU-only B0 screen32 experiment."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def run(*args: str) -> None:
    subprocess.run(args, check=True)


run(sys.executable, "-m", "pip", "install", "--quiet", "paddlepaddle==3.3.1", "paddleocr==3.7.0", "psutil", "pyyaml")
run("bash", "-lc", "mkdir -p /content/itda_ocr_lab && tar -xzf /content/itda_ocr_bundle.tar.gz -C /content/itda_ocr_lab")
os.chdir("/content/itda_ocr_lab")

dataset = Path("/content/drive/MyDrive/상품사진입니다")
weights = dataset / "weights"
if not dataset.is_dir():
    raise RuntimeError(f"Drive dataset missing: {dataset}")
if not weights.is_dir():
    raise RuntimeError(f"Local model weights missing: {weights}")

os.environ["ITDA_WEIGHTS_ROOT"] = str(weights)
run(
    sys.executable,
    "scripts/run_experiment.py",
    "--config", "configs/experiments/b0_pp_ocr_mobile_raw_rule.yaml",
    "--input", str(dataset),
    "--output", "/content/B0_PP_OCR_MOBILE_RAW_RULE_screen32",
    "--labels", "/content/labels.csv",
    "--image-ids-file", "/content/screen32_ids.txt",
    "--device", "cpu",
    "--threads", "4",
)
