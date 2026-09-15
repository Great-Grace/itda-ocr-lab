"""Run the PP-OCRv6 medium recognizer probe on the full screen32 split."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def run(*args: str) -> None:
    subprocess.run(args, check=True)


run("bash", "-lc", "rm -rf /content/itda_ocr_lab && mkdir -p /content/itda_ocr_lab && tar -xzf /content/itda_ocr_bundle.tar.gz -C /content/itda_ocr_lab")
os.chdir("/content/itda_ocr_lab")
dataset = Path("/content/drive/MyDrive/상품사진입니다")
os.environ["ITDA_WEIGHTS_ROOT"] = str(dataset / "weights")
run(
    sys.executable,
    "scripts/run_experiment.py",
    "--config", "configs/experiments/b4_ppocrv6_medium_probe.yaml",
    "--input", str(dataset),
    "--output", "/content/B4_PPOCRV6_MEDIUM_REC_screen32",
    "--labels", "/content/labels.csv",
    "--image-ids-file", "/content/screen32_ids.txt",
    "--device", "cpu",
    "--threads", "4",
)
