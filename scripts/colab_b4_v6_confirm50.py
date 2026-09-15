"""Run PP-OCRv6 medium recognition on a 50-image dev confirmation split."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def run(*args: str) -> None:
    subprocess.run(args, check=True)


run(sys.executable, "-m", "pip", "install", "--quiet", "paddlepaddle==3.3.1", "paddleocr==3.7.0", "psutil", "pyyaml")
run("bash", "-lc", "rm -rf /content/itda_ocr_lab && mkdir -p /content/itda_ocr_lab && tar -xzf /content/itda_ocr_bundle.tar.gz -C /content/itda_ocr_lab")
os.chdir("/content/itda_ocr_lab")
dataset = Path("/content/drive/MyDrive/상품사진입니다")
os.environ["ITDA_WEIGHTS_ROOT"] = str(dataset / "weights")
run(
    sys.executable,
    "scripts/run_experiment.py",
    "--config", "configs/experiments/b4_ppocrv6_medium_probe.yaml",
    "--input", str(dataset),
    "--output", "/content/B4_PPOCRV6_MEDIUM_REC_dev50",
    "--labels", "/content/labels.csv",
    "--image-ids-file", "/content/confirm50_ids.txt",
    "--device", "cpu",
    "--threads", "4",
)
