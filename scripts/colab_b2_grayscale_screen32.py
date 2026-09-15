"""Run the B2 grayscale-contrast OCR ablation on the Drive screen32 split."""
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
    "--config", "configs/experiments/b2_grayscale_contrast_dmy_rule.yaml",
    "--input", str(dataset),
    "--output", "/content/B2_GRAYSCALE_CONTRAST_DMY_RULE_screen32",
    "--labels", "/content/labels.csv",
    "--image-ids-file", "/content/screen32_ids.txt",
    "--device", "cpu",
    "--threads", "4",
)
