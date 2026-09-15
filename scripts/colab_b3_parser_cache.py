"""Run the day-first plus month-name parser ablation with cached B0 tokens."""
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
    "--config", "configs/experiments/b3_parser_dmy_monthname_raw_rule.yaml",
    "--input", str(dataset),
    "--output", "/content/B3_PARSER_DMY_MONTHNAME_RAW_RULE_screen32",
    "--labels", "/content/labels.csv",
    "--image-ids-file", "/content/screen32_ids.txt",
    "--tokens-cache", "/content/b0_screen32_tokens.jsonl",
    "--device", "cpu",
    "--threads", "4",
)
