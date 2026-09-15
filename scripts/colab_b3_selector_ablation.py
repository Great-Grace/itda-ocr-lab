"""Run rule-selector ablations on cached B3 dev100 OCR tokens."""
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
    "scripts/run_ablation.py",
    "--config", "configs/experiments/b3_parser_dmy_monthname_raw_rule.yaml",
    "--input", str(dataset),
    "--output", "/content/B3_SELECTOR_ABLATION_dev100",
    "--labels", "/content/labels.csv",
    "--image-ids-file", "/content/confirm100_ids.txt",
    "--tokens-cache", "/content/b3_dev100_tokens.jsonl",
)
