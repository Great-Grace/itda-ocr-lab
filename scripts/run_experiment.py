#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ocr_lab.config import load_config
from ocr_lab.pipeline import run_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one config-first OCR experiment.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--labels")
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--device", choices=["cpu", "cuda"], help="Override runtime device for this run")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.device:
        config.setdefault("runtime", {})["device"] = args.device
    metrics = run_pipeline(config, args.input, args.output, args.labels, args.max_images)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
