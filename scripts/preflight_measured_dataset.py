#!/usr/bin/env python3
"""Check image/split/weight coverage before a costly measured CPU run."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--split-csv", required=True)
    args = parser.parse_args()
    config_path, image_root, split_path = Path(args.config), Path(args.input), Path(args.split_csv)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    params = ((config.get("ocr") or {}).get("params") or {})
    ids = [Path(row["filename"]).stem for row in csv.DictReader(split_path.open(encoding="utf-8"))]
    indexed = {path.stem for path in image_root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES} if image_root.is_dir() else set()
    weight_dirs = {
        key: os.path.expandvars(str(value))
        for key, value in params.items()
        if key.endswith("_model_dir") and value
    }
    result = {
        "input_exists": image_root.is_dir(),
        "split_rows": len(ids),
        "missing_images": sorted(set(ids) - indexed),
        "weights": {key: {"path": value, "exists": Path(value).is_dir()} for key, value in weight_dirs.items()},
    }
    result["ready"] = result["input_exists"] and not result["missing_images"] and all(item["exists"] for item in result["weights"].values())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
