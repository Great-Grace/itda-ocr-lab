#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

DEFAULT_CANDIDATES = [
    "MyDrive/ITDA_OCR",
    "Shareddrives/ITDA_OCR",
    "MyDrive/ITDA_OCR_DATASET",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Find and validate the shared ITDA OCR Drive folder.")
    parser.add_argument("--mount-root", default="/content/drive")
    parser.add_argument("--folder", action="append", dest="folders")
    parser.add_argument("--expected-count", type=int, default=3352)
    args = parser.parse_args()
    root = Path(args.mount_root)
    candidates = [root / relative for relative in (args.folders or DEFAULT_CANDIDATES)]
    matches = []
    for folder in candidates:
        manifest = folder / "DATASET_MANIFEST.yaml"
        if not manifest.exists():
            continue
        data = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
        image_dir = folder / str(data.get("image_dir", "images"))
        count = sum(
            1 for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ) if image_dir.exists() else 0
        expected = int(data.get("expected_image_count", args.expected_count))
        matches.append({
            "root": str(folder),
            "manifest": str(manifest),
            "dataset_id": data.get("dataset_id"),
            "version": data.get("version"),
            "image_dir": str(image_dir),
            "image_count": count,
            "expected_image_count": expected,
            "count_ok": count == expected,
        })
    if len(matches) != 1:
        print(json.dumps({"status": "needs_input", "matches": matches}, ensure_ascii=False, indent=2))
        return 2
    match = matches[0]
    match["status"] = "ready" if match["count_ok"] else "count_mismatch"
    print(json.dumps(match, ensure_ascii=False, indent=2))
    return 0 if match["count_ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
