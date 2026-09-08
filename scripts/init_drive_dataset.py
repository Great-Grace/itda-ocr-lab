#!/usr/bin/env python3
"""Create the shared-dataset manifest after images are placed in Drive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Initialize DATASET_MANIFEST.yaml for a shared OCR dataset folder.")
    parser.add_argument("root", help="Drive dataset root, e.g. /content/drive/MyDrive/ITDA_OCR")
    parser.add_argument("--dataset-id", default="itda-ocr-public")
    parser.add_argument("--version", default="1")
    parser.add_argument("--image-dir", help="Images directory relative to root; defaults to images/ when present, otherwise root")
    parser.add_argument("--labels-file", help="Optional labels CSV path relative to root")
    parser.add_argument("--force", action="store_true", help="Replace an existing manifest")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.is_dir():
        raise SystemExit(f"Dataset root does not exist: {root}")
    image_rel = args.image_dir or ("images" if (root / "images").is_dir() else ".")
    image_dir = root / image_rel
    images = sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise SystemExit(f"No supported images found in: {image_dir}")
    manifest_path = root / "DATASET_MANIFEST.yaml"
    if manifest_path.exists() and not args.force:
        raise SystemExit(f"Manifest already exists: {manifest_path}. Use --force only after checking the dataset version.")

    manifest = {
        "dataset_id": args.dataset_id,
        "version": str(args.version),
        "image_dir": image_rel,
        "expected_image_count": len(images),
        "allowed_extensions": sorted(ext.lstrip(".") for ext in IMAGE_SUFFIXES),
    }
    if args.labels_file:
        manifest["labels_file"] = args.labels_file
    manifest_path.write_text(yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(json.dumps({"status": "ready", "manifest": str(manifest_path), **manifest}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
