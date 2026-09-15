"""Fail-fast Colab preflight for the ITDA Drive dataset.

This script intentionally performs exactly one mount attempt when requested;
it never retries OAuth or silently falls back to another folder.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mount", action="store_true", help="Attempt one readonly Google Drive mount")
    parser.add_argument("--mount-root", default="/content/drive")
    parser.add_argument("--dataset-name", default="상품사진입니다")
    parser.add_argument("--expected-images", type=int, default=3352)
    args = parser.parse_args()

    if args.mount:
        from google.colab import drive
        drive.mount(args.mount_root, readonly=True)

    root = Path(args.mount_root) / "MyDrive" / args.dataset_name
    if not root.is_dir():
        raise SystemExit(f"Drive preflight failed: dataset folder not found: {root}")
    image_suffixes = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    image_count = sum(1 for path in root.iterdir() if path.is_file() and path.suffix.lower() in image_suffixes)
    if image_count != args.expected_images:
        raise SystemExit(f"Drive preflight failed: expected {args.expected_images} images, found {image_count} in {root}")
    weights = root / "weights" / "paddle"
    required = [weights / "ppocrv5_mobile_det", weights / "korean_ppocrv5_mobile_rec"]
    missing = [str(path) for path in required if not path.is_dir()]
    if missing:
        raise SystemExit("Drive preflight failed: missing weight directories: " + ", ".join(missing))
    result = {"root": str(root), "image_count": image_count, "weights_root": str(root / "weights"), "status": "ok"}
    Path("/content/itda_drive_preflight.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
