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
    parser.add_argument("--discover", action="store_true", help="Search this user's Drive for a likely image dataset folder")
    parser.add_argument("--initialize", action="store_true", help="Create a manifest for one uniquely discovered image folder")
    parser.add_argument("--min-images", type=int, default=2500)
    parser.add_argument("--max-images", type=int, default=5000)
    parser.add_argument("--max-depth", type=int, default=5)
    args = parser.parse_args()
    root = Path(args.mount_root)
    candidates = [root / relative for relative in (args.folders or DEFAULT_CANDIDATES)]
    if args.discover:
        candidates = _manifest_candidates(root) or _image_folder_candidates(
            root, args.min_images, args.max_images, args.max_depth
        )
    matches = []
    for folder in candidates:
        manifest = folder / "DATASET_MANIFEST.yaml"
        if not manifest.exists():
            if args.discover:
                matches.append(_uninitialized_candidate(folder))
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
    if match.get("status") == "candidate_requires_initialization":
        if not args.initialize:
            print(json.dumps(match, ensure_ascii=False, indent=2))
            return 2
        match = _initialize_candidate(match)
    match["status"] = "ready" if match["count_ok"] else "count_mismatch"
    print(json.dumps(match, ensure_ascii=False, indent=2))
    return 0 if match["count_ok"] else 2


def _manifest_candidates(root: Path) -> list[Path]:
    return [path.parent for path in root.rglob("DATASET_MANIFEST.yaml")]


def _image_folder_candidates(root: Path, minimum: int, maximum: int, max_depth: int) -> list[Path]:
    candidates: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_dir() or len(path.relative_to(root).parts) > max_depth:
            continue
        try:
            count = sum(
                1 for child in path.iterdir()
                if child.is_file() and child.suffix.lower() in IMAGE_SUFFIXES
            )
        except OSError:
            continue
        if minimum <= count <= maximum:
            candidates.append(path)
    return candidates


def _uninitialized_candidate(image_folder: Path) -> dict:
    root = image_folder.parent if image_folder.name.lower() == "images" else image_folder
    image_dir = "images" if root / "images" == image_folder else "."
    count = sum(1 for path in image_folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    return {
        "status": "candidate_requires_initialization",
        "root": str(root),
        "image_dir": image_dir,
        "image_count": count,
        "expected_image_count": count,
        "count_ok": True,
    }


def _initialize_candidate(candidate: dict) -> dict:
    root = Path(candidate["root"])
    manifest = {
        "dataset_id": "itda-ocr-auto-discovered",
        "version": "1",
        "image_dir": candidate["image_dir"],
        "expected_image_count": candidate["image_count"],
        "allowed_extensions": sorted(ext.lstrip(".") for ext in IMAGE_SUFFIXES),
    }
    path = root / "DATASET_MANIFEST.yaml"
    path.write_text(yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return {
        "status": "ready",
        "root": str(root),
        "manifest": str(path),
        "dataset_id": manifest["dataset_id"],
        "version": manifest["version"],
        "image_dir": str(root / manifest["image_dir"]),
        "image_count": manifest["expected_image_count"],
        "expected_image_count": manifest["expected_image_count"],
        "count_ok": True,
    }


if __name__ == "__main__":
    raise SystemExit(main())
