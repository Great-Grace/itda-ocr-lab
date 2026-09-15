"""Map local ITDA images to Kaggle detector boxes without downloading the archive.

The Kaggle archive manifest produced by ``build_kaggle_archive_manifest.py``
contains CRC32 and byte offsets. This command uses CRC32 to identify the exact
Kaggle image, then fetches only its small YOLO label member by HTTP Range.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import struct
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


CLASS_NAMES = {
    "region": ["date", "due", "code", "full"],
    "date": ["date", "due", "code"],
    "dmy": ["Y4", "ENG_MON", "NUM2"],
}


def _member_bytes(archive_url: str, row: dict) -> bytes:
    offset = int(row["offset"])
    head = urllib.request.urlopen(
        urllib.request.Request(archive_url, headers={"Range": f"bytes={offset}-{offset + 2047}"}), timeout=60
    ).read()
    fields = struct.unpack_from("<4s5H3L2H", head, 0)
    name_len, extra_len = fields[9], fields[10]
    start = offset + 30 + name_len + extra_len
    end = start + int(row["compressed_size"]) - 1
    payload = urllib.request.urlopen(
        urllib.request.Request(archive_url, headers={"Range": f"bytes={start}-{end}"}), timeout=60
    ).read()
    if int(row.get("compression", 8)) == 0:
        return payload
    if int(row.get("compression", 8)) == 8:
        return zlib.decompress(payload, -15)
    raise ValueError(f"unsupported compression method: {row.get('compression')}")


def _local_images(root: Path, ids: set[str] | None) -> list[Path]:
    paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    if not ids:
        return paths
    return [p for p in paths if p.stem in ids]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-manifest", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", choices=sorted(CLASS_NAMES), default="region")
    parser.add_argument("--ids", help="optional text file containing image stems to include")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    manifest = json.loads(Path(args.archive_manifest).read_text(encoding="utf-8"))
    if isinstance(manifest, dict):
        archive_url, files = manifest["archive_url"], manifest["files"]
    else:
        archive_url, files = "", manifest
    prefix = {"region": "expiry_region_detection_dataset", "date": "expiry_date_detection_dataset", "dmy": "expiry_dmy_detection_dataset"}[args.dataset]
    image_rows = {}
    label_rows = {}
    for row in files:
        name = row["name"]
        if name.startswith(f"{prefix}/{prefix}/images/"):
            image_rows[str(row["crc"])] = row
        elif name.startswith(f"{prefix}/{prefix}/labels/") and name.endswith(".txt"):
            label_rows[name] = row
    ids = set(Path(args.ids).read_text(encoding="utf-8").split()) if args.ids else None
    def build_record(image_path: Path) -> dict:
        raw = image_path.read_bytes()
        crc = str(zlib.crc32(raw) & 0xFFFFFFFF)
        image_row = image_rows.get(crc)
        base = {"image_id": image_path.stem, "image": str(image_path), "crc": crc}
        if image_row is None:
            return {**base, "status": "no_kaggle_image_match"}
        image_name = image_row["name"]
        label_name = image_name.replace("/images/", "/labels/")
        label_name = str(Path(label_name).with_suffix(".txt"))
        label_row = label_rows.get(label_name)
        if label_row is None:
            return {**base, "status": "no_label", "kaggle_image": image_name}
        text = _member_bytes(archive_url, label_row) if archive_url else ""
        boxes = []
        for line in text.decode("utf-8", "replace").splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            cls, cx, cy, width, height = map(float, parts)
            boxes.append({"class": int(cls), "name": CLASS_NAMES[args.dataset][int(cls)], "cx": cx, "cy": cy, "width": width, "height": height})
        return {**base, "status": "ok", "kaggle_image": image_name, "label": label_name, "boxes": boxes}

    image_paths = _local_images(Path(args.images), ids)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        records = list(pool.map(build_record, image_paths))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"dataset": args.dataset, "classes": CLASS_NAMES[args.dataset], "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    counts = {}
    for row in records:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print(json.dumps({"images": len(records), "counts": counts, "output": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
