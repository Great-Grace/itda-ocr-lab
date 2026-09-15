"""Audit detector coverage using Kaggle's region-train YOLO labels.

The Drive images are matched to the Kaggle archive by image CRC32. Only
annotation metadata is downloaded by HTTP range requests; the 2.7GB archive
is never materialized locally.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import struct
import zlib
from pathlib import Path

import requests
from PIL import Image

from ocr_lab.contracts import OCRToken


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--ids", required=True)
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--manifest", default="/private/tmp/kaggle_manifest_corrected.json")
    parser.add_argument("--headers", default="/private/tmp/kaggle_headers")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import os
    import requests

    headers = Path(args.headers).read_text(encoding="utf-8")
    archive_url = re.search(r"^location: (.+)$", headers, re.M | re.I).group(1).strip()
    archive_rows = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    by_crc = {row["crc"]: row for row in archive_rows if row["name"].startswith("expiry_region_detection_dataset/") and "/images/train/" in row["name"]}
    labels_by_stem = {}
    for row in archive_rows:
        name = row["name"]
        if name.startswith("expiry_region_detection_dataset/") and "/labels/train/" in name and name.endswith(".txt"):
            labels_by_stem[Path(name).stem] = row
    token_rows = {row["image_id"]: [OCRToken(**item) for item in row.get("tokens", [])] for row in map(json.loads, Path(args.tokens).read_text().splitlines()) if row}
    output = []
    for image_id in Path(args.ids).read_text().splitlines():
        image_id = image_id.strip()
        if not image_id:
            continue
        image_path = next((Path(args.images) / f"{image_id}{suffix}" for suffix in (".jpg", ".jpeg", ".png", ".webp") if (Path(args.images) / f"{image_id}{suffix}").exists()), None)
        if image_path is None:
            continue
        raw = image_path.read_bytes(); image_crc = zlib.crc32(raw) & 0xffffffff
        image_row = by_crc.get(image_crc)
        if image_row is None:
            output.append({"image_id": image_id, "status": "no_kaggle_image_match"}); continue
        stem = Path(image_row["name"]).stem
        label_row = labels_by_stem.get(stem)
        if label_row is None:
            output.append({"image_id": image_id, "kaggle_image": image_row["name"], "status": "no_label"}); continue
        label_text = _read_zip_member(archive_url, label_row)
        width, height = Image.open(image_path).size
        gt_boxes = []
        for line in label_text.decode("utf-8", "replace").splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            cls, cx, cy, bw, bh = map(float, parts)
            gt_boxes.append({"class": int(cls), "bbox": ((cx - bw / 2) * width, (cy - bh / 2) * height, (cx + bw / 2) * width, (cy + bh / 2) * height)})
        tokens = token_rows.get(image_id, [])
        overlaps = []
        for token in tokens:
            if token.bbox is None:
                continue
            by_class = {str(cls): max((_iou(token.bbox, box["bbox"]) for box in gt_boxes if box["class"] == cls), default=0.0) for cls in sorted({box["class"] for box in gt_boxes})}
            overlaps.append({"text": token.text, "iou": max(by_class.values(), default=0.0), "iou_by_class": by_class, "bbox": token.bbox})
        output.append({"image_id": image_id, "kaggle_image": image_row["name"], "status": "ok", "gt_box_count": len(gt_boxes), "gt_classes": [box["class"] for box in gt_boxes], "gt_boxes": gt_boxes, "ocr_token_count": len(tokens), "max_token_gt_iou": max((item["iou"] for item in overlaps), default=0.0), "max_token_date_iou": max((item["iou_by_class"].get("0", 0.0) for item in overlaps), default=0.0), "token_overlaps": overlaps})
    Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    ok = [row for row in output if row.get("status") == "ok"]
    print(json.dumps({"images": len(output), "matched": len(ok), "with_ocr_overlap": sum(row.get("max_token_gt_iou", 0) > 0.1 for row in ok)}, ensure_ascii=False))
    return 0


def _read_zip_member(url: str, row: dict) -> bytes:
    head = requests.get(url, headers={"Range": f"bytes={row['offset']}-{row['offset'] + 1023}"}, timeout=60).content
    _, _, _, _, _, _, _, _, _, name_len, extra_len = struct.unpack_from("<4s5H3L2H", head, 0)
    start = row["offset"] + 30 + name_len + extra_len
    payload = requests.get(url, headers={"Range": f"bytes={start}-{start + row['compressed_size'] - 1}"}, timeout=60).content
    return zlib.decompress(payload, -15)


def _iou(a, b):
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, right - left) * max(0, bottom - top)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return inter / max(1e-9, area_a + area_b - inter)


if __name__ == "__main__":
    raise SystemExit(main())
