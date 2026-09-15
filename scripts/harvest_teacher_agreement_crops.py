"""Create clean date-crop/transcript pairs from Kaggle boxes and PP-OCRv6.

An image-level date label is not blindly assigned to every annotated box.
Only a date/due crop whose frozen PP-OCRv6 output normalizes exactly to the
image-level label is retained.  This produces high-precision recognizer data
and exposes crop-label mismatch rates explicitly.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


FULL_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def normalized_options(text: str) -> set[str]:
    compact = str(text or "").replace(" ", "")
    values: set[str] = set()
    for y, m, d in re.findall(r"(?<!\d)(\d{4})[^\d]?(\d{1,2})[^\d]?(\d{1,2})(?!\d)", compact):
        values.add(f"{y}-{int(m):02d}-{int(d):02d}")
    for d, m, y in re.findall(r"(?<!\d)(\d{1,2})[^\d]?(\d{1,2})[^\d]?(\d{4})(?!\d)", compact):
        values.add(f"{y}-{int(m):02d}-{int(d):02d}")
    for y, m, d in re.findall(r"(?<!\d)(\d{2})[^\d]?(\d{1,2})[^\d]?(\d{1,2})(?!\d)", compact):
        values.add(f"20{y}-{int(m):02d}-{int(d):02d}")
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--device", default="gpu:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--margin", type=int, default=6)
    args = parser.parse_args()
    from paddleocr import TextRecognition

    labels = {Path(row["filename"]).stem: row["date"] for row in csv.DictReader(Path(args.labels).open(encoding="utf-8", newline=""))}
    rows = json.loads(Path(args.manifest).read_text(encoding="utf-8"))["records"]
    crops, metadata = [], []
    source_counts: Counter[str] = Counter()
    for row in rows:
        image_id, target = row["image_id"], labels.get(row["image_id"], "")
        if row.get("status") != "ok" or not FULL_DATE.fullmatch(target):
            continue
        image = Image.open(row["image"]).convert("RGB")
        for box_index, box in enumerate(row.get("boxes", [])):
            if int(box["class"]) not in (0, 1):
                continue
            left = max(0, int((box["cx"] - box["width"] / 2) * image.width) - args.margin)
            top = max(0, int((box["cy"] - box["height"] / 2) * image.height) - args.margin)
            right = min(image.width, int((box["cx"] + box["width"] / 2) * image.width) + args.margin)
            bottom = min(image.height, int((box["cy"] + box["height"] / 2) * image.height) + args.margin)
            if right <= left or bottom <= top:
                continue
            crop = ImageOps.pad(image.crop((left, top, right, bottom)).convert("L"), (320, 48), color=255, centering=(0, 0))
            crops.append(np.asarray(crop.convert("RGB")))
            metadata.append({"image_id": image_id, "target": target, "source_class": box["name"], "box_index": box_index})
            source_counts[box["name"]] += 1

    recognizer = TextRecognition(model_name="PP-OCRv6_medium_rec", model_dir=args.model_dir, device=args.device)
    best_by_image: dict[str, dict] = {}
    for start in range(0, len(crops), args.batch_size):
        batch = crops[start : start + args.batch_size]
        outputs = recognizer.predict(batch, batch_size=len(batch))
        for meta, output, crop in zip(metadata[start : start + args.batch_size], outputs, batch):
            payload = output if isinstance(output, dict) else {}
            text = str(payload.get("rec_text", ""))
            score = float(payload.get("rec_score", 0.0) or 0.0)
            if meta["target"] not in normalized_options(text):
                continue
            value = {**meta, "teacher_text": text, "teacher_score": score, "crop": crop}
            existing = best_by_image.get(meta["image_id"])
            priority = (score, meta["source_class"] == "due")
            if existing is None or priority > (existing["teacher_score"], existing["source_class"] == "due"):
                best_by_image[meta["image_id"]] = value
        if (start + len(batch)) % 500 < args.batch_size:
            print(f"recognized {min(start + len(batch), len(crops))}/{len(crops)}", flush=True)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for image_id, item in sorted(best_by_image.items()):
        filename = f"{image_id}.png"
        Image.fromarray(item.pop("crop")).save(output / filename)
        records.append({"image_id": image_id, "image": filename, "label": item["target"], "source_class": item["source_class"], "teacher_text": item["teacher_text"], "teacher_score": f"{item['teacher_score']:.6f}"})
    with (output / "labels.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "image", "label", "source_class", "teacher_text", "teacher_score"])
        writer.writeheader(); writer.writerows(records)
    eligible_images = sum(1 for row in rows if row.get("image_id") in labels and FULL_DATE.fullmatch(labels[row["image_id"]]))
    summary = {"manifest_images": len(rows), "candidate_crops": len(crops), "candidate_crop_classes": source_counts, "teacher_agreement_images": len(records), "agreement_rate_per_image": len(records) / max(1, eligible_images)}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=dict), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, default=dict))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
