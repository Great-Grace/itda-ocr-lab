"""Build annotation-based date crops for recognizer training/evaluation."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageOps

parser = argparse.ArgumentParser()
parser.add_argument("--manifest", required=True)
parser.add_argument("--labels", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

labels = {row["image_id"]: row["final_date"] for row in csv.DictReader(open(args.labels, encoding="utf-8"))}
rows = json.loads(Path(args.manifest).read_text())
out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
records = []
for row in rows:
    image_id = row["image_id"]
    label = labels.get(image_id, "")
    if not label or label.upper() == "NONE" or "None" in label:
        continue
    image_path = Path(row["image"])
    if not image_path.exists():
        continue
    date_boxes = [item["bbox"] for item in row.get("gt_boxes", []) if int(item.get("class", -1)) == 0]
    if not date_boxes:
        continue
    image = Image.open(image_path).convert("L")
    left = max(0, int(min(box[0] for box in date_boxes)) - 8)
    top = max(0, int(min(box[1] for box in date_boxes)) - 8)
    right = min(image.width, int(max(box[2] for box in date_boxes)) + 8)
    bottom = min(image.height, int(max(box[3] for box in date_boxes)) + 8)
    crop = ImageOps.pad(image.crop((left, top, right, bottom)), (320, 48), color=255, centering=(0, 0))
    filename = f"{image_id}.png"
    crop.save(out / filename)
    records.append({"image_id": image_id, "image": filename, "label": label, "raw_text": label})
with (out / "labels.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=["image_id", "image", "label", "raw_text"]); writer.writeheader(); writer.writerows(records)
print(json.dumps({"samples": len(records), "output": str(out)}, ensure_ascii=False))
