"""Extract weakly supervised real date crops from a labeled dev split.

Only OCR candidates whose normalized date equals the image label are kept.
The resulting crop set is for recognizer research; it must never consume the
lock split.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageOps

from ocr_lab.contracts import OCRToken
from ocr_lab.modules.date_candidates import candidate_key, parse_final_date
from ocr_lab.modules.regex_selector import KeywordRegexSelector

parser = argparse.ArgumentParser()
parser.add_argument("--images", required=True)
parser.add_argument("--tokens", required=True)
parser.add_argument("--labels", help="CSV with image_id/final_date or filename/date columns")
parser.add_argument("--ids", help="Optional ID list; defaults to every label row")
parser.add_argument("--output", required=True)
args = parser.parse_args()

image_root = Path(args.images)
if not args.labels:
    raise SystemExit("--labels is required")
label_rows = list(csv.DictReader(open(args.labels, encoding="utf-8")))
labels = {
    Path(row.get("filename", row.get("image_id", ""))).stem: row.get("date", row.get("final_date", ""))
    for row in label_rows
    if row.get("filename") or row.get("image_id")
}
tokens = {}
with open(args.tokens, encoding="utf-8") as handle:
    for line in handle:
        row = json.loads(line)
        tokens[row["image_id"]] = [OCRToken(**item) for item in row.get("tokens", [])]

selector = KeywordRegexSelector(allow_day_first=True, allow_month_names=True)
output = Path(args.output)
output.mkdir(parents=True, exist_ok=True)
rows = []
ids = Path(args.ids).read_text().splitlines() if args.ids else sorted(labels)
for image_id in ids:
    image_id = image_id.strip()
    if not image_id or image_id not in labels or parse_final_date(labels[image_id]) is None:
        continue
    target = parse_final_date(labels[image_id])
    candidates = selector.candidates(tokens.get(image_id, []))
    matches = [candidate for candidate in candidates if candidate_key(candidate) == target and candidate.token_indices]
    if not matches:
        continue
    candidate = matches[0]
    source = Image.open(next((image_root / f"{image_id}{suffix}" for suffix in (".jpg", ".jpeg", ".png", ".webp") if (image_root / f"{image_id}{suffix}").exists()), image_root / f"{image_id}.jpg")).convert("L")
    boxes = [tokens[image_id][index].bbox for index in candidate.token_indices if index < len(tokens[image_id]) and tokens[image_id][index].bbox]
    if not boxes:
        continue
    left = max(0, int(min(box[0] for box in boxes)) - 6)
    top = max(0, int(min(box[1] for box in boxes)) - 6)
    right = min(source.width, int(max(box[2] for box in boxes)) + 6)
    bottom = min(source.height, int(max(box[3] for box in boxes)) + 6)
    crop = source.crop((left, top, right, bottom))
    crop = ImageOps.pad(crop, (320, 48), color=255, centering=(0, 0))
    filename = f"{image_id}.png"
    crop.save(output / filename)
    rows.append({"image_id": image_id, "image": filename, "label": labels[image_id], "raw_text": candidate.raw_text})

with (output / "labels.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=["image_id", "image", "label", "raw_text"])
    writer.writeheader(); writer.writerows(rows)
print(json.dumps({"samples": len(rows), "output": str(output)}, ensure_ascii=False))
