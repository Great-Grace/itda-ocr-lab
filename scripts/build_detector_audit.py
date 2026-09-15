"""Build a reviewable detector/recognition audit from cached OCR tokens."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw

from ocr_lab.contracts import OCRToken
from ocr_lab.modules.date_candidates import candidate_key, parse_final_date
from ocr_lab.modules.regex_selector import KeywordRegexSelector

parser = argparse.ArgumentParser()
parser.add_argument("--images", required=True)
parser.add_argument("--tokens", required=True)
parser.add_argument("--labels", required=True)
parser.add_argument("--ids", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

labels = {row["image_id"]: row["final_date"] for row in csv.DictReader(open(args.labels, encoding="utf-8"))}
tokens_by_id = {row["image_id"]: [OCRToken(**token) for token in row.get("tokens", [])] for row in map(json.loads, Path(args.tokens).read_text().splitlines()) if row}
selector = KeywordRegexSelector(allow_day_first=True, allow_month_names=True)
out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
rows = []
for image_id in Path(args.ids).read_text().splitlines():
    image_id = image_id.strip()
    target = parse_final_date(labels.get(image_id, ""))
    if not image_id or target is None:
        continue
    tokens = tokens_by_id.get(image_id, [])
    candidates = selector.candidates(tokens)
    if any(candidate_key(candidate) == target for candidate in candidates):
        continue
    source = next((Path(args.images) / f"{image_id}{suffix}" for suffix in (".jpg", ".jpeg", ".png", ".webp") if (Path(args.images) / f"{image_id}{suffix}").exists()), None)
    if source:
        image = Image.open(source).convert("RGB")
        draw = ImageDraw.Draw(image)
        for token in tokens:
            if token.bbox:
                left, top, right, bottom = token.bbox
                draw.rectangle((left, top, right, bottom), outline="red", width=2)
        image.save(out / f"{image_id}.jpg")
    rows.append({"image_id": image_id, "target": labels[image_id], "token_count": len(tokens), "ocr_text": " | ".join(token.text for token in tokens), "candidate_count": len(candidates)})
(out / "audit.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"missing_candidate_images": len(rows), "output": str(out)}, ensure_ascii=False))
