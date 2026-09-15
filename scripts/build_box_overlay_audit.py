#!/usr/bin/env python3
"""Render OCR polygons/bboxes and predictions for visual box diagnosis."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True)
    p.add_argument("--tokens", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--review", required=True)
    p.add_argument("--ids", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    image_root, out = Path(args.images), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    labels = {r["image_id"]: r["final_date"] for r in csv.DictReader(open(args.labels, encoding="utf-8"))}
    review = {r["image_id"]: r for r in csv.DictReader(open(args.review, encoding="utf-8"))}
    tokens = {}
    with open(args.tokens, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            tokens[row["image_id"]] = row.get("tokens", [])
    manifest = []
    font = ImageFont.load_default()
    for image_id in Path(args.ids).read_text(encoding="utf-8").splitlines():
        image_id = image_id.strip()
        source = next((image_root / f"{image_id}{s}" for s in (".jpg", ".jpeg", ".png", ".webp") if (image_root / f"{image_id}{s}").exists()), None)
        if not source:
            continue
        image = Image.open(source).convert("RGB")
        draw = ImageDraw.Draw(image)
        for index, token in enumerate(tokens.get(image_id, [])):
            bbox = token.get("bbox")
            if not bbox:
                continue
            left, top, right, bottom = map(float, bbox)
            score = float(token.get("confidence", 0.0) or 0.0)
            det = token.get("detection_confidence")
            color = "lime" if score >= 0.8 else ("yellow" if score >= 0.5 else "red")
            draw.rectangle((left, top, right, bottom), outline=color, width=max(2, int(min(image.size) / 500)))
            text = str(token.get("text", ""))[:24]
            draw.text((left, max(0, top - 14)), f"{index}:{text} {score:.2f}", fill=color, font=font)
        r = review.get(image_id, {})
        header = f"{image_id}  GT={labels.get(image_id, 'NONE')}  PRED={r.get('final_date','')}  ERR={r.get('error_category','')}"
        draw.rectangle((0, 0, image.width, 22), fill=(0, 0, 0))
        draw.text((4, 4), header[:180], fill="white", font=font)
        image.save(out / f"{image_id}.jpg", quality=90)
        manifest.append({"image_id": image_id, "gt": labels.get(image_id, "NONE"), **r, "overlay": f"{image_id}.jpg"})
    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        fields = sorted({key for row in manifest for key in row})
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(manifest)
    print(json.dumps({"images": len(manifest), "output": str(out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
