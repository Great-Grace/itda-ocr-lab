"""Evaluate a trained YOLO region detector followed by PP-OCRv6 recognition."""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
from datetime import date
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def normalize(text: str) -> str | None:
    text = str(text or "").replace(" ", "")
    months = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
    month_pattern = r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
    for y, month, d in re.findall(rf"(?<!\d)(\d{{4}})[^A-Za-z0-9]*(\w+)[^A-Za-z0-9]*(\d{{1,2}})(?!\d)", text, re.I):
        key = month[:3].lower()
        if key in months:
            try:
                date(int(y), months[key], int(d)); return f"{int(y):04d}-{months[key]:02d}-{int(d):02d}"
            except ValueError:
                pass
    for d, month, y in re.findall(rf"(?<!\d)(\d{{1,2}})[^A-Za-z0-9]*({month_pattern})[^A-Za-z0-9]*(\d{{4}})(?!\d)", text, re.I):
        key = month[:3].lower()
        try:
            date(int(y), months[key], int(d)); return f"{int(y):04d}-{months[key]:02d}-{int(d):02d}"
        except ValueError:
            pass
    korean = re.search(r"(?<!\d)(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일?", text)
    if korean:
        y, m, d = map(int, korean.groups())
        try:
            date(y, m, d); return f"{y:04d}-{m:02d}-{d:02d}"
        except ValueError:
            pass
    m = re.search(r"(\d{4})[^\d]?(\d{1,2})[^\d]?(\d{1,2})", text)
    if m:
        y, mo, d = map(int, m.groups())
        try:
            date(y, mo, d); return f"{y:04d}-{mo:02d}-{d:02d}"
        except ValueError:
            pass
    m = re.search(r"(\d{2})[^\d]?(\d{1,2})[^\d]?(\d{1,2})", text)
    if m:
        y, mo, d = 2000 + int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            date(y, mo, d); return f"{y:04d}-{mo:02d}-{d:02d}"
        except ValueError:
            pass
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--rec-model-dir", required=True)
    parser.add_argument("--rec-model-name", default="PP-OCRv6_medium_rec")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--rec-device", default="gpu:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--expand", type=float, default=1.0, help="expand detector crop around its center")
    parser.add_argument("--classes", default="0,1,3", help="detector classes to send to recognition")
    args = parser.parse_args()
    from ultralytics import YOLO
    from paddleocr import TextRecognition

    labels = {Path(row["filename"]).stem: row["date"] for row in csv.DictReader(Path(args.labels).open(encoding="utf-8", newline=""))}
    ids = [Path(row["filename"]).stem for row in csv.DictReader(Path(args.labels).open(encoding="utf-8", newline=""))]
    if args.limit:
        ids = ids[: args.limit]
    detector = YOLO(args.weights)
    recognizer = TextRecognition(model_name=args.rec_model_name, model_dir=args.rec_model_dir, device=args.rec_device)
    allowed_classes = {int(value) for value in args.classes.split(",") if value.strip()}
    image_paths = [next((Path(args.images) / f"{image_id}{s}" for s in (".jpg", ".jpeg", ".png", ".webp") if (Path(args.images) / f"{image_id}{s}").exists()), None) for image_id in ids]
    image_paths = [(image_id, path) for image_id, path in zip(ids, image_paths) if path is not None]
    started = time.perf_counter(); candidates_by_id = {image_id: [] for image_id, _ in image_paths}
    for image_id, path in image_paths:
        image = Image.open(path).convert("RGB")
        result = detector.predict(str(path), imgsz=960, conf=0.20, device=args.device, verbose=False)[0]
        crops, metas = [], []
        for box, cls, conf in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.cls.cpu().tolist(), result.boxes.conf.cpu().tolist()):
            if int(cls) not in allowed_classes:
                continue
            raw_left, raw_top, raw_right, raw_bottom = map(float, box)
            center_x, center_y = (raw_left + raw_right) / 2.0, (raw_top + raw_bottom) / 2.0
            width, height = (raw_right - raw_left) * args.expand, (raw_bottom - raw_top) * args.expand
            left = max(0, int(center_x - width / 2.0) - 6)
            top = max(0, int(center_y - height / 2.0) - 6)
            right = min(image.width, int(center_x + width / 2.0) + 6)
            bottom = min(image.height, int(center_y + height / 2.0) + 6)
            if right <= left or bottom <= top:
                continue
            crop = ImageOps.pad(image.crop((left, top, right, bottom)).convert("L"), (320, 48), color=255, centering=(0, 0))
            crops.append(np.asarray(crop.convert("RGB"))); metas.append({"class": int(cls), "det_conf": float(conf), "bbox": [left, top, right, bottom]})
        if not crops:
            continue
        for meta, rec in zip(metas, recognizer.predict(crops, batch_size=min(args.batch_size, len(crops)))):
            payload = rec if isinstance(rec, dict) else {}
            raw = payload.get("rec_text", "")
            score = float(payload.get("rec_score", 0.0) or 0.0)
            candidates_by_id[image_id].append({"raw": raw, "prediction": normalize(raw), "rec_score": score, **meta})
    predictions = []
    for image_id, candidates in candidates_by_id.items():
        target = normalize(labels.get(image_id, ""))
        exact_candidates = [item for item in candidates if item["prediction"] == target and target is not None]
        chosen = max(candidates, key=lambda item: (item["class"] == 1, item["rec_score"] * item["det_conf"]), default=None)
        predictions.append({"image_id": image_id, "target": target, "candidate_recall": bool(exact_candidates), "selection_exact": bool(chosen and chosen["prediction"] == target), "chosen": chosen, "candidates": candidates})
    elapsed = time.perf_counter() - started
    valid = [row for row in predictions if row["target"] is not None]
    result = {"images": len(predictions), "candidate_recall": sum(row["candidate_recall"] for row in valid) / max(1, len(valid)), "selection_accuracy": sum(row["selection_exact"] for row in valid) / max(1, len(valid)), "final_em": sum(row["selection_exact"] for row in valid) / max(1, len(valid)), "seconds": elapsed, "sec_per_image": elapsed / max(1, len(predictions)), "predictions": predictions}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("images", "candidate_recall", "selection_accuracy", "final_em", "seconds", "sec_per_image")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
