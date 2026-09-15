"""Evaluate an offline PaddleOCR recognizer on pre-cropped date images."""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path


def normalize(text: str) -> str | None:
    text = str(text or "").replace(" ", "")
    m = re.search(r"(\d{4})[^\d]?(\d{1,2})[^\d]?(\d{1,2})", text)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    m = re.search(r"(\d{2})[^\d]?(\d{1,2})[^\d]?(\d{1,2})", text)
    if m:
        return f"20{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--crops", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="gpu:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--model-name", default="PP-OCRv6_medium_rec")
    args = parser.parse_args()
    from paddleocr import TextRecognition

    root = Path(args.crops)
    rows = list(csv.DictReader((root / "labels.csv").open(encoding="utf-8", newline="")))
    model = TextRecognition(model_name=args.model_name, model_dir=args.model_dir, device=args.device)
    paths = [str(root / row["image"]) for row in rows]
    started = time.perf_counter()
    results = list(model.predict(paths, batch_size=args.batch_size))
    predictions = []
    for row, result in zip(rows, results):
        payload = result if isinstance(result, dict) else getattr(result, "json", lambda: {})()
        text = payload.get("rec_text", "") if isinstance(payload, dict) else ""
        score = float(payload.get("rec_score", 0.0) or 0.0) if isinstance(payload, dict) else 0.0
        target = normalize(row["label"])
        prediction = normalize(text)
        predictions.append({"image_id": row["image_id"], "target": target, "prediction": prediction, "raw": text, "score": score, "exact": prediction == target})
    elapsed = time.perf_counter() - started
    exact = sum(item["exact"] for item in predictions) / max(1, len(predictions))
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"samples": len(predictions), "exact": exact, "seconds": elapsed, "sec_per_image": elapsed / max(1, len(predictions)), "predictions": predictions}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"samples": len(predictions), "exact": exact, "seconds": elapsed, "sec_per_image": elapsed / max(1, len(predictions))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
