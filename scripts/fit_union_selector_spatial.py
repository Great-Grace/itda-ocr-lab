"""Fit a train-only union selector with detector/token box overlap."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from ocr_lab.contracts import OCRToken
from ocr_lab.modules.date_candidates import candidate_final_date
from ocr_lab.modules.regex_selector import KeywordRegexSelector


def iou(a: list[float] | None, b: list[float] | None) -> float:
    if not a or not b:
        return 0.0
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1]); ab = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(1e-9, aa + ab - inter)


def union_bbox(tokens: list[OCRToken], indices: tuple[int, ...]) -> list[float] | None:
    boxes = [tokens[index].bbox for index in indices if index < len(tokens) and tokens[index].bbox is not None]
    if not boxes:
        return None
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


def load_candidates(tokens_path: Path, yolo_path: Path) -> dict[str, list[dict]]:
    selector = KeywordRegexSelector(allow_day_first=True, allow_month_names=True)
    output: dict[str, list[dict]] = {}; baseline_boxes: dict[str, list[list[float]]] = {}
    for line in tokens_path.open(encoding="utf-8"):
        row = json.loads(line); tokens = [OCRToken(**item) for item in row.get("tokens", [])]; output[row["image_id"]] = []; baseline_boxes[row["image_id"]] = []
        for candidate in selector.candidates(tokens):
            prediction = candidate_final_date(candidate)
            if not prediction:
                continue
            box = union_bbox(tokens, candidate.token_indices); baseline_boxes[row["image_id"]].append(box) if box else None
            output[row["image_id"]].append({"prediction": prediction, "source": "baseline", "score": float(candidate.score), "rec": float(candidate.features.get("recognition_confidence", 0.0)), "det": float(candidate.features.get("detection_confidence", 0.0)), "class": -1, "bbox": box})
    for row in json.loads(yolo_path.read_text(encoding="utf-8"))["predictions"]:
        image_id = row["image_id"]; yolo_boxes = [item.get("bbox") for item in row.get("candidates", []) if item.get("bbox")]
        values = output.setdefault(image_id, [])
        for candidate in row.get("candidates", []):
            if candidate.get("prediction"):
                box = candidate.get("bbox")
                values.append({"prediction": candidate["prediction"], "source": "yolo", "score": math.log(max(float(candidate.get("rec_score", 0.0)), 1e-6)), "rec": float(candidate.get("rec_score", 0.0)), "det": float(candidate.get("det_conf", 0.0)), "class": int(candidate.get("class", 0)), "bbox": box, "spatial": max((iou(box, b) for b in baseline_boxes.get(image_id, [])), default=0.0)})
        for value in values:
            if value["source"] == "baseline":
                value["spatial"] = max((iou(value.get("bbox"), box) for box in yolo_boxes), default=0.0)
    return output


def labels(path: Path) -> dict[str, str]:
    return {Path(row["filename"]).stem: row["date"] for row in csv.DictReader(path.open(encoding="utf-8", newline=""))}


def evaluate(candidates: dict[str, list[dict]], targets: dict[str, str], w: dict[str, float]) -> dict[str, float]:
    found = selected = 0
    for image_id, target in targets.items():
        pool = candidates.get(image_id, []); found += any(c["prediction"] == target for c in pool)
        if not pool:
            continue
        def rank(c: dict) -> float:
            source = w["baseline_source"] if c["source"] == "baseline" else w["yolo_source"] + w["yolo_bonus"]
            return source + w["candidate_score"] * c["score"] + w["rec_log"] * math.log(max(c["rec"], 1e-6)) + w["det_log"] * math.log(max(c["det"], 1e-6)) + w["spatial"] * c.get("spatial", 0.0)
        selected += max(pool, key=rank)["prediction"] == target
    n = max(1, len(targets)); return {"images": len(targets), "candidate_recall": found / n, "final_em": selected / n, "selection_accuracy": selected / max(1, found)}


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--train-tokens", required=True); parser.add_argument("--train-yolo", required=True); parser.add_argument("--train-labels", required=True); parser.add_argument("--val-tokens", required=True); parser.add_argument("--val-yolo", required=True); parser.add_argument("--val-labels", required=True); parser.add_argument("--output", required=True); args = parser.parse_args()
    train = load_candidates(Path(args.train_tokens), Path(args.train_yolo)); train_labels = labels(Path(args.train_labels)); val = load_candidates(Path(args.val_tokens), Path(args.val_yolo)); val_labels = labels(Path(args.val_labels))
    best = None
    for spatial in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
        for yolo_bonus in (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0):
            for candidate_score in (0.5, 0.75, 1.0, 1.5, 2.0):
                for rec_log in (0.5, 0.75, 1.0, 1.5):
                    for det_log in (0.0, 0.1, 0.25, 0.5, 1.0):
                        w = {"baseline_source": 0.0, "yolo_source": 0.0, "yolo_bonus": yolo_bonus, "candidate_score": candidate_score, "rec_log": rec_log, "det_log": det_log, "spatial": spatial}; score = evaluate(train, train_labels, w)
                        key = (score["final_em"], score["selection_accuracy"], score["candidate_recall"])
                        if best is None or key > best["key"]: best = {"key": key, "weights": w, "train": score}
    assert best is not None; best["val"] = evaluate(val, val_labels, best["weights"]); best.pop("key")
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8"); print(json.dumps(best, ensure_ascii=False)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
