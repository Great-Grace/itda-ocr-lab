"""Fit a small, transparent source-aware selector on train candidates."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from ocr_lab.contracts import OCRToken
from ocr_lab.modules.date_candidates import candidate_final_date
from ocr_lab.modules.regex_selector import KeywordRegexSelector


def load_candidates(tokens_path: Path, yolo_path: Path) -> dict[str, list[dict]]:
    selector = KeywordRegexSelector(allow_day_first=True, allow_month_names=True)
    output: dict[str, list[dict]] = {}
    for line in tokens_path.open(encoding="utf-8"):
        row = json.loads(line)
        tokens = [OCRToken(**item) for item in row.get("tokens", [])]
        output[row["image_id"]] = []
        for candidate in selector.candidates(tokens):
            prediction = candidate_final_date(candidate)
            if prediction:
                output[row["image_id"]].append({"prediction": prediction, "source": "baseline", "score": float(candidate.score), "rec": float(candidate.features.get("recognition_confidence", 0.0)), "det": float(candidate.features.get("detection_confidence", 0.0)), "class": -1})
    for row in json.loads(yolo_path.read_text(encoding="utf-8"))["predictions"]:
        values = output.setdefault(row["image_id"], [])
        for candidate in row.get("candidates", []):
            if candidate.get("prediction"):
                values.append({"prediction": candidate["prediction"], "source": "yolo", "score": math.log(max(float(candidate.get("rec_score", 0.0)), 1e-6)), "rec": float(candidate.get("rec_score", 0.0)), "det": float(candidate.get("det_conf", 0.0)), "class": int(candidate.get("class", 0))})
    return output


def load_labels(path: Path) -> dict[str, str]:
    return {Path(row["filename"]).stem: row["date"] for row in csv.DictReader(path.open(encoding="utf-8", newline=""))}


def evaluate(candidates: dict[str, list[dict]], labels: dict[str, str], weights: dict[str, float]) -> dict[str, float]:
    rows = []
    for image_id, target in labels.items():
        pool = candidates.get(image_id, [])
        if pool:
            def rank(item: dict) -> float:
                return (weights["baseline_source"] if item["source"] == "baseline" else weights["yolo_source"] + weights["yolo_bonus"]) + weights["candidate_score"] * item["score"] + weights["rec_log"] * math.log(max(item["rec"], 1e-6)) + weights["det_log"] * math.log(max(item["det"], 1e-6)) + weights["due_bonus"] * (item["class"] == 1)
            chosen = max(pool, key=rank)
            selected = chosen["prediction"] == target
        else:
            selected = False
        exists = any(item["prediction"] == target for item in pool)
        rows.append((exists, selected))
    exists = sum(item[0] for item in rows); selected = sum(item[1] for item in rows)
    return {"candidate_recall": exists / max(1, len(rows)), "final_em": selected / max(1, len(rows)), "selection_accuracy": selected / max(1, exists), "images": len(rows)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-tokens", required=True); parser.add_argument("--train-yolo", required=True); parser.add_argument("--train-labels", required=True)
    parser.add_argument("--val-tokens", required=True); parser.add_argument("--val-yolo", required=True); parser.add_argument("--val-labels", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    train = load_candidates(Path(args.train_tokens), Path(args.train_yolo)); train_labels = load_labels(Path(args.train_labels))
    val = load_candidates(Path(args.val_tokens), Path(args.val_yolo)); val_labels = load_labels(Path(args.val_labels))
    best = None
    for baseline_source in (0.0, 0.25, 0.5, 1.0, 1.5):
        for yolo_source in (0.0, 0.25, 0.5, 1.0, 1.5):
            for yolo_bonus in (0.0, 0.25, 0.5, 1.0, 2.0):
                for candidate_score in (0.5, 1.0, 2.0):
                    for rec_log in (0.0, 0.25, 0.5, 1.0):
                        for det_log in (0.0, 0.1, 0.25, 0.5):
                            for due_bonus in (0.0, 0.1, 0.25):
                                weights = {"baseline_source": baseline_source, "yolo_source": yolo_source, "yolo_bonus": yolo_bonus, "candidate_score": candidate_score, "rec_log": rec_log, "det_log": det_log, "due_bonus": due_bonus}
                                metrics = evaluate(train, train_labels, weights)
                                key = (metrics["final_em"], metrics["selection_accuracy"], metrics["candidate_recall"])
                                if best is None or key > best["key"]:
                                    best = {"key": key, "weights": weights, "train": metrics}
    assert best is not None
    best["val"] = evaluate(val, val_labels, best["weights"])
    best.pop("key")
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(best, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
