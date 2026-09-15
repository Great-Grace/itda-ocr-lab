"""Diagnostic union of baseline OCR candidates and detector-crop candidates."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from ocr_lab.contracts import OCRToken
from ocr_lab.modules.date_candidates import candidate_final_date
from ocr_lab.modules.regex_selector import KeywordRegexSelector


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--yolo-eval", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    labels = {Path(row["filename"]).stem: row["date"] for row in csv.DictReader(Path(args.labels).open(encoding="utf-8", newline=""))}
    tokens = {}
    for line in Path(args.tokens).open(encoding="utf-8"):
        row = json.loads(line)
        tokens[row["image_id"]] = [OCRToken(**item) for item in row.get("tokens", [])]
    selector = KeywordRegexSelector(allow_day_first=True, allow_month_names=True)
    candidates = {}
    for image_id, token_list in tokens.items():
        candidates[image_id] = []
        for candidate in selector.candidates(token_list):
            prediction = candidate_final_date(candidate)
            if prediction:
                candidates[image_id].append({"prediction": prediction, "score": float(candidate.score), "det": float(candidate.features.get("detection_confidence", 0.0)), "source": "baseline"})
    yolo = json.loads(Path(args.yolo_eval).read_text(encoding="utf-8"))
    for row in yolo["predictions"]:
        for item in row.get("candidates", []):
            if item.get("prediction"):
                candidates.setdefault(row["image_id"], []).append({"prediction": item["prediction"], "score": math.log(max(float(item.get("rec_score", 0.0)), 1e-6)), "det": float(item.get("det_conf", 0.0)), "source": "yolo"})
    rows = []
    for image_id, target in labels.items():
        pool = candidates.get(image_id, [])
        exact_exists = any(item["prediction"] == target for item in pool)
        def rank(item: dict) -> float:
            source_bonus = 1.0 if item["source"] == "baseline" else 2.0
            return source_bonus + item["score"] + 0.4 * math.log(max(item["det"], 1e-6))
        chosen = max(pool, key=rank, default=None)
        rows.append({"image_id": image_id, "target": target, "candidate_exists": exact_exists, "selection_exact": bool(chosen and chosen["prediction"] == target), "chosen": chosen, "candidate_count": len(pool)})
    denominator = max(1, len(rows)); exists = sum(row["candidate_exists"] for row in rows); selected = sum(row["selection_exact"] for row in rows)
    result = {"images": len(rows), "candidate_recall": exists / denominator, "selection_accuracy_given_candidate": selected / max(1, exists), "final_em": selected / denominator, "rows": rows}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("images", "candidate_recall", "selection_accuracy_given_candidate", "final_em")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
