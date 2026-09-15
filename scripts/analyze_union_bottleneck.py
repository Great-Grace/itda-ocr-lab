"""Attribute union-OCR errors to candidate generation or ranking sources."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from ocr_lab.contracts import OCRToken
from ocr_lab.modules.date_candidates import candidate_final_date
from ocr_lab.modules.regex_selector import KeywordRegexSelector


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--yolo-eval", required=True)
    parser.add_argument("--union", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    labels = {Path(row["filename"]).stem: row["date"] for row in csv.DictReader(Path(args.labels).open(encoding="utf-8", newline=""))}
    selector = KeywordRegexSelector(allow_day_first=True, allow_month_names=True)
    baseline: dict[str, list[str]] = {}
    for line in Path(args.tokens).open(encoding="utf-8"):
        row = json.loads(line)
        tokens = [OCRToken(**item) for item in row.get("tokens", [])]
        baseline[row["image_id"]] = [prediction for candidate in selector.candidates(tokens) if (prediction := candidate_final_date(candidate))]
    yolo = json.loads(Path(args.yolo_eval).read_text(encoding="utf-8"))
    yolo_candidates = {
        row["image_id"]: [item["prediction"] for item in row.get("candidates", []) if item.get("prediction")]
        for row in yolo["predictions"]
    }
    union = {row["image_id"]: row for row in json.loads(Path(args.union).read_text(encoding="utf-8"))["rows"]}

    rows = []
    for image_id, target in labels.items():
        base = baseline.get(image_id, [])
        detector = yolo_candidates.get(image_id, [])
        base_match, detector_match = target in base, target in detector
        row = union.get(image_id, {})
        if base_match and detector_match:
            source = "both_cover"
        elif base_match:
            source = "baseline_only_cover"
        elif detector_match:
            source = "detector_only_cover"
        elif not base and not detector:
            source = "no_candidate_both"
        elif not base:
            source = "baseline_empty_detector_wrong"
        elif not detector:
            source = "detector_empty_baseline_wrong"
        else:
            source = "both_wrong_candidate"
        ranking_error = bool(row.get("candidate_exists") and not row.get("selection_exact"))
        rows.append({
            "image_id": image_id,
            "target": target,
            "source_category": source,
            "ranking_error": ranking_error,
            "chosen_source": (row.get("chosen") or {}).get("source"),
            "baseline_candidates": base,
            "detector_candidates": detector,
        })
    summary = Counter(row["source_category"] for row in rows)
    output = {"images": len(rows), "source_categories": summary, "ranking_errors": sum(row["ranking_error"] for row in rows), "ranking_error_chosen_source": Counter(row["chosen_source"] for row in rows if row["ranking_error"]), "rows": rows}
    target = Path(args.output); target.parent.mkdir(parents=True, exist_ok=True); target.write_text(json.dumps(output, ensure_ascii=False, indent=2, default=dict), encoding="utf-8")
    print(json.dumps({"images": len(rows), "source_categories": summary, "ranking_errors": output["ranking_errors"], "ranking_error_chosen_source": output["ranking_error_chosen_source"]}, ensure_ascii=False, default=dict))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
