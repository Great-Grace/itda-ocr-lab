#!/usr/bin/env python3
"""Measure selector accuracy only on images with competing valid candidates."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ocr_lab.modules.date_candidates import parse_final_date


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    labels = {row["image_id"]: row["final_date"] for row in csv.DictReader(Path(args.labels).open(encoding="utf-8"))}
    hard_rows = []
    for row in csv.DictReader(Path(args.review).open(encoding="utf-8")):
        target = parse_final_date(labels.get(row["image_id"], ""))
        candidates = [parse_final_date(value) for value in row.get("candidate_set", "").split(" | ") if value]
        if target is not None and target in candidates and int(row.get("candidate_count", 0)) >= 2:
            hard_rows.append({
                "image_id": row["image_id"],
                "target": labels[row["image_id"]],
                "prediction": row["final_date"],
                "correct": parse_final_date(row["final_date"]) == target,
                "candidate_set": row["candidate_set"],
            })
    correct = sum(row["correct"] for row in hard_rows)
    result = {
        "definition": "at least two generated date candidates and the correct date present",
        "image_count": len(hard_rows),
        "correct_count": correct,
        "selection_accuracy": correct / len(hard_rows) if hard_rows else None,
        "failures": [row for row in hard_rows if not row["correct"]],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
