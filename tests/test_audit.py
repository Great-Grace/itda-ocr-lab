import csv
import json
from pathlib import Path

from scripts.audit_ai_labels import audit_labels


def test_audit_labels_flags_invalid_calendar_and_outliers(tmp_path: Path) -> None:
    labels_csv = tmp_path / "test_labels.csv"
    gt_csv = tmp_path / "gt_labels.csv"
    tokens_cache = tmp_path / "tokens.jsonl"
    out_dir = tmp_path / "audit_out"

    labels_data = [
        {"image_id": "clean_1", "year": "2024", "month": "05", "day": "20", "final_date": "2024-05-20"},
        {"image_id": "bad_cal", "year": "2024", "month": "02", "day": "30", "final_date": "2024-02-30"},
        {"image_id": "outlier_yr", "year": "1995", "month": "01", "day": "01", "final_date": "1995-01-01"},
        {"image_id": "mismatch_gt", "year": "2022", "month": "10", "day": "15", "final_date": "2022-10-15"},
        {"image_id": "hallucination", "year": "2023", "month": "11", "day": "11", "final_date": "2023-11-11"},
    ]
    with open(labels_csv, "w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["image_id", "year", "month", "day", "final_date"])
        writer.writeheader()
        writer.writerows(labels_data)

    gt_data = [
        {"image_id": "clean_1", "final_date": "2024-05-20"},
        {"image_id": "mismatch_gt", "final_date": "2022-10-20"},
    ]
    with open(gt_csv, "w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["image_id", "final_date"])
        writer.writeheader()
        writer.writerows(gt_data)

    tokens = [
        {"image_id": "clean_1", "tokens": [{"text": "2024.05.20"}]},
        {"image_id": "bad_cal", "tokens": [{"text": "2024.02.30"}]},
        {"image_id": "outlier_yr", "tokens": [{"text": "1995.01.01"}]},
        {"image_id": "mismatch_gt", "tokens": [{"text": "2022.10.15"}]},
        {"image_id": "hallucination", "tokens": [{"text": "COMPLETELY_UNRELATED_TEXT"}]},
    ]
    with open(tokens_cache, "w", encoding="utf-8") as fp:
        for t in tokens:
            fp.write(json.dumps(t) + "\n")

    res = audit_labels(
        labels_path=labels_csv,
        gt_path=gt_csv,
        tokens_cache_paths=[tokens_cache],
        output_dir=out_dir,
        min_year=2018,
        max_year=2035,
    )

    assert res["total_images"] == 5
    assert res["clean_count"] == 1  # Only clean_1 passes all checks
    assert res["flagged_count"] == 4

    flagged_reasons = {}
    with open(out_dir / "flagged_suspicious.csv", encoding="utf-8") as fp:
        for row in csv.DictReader(fp):
            flagged_reasons[row["image_id"]] = row["flags"]

    assert "CALENDAR_INVALID" in flagged_reasons["bad_cal"]
    assert "YEAR_OUTLIER" in flagged_reasons["outlier_yr"]
    assert "GT_MISMATCH" in flagged_reasons["mismatch_gt"]
    assert "OCR_HALLUCINATION_OR_MISSING" in flagged_reasons["hallucination"]

    assert (out_dir / "verified_clean.csv").exists()
    assert (out_dir / "audit_summary.md").exists()
