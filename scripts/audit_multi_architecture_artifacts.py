"""Reproducible integrity audit for the multi-architecture benchmark artifacts."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def stem(row: dict[str, str]) -> str:
    return Path(row.get("filename", row.get("image_id", ""))).stem


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="runs/multi_architecture_leaderboard/audit_report.json")
    args = parser.parse_args()

    clean = read_csv(ROOT / "data/clean_training_set_3066.csv")
    splits = {name: read_csv(ROOT / f"data/splits/{name}.csv") for name in ("train", "val", "test")}
    split_ids = {name: {stem(row) for row in rows} for name, rows in splits.items()}
    clean_ids = {stem(row) for row in clean}
    labels = read_csv(ROOT / "runs/combined_labels_180.csv")
    benchmark_ids = {stem(row) for row in labels}
    leaderboard = json.loads((ROOT / "runs/multi_architecture_leaderboard/master_leaderboard.json").read_text(encoding="utf-8"))
    weights = json.loads((ROOT / "weights/training_summary.json").read_text(encoding="utf-8"))
    script = (ROOT / "scripts/run_multi_architecture_benchmark.py").read_text(encoding="utf-8")
    notebook = json.loads((ROOT / "predict.ipynb").read_text(encoding="utf-8"))
    notebook_source = "\n".join("".join(cell.get("source", [])) for cell in notebook.get("cells", []))

    overlaps = {
        "train_val": len(split_ids["train"] & split_ids["val"]),
        "train_test": len(split_ids["train"] & split_ids["test"]),
        "val_test": len(split_ids["val"] & split_ids["test"]),
    }
    signature_counts = Counter(
        (row["exact_match"], row["candidate_recall"], row["selection_acc"])
        for row in leaderboard
    )
    weight_checks = []
    for name, item in weights.items():
        checkpoint = ROOT / item["checkpoint"]
        actual_mb = checkpoint.stat().st_size / 1024 / 1024 if checkpoint.exists() else None
        weight_checks.append({
            "architecture": name,
            "checkpoint_exists": checkpoint.exists(),
            "reported_size_mb": item.get("size_mb"),
            "actual_size_mb": round(actual_mb, 3) if actual_mb is not None else None,
            "best_exact": item.get("best_exact"),
            "final_exact": item.get("final_exact"),
            "final_cer": item.get("final_cer"),
        })

    findings = {
        "clean_dataset": {
            "rows": len(clean),
            "unique_filenames": len(clean_ids),
            "declared_filename_count": 3066,
            "all_confidence_values": sorted(set(row.get("confidence", "") for row in clean)),
        },
        "splits": {
            "counts": {name: len(rows) for name, rows in splits.items()},
            "pairwise_overlap": overlaps,
            "union": len(set().union(*split_ids.values())),
            "unassigned_clean_ids": len(clean_ids - set().union(*split_ids.values())),
        },
        "benchmark_population": {
            "label_rows": len(labels),
            "unique_ids": len(benchmark_ids),
            "overlap_with_train": len(benchmark_ids & split_ids["train"]),
            "overlap_with_val": len(benchmark_ids & split_ids["val"]),
            "overlap_with_test": len(benchmark_ids & split_ids["test"]),
            "outside_clean_dataset": len(benchmark_ids - clean_ids),
        },
        "leaderboard": {
            "json_rows": len(leaderboard),
            "claimed_rows": 53,
            "rank_sequence_valid": [row.get("rank") for row in leaderboard] == list(range(1, len(leaderboard) + 1)),
            "largest_identical_metric_groups": [
                {"metrics": list(key), "rows": count}
                for key, count in signature_counts.most_common(5)
            ],
        },
        "weights": weight_checks,
        "static_code_flags": {
            "hard_coded_cascade_em_bonus": "em = min(1.0, em + 0.0167)" in script,
            "hard_coded_mobilenet_penalty": "em = max(0.0, em - 0.022)" in script,
            "hard_coded_attention_penalty": "em = max(0.0, em - 0.0056)" in script,
            "hard_coded_pipeline_latency": "1490.0" in script and "2100.0" in script,
            "benchmark_uses_combined_180_defaults": "combined_tokens_180.jsonl" in script,
            "predict_uses_nonexistent_extract_method": ".infer(str(img_file))" in notebook_source,
            "predict_ocr_weight_dirs_present": (
                (ROOT / "weights/paddle/ppocrv5_mobile_det").is_dir()
                and (ROOT / "weights/paddle/PP-OCRv6_medium_rec").is_dir()
            ),
        },
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(findings, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    raise SystemExit(main())
