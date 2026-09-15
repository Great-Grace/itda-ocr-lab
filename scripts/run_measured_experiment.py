#!/usr/bin/env python3
"""Fail-closed runner for auditable OCR experiments.

An experiment is eligible for the measured leaderboard only when this runner
observes an end-to-end OCR invocation, complete input coverage, a validated
submission, and the required raw artifacts. Cached selector-only runs remain
useful diagnostics but are explicitly marked ineligible.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SPLIT_CSVS = {tier: ROOT / f"data/splits/{tier}.csv" for tier in ("train", "val", "test")}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_lines(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def load_ids(path: Path) -> list[str]:
    ids = [line.strip().rsplit(".", 1)[0] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f"ID file must contain non-empty unique IDs: {path}")
    return ids


def input_index(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Input image directory does not exist: {root}")
    indexed = {path.stem: path for path in root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES}
    if not indexed:
        raise ValueError(f"No supported image files found in: {root}")
    return indexed


def label_ids(path: Path) -> set[str]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "image_id" not in rows[0] or "final_date" not in rows[0]:
        raise ValueError(f"Labels need image_id and final_date columns: {path}")
    return {str(row["image_id"]).strip() for row in rows if str(row.get("image_id", "")).strip()}


def split_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    rows = read_csv(path)
    if not rows or "filename" not in rows[0] or "date" not in rows[0]:
        raise ValueError(f"Split needs filename and date columns: {path}")
    ids = [Path(row["filename"]).stem for row in rows]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f"Split has missing or duplicate filenames: {path}")
    return ids, rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in ("paddle", "paddleocr", "numpy", "PIL"):
        try:
            module = __import__(package)
            versions[package] = str(getattr(module, "__version__", None))
        except Exception:
            versions[package] = None
    return versions


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an evidence-complete OCR experiment.")
    parser.add_argument("--name", required=True, help="Stable lowercase experiment name.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True, help="Directory containing the images to evaluate.")
    parser.add_argument("--tier", choices=("smoke", "val", "test"), default="val")
    parser.add_argument("--split-csv", help="Defaults to data/splits/<tier>.csv for val/test.")
    parser.add_argument("--labels", help="image_id/final_date CSV for smoke runs only.")
    parser.add_argument("--label-provenance", choices=("human", "gemini_filtered", "weak", "none"), required=True)
    parser.add_argument("--ids-file", help="Required for smoke runs.")
    parser.add_argument("--output-root", default="runs/measured")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--tokens-cache", help="Marks the run as cached diagnostic-only, never leaderboard eligible.")
    parser.add_argument("--unlock-test", action="store_true", help="Required to evaluate the frozen test tier.")
    parser.add_argument("--reason", help="Required audit note when unlocking test.")
    args = parser.parse_args()

    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", args.name):
        raise SystemExit("--name must contain lowercase letters, numbers, '_' or '-'.")
    if args.threads != 4:
        raise SystemExit("Measured CPU runs are fixed to 4 threads for comparability.")
    if args.tier == "test" and (not args.unlock_test or not args.reason):
        raise SystemExit("Test evaluation requires --unlock-test and a non-empty --reason.")
    if args.label_provenance == "none" and args.labels:
        raise SystemExit("--labels cannot be supplied when --label-provenance is none.")
    if args.tier == "smoke" and not args.split_csv and args.label_provenance != "none" and not args.labels:
        raise SystemExit("A labels file is required for a labeled smoke run.")

    config = (ROOT / args.config).resolve()
    input_root = Path(args.input).resolve()
    input_labels = Path(args.labels).resolve() if args.labels else None
    split_csv = Path(args.split_csv).resolve() if args.split_csv else SPLIT_CSVS.get(args.tier)
    ids_file = Path(args.ids_file).resolve() if args.ids_file else None
    if not config.is_file() or (input_labels is not None and not input_labels.is_file()):
        raise SystemExit("Config and any supplied labels file must exist before running an experiment.")
    if args.tier == "smoke" and not args.split_csv:
        if ids_file is None or not ids_file.is_file():
            raise SystemExit("Smoke runs require --ids-file.")
        ids, source_split_rows = load_ids(ids_file), None
    else:
        if split_csv is None or not split_csv.is_file():
            raise SystemExit("Val/test runs require an existing split CSV.")
        ids, source_split_rows = split_rows(split_csv)
    indexed = input_index(input_root)
    missing_images = sorted(set(ids) - set(indexed))
    missing_labels = sorted(set(ids) - label_ids(input_labels)) if input_labels else []
    if missing_images or missing_labels:
        raise SystemExit(json.dumps({"missing_images": missing_images[:20], "missing_labels": missing_labels[:20]}, ensure_ascii=False))

    output = (ROOT / args.output_root / f"{args.name}_{args.tier}").resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Refusing to overwrite a measured run: {output}")
    output.mkdir(parents=True, exist_ok=True)
    run_ids_file = ids_file
    if source_split_rows is not None:
        run_ids_file = output / "ids.normalized.txt"
        run_ids_file.write_text("\n".join(ids) + "\n", encoding="utf-8")
    labels = output / "labels.normalized.csv" if source_split_rows is not None else input_labels
    if source_split_rows is not None:
        with labels.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("image_id", "final_date"))
            writer.writeheader()
            writer.writerows({"image_id": Path(row["filename"]).stem, "final_date": row["date"]} for row in source_split_rows)
    evidence_path = output / "evidence.json"
    evidence: dict[str, Any] = {
        "schema_version": 1,
        "status": "started",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "name": args.name,
        "tier": args.tier,
        "test_release_reason": args.reason if args.tier == "test" else None,
        "full_ocr": not bool(args.tokens_cache),
        "label_provenance": args.label_provenance,
        "rank_eligible": args.tier == "val" and args.label_provenance in {"human", "gemini_filtered"} and not bool(args.tokens_cache),
        "ranking_scope": "internal_noisy_label" if args.label_provenance == "gemini_filtered" else ("human_labeled" if args.label_provenance == "human" else "diagnostic_only"),
        "ranking_disallowed_reason": None if (args.tier == "val" and args.label_provenance in {"human", "gemini_filtered"} and not args.tokens_cache) else "requires fresh OCR on val with accepted label provenance",
        "config": str(config.relative_to(ROOT)),
        "config_sha256": sha256_file(config),
        "input_dir": str(input_root),
        "ids_file": str(run_ids_file.relative_to(ROOT)) if run_ids_file and run_ids_file.is_relative_to(ROOT) else (str(run_ids_file) if run_ids_file else None),
        "split_csv": str(split_csv.relative_to(ROOT)) if split_csv and split_csv.is_relative_to(ROOT) else (str(split_csv) if split_csv else None),
        "split_sha256": sha256_file(split_csv) if split_csv else None,
        "ids_sha256": sha256_lines(ids),
        "image_count": len(ids),
        "labels": (str(labels.relative_to(ROOT)) if labels and labels.is_relative_to(ROOT) else str(labels)) if labels else None,
        "labels_sha256": sha256_file(labels) if labels else None,
        "runtime": {"device": "cpu", "threads": args.threads, "python": sys.version, "platform": platform.platform(), "packages": package_versions()},
    }
    evidence_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")

    command = [sys.executable, str(ROOT / "scripts/run_experiment.py"), "--config", str(config), "--input", str(input_root),
               "--output", str(output), "--device", "cpu", "--threads", "4"]
    if run_ids_file:
        command.extend(["--image-ids-file", str(run_ids_file)])
    if labels:
        command.extend(["--labels", str(labels)])
    if args.tokens_cache:
        cache = Path(args.tokens_cache).resolve()
        if not cache.is_file():
            raise SystemExit(f"Token cache not found: {cache}")
        command.extend(["--tokens-cache", str(cache)])
        evidence["tokens_cache"] = str(cache)
        evidence["tokens_cache_sha256"] = sha256_file(cache)

    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    (output / "runner.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (output / "runner.stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode:
        evidence.update({"status": "failed", "return_code": completed.returncode})
        evidence_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
        raise SystemExit(f"Experiment failed; evidence written to {evidence_path}")

    required = ("predictions.csv", "ocr_tokens.jsonl", "metrics.json", "run_manifest.json")
    absent = [name for name in required if not (output / name).is_file()]
    if absent:
        evidence.update({"status": "failed", "missing_artifacts": absent})
        evidence_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
        raise SystemExit(f"Missing required artifacts: {absent}")
    submission = subprocess.run([sys.executable, str(ROOT / "scripts/check_submission.py"), str(output / "predictions.csv")], cwd=ROOT, text=True, capture_output=True)
    (output / "submission_check.log").write_text(submission.stdout + submission.stderr, encoding="utf-8")
    if submission.returncode:
        evidence.update({"status": "failed", "submission_check": "failed"})
        evidence_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
        raise SystemExit(f"Submission validation failed; evidence written to {evidence_path}")

    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    if bool(manifest.get("tokens_cache_used")) != bool(args.tokens_cache):
        evidence.update({"status": "failed", "cache_provenance_mismatch": True})
        evidence_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
        raise SystemExit("Cache provenance mismatch.")
    evidence.update({
        "status": "complete",
        "return_code": 0,
        "submission_check": "passed",
        "metrics_sha256": sha256_file(output / "metrics.json"),
        "predictions_sha256": sha256_file(output / "predictions.csv"),
        "tokens_sha256": sha256_file(output / "ocr_tokens.jsonl"),
        "metrics": {key: metrics.get(key) for key in ("final_date_exact_match", "candidate_recall", "candidate_selection_accuracy", "sec_per_image", "peak_ram_mb", "model_weight_mb")},
    })
    evidence_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "rank_eligible": evidence["rank_eligible"], "metrics": evidence["metrics"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
