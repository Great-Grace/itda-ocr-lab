"""Run AMSC-OCR benchmark and hard-sample profiling on full 459 validation images in Colab."""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path("/content/itda_lab")
sys.path.insert(0, str(ROOT / "src"))

from ocr_lab.contracts import OCRToken
from ocr_lab.modules.amsc_cascade_ocr import AMSC_CascadeOCRBackend
from ocr_lab.modules.date_normalizer import DateNormalizer
from ocr_lab.modules.union_spatial_selector import UnionSpatialSelector

DRIVE_DIR = Path("/content/drive/MyDrive/상품사진입니다")
VAL_CSV = ROOT / "data/splits/val.csv"
OUTPUT_CSV = Path("/content/amsc_val459_predictions.csv")
REPORT_JSON = Path("/content/amsc_val459_metrics.json")


def find_image(image_id: str) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        p = DRIVE_DIR / f"{image_id}{ext}"
        if p.is_file():
            return p
    return None


def main():
    print("=== [AMSC-OCR Full 459 Val Evaluation Starting] ===")
    if not DRIVE_DIR.exists():
        raise RuntimeError(f"Google Drive not mounted at {DRIVE_DIR}")

    val_df = pd.read_csv(VAL_CSV)
    print(f"Loaded {len(val_df)} rows from {VAL_CSV}")

    # Discover weights
    weights_root = ROOT / "weights"
    det_dir = weights_root / "paddle/ppocrv5_mobile_det"
    rec_dir = weights_root / "paddle/PP-OCRv6_medium_rec"
    yolo_weight = weights_root / "final_kaggle/expiry_binary_yolov8n_1280_best.pt"

    backend = AMSC_CascadeOCRBackend(
        runtime_device="cuda",
        fast_exit_enabled=True,
        fast_exit_conf=0.85,
        calendar_min_year=2020,
        calendar_max_year=2035,
        yolo_weights=str(yolo_weight) if yolo_weight.is_file() else None,
        yolo_imgsz=960,
        yolo_conf=0.20,
        yolo_expand=1.0,
        yolo_margin=6,
        yolo_batch_size=8,
        yolo_rec_model_name="PP-OCRv6_medium_rec",
        yolo_rec_model_dir=str(rec_dir) if rec_dir.is_dir() else None,
        full_ocr_params={
            "require_local_weights": False,
            "text_detection_model_name": "PP-OCRv5_mobile_det",
            "text_recognition_model_name": "PP-OCRv6_medium_rec",
            "text_detection_model_dir": str(det_dir) if det_dir.is_dir() else None,
            "text_recognition_model_dir": str(rec_dir) if rec_dir.is_dir() else None,
            "det_max_side": 960,
            "det_thresh": 0.25,
            "box_thresh": 0.50,
            "unclip_ratio": 1.8,
            "recognition_batch_size": 8,
            "fallback_enabled": False,
            "engine": "paddle",
            "enable_mkldnn": False,
        },
    )

    selector = UnionSpatialSelector(
        allow_day_first=True,
        allow_month_names=True,
        keyword_weight=0.7,
        negative_keyword_weight=0.45,
        confidence_weight=0.2,
        detection_confidence_weight=0.2,
        calendar_weight=0.5,
        pattern_weight=0.25,
        position_weight=0.25,
        partial_date_penalty=0.45,
        time_like_penalty=0.60,
        future_date_bonus=0.0,
        yolo_bonus=1.5,
        candidate_score_weight=1.5,
        recognition_log_weight=1.5,
        detection_log_weight=0.5,
        bbox_iou_weight=0.5,
    )
    normalizer = DateNormalizer(none_token="NONE", year_missing_token="NONE", day_missing_token="NONE")

    results = []
    tier_counts = {1: 0, 2: 0, 3: 0}
    timings = []

    print(f"Evaluating {len(val_df)} validation images on GPU...")
    t0_all = time.time()
    for idx, row in val_df.iterrows():
        img_fn = row["filename"]
        img_id = Path(img_fn).stem
        img_path = find_image(img_id)
        if img_path is None:
            print(f"WARNING: Image not found: {img_fn}")
            results.append({
                "image_id": img_id,
                "year": "NONE",
                "month": "NONE",
                "day": "NONE",
                "final_date": "NONE",
            })
            continue

        t0 = time.perf_counter()
        tokens = backend.extract(img_path)
        metrics = backend.last_stage_metrics or {}
        tier = metrics.get("tier_reached", 2)
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
        elapsed = (time.perf_counter() - t0) * 1000
        timings.append(elapsed)

        selected = selector.select(tokens)
        pred = normalizer.normalize(img_id, selected)

        results.append({
            "image_id": img_id,
            "year": str(pred.year),
            "month": str(pred.month),
            "day": str(pred.day),
            "final_date": str(pred.final_date),
            "tier_reached": tier,
            "elapsed_ms": elapsed,
        })

        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{len(val_df)} images ({elapsed:.1f}ms/img, Tier1={tier_counts[1]}, Tier2={tier_counts[2]}, Tier3={tier_counts[3]})")

    total_time = time.time() - t0_all
    pred_df = pd.DataFrame(results)
    pred_df.to_csv(OUTPUT_CSV, index=False)

    # Accuracy against GT
    merged = pd.merge(pred_df, val_df, left_on="image_id", right_on="image_id", suffixes=("_pred", "_gt"))
    em = (merged["final_date_pred"] == merged["final_date_gt"]).sum()
    year_match = (merged["year_pred"] == merged["year_gt"]).sum()
    month_match = (merged["month_pred"] == merged["month_gt"]).sum()
    day_match = (merged["day_pred"] == merged["day_gt"]).sum()
    n = len(merged)

    summary = {
        "total_images": n,
        "exact_match": int(em),
        "exact_match_pct": float(em / n * 100),
        "year_match_pct": float(year_match / n * 100),
        "month_match_pct": float(month_match / n * 100),
        "day_match_pct": float(day_match / n * 100),
        "avg_component_pct": float((year_match + month_match + day_match) / (3 * n) * 100),
        "tier_distribution": tier_counts,
        "avg_latency_ms": float(sum(timings) / len(timings)) if timings else 0.0,
        "total_elapsed_sec": total_time,
    }

    REPORT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n=======================================================")
    print(f"EXACT MATCH:           {summary['exact_match_pct']:.2f}% ({em}/{n})")
    print(f"YEAR MATCH:            {summary['year_match_pct']:.2f}%")
    print(f"MONTH MATCH:           {summary['month_match_pct']:.2f}%")
    print(f"DAY MATCH:             {summary['day_match_pct']:.2f}%")
    print(f"AVG COMPONENT:         {summary['avg_component_pct']:.2f}%")
    print(f"TIER DISTRIBUTION:     {tier_counts}")
    print(f"AVG LATENCY:           {summary['avg_latency_ms']:.1f} ms/image")
    print("=======================================================")

    # Save mismatches for targeted fine-tuning
    mismatches = merged[merged["final_date_pred"] != merged["final_date_gt"]]
    mismatches.to_csv("/content/amsc_val459_mismatches.csv", index=False)
    print(f"Saved {len(mismatches)} failure cases to /content/amsc_val459_mismatches.csv")


if __name__ == "__main__":
    main()
