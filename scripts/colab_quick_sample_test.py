"""Run a quick sample benchmark on Colab (e.g. 30 or 50 images) to approximate latency and exact match."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

# Auto-detect repository / payload root
CANDIDATE_ROOTS = [
    Path("/content/itda_final_payload"),
    Path("/content/itda_lab"),
    Path(__file__).resolve().parents[1],
]
ROOT = next((p for p in CANDIDATE_ROOTS if (p / "predict.ipynb").is_file() or (p / "src").is_dir()), CANDIDATE_ROOTS[0])
sys.path.insert(0, str(ROOT / "src"))

from ocr_lab.modules.amsc_cascade_ocr import AMSC_CascadeOCRBackend
from ocr_lab.modules.date_normalizer import DateNormalizer
from ocr_lab.modules.union_spatial_selector import UnionSpatialSelector

CANDIDATE_DRIVES = [
    Path("/content/drive/MyDrive/상품사진입니다"),
    Path("/content/itda_locked_internal_test/images"),
    ROOT / "data" / "sample",
]
CANDIDATE_TEST_CSVS = [
    Path("/content/itda_locked_test.csv"),
    ROOT / "data" / "splits" / "test.csv",
    ROOT / "data" / "splits" / "val.csv",
]


def find_image(image_root: Path, image_id: str) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
        p = image_root / f"{image_id}{ext}"
        if p.is_file():
            return p
    return None


def run_quick_benchmark(sample_size: int = 30, device: str = "cpu"):
    drive_dir = next((p for p in CANDIDATE_DRIVES if p.is_dir()), None)
    if drive_dir is None:
        raise RuntimeError("No image directory found. Please ensure Google Drive is mounted at /content/drive/MyDrive/상품사진입니다")

    test_csv_path = next((p for p in CANDIDATE_TEST_CSVS if p.is_file()), None)
    if test_csv_path is None:
        raise RuntimeError("Test split CSV not found.")

    with test_csv_path.open(encoding="utf-8", newline="") as handle:
        all_rows = list(csv.DictReader(handle))

    # Evenly stride across the test set for an unbiased representation
    total_available = len(all_rows)
    stride = max(1, total_available // sample_size)
    sampled_rows = all_rows[::stride][:sample_size]

    print(f"\n=======================================================")
    print(f"🚀 [AMSC-OCR Fast Approximation Benchmark]")
    print(f"• Root:       {ROOT}")
    print(f"• Split CSV:  {test_csv_path.name} (Total {total_available} items)")
    print(f"• Sample:     {len(sampled_rows)} images (strided)")
    print(f"• Device:     {device.upper()}")
    print(f"• Images Dir: {drive_dir}")
    print(f"=======================================================\n")

    weights_root = ROOT / "weights"
    det_dir = weights_root / "paddle" / "ppocrv5_mobile_det"
    rec_dir = Path(os.environ.get("ITDA_V6_WEIGHTS_ROOT", str(weights_root / "paddle" / "PP-OCRv6_medium_rec")))
    yolo_weight = Path(os.environ.get("ITDA_YOLO_WEIGHTS", str(weights_root / "final_kaggle" / "expiry_binary_yolov8n_1280_best.pt")))

    backend = AMSC_CascadeOCRBackend(
        runtime_device=device,
        fast_exit_enabled=True,
        fast_exit_conf=0.85,
        calendar_min_year=2020,
        calendar_max_year=2035,
        yolo_weights=str(yolo_weight),
        yolo_imgsz=960,
        yolo_conf=0.20,
        yolo_expand=1.0,
        yolo_margin=6,
        yolo_batch_size=8,
        yolo_max_boxes=4,
        yolo_rec_model_name="PP-OCRv6_medium_rec",
        yolo_rec_model_dir=str(rec_dir),
        full_ocr_params={
            "require_local_weights": True,
            "text_detection_model_name": "PP-OCRv5_mobile_det",
            "text_recognition_model_name": "PP-OCRv6_medium_rec",
            "text_detection_model_dir": str(det_dir),
            "text_recognition_model_dir": str(rec_dir),
            "det_max_side": 960,
            "det_thresh": 0.25,
            "box_thresh": 0.50,
            "unclip_ratio": 1.8,
            "recognition_batch_size": 1,
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

    latencies = []
    em_count = 0
    mismatches = []

    for idx, row in enumerate(sampled_rows, 1):
        filename = row["filename"]
        target_date = row.get("date", row.get("final_date", "NONE"))
        image_id = Path(filename).stem
        img_path = find_image(drive_dir, image_id)

        if img_path is None:
            print(f"[{idx:02d}/{len(sampled_rows)}] ⚠️ Image not found: {filename}")
            continue

        t0 = time.perf_counter()
        tokens = backend.extract(img_path)
        selected = selector.select(tokens)
        pred = normalizer.normalize(image_id, selected)
        dt = (time.perf_counter() - t0) * 1000
        latencies.append(dt)

        matched = (pred.final_date == target_date)
        if matched:
            em_count += 1
            status = "✅ MATCH"
        else:
            status = f"❌ WRONG (pred={pred.final_date} vs gt={target_date})"
            mismatches.append({"image_id": image_id, "pred": pred.final_date, "gt": target_date})

        print(f"[{idx:02d}/{len(sampled_rows)}] {image_id} | {dt:6.1f}ms | {status}")

    n = len(latencies)
    if n == 0:
        print("No images were evaluated.")
        return

    avg_latency = sum(latencies) / n
    sec_per_img = avg_latency / 1000
    em_pct = (em_count / n) * 100
    est_500_min = (sec_per_img * 500) / 60

    print("\n=======================================================")
    print("📊 [BENCHMARK APPROXIMATION RESULT]")
    print(f"• Evaluated Images:  {n}/{len(sampled_rows)}")
    print(f"• Exact Match (EM):  {em_pct:.2f}% ({em_count}/{n})")
    print(f"• Avg Latency:       {avg_latency:.1f} ms/image ({sec_per_img:.2f} s/image)")
    print(f"• Est. 500 Images:   {est_500_min:.1f} minutes (Budget: 40.0 min)")
    print("=======================================================\n")

    if mismatches:
        print(f"🔍 Top {min(5, len(mismatches))} Mismatches for quick check:")
        for m in mismatches[:5]:
            print(f"   - {m['image_id']}: pred={m['pred']} != gt={m['gt']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=30, help="Number of images to sample")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Inference device")
    args = parser.parse_args()
    run_quick_benchmark(sample_size=args.sample_size, device=args.device)
