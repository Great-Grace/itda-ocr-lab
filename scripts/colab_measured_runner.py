#!/usr/bin/env python3
"""Colab Orchestrator for Fail-Closed Measured OCR Experiments.

Executes strictly according to protocol:
1. Verifies Google Drive mount and checks all 459 Val images + local Paddle weights.
2. Runs Val Smoke test (10 images, seed=42) to verify runtime.
3. Executes preflight on full Val 459 images.
4. Executes fresh OCR on Val 459 images via run_measured_experiment.py.
5. Archives evidence-complete output for local leaderboard ingestion.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path("/content/itda_ocr")
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))

IMAGE_ROOT = Path("/content/drive/MyDrive/상품사진입니다")
VAL_CSV = ROOT / "data/splits/val.csv"
SMOKE_CSV = ROOT / "data/splits/val_smoke_10_labels.csv"
SMOKE_IDS = ROOT / "data/splits/val_smoke_10_ids.txt"
CONFIG_PATH = ROOT / "configs/experiments/best_cpu_baseline_v6_box60.yaml"

print("================================================================")
print("=== [ITDA Measured Benchmark: Colab Execution Starting] ===")
print("================================================================")

# Step 1: Verify Drive Mount
if not IMAGE_ROOT.is_dir():
    print(f"[FAIL] Drive dataset root not found: {IMAGE_ROOT}")
    print("Please ensure Google Drive is mounted in Colab.")
    sys.exit(1)

print(f"[PASS] Google Drive dataset found at: {IMAGE_ROOT}")

# Check 459 Val images coverage
with open(VAL_CSV, encoding="utf-8") as f:
    val_rows = list(csv.DictReader(f))
val_ids = [Path(r["filename"]).stem for r in val_rows]
indexed_stems = {p.stem for p in IMAGE_ROOT.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}}
missing_val = sorted(set(val_ids) - indexed_stems)
if missing_val:
    print(f"[FAIL] Missing {len(missing_val)} images out of 459 in Drive: {missing_val[:10]}")
    sys.exit(1)
print(f"[PASS] 100% Image Coverage Verified: All {len(val_ids)} Val images exist.")

# Locate Paddle weights in Drive
weights_root = IMAGE_ROOT / "weights"
det_dir = weights_root / "paddle" / "ppocrv5_mobile_det"
if not det_dir.is_dir():
    print(f"[FAIL] Missing detector weight dir: {det_dir}")
    sys.exit(1)

# Check V6 recognizer weights
v6_candidates = [
    weights_root / "paddle" / "PP-OCRv6_medium_rec",
    weights_root / "paddle" / "ppocrv6_medium_rec",
    Path("/tmp/PP-OCRv6_medium_rec"),
]
v6_dir = None
for c in v6_candidates:
    if c.is_dir():
        v6_dir = c
        break

if not v6_dir:
    tar_cand = weights_root / "ppocrv6_medium_rec.tar.gz"
    if tar_cand.is_file():
        print(f"[SETUP] Extracting {tar_cand} -> /tmp/PP-OCRv6_medium_rec...")
        v6_dir = Path("/tmp/PP-OCRv6_medium_rec")
        v6_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(["tar", "-xzf", str(tar_cand), "-C", "/tmp"], check=True)

if not v6_dir or not v6_dir.is_dir():
    print(f"[FAIL] Missing PP-OCRv6 medium recognizer weights in {weights_root}")
    sys.exit(1)

print(f"[PASS] Detector weights verified: {det_dir}")
print(f"[PASS] Recognizer weights verified: {v6_dir}")

env = os.environ.copy()
env["ITDA_WEIGHTS_ROOT"] = str(weights_root)
env["ITDA_V6_WEIGHTS_ROOT"] = str(v6_dir)

# Step 2: Val Smoke Test (10 images, seed=42)
print("\n--- [Step 2: Running Smoke Test on 10 Images] ---")
import shutil; shutil.rmtree(ROOT / "runs/measured/smoke_val_10_smoke", ignore_errors=True)
shutil.rmtree(ROOT / "runs/measured/ppocrv6_raw_box60_val", ignore_errors=True)
smoke_cmd = [
    sys.executable, "scripts/run_measured_experiment.py",
    "--name", "smoke_val_10",
    "--config", str(CONFIG_PATH),
    "--input", str(IMAGE_ROOT),
    "--tier", "smoke",
    "--ids-file", str(SMOKE_IDS),
    "--labels", str(SMOKE_CSV),
    "--label-provenance", "gemini_filtered",
    "--threads", "4",
]
smoke_proc = subprocess.run(smoke_cmd, env=env, text=True, capture_output=True)
if smoke_proc.returncode != 0:
    print("[FAIL] Smoke run failed!")
    print("STDOUT:\n", smoke_proc.stdout)
    print("STDERR:\n", smoke_proc.stderr)
    sys.exit(1)
print("[PASS] Smoke test completed successfully (10 images).")

# Step 3: Preflight on full Val 459 images
print("\n--- [Step 3: Running Preflight on Full Val Split (459 images)] ---")
preflight_cmd = [
    sys.executable, "scripts/preflight_measured_dataset.py",
    "--config", str(CONFIG_PATH),
    "--input", str(IMAGE_ROOT),
    "--split-csv", str(VAL_CSV),
]
preflight_proc = subprocess.run(preflight_cmd, env=env, text=True, capture_output=True)
print("Preflight Output:\n", preflight_proc.stdout)
if preflight_proc.returncode != 0:
    print("[FAIL] Preflight check failed! Full run blocked.")
    sys.exit(1)
print("[PASS] Preflight check ready: true.")

# Step 4: Full Official Fresh Run on Val (459 images)
print("\n--- [Step 4: Running Official Measured Baseline (ppocrv6_raw_box60)] ---")
full_cmd = [
    sys.executable, "scripts/run_measured_experiment.py",
    "--name", "ppocrv6_raw_box60",
    "--config", str(CONFIG_PATH),
    "--input", str(IMAGE_ROOT),
    "--tier", "val",
    "--label-provenance", "gemini_filtered",
    "--threads", "4",
]
print(f"[RUN] {" ".join(full_cmd)}")
full_proc = subprocess.run(full_cmd, env=env, text=True)
if full_proc.returncode != 0:
    print("[FAIL] Official full run failed with return code:", full_proc.returncode)
    sys.exit(1)

# Step 5: Package artifacts
out_run_dir = ROOT / "runs/measured/ppocrv6_raw_box60_val"
tar_dest = Path("/content/measured_run_artifacts.tar.gz")
print(f"\n--- [Step 5: Packaging Artifacts -> {tar_dest}] ---")
with tarfile.open(tar_dest, "w:gz") as tar:
    tar.add(out_run_dir, arcname="ppocrv6_raw_box60_val")

print("=== [Colab Measured Experiment Pipeline 100% COMPLETE] ===")
