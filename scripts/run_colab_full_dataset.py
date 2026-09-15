#!/usr/bin/env python3
"""Run full dataset OCR extraction, pseudo-labeling, and quality census audit on Colab.

Uses the active mounted Colab session (itda-test).
Zero external paid API cost.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts" / "colab_fixed_cli.py"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run dataset OCR & audit on Colab.")
    parser.add_argument("--session", default="itda-test", help="Colab session name")
    parser.add_argument("--max-images", type=int, help="Optional image limit for test")
    parser.add_argument("--output-dir", default="runs/full_3352_audit", help="Local download directory")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = out_dir / "itda_payload.tar.gz"

    # 1. Build payload bundle
    print(f"[1/4] Packing local payload -> {bundle_path}...")
    with tarfile.open(bundle_path, "w:gz") as tar:
        for item in [
            "src",
            "configs",
            "scripts/run_experiment.py",
            "scripts/build_pseudo_labels.py",
            "scripts/audit_ai_labels.py",
            "scripts/check_submission.py",
            "runs/B0_test_v1/labels.csv",
            "requirements.txt",
        ]:
            p = ROOT / item
            if p.exists():
                tar.add(p, arcname=item)

    # 2. Generate remote worker script
    max_img_code = f', "--max-images", "{args.max_images}"' if args.max_images else ""
    remote_script_content = f'''from pathlib import Path
import os, subprocess, sys, tarfile, json

print("=== [Colab Remote Worker Starting] ===")

# Unpack code
repo_dir = Path("/content/itda_ocr")
repo_dir.mkdir(parents=True, exist_ok=True)
with tarfile.open("/content/itda_payload.tar.gz", "r:gz") as tar:
    tar.extractall(repo_dir)

os.chdir(repo_dir)
sys.path.insert(0, str(repo_dir / "src"))

# Setup weights
dataset_dir = Path("/content/drive/MyDrive/상품사진입니다")
weights_dir = dataset_dir / "weights"
v6_archive = weights_dir / "ppocrv6_medium_rec.tar.gz"
v6_dir = Path("/tmp/PP-OCRv6_medium_rec")
if not v6_dir.is_dir():
    if v6_archive.is_file():
        print("Extracting PP-OCRv6 medium weights...")
        subprocess.run(["tar", "-xzf", str(v6_archive), "-C", "/tmp"], check=True)
    elif (weights_dir / "paddle" / "PP-OCRv6_medium_rec").is_dir():
        v6_dir = weights_dir / "paddle" / "PP-OCRv6_medium_rec"

env = os.environ.copy()
env["ITDA_WEIGHTS_ROOT"] = str(weights_dir)
env["ITDA_V6_WEIGHTS_ROOT"] = str(v6_dir)
env["PYTHONPATH"] = str(repo_dir / "src")
env["FLAGS_use_mkldnn"] = "0"
env["OMP_NUM_THREADS"] = "4"
env["PYTHONUNBUFFERED"] = "1"

# Run OCR extraction
out_dir = Path("/content/runs/full_3352_b39")
out_dir.mkdir(parents=True, exist_ok=True)
cmd = [
    sys.executable, "-u", "scripts/run_experiment.py",
    "--config", "configs/experiments/b39-boundary-and-date-priority.yaml",
    "--input", str(dataset_dir),
    "--output", str(out_dir),
    "--device", "cpu",
    "--threads", "4"
    {max_img_code}
]
print("Running OCR extraction command:", cmd, flush=True)
proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in proc.stdout:
    sys.stdout.write(line)
    sys.stdout.flush()
ret = proc.wait()
if ret != 0:
    raise RuntimeError(f"OCR command failed with exit code {{ret}}")

# Run pseudo-label generation
pseudo_dir = Path("/content/runs/full_3352_pseudo")
pseudo_dir.mkdir(parents=True, exist_ok=True)

cmd_pseudo = [
    sys.executable, "scripts/build_pseudo_labels.py",
    "--tokens-cache", str(out_dir / "ocr_tokens.jsonl"),
    "--existing-labels", "runs/B0_test_v1/labels.csv",
    "--output-dir", str(pseudo_dir),
]
print("Running pseudo-label generation...")
subprocess.run(cmd_pseudo, check=True)

# Run 5-stage quality audit
cmd_audit = [
    sys.executable, "scripts/audit_ai_labels.py",
    "--labels", str(pseudo_dir / "pseudo_labels.csv"),
    "--gt-labels", "runs/B0_test_v1/labels.csv",
    "--tokens-cache", str(out_dir / "ocr_tokens.jsonl"),
    "--output-dir", str(pseudo_dir / "audit"),
]
print("Running 5-stage quality audit...")
subprocess.run(cmd_audit, check=True)

# Pack artifacts
archive_path = Path("/content/full_3352_results.tar.gz")
with tarfile.open(archive_path, "w:gz") as tar:
    tar.add(out_dir / "ocr_tokens.jsonl", arcname="ocr_tokens.jsonl")
    tar.add(out_dir / "predictions.csv", arcname="predictions.csv")
    tar.add(out_dir / "metrics.json", arcname="metrics.json")
    tar.add(str(pseudo_dir), arcname="pseudo_labeled")

print("SUCCESS: Artifacts packed to", archive_path, archive_path.stat().st_size)
'''
    driver_path = out_dir / "driver.py"
    driver_path.write_text(remote_script_content, encoding="utf-8")

    cli_cmd = [sys.executable, str(CLI)]

    # 3. Upload bundle and driver
    print(f"[2/4] Uploading payload to Colab session {args.session}...")
    subprocess.run(cli_cmd + ["upload", "-s", args.session, str(bundle_path), "/content/itda_payload.tar.gz"], check=True)

    # 4. Execute on Colab
    print(f"[3/4] Executing remote pipeline on Colab ({args.session})...")
    timeout = 180 if args.max_images else 7200
    subprocess.run(cli_cmd + ["exec", "-s", args.session, "-f", str(driver_path), "--timeout", str(timeout)], check=True)

    # 5. Download results
    local_archive = out_dir / "full_3352_results.tar.gz"
    print(f"[4/4] Downloading artifacts -> {local_archive}...")
    subprocess.run(cli_cmd + ["download", "-s", args.session, "/content/full_3352_results.tar.gz", str(local_archive)], check=True)

    print(f"[EXTRACT] Extracting {local_archive}...")
    with tarfile.open(local_archive, "r:gz") as tar:
        tar.extractall(out_dir)

    print(f"\n=== [COMPLETE: Full dataset results saved to {out_dir}] ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
