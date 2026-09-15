#!/usr/bin/env python3
"""Automated Colab GPU runner for full-dataset (3,352 images) OCR extraction and audit.

Orchestrates:
1. Provisioning a Colab GPU (T4) session via colab_fixed_cli.
2. Mounting Google Drive (MyDrive/상품사진입니다).
3. High-speed GPU OCR token extraction across all images.
4. Offline consensus pseudo-label generation and 5-stage quality census audit.
5. Downloading clean labels, flagged anomalies, and evidence reports back locally.
6. Zero external paid API guarantee (0 cost).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts" / "colab_fixed_cli.py"


def run_cmd(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    print(f"[RUN] {' '.join(cmd)}")
    return subprocess.run(cmd, check=check, text=True)


def build_payload_bundle(bundle_path: Path) -> None:
    print(f"[PACK] Building payload bundle -> {bundle_path}")
    with tarfile.open(bundle_path, "w:gz") as tar:
        for item in [
            "src",
            "configs/experiments/b39-boundary-and-date-priority.yaml",
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


REMOTE_SCRIPT = '''#!/usr/bin/env python3
import os
import subprocess
import sys
import tarfile
from pathlib import Path

print("=== [Colab Remote Worker Initializing] ===")
bundle = Path("/content/itda_payload.tar.gz")
with tarfile.open(bundle, "r:gz") as tar:
    tar.extractall("/content/itda_ocr")

os.chdir("/content/itda_ocr")
sys.path.insert(0, "/content/itda_ocr/src")

# 1. Install dependencies
print("[1/5] Installing dependencies...")
subprocess.run([
    sys.executable, "-m", "pip", "install", "--quiet",
    "paddlepaddle-gpu==3.3.1" if subprocess.run(["which", "nvcc"], capture_output=True).returncode == 0 else "paddlepaddle==3.3.1",
    "paddleocr==3.7.0", "pyyaml", "psutil"
], check=False)

dataset_dir = Path("/content/drive/MyDrive/상품사진입니다")
if not dataset_dir.is_dir():
    raise SystemExit(f"[ERROR] Dataset folder not found in Drive: {dataset_dir}")

weights_dir = dataset_dir / "weights"
v6_archive = weights_dir / "ppocrv6_medium_rec.tar.gz"
v6_dir = Path("/tmp/PP-OCRv6_medium_rec")
if not v6_dir.is_dir():
    if v6_archive.is_file():
        print("[SETUP] Extracting PP-OCRv6 medium weights to /tmp...")
        v6_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(["tar", "-xzf", str(v6_archive), "-C", "/tmp"], check=True)
    elif (weights_dir / "paddle" / "PP-OCRv6_medium_rec").is_dir():
        v6_dir = weights_dir / "paddle" / "PP-OCRv6_medium_rec"

env = os.environ.copy()
if weights_dir.is_dir():
    env["ITDA_WEIGHTS_ROOT"] = str(weights_dir)
env["ITDA_V6_WEIGHTS_ROOT"] = str(v6_dir)

# 2. Run OCR extraction
print("[2/5] Running High-Speed GPU OCR Extraction...")
out_dir = Path("/content/ocr_output")
out_dir.mkdir(parents=True, exist_ok=True)

ocr_cmd = [
    sys.executable, "scripts/run_experiment.py",
    "--config", "configs/experiments/b39-boundary-and-date-priority.yaml",
    "--input", str(dataset_dir),
    "--output", str(out_dir),
    "--device", "cuda",
    "--threads", "4",
]
__MAX_IMAGES_ARG__

subprocess.run(ocr_cmd, env=env, check=True)

# 3. Run Pseudo-Label Generation
print("[3/5] Generating Strict Consensus Pseudo-Labels...")
pseudo_dir = Path("/content/pseudo_labeled")
pseudo_dir.mkdir(parents=True, exist_ok=True)

pseudo_cmd = [
    sys.executable, "scripts/build_pseudo_labels.py",
    "--tokens-cache", str(out_dir / "ocr_tokens.jsonl"),
    "--existing-labels", "runs/B0_test_v1/labels.csv",
    "--output-dir", str(pseudo_dir),
]
subprocess.run(pseudo_cmd, check=True)

# 4. Run Quality Census Audit
print("[4/5] Running 5-Stage Full Census & Quality Audit...")
audit_dir = pseudo_dir / "audit"
audit_cmd = [
    sys.executable, "scripts/audit_ai_labels.py",
    "--labels", str(pseudo_dir / "pseudo_labels.csv"),
    "--gt-labels", "runs/B0_test_v1/labels.csv",
    "--tokens-cache", str(out_dir / "ocr_tokens.jsonl"),
    "--output-dir", str(audit_dir),
]
subprocess.run(audit_cmd, check=True)

# 5. Pack Artifacts
print("[5/5] Packing results...")
with tarfile.open("/content/full_audit_artifacts.tar.gz", "w:gz") as tar:
    tar.add(out_dir / "ocr_tokens.jsonl", arcname="ocr_tokens.jsonl")
    tar.add(out_dir / "predictions.csv", arcname="predictions.csv")
    tar.add(out_dir / "metrics.json", arcname="metrics.json")
    tar.add(str(pseudo_dir), arcname="pseudo_labeled")

print("=== [Colab Remote Worker Completed Successfully] ===")
'''


def main() -> int:
    parser = argparse.ArgumentParser(description="Run full dataset labeling & audit on Colab GPU.")
    parser.add_argument("--session", default="itda-full-audit", help="Colab session name")
    parser.add_argument("--gpu", default="T4", choices=["T4", "A100", "V100"], help="GPU type")
    parser.add_argument("--max-images", type=int, help="Limit image count for smoke check")
    parser.add_argument("--keep-session", action="store_true", help="Keep session alive on finish")
    parser.add_argument("--output-dir", default="runs/full_dataset_audit", help="Local output destination")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle_path = out_dir / "itda_payload.tar.gz"
    build_payload_bundle(bundle_path)

    # Customize remote script
    remote_script_code = REMOTE_SCRIPT
    if args.max_images:
        remote_script_code = remote_script_code.replace(
            "__MAX_IMAGES_ARG__", f'ocr_cmd.extend(["--max-images", "{args.max_images}"])'
        )
    else:
        remote_script_code = remote_script_code.replace("__MAX_IMAGES_ARG__", "")

    remote_py = out_dir / "remote_worker.py"
    remote_py.write_text(remote_script_code, encoding="utf-8")

    session = args.session
    cli_cmd = [sys.executable, str(CLI)]

    try:
        # 1. Create session (attempt requested GPU, fallback to T4 if needed)
        print(f"[COLAB] Provisioning GPU session: {session} ({args.gpu})...")
        res = run_cmd(cli_cmd + ["new", "-s", session, "--gpu", args.gpu], check=False)
        if res.returncode != 0 and args.gpu != "T4":
            print(f"[COLAB] {args.gpu} unavailable or quota reached. Falling back to T4...")
            run_cmd(cli_cmd + ["new", "-s", session, "--gpu", "T4"], check=True)
        elif res.returncode != 0:
            raise RuntimeError(f"Failed to provision Colab GPU session: {session}")

        # 2. Mount drive
        print("[COLAB] Mounting Google Drive...")
        run_cmd(cli_cmd + ["drivemount", "-s", session])

        # 3. Upload bundle and worker script
        print("[COLAB] Uploading source bundle and worker script...")
        run_cmd(cli_cmd + ["upload", "-s", session, str(bundle_path), "/content/itda_payload.tar.gz"])
        run_cmd(cli_cmd + ["upload", "-s", session, str(remote_py), "/content/remote_worker.py"])

        # 4. Execute remote worker
        print("[COLAB] Executing remote pipeline...")
        run_cmd(cli_cmd + ["exec", "-s", session, "-f", "/content/remote_worker.py"])

        # 5. Download results
        archive_dest = out_dir / "full_audit_artifacts.tar.gz"
        print(f"[COLAB] Downloading artifacts -> {archive_dest}...")
        run_cmd(cli_cmd + ["download", "-s", session, "/content/full_audit_artifacts.tar.gz", str(archive_dest)])

        # 6. Extract locally
        print(f"[EXTRACT] Unpacking artifacts into {out_dir}...")
        with tarfile.open(archive_dest, "r:gz") as tar:
            tar.extractall(out_dir)

        print(f"\n[SUCCESS] Full dataset labeling & audit complete! Results saved in {out_dir}")
        return 0

    finally:
        if not args.keep_session:
            print(f"[COLAB] Stopping session: {session}...")
            run_cmd(cli_cmd + ["stop", "-s", session], check=False)


if __name__ == "__main__":
    raise SystemExit(main())
