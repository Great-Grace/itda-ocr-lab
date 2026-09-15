#!/usr/bin/env python3
"""Run full 3,352 dataset Gemini 3.6 Flash VLM labeling on Colab itda-labeler session.

Zero external cost (Free Tier 1,500 RPD compliance).
Processes ~1,118 batches (3-in-1 images) with resume capability.
"""
from __future__ import annotations

import argparse
import base64
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts" / "colab_fixed_cli.py"


def main():
    parser = argparse.ArgumentParser(description="Run full Gemini VLM labeling on Colab.")
    parser.add_argument("--session", default="itda-labeler", help="Colab session name")
    parser.add_argument("--batch-size", type=int, default=3, help="Images per call")
    parser.add_argument("--delay", type=float, default=4.0, help="Seconds between calls")
    parser.add_argument("--local-out", default="runs/gemini_vlm_labels", help="Local output directory")
    args = parser.parse_args()

    local_out = ROOT / args.local_out
    local_out.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        env_file = ROOT / ".env"
        if env_file.is_file():
            for line in env_file.read_text().splitlines():
                if line.startswith("GEMINI_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()

    labeler_code = (ROOT / "scripts" / "gemini_vlm_labeler.py").read_text(encoding="utf-8")
    labeler_b64 = base64.b64encode(labeler_code.encode("utf-8")).decode("ascii")

    # Prepare remote runner script
    remote_runner = f"""
import base64, os, sys
from pathlib import Path

print("=== [Colab Remote Driver Starting] ===", flush=True)

# Deploy labeler script directly
labeler_path = Path("/content/gemini_vlm_labeler.py")
labeler_code = base64.b64decode("{labeler_b64}").decode("utf-8")
labeler_path.write_text(labeler_code, encoding="utf-8")

os.environ['GEMINI_API_KEY'] = '{api_key}'
sys.path.insert(0, "/content")
sys.argv = [
    "gemini_vlm_labeler.py",
    "--input-dir", "/content/drive/MyDrive/상품사진입니다",
    "--output-dir", "/content/runs/gemini_vlm_labels",
    "--batch-size", "{args.batch_size}",
    "--delay", "{args.delay}"
]

import gemini_vlm_labeler
gemini_vlm_labeler.main()
print("=== [VLM Labeling FINISHED] ===", flush=True)
"""
    runner_file = local_out / "remote_driver.py"
    runner_file.write_text(remote_runner, encoding="utf-8")

    # Execute on Colab
    print(f"[1/2] Launching remote labeling process on Colab ({args.session})...")
    print(f"Total: 3,352 images in batches of {args.batch_size} (~1,118 API calls, safe 15 RPM delay {args.delay}s).")
    subprocess.run([
        sys.executable, str(CLI), "exec", "-s", args.session,
        "-f", str(runner_file),
        "--timeout", "10800"
    ], check=True)

    # Download final labels
    print(f"[2/2] Downloading final labeled dataset to {local_out}...")
    subprocess.run([
        sys.executable, str(CLI), "download", "-s", args.session,
        "/content/runs/gemini_vlm_labels/gemini_labels.csv",
        str(local_out / "gemini_labels.csv")
    ], check=True)
    subprocess.run([
        sys.executable, str(CLI), "download", "-s", args.session,
        "/content/runs/gemini_vlm_labels/gemini_labels.jsonl",
        str(local_out / "gemini_labels.jsonl")
    ], check=True)

    print(f"\nSUCCESS: Full 3,352 dataset VLM labels downloaded to {local_out}!")


if __name__ == "__main__":
    main()
