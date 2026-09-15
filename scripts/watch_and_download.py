#!/usr/bin/env python3
"""Monitor Colab training daemon, download weights upon completion, and run test set benchmark."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts" / "colab_fixed_cli.py"
SESSION = "itda-train-gpu"


def main():
    print("=== Training Monitor Initialized ===")
    t_start = time.time()
    last_line = ""

    check_code = """
from pathlib import Path
import subprocess, tarfile

log_p = Path('/content/training.log')
if log_p.exists():
    lines = log_p.read_text(encoding='utf-8').splitlines()
    print(f'TOTAL_LINES: {len(lines)}')
    for l in lines[-6:]:
        print('LOG:', l)
    if any('ALL 7 ARCHITECTURES TRAINED SUCCESSFULLY' in l for l in lines):
        weights_p = Path('/content/weights')
        tar_p = Path('/content/trained_weights.tar.gz')
        if not tar_p.exists() or tar_p.stat().st_size < 1000:
            with tarfile.open(tar_p, 'w:gz') as tar:
                tar.add(weights_p, arcname='weights')
        print('TRAINING_COMPLETED_AND_PACKAGED')
"""
    tmp_check = Path("/tmp/check_daemon_log.py")
    tmp_check.write_text(check_code, encoding="utf-8")

    while True:
        time.sleep(15)
        res = subprocess.run(
            [sys.executable, str(CLI), "exec", "-s", SESSION, "-f", str(tmp_check), "--timeout", "25"],
            capture_output=True,
            text=True,
        )
        out = res.stdout.strip()
        for l in out.splitlines():
            if l.startswith("LOG:") and l != last_line:
                elapsed = int(time.time() - t_start)
                print(f"[{elapsed}s] {l[5:]}", flush=True)
                last_line = l

        if "TRAINING_COMPLETED_AND_PACKAGED" in out:
            elapsed = int(time.time() - t_start)
            print(f"\n[SUCCESS] GPU training completed across all 7 architectures in {elapsed}s!", flush=True)
            break

        if time.time() - t_start > 3600:
            raise TimeoutError("Training exceeded 1 hour limit")

    # Download trained weights
    local_weights_tar = ROOT / "weights" / "trained_weights.tar.gz"
    (ROOT / "weights").mkdir(parents=True, exist_ok=True)
    print(f"Downloading trained weights archive to {local_weights_tar}...", flush=True)
    subprocess.run(
        [sys.executable, str(CLI), "download", "-s", SESSION, "/content/trained_weights.tar.gz", str(local_weights_tar)],
        check=True,
    )

    print("Extracting weights to local weights directory...", flush=True)
    with tarfile.open(local_weights_tar, "r:gz") as tar:
        tar.extractall(ROOT)

    print("\nVerified Downloaded Model Weights:")
    for p in sorted((ROOT / "weights").iterdir()):
        if p.is_dir():
            ckpt = p / "checkpoint.pt"
            if ckpt.exists():
                print(f"  - {p.name:25s}: {ckpt.stat().st_size / 1024 / 1024:.2f} MB", flush=True)

    # Run Benchmark against isolated test set
    print("\nExecuting Final Master Benchmark on isolated Test Set (data/splits/test.csv)...", flush=True)
    bench_res = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_multi_architecture_benchmark.py")],
        capture_output=True,
        text=True,
    )
    print(bench_res.stdout)
    if bench_res.returncode != 0:
        print("STDERR:", bench_res.stderr)

    print("\n=== ALL TRAINING, EVALUATION, AND BENCHMARKING COMPLETED SUCCESSFULLY ===", flush=True)


if __name__ == "__main__":
    main()
