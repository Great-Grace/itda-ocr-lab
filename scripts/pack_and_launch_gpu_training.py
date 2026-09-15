#!/usr/bin/env python3
"""Pack disjoint splits (train/val/test), upload to Colab, launch detached daemon training, monitor, and download weights."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts" / "colab_fixed_cli.py"


def main():
    parser = argparse.ArgumentParser(description="Launch detached multi-architecture neural training on Colab GPU.")
    parser.add_argument("--session", default="itda-train-gpu", help="Colab session name")
    parser.add_argument("--epochs", type=int, default=8, help="Epochs per architecture")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--multiplier", type=int, default=3, help="On-the-fly augmentation multiplier")
    parser.add_argument("--arch", default="all", help="Architecture or 'all'")
    args = parser.parse_args()

    # Step 1: Create tar payload with disjoint data splits
    payload_tar = Path("/tmp/itda_train_payload.tar.gz")
    print(f"[1/5] Packaging code and disjoint dataset splits into {payload_tar}...")
    with tarfile.open(payload_tar, "w:gz") as tar:
        tar.add(ROOT / "src", arcname="src")
        tar.add(ROOT / "scripts" / "train_multi_architecture_suite.py", arcname="scripts/train_multi_architecture_suite.py")
        tar.add(ROOT / "data" / "splits", arcname="data/splits")
    print(f"      Payload size: {payload_tar.stat().st_size / 1024:.1f} KB")

    # Step 2: Upload payload to Colab
    print(f"[2/5] Uploading payload to Colab session ({args.session})...")
    cmd_upload = [sys.executable, str(CLI), "upload", "-s", args.session, str(payload_tar), "/content/itda_train_payload.tar.gz"]
    subprocess.run(cmd_upload, check=True)
    print("      Upload completed successfully.")

    # Step 3: Launch detached daemon process on Colab
    print(f"[3/5] Launching detached training daemon on Tesla T4 GPU...")
    starter_code = f"""
import os, subprocess, sys, tarfile
from pathlib import Path

work_dir = Path("/content/ocr_train_lab")
if work_dir.exists():
    import shutil
    shutil.rmtree(work_dir)
work_dir.mkdir(parents=True, exist_ok=True)

with tarfile.open("/content/itda_train_payload.tar.gz", "r:gz") as tar:
    tar.extractall(work_dir)

os.chdir(work_dir)
weights_out = Path("/content/weights")
weights_out.mkdir(parents=True, exist_ok=True)

# Terminate any stale runners
subprocess.run(["pkill", "-9", "-f", "train_multi_architecture_suite.py"])

log_path = Path("/content/training.log")
log_file = open(log_path, "w", encoding="utf-8")

cmd = [
    sys.executable, "-u", "scripts/train_multi_architecture_suite.py",
    "--arch", "{args.arch}",
    "--output-root", str(weights_out),
    "--train-csv", "data/splits/train.csv",
    "--val-csv", "data/splits/val.csv",
    "--epochs", "{args.epochs}",
    "--batch-size", "{args.batch_size}",
    "--multiplier", "{args.multiplier}",
    "--device", "cuda"
]

env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"

proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True)
print(f"DAEMON_STARTED: PID {{proc.pid}}", flush=True)
"""
    starter_file = Path("/tmp/launch_remote_daemon.py")
    starter_file.write_text(starter_code, encoding="utf-8")

    res = subprocess.run([sys.executable, str(CLI), "exec", "-s", args.session, "-f", str(starter_file), "--timeout", "30"], capture_output=True, text=True)
    print("      Daemon launcher response:", res.stdout.strip())
    if "DAEMON_STARTED" not in res.stdout:
        print("STDERR:", res.stderr)
        raise RuntimeError("Failed to start detached training daemon on Colab")

    # Step 4: Monitor daemon until completion
    print(f"[4/5] Monitoring training progress via /content/training.log...")
    check_script = Path("/tmp/check_log.py")
    check_script.write_text("""
from pathlib import Path
log_file = Path('/content/training.log')
if log_file.exists():
    lines = log_file.read_text(encoding='utf-8').splitlines()
    print(f"TOTAL_LINES: {len(lines)}")
    for l in lines[-8:]:
        print("LOG:", l)
    if any("ALL 7 ARCHITECTURES TRAINED SUCCESSFULLY" in l or "ALL" in l and "TRAINED SUCCESSFULLY" in l for l in lines):
        # Package weights
        import subprocess, tarfile
        with tarfile.open('/content/trained_weights.tar.gz', 'w:gz') as tar:
            tar.add('/content/weights', arcname='weights')
        print("TRAINING_COMPLETED_AND_PACKAGED")
else:
    print("Log file does not exist yet.")
""", encoding="utf-8")

    t_start = time.time()
    last_reported_line = ""
    while True:
        time.sleep(15)
        res = subprocess.run([sys.executable, str(CLI), "exec", "-s", args.session, "-f", str(check_script), "--timeout", "20"], capture_output=True, text=True)
        out = res.stdout.strip()
        for line in out.splitlines():
            if line.startswith("LOG:") and line != last_reported_line:
                print(f"      [{int(time.time() - t_start)}s] {line[5:]}")
                last_reported_line = line
        if "TRAINING_COMPLETED_AND_PACKAGED" in out:
            print(f"      Training completed and weights packaged in {time.time() - t_start:.1f}s!")
            break
        if time.time() - t_start > 3600:
            raise TimeoutError("Training exceeded 1 hour limit")

    # Step 5: Download trained weights
    local_weights_tar = ROOT / "weights" / "trained_weights.tar.gz"
    (ROOT / "weights").mkdir(parents=True, exist_ok=True)
    print(f"[5/5] Downloading trained weights archive to {local_weights_tar}...")
    cmd_dl = [
        sys.executable, str(CLI), "download", "-s", args.session,
        "/content/trained_weights.tar.gz", str(local_weights_tar)
    ]
    subprocess.run(cmd_dl, check=True)

    print(f"      Extracting weights to {ROOT / 'weights'}...")
    with tarfile.open(local_weights_tar, "r:gz") as tar:
        tar.extractall(ROOT)

    print(f"\nSUCCESS: All architecture checkpoints synchronized to {ROOT / 'weights'}!")
    for p in (ROOT / "weights").iterdir():
        if p.is_dir():
            ckpt = p / "checkpoint.pt"
            if ckpt.exists():
                print(f"  - {p.name:25s}: {ckpt.stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
