#!/usr/bin/env python3
"""Run a 10-image validation smoke gate, then the full 459-image val run."""
from __future__ import annotations

import os
import csv
import subprocess
import sys
import tarfile
from pathlib import Path


def run(command: list[str], env: dict[str, str]) -> None:
    print("[RUN]", " ".join(command), flush=True)
    subprocess.run(command, env=env, check=True)


dataset = Path("/content/drive/MyDrive/상품사진입니다")
root = Path.cwd()
if not dataset.is_dir():
    raise SystemExit(f"Drive dataset unavailable: {dataset}")
archive = dataset / "weights/ppocrv6_medium_rec.tar.gz"
v6_root = Path("/content/itda_v6_weights")
if not v6_root.is_dir():
    v6_root.mkdir(parents=True, exist_ok=True)
    run(["tar", "-xzf", str(archive), "-C", str(v6_root)], os.environ.copy())
v6_dirs = [path.parent for path in v6_root.rglob("inference.yml")]
if not v6_dirs:
    raise SystemExit("No V6 model directory after archive extraction")
env = {**os.environ, "ITDA_WEIGHTS_ROOT": str(dataset / "weights"), "ITDA_V6_WEIGHTS_ROOT": str(v6_dirs[0])}
run([sys.executable, "-m", "pip", "install", "--quiet", "paddlepaddle==3.3.1", "paddleocr==3.7.0", "pyyaml", "psutil"], env)
config = "configs/experiments/best_cpu_baseline_v6_box60.yaml"
smoke_csv = Path("data/splits/val_smoke10.csv")
if not smoke_csv.is_file():
    with Path("data/splits/val.csv").open(encoding="utf-8", newline="") as handle:
        val_rows = list(csv.DictReader(handle))
    with smoke_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=val_rows[0].keys())
        writer.writeheader(); writer.writerows(val_rows[:10])

# Smoke is derived from val.csv and is never rank-eligible.
run([sys.executable, "scripts/preflight_measured_dataset.py", "--config", config, "--input", str(dataset), "--split-csv", str(smoke_csv)], env)
run([sys.executable, "scripts/run_measured_experiment.py", "--name", "ppocrv6-raw-box60-smoke", "--config", config,
     "--input", str(dataset), "--tier", "smoke", "--split-csv", str(smoke_csv), "--label-provenance", "gemini_filtered"], env)

# Full validation is run only after the smoke gate completes successfully.
run([sys.executable, "scripts/preflight_measured_dataset.py", "--config", config, "--input", str(dataset), "--split-csv", "data/splits/val.csv"], env)
run([sys.executable, "scripts/run_measured_experiment.py", "--name", "ppocrv6-raw-box60", "--config", config,
     "--input", str(dataset), "--tier", "val", "--label-provenance", "gemini_filtered"], env)
run([sys.executable, "scripts/build_measured_leaderboard.py"], env)
with tarfile.open("/content/measured_val_artifacts.tar.gz", "w:gz") as archive_out:
    archive_out.add(root / "runs/measured", arcname="measured")
print("MEASURED_VAL_COMPLETE", flush=True)
