#!/usr/bin/env python3
"""Colab-side worker for one fresh, evidence-complete CPU validation run."""
from __future__ import annotations

import os
import subprocess
import sys
import tarfile
from pathlib import Path


def run(command: list[str], env: dict[str, str]) -> None:
    print("[RUN]", " ".join(command), flush=True)
    subprocess.run(command, env=env, check=True)


def main() -> int:
    dataset = Path("/content/drive/MyDrive/상품사진입니다")
    if not dataset.is_dir():
        raise SystemExit(f"Drive dataset unavailable: {dataset}")
    root = Path.cwd()
    archive = dataset / "weights/ppocrv6_medium_rec.tar.gz"
    v6_root = Path("/content/itda_v6_weights")
    if not v6_root.is_dir():
        v6_root.mkdir(parents=True, exist_ok=True)
        if not archive.is_file():
            raise SystemExit(f"Missing offline V6 archive: {archive}")
        run(["tar", "-xzf", str(archive), "-C", str(v6_root)], os.environ.copy())
    candidates = [path.parent for path in v6_root.rglob("inference.yml")]
    if not candidates:
        raise SystemExit("V6 archive extraction produced no inference.yml")
    env = {**os.environ, "ITDA_WEIGHTS_ROOT": str(dataset / "weights"), "ITDA_V6_WEIGHTS_ROOT": str(candidates[0])}
    run([sys.executable, "-m", "pip", "install", "--quiet", "paddlepaddle==3.3.1", "paddleocr==3.7.0", "pyyaml", "psutil"], env)
    config = "configs/experiments/best_cpu_baseline_v6_box60.yaml"
    run([sys.executable, "scripts/preflight_measured_dataset.py", "--config", config, "--input", str(dataset), "--split-csv", "data/splits/val.csv"], env)
    run([sys.executable, "scripts/run_measured_experiment.py", "--name", "ppocrv6-raw-box60", "--config", config,
         "--input", str(dataset), "--tier", "val", "--label-provenance", "gemini_filtered"], env)
    run([sys.executable, "scripts/build_measured_leaderboard.py"], env)
    output = root / "runs/measured"
    with tarfile.open("/content/measured_val_artifacts.tar.gz", "w:gz") as tar:
        tar.add(output, arcname="measured")
    print("MEASURED_VAL_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
