#!/usr/bin/env python3
"""Compare CPU process/thread layouts without copying the image dataset."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path


LAYOUTS = ((1, 4), (2, 2), (4, 1))


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark CPU OCR layouts: 1x4, 2x2, and 4x1.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--labels")
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--layouts", default="1x4,2x2,4x1", help="Comma-separated process x thread layouts")
    args = parser.parse_args()

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for processes, threads in _parse_layouts(args.layouts):
        rows.append(_run_layout(args, output_root, processes, threads))
    fields = sorted({key for row in rows for key in row})
    with (output_root / "cpu_benchmark.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    (output_root / "cpu_benchmark.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"layouts": len(rows), "summary": str(output_root / "cpu_benchmark.csv")}, ensure_ascii=False, indent=2))
    return 0


def _parse_layouts(value: str) -> list[tuple[int, int]]:
    layouts = []
    for item in value.split(","):
        processes, threads = item.strip().lower().split("x", 1)
        pair = (int(processes), int(threads))
        if pair[0] < 1 or pair[1] < 1:
            raise SystemExit("process and thread counts must be positive")
        layouts.append(pair)
    return layouts


def _run_layout(args: argparse.Namespace, root: Path, processes: int, threads: int) -> dict[str, object]:
    layout_dir = root / f"p{processes}_t{threads}"
    layout_dir.mkdir(parents=True, exist_ok=True)
    children = []
    base_env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        base_env[key] = str(threads)
    started = time.perf_counter()
    for shard in range(processes):
        output_dir = layout_dir / f"part_{shard:02d}"
        command = [sys.executable, str(Path(__file__).with_name("run_experiment.py")),
            "--config", args.config, "--input", args.input, "--output", str(output_dir),
            "--device", "cpu", "--threads", str(threads), "--shard-index", str(shard), "--shard-count", str(processes)]
        if args.labels:
            command.extend(["--labels", args.labels])
        if args.max_images:
            command.extend(["--max-images", str(args.max_images)])
        children.append(subprocess.Popen(command, env=base_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
    failures = []
    for child in children:
        stdout, stderr = child.communicate()
        if child.returncode:
            failures.append({"returncode": child.returncode, "stdout": stdout[-2000:], "stderr": stderr[-2000:]})
    wall_seconds = time.perf_counter() - started
    if failures:
        raise RuntimeError(json.dumps({"layout": [processes, threads], "failures": failures}, ensure_ascii=False))
    metrics = [_load_json(path) for path in sorted(layout_dir.glob("part_*/metrics.json"))]
    image_count = sum(int(item.get("image_count", 0)) for item in metrics)
    peak_ram = max((float(item.get("peak_ram_mb") or 0.0) for item in metrics), default=0.0)
    model_size = max((float(item.get("model_weight_mb") or 0.0) for item in metrics), default=0.0)
    accuracy = _weighted(metrics, "final_date_exact_match", "labeled_count")
    candidate_recall = _ratio(metrics, "candidate_recall_count", "candidate_recall_denominator")
    selection = _ratio(metrics, "candidate_selection_accuracy", "candidate_selection_denominator")
    return {"processes": processes, "threads_per_process": threads, "image_count": image_count,
        "wall_seconds": wall_seconds, "sec_per_image": wall_seconds / image_count if image_count else 0.0,
        "images_per_sec": image_count / wall_seconds if wall_seconds and image_count else 0.0,
        "final_date_exact_match": accuracy, "candidate_recall": candidate_recall,
        "candidate_selection_accuracy": selection, "peak_ram_mb": peak_ram, "model_weight_mb": model_size}


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _weighted(metrics: list[dict[str, object]], key: str, denominator_key: str) -> float | None:
    denominator = sum(int(item.get(denominator_key, 0) or 0) for item in metrics)
    if not denominator:
        return None
    numerator = sum(float(item.get(key, 0.0) or 0.0) * int(item.get(denominator_key, 0) or 0) for item in metrics)
    return numerator / denominator


def _ratio(metrics: list[dict[str, object]], numerator_key: str, denominator_key: str) -> float | None:
    denominator = sum(int(item.get(denominator_key, 0) or 0) for item in metrics)
    if not denominator:
        return None
    if numerator_key.endswith("accuracy"):
        numerator = sum(float(item.get(numerator_key, 0.0) or 0.0) * int(item.get(denominator_key, 0) or 0) for item in metrics)
    else:
        numerator = sum(int(item.get(numerator_key, 0) or 0) for item in metrics)
    return numerator / denominator


if __name__ == "__main__":
    raise SystemExit(main())
