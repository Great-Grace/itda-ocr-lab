#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ocr_lab.config import load_config, set_dotted
from ocr_lab.pipeline import run_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a small Cartesian ablation from one YAML config.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--labels")
    parser.add_argument("--max-images", type=int)
    args = parser.parse_args()

    base = load_config(args.config)
    ablation = base.get("ablation", {}) or {}
    grid = ablation.get("grid", {}) or {}
    max_runs = int(ablation.get("max_runs", 8))
    if not grid:
        raise SystemExit("No ablation.grid entries found")
    keys = list(grid)
    values = [value if isinstance(value, list) else [value] for value in (grid[key] for key in keys)]
    combinations = list(itertools.islice(itertools.product(*values), max_runs))
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    summary = []
    for index, combination in enumerate(combinations, start=1):
        config = base
        for key, value in zip(keys, combination):
            config = set_dotted(config, key, value)
        run_dir = output_root / f"run_{index:03d}"
        metrics = run_pipeline(config, args.input, run_dir, args.labels, args.max_images)
        row = {"run": run_dir.name, **metrics}
        for key, value in zip(keys, combination):
            row[key] = value
        summary.append(row)
    fields = sorted({key for row in summary for key in row})
    with (output_root / "ablation_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    (output_root / "ablation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"runs": len(summary), "summary": str(output_root / "ablation_summary.csv")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
