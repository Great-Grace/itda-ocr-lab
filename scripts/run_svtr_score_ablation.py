"""Evaluate source-specific confidence multipliers for SVTR ROI tokens."""
from __future__ import annotations

import json
import subprocess
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--source", default="runs/B11_LOCAL_SVTR_ROI_SCREEN32/svtr_merged_tokens.jsonl")
parser.add_argument("--input", default="runs/B11_LOCAL_SVTR_ROI_SCREEN32/cache_input")
parser.add_argument("--ids-file", default="runs/B0_test_v1/screen32_ids.txt")
parser.add_argument("--out", default="runs/B13_LOCAL_SVTR_SCORE_ABLATION")
args = parser.parse_args()
source = ROOT / args.source
out_root = ROOT / args.out
out_root.mkdir(parents=True, exist_ok=True)
rows = []
for multiplier in (0.25, 0.50, 0.75, 1.00):
    cache = out_root / f"tokens_{multiplier:.2f}.jsonl"
    with source.open(encoding="utf-8") as src, cache.open("w", encoding="utf-8") as dst:
        for line in src:
            row = json.loads(line)
            for token in row.get("tokens", []):
                if token.get("extras", {}).get("source") == "svtr_local_roi":
                    token["confidence"] = float(token.get("confidence", 0.0)) * multiplier
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
    run_dir = out_root / f"multiplier_{multiplier:.2f}"
    command = ["/opt/anaconda3/bin/python", "scripts/run_experiment.py", "--config", "configs/experiments/b8_extended_date_parser.yaml", "--input", args.input, "--output", str(run_dir), "--labels", "runs/B0_test_v1/labels.csv", "--image-ids-file", args.ids_file, "--tokens-cache", str(cache), "--device", "cpu", "--threads", "4"]
    subprocess.run(command, cwd=ROOT, check=True, env={**__import__("os").environ, "PYTHONPATH": "src"}, stdout=subprocess.DEVNULL)
    metrics = json.loads((run_dir / "metrics.json").read_text())
    rows.append({"multiplier": multiplier, "final_em": metrics["final_date_exact_match"], "candidate_recall": metrics["candidate_recall"], "selection": metrics["candidate_selection_accuracy"]})
(out_root / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
print(json.dumps(rows, indent=2))
