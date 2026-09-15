#!/usr/bin/env python3
"""Occam's Razor feature ablation script for date selector.

Measures the marginal contribution (Delta EM) of each individual heuristic:
- Bare regex baseline
- + Keyword scoring
- + Negative keyword penalization
- + Calendar / valid date priority
- + Spatial position prior
- + Day-first (DMY) parsing
- + Month name parsing
- + Partial date penalty
- + Time-like string penalty
- + Future date bonus

Any heuristic providing < 0.5%p gain is flagged for pruning under Occam's Razor.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ocr_lab.contracts import OCRToken
from ocr_lab.modules.date_candidates import parse_final_date
from ocr_lab.modules.date_normalizer import DateNormalizer
from ocr_lab.modules.regex_selector import KeywordRegexSelector


def load_dataset(tokens_cache_path: str | Path, labels_path: str | Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    labels: dict[str, str] = {}
    with Path(labels_path).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("image_id"):
                labels[str(row["image_id"])] = str(row.get("final_date", "NONE"))

    dataset: list[dict[str, Any]] = []
    with Path(tokens_cache_path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            img_id = str(item["image_id"])
            if img_id in labels:
                tokens = [OCRToken(**t) for t in item.get("tokens", [])]
                dataset.append({"image_id": img_id, "tokens": tokens, "gt": labels[img_id]})
    return dataset, labels


def eval_selector(dataset: list[dict[str, Any]], selector_kwargs: dict[str, Any]) -> dict[str, float]:
    selector = KeywordRegexSelector(**selector_kwargs)
    normalizer = DateNormalizer()

    total = len(dataset)
    correct = 0
    t0 = time.perf_counter()
    for item in dataset:
        img_id = item["image_id"]
        gt = item["gt"]
        tokens = item["tokens"]

        selected = selector.select(tokens)
        pred = normalizer.normalize(img_id, selected).final_date

        gt_parsed = parse_final_date(gt)
        if pred == gt or (gt_parsed is not None and parse_final_date(pred) == gt_parsed):
            correct += 1
    elapsed_ms = (time.perf_counter() - t0) * 1000 / total if total else 0.0

    return {
        "exact_match": round(correct / total, 4) if total else 0.0,
        "correct": correct,
        "total": total,
        "ms_per_image": round(elapsed_ms, 3),
    }


def main():
    parser = argparse.ArgumentParser(description="Run Occam's Razor feature ablation.")
    parser.add_argument("--tokens-cache", required=True, help="Path to ocr_tokens.jsonl")
    parser.add_argument("--labels", required=True, help="Path to labels.csv")
    parser.add_argument("--output", default="runs/occam_ablation", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {args.tokens_cache} and {args.labels}...")
    dataset, _ = load_dataset(args.tokens_cache, args.labels)
    print(f"Loaded {len(dataset)} items.")

    # 1. Bare baseline: pure date matching without any heuristics
    stages = [
        ("00_bare_regex", {
            "allow_day_first": False,
            "allow_month_names": False,
            "keyword_weight": 0.0,
            "negative_keyword_weight": 0.0,
            "confidence_weight": 0.0,
            "detection_confidence_weight": 0.0,
            "calendar_weight": 0.0,
            "pattern_weight": 0.0,
            "position_weight": 0.0,
            "partial_date_penalty": 0.0,
            "time_like_penalty": 0.0,
            "future_date_bonus": 0.0,
        }),
        ("01_+calendar_priority", {
            "calendar_weight": 0.5,
        }),
        ("02_+keyword_scoring", {
            "calendar_weight": 0.5,
            "keyword_weight": 0.5,
        }),
        ("03_+negative_keywords", {
            "calendar_weight": 0.5,
            "keyword_weight": 0.5,
            "negative_keyword_weight": 0.3,
        }),
        ("04_+day_first_dmy", {
            "calendar_weight": 0.5,
            "keyword_weight": 0.5,
            "negative_keyword_weight": 0.3,
            "allow_day_first": True,
        }),
        ("05_+month_names", {
            "calendar_weight": 0.5,
            "keyword_weight": 0.5,
            "negative_keyword_weight": 0.3,
            "allow_day_first": True,
            "allow_month_names": True,
        }),
        ("06_+position_prior", {
            "calendar_weight": 0.5,
            "keyword_weight": 0.5,
            "negative_keyword_weight": 0.3,
            "allow_day_first": True,
            "allow_month_names": True,
            "position_weight": 0.15,
        }),
        ("07_+partial_penalty", {
            "calendar_weight": 0.5,
            "keyword_weight": 0.5,
            "negative_keyword_weight": 0.3,
            "allow_day_first": True,
            "allow_month_names": True,
            "position_weight": 0.15,
            "partial_date_penalty": 0.35,
        }),
        ("08_+time_penalty", {
            "calendar_weight": 0.5,
            "keyword_weight": 0.5,
            "negative_keyword_weight": 0.3,
            "allow_day_first": True,
            "allow_month_names": True,
            "position_weight": 0.15,
            "partial_date_penalty": 0.35,
            "time_like_penalty": 0.50,
        }),
        ("09_+future_bonus_full", {
            "calendar_weight": 0.5,
            "keyword_weight": 0.5,
            "negative_keyword_weight": 0.3,
            "allow_day_first": True,
            "allow_month_names": True,
            "position_weight": 0.15,
            "partial_date_penalty": 0.35,
            "time_like_penalty": 0.50,
            "future_date_bonus": 0.15,
        }),
    ]

    results = []
    prev_em = 0.0

    print("\n=== Running Occam's Razor Sequential Ablation ===")
    for name, overrides in stages:
        base_params = {
            "allow_day_first": False,
            "allow_month_names": False,
            "keyword_weight": 0.0,
            "negative_keyword_weight": 0.0,
            "confidence_weight": 0.2,
            "detection_confidence_weight": 0.2,
            "calendar_weight": 0.0,
            "pattern_weight": 0.2,
            "position_weight": 0.0,
            "partial_date_penalty": 0.0,
            "time_like_penalty": 0.0,
            "future_date_bonus": 0.0,
        }
        base_params.update(overrides)
        res = eval_selector(dataset, base_params)
        em = res["exact_match"]
        delta = em - prev_em if results else 0.0
        prev_em = em

        status = "KEEP" if delta >= 0.005 or not results else ("NEUTRAL" if delta == 0.0 else "PRUNE")
        print(f"[{name}] EM: {em*100:.1f}% (Delta: {delta*100:+.1f}%p) | latency: {res['ms_per_image']:.2f}ms | {status}")

        results.append({
            "stage": name,
            "exact_match": em,
            "delta_em": round(delta, 4),
            "latency_ms": res["ms_per_image"],
            "decision": status,
            **overrides,
        })

    # Save report
    md_lines = [
        "# Occam's Razor Feature Ablation Report",
        "",
        f"- Dataset: **{len(dataset)}** images",
        "- Policy: Marginal Delta EM $\\ge +0.5\\%p$ to justify code complexity.",
        "",
        "| Stage | Final EM | Marginal Delta | Latency (ms) | Occam Decision | Added Feature |",
        "|---|---:|---:|---:|:---:|---|",
    ]
    for r in results:
        delta_str = f"{r['delta_em']*100:+.1f}%p" if r["stage"] != "00_bare_regex" else "baseline"
        md_lines.append(
            f"| `{r['stage']}` | **{r['exact_match']*100:.1f}%** | {delta_str} | {r['latency_ms']:.2f}ms | **{r['decision']}** | `{list(r.keys())[-1]}` |"
        )

    (out_dir / "occam_ablation_report.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    with open(out_dir / "occam_ablation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nReport generated at {out_dir / 'occam_ablation_report.md'}")


if __name__ == "__main__":
    main()
