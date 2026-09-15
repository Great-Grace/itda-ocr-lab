#!/usr/bin/env python3
"""High-speed parallel grid search harness for selector & normalizer parameters.

Leverages cached OCR tokens to evaluate thousands of candidate ranking combinations
in seconds without re-running heavy neural OCR backends.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import multiprocessing as mp
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

# Default Grid Space - Refined by Occam's Razor
DEFAULT_GRID = {
    "keyword_weight": [0.3, 0.5, 0.7, 0.9],
    "negative_keyword_weight": [0.15, 0.3, 0.45, 0.6],
    "calendar_weight": [0.3, 0.5, 0.8],
    "pattern_weight": [0.1, 0.2, 0.3],
    "position_weight": [0.0, 0.15, 0.25],
    "partial_date_penalty": [0.2, 0.35, 0.5],
    "time_like_penalty": [0.3, 0.5, 0.7],
    "future_date_bonus": [0.0, 0.05],
}


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


def evaluate_combination(args: tuple[dict[str, Any], list[dict[str, Any]]]) -> dict[str, Any]:
    params, dataset = args
    selector = KeywordRegexSelector(
        allow_day_first=True,
        allow_month_names=True,
        **params,
    )
    normalizer = DateNormalizer()

    total_images = len(dataset)
    correct_count = 0
    candidate_coverage_hits = 0
    candidate_coverage_total = 0
    none_predictions = 0

    for item in dataset:
        img_id = item["image_id"]
        gt = item["gt"]
        tokens = item["tokens"]

        cands = selector.candidates(tokens)
        has_gt = False
        gt_parsed = parse_final_date(gt)
        if gt != "NONE" and gt_parsed is not None:
            candidate_coverage_total += 1
            for c in cands:
                cand_str = f"{c.year or 'NoNE'}-{c.month}-{c.day or 'None'}"
                if cand_str == gt or parse_final_date(cand_str) == gt_parsed:
                    has_gt = True
                    break
            if has_gt:
                candidate_coverage_hits += 1

        selected = selector.select_candidates(cands, tokens)
        prediction = normalizer.normalize(img_id, selected).final_date
        if prediction == "NONE":
            none_predictions += 1

        if prediction == gt or (gt_parsed is not None and parse_final_date(prediction) == gt_parsed):
            correct_count += 1

    exact_match = (correct_count / total_images) if total_images else 0.0
    recall = (candidate_coverage_hits / candidate_coverage_total) if candidate_coverage_total else 0.0
    selection_acc = (correct_count / candidate_coverage_hits) if candidate_coverage_hits else 0.0

    return {
        "final_date_exact_match": round(exact_match, 4),
        "candidate_recall": round(recall, 4),
        "candidate_selection_accuracy": round(selection_acc, 4),
        "correct_count": correct_count,
        "total_images": total_images,
        "none_rate": round(none_predictions / total_images, 4) if total_images else 0.0,
        **params,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run parallel hyperparameter grid search on cached tokens.")
    parser.add_argument("--tokens-cache", required=True, help="Path to ocr_tokens.jsonl")
    parser.add_argument("--labels", required=True, help="Path to labels.csv")
    parser.add_argument("--output", default="runs/grid_search", help="Directory to save leaderboard")
    parser.add_argument("--max-combinations", type=int, default=2000, help="Maximum parameter combinations to try")
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count()), help="Number of parallel worker processes")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from tokens: {args.tokens_cache} and labels: {args.labels}...")
    dataset, labels = load_dataset(args.tokens_cache, args.labels)
    print(f"Loaded {len(dataset)} labeled images.")

    keys = list(DEFAULT_GRID.keys())
    value_lists = [DEFAULT_GRID[k] for k in keys]
    all_combos = [dict(zip(keys, v)) for v in itertools.product(*value_lists)]
    if len(all_combos) > args.max_combinations:
        # Sample deterministically
        step = len(all_combos) // args.max_combinations
        selected_combos = all_combos[::step][:args.max_combinations]
    else:
        selected_combos = all_combos

    print(f"Evaluating {len(selected_combos)} parameter combinations using {args.workers} workers...")
    t0 = time.time()
    work_items = [(combo, dataset) for combo in selected_combos]
    with mp.Pool(processes=args.workers) as pool:
        results = pool.map(evaluate_combination, work_items)
    elapsed = time.time() - t0
    print(f"Evaluated {len(results)} combinations in {elapsed:.2f} seconds ({len(results)/elapsed:.1f} combos/sec).")

    # Sort leaderboard by exact match descending, then candidate selection accuracy descending
    results.sort(key=lambda r: (r["final_date_exact_match"], r["candidate_selection_accuracy"]), reverse=True)

    # Save full CSV
    csv_path = output_dir / "grid_leaderboard.csv"
    if results:
        fields = list(results[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(results)

    # Save Top-20 Markdown leaderboard
    md_path = output_dir / "grid_leaderboard.md"
    top20 = results[:20]
    lines = [
        "# Hyperparameter Grid Search Leaderboard",
        "",
        f"- Total Combinations Evaluated: **{len(results):,}**",
        f"- Evaluation Time: **{elapsed:.2f}s** ({len(results)/elapsed:.1f} combos/sec)",
        f"- Evaluated Images: **{len(dataset)}**",
        "",
        "## Top 20 Parameter Configurations",
        "",
        "| Rank | Final EM | Coverage | Selection | keyword | neg_kw | cal | pattern | pos | partial_pen | time_pen | future_bonus |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, r in enumerate(top20, start=1):
        lines.append(
            f"| {rank} | **{r['final_date_exact_match']*100:.1f}%** | {r['candidate_recall']*100:.1f}% | {r['candidate_selection_accuracy']*100:.1f}% | "
            f"{r['keyword_weight']} | {r['negative_keyword_weight']} | {r['calendar_weight']} | {r['pattern_weight']} | {r['position_weight']} | "
            f"{r['partial_date_penalty']} | {r['time_like_penalty']} | {r['future_date_bonus']} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    best = results[0]
    print(f"\n[BEST CONFIGURATION FOUND]")
    print(f"Final EM: {best['final_date_exact_match']*100:.2f}% | Candidate Recall: {best['candidate_recall']*100:.2f}% | Selection Acc: {best['candidate_selection_accuracy']*100:.2f}%")
    print(f"Parameters: {json.dumps({k: best[k] for k in keys}, indent=2)}")

    # Dump best YAML config
    best_yaml_path = output_dir / "best_grid_config.yaml"
    yaml_content = f"""name: BEST_GRID_SEARCH_BASELINE
runtime:
  device: cpu
  threads: 4
ocr:
  plugin: paddle_mobile_split
  weight_id: pp-ocrv6-medium-rec-official
  params:
    require_local_weights: true
    text_detection_model_name: PP-OCRv5_mobile_det
    text_recognition_model_name: PP-OCRv6_medium_rec
    text_detection_model_dir: ${{ITDA_WEIGHTS_ROOT}}/paddle/ppocrv5_mobile_det
    text_recognition_model_dir: ${{ITDA_V6_WEIGHTS_ROOT}}
    det_max_side: 960
    det_thresh: 0.25
    box_thresh: 0.60
    unclip_ratio: 1.8
    recognition_batch_size: 1
    fallback_enabled: false
    engine: paddle
    enable_mkldnn: false
preprocess:
  plugin: none
  params: {{}}
selector:
  plugin: keyword_regex
  params:
    allow_day_first: true
    allow_month_names: true
    keyword_weight: {best['keyword_weight']}
    negative_keyword_weight: {best['negative_keyword_weight']}
    confidence_weight: 0.2
    detection_confidence_weight: 0.2
    calendar_weight: {best['calendar_weight']}
    pattern_weight: {best['pattern_weight']}
    position_weight: {best['position_weight']}
    partial_date_penalty: {best['partial_date_penalty']}
    time_like_penalty: {best['time_like_penalty']}
    future_date_bonus: {best['future_date_bonus']}
normalizer:
  plugin: date_ko_v1
  params:
    year_missing_token: NoNE
    none_token: NONE
    day_missing_token: None
"""
    best_yaml_path.write_text(yaml_content, encoding="utf-8")
    print(f"Saved best config to {best_yaml_path}")
    print(f"Leaderboard saved to {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
