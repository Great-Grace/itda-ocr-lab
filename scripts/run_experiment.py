#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ocr_lab.config import load_config
from ocr_lab.pipeline import run_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one config-first OCR experiment.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--labels")
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--device", choices=["cpu", "cuda"], help="Override runtime device for this run")
    parser.add_argument("--tokens-cache", help="Path to ocr_tokens.jsonl to reuse previously extracted OCR tokens")
    parser.add_argument("--image-ids-file", help="Optional newline-delimited image stems to evaluate")
    parser.add_argument("--threads", type=int, help="Override CPU thread count for this process")
    parser.add_argument("--shard-index", type=int, default=0, help="Shard index for multi-process CPU benchmarking")
    parser.add_argument("--shard-count", type=int, default=1, help="Number of shards for multi-process CPU benchmarking")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.device:
        config.setdefault("runtime", {})["device"] = args.device
    if args.threads:
        config.setdefault("runtime", {})["threads"] = args.threads
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[key] = str(args.threads)
    metrics = run_pipeline(
        config,
        args.input,
        args.output,
        args.labels,
        args.max_images,
        tokens_cache_path=args.tokens_cache,
        image_ids_path=args.image_ids_file,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
