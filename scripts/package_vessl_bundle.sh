#!/usr/bin/env bash
set -euo pipefail

# Run locally. This packages code and the approved 3,063-label split metadata,
# but deliberately excludes images, weights, historical runs, and the deleted
# 200-label experiment family.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="$ROOT_DIR/runs/vessl"
OUTPUT_FILE="$OUTPUT_DIR/itda_ocr_code_bundle.tar.gz"
mkdir -p "$OUTPUT_DIR"

tar -C "$ROOT_DIR" -czf "$OUTPUT_FILE" \
  --exclude='scripts/colab_*' \
  --exclude='scripts/run_multi_architecture_benchmark.py' \
  --exclude='data/sample/*.ocr.json' \
  src configs scripts tests requirements.txt README.md \
  data/splits data/clean_training_set_3066.csv data/sample

echo "$OUTPUT_FILE"
