# Measured benchmark harness

This is the only supported route for a new internal leaderboard row. It is designed to fail closed: an incomplete image set, missing local weights, failed submission validation, cache-provenance mismatch, or modified artifact excludes a result from ranking.

## Label policy

`data/clean_training_set_3066.csv` contains **3,063** accepted Gemini-generated labels. They are the working dataset authorized for this project, but their provenance is recorded as `gemini_filtered`, not human ground truth.

- Training may use `data/splits/train.csv`.
- Configuration selection and internal comparison may use `data/splits/val.csv`.
- `data/splits/test.csv` is frozen. It requires `--unlock-test --reason "..."` and is never included in the internal model-selection leaderboard.
- The deleted 200-image generated-label experiment set must not be recreated, cited, or mixed into any measured result.

## Required inputs

The split CSVs contain filenames and labels, not image bytes. Before a real run, provide a local image directory containing every filename in the selected split and local model directories referenced by the config. The harness refuses partial coverage and never downloads weights.

## Run a fresh validation experiment

First run the read-only coverage check. It must report `"ready": true`:

```bash
ITDA_WEIGHTS_ROOT=/absolute/path/to/weights \
ITDA_V6_WEIGHTS_ROOT=/absolute/path/to/PP-OCRv6_medium_rec \
python scripts/preflight_measured_dataset.py \
  --config configs/experiments/best_cpu_baseline_v6_box60.yaml \
  --input /absolute/path/to/3063-images \
  --split-csv data/splits/val.csv
```

Then execute the measured run:

```bash
ITDA_WEIGHTS_ROOT=/absolute/path/to/weights \
ITDA_V6_WEIGHTS_ROOT=/absolute/path/to/PP-OCRv6_medium_rec \
python scripts/run_measured_experiment.py \
  --name ppocrv6-box60-raw \
  --config configs/experiments/best_cpu_baseline_v6_box60.yaml \
  --input /absolute/path/to/3063-images \
  --tier val \
  --label-provenance gemini_filtered
```

The run records a normalized label file, input ID hash, config hash, runtime metadata, raw predictions, OCR tokens, metrics, logs, and a submission check under `runs/measured/<name>_val/`.

## Build the internal leaderboard

```bash
python scripts/build_measured_leaderboard.py
```

Only fresh validation runs whose `evidence.json` hashes still match `predictions.csv`, `ocr_tokens.jsonl`, and `metrics.json` are included. Selector-only cache ablations and every test-tier run remain unranked diagnostics.
