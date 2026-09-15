# Baseline 0 — PP-OCR mobile / raw image / rule selector

This baseline is deliberately diagnostic rather than SOTA-oriented:

```text
raw image
  -> PP-OCRv5_mobile_det
  -> korean_PP-OCRv5_mobile_rec
  -> date candidate generation + calendar validation
  -> confidence/context/bbox rule ranking
  -> date normalization
```

The config is `configs/experiments/b0_pp_ocr_mobile_raw_rule.yaml`.
Inference never downloads weights. Put the detector and recognizer directories
under the mounted dataset's `weights/` directory, register their provenance in
`configs/weights.yaml`, and set `ITDA_WEIGHTS_ROOT` before running.

The checked-in B0 settings are `det_max_side=1280`, `det_thresh=0.25`,
`box_thresh=0.50`, `unclip_ratio=1.8`, recognition batch size 8, CPU device,
four runtime threads, and oneDNN/MKLDNN requested. Recognition scores are not
filtered before date parsing. Every token sidecar retains text, recognition and
detection confidence, polygon/bbox, center, width, and height. Low-confidence
images can take an isolated 1600px detector retry with crop-only contrast,
sharpening, and upscale; the retry is guarded so a Paddle runtime incompatibility
cannot crash the main inference.

The complete detector/threshold/batch ablation grid is defined in
`configs/experiments/b0_cpu_ablation.yaml` (3 max-sides × 3 detector thresholds
× 3 box thresholds × 4 recognition batches = 108 runs). It is intended to run
on a fixed screen/dev split, never on the lock split.

## Commands

Smoke-test the contracts without PaddleOCR:

```bash
.venv/bin/python scripts/run_experiment.py \
  --config configs/experiments/b0_smoke_mock.yaml \
  --input data/sample \
  --labels data/sample/labels.csv \
  --output runs/b0_smoke \
  --device cpu
.venv/bin/python scripts/check_submission.py runs/b0_smoke/predictions.csv
```

Run the real baseline on a mounted/available dataset:

```bash
.venv/bin/python scripts/run_cpu_benchmark.py \
  --config configs/experiments/b0_pp_ocr_mobile_raw_rule.yaml \
  --input "$ITDA_INPUT_DIR" \
  --labels "$ITDA_LABELS_PATH" \
  --output runs/B0_PP_OCR_MOBILE_RAW_RULE
```

The `paddle_mobile_split` adapter times detection and recognition separately;
`metrics.json` reports final exact match, candidate recall, candidate
selection accuracy, date-component accuracy, optional CER/string accuracy,
per-stage latency, throughput, peak RSS, and local model size. If the OCR
backend does not expose separate detector/recognizer timings, those two fields
remain null and the combined `ocr_ms` is reported explicitly.

The error labels are diagnostic heuristics when only final-date labels exist:
`DET`, `REC`, `GEN`, `SEL`, `NORM`, and `AMBIGUOUS / UNKNOWN`. True CER and
recognition exact-string accuracy require an optional `ocr_text` or `text`
column in the labels file.

The current team-facing table and caveats are in `runs/benchmark_review.md`.
The latest fresh B0 screen32 run is stored under
`runs/B0_CURRENT_screen32/B0_CURRENT_screen32`; its submission file passes
`check_submission.py`.

For selective second-pass experiments, `keyword_regex` also accepts
`source_confidence_multipliers`. This is a calibrated prior for auxiliary
recognizers such as `svtr_local_roi`; it downweights their evidence without
discarding their candidates or changing the pretrained model dictionary.
