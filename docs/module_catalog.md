# Module catalog

## Preprocessors

| name | purpose | status |
|---|---|---|
| `none` / `passthrough` | default no-op pass-through | ready |
| `resize` | aspect-ratio preserving downscale (`max_size`) | ready |
| `grayscale_contrast` | grayscale conversion with contrast enhancement (`contrast_factor`) | ready |

## OCR backends

| name | purpose | status |
|---|---|---|
| `mock` | offline contract and pipeline smoke test using OCR sidecars | ready |
| `paddle_mobile` | PP-OCRv5 mobile all-in-one adapter with local-weight enforcement | optional reference adapter |
| `paddle_mobile_split` | explicit PP-OCRv5 mobile detector → crop → recognizer adapter with stage timings | selected by Baseline 0; install and pin only when an experiment selects it |
| `paddle_yolo_union` | full-image PP-OCRv5/PP-OCRv6 tokens plus YOLO expiry crop tokens, preserving source and bbox metadata | current SOTA research backend; requires local YOLO and Paddle weights |

## Selection and normalization

| name | purpose |
|---|---|
| `keyword_regex` | all supported date candidate extraction, calendar validation, confidence/context/bbox rule ranking |
| `union_spatial` | train-fitted-style source bonus, confidence logs, and cross-branch bbox IoU ranking | current union selector |
| `date_ko_v1` | year/month/day normalization and `NONE` fallback |

New modules must preserve the contracts in `src/ocr_lab/contracts.py` and document accepted parameters in this file or a linked module note.

## Baseline 0

`configs/experiments/b0_pp_ocr_mobile_raw_rule.yaml` is the CPU-only baseline
candidate. It uses raw images, PP-OCRv5 mobile detection, the Korean mobile
recognizer, rule-based date candidate ranking, and `NoNE` for a missing year.
Run `scripts/run_cpu_benchmark.py` to compare 1 process × 4 threads, 2 × 2,
and 4 × 1 without copying the dataset.
