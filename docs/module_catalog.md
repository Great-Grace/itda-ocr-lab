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
| `paddle_mobile` | optional local PaddleOCR adapter | adapter included; install and pin PaddleOCR separately |

## Selection and normalization

| name | purpose |
|---|---|
| `keyword_regex` | date-like candidate extraction plus expiry keyword scoring |
| `date_ko_v1` | year/month/day normalization and `NONE` fallback |

New modules must preserve the contracts in `src/ocr_lab/contracts.py` and document accepted parameters in this file or a linked module note.
