# Current SOTA union runbook

The current research config is `configs/experiments/current-sota-union.yaml`.
It combines full-image PP-OCR tokens with an expiry-only YOLO crop branch and
the `union_spatial` selector.  It is intentionally separate from Baseline 0.

## Local or Colab environment variables

```bash
export ITDA_WEIGHTS_ROOT=/path/to/weights
export ITDA_V6_WEIGHTS_ROOT=/path/to/PP-OCRv6_medium_rec
export ITDA_YOLO_WEIGHTS=/path/to/expiry_binary_yolov8n_1280_best.pt
```

`ITDA_WEIGHTS_ROOT` must contain:

```text
paddle/ppocrv5_mobile_det/
```

The PP-OCRv6 directory may be supplied separately through
`ITDA_V6_WEIGHTS_ROOT`; the YOLO checkpoint is supplied separately through
`ITDA_YOLO_WEIGHTS` so a missing checkpoint fails closed.

## Smoke test

Use the existing mock smoke config for local contract validation.  The union
config itself requires the real offline Paddle and YOLO weights.

```bash
PYTHONPATH=.:src .venv/bin/pytest -q
.venv/bin/python scripts/validate_submission_readiness.py --sample-input data/sample
```

## Measured validation gate

The union backend must be added to the measured runner before its result is
ranked.  Run validation only with `data/splits/val.csv`; keep `test.csv` locked.
The final CPU deployment setting is `device=cpu`, `threads=4`, and
`enable_mkldnn=false` because Paddle 3.3 CPU PIR/oneDNN fails on the target
runtime.  Colab free CPU is useful for compatibility checks but has 2 vCPUs,
so it is not a substitute for the competition's 4-core latency gate.
