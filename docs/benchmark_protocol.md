# Measured OCR Benchmark Protocol

## Dataset authority

`data/clean_training_set_3066.csv` is a legacy filename for the **3,063-row filtered Gemini-label dataset**. It is the project-approved working dataset. Its labels must be recorded as `gemini_filtered` in every measured run; do not describe them as human-verified ground truth.

The former 200-image generated-label set and every crop/split derived from it have been removed and are prohibited from future experiments.

## Split policy

| Split | Rows | Use |
|---|---:|---|
| `data/splits/train.csv` | 2,145 | Training only |
| `data/splits/val.csv` | 459 | Architecture selection and internal leaderboard |
| `data/splits/test.csv` | 459 | Frozen final report only |

The CSVs are filename-disjoint. Test execution requires an explicit unlock reason and is excluded from the selection leaderboard.

## Measured-run gate

Every architecture, preprocessing, or detector comparison must run through `scripts/run_measured_experiment.py`.

A rankable validation result requires all of the following:

- Complete coverage of the selected split's real image files
- Local model weights already present before inference
- Fresh end-to-end OCR; token-cache runs are diagnostic-only
- Raw predictions, OCR tokens, metrics, runtime environment, split hash, config hash, and artifact hashes
- Submission-schema validation

`scripts/build_measured_leaderboard.py` ranks only evidence-verified validation runs. It does not ingest historical reports, hard-coded tables, cache-only ablations, or test runs.

## Interpretation

Candidate Recall, Selection Accuracy, runtime, memory, and model size are reported alongside EM. Since the labels are Gemini-filtered, the leaderboard is an internal noisy-label comparison; do not present it as a human-ground-truth score without a separate human audit.
