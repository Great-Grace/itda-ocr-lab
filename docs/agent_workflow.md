# Agent workflow

The human-facing request can be a paper PDF, a paper link, or a short idea. The agent must convert it into a small, reviewable experiment before running expensive work.

## Required intake fields

- hypothesis: what paper idea is being tested
- target: preprocess, OCR, candidate selection, normalizer, or runtime
- baseline config
- data split and label path
- metric and acceptance threshold
- GPU type and time budget
- CPU-only validation requirement

If a field is missing, ask for it or apply the project default and state the default. Do not silently invent a weight, dataset split, or metric.

## Agent command sequence

```text
read AGENTS.md and docs/module_catalog.md
inspect the baseline config
write a new config under configs/experiments/
run a small smoke test first
run GPU training/ablation only after the smoke test passes
run CPU validation on the winning candidate
write a result summary and representative failures
```

The agent may edit its feature branch. It must not merge to `main`, publish credentials, commit weights, or claim success without `metrics.json` and `check_submission.py` output.

## Ablation

An ablation card can define a small Cartesian grid. Keep it small enough to finish on the chosen accelerator:

```yaml
ablation:
  max_runs: 4
  grid:
    selector.params.keyword_weight: [0.5, 0.7]
    selector.params.confidence_weight: [0.2, 0.4]
```

Run it with:

```bash
python scripts/run_ablation.py \
  --config configs/experiments/my_experiment.yaml \
  --input data/dev_images \
  --labels data/dev_labels.csv \
  --output runs/my_ablation
```
