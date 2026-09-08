# Colab CLI workflow

Colab CLI is an optional execution backend. The repository remains runnable locally with the same command.

## GPU development

```bash
colab run --gpu T4 scripts/run_experiment.py \
  --config configs/experiments/paddle_mobile.yaml \
  --input data/dev_images \
  --labels data/dev_labels.csv \
  --output runs/paddle_mobile_t4
```

## CPU gate

```bash
python scripts/run_experiment.py \
  --config configs/experiments/paddle_mobile.yaml \
  --device cpu \
  --input data/dev_images \
  --labels data/dev_labels.csv \
  --output runs/paddle_mobile_cpu

python scripts/check_submission.py runs/paddle_mobile_cpu/predictions.csv
```

GPU runs are for exploration or training. The final candidate must use local weights and pass the CPU path without downloading anything during inference.
