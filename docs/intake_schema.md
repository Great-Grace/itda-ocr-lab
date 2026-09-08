# Experiment intake

An agent should turn a natural-language request into a small YAML request before running work. `scripts/intake_experiment.py` applies safe defaults and returns a question list when a required decision is missing.

```yaml
hypothesis: "MSR-style resizing reduces date recognition errors under aspect-ratio distortion"
target: preprocess
baseline_config: configs/experiments/paddle_mobile.yaml
input_dir: data/dev_images
labels_path: data/dev_labels.csv
output_name: msr_resize_ablation
metric: final_date_exact_match
gpu_type: T4
max_gpu_hours: 2
cpu_required: true
```

Check before execution:

```bash
python scripts/intake_experiment.py configs/experiments/request.yaml \
  --output configs/experiments/resolved_request.yaml
```

The agent should not silently fill in `hypothesis`, `target`, `input_dir`, or `output_name`. It should ask a follow-up question for those fields. Defaults are allowed for metric, GPU type, budget, and the CPU gate, but the resolved plan must show them.
