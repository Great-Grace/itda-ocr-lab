# ITDA OCR Lab

Colab-first, CLI-driven OCR experiment harness for the ITDA competition.

The repository is the shared source of truth. Colab is an execution worker, not the place where the canonical code lives. Team members can use Codex, Claude Code, Antigravity, or plain Colab; every experiment is represented by a versioned YAML config and produces reviewable artifacts.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/run_experiment.py \
  --config configs/baseline_mock.yaml \
  --input data/dev_images \
  --output runs/local_smoke
```

The smoke backend reads optional sidecar files named `<image>.ocr.json`. This lets the pipeline and review flow be tested before a real OCR weight is installed.

## Colab execution

The intended path for GPU work is the Colab CLI. It is deliberately separate from final submission inference:

```bash
colab run --gpu T4 scripts/run_experiment.py \
  --config configs/experiments/my_experiment.yaml \
  --input data/dev_images \
  --output runs/colab_my_experiment
```

The final submission must run with `--device cpu`, local weights, no network download, and the competition environment variables. See `docs/colab_workflow.md` and `docs/agent_workflow.md`.

## Artifacts

Every run writes human-readable and machine-readable results:

```text
predictions.csv       # final_date output
review.csv            # tokens, candidates, selected result, error label
ocr_tokens.jsonl      # cached OCR stage for cheap Selection experiments
metrics.json          # accuracy, latency, NONE rate when labels exist
run_manifest.json     # config, git revision, device, timestamps
```

`runs/` and `weights/` are ignored by Git. Weight provenance belongs in `configs/weights.yaml` and the actual files belong in a release asset or an approved shared storage location.

## Team workflow

1. Create a branch: `feature/<name>-<experiment>`.
2. Add or edit one file under `configs/experiments/`.
3. Run the experiment and inspect `review.csv`/`review.html` when available.
4. Commit code/config plus a short result summary. Never commit weights or raw data.
5. Open a PR. The PR template asks for the exact config, baseline comparison, CPU timing, and representative failures.

The project is intentionally config-first: a new model should be an adapter plus a config, not a rewrite of the pipeline.
