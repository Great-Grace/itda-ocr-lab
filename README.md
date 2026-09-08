# ITDA OCR Lab

Colab-first, CLI-driven OCR experiment harness for the ITDA competition.

The repository is the shared source of truth. Colab is an execution worker, not the place where the canonical code lives. Team members can use Codex, Claude Code, Antigravity, or plain Colab; every experiment is represented by a versioned YAML config and produces reviewable artifacts.

The harness is model- and weight-agnostic. `mock` is only the built-in smoke backend; PaddleOCR is only an optional reference adapter. A team experiment selects its own adapter, approved weight ID, and model-specific parameters through its config.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Instant 1-second smoke test using bundled sample fixture
python scripts/run_experiment.py \
  --config configs/baseline_mock.yaml \
  --input data/sample \
  --output runs/local_smoke \
  --labels data/sample/labels.csv
```

The smoke backend reads optional sidecar files named `<image>.ocr.json`. This lets the entire pipeline, selection scoring, and visual review flow be tested instantly without downloading heavy OCR weights.

## Colab & Notebook execution

1. **Interactive GPU Experimentation**:
   Open `notebooks/run_on_colab.ipynb` directly in Google Colab to mount Google Drive, pull repository changes, run GPU experiments, and execute the CPU compliance gate.
2. **Headless CLI Execution**:
   ```bash
   colab run --gpu T4 scripts/run_experiment.py \
     --config configs/experiments/my_experiment.yaml \
     --input data/dev_images \
     --output runs/colab_my_experiment
   ```
3. **Official Competition Submission**:
   Use `notebooks/predict.ipynb` for the official offline, CPU-only evaluation complying with ITDA competition rules.

## Artifacts

Every run writes human-readable and machine-readable results:

```text
predictions.csv       # final_date output formatted for submission
review.csv            # tokens, candidates, selected result, error label
summary.md            # non-developer friendly KPI summary and failure analysis
review.html           # interactive single-file offline visual dashboard
ocr_tokens.jsonl      # cached OCR stage for fast Selection iterations (--tokens-cache)
metrics.json          # accuracy, latency, NONE rate when labels exist
run_manifest.json     # config, git revision, device, timestamps
architecture.md       # human-readable snapshot of modules & parameters
```


`runs/` and `weights/` are ignored by Git. Weight provenance belongs in `configs/weights.yaml` and the actual files belong in a release asset or an approved shared storage location.

## Team workflow

1. Create a branch: `feature/<name>-<experiment>`.
2. Add or edit one file under `configs/experiments/`.
3. Run the experiment and inspect `review.csv`/`review.html` when available.
4. Commit code/config plus a short result summary. Never commit weights or raw data.
5. Open a PR. The PR template asks for the exact config, baseline comparison, CPU timing, and representative failures.

The project is intentionally config-first: a new model should be an adapter plus a config, not a rewrite of the pipeline.
