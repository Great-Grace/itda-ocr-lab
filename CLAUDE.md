# Claude Code Agent Guidelines

Read and strictly adhere to [AGENTS.md](AGENTS.md).

### Key Rules for Claude Code
1. **Config-First & Modularity**: Do not edit `src/ocr_lab/pipeline.py` to change experiments. Create or edit files under `configs/experiments/`. Keep all module contracts in `src/ocr_lab/contracts.py` stable.
2. **Smoke Test First**: Before expensive execution, run quick smoke verification on `data/sample/`:
   ```bash
   python scripts/run_experiment.py --config configs/baseline_mock.yaml --input data/sample --output runs/smoke_check --labels data/sample/labels.csv
   ```
3. **Submission Gate**: Always validate generated predictions with `python scripts/check_submission.py runs/<name>/predictions.csv`.
4. **Token Caching**: Use `--tokens-cache <prev_tokens.jsonl>` for selector/normalizer iterations to avoid redundant OCR.
5. **Non-Developer Friendly**: Summarize results clearly using `summary.md` and `review.html` generated in the run directory.
