# ITDA OCR Lab agent rules

- Treat `configs/` as the experiment interface. Do not edit the pipeline to change a normal experiment.
- Keep module contracts in `src/ocr_lab/contracts.py` stable.
- Every experiment must record its config, Git revision, device, weight identifiers, and latency.
- GPU is for development/training. Every candidate must pass the CPU-only validation path.
- Never download weights during final inference. Never call external APIs during inference.
- Do not commit raw images, labels, weights, credentials, or `runs/` artifacts.
- Before claiming success, run `python scripts/check_submission.py` on the generated CSV.
- New OCR backends must return `OCRToken` objects and expose model-specific settings under `params`.
