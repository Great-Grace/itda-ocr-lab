# GitHub Copilot & Codex Instructions

This repository is **ITDA OCR Lab**, a modular, config-first OCR experiment harness.

### Operating Principles
- **Modularity**: Never write monolithic pipelines. All functional changes must be modular classes in `src/ocr_lab/modules/` implementing standard contracts in `src/ocr_lab/contracts.py`.
- **Config-First**: Configure experiments via YAML in `configs/experiments/`. Do not edit `src/ocr_lab/pipeline.py`.
- **Preprocessors**: Register in `PREPROCESSORS` in `src/ocr_lab/modules/__init__.py`.
- **OCR Backends**: Register in `OCR_BACKENDS` returning `list[OCRToken]`.
- **Selectors**: Register in `SELECTORS` returning `DateCandidate | None`.
- **Normalizers**: Register in `NORMALIZERS` returning `Prediction`.
- **Offline & CPU Gate**: Final inference must run with `--device cpu`, zero internet calls, and validated with `scripts/check_submission.py`.
- **Refer to [AGENTS.md](../AGENTS.md)** for detailed multi-agent workflows.
