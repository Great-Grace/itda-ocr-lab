from __future__ import annotations

import re
import tempfile
import time
from pathlib import Path
from typing import Any

from ..contracts import OCRToken
from .preprocessor import AdaptiveContrastPreprocessor

_DATEISH = re.compile(r"\d{2,4}\s*[.\-/년]\s*\d{1,2}|\d{6,8}")


class CascadeOCRBackend:
    """Conditional 2-pass OCR cascade.

    Pass 1: Runs primary OCR on raw image.
    Check: If any token has date-like patterns, return immediately (fast path).
    Pass 2: If no date-like pattern detected, apply adaptive contrast (CLAHE/unsharp)
            and re-run OCR on the preprocessed image.
    """

    def __init__(self, primary_backend: str = "mock", primary_params: dict[str, Any] | None = None, **params: Any) -> None:
        from . import OCR_BACKENDS
        self.primary_backend_name = primary_backend
        backend_cls = OCR_BACKENDS.get(primary_backend)
        if backend_cls is None:
            raise ValueError(f"Unknown primary OCR backend: {primary_backend}")
        primary_config = dict(primary_params or params)
        # Cascade-only controls must not leak into the underlying OCR adapter.
        primary_config.pop("cutoff", None)
        self.primary = backend_cls(**primary_config)
        if params.get("retry_preprocessor") == "clahe_local":
            from .preprocessor import LocalClahePreprocessor
            self.preprocessor = LocalClahePreprocessor(
                clip_limit=params.get("clip_limit", 2.0),
                tile_grid_size=params.get("tile_grid_size", 8),
            )
        else:
            self.preprocessor = AdaptiveContrastPreprocessor(cutoff=params.get("cutoff", 2.0), unsharp=True)
        self.last_stage_metrics: dict[str, Any] = {
            "stage_timing_available": False,
            "detection_ms": None,
            "recognition_ms": None,
            "ocr_ms": None,
            "cascade_pass2_triggered": False,
        }

    def extract(self, image_path: Path) -> list[OCRToken]:
        t0 = time.perf_counter()
        tokens = self.primary.extract(image_path)
        has_date = any(_DATEISH.search(t.text) for t in tokens)

        if has_date:
            elapsed = (time.perf_counter() - t0) * 1000
            base_metrics = getattr(self.primary, "last_stage_metrics", {}) or {}
            self.last_stage_metrics = {
                **base_metrics,
                "ocr_ms": elapsed,
                "cascade_pass2_triggered": False,
            }
            return tokens

        # Conditional 2-pass retry on difficult images
        with tempfile.TemporaryDirectory() as tmp_dir:
            prep_path = self.preprocessor.preprocess(image_path, Path(tmp_dir))
            retry_tokens = self.primary.extract(prep_path)

        elapsed = (time.perf_counter() - t0) * 1000
        base_metrics = getattr(self.primary, "last_stage_metrics", {}) or {}
        self.last_stage_metrics = {
            **base_metrics,
            "ocr_ms": elapsed,
            "cascade_pass2_triggered": True,
        }
        retry_has_date = any(_DATEISH.search(t.text) for t in retry_tokens)
        return retry_tokens if retry_has_date else (tokens or retry_tokens)
