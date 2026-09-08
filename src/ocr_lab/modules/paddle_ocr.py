from __future__ import annotations

from pathlib import Path
from typing import Any

from ..contracts import OCRToken


class PaddleOCRBackend:
    """Adapter for installed PaddleOCR versions.

    Model-specific keyword arguments stay in the experiment config. The adapter
    intentionally does not download weights; callers must provide local model
    directories in `params` for offline/reproducible runs.
    """

    def __init__(self, **params: Any) -> None:
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise RuntimeError(
                "PaddleOCR is not installed. Install the chosen adapter's requirements first."
            ) from exc
        self._engine = PaddleOCR(**params)

    def extract(self, image_path: Path) -> list[OCRToken]:
        result = self._engine.predict(str(image_path)) if hasattr(self._engine, "predict") else self._engine.ocr(str(image_path))
        return _parse_paddle_result(result)


def _parse_paddle_result(result: Any) -> list[OCRToken]:
    tokens: list[OCRToken] = []
    if result is None:
        return tokens
    items = result if isinstance(result, list) else [result]
    for page in items:
        data = page if isinstance(page, dict) else getattr(page, "json", lambda: {})()
        if not isinstance(data, dict):
            continue
        texts = data.get("rec_texts") or data.get("text") or []
        scores = data.get("rec_scores") or data.get("scores") or []
        boxes = data.get("rec_polys") or data.get("dt_polys") or data.get("boxes") or []
        for index, text in enumerate(texts):
            score = float(scores[index]) if index < len(scores) else 0.0
            box = boxes[index] if index < len(boxes) else None
            bbox = None
            if box:
                xs = [float(point[0]) for point in box]
                ys = [float(point[1]) for point in box]
                bbox = (min(xs), min(ys), max(xs), max(ys))
            tokens.append(OCRToken(text=str(text), confidence=score, bbox=bbox))
    return tokens
