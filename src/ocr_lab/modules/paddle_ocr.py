from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ..contracts import OCRToken


class PaddleOCRBackend:
    """Adapter for installed PaddleOCR versions.

    Model-specific keyword arguments stay in the experiment config. The adapter
    intentionally does not download weights; callers must provide local model
    directories in `params` for offline/reproducible runs.
    """

    def __init__(self, runtime_device: str = "cpu", **params: Any) -> None:
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise RuntimeError(
                "PaddleOCR is not installed. Install the chosen adapter's requirements first."
            ) from exc
        params = _expand_paths(params)
        require_local_weights = bool(params.pop("require_local_weights", True))
        _validate_local_weights(params, require_local_weights)

        # PaddleOCR v3 uses `device`; older v2 releases use `use_gpu`.
        # The top-level runtime device is authoritative for team experiments.
        requested_device = "gpu:0" if runtime_device == "cuda" else "cpu"
        params.pop("use_gpu", None)
        params["device"] = requested_device
        try:
            self._engine = PaddleOCR(**params)
        except TypeError as exc:
            if "device" not in str(exc):
                raise
            legacy_params = dict(params)
            legacy_params.pop("device", None)
            legacy_params["use_gpu"] = runtime_device == "cuda"
            self._engine = PaddleOCR(**legacy_params)

    def extract(self, image_path: Path) -> list[OCRToken]:
        result = self._engine.predict(str(image_path)) if hasattr(self._engine, "predict") else self._engine.ocr(str(image_path))
        return _parse_paddle_result(result)


def _parse_paddle_result(result: Any) -> list[OCRToken]:
    tokens: list[OCRToken] = []
    if result is None:
        return tokens
    items = result if isinstance(result, list) else [result]
    for page in items:
        if isinstance(page, list):
            tokens.extend(_parse_paddle_v2_page(page))
            continue
        data = page if isinstance(page, dict) else getattr(page, "json", lambda: {})()
        if not isinstance(data, dict):
            continue
        texts = data.get("rec_texts") or data.get("text") or []
        scores = data.get("rec_scores") or data.get("scores") or []
        boxes = data.get("rec_polys") or data.get("dt_polys") or data.get("boxes") or []
        for index, text in enumerate(texts):
            score = float(scores[index]) if index < len(scores) else 0.0
            box = boxes[index] if index < len(boxes) else None
            tokens.append(OCRToken(text=str(text), confidence=score, bbox=_bbox_from_poly(box)))
    return tokens


def _parse_paddle_v2_page(page: list[Any]) -> list[OCRToken]:
    tokens: list[OCRToken] = []
    for line in page:
        if not isinstance(line, (list, tuple)) or len(line) < 2:
            continue
        box, recognition = line[0], line[1]
        if not isinstance(recognition, (list, tuple)) or not recognition:
            continue
        text = recognition[0]
        score = float(recognition[1]) if len(recognition) > 1 else 0.0
        tokens.append(OCRToken(text=str(text), confidence=score, bbox=_bbox_from_poly(box)))
    return tokens


def _bbox_from_poly(box: Any) -> tuple[float, float, float, float] | None:
    if box is None:
        return None
    if hasattr(box, "tolist"):
        box = box.tolist()
    if not box:
        return None
    xs = [float(point[0]) for point in box]
    ys = [float(point[1]) for point in box]
    return (min(xs), min(ys), max(xs), max(ys))


def _expand_paths(params: dict[str, Any]) -> dict[str, Any]:
    expanded = dict(params)
    for key, value in expanded.items():
        if key.endswith(("_dir", "_path")) and isinstance(value, str):
            expanded[key] = os.path.expandvars(value)
    return expanded


def _validate_local_weights(params: dict[str, Any], required: bool) -> None:
    if not required:
        return
    detector_keys = ("det_model_dir", "text_detection_model_dir")
    recognizer_keys = ("rec_model_dir", "text_recognition_model_dir")
    missing = []
    for label, keys in (("detector", detector_keys), ("recognizer", recognizer_keys)):
        configured = next((params[key] for key in keys if params.get(key)), None)
        if not configured or not Path(configured).is_dir():
            missing.append(label)
    if missing:
        raise RuntimeError(
            "Local PaddleOCR weights are required for reproducible inference; "
            f"missing or invalid: {', '.join(missing)}. "
            "Set det_model_dir/rec_model_dir (or v3 aliases) to existing local directories."
        )
