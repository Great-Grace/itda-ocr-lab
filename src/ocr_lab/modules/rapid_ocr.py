from __future__ import annotations

import time
import os
from pathlib import Path
from typing import Any

from ..contracts import OCRToken


class RapidOCRBackend:
    """Offline RapidOCR/ONNX Runtime adapter.

    RapidOCR returns ``[polygon, text, confidence]`` rows and exposes elapsed
    detection/classification/recognition timings. Model paths are mandatory so
    inference cannot silently download weights.
    """

    def __init__(self, runtime_device: str = "cpu", **params: Any) -> None:
        if runtime_device != "cpu":
            raise ValueError("RapidOCRBackend is CPU-only")
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as exc:  # pragma: no cover - optional runtime
            raise RuntimeError("rapidocr-onnxruntime is required for rapid_onnx") from exc
        det_model_path = Path(os.path.expandvars(str(params.pop("det_model_path")))).expanduser()
        rec_model_path = Path(os.path.expandvars(str(params.pop("rec_model_path")))).expanduser()
        for path in (det_model_path, rec_model_path):
            if not path.is_file():
                raise FileNotFoundError(f"RapidOCR local model is missing: {path}")
        params.setdefault("use_angle_cls", False)
        params["det_model_path"] = str(det_model_path)
        params["rec_model_path"] = str(rec_model_path)
        self._engine = RapidOCR(**params)
        self.last_stage_metrics: dict[str, Any] = {
            "stage_timing_available": True,
            "detection_ms": None,
            "recognition_ms": None,
        }

    def extract(self, image_path: Path) -> list[OCRToken]:
        started = time.perf_counter()
        rows, elapsed = self._engine(str(image_path))
        total_ms = (time.perf_counter() - started) * 1000
        rows = rows or []
        # RapidOCR reports [det_ms, cls_ms, rec_ms] in seconds.
        detection_ms = float(elapsed[0]) * 1000 if isinstance(elapsed, (list, tuple)) and len(elapsed) > 0 else None
        recognition_ms = float(elapsed[2]) * 1000 if isinstance(elapsed, (list, tuple)) and len(elapsed) > 2 else None
        tokens: list[OCRToken] = []
        for row in rows:
            if not isinstance(row, (list, tuple)) or len(row) < 3:
                continue
            polygon, text, score = row[0], row[1], row[2]
            bbox = _bbox(polygon)
            extras: dict[str, Any] = {"polygon": polygon, "source": "rapid_onnx"}
            if bbox:
                left, top, right, bottom = bbox
                extras.update({"center_x": (left + right) / 2, "center_y": (top + bottom) / 2,
                               "width": right - left, "height": bottom - top})
            tokens.append(OCRToken(text=str(text), confidence=float(score or 0.0), bbox=bbox,
                                   extras=extras, detection_confidence=None))
        self.last_stage_metrics = {
            "stage_timing_available": True,
            "detection_ms": detection_ms,
            "recognition_ms": recognition_ms,
            "ocr_ms": total_ms,
        }
        return tokens


def _bbox(polygon: Any) -> tuple[float, float, float, float] | None:
    try:
        points = [(float(point[0]), float(point[1])) for point in polygon]
        if not points:
            return None
        xs, ys = zip(*points)
        return min(xs), min(ys), max(xs), max(ys)
    except (TypeError, ValueError, IndexError):
        return None
