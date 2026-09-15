"""Full-image Paddle OCR plus an expiry-region YOLO crop branch.

The backend keeps both sources in one token stream.  The selector can therefore
make a decision with the original full-image context while still benefiting
from a focused expiry crop.  All model paths are explicit and validated so a
missing offline weight fails closed instead of silently downloading a model.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from ..contracts import OCRToken
from .paddle_ocr import (
    PaddleOCRSplitBackend,
    _bbox_from_poly,
    _first_present,
    _result_payload,
)


class PaddleYoloUnionBackend:
    """Combine full-image PP-OCR and expiry-region YOLO crop recognition."""

    def __init__(self, runtime_device: str = "cpu", **params: Any) -> None:
        self._runtime_device = runtime_device
        self._device = "gpu:0" if runtime_device == "cuda" else "cpu"

        full_params = dict(params.pop("full_ocr_params", {}) or {})
        # Allow the common flat names as a convenience for config migration.
        for key in (
            "text_detection_model_name",
            "text_recognition_model_name",
            "text_detection_model_dir",
            "text_recognition_model_dir",
            "det_max_side",
            "det_thresh",
            "box_thresh",
            "unclip_ratio",
            "recognition_batch_size",
            "fallback_enabled",
            "engine",
            "enable_mkldnn",
        ):
            if key in params and key not in full_params:
                full_params[key] = params.pop(key)
        full_params.setdefault("require_local_weights", True)
        self._full_backend = PaddleOCRSplitBackend(
            runtime_device=runtime_device,
            **full_params,
        )

        weights = Path(os.path.expandvars(str(params.pop("yolo_weights")))).expanduser()
        if not weights.is_file():
            raise FileNotFoundError(f"YOLO expiry weight is missing: {weights}")
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - optional runtime
            raise RuntimeError("ultralytics is required for PaddleYoloUnionBackend") from exc
        self._yolo = YOLO(str(weights))
        self._yolo_imgsz = int(params.pop("yolo_imgsz", 960))
        self._yolo_conf = float(params.pop("yolo_conf", 0.20))
        raw_classes = params.pop("yolo_classes", "0")
        if isinstance(raw_classes, str):
            self._yolo_classes = {int(value.strip()) for value in raw_classes.split(",") if value.strip()}
        else:
            self._yolo_classes = {int(value) for value in raw_classes}
        self._yolo_expand = float(params.pop("yolo_expand", 1.0))
        self._yolo_margin = int(params.pop("yolo_margin", 6))
        self._yolo_batch_size = max(1, int(params.pop("yolo_batch_size", 8)))
        self._rec_model_name = str(params.pop("yolo_rec_model_name", "PP-OCRv6_medium_rec"))
        rec_dir = params.pop("yolo_rec_model_dir", full_params.get("text_recognition_model_dir"))
        if rec_dir is None:
            raise ValueError("yolo_rec_model_dir or full_ocr_params.text_recognition_model_dir is required")
        rec_dir_path = Path(os.path.expandvars(str(rec_dir))).expanduser()
        if not rec_dir_path.is_dir():
            raise FileNotFoundError(f"YOLO crop recognizer weight directory is missing: {rec_dir_path}")
        try:
            from paddleocr import TextRecognition
        except ImportError as exc:  # pragma: no cover - optional runtime
            raise RuntimeError("PaddleOCR is required for YOLO crop recognition") from exc
        self._yolo_recognizer = TextRecognition(
            model_name=self._rec_model_name,
            model_dir=str(rec_dir_path),
            device=self._device,
        )
        self.last_stage_metrics: dict[str, Any] = {
            "stage_timing_available": True,
            "detection_ms": None,
            "recognition_ms": None,
            "ocr_ms": None,
            "yolo_detection_ms": None,
            "yolo_recognition_ms": None,
        }

    def extract(self, image_path: Path) -> list[OCRToken]:
        full_tokens = self._full_backend.extract(image_path)
        for token in full_tokens:
            token.extras = {**token.extras, "source": "baseline"}

        image = Image.open(image_path).convert("RGB")
        yolo_device: int | str = 0 if self._runtime_device == "cuda" else "cpu"
        detection_started = time.perf_counter()
        result = self._yolo.predict(
            str(image_path),
            imgsz=self._yolo_imgsz,
            conf=self._yolo_conf,
            device=yolo_device,
            verbose=False,
        )[0]
        yolo_detection_ms = (time.perf_counter() - detection_started) * 1000

        crops: list[np.ndarray] = []
        metas: list[tuple[list[float], float | None]] = []
        for box, cls, conf in zip(
            result.boxes.xyxy.cpu().tolist(),
            result.boxes.cls.cpu().tolist(),
            result.boxes.conf.cpu().tolist(),
        ):
            if int(cls) not in self._yolo_classes:
                continue
            left, top, right, bottom = map(float, box)
            center_x, center_y = (left + right) / 2.0, (top + bottom) / 2.0
            width = (right - left) * self._yolo_expand
            height = (bottom - top) * self._yolo_expand
            crop_left = max(0, int(center_x - width / 2.0) - self._yolo_margin)
            crop_top = max(0, int(center_y - height / 2.0) - self._yolo_margin)
            crop_right = min(image.width, int(center_x + width / 2.0) + self._yolo_margin)
            crop_bottom = min(image.height, int(center_y + height / 2.0) + self._yolo_margin)
            if crop_right <= crop_left or crop_bottom <= crop_top:
                continue
            crop = ImageOps.pad(
                image.crop((crop_left, crop_top, crop_right, crop_bottom)).convert("L"),
                (320, 48),
                color=255,
                centering=(0, 0),
            ).convert("RGB")
            crops.append(np.asarray(crop))
            metas.append(([crop_left, crop_top, crop_right, crop_bottom], float(conf)))

        if not crops:
            full_metrics = self._full_backend.last_stage_metrics or {}
            self.last_stage_metrics = {
                "stage_timing_available": True,
                "detection_ms": full_metrics.get("detection_ms", 0.0) + yolo_detection_ms,
                "recognition_ms": full_metrics.get("recognition_ms", 0.0),
                "yolo_detection_ms": yolo_detection_ms,
                "yolo_recognition_ms": 0.0,
                "ocr_ms": full_metrics.get("ocr_ms", 0.0) + yolo_detection_ms,
            }
            return full_tokens

        recognition_started = time.perf_counter()
        rec_results: list[Any] = []
        for start in range(0, len(crops), self._yolo_batch_size):
            batch = crops[start : start + self._yolo_batch_size]
            rec_results.extend(list(self._yolo_recognizer.predict(batch, batch_size=len(batch))))
        yolo_recognition_ms = (time.perf_counter() - recognition_started) * 1000

        yolo_tokens: list[OCRToken] = []
        for meta, result_item in zip(metas, rec_results):
            payload = _result_payload(result_item)
            texts = _first_present(payload, "rec_texts", "text", "rec_text")
            scores = _first_present(payload, "rec_scores", "scores", "rec_score")
            text = texts[0] if isinstance(texts, (list, tuple)) and texts else texts
            score = scores[0] if isinstance(scores, (list, tuple)) and scores else scores
            if text is None:
                continue
            bbox, detection_confidence = meta
            yolo_tokens.append(
                OCRToken(
                    text=str(text),
                    confidence=float(score or 0.0),
                    bbox=tuple(bbox),
                    extras={
                        "source": "yolo",
                        "detector": "expiry_binary",
                        "center_x": (bbox[0] + bbox[2]) / 2.0,
                        "center_y": (bbox[1] + bbox[3]) / 2.0,
                        "width": bbox[2] - bbox[0],
                        "height": bbox[3] - bbox[1],
                    },
                    detection_confidence=detection_confidence,
                )
            )

        full_metrics = self._full_backend.last_stage_metrics or {}
        self.last_stage_metrics = {
            "stage_timing_available": True,
            "detection_ms": float(full_metrics.get("detection_ms") or 0.0) + yolo_detection_ms,
            "recognition_ms": float(full_metrics.get("recognition_ms") or 0.0) + yolo_recognition_ms,
            "yolo_detection_ms": yolo_detection_ms,
            "yolo_recognition_ms": yolo_recognition_ms,
            "ocr_ms": float(full_metrics.get("ocr_ms") or 0.0) + yolo_detection_ms + yolo_recognition_ms,
        }
        return full_tokens + yolo_tokens
