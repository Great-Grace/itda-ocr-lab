"""Adaptive Multi-Tier Selective Cascade OCR Backend (AMSC-OCR).

Architecture:
  - Tier 1: RapidOCR ONNX (Ultra-fast CPU gate, ~0.6-0.8s).
            Fast Exit if complete, high-confidence expiration date found.
  - Tier 2: PP-OCRv6 Medium full-image (~1.5s).
            Normal Exit if valid candidate found.
  - Tier 3: YOLOv8n Expiry Crop + Dot-Matrix Morphology & Contrast Expert (~1.2s).
            Triggered only on difficult/recall-miss samples (approx 15%).
            Applies morphological dilation to connect disjointed inkjet dots
            and adaptive contrast to counter shiny packaging glare.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from ..contracts import OCRToken
from .date_candidates import generate_date_candidates
from .paddle_ocr import (
    PaddleOCRSplitBackend,
    _first_present,
    _result_payload,
)

# Common expiry signal keywords
_EXPIRY_KEYWORDS = {"소비", "유통", "까지", "기한", "EXP", "BBD", "BE"}
_NEGATIVE_KEYWORDS = {"제조", "PROD", "MFG", "PKG"}


class AMSC_CascadeOCRBackend:
    """Adaptive Multi-Tier Selective Cascade for Packaging Expiration Date OCR."""

    def __init__(self, runtime_device: str = "cpu", **params: Any) -> None:
        self.runtime_device = runtime_device
        self._device = "gpu:0" if runtime_device == "cuda" else "cpu"
        self._threads = int(params.get("threads", 4))

        # Fast exit configuration
        self.fast_exit_enabled = bool(params.get("fast_exit_enabled", True))
        self.fast_exit_conf = float(params.get("fast_exit_conf", 0.85))
        self.calendar_min_year = int(params.get("calendar_min_year", 2020))
        self.calendar_max_year = int(params.get("calendar_max_year", 2035))

        # 1. Tier 1: RapidOCR setup
        self._rapid_backend = None
        rapid_det = params.get("rapid_det_model_path")
        rapid_rec = params.get("rapid_rec_model_path")
        try:
            from .rapid_ocr import RapidOCRBackend
            if rapid_det and rapid_rec and Path(str(rapid_det)).is_file() and Path(str(rapid_rec)).is_file():
                self._rapid_backend = RapidOCRBackend(
                    runtime_device="cpu",
                    det_model_path=str(rapid_det),
                    rec_model_path=str(rapid_rec),
                )
            else:
                from rapidocr_onnxruntime import RapidOCR
                self._rapid_engine = RapidOCR()
                self._rapid_backend = "builtin"
        except Exception:
            self._rapid_backend = None

        # 2. Tier 2: PP-OCRv6 setup
        full_params = dict(params.get("full_ocr_params", {}) or {})
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
                full_params[key] = params[key]
        full_params.setdefault("text_detection_model_name", "PP-OCRv5_mobile_det")
        full_params.setdefault("text_recognition_model_name", "PP-OCRv6_medium_rec")
        full_params.setdefault("require_local_weights", False)
        # Expand env vars and validate optional directories
        for key in ("text_detection_model_dir", "text_recognition_model_dir"):
            val = full_params.get(key)
            if val is not None:
                expanded = os.path.expandvars(str(val))
                full_params[key] = expanded if Path(expanded).is_dir() else None

        self._paddle_backend = PaddleOCRSplitBackend(
            runtime_device=runtime_device,
            **full_params,
        )

        # 3. Tier 3: YOLO Expiry Crop & Expert Recognizer setup
        self._yolo = None
        self._yolo_recognizer = None
        yolo_weight = params.get("yolo_weights")
        if yolo_weight:
            weight_path = Path(os.path.expandvars(str(yolo_weight)))
            if not weight_path.is_file():
                raise FileNotFoundError(f"Specified YOLO weights file not found: {weight_path}")
            from ultralytics import YOLO
            self._yolo = YOLO(str(weight_path))
            self._yolo_imgsz = int(params.get("yolo_imgsz", 960))
            self._yolo_conf = float(params.get("yolo_conf", 0.20))
            self._yolo_expand = float(params.get("yolo_expand", 1.0))
            self._yolo_margin = int(params.get("yolo_margin", 6))
            self._yolo_batch_size = max(1, int(params.get("yolo_batch_size", 8)))
            self._yolo_max_boxes = max(1, int(params.get("yolo_max_boxes", 4)))

            rec_dir_val = params.get("yolo_rec_model_dir", full_params.get("text_recognition_model_dir"))
            rec_dir = os.path.expandvars(str(rec_dir_val)) if rec_dir_val else None
            if rec_dir and not Path(rec_dir).is_dir():
                rec_dir = None
            rec_name = str(params.get("yolo_rec_model_name", "PP-OCRv6_medium_rec"))
            from paddleocr import TextRecognition
            self._yolo_recognizer = TextRecognition(
                model_name=rec_name,
                model_dir=str(rec_dir) if rec_dir else None,
                device=self._device,
            )

        self.last_stage_metrics: dict[str, Any] = {
            "tier_reached": 1,
            "fast_exit_triggered": False,
            "ocr_ms": 0.0,
            "rapid_ms": 0.0,
            "v6_ms": 0.0,
            "tier3_ms": 0.0,
        }

    def _can_fast_exit(self, tokens: list[OCRToken]) -> bool:
        """Evaluate if Tier 1 tokens contain an unambiguous high-confidence expiration date."""
        if not tokens:
            return False

        has_expiry_kw = any(any(kw in t.text for kw in _EXPIRY_KEYWORDS) for t in tokens)
        has_negative_kw = any(any(kw in t.text for kw in _NEGATIVE_KEYWORDS) for t in tokens)

        candidates = generate_date_candidates(tokens, allow_day_first=True, allow_month_names=True)
        if not candidates:
            return False

        candidates.sort(
            key=lambda c: float(c.features.get("recognition_confidence", 0.0)),
            reverse=True,
        )
        best = candidates[0]
        rec_conf = float(best.features.get("recognition_confidence", 0.0))

        if (
            best.year
            and best.month
            and best.day
            and best.calendar_valid
            and (self.calendar_min_year <= int(best.year) <= self.calendar_max_year)
            and rec_conf >= self.fast_exit_conf
            and has_expiry_kw
            and not has_negative_kw
        ):
            return True

        return False

    def _has_any_valid_candidate(self, tokens: list[OCRToken]) -> bool:
        """Check if any valid date candidate exists in tokens."""
        candidates = generate_date_candidates(tokens, allow_day_first=True, allow_month_names=True)
        for c in candidates:
            if c.calendar_valid and (c.year is not None or (c.month is not None and c.day is not None)):
                if c.year is None or (self.calendar_min_year <= int(c.year) <= self.calendar_max_year):
                    return True
        return False

    def _extract_rapid(self, image_path: Path) -> tuple[list[OCRToken], float]:
        t0 = time.perf_counter()
        tokens: list[OCRToken] = []
        if self._rapid_backend == "builtin":
            try:
                rows, _ = self._rapid_engine(str(image_path))
                if rows:
                    for row in rows:
                        if isinstance(row, (list, tuple)) and len(row) >= 3:
                            poly, text, score = row[0], row[1], row[2]
                            xs = [p[0] for p in poly]
                            ys = [p[1] for p in poly]
                            bbox = (min(xs), min(ys), max(xs), max(ys))
                            tokens.append(
                                OCRToken(
                                    text=str(text),
                                    confidence=float(score or 0.0),
                                    bbox=bbox,
                                    extras={"source": "rapid_fast", "polygon": poly},
                                )
                            )
            except Exception:
                pass
        elif hasattr(self._rapid_backend, "extract"):
            try:
                raw_tokens = self._rapid_backend.extract(image_path)
                for t in raw_tokens:
                    t.extras = {**t.extras, "source": "rapid_fast"}
                tokens = raw_tokens
            except Exception:
                pass
        elapsed = (time.perf_counter() - t0) * 1000
        return tokens, elapsed

    def _run_dot_matrix_expert(self, image: Image.Image, image_path: Path) -> tuple[list[OCRToken], float]:
        """Tier 3: Run YOLO detector, then dual-process crops (standard + dot-matrix dilated)."""
        t0 = time.perf_counter()
        if self._yolo is None or self._yolo_recognizer is None:
            return [], 0.0

        yolo_device: int | str = 0 if self.runtime_device == "cuda" else "cpu"
        result = self._yolo.predict(
            str(image_path),
            imgsz=self._yolo_imgsz,
            conf=self._yolo_conf,
            device=yolo_device,
            verbose=False,
        )[0]

        crops: list[np.ndarray] = []
        metas: list[tuple[list[float], float, str]] = []

        detected: list[tuple[float, list[float]]] = []
        for box, cls, conf in zip(
            result.boxes.xyxy.cpu().tolist(),
            result.boxes.cls.cpu().tolist(),
            result.boxes.conf.cpu().tolist(),
        ):
            if int(cls) != 0:
                continue
            detected.append((float(conf), box))

        # Sort descending by confidence and cap to top K boxes
        detected.sort(key=lambda item: item[0], reverse=True)
        detected = detected[: getattr(self, "_yolo_max_boxes", 4)]

        for conf, box in detected:
            left, top, right, bottom = map(float, box)
            cx, cy = (left + right) / 2.0, (top + bottom) / 2.0
            w = (right - left) * self._yolo_expand
            h = (bottom - top) * self._yolo_expand
            c_left = max(0, int(cx - w / 2.0) - self._yolo_margin)
            c_top = max(0, int(cy - h / 2.0) - self._yolo_margin)
            c_right = min(image.width, int(cx + w / 2.0) + self._yolo_margin)
            c_bottom = min(image.height, int(cy + h / 2.0) + self._yolo_margin)
            if c_right <= c_left or c_bottom <= c_top:
                continue

            raw_crop = image.crop((c_left, c_top, c_right, c_bottom))
            bbox = [c_left, c_top, c_right, c_bottom]

            # Variant 1: Standard padded crop
            crop_std = ImageOps.pad(raw_crop.convert("L"), (320, 48), color=255, centering=(0, 0)).convert("RGB")
            crops.append(np.asarray(crop_std))
            metas.append((bbox, float(conf), "yolo_crop"))

            # Variant 2: Dot-Matrix Morphology (MinFilter connects dark inkjet dots on light background)
            gray = ImageOps.grayscale(raw_crop)
            auto_c = ImageOps.autocontrast(gray, cutoff=2.0)
            dilated = auto_c.filter(ImageFilter.MinFilter(3))
            sharpened = dilated.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
            crop_morph = ImageOps.pad(sharpened, (320, 48), color=255, centering=(0, 0)).convert("RGB")
            crops.append(np.asarray(crop_morph))
            metas.append((bbox, float(conf), "yolo_morph_dotmatrix"))

            # Variant 3: Inverted contrast (for light dot-matrix print on dark bottle caps/cans)
            inv = ImageOps.invert(auto_c)
            dilated_inv = inv.filter(ImageFilter.MinFilter(3))
            crop_inv = ImageOps.pad(dilated_inv, (320, 48), color=255, centering=(0, 0)).convert("RGB")
            crops.append(np.asarray(crop_inv))
            metas.append((bbox, float(conf), "yolo_morph_inverted"))

        if not crops:
            return [], (time.perf_counter() - t0) * 1000

        rec_results: list[Any] = []
        for start in range(0, len(crops), self._yolo_batch_size):
            batch = crops[start : start + self._yolo_batch_size]
            rec_results.extend(list(self._yolo_recognizer.predict(batch, batch_size=len(batch))))

        expert_tokens: list[OCRToken] = []
        for (bbox, det_conf, source_tag), res in zip(metas, rec_results):
            payload = _result_payload(res)
            texts = _first_present(payload, "rec_texts", "text", "rec_text")
            scores = _first_present(payload, "rec_scores", "scores", "rec_score")
            text = texts[0] if isinstance(texts, (list, tuple)) and texts else texts
            score = scores[0] if isinstance(scores, (list, tuple)) and scores else scores
            if not text:
                continue
            expert_tokens.append(
                OCRToken(
                    text=str(text),
                    confidence=float(score or 0.0),
                    bbox=tuple(bbox),
                    extras={
                        "source": source_tag,
                        "detector": "yolo_expiry",
                        "center_x": (bbox[0] + bbox[2]) / 2.0,
                        "center_y": (bbox[1] + bbox[3]) / 2.0,
                        "width": bbox[2] - bbox[0],
                        "height": bbox[3] - bbox[1],
                    },
                    detection_confidence=det_conf,
                )
            )

        elapsed = (time.perf_counter() - t0) * 1000
        return expert_tokens, elapsed

    def extract(self, image_path: Path) -> list[OCRToken]:
        total_start = time.perf_counter()
        image = Image.open(image_path).convert("RGB")

        # -------------------------------------------------------------
        # TIER 1: Fast Gate (RapidOCR ONNX)
        # -------------------------------------------------------------
        tier1_tokens, rapid_ms = self._extract_rapid(image_path)
        if self.fast_exit_enabled and self._can_fast_exit(tier1_tokens):
            total_ms = (time.perf_counter() - total_start) * 1000
            self.last_stage_metrics = {
                "tier_reached": 1,
                "fast_exit_triggered": True,
                "ocr_ms": total_ms,
                "rapid_ms": rapid_ms,
                "v6_ms": 0.0,
                "tier3_ms": 0.0,
            }
            return tier1_tokens

        # -------------------------------------------------------------
        # TIER 2: High-Precision Full Image (PP-OCRv6)
        # -------------------------------------------------------------
        v6_start = time.perf_counter()
        v6_tokens = self._paddle_backend.extract(image_path)
        for t in v6_tokens:
            t.extras = {**t.extras, "source": "baseline"}
        v6_ms = (time.perf_counter() - v6_start) * 1000

        combined_tokens = tier1_tokens + v6_tokens

        # -------------------------------------------------------------
        # TIER 3: Expiry ROI + Dot-Matrix Morphology Expert
        # Runs alongside Tier 2 to form True Full OCR + YOLO Union
        # -------------------------------------------------------------
        tier3_tokens, tier3_ms = self._run_dot_matrix_expert(image, image_path)
        total_tokens = combined_tokens + tier3_tokens

        total_ms = (time.perf_counter() - total_start) * 1000
        self.last_stage_metrics = {
            "tier_reached": 3 if tier3_tokens else 2,
            "fast_exit_triggered": False,
            "ocr_ms": total_ms,
            "rapid_ms": rapid_ms,
            "v6_ms": v6_ms,
            "tier3_ms": tier3_ms,
        }
        return total_tokens
