from __future__ import annotations

import os
import time
import json
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
        self.last_stage_metrics: dict[str, Any] = {
            "stage_timing_available": False,
            "detection_ms": None,
            "recognition_ms": None,
        }

    def extract(self, image_path: Path) -> list[OCRToken]:
        started = time.perf_counter()
        result = self._engine.predict(str(image_path)) if hasattr(self._engine, "predict") else self._engine.ocr(str(image_path))
        elapsed_ms = (time.perf_counter() - started) * 1000
        self.last_stage_metrics = {
            "stage_timing_available": False,
            "detection_ms": None,
            "recognition_ms": None,
            "ocr_ms": elapsed_ms,
        }
        return _parse_paddle_result(result)


class PaddleOCRSplitBackend:
    """Explicit detector -> crop -> recognizer adapter with stage timings."""

    def __init__(self, runtime_device: str = "cpu", **params: Any) -> None:
        try:
            from paddleocr import TextDetection, TextRecognition
        except ImportError as exc:
            raise RuntimeError(
                "PaddleOCR is not installed. Install the chosen adapter's requirements first."
            ) from exc
        from PIL import Image  # noqa: F401 - validates the split adapter dependency

        params = _expand_paths(params)
        require_local_weights = bool(params.pop("require_local_weights", True))
        _validate_local_weights(params, require_local_weights)
        requested_device = "gpu:0" if runtime_device == "cuda" else "cpu"
        det_kwargs = {
            "model_name": params.pop("text_detection_model_name", "PP-OCRv5_mobile_det"),
            "model_dir": params.pop("text_detection_model_dir", params.pop("det_model_dir", None)),
            "device": requested_device,
        }
        rec_kwargs = {
            "model_name": params.pop("text_recognition_model_name", "korean_PP-OCRv5_mobile_rec"),
            "model_dir": params.pop("text_recognition_model_dir", params.pop("rec_model_dir", None)),
            "device": requested_device,
        }
        for key in ("input_shape", "text_recognition_input_shape", "rec_input_shape"):
            if key in params:
                rec_kwargs["input_shape"] = params.pop(key)
                break
        # TextDetection exposes these controls directly. Keep them in the
        # experiment config so threshold/side-length ablations do not require
        # changing pipeline code.
        for source, target in (
            ("det_max_side", "limit_side_len"),
            ("text_det_limit_side_len", "limit_side_len"),
            ("det_thresh", "thresh"),
            ("box_thresh", "box_thresh"),
            ("unclip_ratio", "unclip_ratio"),
            ("text_det_limit_type", "limit_type"),
        ):
            if source in params:
                det_kwargs[target] = params.pop(source)
        det_kwargs.setdefault("limit_type", "max")
        recognition_batch_size = int(params.pop("recognition_batch_size", params.pop("rec_batch_size", 8)))
        self._box_priority = str(params.pop("box_priority", "none"))
        self._max_recognition_boxes = int(params.pop("max_recognition_boxes", 0))
        if self._box_priority not in {"none", "aspect_desc"}:
            raise ValueError("box_priority must be 'none' or 'aspect_desc'")
        self._crop_mode = str(params.pop("crop_mode", "axis_aligned"))
        if self._crop_mode not in {"axis_aligned", "perspective"}:
            raise ValueError("crop_mode must be 'axis_aligned' or 'perspective'")
        self._fallback_enabled = bool(params.pop("fallback_enabled", True))
        self._fallback_max_side = int(params.pop("fallback_max_side", 1600))
        self._fallback_confidence = float(params.pop("fallback_confidence", 0.20))
        self._detector_kwargs = dict(det_kwargs)
        for key in ("engine", "enable_hpi", "use_tensorrt", "enable_mkldnn", "cpu_threads"):
            if key in params:
                det_kwargs[key] = params[key]
                rec_kwargs[key] = params[key]
        self._detector = _construct_with_optional_kwargs(TextDetection, det_kwargs)
        self._recognizer = _construct_with_optional_kwargs(TextRecognition, rec_kwargs)
        self._recognition_batch_size = max(1, recognition_batch_size)
        self._fallback_detector = None
        self.last_stage_metrics: dict[str, Any] = {
            "stage_timing_available": True,
            "detection_ms": None,
            "recognition_ms": None,
        }

    def extract(self, image_path: Path) -> list[OCRToken]:
        import numpy as np
        from PIL import Image

        detection_started = time.perf_counter()
        detection_result = list(self._detector.predict(str(image_path), batch_size=1))
        detection_ms = (time.perf_counter() - detection_started) * 1000
        payload = _result_payload(detection_result[0]) if detection_result else {}
        polygons = _first_present(payload, "dt_polys", "polys", "boxes")
        detection_scores = _first_present(payload, "dt_scores", "det_scores")
        source = np.asarray(Image.open(image_path).convert("RGB"))
        crop_items = []
        for index, polygon in enumerate(polygons):
            crop = _crop_polygon(source, polygon, self._crop_mode)
            if crop is not None:
                detection_score = detection_scores[index] if index < len(detection_scores) else None
                crop_items.append((polygon, detection_score, crop))
        selected_items = list(enumerate(crop_items))
        if self._box_priority == "aspect_desc":
            selected_items.sort(key=lambda item: _aspect_ratio(item[1][0]), reverse=True)
        if self._max_recognition_boxes > 0:
            selected_items = selected_items[:self._max_recognition_boxes]
        crops = [item[1][2] for item in selected_items]
        if not crops:
            self.last_stage_metrics = {"stage_timing_available": True, "detection_ms": detection_ms, "recognition_ms": 0.0, "ocr_ms": detection_ms}
            return []
        recognition_started = time.perf_counter()
        recognition_results = []
        for start in range(0, len(crops), self._recognition_batch_size):
            recognition_results.extend(list(self._recognizer.predict(
                crops[start:start + self._recognition_batch_size],
                batch_size=min(self._recognition_batch_size, len(crops) - start),
            )))
        recognition_ms = (time.perf_counter() - recognition_started) * 1000
        tokens: list[OCRToken] = []
        for index, result in enumerate(recognition_results):
            rec_payload = _result_payload(result)
            texts = _first_present(rec_payload, "rec_texts", "text", "rec_text")
            scores = _first_present(rec_payload, "rec_scores", "scores", "rec_score")
            text = texts[0] if isinstance(texts, (list, tuple)) and texts else texts
            score = scores[0] if isinstance(scores, (list, tuple)) and scores else scores
            if text is None:
                continue
            source_index = selected_items[index][0]
            polygon, detection_score = crop_items[source_index][0], crop_items[source_index][1]
            bbox = _bbox_from_poly(polygon)
            extras = {"polygon": polygon.tolist() if hasattr(polygon, "tolist") else polygon}
            if bbox is not None:
                left, top, right, bottom = bbox
                extras.update({"center_x": (left + right) / 2.0, "center_y": (top + bottom) / 2.0,
                               "width": right - left, "height": bottom - top})
            extras["source_detection_index"] = source_index
            tokens.append(OCRToken(text=str(text), confidence=float(score or 0.0), bbox=bbox, extras=extras, detection_confidence=float(detection_score) if detection_score is not None else None))
        tokens.sort(key=lambda token: int(token.extras.get("source_detection_index", 0)))
        fallback_used = False
        if self._fallback_enabled and (not tokens or sum(token.confidence for token in tokens) / len(tokens) < self._fallback_confidence):
            fallback_tokens, fallback_detection_ms, fallback_recognition_ms = self._extract_fallback(image_path)
            if fallback_tokens:
                tokens = fallback_tokens
                detection_ms += fallback_detection_ms
                recognition_ms += fallback_recognition_ms
                fallback_used = True
        self.last_stage_metrics = {"stage_timing_available": True, "detection_ms": detection_ms, "recognition_ms": recognition_ms, "ocr_ms": detection_ms + recognition_ms, "fallback_used": fallback_used}
        return tokens

    def _extract_fallback(self, image_path: Path) -> tuple[list[OCRToken], float, float]:
        """Retry only low-confidence images with a larger detector side."""

        import numpy as np
        from PIL import Image, ImageEnhance, ImageFilter

        kwargs = dict(self._detector_kwargs)
        kwargs["limit_side_len"] = self._fallback_max_side
        try:
            detector = _construct_with_optional_kwargs(type(self._detector), kwargs)
        except Exception:
            return [], 0.0, 0.0
        started = time.perf_counter()
        try:
            results = list(detector.predict(str(image_path), batch_size=1))
        except Exception:
            # A fallback must never turn a low-confidence image into a whole
            # pipeline failure (notably on Paddle/PIR versions with an
            # incompatible oneDNN graph).
            return [], (time.perf_counter() - started) * 1000, 0.0
        detection_ms = (time.perf_counter() - started) * 1000
        payload = _result_payload(results[0]) if results else {}
        polygons = _first_present(payload, "dt_polys", "polys", "boxes")
        scores = _first_present(payload, "dt_scores", "det_scores")
        source = np.asarray(Image.open(image_path).convert("RGB"))
        crops, metas = [], []
        for index, polygon in enumerate(polygons):
            crop = _crop_polygon(source, polygon, self._crop_mode)
            if crop is None:
                continue
            # Enhancement is performed in grayscale, then restored to RGB.
            # PaddleOCR v3 recognizers expect an HxWx3 array even for
            # visually grayscale text crops.
            image = Image.fromarray(crop).convert("L")
            image = ImageEnhance.Contrast(image).enhance(1.15).filter(ImageFilter.SHARPEN)
            image = image.resize((max(8, image.width * 2), max(8, image.height * 2))).convert("RGB")
            crops.append(np.asarray(image))
            metas.append((polygon, scores[index] if index < len(scores) else None))
        if not crops:
            return [], detection_ms, 0.0
        started = time.perf_counter()
        results = []
        for start in range(0, len(crops), self._recognition_batch_size):
            results.extend(list(self._recognizer.predict(crops[start:start + self._recognition_batch_size], batch_size=min(self._recognition_batch_size, len(crops) - start))))
        recognition_ms = (time.perf_counter() - started) * 1000
        tokens: list[OCRToken] = []
        for index, result in enumerate(results):
            payload = _result_payload(result)
            texts = _first_present(payload, "rec_texts", "text", "rec_text")
            scores_rec = _first_present(payload, "rec_scores", "scores", "rec_score")
            text = texts[0] if isinstance(texts, (list, tuple)) and texts else texts
            score = scores_rec[0] if isinstance(scores_rec, (list, tuple)) and scores_rec else scores_rec
            if text is None:
                continue
            polygon, detection_score = metas[index]
            bbox = _bbox_from_poly(polygon)
            extras = {"polygon": polygon.tolist() if hasattr(polygon, "tolist") else polygon, "source": "fallback"}
            if bbox is not None:
                left, top, right, bottom = bbox
                extras.update({"center_x": (left + right) / 2.0, "center_y": (top + bottom) / 2.0, "width": right - left, "height": bottom - top})
            tokens.append(OCRToken(text=str(text), confidence=float(score or 0.0), bbox=bbox, extras=extras, detection_confidence=float(detection_score) if detection_score is not None else None))
        return tokens, detection_ms, recognition_ms


def _parse_paddle_result(result: Any) -> list[OCRToken]:
    tokens: list[OCRToken] = []
    if result is None:
        return tokens
    items = result if isinstance(result, list) else [result]
    for page in items:
        if isinstance(page, list):
            tokens.extend(_parse_paddle_v2_page(page))
            continue
        data = _result_payload(page)
        if not isinstance(data, dict):
            continue
        texts = _first_present(data, "rec_texts", "text")
        scores = _first_present(data, "rec_scores", "scores")
        boxes = _first_present(data, "rec_polys", "dt_polys", "boxes")
        detection_scores = _first_present(data, "dt_scores", "det_scores", "text_detection_scores")
        for index, text in enumerate(texts):
            score = float(scores[index]) if index < len(scores) else 0.0
            box = boxes[index] if index < len(boxes) else None
            detection_score = float(detection_scores[index]) if index < len(detection_scores) else None
            bbox = _bbox_from_poly(box)
            extras = {"polygon": box.tolist() if hasattr(box, "tolist") else box}
            if bbox is not None:
                left, top, right, bottom = bbox
                extras.update({"center_x": (left + right) / 2.0, "center_y": (top + bottom) / 2.0,
                               "width": right - left, "height": bottom - top})
            tokens.append(OCRToken(text=str(text), confidence=score, bbox=bbox, extras=extras, detection_confidence=detection_score))
    return tokens


def _result_payload(result: Any) -> dict[str, Any]:
    data = result if isinstance(result, dict) else getattr(result, "json", lambda: {})()
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            data = {}
    if isinstance(data, dict) and isinstance(data.get("res"), dict):
        data = data["res"]
    return data if isinstance(data, dict) else {}


def _crop_bbox(image: Any, polygon: Any) -> Any:
    if hasattr(polygon, "tolist"):
        polygon = polygon.tolist()
    if not polygon:
        return None
    xs = [int(point[0]) for point in polygon]
    ys = [int(point[1]) for point in polygon]
    height, width = image.shape[:2]
    left, right = max(0, min(xs)), min(width, max(xs) + 1)
    top, bottom = max(0, min(ys)), min(height, max(ys) + 1)
    if right <= left or bottom <= top:
        return None
    return image[top:bottom, left:right]


def _aspect_ratio(polygon: Any) -> float:
    bbox = _bbox_from_poly(polygon)
    if bbox is None:
        return 0.0
    width = max(1.0, bbox[2] - bbox[0])
    height = max(1.0, bbox[3] - bbox[1])
    return width / height


def _crop_polygon(image: Any, polygon: Any, mode: str = "axis_aligned") -> Any:
    """Crop a detected text polygon, optionally rectifying its perspective."""

    if mode == "axis_aligned":
        return _crop_bbox(image, polygon)
    try:
        import cv2
        import numpy as np
        points = np.asarray(polygon.tolist() if hasattr(polygon, "tolist") else polygon, dtype=np.float32)
        if points.shape != (4, 2):
            return _crop_bbox(image, polygon)
        # Order arbitrary DB polygon points as top-left, top-right,
        # bottom-right, bottom-left before the perspective transform.
        ordered = np.zeros((4, 2), dtype=np.float32)
        sums, diffs = points.sum(axis=1), np.diff(points, axis=1).reshape(-1)
        ordered[0], ordered[2] = points[np.argmin(sums)], points[np.argmax(sums)]
        ordered[1], ordered[3] = points[np.argmin(diffs)], points[np.argmax(diffs)]
        width = max(1, int(max(np.linalg.norm(ordered[1] - ordered[0]), np.linalg.norm(ordered[2] - ordered[3]))))
        height = max(1, int(max(np.linalg.norm(ordered[3] - ordered[0]), np.linalg.norm(ordered[2] - ordered[1]))))
        destination = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
        crop = cv2.warpPerspective(image, cv2.getPerspectiveTransform(ordered, destination), (width, height), borderValue=(255, 255, 255))
        if crop is not None and crop.shape[0] > crop.shape[1] * 1.5:
            crop = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return crop
    except Exception:
        return _crop_bbox(image, polygon)


def _first_present(data: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = data.get(key)
        if value is not None:
            return value
    return []


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
        bbox = _bbox_from_poly(box)
        extras = {"polygon": box.tolist() if hasattr(box, "tolist") else box}
        if bbox is not None:
            left, top, right, bottom = bbox
            extras.update({"center_x": (left + right) / 2.0, "center_y": (top + bottom) / 2.0,
                           "width": right - left, "height": bottom - top})
        tokens.append(OCRToken(text=str(text), confidence=score, bbox=bbox, extras=extras, detection_confidence=None))
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


def _construct_with_optional_kwargs(factory: Any, kwargs: dict[str, Any]) -> Any:
    """Construct PaddleOCR objects, retrying without version-specific options."""

    try:
        return factory(**kwargs)
    except TypeError as exc:
        optional = ("limit_type", "limit_side_len", "thresh", "box_thresh", "unclip_ratio",
                    "text_det_limit_type", "text_det_limit_side_len", "text_det_thresh",
                    "text_det_box_thresh", "text_det_unclip_ratio", "input_shape")
        message = str(exc)
        trimmed = dict(kwargs)
        removed = False
        for key in optional:
            if key in trimmed and (key in message or "unexpected keyword" in message):
                trimmed.pop(key)
                removed = True
        if not removed:
            raise
        return factory(**trimmed)
