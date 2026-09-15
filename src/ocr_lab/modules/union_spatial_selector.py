"""Train-fitted-style ranking for union OCR candidates."""

from __future__ import annotations

import math
from typing import Iterable

from ..contracts import DateCandidate, OCRToken
from .regex_selector import KeywordRegexSelector


def _iou(left: list[float] | tuple[float, ...] | None, right: list[float] | tuple[float, ...] | None) -> float:
    if not left or not right:
        return 0.0
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_left = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    area_right = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return inter / max(1e-9, area_left + area_right - inter)


class UnionSpatialSelector(KeywordRegexSelector):
    """Rank date candidates from baseline and YOLO token sources."""

    def __init__(self, **params: object) -> None:
        self.yolo_bonus = float(params.pop("yolo_bonus", 1.5))
        self.candidate_score_weight = float(params.pop("candidate_score_weight", 1.5))
        self.recognition_log_weight = float(params.pop("recognition_log_weight", 1.5))
        self.detection_log_weight = float(params.pop("detection_log_weight", 0.5))
        self.bbox_iou_weight = float(params.pop("bbox_iou_weight", 0.5))
        super().__init__(**params)

    def select_candidates(self, candidates: Iterable[DateCandidate], tokens: Iterable[OCRToken]) -> DateCandidate | None:
        candidate_list = list(candidates)
        token_list = list(tokens)
        baseline_boxes = [candidate_box(token_list, c) for c in candidate_list if candidate_source(token_list, c) == "baseline"]
        yolo_boxes = [candidate_box(token_list, c) for c in candidate_list if candidate_source(token_list, c) == "yolo"]

        def rank(candidate: DateCandidate) -> float:
            source = candidate_source(token_list, candidate)
            box = candidate_box(token_list, candidate)
            counterpart_boxes = yolo_boxes if source == "baseline" else baseline_boxes
            spatial = max((_iou(box, other) for other in counterpart_boxes), default=0.0)
            rec = max(float(candidate.features.get("recognition_confidence", 0.0)), 1e-6)
            det = max(float(candidate.features.get("detection_confidence", 0.0)), 1e-6)
            return (
                (self.yolo_bonus if source == "yolo" else 0.0)
                + self.candidate_score_weight * float(candidate.score)
                + self.recognition_log_weight * math.log(rec)
                + self.detection_log_weight * math.log(det)
                + self.bbox_iou_weight * spatial
            )

        return max(candidate_list, key=rank, default=None)


def candidate_source(tokens: list[OCRToken], candidate: DateCandidate) -> str:
    sources = {str(tokens[i].extras.get("source", "baseline")) for i in candidate.token_indices if i < len(tokens)}
    return "yolo" if sources == {"yolo"} else "baseline"


def candidate_box(tokens: list[OCRToken], candidate: DateCandidate) -> list[float] | None:
    boxes = [tokens[i].bbox for i in candidate.token_indices if i < len(tokens) and tokens[i].bbox is not None]
    if not boxes:
        return None
    return [min(box[0] for box in boxes), min(box[1] for box in boxes), max(box[2] for box in boxes), max(box[3] for box in boxes)]
