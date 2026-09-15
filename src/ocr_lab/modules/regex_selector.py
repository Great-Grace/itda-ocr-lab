from __future__ import annotations

import math
import re
from typing import Iterable

from ..contracts import DateCandidate, OCRToken
from .date_candidates import generate_date_candidates
from .text_sanitizer import sanitize_tokens


POSITIVE_KEYWORDS = (
    "소비기한",
    "유통기한",
    "품질유지기한",
    "exp",
    "expiry",
    "exp date",
    "best before",
    "use by",
    "bbe",
    "bb",
    "까지",
)
NEGATIVE_KEYWORDS = ("제조일", "제조일자", "제조", "mfg", "mfd", "prod", "lot", "부터")


class KeywordRegexSelector:
    """Generate date candidates and rank them with interpretable features."""

    def __init__(self, **params: object) -> None:
        self.keyword_weight = float(params.get("keyword_weight", 0.7))
        self.negative_keyword_weight = float(params.get("negative_keyword_weight", 0.45))
        self.confidence_weight = float(params.get("confidence_weight", 0.2))
        self.detection_confidence_weight = float(params.get("detection_confidence_weight", 0.2))
        self.calendar_weight = float(params.get("calendar_weight", 0.5))
        self.pattern_weight = float(params.get("pattern_weight", 0.25))
        self.position_weight = float(params.get("position_weight", 0.25))
        self.same_line_bonus = float(params.get("same_line_bonus", 0.15))
        self.keep_invalid = bool(params.get("keep_invalid", False))
        self.allow_day_first = bool(params.get("allow_day_first", False))
        self.allow_month_names = bool(params.get("allow_month_names", False))
        self.sanitization_mode = str(params.get("sanitization_mode", "none"))
        raw_priors = params.get("source_confidence_multipliers", {}) or {}
        self.source_confidence_multipliers = {str(key): float(value) for key, value in raw_priors.items()} if isinstance(raw_priors, dict) else {}
        self.partial_date_penalty = float(params.get("partial_date_penalty", 0.45))
        self.time_like_penalty = float(params.get("time_like_penalty", 0.60))
        self.future_date_bonus = float(params.get("future_date_bonus", 0.15))

    def candidates(self, tokens: Iterable[OCRToken]) -> list[DateCandidate]:
        from datetime import date
        token_list = list(tokens)
        parser_tokens = sanitize_tokens(token_list, self.sanitization_mode)
        candidates = generate_date_candidates(
            parser_tokens,
            keep_invalid=self.keep_invalid,
            allow_day_first=self.allow_day_first,
            allow_month_names=self.allow_month_names,
        )
        token_dates: dict[str, date] = {}
        for c in candidates:
            if c.calendar_valid and c.year and c.month and c.day:
                try:
                    d_val = date(int(c.year), int(c.month), int(c.day))
                    if c.raw_text not in token_dates or c.pattern.startswith("ymd"):
                        token_dates[c.raw_text] = d_val
                except (ValueError, TypeError):
                    pass
        comp_dates = list(token_dates.values())
        max_complete_date = max(comp_dates) if len(set(comp_dates)) > 1 else None
        min_complete_date = min(comp_dates) if len(set(comp_dates)) > 1 else None

        for candidate in candidates:
            self._score(
                candidate,
                parser_tokens,
                has_complete=bool(comp_dates),
                max_complete_date=max_complete_date,
                min_complete_date=min_complete_date,
            )
            if candidate.pattern == "dmy_separated" and "/" not in candidate.raw_text:
                candidate.score -= 0.6
            elif candidate.pattern == "dmy_separated" and "/" in candidate.raw_text and int(candidate.year or 0) <= 2026:
                candidate.score += 0.4

        return sorted(candidates, key=lambda item: item.score, reverse=True)

    def select_candidates(self, candidates: Iterable[DateCandidate], tokens: Iterable[OCRToken]) -> DateCandidate | None:
        ranked = sorted(candidates, key=lambda item: item.score, reverse=True)
        return ranked[0] if ranked else None

    def select(self, tokens: Iterable[OCRToken]) -> DateCandidate | None:
        token_list = list(tokens)
        return self.select_candidates(self.candidates(token_list), token_list)

    def _score(
        self,
        candidate: DateCandidate,
        tokens: list[OCRToken],
        *,
        has_complete: bool = False,
        max_complete_date: object = None,
        min_complete_date: object = None,
    ) -> None:
        from datetime import date
        recognition = float(candidate.features.get("recognition_confidence", 0.0))
        detection = float(candidate.features.get("detection_confidence", 0.0))
        source_prior = self._source_prior(candidate, tokens)
        recognition *= source_prior
        pattern_reliability = {
            "ymd_korean": 1.0,
            "ymd_separated": 1.0,
            "ymd_lot_attached": 0.96,
            "dmy_separated": 0.92,
            "mdy_separated": 0.90,
            "dmy_lot_attached": 0.92,
            "ymd_spaced": 0.88,
            "dmy_spaced": 0.86,
            "ymd_mixed_compact": 0.84,
            "mdy_month_name": 0.95,
            "dmy_month_name": 0.92,
            "ym_korean": 0.78,
            "ym_separated": 0.74,
            "my_separated": 0.72,
            "ymd_compact": 0.9,
            "md_korean": 0.72,
            "md_separated": 0.70,
        }.get(candidate.pattern, 0.5)
        positive, negative, spatial = self._context_features(candidate, tokens)
        calendar_valid = 1.0 if candidate.calendar_valid else -1.0
        score = (
            self.confidence_weight * recognition
            + self.detection_confidence_weight * detection
            + self.calendar_weight * calendar_valid
            + self.pattern_weight * pattern_reliability
            + self.keyword_weight * positive
            - self.negative_keyword_weight * negative
            + self.position_weight * spatial
        )
        partial_penalty = self.partial_date_penalty if has_complete and not candidate.year else 0.0
        time_like_penalty = self.time_like_penalty if re.fullmatch(r"\d{1,2}:\d{2}", candidate.raw_text.strip()) else 0.0
        score -= partial_penalty + time_like_penalty

        # If multiple complete dates exist, packaged food expiry is later than manufacture date
        if max_complete_date and candidate.year and candidate.month and candidate.day:
            try:
                cand_d = date(int(candidate.year), int(candidate.month), int(candidate.day))
                if cand_d == max_complete_date:
                    score += (self.future_date_bonus * source_prior)
                    candidate.evidence.append("later_expiry_date")
                elif min_complete_date and cand_d == min_complete_date and positive == 0:
                    score -= (self.future_date_bonus * 0.5 * source_prior)
            except (ValueError, TypeError):
                pass

        candidate.score = score
        candidate.features.update(
            {
                "pattern_reliability": pattern_reliability,
                "calendar_valid": calendar_valid,
                "positive_keyword": positive,
                "negative_keyword": negative,
                "spatial_relation": spatial,
                "source_prior": source_prior,
                "partial_date_penalty": partial_penalty,
                "time_like_penalty": time_like_penalty,
                "future_date_bonus": self.future_date_bonus if max_complete_date else 0.0,
            }
        )
        if candidate.calendar_valid:
            candidate.evidence.append("calendar_valid")
        else:
            candidate.evidence.append("calendar_invalid")
        if positive > 0:
            candidate.evidence.append("expiry_keyword")
        if negative > 0:
            candidate.evidence.append("manufacturing_keyword")
        if spatial > 0:
            candidate.evidence.append("bbox_spatial_relation")

    def _source_prior(self, candidate: DateCandidate, tokens: list[OCRToken]) -> float:
        if not self.source_confidence_multipliers or not candidate.token_indices:
            return 1.0
        priors = []
        for index in candidate.token_indices:
            if index >= len(tokens):
                continue
            source = str(tokens[index].extras.get("source", ""))
            priors.append(self.source_confidence_multipliers.get(source, 1.0))
        return sum(priors) / len(priors) if priors else 1.0

    def _context_features(self, candidate: DateCandidate, tokens: list[OCRToken]) -> tuple[float, float, float]:
        if not candidate.token_indices:
            return 0.0, 0.0, 0.0
        candidate_index = candidate.token_indices[0]
        candidate_token = tokens[candidate_index] if candidate_index < len(tokens) else None
        positive = 0.0
        negative = 0.0
        spatial = 0.0
        for index, token in enumerate(tokens):
            text = token.text.lower().strip()
            matched_positive = any(keyword in text for keyword in POSITIVE_KEYWORDS)
            matched_negative = any(keyword in text for keyword in NEGATIVE_KEYWORDS)
            if not matched_positive and not matched_negative:
                continue
            distance = abs(index - candidate_index)
            proximity = 1.0 / (1.0 + distance)
            relation = self._spatial_relation(token, candidate_token)
            if matched_positive:
                positive = max(positive, proximity + relation)
            if matched_negative:
                negative = max(negative, proximity + relation)
            spatial = max(spatial, relation)
        return min(1.5, positive), min(1.5, negative), min(1.5, spatial)

    def _spatial_relation(self, keyword: OCRToken, candidate: OCRToken | None) -> float:
        if keyword.bbox is None or candidate is None or candidate.bbox is None:
            return 0.0
        kx = (keyword.bbox[0] + keyword.bbox[2]) / 2.0
        ky = (keyword.bbox[1] + keyword.bbox[3]) / 2.0
        cx = (candidate.bbox[0] + candidate.bbox[2]) / 2.0
        cy = (candidate.bbox[1] + candidate.bbox[3]) / 2.0
        k_height = max(1.0, keyword.bbox[3] - keyword.bbox[1])
        c_height = max(1.0, candidate.bbox[3] - candidate.bbox[1])
        same_line = abs(ky - cy) <= max(k_height, c_height)
        distance = math.hypot(kx - cx, ky - cy)
        normalized_distance = distance / max(1.0, max(candidate.bbox[2], candidate.bbox[3], keyword.bbox[2], keyword.bbox[3]))
        relation = 1.0 / (1.0 + 4.0 * normalized_distance)
        if same_line and cx >= kx:
            relation += self.same_line_bonus
        elif cy >= ky:
            relation += self.same_line_bonus * 0.75
        return min(1.5, relation)
