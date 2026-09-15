from __future__ import annotations

from ..contracts import DateCandidate, Prediction
from .date_candidates import candidate_key, parse_date_text


class DateNormalizer:
    def __init__(self, **params: object) -> None:
        self.none_token = str(params.get("none_token", "NONE"))
        self.year_missing_token = str(params.get("year_missing_token", self.none_token))
        self.day_missing_token = str(params.get("day_missing_token", self.none_token))

    def normalize(self, image_id: str, candidate: DateCandidate | None) -> Prediction:
        if candidate is None:
            return Prediction(image_id=image_id, year=self.none_token, month=self.none_token, day=self.none_token, final_date=self.none_token)
        parsed = candidate
        if candidate_key(parsed) is None:
            parsed = parse_date_text(candidate.raw_text, pattern=candidate.pattern) or candidate
        key = candidate_key(parsed)
        if key is None:
            return Prediction(
                image_id=image_id,
                year=self.none_token,
                month=self.none_token,
                day=self.none_token,
                final_date=self.none_token,
                confidence=candidate.score,
                evidence=[*candidate.evidence, "normalization_failed"],
            )
        year, month, day = key
        output_year = year or self.year_missing_token
        output_month = month or self.none_token
        output_day = day or self.day_missing_token
        if output_year == self.none_token and output_month == self.none_token and output_day == self.none_token:
            final_date = self.none_token
        else:
            final_date = f"{output_year}-{output_month}-{output_day}"
        return Prediction(
            image_id=image_id,
            year=output_year,
            month=output_month,
            day=output_day,
            final_date=final_date,
            confidence=candidate.score,
            evidence=candidate.evidence,
        )
