from __future__ import annotations

import re

from ..contracts import DateCandidate, Prediction


class DateNormalizer:
    def normalize(self, image_id: str, candidate: DateCandidate | None) -> Prediction:
        if candidate is None:
            return Prediction(image_id=image_id)
        digits = re.findall(r"\d+", candidate.raw_text)
        if len(digits) == 3:
            year, month, day = digits
        elif len(digits) == 1 and len(digits[0]) == 8:
            year, month, day = digits[0][:4], digits[0][4:6], digits[0][6:]
        else:
            return Prediction(image_id=image_id, confidence=candidate.score, evidence=candidate.evidence)
        if len(year) == 2:
            year = f"20{year}"
        if not (len(year) == 4 and 1 <= int(month) <= 12 and 1 <= int(day) <= 31):
            return Prediction(image_id=image_id, confidence=candidate.score, evidence=candidate.evidence)
        month, day = month.zfill(2), day.zfill(2)
        return Prediction(
            image_id=image_id,
            year=year,
            month=month,
            day=day,
            final_date=f"{year}-{month}-{day}",
            confidence=candidate.score,
            evidence=candidate.evidence,
        )
