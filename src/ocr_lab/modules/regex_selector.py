from __future__ import annotations

import re
from typing import Iterable

from ..contracts import DateCandidate, OCRToken


DATE_LIKE = re.compile(r"(?<!\d)(\d{2,4})\s*[.\-/년]\s*(\d{1,2})\s*[.\-/월]\s*(\d{1,2})")
COMPACT_DATE = re.compile(r"(?<!\d)(\d{8})(?!\d)")
KEYWORDS = ("소비기한", "유통기한", "까지", "exp", "best before")


class KeywordRegexSelector:
    def __init__(self, **params: object) -> None:
        self.keyword_weight = float(params.get("keyword_weight", 0.7))
        self.confidence_weight = float(params.get("confidence_weight", 0.2))
        self.position_weight = float(params.get("position_weight", 0.1))

    def candidates(self, tokens: Iterable[OCRToken]) -> list[DateCandidate]:
        token_list = list(tokens)
        joined = " ".join(token.text for token in token_list)
        candidates: list[DateCandidate] = []
        for index, token in enumerate(token_list):
            text = token.text.strip()
            for match in DATE_LIKE.finditer(text):
                candidate = self._candidate(match.group(0), token, index, token_list)
                candidates.append(candidate)
            for match in COMPACT_DATE.finditer(text):
                raw = match.group(1)
                candidate = self._candidate(f"{raw[:4]}.{raw[4:6]}.{raw[6:]}", token, index, token_list)
                candidates.append(candidate)
        if not candidates and joined:
            for match in DATE_LIKE.finditer(joined):
                candidates.append(DateCandidate(raw_text=match.group(0), score=0.1, evidence=["joined_text"]))
        return sorted(candidates, key=lambda item: item.score, reverse=True)

    def _candidate(self, raw: str, token: OCRToken, index: int, tokens: list[OCRToken]) -> DateCandidate:
        context = " ".join(t.text.lower() for t in tokens[max(0, index - 2): index + 3])
        keyword = any(word in context for word in KEYWORDS)
        score = self.confidence_weight * token.confidence
        evidence = ["date_pattern"]
        if keyword:
            score += self.keyword_weight
            evidence.append("expiry_keyword")
        return DateCandidate(raw_text=raw, score=score, evidence=evidence, token_indices=[index])

    def select(self, tokens: Iterable[OCRToken]) -> DateCandidate | None:
        candidates = self.candidates(tokens)
        return candidates[0] if candidates else None
