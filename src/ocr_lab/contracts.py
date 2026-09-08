from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class OCRToken:
    text: str
    confidence: float = 0.0
    bbox: Optional[tuple[float, float, float, float]] = None
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class DateCandidate:
    raw_text: str
    normalized: Optional[str] = None
    score: float = 0.0
    evidence: list[str] = field(default_factory=list)
    token_indices: list[int] = field(default_factory=list)


@dataclass
class Prediction:
    image_id: str
    year: str = "NONE"
    month: str = "NONE"
    day: str = "NONE"
    final_date: str = "NONE"
    confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)

