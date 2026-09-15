"""Conservative OCR output sanitization for recognition ablations.

The recognizer's raw output is never overwritten in the token cache.  A
sanitized copy is used only for candidate generation, making this a reversible
post-processing experiment rather than an unsafe charset change to pretrained
weights.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Iterable

from ..contracts import OCRToken


_ALLOWED = re.compile(r"[^0-9A-Za-z가-힣\s./:-]")
_DATEISH = re.compile(r"(?:\d|[OIil|])[^\n]{0,18}(?:\d|[OIil|])", re.I)
_NUMERIC_DATE_ALLOWED = re.compile(r"[^0-9OIil|./:-]", re.I)


def sanitize_text(text: str, mode: str = "none") -> str:
    """Return a parser-facing text view while preserving raw text elsewhere."""

    if mode == "none":
        return text
    if mode == "soft":
        return _ALLOWED.sub("", text)
    if mode == "date_context":
        return _ALLOWED.sub("", text) if _DATEISH.search(text) else text
    if mode == "numeric_date":
        # Restrict only date-like tokens to the characters a numeric/date
        # recognizer can reliably emit. Non-date tokens remain untouched so
        # expiry/manufacturing keywords are still available to the selector.
        return _NUMERIC_DATE_ALLOWED.sub("", text) if _DATEISH.search(text) else text
    raise ValueError(f"Unknown sanitization mode: {mode}")


def sanitize_tokens(tokens: Iterable[OCRToken], mode: str = "none") -> list[OCRToken]:
    sanitized: list[OCRToken] = []
    for token in tokens:
        text = sanitize_text(token.text, mode)
        extras = dict(token.extras)
        if text != token.text:
            extras["raw_text"] = token.text
            extras["sanitizer_mode"] = mode
        sanitized.append(replace(token, text=text, extras=extras))
    return sanitized
