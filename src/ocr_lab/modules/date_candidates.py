"""Date candidate generation for expiry-date OCR experiments.

The generator deliberately does not choose a date.  It extracts every
date-like span, validates calendar components, and leaves ranking to the
rule-based selector.  This separation lets the experiment report candidate
recall independently from selection accuracy.
"""

from __future__ import annotations

import re
from dataclasses import replace
from datetime import date
from typing import Iterable

from ..contracts import DateCandidate, OCRToken
from .line_reconstruction import reconstruct_lines


_FULL_SEPARATED = re.compile(
    r"(?<!\d)(?<!\d[.\-/:,])(?P<year>\d{2,4})\s*[.\-/:,]\s*"
    r"(?P<month>\d{1,2})\s*[.\-/:,]\s*(?P<day>\d{1,2})(?!\d)"
)
_FULL_DMY_SEPARATED = re.compile(
    r"(?<!\d)(?<!\d[.\-/:,])(?P<day>\d{1,2})\s*[.\-/:,]\s*"
    r"(?P<month>\d{1,2})\s*[.\-/:,]\s*(?P<year>\d{2,4})(?!\d)"
)
_FULL_DMY_LOT_ATTACHED = re.compile(
    r"(?<!\d)(?<!\d[.\-/:,])(?P<day>\d{1,2})\s*[.\-/:,]\s*"
    r"(?P<month>\d{1,2})\s*[.\-/:,]\s*(?P<year>20\d{2})(?=[A-Za-z0-9]*[A-Za-z])"
)
_FULL_YMD_LOT_ATTACHED = re.compile(
    r"(?<!\d)(?<!\d[.\-/:,])(?P<year>20\d{2})\s*[.\-/:,]\s*"
    r"(?P<month>\d{1,2})\s*[.\-/:,]\s*(?P<day>\d{1,2})(?=[A-Za-z0-9]*[A-Za-z])"
)
_FULL_YMD_SPACED = re.compile(
    r"(?<!\d)(?P<year>\d{4})\s+(?P<month>\d{1,2})\s+(?P<day>\d{1,2})(?!\d)"
)
_FULL_DMY_SPACED = re.compile(
    r"(?<!\d)(?P<day>\d{1,2})\s+(?P<month>\d{1,2})\s+(?P<year>\d{2,4})(?!\d)"
)
_FULL_MIXED_COMPACT = re.compile(
    r"(?<![\d.\-/,])(?P<year>\d{4})\s*[.\-/,]\s*(?P<month>\d{2})(?P<day>\d{2})(?!\d)"
)
_MONTH_NAME = r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
_FULL_MDY_MONTH_NAME = re.compile(
    rf"(?<![A-Za-z0-9])(?P<month_name>{_MONTH_NAME})\s*[.\-/,]?\s*"
    r"(?P<day>\d{1,2})(?:\s*[.\-/,]+\s*|\s+)(?P<year>\d{2,4})(?!\d)",
    re.I,
)
_FULL_DMY_MONTH_NAME = re.compile(
    rf"(?<![A-Za-z0-9])(?P<day>\d{{1,2}})\s*[.\-/,]?\s*"
    rf"(?P<month_name>{_MONTH_NAME})(?:\s*[.\-/,]?\s*)(?P<year>\d{{2,4}})(?!\d)",
    re.I,
)
_FULL_MDY_SEPARATED = re.compile(
    r"(?<!\d)(?<!\d[.\-/:,])(?P<month>0?[1-9]|1[0-2])\s*[.\-/:,]\s*"
    r"(?P<day>1[3-9]|2\d|3[01])\s*[.\-/:,]\s*(?P<year>\d{2,4})(?!\d)"
)
_YEAR_MONTH_SEPARATED = re.compile(
    r"(?<![\d.\-/:,])(?P<year>\d{4})\s*[.\-/:,]\s*(?P<month>\d{1,2})(?!\d|\s*[.\-/:,]\s*\d)"
)
_MONTH_YEAR_SEPARATED = re.compile(
    r"(?<![\d.\-/:,])(?P<month>\d{1,2})\s*[.\-/:,]\s*(?P<year>\d{4})(?!\d)"
)
_YEAR_MONTH_KOREAN = re.compile(
    r"(?<!\d)(?P<year>\d{2,4})\s*년\s*(?P<month>\d{1,2})\s*월(?!\s*\d)"
)
_FULL_KOREAN = re.compile(
    r"(?<!\d)(?P<year>\d{2,4})\s*년\s*"
    r"(?P<month>\d{1,2})\s*월\s*(?P<day>\d{1,2})\s*일?"
)
_COMPACT = re.compile(r"(?<!\d)(?P<digits>\d{8}|\d{6})(?!\d)")
_MONTH_DAY_SEPARATED = re.compile(
    r"(?<![\d.\-/,])(?P<month>\d{1,2})\s*[.\-/:,]\s*"
    r"(?P<day>\d{1,2})(?!\d)"
)
_MONTH_DAY_KOREAN = re.compile(
    r"(?<![\d년])(?P<month>\d{1,2})\s*월\s*"
    r"(?P<day>\d{1,2})\s*일?"
)
_MONTH_NUMBER = {
    "jan": "01", "feb": "02", "mar": "03", "apr": "04", "may": "05", "jun": "06",
    "jul": "07", "aug": "08", "sep": "09", "oct": "10", "nov": "11", "dec": "12",
}


def _correct_numeric_context(text: str) -> str:
    """Correct OCR confusions only when the span looks numeric/date-like."""

    pattern = re.compile(r"[0-9OIil|][0-9OIil|.\-/,:년월일\s]{1,}[0-9OIil|]", re.I)
    table = str.maketrans({"O": "0", "o": "0", "I": "1", "i": "1", "l": "1", "|": "1"})
    return pattern.sub(lambda match: match.group(0).translate(table), text)


def parse_date_text(
    raw_text: str,
    *,
    pattern: str = "",
    allow_day_first: bool = False,
    allow_month_names: bool = False,
) -> DateCandidate | None:
    """Parse one already-isolated date string into a candidate.

    The function accepts both four/two digit years and month/day-only forms.
    Two digit years are normalized to the 20xx range for deterministic
    comparison with competition labels.  Missing years remain ``None``.
    """

    text = _correct_numeric_context(raw_text.strip())
    matchers = [
        ("ymd_korean", _FULL_KOREAN),
        ("ymd_separated", _FULL_SEPARATED),
        ("ymd_lot_attached", _FULL_YMD_LOT_ATTACHED),
        ("ymd_spaced", _FULL_YMD_SPACED),
        ("ymd_mixed_compact", _FULL_MIXED_COMPACT),
        ("ym_korean", _YEAR_MONTH_KOREAN),
        ("ym_separated", _YEAR_MONTH_SEPARATED),
    ]
    if allow_day_first:
        matchers.append(("dmy_separated", _FULL_DMY_SEPARATED))
        matchers.append(("dmy_lot_attached", _FULL_DMY_LOT_ATTACHED))
        matchers.append(("dmy_spaced", _FULL_DMY_SPACED))
        matchers.append(("my_separated", _MONTH_YEAR_SEPARATED))
        matchers.append(("mdy_separated", _FULL_MDY_SEPARATED))
    if allow_month_names:
        matchers.append(("mdy_month_name", _FULL_MDY_MONTH_NAME))
        if allow_day_first:
            matchers.append(("dmy_month_name", _FULL_DMY_MONTH_NAME))
    matchers.extend((
        ("ymd_compact", _COMPACT),
        ("md_korean", _MONTH_DAY_KOREAN),
        ("md_separated", _MONTH_DAY_SEPARATED),
    ))
    if pattern:
        # Prioritize testing the requested pattern first if available
        matchers.sort(key=lambda item: 0 if item[0] == pattern else 1)

    for name, matcher in matchers:
        match = matcher.fullmatch(text)
        if not match:
            continue
        groups = match.groupdict()
        if name == "ymd_compact":
            digits = groups["digits"]
            if len(digits) == 8:
                cand_year, cand_month, cand_day = digits[:4], digits[4:6], digits[6:]
                if not _calendar_valid(cand_year, cand_month, cand_day) and allow_day_first:
                    # Try 8-digit compact DMY (e.g. 11122021 -> day=11, month=12, year=2021)
                    alt_day, alt_month, alt_year = digits[:2], digits[2:4], digits[4:]
                    if _calendar_valid(alt_year, alt_month, alt_day):
                        year, month, day = alt_year, alt_month, alt_day
                    else:
                        year, month, day = cand_year, cand_month, cand_day
                else:
                    year, month, day = cand_year, cand_month, cand_day
            else:
                year, month, day = f"20{digits[:2]}", digits[2:4], digits[4:]
        else:
            year = groups.get("year")
            month = groups.get("month") or _MONTH_NUMBER[groups["month_name"].lower()[:3]]
            day = groups.get("day")
            if year and len(year) not in (2, 4):
                continue
            if year and len(year) == 2:
                year = f"20{year}"
                # In packaged food, an expiration year outside plausible range (e.g. 2000-2017)
                # or slash-separated DD/MM/YY where d0 > 12 (e.g. 28/02/22) is typically
                # a swapped DD-MM-YY where YY is in 2018-2035.
                if allow_day_first and day and name.startswith("ymd"):
                    d0_val = int(year) - 2000
                    d2_val = int(day)
                    is_slash = "/" in text
                    if (d0_val < 18 or (is_slash and d0_val > 12)) and 18 <= d2_val <= 35:
                        alt_year = f"20{d2_val}"
                        alt_day = str(d0_val).zfill(2)
                        if _calendar_valid(alt_year, month, alt_day):
                            year, day = alt_year, alt_day
        valid = _calendar_valid(year, month, day)
        return DateCandidate(
            raw_text=text,
            year=year,
            month=month.zfill(2),
            day=day.zfill(2) if day else None,
            calendar_valid=valid,
            pattern=pattern or name,
        )
    return None


def parse_final_date(value: str) -> tuple[str | None, str, str | None] | None:
    """Parse a normalized label/prediction into comparable components."""

    text = str(value or "").strip()
    if text.upper() == "NONE" or not text:
        return None
    match = re.fullmatch(r"(?P<year>\d{4}|NoNE|NONE)-(?P<month>\d{2})-(?P<day>\d{2}|None)", text, re.I)
    if not match:
        candidate = parse_date_text(text)
        if not candidate or not candidate.calendar_valid or not candidate.month:
            return None
        return candidate.year, candidate.month, candidate.day
    year = match.group("year")
    if year.upper() == "NONE":
        year = None
    day = match.group("day")
    return year, match.group("month"), None if day.lower() == "none" else day


def candidate_key(candidate: DateCandidate) -> tuple[str | None, str, str | None] | None:
    if not candidate.calendar_valid or not candidate.month:
        return None
    return candidate.year, candidate.month.zfill(2), candidate.day.zfill(2) if candidate.day else None


def candidate_final_date(candidate: DateCandidate, year_missing_token: str = "NoNE", day_missing_token: str = "None") -> str | None:
    key = candidate_key(candidate)
    if key is None:
        return None
    year, month, day = key
    return f"{year or year_missing_token}-{month}-{day or day_missing_token}"


def generate_date_candidates(
    tokens: Iterable[OCRToken],
    *,
    keep_invalid: bool = False,
    allow_day_first: bool = False,
    allow_month_names: bool = False,
) -> list[DateCandidate]:
    """Generate all non-overlapping date candidates from OCR tokens."""

    token_list = list(tokens)
    generated: list[DateCandidate] = []
    for token_index, token in enumerate(token_list):
        text = token.text.strip()
        if not text:
            continue
        spans: list[tuple[int, int, str, DateCandidate]] = []
        corrected_text = _correct_numeric_context(text)
        # Prefer full dates; partial patterns overlapping a full date are not
        # separate candidates (e.g. the ``10.14`` suffix of ``2026.10.14``).
        full_matchers = [
            ("ymd_korean", _FULL_KOREAN), ("ymd_separated", _FULL_SEPARATED),
            ("ymd_lot_attached", _FULL_YMD_LOT_ATTACHED),
            ("ymd_spaced", _FULL_YMD_SPACED), ("ymd_mixed_compact", _FULL_MIXED_COMPACT),
            ("ym_korean", _YEAR_MONTH_KOREAN), ("ym_separated", _YEAR_MONTH_SEPARATED),
        ]
        if allow_day_first:
            full_matchers.append(("dmy_separated", _FULL_DMY_SEPARATED))
            full_matchers.append(("dmy_lot_attached", _FULL_DMY_LOT_ATTACHED))
            full_matchers.append(("dmy_spaced", _FULL_DMY_SPACED))
            full_matchers.append(("my_separated", _MONTH_YEAR_SEPARATED))
            full_matchers.append(("mdy_separated", _FULL_MDY_SEPARATED))
        if allow_month_names:
            full_matchers.append(("mdy_month_name", _FULL_MDY_MONTH_NAME))
            if allow_day_first:
                full_matchers.append(("dmy_month_name", _FULL_DMY_MONTH_NAME))
        full_matchers.append(("ymd_compact", _COMPACT))
        for name, matcher in full_matchers:
            for match in matcher.finditer(corrected_text):
                parsed = parse_date_text(
                    match.group(0),
                    pattern=name,
                    allow_day_first=allow_day_first,
                    allow_month_names=allow_month_names,
                )
                if parsed:
                    spans.append((match.start(), match.end(), name, parsed))
        occupied = [(start, end) for start, end, _, _ in spans]
        for name, matcher in (("md_korean", _MONTH_DAY_KOREAN), ("md_separated", _MONTH_DAY_SEPARATED)):
            for match in matcher.finditer(corrected_text):
                if any(match.start() < end and start < match.end() for start, end in occupied):
                    continue
                parsed = parse_date_text(match.group(0), pattern=name)
                if parsed:
                    spans.append((match.start(), match.end(), name, parsed))
        for start, end, name, parsed in sorted(spans, key=lambda item: (item[0], -(item[1] - item[0]))):
            if not parsed.calendar_valid and not keep_invalid:
                continue
            candidate = replace(
                parsed,
                token_indices=[token_index],
                features={
                    "recognition_confidence": float(token.confidence),
                    "detection_confidence": float(token.detection_confidence if token.detection_confidence is not None else token.extras.get("detection_confidence", 0.0)),
                },
            )
            generated.append(candidate)

    # Some OCR backends split ``2026`` / ``.10.14`` across adjacent boxes.  A
    # joined-text pass recovers the complete span while retaining token IDs.
    if token_list:
        # Reconstruct each physical row before trying a cross-box date.  This
        # prevents unrelated dates on different rows from being concatenated.
        line_variants: list[tuple[str, tuple[int, ...]]] = []
        for line in reconstruct_lines(token_list):
            line_variants.extend(((line.text, line.token_indices), (line.text.replace(" ", ""), line.token_indices)))
        if not generated:
            for joined, line_indices in line_variants:
                joined = _correct_numeric_context(joined)
                spans: list[tuple[int, int, str, DateCandidate]] = []
                joined_full_matchers = [
                    ("ymd_korean", _FULL_KOREAN), ("ymd_separated", _FULL_SEPARATED),
                    ("ymd_lot_attached", _FULL_YMD_LOT_ATTACHED),
                    ("ymd_spaced", _FULL_YMD_SPACED), ("ymd_mixed_compact", _FULL_MIXED_COMPACT),
                    ("ym_korean", _YEAR_MONTH_KOREAN), ("ym_separated", _YEAR_MONTH_SEPARATED),
                ]
                if allow_day_first:
                    joined_full_matchers.append(("dmy_separated", _FULL_DMY_SEPARATED))
                    joined_full_matchers.append(("dmy_lot_attached", _FULL_DMY_LOT_ATTACHED))
                    joined_full_matchers.append(("dmy_spaced", _FULL_DMY_SPACED))
                    joined_full_matchers.append(("my_separated", _MONTH_YEAR_SEPARATED))
                if allow_month_names:
                    joined_full_matchers.append(("mdy_month_name", _FULL_MDY_MONTH_NAME))
                    if allow_day_first:
                        joined_full_matchers.append(("dmy_month_name", _FULL_DMY_MONTH_NAME))
                joined_full_matchers.append(("ymd_compact", _COMPACT))
                for name, matcher in joined_full_matchers:
                    for match in matcher.finditer(joined):
                        parsed = parse_date_text(
                            match.group(0),
                            pattern=name,
                            allow_day_first=allow_day_first,
                            allow_month_names=allow_month_names,
                        )
                        if parsed:
                            spans.append((match.start(), match.end(), name, parsed))
                occupied = [(start, end) for start, end, _, _ in spans]
                for name, matcher in (("md_korean", _MONTH_DAY_KOREAN), ("md_separated", _MONTH_DAY_SEPARATED)):
                    for match in matcher.finditer(joined):
                        if any(match.start() < end and start < match.end() for start, end in occupied):
                            continue
                        parsed = parse_date_text(match.group(0), pattern=name, allow_day_first=allow_day_first)
                        if parsed:
                            spans.append((match.start(), match.end(), name, parsed))
                for start, end, name, parsed in spans:
                    if not parsed.calendar_valid and not keep_invalid:
                        continue
                    indices = list(line_indices)
                    confidence_values = [float(token_list[i].confidence) for i in indices if i < len(token_list)]
                    detection_values = [float(token_list[i].detection_confidence if token_list[i].detection_confidence is not None else token_list[i].extras.get("detection_confidence", 0.0)) for i in indices if i < len(token_list)]
                    generated.append(replace(
                        parsed,
                        token_indices=indices,
                        evidence=["joined_text"],
                        features={
                            "recognition_confidence": sum(confidence_values) / len(confidence_values) if confidence_values else 0.0,
                            "detection_confidence": sum(detection_values) / len(detection_values) if detection_values else 0.0,
                        },
                    ))
                if generated:
                    break
    # Stable de-duplication by normalized components and source span.
    unique: dict[tuple[object, ...], DateCandidate] = {}
    for candidate in generated:
        key = (candidate.year, candidate.month, candidate.day, candidate.pattern, tuple(candidate.token_indices))
        unique.setdefault(key, candidate)
    return list(unique.values())


def _calendar_valid(year: str | None, month: str, day: str | None) -> bool:
    try:
        month_int = int(month)
        if not 1 <= month_int <= 12:
            return False
        if day is None:
            return True
        day_int = int(day)
        if year:
            date(int(year), month_int, day_int)
        else:
            # 2000 is a leap year and therefore the least surprising reference
            # for month/day-only candidates.
            date(2000, month_int, day_int)
        return True
    except (TypeError, ValueError):
        return False
