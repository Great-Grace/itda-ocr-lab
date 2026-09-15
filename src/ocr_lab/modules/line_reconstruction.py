"""Geometry-aware reconstruction of OCR tokens into logical text lines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..contracts import OCRToken


@dataclass(frozen=True)
class OCRLine:
    """A logical line and the source token indices that formed it."""

    text: str
    token_indices: tuple[int, ...]


def reconstruct_lines(tokens: Iterable[OCRToken], y_tolerance: float = 0.6) -> list[OCRLine]:
    """Group nearby OCR boxes by row and order each row left-to-right.

    ``y_tolerance`` is relative to the larger box height.  Tokens without a
    bbox remain in their original order as a conservative fallback.
    """

    token_list = list(tokens)
    if not token_list:
        return []
    positioned: list[tuple[int, OCRToken, float, float]] = []
    unpositioned: list[int] = []
    for index, token in enumerate(token_list):
        if token.bbox is None:
            unpositioned.append(index)
            continue
        left, top, right, bottom = token.bbox
        positioned.append((index, token, (top + bottom) / 2.0, max(1.0, bottom - top)))

    groups: list[list[tuple[int, OCRToken, float, float]]] = []
    for item in sorted(positioned, key=lambda value: (value[2], value[1].bbox[0] if value[1].bbox else 0.0)):
        _, _, center_y, height = item
        chosen = None
        for group_index, group in enumerate(groups):
            group_center = sum(value[2] for value in group) / len(group)
            group_height = max(value[3] for value in group)
            if abs(center_y - group_center) <= y_tolerance * max(height, group_height):
                chosen = group_index
                break
        if chosen is None:
            groups.append([item])
        else:
            groups[chosen].append(item)

    lines: list[OCRLine] = []
    for group in sorted(groups, key=lambda value: min(item[2] for item in value)):
        ordered = sorted(group, key=lambda value: value[1].bbox[0] if value[1].bbox else 0.0)
        indices = tuple(item[0] for item in ordered)
        lines.append(OCRLine(" ".join(token_list[index].text.strip() for index in indices if token_list[index].text.strip()), indices))
    if unpositioned:
        lines.append(OCRLine(" ".join(token_list[index].text.strip() for index in unpositioned if token_list[index].text.strip()), tuple(unpositioned)))
    return [line for line in lines if line.text]
