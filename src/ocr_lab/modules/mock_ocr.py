from __future__ import annotations

import json
from pathlib import Path

from ..contracts import OCRToken


class MockOCRBackend:
    """Offline smoke backend; reads `<image>.ocr.json` sidecars when present."""

    def __init__(self, **_: object) -> None:
        pass

    def extract(self, image_path: Path) -> list[OCRToken]:
        sidecar = image_path.with_suffix(image_path.suffix + ".ocr.json")
        if not sidecar.exists():
            return []
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        return [OCRToken(**item) for item in payload]
