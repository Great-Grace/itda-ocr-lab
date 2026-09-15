"""Keep SVTR ROI tokens only for images with no base date candidate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ocr_lab.contracts import OCRToken
from ocr_lab.modules.regex_selector import KeywordRegexSelector

parser = argparse.ArgumentParser()
parser.add_argument("--base", required=True)
parser.add_argument("--svtr", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

base_rows = {row["image_id"]: row for row in map(json.loads, Path(args.base).read_text().splitlines()) if row}
svtr_rows = {row["image_id"]: row for row in map(json.loads, Path(args.svtr).read_text().splitlines()) if row}
selector = KeywordRegexSelector(allow_day_first=True, allow_month_names=True)
used_second_pass = 0
out = []
for image_id, row in base_rows.items():
    base_tokens = [item for item in row.get("tokens", []) if item.get("extras", {}).get("source") != "svtr_local_roi"]
    base_objects = [OCRToken(**item) for item in base_tokens]
    if selector.candidates(base_objects):
        tokens = base_tokens
    else:
        svtr_tokens = svtr_rows.get(image_id, {}).get("tokens", [])
        tokens = base_tokens + svtr_tokens
        used_second_pass += 1
    out.append({"image_id": image_id, "tokens": tokens})
Path(args.output).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in out), encoding="utf-8")
print(json.dumps({"images": len(out), "second_pass_images": used_second_pass}, ensure_ascii=False))
