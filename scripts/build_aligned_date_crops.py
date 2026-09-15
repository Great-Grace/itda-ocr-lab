"""Build one date-line crop per image by aligning GT date boxes to OCR candidates."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageOps

from ocr_lab.contracts import OCRToken
from ocr_lab.modules.date_candidates import parse_final_date
from ocr_lab.modules.regex_selector import KeywordRegexSelector


def box_iou(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_l = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    area_r = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = area_l + area_r - inter
    return inter / union if union else 0.0


def token_box(token: OCRToken) -> tuple[float, float, float, float] | None:
    if token.bbox is None:
        return None
    return tuple(float(value) for value in token.bbox)


def union_boxes(boxes: list[tuple[float, float, float, float]]) -> tuple[float, float, float, float] | None:
    if not boxes:
        return None
    return min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)


def load_tokens(path: Path) -> dict[str, list[OCRToken]]:
    rows: dict[str, list[OCRToken]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows[row["image_id"]] = [OCRToken(**item) for item in row.get("tokens", [])]
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--tokens-cache", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--margin", type=int, default=5)
    args = parser.parse_args()

    labels = {row["image_id"]: row["final_date"] for row in csv.DictReader(Path(args.labels).open(encoding="utf-8"))}
    tokens_by_id: dict[str, list[OCRToken]] = {}
    for cache in args.tokens_cache:
        tokens_by_id.update(load_tokens(Path(cache)))
    selector = KeywordRegexSelector(allow_day_first=True, allow_month_names=True)
    target_rows = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    records, skipped = [], {}
    for row in target_rows:
        image_id = row["image_id"]
        target = parse_final_date(labels.get(image_id, ""))
        tokens = tokens_by_id.get(image_id, [])
        if target is None or not tokens:
            skipped[image_id] = "missing_label_or_tokens"
            continue
        candidates = [candidate for candidate in selector.candidates(tokens) if parse_final_date(
            f"{candidate.year or 'NoNE'}-{candidate.month}-{candidate.day or 'None'}"
        ) == target]
        if not candidates:
            skipped[image_id] = "no_matching_ocr_candidate"
            continue
        date_boxes = [tuple(float(v) for v in item["bbox"]) for item in row.get("gt_boxes", []) if int(item.get("class", -1)) == 0]
        if not date_boxes:
            skipped[image_id] = "no_gt_date_box"
            continue
        ranked = []
        for candidate in candidates:
            candidate_boxes = [token_box(tokens[index]) for index in candidate.token_indices if index < len(tokens)]
            candidate_box = union_boxes([box for box in candidate_boxes if box is not None])
            if candidate_box is None:
                continue
            for date_box in date_boxes:
                ranked.append((box_iou(candidate_box, date_box), candidate.score, date_box, candidate.raw_text))
        if not ranked:
            skipped[image_id] = "candidate_without_bbox"
            continue
        overlap, score, crop_box, raw_text = max(ranked, key=lambda item: (item[0], item[1]))
        # Class-0 boxes can contain two adjacent date lines. Other annotated
        # class boxes often mark the neighboring line; trim toward that box so
        # a recognizer target does not include a second date.
        all_boxes = [(int(item.get("class", -1)), tuple(float(v) for v in item["bbox"])) for item in row.get("gt_boxes", [])]
        left, top, right, bottom = crop_box
        for cls, other in all_boxes:
            if cls == 0:
                continue
            horizontal = max(0.0, min(right, other[2]) - max(left, other[0]))
            if horizontal < 0.25 * max(1.0, min(right - left, other[2] - other[0])):
                continue
            if other[1] >= top and other[1] < bottom and other[3] > top:
                bottom = min(bottom, other[1] - 1.0)
            elif other[3] <= bottom and other[3] > top and other[1] < top:
                top = max(top, other[3] + 1.0)
        crop_box = (left, top, right, bottom)
        if right <= left or bottom <= top + 3:
            skipped[image_id] = "trimmed_box_empty"
            continue
        image_path = Path(row["image"])
        image = Image.open(image_path).convert("L")
        left = max(0, int(crop_box[0]) - args.margin)
        top = max(0, int(crop_box[1]) - args.margin)
        right = min(image.width, int(crop_box[2]) + args.margin)
        bottom = min(image.height, int(crop_box[3]) + args.margin)
        crop = ImageOps.pad(image.crop((left, top, right, bottom)), (320, 48), color=255, centering=(0, 0))
        filename = f"{image_id}.png"; crop.save(output / filename)
        records.append({"image_id": image_id, "image": filename, "label": labels[image_id], "raw_text": raw_text, "box_iou": f"{overlap:.4f}"})
    with (output / "labels.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "image", "label", "raw_text", "box_iou"]); writer.writeheader(); writer.writerows(records)
    (output / "manifest.json").write_text(json.dumps({"samples": len(records), "skipped": skipped, "margin": args.margin}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"samples": len(records), "skipped": len(skipped), "output": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
