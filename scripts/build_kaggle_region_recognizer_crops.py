"""Build recognizer crops from Kaggle region boxes plus ITDA date labels.

Prefer the semantic ``due`` box (class 1) because the task asks for expiry
date; fall back to ``date`` (class 0) when no due box exists. This avoids
training on the entire product image or on an unrelated code box.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageOps


def labels_by_id(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {Path(row["filename"]).stem: row["date"] for row in csv.DictReader(handle)}


def build(manifest_path: Path, labels: dict[str, str], output: Path, margin: int) -> int:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in data["records"]:
        if row.get("status") != "ok" or row["image_id"] not in labels:
            continue
        image = Image.open(row["image"]).convert("L")
        candidates = [box for box in row.get("boxes", []) if box["class"] in (1, 0)]
        if not candidates:
            continue
        # Due/expiry semantic class wins; largest area breaks ties.
        candidates.sort(key=lambda box: (box["class"] == 1, box["width"] * box["height"]), reverse=True)
        box = candidates[0]
        left = max(0, int((box["cx"] - box["width"] / 2) * image.width) - margin)
        top = max(0, int((box["cy"] - box["height"] / 2) * image.height) - margin)
        right = min(image.width, int((box["cx"] + box["width"] / 2) * image.width) + margin)
        bottom = min(image.height, int((box["cy"] + box["height"] / 2) * image.height) + margin)
        if right <= left or bottom <= top:
            continue
        filename = f"{row['image_id']}.png"
        crop = ImageOps.pad(image.crop((left, top, right, bottom)), (320, 48), color=255, centering=(0, 0))
        crop.save(output / filename)
        rows.append({"image_id": row["image_id"], "image": filename, "label": labels[row["image_id"]], "source_class": box["name"]})
    with (output / "labels.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "image", "label", "source_class"])
        writer.writeheader(); writer.writerows(rows)
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--margin", type=int, default=8)
    args = parser.parse_args()
    labels = labels_by_id(Path(args.labels)); root = Path(args.output)
    train = build(Path(args.train_manifest), labels, root / "train", args.margin)
    val = build(Path(args.val_manifest), labels, root / "val", args.margin)
    (root / "manifest.json").write_text(json.dumps({"train": train, "val": val, "margin": args.margin}, indent=2), encoding="utf-8")
    print(json.dumps({"train": train, "val": val, "output": str(root)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
