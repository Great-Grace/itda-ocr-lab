#!/usr/bin/env python3
"""Materialize OCR crop CSVs as PaddleOCR SimpleDataSet lists."""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


def write_split(source: Path, output: Path) -> int:
    output.mkdir(parents=True, exist_ok=True)
    image_dir = output / "images"
    image_dir.mkdir(exist_ok=True)
    rows = list(csv.DictReader((source / "labels.csv").open(encoding="utf-8")))
    kept = 0
    lines = []
    for row in rows:
        digits = "".join(c for c in row.get("label", "") if c.isdigit())
        if len(digits) != 8:
            continue
        src = (source / row["image"]).resolve()
        if not src.is_file():
            raise FileNotFoundError(src)
        dst = image_dir / row["image"]
        if not dst.exists():
            os.symlink(src, dst)
        # Preserve the separators expected by the date parser while training
        # on the actual date crop, not a synthetic rendering.
        lines.append(f"images/{row['image']}\t{row['label']}\n")
        kept += 1
    (output / f"{output.name}_list.txt").write_text("".join(lines), encoding="utf-8")
    return kept


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-crops", required=True)
    parser.add_argument("--val-crops", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.output)
    train = write_split(Path(args.train_crops), root / "train")
    val = write_split(Path(args.val_crops), root / "val")
    print({"train": train, "val": val, "output": str(root)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
