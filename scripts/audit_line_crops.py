"""Audit date crops for multiple horizontal text lines and build a clean subset."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def line_groups(image: Image.Image) -> list[tuple[int, int]]:
    array = np.asarray(image.convert("L"), dtype=np.uint8)
    # Ink density after a conservative foreground threshold. Smooth a few rows
    # so tilted/dotted characters form a continuous text-line signal.
    density = (array < 205).sum(axis=1).astype(float)
    density = np.convolve(density, np.ones(3) / 3, mode="same")
    active = density >= max(2.0, 0.03 * array.shape[1])
    groups: list[tuple[int, int]] = []
    start = None
    for index, value in enumerate(active):
        if value and start is None:
            start = index
        if start is not None and (not value or index == len(active) - 1):
            end = index if value and index == len(active) - 1 else index - 1
            if end - start + 1 >= 2:
                groups.append((start, end))
            start = None
    return groups


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--crops", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-lines", type=int, default=1)
    args = parser.parse_args()
    source, output = Path(args.crops), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader((source / "labels.csv").open(encoding="utf-8")))
    audit, clean = [], []
    for row in rows:
        groups = line_groups(Image.open(source / row["image"]))
        audited = {**row, "line_group_count": len(groups), "line_groups": json.dumps(groups)}
        audit.append(audited)
        if 1 <= len(groups) <= args.max_lines:
            clean.append(row)
            shutil.copy2(source / row["image"], output / row["image"])
    fields = list(rows[0]) + ["line_group_count", "line_groups"] if rows else ["image"]
    with (output / "audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(audit)
    with (output / "labels.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["image"]); writer.writeheader(); writer.writerows(clean)
    (output / "summary.json").write_text(json.dumps({"total": len(rows), "clean": len(clean), "max_lines": args.max_lines}, indent=2), encoding="utf-8")
    print(json.dumps({"total": len(rows), "clean": len(clean), "output": str(output)}))


if __name__ == "__main__":
    raise SystemExit(main())
