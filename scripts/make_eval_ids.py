#!/usr/bin/env python3
"""Create a deterministic newline-delimited image-id subset from a labels CSV."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels")
    parser.add_argument("--source-ids", help="Optional existing newline-delimited IDs")
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int)
    args = parser.parse_args()

    if not args.labels and not args.source_ids:
        raise SystemExit("provide --labels or --source-ids")
    if args.source_ids:
        ids = [line.strip() for line in Path(args.source_ids).read_text(encoding="utf-8").splitlines() if line.strip()]
        rows = [{"image_id": image_id} for image_id in ids]
    else:
        with Path(args.labels).open(encoding="utf-8", newline="") as handle:
            rows = [row for row in csv.DictReader(handle) if row.get("image_id")]
    if not rows:
        raise SystemExit("labels CSV has no image_id rows")
    if args.count is None or args.count >= len(rows):
        selected = rows
    elif args.count < 2:
        selected = rows[:1]
    else:
        indexes = sorted({round(i * (len(rows) - 1) / (args.count - 1)) for i in range(args.count)})
        selected = [rows[index] for index in indexes]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(row["image_id"] for row in selected) + "\n", encoding="utf-8")
    print(f"wrote {len(selected)} ids to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
