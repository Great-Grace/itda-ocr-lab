#!/usr/bin/env python3
"""Create a deterministic small validation smoke split from val.csv."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/splits/val.csv")
    parser.add_argument("--output", default="runs/measured/val_smoke10.csv")
    parser.add_argument("--count", type=int, default=10)
    args = parser.parse_args()
    source, output = Path(args.source), Path(args.output)
    with source.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if args.count < 1 or args.count > len(rows):
        raise SystemExit("count must be within the source split")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows[:args.count])
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
