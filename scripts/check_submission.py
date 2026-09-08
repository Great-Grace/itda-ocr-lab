#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys


REQUIRED = ["image_id", "year", "month", "day", "final_date"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the competition submission schema.")
    parser.add_argument("csv_path")
    args = parser.parse_args()
    with open(args.csv_path, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != REQUIRED:
            raise SystemExit(f"Invalid columns: expected {REQUIRED}, got {reader.fieldnames}")
        rows = list(reader)
    if not rows:
        raise SystemExit("Submission is empty")
    for index, row in enumerate(rows, start=2):
        if row["final_date"] != "NONE" and not (len(row["year"]) == 4 and len(row["month"]) == 2 and len(row["day"]) == 2 and row["final_date"] == f"{row['year']}-{row['month']}-{row['day']}"):
            raise SystemExit(f"Invalid normalized date at CSV line {index}: {row}")
    print(f"OK: {len(rows)} rows, columns={REQUIRED}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
