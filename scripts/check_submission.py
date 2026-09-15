#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import sys


REQUIRED = ["image_id", "year", "month", "day", "final_date"]
MISSING_TOKEN = "NONE"
PARTIAL_DATE = re.compile(r"^(?:\d{4}|NONE)-(?:\d{2}|NONE)-(?:\d{2}|NONE)$")


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
        year, month, day, final_date = row["year"], row["month"], row["day"], row["final_date"]
        
        # Validate year
        if year != MISSING_TOKEN and not (len(year) == 4 and year.isdigit()):
            raise SystemExit(f"Invalid year at CSV line {index}: {row}")
        
        # Validate month
        if month != MISSING_TOKEN and not (len(month) == 2 and month.isdigit() and 1 <= int(month) <= 12):
            raise SystemExit(f"Invalid month at CSV line {index}: {row}")
            
        # Validate day
        if day != MISSING_TOKEN and not (len(day) == 2 and day.isdigit() and 1 <= int(day) <= 31):
            raise SystemExit(f"Invalid day at CSV line {index}: {row}")
            
        # Validate final_date
        if year == MISSING_TOKEN and month == MISSING_TOKEN and day == MISSING_TOKEN:
            if final_date != MISSING_TOKEN:
                raise SystemExit(f"Expected final_date 'NONE' when all components missing at line {index}: {row}")
        else:
            expected_final = f"{year}-{month}-{day}"
            if final_date != expected_final:
                raise SystemExit(f"Expected final_date '{expected_final}', got '{final_date}' at line {index}: {row}")
    print(f"OK: {len(rows)} rows, columns={REQUIRED}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
