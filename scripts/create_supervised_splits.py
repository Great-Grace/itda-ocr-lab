"""Create deterministic 150/50 supervised research splits from the 200 labels."""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


def label_group(value: str) -> str:
    if value.upper() == "NONE":
        return "none"
    year, month, day = value.split("-")
    if year.lower() == "none":
        return "year_missing"
    if day.lower() == "none":
        return "day_missing"
    return "full"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--holdout", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260910)
    args = parser.parse_args()
    rows = list(csv.DictReader(Path(args.labels).open(encoding="utf-8")))
    rng = random.Random(args.seed)
    groups = defaultdict(list)
    for row in rows:
        groups[(label_group(row["final_date"]), row.get("confidence", "unknown"))].append(row)
    for values in groups.values():
        rng.shuffle(values)
    holdout = []
    # Proportional allocation across semantic/date-confidence strata.
    allocations = {key: int(round(len(values) * args.holdout / len(rows))) for key, values in groups.items()}
    while sum(allocations.values()) < args.holdout:
        key = max(groups, key=lambda item: len(groups[item]) - allocations[item])
        allocations[key] += 1
    while sum(allocations.values()) > args.holdout:
        key = max(allocations, key=lambda item: allocations[item])
        allocations[key] -= 1
    for key, values in groups.items():
        holdout.extend(values[:allocations[key]])
    holdout_ids = {row["image_id"] for row in holdout}
    train = [row for row in rows if row["image_id"] not in holdout_ids]
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    for name, values in (("train150", train), ("holdout50", holdout)):
        (output / f"{name}_ids.txt").write_text("\n".join(row["image_id"] for row in values) + "\n", encoding="utf-8")
        with (output / f"{name}_labels.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(values)
    manifest = {"seed": args.seed, "train_count": len(train), "holdout_count": len(holdout), "train_groups": {key[0] + ":" + key[1]: sum(1 for row in train if label_group(row["final_date"]) == key[0] and row.get("confidence") == key[1]) for key in groups}, "holdout_groups": {key[0] + ":" + key[1]: sum(1 for row in holdout if label_group(row["final_date"]) == key[0] and row.get("confidence") == key[1]) for key in groups}}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
