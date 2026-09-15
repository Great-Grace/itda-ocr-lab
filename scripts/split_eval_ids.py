#!/usr/bin/env python3
"""Split a labeled image list deterministically into dev and locked holdout IDs."""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--dev-output", required=True)
    parser.add_argument("--lock-output", required=True)
    parser.add_argument("--dev-count", type=int, default=100)
    args = parser.parse_args()

    with Path(args.labels).open(encoding="utf-8", newline="") as handle:
        ids = [row["image_id"] for row in csv.DictReader(handle) if row.get("image_id")]
    if not ids or not 0 < args.dev_count < len(ids):
        raise SystemExit("dev-count must be between 1 and the number of labeled images minus one")
    ranked = sorted(ids, key=lambda image_id: hashlib.sha256(image_id.encode("utf-8")).hexdigest())
    dev, lock = ranked[:args.dev_count], ranked[args.dev_count:]
    for target, values in ((args.dev_output, dev), (args.lock_output, lock)):
        path = Path(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(values) + "\n", encoding="utf-8")
    print(f"dev={len(dev)} lock={len(lock)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
