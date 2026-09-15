"""Build train/holdout manifests from Kaggle region detection annotations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--train-audit", required=True)
parser.add_argument("--holdout-audit", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--train-ids", default="runs/recognizer_dataset_v1/train150_ids.txt")
parser.add_argument("--holdout-ids", default="runs/recognizer_dataset_v1/holdout50_ids.txt")
args = parser.parse_args()

out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
all_rows = json.loads(Path(args.train_audit).read_text()) + json.loads(Path(args.holdout_audit).read_text())
for name, ids_path in (("train150", args.train_ids), ("holdout50", args.holdout_ids)):
    wanted = set(Path(ids_path).read_text().split())
    rows = [row for row in all_rows if row.get("image_id") in wanted]
    payload = []
    for row in rows:
        image = next((Path(root) / f"{row['image_id']}{suffix}" for root in ("/private/tmp/itda_drive_dev100", "/private/tmp/itda_drive_lock100") for suffix in (".jpg", ".jpeg", ".png", ".webp") if (Path(root) / f"{row['image_id']}{suffix}").exists()), None)
        payload.append({"image_id": row["image_id"], "image": str(image) if image else None, "gt_boxes": row.get("gt_boxes", []), "status": row.get("status")})
    (out / f"{name}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"train": len(json.loads((out/'train150.json').read_text())), "holdout": len(json.loads((out/'holdout50.json').read_text()))}))
