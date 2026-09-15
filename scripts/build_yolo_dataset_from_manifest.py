"""Convert range-fetched Kaggle box manifests into a YOLO dataset."""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


CLASS_MODES = {
    "region4": (["date", "due", "code", "full"], {0: 0, 1: 1, 2: 2, 3: 3}),
    "expiry_binary": (["expiry"], {0: 0, 1: 0}),
    "expiry_full": (["expiry", "full"], {0: 0, 1: 0, 3: 1}),
    "date_due_full": (["date", "due", "full"], {0: 0, 1: 1, 3: 2}),
}


def materialize(manifest_path: Path, split: str, output: Path, link: bool, class_map: dict[int, int]) -> int:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    classes = data["classes"]
    records = [row for row in data["records"] if row.get("status") == "ok"]
    image_dir = output / "images" / split
    label_dir = output / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    for row in records:
        source = Path(row["image"])
        target = image_dir / f"{row['image_id']}{source.suffix.lower()}"
        if not target.exists():
            if link:
                try:
                    target.symlink_to(source)
                except OSError:
                    shutil.copy2(source, target)
            else:
                shutil.copy2(source, target)
        with (label_dir / f"{row['image_id']}.txt").open("w", encoding="utf-8") as handle:
            for box in row.get("boxes", []):
                source_class = int(box["class"])
                if source_class not in class_map:
                    continue
                handle.write(f"{class_map[source_class]} {box['cx']:.6f} {box['cy']:.6f} {box['width']:.6f} {box['height']:.6f}\n")
    return len(records)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--class-mode", choices=sorted(CLASS_MODES), default="region4")
    parser.add_argument("--copy", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    class_names, class_map = CLASS_MODES[args.class_mode]
    count_train = materialize(Path(args.train_manifest), "train", output, not args.copy, class_map)
    count_val = materialize(Path(args.val_manifest), "val", output, not args.copy, class_map)
    payload = {
        "path": str(output),
        "train": "images/train",
        "val": "images/val",
        "nc": len(class_names),
        "names": class_names,
    }
    (output / "data.yaml").write_text("\n".join([f"path: {output}", "train: images/train", "val: images/val", f"nc: {payload['nc']}", f"names: {payload['names']}" ]), encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps({**payload, "class_mode": args.class_mode, "counts": {"train": count_train, "val": count_val}}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"train": count_train, "val": count_val, "output": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
