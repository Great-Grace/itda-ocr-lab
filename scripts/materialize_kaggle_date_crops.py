"""Materialize only Kaggle date-crop images needed for recognizer training.

The 2.7 GB Kaggle archive is never downloaded in full. Image members are
fetched with HTTP Range requests and paired with our existing Gemini-filtered
date labels. Splits come from ``data/splits`` so no Kaggle random split leaks
into the measured benchmark.
"""
from __future__ import annotations

import argparse
import csv
import json
import struct
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def member_bytes(url: str, row: dict) -> bytes:
    offset = int(row["offset"])
    head = urllib.request.urlopen(urllib.request.Request(url, headers={"Range": f"bytes={offset}-{offset + 2047}"}), timeout=60).read()
    fields = struct.unpack_from("<4s5H3L2H", head, 0)
    name_len, extra_len = fields[9], fields[10]
    start = offset + 30 + name_len + extra_len
    end = start + int(row["compressed_size"]) - 1
    payload = urllib.request.urlopen(urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"}), timeout=60).read()
    if int(row.get("compression", 8)) == 0:
        return payload
    return zlib.decompress(payload, -15)


def read_csv(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {Path(row["filename"]).stem: row for row in csv.DictReader(handle)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-manifest", required=True)
    parser.add_argument("--labels", required=True, help="clean_training_set_3066.csv")
    parser.add_argument("--split-dir", required=True, help="directory containing train.csv/val.csv/test.csv")
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-images", type=int, default=0, help="smoke limit; 0 means all")
    parser.add_argument("--include-test", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(Path(args.archive_manifest).read_text(encoding="utf-8"))
    url, files = manifest["archive_url"], manifest["files"]
    image_rows = {}
    prefix = "expiry_date_detection_dataset/expiry_date_detection_dataset/images/"
    for row in files:
        if row["name"].startswith(prefix):
            image_rows[Path(row["name"]).stem] = row
    labels = read_csv(Path(args.labels))
    split_by_id = {}
    for split in ("train", "val", "test"):
        split_path = Path(args.split_dir) / f"{split}.csv"
        if not split_path.exists():
            continue
        for image_id in read_csv(split_path):
            split_by_id[image_id] = split
    selected = [image_id for image_id in labels if image_id in image_rows and image_id in split_by_id and (args.include_test or split_by_id[image_id] != "test")]
    selected.sort(key=lambda image_id: (split_by_id[image_id], int(image_id) if image_id.isdigit() else image_id))
    if args.max_images:
        selected = selected[: args.max_images]
    output = Path(args.output)
    jobs = []
    for image_id in selected:
        split = split_by_id[image_id]
        row = image_rows[image_id]
        destination = output / split / "images" / f"{image_id}.jpg"
        destination.parent.mkdir(parents=True, exist_ok=True)
        jobs.append((image_id, split, row, destination))

    def fetch(job: tuple[str, str, dict, Path]) -> dict:
        image_id, split, row, destination = job
        if not destination.exists():
            destination.write_bytes(member_bytes(url, row))
        return {"image_id": image_id, "split": split, "image": str(destination), "label": labels[image_id]["date"], "kaggle_image": row["name"], "kaggle_crc": row["crc"]}

    records = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(fetch, job) for job in jobs]
        for index, future in enumerate(as_completed(futures), 1):
            records.append(future.result())
            if index % 100 == 0 or index == len(futures):
                print(f"materialized {index}/{len(futures)}", flush=True)
    records.sort(key=lambda row: (row["split"], row["image_id"]))
    for split in ("train", "val", "test"):
        split_rows = [row for row in records if row["split"] == split]
        if not split_rows:
            continue
        (output / split).mkdir(parents=True, exist_ok=True)
        with (output / split / "labels.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["image_id", "image", "label"])
            writer.writeheader()
            writer.writerows({"image_id": row["image_id"], "image": str(Path(row["image"]).relative_to(output / split)), "label": row["label"]} for row in split_rows)
    (output / "manifest.json").write_text(json.dumps({"source": "kaggle_expiry_date_detection_dataset", "label_source": str(args.labels), "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"selected": len(selected), "materialized": len(records), "splits": {split: sum(row["split"] == split for row in records) for split in ("train", "val", "test")}}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
