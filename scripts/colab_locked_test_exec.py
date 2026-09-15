"""Run the current submission notebook once against the isolated internal test split in Colab."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path("/content/itda_final_payload")
DRIVE_IMAGES = Path("/content/drive/MyDrive/상품사진입니다")
TEST_CSV = Path("/content/itda_locked_test.csv")
OUT_DIR = Path("/content/itda_locked_internal_test")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_image(image_id: str) -> Path:
    for suffix in IMAGE_EXTENSIONS:
        candidate = DRIVE_IMAGES / f"{image_id}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"missing test image: {image_id}")


def main() -> None:
    if OUT_DIR.exists():
        raise FileExistsError(f"refusing to overwrite prior test output: {OUT_DIR}")
    with TEST_CSV.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 459:
        raise RuntimeError(f"expected 459 locked test rows, found {len(rows)}")
    targets = {Path(row["filename"]).stem: row["date"] for row in rows}
    if len(targets) != 459:
        raise RuntimeError("duplicate test image IDs")

    OUT_DIR.mkdir(parents=True)
    staged = OUT_DIR / "images"
    staged.mkdir()
    for image_id in targets:
        image = find_image(image_id)
        (staged / image.name).symlink_to(image)

    weights = ROOT / "weights"
    required_weights = {
        "detector": weights / "paddle/ppocrv5_mobile_det/inference.pdiparams",
        "recognizer": weights / "paddle/PP-OCRv6_medium_rec/inference.pdiparams",
        "yolo": weights / "final_kaggle/expiry_binary_yolov8n_1280_best.pt",
    }
    missing = [str(path) for path in required_weights.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing local weights: " + ", ".join(missing))

    output_csv = OUT_DIR / "predictions.csv"
    env = os.environ | {
        "PYTHONPATH": str(ROOT / "src"),
        "ITDA_INPUT_DIR": str(staged),
        "ITDA_OUTPUT_PATH": str(output_csv),
        "ITDA_WEIGHTS_ROOT": str(weights),
        "ITDA_V6_WEIGHTS_ROOT": str(weights / "paddle/PP-OCRv6_medium_rec"),
        "ITDA_YOLO_WEIGHTS": str(weights / "final_kaggle/expiry_binary_yolov8n_1280_best.pt"),
    }
    started = time.monotonic()
    subprocess.run([
        sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
        str(ROOT / "predict.ipynb"), "--ExecutePreprocessor.timeout=2400",
        "--output", "executed_predict.ipynb", "--output-dir", str(OUT_DIR),
    ], cwd=ROOT, env=env, check=True)
    elapsed = time.monotonic() - started

    with output_csv.open(encoding="utf-8", newline="") as handle:
        predictions = {row["image_id"]: row["final_date"] for row in csv.DictReader(handle)}
    if set(predictions) != set(targets):
        raise RuntimeError("prediction IDs do not match the locked test split")
    exact = sum(predictions[image_id] == target for image_id, target in targets.items())
    evidence = {
        "images": len(targets),
        "exact_match": exact,
        "exact_match_pct": round(exact / len(targets) * 100, 4),
        "elapsed_seconds": round(elapsed, 2),
        "seconds_per_image": round(elapsed / len(targets), 4),
        "weights_sha256": {name: sha256(path) for name, path in required_weights.items()},
    }
    (OUT_DIR / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
