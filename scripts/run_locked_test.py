#!/usr/bin/env python3
"""Execute the frozen submission notebook exactly once on the isolated test split.

The script deliberately has no split override: it always stages only the IDs
from ``data/splits/test.csv`` before executing ``predict.ipynb``.  It is for
the final audit after the architecture and parameters have been frozen.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST_SPLIT = ROOT / "data" / "splits" / "test.csv"
REQUIRED_COLUMNS = ["image_id", "year", "month", "day", "final_date"]
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_test_targets() -> dict[str, str]:
    with TEST_SPLIT.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 459:
        raise RuntimeError(f"Locked test split must contain 459 rows; found {len(rows)}")
    targets = {Path(row["filename"]).stem: row["date"] for row in rows}
    if len(targets) != 459:
        raise RuntimeError("Locked test split contains duplicate image IDs")
    return targets


def image_for_id(image_root: Path, image_id: str) -> Path:
    for extension in IMAGE_EXTENSIONS:
        candidate = image_root / f"{image_id}{extension}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Missing locked test image: {image_id} under {image_root}")


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def require_clean_git() -> None:
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    if status:
        raise RuntimeError(
            "Refusing to consume the locked test with uncommitted changes. "
            "Freeze the reviewed submission revision first."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, type=Path, help="Folder containing the original competition images")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "locked_test_final")
    args = parser.parse_args()

    image_root = args.images.resolve()
    if not image_root.is_dir():
        raise FileNotFoundError(f"Image folder does not exist: {image_root}")
    require_clean_git()
    targets = load_test_targets()
    sources = {image_id: image_for_id(image_root, image_id) for image_id in targets}

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite prior locked-test evidence: {output_dir}")
    output_dir.mkdir(parents=True)

    weights_root = Path(os.environ.get("ITDA_WEIGHTS_ROOT", ROOT / "weights")).resolve()
    v6_root = Path(os.environ.get("ITDA_V6_WEIGHTS_ROOT", weights_root / "paddle" / "PP-OCRv6_medium_rec")).resolve()
    yolo_weight = Path(os.environ.get("ITDA_YOLO_WEIGHTS", weights_root / "final_kaggle" / "expiry_binary_yolov8n_1280_best.pt")).resolve()
    model_files = {
        "ppocrv5_detector": weights_root / "paddle" / "ppocrv5_mobile_det" / "inference.pdiparams",
        "ppocrv6_recognizer": v6_root / "inference.pdiparams",
        "yolo_expiry_detector": yolo_weight,
    }
    missing_models = [str(path) for path in model_files.values() if not path.is_file()]
    if missing_models:
        raise FileNotFoundError("Missing required local weights:\n" + "\n".join(missing_models))

    with tempfile.TemporaryDirectory(prefix="itda_locked_test_") as tmp_dir:
        staged = Path(tmp_dir) / "images"
        staged.mkdir()
        for image_id, source in sources.items():
            (staged / source.name).symlink_to(source)

        prediction_csv = output_dir / "predictions.csv"
        env = os.environ | {
            "ITDA_INPUT_DIR": str(staged),
            "ITDA_OUTPUT_PATH": str(prediction_csv),
            "ITDA_WEIGHTS_ROOT": str(weights_root),
            "ITDA_V6_WEIGHTS_ROOT": str(v6_root),
            "ITDA_YOLO_WEIGHTS": str(yolo_weight),
            "PYTHONPATH": str(ROOT / "src"),
        }
        command = [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            str(ROOT / "predict.ipynb"),
            "--ExecutePreprocessor.timeout=2400",
            "--output",
            "executed_predict.ipynb",
            "--output-dir",
            str(output_dir),
        ]
        subprocess.run(command, cwd=ROOT, env=env, check=True)

    with prediction_csv.open(encoding="utf-8", newline="") as handle:
        predictions = list(csv.DictReader(handle))
    if not predictions or list(predictions[0]) != REQUIRED_COLUMNS:
        raise RuntimeError("Prediction CSV has an invalid schema")
    predicted_dates = {row["image_id"]: row["final_date"] for row in predictions}
    if set(predicted_dates) != set(targets):
        raise RuntimeError("Prediction IDs do not exactly match the locked test IDs")

    exact_matches = sum(predicted_dates[image_id] == target for image_id, target in targets.items())
    evidence = {
        "git_sha": git_sha(),
        "test_split": str(TEST_SPLIT),
        "test_images": len(targets),
        "exact_match": exact_matches,
        "exact_match_pct": round(exact_matches / len(targets) * 100, 4),
        "weights_sha256": {name: sha256(path) for name, path in model_files.items()},
        "prediction_csv": str(prediction_csv),
    }
    (output_dir / "locked_test_evidence.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(evidence, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
