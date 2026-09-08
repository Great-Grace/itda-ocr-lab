import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_drive_resolver_counts_only_images(tmp_path: Path) -> None:
    dataset = tmp_path / "MyDrive" / "ITDA_OCR"
    images = dataset / "images"
    images.mkdir(parents=True)
    (images / "1.png").write_bytes(b"image")
    (images / "1.png.ocr.json").write_text("[]", encoding="utf-8")
    (images / "labels.csv").write_text("image_id,final_date\n", encoding="utf-8")
    (dataset / "DATASET_MANIFEST.yaml").write_text(
        "dataset_id: test\nversion: 1\nimage_dir: images\nexpected_image_count: 1\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            "scripts/resolve_drive_dataset.py",
            "--mount-root", str(tmp_path),
            "--folder", "MyDrive/ITDA_OCR",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["image_count"] == 1
    assert payload["status"] == "ready"


def test_submission_notebook_uses_injected_paths() -> None:
    notebook = json.loads((ROOT / "notebooks/predict.ipynb").read_text(encoding="utf-8"))
    first_cell = "".join(notebook["cells"][0]["source"])
    all_code = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert notebook["cells"][0]["cell_type"] == "code"
    assert 'os.environ.get("ITDA_INPUT_DIR"' in first_cell
    assert 'os.environ.get("ITDA_OUTPUT_PATH"' in first_cell
    assert "shutil.copyfile(run_dir / \"predictions.csv\", output_path)" in all_code
