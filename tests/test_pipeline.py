import json
from pathlib import Path

from ocr_lab.config import load_config
from ocr_lab.pipeline import run_pipeline


def test_mock_pipeline_writes_submission(tmp_path: Path) -> None:
    image = tmp_path / "1.png"
    image.write_bytes(b"not a real image; mock backend does not decode it")
    image.with_suffix(".png.ocr.json").write_text(
        '[{"text":"소비기한","confidence":0.99},{"text":"2026.05.29","confidence":0.95}]',
        encoding="utf-8",
    )
    labels = tmp_path / "labels.csv"
    labels.write_text("image_id,final_date\n1,2026-05-29\n", encoding="utf-8")

    config = load_config(Path(__file__).parents[1] / "configs/baseline_mock.yaml")
    run_dir = tmp_path / "run"
    metrics = run_pipeline(config, tmp_path, run_dir, labels_path=labels)

    assert metrics["image_count"] == 1
    assert metrics["final_date_exact_match"] == 1.0
    assert (run_dir / "predictions.csv").exists()
    assert (run_dir / "summary.md").exists()
    assert (run_dir / "review.html").exists()
    assert "2026-05-29" in (run_dir / "predictions.csv").read_text(encoding="utf-8")
    assert "ITDA OCR Lab — Review Dashboard" in (run_dir / "review.html").read_text(encoding="utf-8")


def test_token_caching_reuses_tokens(tmp_path: Path) -> None:
    cache_file = tmp_path / "cached_tokens.jsonl"
    cached_row = {
        "image_id": "cached_img",
        "tokens": [
            {"text": "유통기한", "confidence": 0.99, "bbox": None, "extras": {}},
            {"text": "2027.12.31", "confidence": 0.95, "bbox": None, "extras": {}},
        ],
    }
    cache_file.write_text(json.dumps(cached_row) + "\n", encoding="utf-8")

    # Create dummy image without ocr sidecar
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    (img_dir / "cached_img.png").write_bytes(b"dummy")

    config = load_config(Path(__file__).parents[1] / "configs/baseline_mock.yaml")
    run_dir = tmp_path / "cache_run"
    metrics = run_pipeline(config, img_dir, run_dir, tokens_cache_path=cache_file)

    assert metrics["image_count"] == 1
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["tokens_cache_used"] is True
    pred_csv = (run_dir / "predictions.csv").read_text(encoding="utf-8")
    assert "2027-12-31" in pred_csv


def test_review_escapes_ocr_html(tmp_path: Path) -> None:
    image = tmp_path / "unsafe.png"
    image.write_bytes(b"dummy")
    image.with_suffix(".png.ocr.json").write_text(
        '[{"text":"<script>alert(1)</script>","confidence":0.9}]',
        encoding="utf-8",
    )
    config = load_config(Path(__file__).parents[1] / "configs/baseline_mock.yaml")
    run_dir = tmp_path / "unsafe_run"
    run_pipeline(config, tmp_path, run_dir)
    html = (run_dir / "review.html").read_text(encoding="utf-8")
    assert "\\u003cscript\\u003e" in html
