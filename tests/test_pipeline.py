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
    config = load_config(Path(__file__).parents[1] / "configs/baseline_mock.yaml")
    metrics = run_pipeline(config, tmp_path, tmp_path / "run")
    assert metrics["image_count"] == 1
    assert (tmp_path / "run/predictions.csv").exists()
    assert "2026-05-29" in (tmp_path / "run/predictions.csv").read_text(encoding="utf-8")
