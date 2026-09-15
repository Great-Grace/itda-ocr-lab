from pathlib import Path

from ocr_lab.modules.paddle_ocr import _validate_local_weights


def test_local_weight_validation_rejects_missing_directories(tmp_path: Path) -> None:
    params = {
        "text_detection_model_dir": str(tmp_path / "missing-det"),
        "text_recognition_model_dir": str(tmp_path / "missing-rec"),
    }

    try:
        _validate_local_weights(params, required=True)
    except RuntimeError as exc:
        assert "detector, recognizer" in str(exc)
    else:
        raise AssertionError("missing local model directories must fail closed")
