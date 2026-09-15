from __future__ import annotations

from pathlib import Path

from ocr_lab.contracts import OCRToken
from ocr_lab.modules.date_candidates import generate_date_candidates
from ocr_lab.modules.date_normalizer import DateNormalizer
from ocr_lab.modules.regex_selector import KeywordRegexSelector
from ocr_lab.modules.union_spatial_selector import UnionSpatialSelector
from ocr_lab.modules.line_reconstruction import reconstruct_lines
from ocr_lab.modules.text_sanitizer import sanitize_text
from ocr_lab.modules.paddle_ocr import _crop_polygon
from ocr_lab.pipeline import run_pipeline


def test_baseline0_date_formats_and_calendar_validation() -> None:
    supported = {
        "2026.10.14": "2026-10-14",
        "26-10-14": "2026-10-14",
        "10/14": "NONE-10-14",
        "20261014": "2026-10-14",
        "261014": "2026-10-14",
        "2026년 3월 7일": "2026-03-07",
        "3월 7일": "NONE-03-07",
        "10.14 09:45 PM": "NONE-10-14",
        "2021:04:13": "2021-04-13",
    }
    selector = KeywordRegexSelector()
    normalizer = DateNormalizer()
    for text, expected in supported.items():
        token = OCRToken(text=text, confidence=0.95)
        candidate = selector.select([token])
        assert candidate is not None
        assert normalizer.normalize("sample", candidate).final_date == expected
    assert selector.select([OCRToken(text="13.32", confidence=0.99)]) is None
    assert selector.select([OCRToken(text="2026.02.31", confidence=0.99)]) is None
    assert selector.select([OCRToken(text="209.10.29", confidence=0.99)]) is None
    split = generate_date_candidates([OCRToken(text="2026년"), OCRToken(text="3월"), OCRToken(text="7일")])
    assert [(item.year, item.month, item.day) for item in split] == [("2026", "03", "07")]


def test_day_first_parser_is_an_explicit_ablation() -> None:
    assert generate_date_candidates([OCRToken(text="31-03-2022")]) == []
    candidates = generate_date_candidates([OCRToken(text="31-03-2022")], allow_day_first=True)
    assert [(item.year, item.month, item.day) for item in candidates] == [("2022", "03", "31")]
    candidates = generate_date_candidates([OCRToken(text="BEST BEFORE:21-04-2023")], allow_day_first=True)
    assert [(item.year, item.month, item.day) for item in candidates] == [("2023", "04", "21")]


def test_english_month_name_parser_is_an_explicit_ablation() -> None:
    candidates = generate_date_candidates(
        [OCRToken(text="OCT.21.2021")],
        allow_day_first=True,
        allow_month_names=True,
    )
    assert [(item.year, item.month, item.day) for item in candidates] == [("2021", "10", "21")]


def test_year_month_only_date_is_preserved() -> None:
    candidates = generate_date_candidates([OCRToken(text="04/2023")], allow_day_first=True)
    assert [(item.year, item.month, item.day) for item in candidates] == [("2023", "04", None)]
    normalizer = DateNormalizer()
    assert normalizer.normalize("sample", candidates[0]).final_date == "2023-04-NONE"


def test_spaced_and_month_name_date_formats() -> None:
    token_texts = ["2022 08 03", "12 09 2021", "SEP 18 2021", "29/SEP/2022", "2021.1217"]
    expected = [
        ("2022", "08", "03"), ("2021", "09", "12"), ("2021", "09", "18"),
        ("2022", "09", "29"), ("2021", "12", "17"),
    ]
    actual = []
    for text in token_texts:
        candidates = generate_date_candidates([OCRToken(text=text)], allow_day_first=True, allow_month_names=True)
        actual.append((candidates[0].year, candidates[0].month, candidates[0].day))
    assert actual == expected


def test_baseline0_bbox_context_prefers_expiry_over_manufacture() -> None:
    tokens = [
        OCRToken(text="제조일자", confidence=0.99, bbox=(0, 0, 60, 20)),
        OCRToken(text="2025.01.01", confidence=0.99, bbox=(70, 0, 150, 20)),
        OCRToken(text="소비기한", confidence=0.80, bbox=(0, 40, 60, 60)),
        OCRToken(text="2026.01.01", confidence=0.80, bbox=(70, 40, 150, 60)),
    ]
    selected = KeywordRegexSelector().select(tokens)
    assert selected is not None
    assert selected.year == "2026"


def test_selector_supports_exp_date_and_mfd_cues() -> None:
    tokens = [
        OCRToken(text="MFD", confidence=0.99, bbox=(0, 0, 40, 20)),
        OCRToken(text="2025.01.01", confidence=0.99, bbox=(50, 0, 130, 20)),
        OCRToken(text="EXP DATE", confidence=0.80, bbox=(0, 40, 70, 60)),
        OCRToken(text="2026.01.01", confidence=0.80, bbox=(80, 40, 160, 60)),
    ]
    selected = KeywordRegexSelector().select(tokens)
    assert selected is not None and selected.year == "2026"


def test_recognition_sanitizer_is_reversible_and_context_aware() -> None:
    raw = "EXP: 2027.10.14???"
    assert sanitize_text(raw, "soft") == "EXP: 2027.10.14"
    assert sanitize_text("소비기한", "date_context") == "소비기한"
    assert sanitize_text(raw, "date_context") == "EXP: 2027.10.14"
    assert sanitize_text("2027.10.O1???", "numeric_date") == "2027.10.O1"
    assert sanitize_text("소비기한", "numeric_date") == "소비기한"


def test_selector_can_downweight_second_pass_source() -> None:
    tokens = [
        OCRToken(text="2021.01.01", confidence=0.99, bbox=(0, 0, 100, 20)),
        OCRToken(text="2026.01.01", confidence=0.99, bbox=(0, 40, 100, 60), extras={"source": "svtr_local_roi"}),
    ]
    selector = KeywordRegexSelector(source_confidence_multipliers={"svtr_local_roi": 0.25})
    selected = selector.select(tokens)
    assert selected is not None and selected.year == "2021"


def test_complete_date_beats_time_like_partial_candidate() -> None:
    tokens = [
        OCRToken(text="까지", confidence=0.9, bbox=(0, 0, 30, 20)),
        OCRToken(text="05:16", confidence=0.99, bbox=(40, 0, 80, 20)),
        OCRToken(text="21.08.06", confidence=0.99, bbox=(90, 0, 160, 20)),
    ]
    selected = KeywordRegexSelector().select(tokens)
    assert selected is not None and selected.year == "2021" and selected.month == "08"


def test_perspective_crop_keeps_polygon_text_region() -> None:
    import numpy as np
    image = np.full((40, 80, 3), 255, dtype=np.uint8)
    image[12:28, 20:60] = 0
    polygon = [[20, 12], [60, 12], [60, 28], [20, 28]]
    crop = _crop_polygon(image, polygon, "perspective")
    assert crop is not None and crop.shape[0] >= 15 and crop.shape[1] >= 39
    assert crop.mean() < 30


def test_pipeline_image_ids_filter(tmp_path: Path) -> None:
    ids = tmp_path / "ids.txt"
    ids.write_text("sample_01\n", encoding="utf-8")
    config = {
        "runtime": {"device": "cpu", "threads": 1},
        "ocr": {"plugin": "mock", "params": {}},
        "preprocess": {"plugin": "none", "params": {}},
        "selector": {"plugin": "keyword_regex", "params": {}},
        "normalizer": {"plugin": "date_ko_v1", "params": {}},
    }
    metrics = run_pipeline(config, "data/sample", tmp_path / "run", image_ids_path=ids)
    assert metrics["image_count"] == 1


def test_pipeline_cache_skips_ocr_initialization(tmp_path: Path) -> None:
    ids = tmp_path / "ids.txt"
    ids.write_text("sample_01\n", encoding="utf-8")
    cache = tmp_path / "tokens.jsonl"
    cache.write_text('{"image_id":"sample_01","tokens":[{"text":"2026.01.02","confidence":0.9}]}\n', encoding="utf-8")
    config = {
        "runtime": {"device": "cpu", "threads": 1},
        "ocr": {"plugin": "mock", "params": {}},
        "preprocess": {"plugin": "none", "params": {}},
        "selector": {"plugin": "keyword_regex", "params": {}},
        "normalizer": {"plugin": "date_ko_v1", "params": {}},
    }
    metrics = run_pipeline(config, "data/sample", tmp_path / "run", labels_path=None,
                           tokens_cache_path=cache, image_ids_path=ids)
    assert metrics["image_count"] == 1


def test_geometry_aware_line_reconstruction_rejoins_split_date() -> None:
    tokens = [
        OCRToken(text="2027.", confidence=0.9, bbox=(100, 10, 140, 30)),
        OCRToken(text="10.", confidence=0.9, bbox=(145, 11, 170, 30)),
        OCRToken(text="14", confidence=0.9, bbox=(175, 10, 195, 30)),
        OCRToken(text="제조일", confidence=0.9, bbox=(100, 80, 145, 100)),
    ]
    lines = reconstruct_lines(tokens)
    assert [line.text for line in lines] == ["2027. 10. 14", "제조일"]
    candidate = KeywordRegexSelector().select(tokens)
    assert candidate is not None
    assert (candidate.year, candidate.month, candidate.day) == ("2027", "10", "14")


def test_numeric_confusion_is_corrected_only_inside_date_context() -> None:
    token = OCRToken(text="EXP 2027.O1.14", confidence=0.9)
    candidate = KeywordRegexSelector().select([token])
    assert candidate is not None
    assert (candidate.year, candidate.month, candidate.day) == ("2027", "01", "14")


def test_union_spatial_selector_keeps_yolo_source_bonus_interpretable() -> None:
    tokens = [
        OCRToken(text="2025.01.01", confidence=0.99, bbox=(0, 0, 100, 20), extras={"source": "baseline"}),
        OCRToken(text="2026.01.01", confidence=0.90, bbox=(0, 0, 100, 20), extras={"source": "yolo"}, detection_confidence=0.95),
    ]
    selector = UnionSpatialSelector(
        keyword_weight=0.0,
        negative_keyword_weight=0.0,
        yolo_bonus=1.5,
        candidate_score_weight=0.0,
        recognition_log_weight=0.0,
        detection_log_weight=0.0,
        bbox_iou_weight=0.5,
    )
    selected = selector.select(tokens)
    assert selected is not None and selected.year == "2026"


def test_teammate_date_ambiguity_and_format_suite() -> None:
    selector = UnionSpatialSelector(allow_day_first=True, allow_month_names=True)
    normalizer = DateNormalizer(none_token="NONE", year_missing_token="NONE", day_missing_token="NONE")

    def _pred(tokens: list[OCRToken]) -> str:
        c = selector.select(tokens)
        return normalizer.normalize("test", c).final_date

    # 1. Month-name single token without separators (000982 "USE BY 15FEB22")
    assert _pred([OCRToken(text="15FEB22", confidence=0.9)]) == "2022-02-15"

    # 2. Month-name single token with dots (003013 "29.Jan.2024")
    assert _pred([OCRToken(text="29.Jan.2024", confidence=0.9)]) == "2024-01-29"

    # 3. Ambiguous two-digit triple resolves via year plausibility (000928 "13/03/22")
    assert _pred([OCRToken(text="13/03/22", confidence=0.9)]) == "2022-03-13"

    # 4. European slash format with day > 12 (002866 "28/02/22")
    assert _pred([OCRToken(text="28/02/22", confidence=0.9)]) == "2022-02-28"

    # 5. Genuine tie keeps Korean year-first default ("20.06.25")
    assert _pred([OCRToken(text="20.06.25", confidence=0.9)]) == "2020-06-25"

    # 6. Unambiguous 4-digit year year-first ("2022.03.05")
    assert _pred([OCRToken(text="2022.03.05", confidence=0.9)]) == "2022-03-05"

    # 7. Unambiguous 4-digit year year-last (000516 "16/07/2026")
    assert _pred([OCRToken(text="16/07/2026", confidence=0.9)]) == "2026-07-16"

    # 8. US month-first unambiguous when middle token > 12 ("05.20.24")
    assert _pred([OCRToken(text="05.20.24", confidence=0.9)]) == "2024-05-20"

