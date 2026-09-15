from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_locked_test_runner_is_fixed_to_the_test_split() -> None:
    source = (ROOT / "scripts" / "run_locked_test.py").read_text(encoding="utf-8")
    assert 'TEST_SPLIT = ROOT / "data" / "splits" / "test.csv"' in source
    assert 'parser.add_argument("--images", required=True' in source
    assert 'parser.add_argument("--split"' not in source
    assert "Refusing to overwrite prior locked-test evidence" in source
    assert "Refusing to consume the locked test with uncommitted changes" in source
