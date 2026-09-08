import subprocess
import sys
from pathlib import Path


def test_intake_reports_missing_fields(tmp_path: Path) -> None:
    request = tmp_path / "request.yaml"
    request.write_text("hypothesis: test\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "scripts/intake_experiment.py", str(request)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "needs_input" in result.stdout
    assert "target" in result.stdout
