"""Read the terminal evidence from a completed/failed Colab locked-test run."""
from __future__ import annotations

import json
from pathlib import Path

out = Path("/content/itda_locked_internal_test")
print("files", [str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()])
notebook = out / "executed_predict.ipynb"
if notebook.is_file():
    data = json.loads(notebook.read_text(encoding="utf-8"))
    for index, cell in enumerate(data.get("cells", [])):
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                print("error_cell", index)
                print("\n".join(output.get("traceback", [])))
if (out / "predictions.csv").is_file():
    print("prediction_bytes", (out / "predictions.csv").stat().st_size)
