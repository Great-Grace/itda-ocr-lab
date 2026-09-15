"""Install the submission's pinned runtime dependencies in Colab."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


root = Path("/content/itda_final_payload")
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "-r", str(root / "requirements.txt")], check=True)
import paddle
import paddleocr
import ultralytics
import rapidocr_onnxruntime

print({
    "python": sys.version.split()[0],
    "paddle": paddle.__version__,
    "paddleocr": getattr(paddleocr, "__version__", "unknown"),
    "ultralytics": ultralytics.__version__,
    "rapidocr": getattr(rapidocr_onnxruntime, "__version__", "unknown"),
})
