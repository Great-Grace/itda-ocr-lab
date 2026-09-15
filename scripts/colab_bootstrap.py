
import os, sys, subprocess, tarfile
from pathlib import Path

print("=== [Colab Bootstrap Starting] ===")
root = Path("/content/itda_ocr")
root.mkdir(parents=True, exist_ok=True)

with tarfile.open("/content/itda_payload.tar.gz", "r:gz") as tar:
    tar.extractall(root)
print("Extracted itda_payload.tar.gz to", root)

print("Installing paddlepaddle, paddleocr, etc...")
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "paddlepaddle==3.3.1", "paddleocr==3.7.0", "pyyaml", "psutil"], check=True)

print("Verifying paddle import...")
import paddle, paddleocr
print("Paddle version:", paddle.__version__, "| PaddleOCR version:", paddleocr.__version__)

print("=== [Bootstrap Complete] ===")
