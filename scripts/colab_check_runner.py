
import os
from pathlib import Path

p = Path("/content/itda_ocr/scripts/colab_measured_runner.py")
print("colab_measured_runner exists:", p.exists())
print("/content/itda_ocr/scripts:", os.listdir("/content/itda_ocr/scripts"))
