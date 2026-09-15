import os, sys
from pathlib import Path
print(os.getcwd(), Path('/content/itda_ocr_lab').exists(), list(Path('/content/itda_ocr_lab').glob('*'))[:10])
print('/content/itda_ocr_lab/src' in sys.path, sys.path[:5])
print(list(Path('/content/itda_ocr_lab/src').glob('*')) if Path('/content/itda_ocr_lab/src').exists() else 'NO_SRC')
