from __future__ import annotations

import subprocess
import sys
from pathlib import Path

subprocess.run([
    sys.executable, '-m', 'pip', 'install', '--quiet',
    'paddlepaddle==3.3.1', 'paddleocr==3.7.0'
], check=True)
from paddleocr import TextRecognition

TextRecognition(model_name='PP-OCRv6_medium_rec', device='cpu')
source = Path('/root/.paddlex/official_models/PP-OCRv6_medium_rec')
if not source.is_dir():
    raise RuntimeError(f'missing {source}')
subprocess.run([
    'tar', '-czf', '/content/ppocrv6_medium_rec.tar.gz',
    '-C', '/root/.paddlex/official_models', 'PP-OCRv6_medium_rec'
], check=True)
print('B4_WEIGHT_TAR_READY')
