"""Run the measured YOLO-crop + PP-OCRv6 recognizer validation on Colab."""
from __future__ import annotations

import runpy
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

bundle = Path('/content/itda_ocr_code_bundle.tar.gz')
root = Path('/content/itda_lab')
if root.exists():
    shutil.rmtree(root)
root.mkdir(parents=True)
with tarfile.open(bundle, 'r:gz') as archive:
    archive.extractall(root)

try:
    import ultralytics  # noqa: F401
except Exception:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', '--no-deps', 'ultralytics==8.4.150'], check=True)

v6_archive = Path('/content/drive/MyDrive/상품사진입니다/weights/ppocrv6_medium_rec.tar.gz')
v6_root = Path('/content/PP-OCRv6_medium_rec')
if not v6_root.exists():
    with tarfile.open(v6_archive, 'r:gz') as archive:
        archive.extractall('/content')
    candidates = list(Path('/content').glob('**/inference.yml'))
    found = next((p.parent for p in candidates if 'PP-OCRv6_medium_rec' in str(p.parent)), None)
    if found and found != v6_root:
        shutil.copytree(found, v6_root, dirs_exist_ok=True)

print('bundle', root, 'val', root / 'data/splits/val.csv')
print('v6_root', v6_root, 'exists', v6_root.exists())

import os
os.chdir(root)
os.environ['PYTHONPATH'] = str(root / 'src')
sys.argv = [
    'scripts/eval_yolo_paddle_pipeline.py',
    '--weights', '/content/expiry_binary_yolov8n_1280_best.pt',
    '--images', '/content/drive/MyDrive/상품사진입니다',
    '--labels', 'data/splits/val.csv',
    '--rec-model-dir', str(v6_root),
    '--rec-model-name', 'PP-OCRv6_medium_rec',
    '--device', '0',
    '--rec-device', 'gpu:0',
    '--batch-size', '32',
    '--limit', '10',
    '--expand', '1.0',
    '--classes', '0',
    '--output', '/content/colab_accuracy_smoke10.json',
]
runpy.run_path(str(root / 'scripts/eval_yolo_paddle_pipeline.py'), run_name='__main__')
