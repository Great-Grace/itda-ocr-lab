"""Real-weight smoke test for the config-first union backend on CPU."""
from __future__ import annotations
import os
import shutil
import tarfile
from pathlib import Path

root = Path('/content/itda_union_lab')
if root.exists():
    shutil.rmtree(root)
root.mkdir(parents=True)
with tarfile.open('/content/itda_ocr_code_bundle.tar.gz', 'r:gz') as archive:
    archive.extractall(root)
v6_root = Path('/content/PP-OCRv6_medium_rec')
if not v6_root.exists():
    with tarfile.open('/content/drive/MyDrive/상품사진입니다/weights/ppocrv6_medium_rec.tar.gz', 'r:gz') as archive:
        archive.extractall('/content')
    candidates = list(Path('/content').glob('**/inference.yml'))
    found = next((p.parent for p in candidates if 'PP-OCRv6_medium_rec' in str(p.parent)), None)
    if found and found != v6_root:
        shutil.copytree(found, v6_root, dirs_exist_ok=True)
os.chdir(root)
os.environ.update({
    'ITDA_WEIGHTS_ROOT': '/content/drive/MyDrive/상품사진입니다/weights',
    'ITDA_V6_WEIGHTS_ROOT': str(v6_root),
    'ITDA_YOLO_WEIGHTS': '/content/expiry_binary_yolov8n_1280_best.pt',
    'FLAGS_use_mkldnn': '0',
    'OMP_NUM_THREADS': '2',
    'MKL_NUM_THREADS': '2',
    'OPENBLAS_NUM_THREADS': '2',
    'PYTHONPATH': str(root / 'src'),
})
import runpy
import sys
sys.argv = [
    'run_experiment.py',
    '--config', 'configs/experiments/current-sota-union.yaml',
    '--input', '/content/drive/MyDrive/상품사진입니다',
    '--output', '/content/union_smoke',
    '--device', 'cpu',
    '--threads', '2',
    '--max-images', '1',
]
runpy.run_path(str(root / 'scripts/run_experiment.py'), run_name='__main__')
