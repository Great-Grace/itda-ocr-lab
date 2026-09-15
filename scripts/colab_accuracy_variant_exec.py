"""Run a parameterized detector-size accuracy variant using the bundled evaluator."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

root = Path('/content/itda_lab')
source = (root / 'scripts/eval_yolo_paddle_pipeline.py').read_text(encoding='utf-8')
source = source.replace('imgsz=960, conf=0.20', 'imgsz=1280, conf=0.20')
variant = Path('/content/eval_yolo_paddle_pipeline_1280.py')
variant.write_text(source, encoding='utf-8')
sys.argv = [
    'eval_yolo_paddle_pipeline_1280.py',
    '--weights', '/content/expiry_binary_yolov8n_1280_best.pt',
    '--images', '/content/drive/MyDrive/상품사진입니다',
    '--labels', str(root / 'data/splits/val.csv'),
    '--rec-model-dir', '/content/PP-OCRv6_medium_rec',
    '--rec-model-name', 'PP-OCRv6_medium_rec',
    '--device', '0',
    '--rec-device', 'gpu:0',
    '--batch-size', '32',
    '--limit', '459',
    '--expand', '1.0',
    '--classes', '0',
    '--output', '/content/direct_eval459_1280.json',
]
runpy.run_path(str(variant), run_name='__main__')
