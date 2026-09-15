"""CPU SVTR date-ROI second pass merged with cached B4 OCR tokens."""
from __future__ import annotations

import json
import importlib
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import paddle
from PIL import Image, ImageOps

ROOT = Path('/content/itda_ocr_lab')
subprocess.run(['bash', '-lc', 'mkdir -p /content/itda_ocr_lab && tar -xzf /content/itda_ocr_bundle.tar.gz -C /content/itda_ocr_lab'], check=True)
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / 'src'))
importlib.invalidate_caches()
from ocr_lab.modules.attention_svtr import SVTRTinyCTC, ctc_greedy_decode
from ocr_lab.modules.paddle_ocr import _bbox_from_poly, _crop_bbox, _first_present, _result_payload
from ocr_lab.contracts import OCRToken


def run(*args: str) -> None:
    subprocess.run(args, check=True)


dataset = Path('/content/drive/MyDrive/상품사진입니다')
ids = [line.strip() for line in Path('/content/screen32_ids.txt').read_text().splitlines() if line.strip()]
b4_tokens = {}
with Path('/content/b4_screen32_tokens.jsonl').open(encoding='utf-8') as handle:
    for line in handle:
        row = json.loads(line); b4_tokens[row['image_id']] = row['tokens']

checkpoint = torch.load('/content/svtr_checkpoint.pt', map_location='cpu', weights_only=False)
charset = checkpoint['charset']
model = SVTRTinyCTC(len(charset))
model.load_state_dict(checkpoint['state_dict'])
model.eval()

from paddleocr import TextDetection
# Paddle 3.3.1's PIR oneDNN path cannot execute this downloaded mobile-det
# graph on the CPU-only runtime. The production B0 config still requests
# oneDNN; this diagnostic second pass disables it only for compatibility.
paddle.set_flags({'FLAGS_use_mkldnn': False})
detector = TextDetection(
    model_name='PP-OCRv5_mobile_det',
    model_dir=str(dataset / 'weights' / 'paddle' / 'ppocrv5_mobile_det'),
    device='cpu',
    enable_mkldnn=False,
    enable_hpi=False,
)

merged = []
started_total = time.perf_counter()
for image_id in ids:
    image_path = dataset / f'{image_id}.jpg'
    if not image_path.exists():
        image_path = next((dataset / f'{image_id}{suffix}' for suffix in ('.jpeg', '.png', '.webp') if (dataset / f'{image_id}{suffix}').exists()), image_path)
    source = np.asarray(Image.open(image_path).convert('RGB'))
    result = list(detector.predict(str(image_path), batch_size=1))
    payload = _result_payload(result[0]) if result else {}
    polygons = _first_present(payload, 'dt_polys', 'polys', 'boxes')
    scores = _first_present(payload, 'dt_scores', 'det_scores')
    crops, crop_meta = [], []
    for index, polygon in enumerate(polygons):
        crop = _crop_bbox(source, polygon)
        if crop is None:
            continue
        image = Image.fromarray(crop).convert('L')
        ratio = 48 / max(1, image.height)
        resized = image.resize((max(8, int(image.width * ratio)), 48), Image.Resampling.BILINEAR)
        padded = ImageOps.pad(resized, (320, 48), color=255, centering=(0, 0))
        crops.append(torch.from_numpy(np.asarray(padded, dtype=np.float32) / 255.0).unsqueeze(0))
        crop_meta.append((polygon, scores[index] if index < len(scores) else 0.0))
    svtr_tokens = []
    if crops:
        with torch.inference_mode():
            logits = model(torch.stack(crops))
            texts = ctc_greedy_decode(logits, charset)
            confidences = logits.exp().max(dim=-1).values.mean(dim=0).tolist()
        for index, text in enumerate(texts):
            if not text:
                continue
            polygon, detection_score = crop_meta[index]
            svtr_tokens.append({
                'text': text,
                'confidence': float(confidences[index]),
                'bbox': _bbox_from_poly(polygon),
                'extras': {'polygon': polygon.tolist() if hasattr(polygon, 'tolist') else polygon, 'source': 'svtr_roi_second_pass'},
                'detection_confidence': float(detection_score or 0.0),
            })
    merged.append({'image_id': image_id, 'tokens': b4_tokens.get(image_id, []) + svtr_tokens})

Path('/content/svtr_merged_tokens.jsonl').write_text(
    ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in merged), encoding='utf-8'
)
Path('/content/B9_roi_runtime.json').write_text(json.dumps({
    'wall_seconds': time.perf_counter() - started_total,
    'peak_rss_mb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    'image_count': len(ids),
}, ensure_ascii=False, indent=2), encoding='utf-8')
cache_input = Path('/content/cache_input'); cache_input.mkdir(exist_ok=True)
for image_id in ids:
    (cache_input / f'{image_id}.jpg').touch()
run(
    sys.executable, 'scripts/run_experiment.py',
    '--config', 'configs/experiments/b8_extended_date_parser.yaml',
    '--input', str(cache_input),
    '--output', '/content/B9_SVTR_ROI_SECOND_PASS_screen32',
    '--labels', '/content/labels.csv',
    '--image-ids-file', '/content/screen32_ids.txt',
    '--tokens-cache', '/content/svtr_merged_tokens.jsonl',
    '--device', 'cpu', '--threads', '4',
)
