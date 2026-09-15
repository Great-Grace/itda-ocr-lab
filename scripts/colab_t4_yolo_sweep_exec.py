"""T4 GPU detector throughput sweep for the conditional YOLO-first architecture."""
from __future__ import annotations

import csv
import json
import shutil
import time
from pathlib import Path

import torch

from ultralytics import YOLO

DRIVE = Path('/content/drive/MyDrive/상품사진입니다')
ROOT = Path('/content/itda_lab')
WEIGHTS = Path('/content/expiry_binary_yolov8n_1280_best.pt')
WORK = Path('/content/itda_t4_val_images')
OUT = Path('/content/colab_t4_yolo_sweep.json')


def find_image(image_id: str) -> Path | None:
    for suffix in ('.jpg', '.jpeg', '.png', '.webp'):
        candidate = DRIVE / f'{image_id}{suffix}'
        if candidate.exists():
            return candidate
    return None


def main() -> None:
    labels_path = ROOT / 'data/splits/val.csv'
    ids = [Path(row['filename']).stem for row in csv.DictReader(labels_path.open(encoding='utf-8', newline=''))]
    source_paths = [find_image(image_id) for image_id in ids]
    source_paths = [path for path in source_paths if path is not None]
    WORK.mkdir(parents=True, exist_ok=True)
    local_paths = []
    for source in source_paths:
        destination = WORK / source.name
        if not destination.exists():
            shutil.copyfile(source, destination)
        local_paths.append(destination)
    print(f'val_images={len(local_paths)}')

    model = YOLO(str(WEIGHTS))
    # Force CUDA context and discard warm-up from reported timings.
    _ = model.predict(source=[str(local_paths[0])], imgsz=960, conf=0.20, device=0, verbose=False)
    torch.cuda.synchronize()
    configs = [(size, conf) for size in (960, 1280, 1536) for conf in (0.20, 0.25, 0.30)]
    rows = []
    for size, conf in configs:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        # Explicit micro-batches avoid Ultralytics' large-list dataloader stalls
        # and make progress/recovery possible on ephemeral Colab kernels.
        results = []
        for offset in range(0, len(local_paths), 32):
            batch_paths = [str(path) for path in local_paths[offset : offset + 32]]
            results.extend(model.predict(
                source=batch_paths,
                imgsz=size,
                conf=conf,
                device=0,
                batch=32,
                workers=0,
                verbose=False,
                stream=False,
            ))
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        box_images = sum(1 for result in results if result.boxes is not None and len(result.boxes) > 0)
        box_count = sum(len(result.boxes) for result in results)
        rows.append({
            'imgsz': size,
            'conf': conf,
            'images': len(local_paths),
            'seconds': elapsed,
            'sec_per_image': elapsed / max(1, len(local_paths)),
            'images_per_second': len(local_paths) / max(elapsed, 1e-9),
            'images_with_boxes': box_images,
            'box_count': box_count,
            'peak_gpu_mem_mb': torch.cuda.max_memory_allocated() / (1024 * 1024),
        })
        print(json.dumps(rows[-1]), flush=True)
        # Checkpoint after every setting so a later kernel interruption keeps data.
        OUT.write_text(json.dumps({'hardware': torch.cuda.get_device_name(0), 'torch': torch.__version__, 'weights': str(WEIGHTS), 'rows': rows}, ensure_ascii=False, indent=2), encoding='utf-8')
    payload = {
        'hardware': torch.cuda.get_device_name(0),
        'torch': torch.__version__,
        'weights': str(WEIGHTS),
        'rows': rows,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
