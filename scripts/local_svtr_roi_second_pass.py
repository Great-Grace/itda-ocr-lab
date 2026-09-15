"""Run the locally trained SVTR-Tiny as a real-image ROI second pass."""
from __future__ import annotations

import json
import re
import resource
import time
import argparse
from pathlib import Path

import numpy as np
import paddle
import torch
from PIL import Image, ImageOps
from paddleocr import TextDetection

from ocr_lab.modules.attention_svtr import SVTRTinyCTC, ctc_greedy_decode
from ocr_lab.modules.paddle_ocr import _bbox_from_poly, _crop_bbox, _first_present, _result_payload


ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--image-root", default="/private/tmp/itda_drive_images")
parser.add_argument("--ids-file", default="runs/B0_test_v1/screen32_ids.txt")
parser.add_argument("--base-cache", default="runs/B0_LOCAL_DRIVE_SCREEN32/ocr_tokens.jsonl")
parser.add_argument("--out", default="runs/B12_LOCAL_SVTR_ROI_DATELIKE_SCREEN32")
args = parser.parse_args()
IMAGE_ROOT = Path(args.image_root)
WEIGHTS_ROOT = Path("/private/tmp/itda_drive_weights/paddle")
IDS = [line.strip() for line in (ROOT / args.ids_file).read_text().splitlines() if line.strip()]
BASE_CACHE = ROOT / args.base_cache
OUT = ROOT / args.out
OUT.mkdir(parents=True, exist_ok=True)


def _looks_date_like(text: str) -> bool:
    """Keep only SVTR outputs that can plausibly contribute a date."""

    digits = len(re.findall(r"[0-9OIil|]", text))
    has_separator = bool(re.search(r"[./:/\-]", text))
    return digits >= 4 or (digits >= 2 and has_separator)

b4_tokens = {}
with BASE_CACHE.open(encoding="utf-8") as handle:
    for line in handle:
        row = json.loads(line)
        b4_tokens[row["image_id"]] = row["tokens"]

checkpoint = torch.load(ROOT / "runs/attention_suite_svtr_long/checkpoint_epoch_019.pt", map_location="cpu", weights_only=False)
model = SVTRTinyCTC(len(checkpoint["charset"]))
model.load_state_dict(checkpoint["state_dict"])
model.eval()

paddle.set_flags({"FLAGS_use_mkldnn": False})
detector = TextDetection(
    model_name="PP-OCRv5_mobile_det",
    model_dir=str(WEIGHTS_ROOT / "ppocrv5_mobile_det"),
    device="cpu",
    limit_side_len=960,
    limit_type="max",
    thresh=0.25,
        box_thresh=0.60,
    unclip_ratio=1.8,
    enable_mkldnn=False,
    enable_hpi=False,
)

started = time.perf_counter()
merged = []
for image_id in IDS:
    image_path = IMAGE_ROOT / f"{image_id}.jpg"
    if not image_path.exists():
        image_path = next((IMAGE_ROOT / f"{image_id}{suffix}" for suffix in (".jpeg", ".png", ".webp") if (IMAGE_ROOT / f"{image_id}{suffix}").exists()), image_path)
    source = np.asarray(Image.open(image_path).convert("RGB"))
    result = list(detector.predict(str(image_path), batch_size=1))
    payload = _result_payload(result[0]) if result else {}
    polygons = _first_present(payload, "dt_polys", "polys", "boxes")
    scores = _first_present(payload, "dt_scores", "det_scores")
    crops, meta = [], []
    for index, polygon in enumerate(polygons):
        crop = _crop_bbox(source, polygon)
        if crop is None:
            continue
        image = Image.fromarray(crop).convert("L")
        ratio = 48 / max(1, image.height)
        resized = image.resize((max(8, int(image.width * ratio)), 48), Image.Resampling.BILINEAR)
        padded = ImageOps.pad(resized, (320, 48), color=255, centering=(0, 0))
        crops.append(torch.from_numpy(np.asarray(padded, dtype=np.float32) / 255.0).unsqueeze(0))
        meta.append((polygon, scores[index] if index < len(scores) else 0.0))
    svtr_tokens = []
    if crops:
        with torch.inference_mode():
            logits = model(torch.stack(crops))
            texts = ctc_greedy_decode(logits, checkpoint["charset"])
            conf = logits.exp().max(dim=-1).values.mean(dim=0).tolist()
        for index, text in enumerate(texts):
            if not text or not _looks_date_like(text):
                continue
            polygon, det_score = meta[index]
            bbox = _bbox_from_poly(polygon)
            extras = {"polygon": polygon.tolist() if hasattr(polygon, "tolist") else polygon, "source": "svtr_local_roi"}
            if bbox:
                left, top, right, bottom = bbox
                extras.update({"center_x": (left + right) / 2, "center_y": (top + bottom) / 2, "width": right - left, "height": bottom - top})
            svtr_tokens.append({"text": text, "confidence": float(conf[index]), "bbox": bbox, "extras": extras, "detection_confidence": float(det_score or 0.0)})
    merged.append({"image_id": image_id, "tokens": b4_tokens.get(image_id, []) + svtr_tokens})

merged_path = OUT / "svtr_merged_tokens.jsonl"
merged_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in merged), encoding="utf-8")
cache_input = OUT / "cache_input"
cache_input.mkdir(exist_ok=True)
for image_id in IDS:
    (cache_input / f"{image_id}.jpg").touch()
(OUT / "runtime.json").write_text(json.dumps({"wall_seconds": time.perf_counter() - started, "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, "image_count": len(IDS)}, indent=2), encoding="utf-8")
print(merged_path)
