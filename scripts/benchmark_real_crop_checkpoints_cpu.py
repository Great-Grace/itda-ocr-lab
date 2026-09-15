#!/usr/bin/env python3
"""CPU crop-level sanity/latency benchmark for real-crop checkpoints."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from ocr_lab.modules.neural_recognizer_suite import create_neural_recognizer, ctc_greedy_decode


def edit_distance(a: str, b: str) -> int:
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        nxt = [i]
        for j, y in enumerate(b, 1):
            nxt.append(min(nxt[-1] + 1, row[j] + 1, nxt[-1] + 1, row[j - 1] + (x != y)))
        row = nxt
    return row[-1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--crops", required=True)
    parser.add_argument("--checkpoints", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    rows = list(csv.DictReader((Path(args.crops) / "labels.csv").open(encoding="utf-8")))
    rows = [row for row in rows if len("".join(c for c in row["label"] if c.isdigit())) == 8]
    source_images, targets = [], []
    for row in rows:
        image = Image.open(Path(args.crops) / row["image"]).convert("L")
        source_images.append(image.copy())
        targets.append("".join(c for c in row["label"] if c.isdigit()))
    batches: dict[tuple[int, int], torch.Tensor] = {}

    def batch_for(image_size: tuple[int, int]) -> torch.Tensor:
        if image_size not in batches:
            tensors = [torch.from_numpy(np.asarray(ImageOps.pad(image, image_size, color=255, centering=(0, 0)), dtype=np.float32) / 255.0).unsqueeze(0) for image in source_images]
            batches[image_size] = torch.stack(tensors)
        return batches[image_size]
    results = []
    for ckpt_path in sorted(Path(args.checkpoints).rglob("best.pt")):
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        name = checkpoint["architecture"]
        base = checkpoint.get("base_architecture", name)
        params = checkpoint.get("model_params", {})
        input_size = tuple(checkpoint.get("input_size", (320, 48)))
        attention = base == "crnn_bilstm_attn"
        model = create_neural_recognizer(base, 10 if attention else 11, **params); model.load_state_dict(checkpoint["state_dict"]); model.eval()
        batch = batch_for(input_size)
        with torch.inference_mode():
            _ = model(batch[:1], None, max_length=10) if attention else model(batch[:1])
        timings = []
        predictions = []
        with torch.inference_mode():
            for start in range(0, len(source_images), 32):
                current = batch[start:start + 32]
                t0 = time.perf_counter(); output = model(current, None, max_length=10) if attention else model(current); timings.append((time.perf_counter() - t0) * 1000 / len(current))
                if attention:
                    for sequence in output.argmax(dim=-1).tolist():
                        text = "".join(str(index) for index in sequence if 0 <= index < 10)
                        predictions.append(text[:8])
                else:
                    predictions.extend(ctc_greedy_decode(output, "0123456789"))
        exact = sum(p == t for p, t in zip(predictions, targets)) / max(1, len(targets))
        cer = sum(edit_distance(p, t) for p, t in zip(predictions, targets)) / max(1, sum(len(t) for t in targets))
        results.append({"architecture": name, "base_architecture": base, "model_params": params, "input_size": input_size, "checkpoint": str(ckpt_path), "crop_count": len(targets), "crop_exact": exact, "crop_cer": cer, "median_ms_per_crop": float(np.median(timings)), "p95_ms_per_crop": float(np.percentile(timings, 95))})
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
