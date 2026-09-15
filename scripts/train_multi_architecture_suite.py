#!/usr/bin/env python3
"""Unified Multi-Architecture Training Engine for OCR Date Recognition.

Trains and evaluates any recognizer architecture from neural_recognizer_suite:
- svtr_tiny_ctc
- crnn_bilstm_ctc
- crnn_bigru_ctc
- crnn_bilstm_attn
- mobilenetv3_ctc
- mobilenetv3_bilstm_ctc
- resnet18_bilstm_ctc

Powered by:
- Authoritative 3,063 clean ground truth dataset (data/clean_training_set_3066.csv)
- Domain-specific on-the-fly 3x dynamic augmentation (dot-matrix erosion, rotation +-5 deg, blur, noise)
- Strictly label-preserving (no flips, no cutouts)
- Isolated checkpoints saved to weights/<arch>/checkpoint.pt
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ocr_lab.modules.neural_recognizer_suite import (
    create_neural_recognizer,
    charset_from_labels,
    ctc_greedy_decode,
    CRNNAttention,
)


def _available_fonts() -> list[str]:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    return [p for p in candidates if Path(p).exists()]


def load_clean_ground_truth_texts(csv_path: Path) -> list[str]:
    """Extract all text targets from verified 3,063 high-confidence samples."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"Clean training dataset not found at {csv_path}")

    texts: list[str] = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            raw = r.get("text_found", "").strip()
            if not raw:
                continue
            for line in raw.split("\n"):
                line = line.strip()
                if len(line) >= 3:
                    texts.append(line)

    # De-duplicate while preserving order
    unique_texts = list(dict.fromkeys(texts))
    print(f"Loaded {len(unique_texts)} unique high-precision ground truth date texts from {csv_path.name}")
    return unique_texts


def _render_augmented_crop(
    text: str,
    rng: random.Random,
    fonts: list[str],
    augment: bool = True,
    target_h: int = 48,
    target_w: int = 320,
) -> torch.Tensor:
    font_path = rng.choice(fonts) if fonts else None
    font_size = rng.randint(22, 30)
    font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
    try:
        w_text = int(font.getlength(text))
    except Exception:
        w_text = len(text) * 12
    img_w = max(120, w_text + 28)
    bg_val = rng.randint(210, 255)
    fg_val = rng.randint(0, 60)
    image = Image.new("L", (img_w, target_h), color=bg_val)
    draw = ImageDraw.Draw(image)
    draw.text((10, rng.randint(4, 9)), text, fill=fg_val, font=font)

    if augment:
        # 1. Dot-matrix / Inkjet degradation (prob 0.35)
        if rng.random() < 0.35:
            arr = np.asarray(image).copy()
            arr[::3, ::3] = np.clip(arr[::3, ::3] + rng.randint(40, 100), 0, 255)
            image = Image.fromarray(arr, mode="L")

        # 2. Geometric: Slight Rotation & Affine Shear (prob 0.40, bounded to +-5 deg)
        if rng.random() < 0.40:
            angle = rng.uniform(-5.0, 5.0)
            image = image.rotate(angle, resample=Image.Resampling.BILINEAR, expand=False, fillcolor=bg_val)

        # 3. Photometric: Soft Gaussian Blur & Noise (prob 0.35)
        if rng.random() < 0.35:
            image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.15, 0.55)))

        arr = np.asarray(image, dtype=np.float32)
        noise = np.random.default_rng(rng.randint(0, 10_000_000)).normal(0, rng.uniform(0, 7), arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        image = Image.fromarray(arr, mode="L")

    image = ImageOps.pad(image, (target_w, target_h), color=255, centering=(0, 0))
    tensor = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).unsqueeze(0)
    return tensor


class CleanGroundTruthDataset(Dataset[tuple[torch.Tensor, str]]):
    """Dynamic on-the-fly augmented dataset from verified ground truth texts."""

    def __init__(self, texts: list[str], multiplier: int = 3, augment: bool = True, seed: int = 42) -> None:
        self.texts = texts
        self.multiplier = multiplier
        self.augment = augment
        self.seed = seed
        self.fonts = _available_fonts()
        self.total_len = len(self.texts) * multiplier

    def __len__(self) -> int:
        return self.total_len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        text_idx = idx % len(self.texts)
        text = self.texts[text_idx]
        rng = random.Random(self.seed + idx * 7919)
        crop_tensor = _render_augmented_crop(text, rng, self.fonts, augment=self.augment)
        return crop_tensor, text


def _collate_ctc(batch: list[tuple[torch.Tensor, str]], charset: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    images, labels = zip(*batch)
    targets = torch.tensor([charset.index(char) + 1 for label in labels for char in label], dtype=torch.long)
    lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)
    return torch.stack(images), targets, lengths, list(labels)


def _collate_attn(batch: list[tuple[torch.Tensor, str]], charset: str) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    images, labels = zip(*batch)
    n = len(charset)
    max_len = max(len(label) for label in labels) + 1
    targets = torch.full((len(labels), max_len), -100, dtype=torch.long)
    for row, label in enumerate(labels):
        targets[row, :len(label)] = torch.tensor([charset.index(char) for char in label])
        targets[row, len(label)] = n + 1  # EOS
    return torch.stack(images), targets, list(labels)


def _edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_single_architecture(
    arch_name: str,
    output_dir: Path,
    train_texts: list[str],
    val_texts: list[str],
    charset: str,
    epochs: int = 8,
    batch_size: int = 128,
    multiplier: int = 3,
    lr: float = 3e-4,
    device_str: str = "cuda",
    seed: int = 42,
) -> dict:
    start_time = time.time()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device(device_str if (device_str == "cpu" or torch.cuda.is_available()) else "cpu")
    print(f"\n=======================================================")
    print(f"[{arch_name}] Starting training on {device} ({epochs} epochs, batch {batch_size}, {multiplier}x dynamic aug)")
    print(f"=======================================================")

    num_classes = len(charset)
    is_attn = "attn" in arch_name.lower()

    train_dataset = CleanGroundTruthDataset(train_texts, multiplier=multiplier, augment=True, seed=seed)
    val_dataset = CleanGroundTruthDataset(val_texts, multiplier=1, augment=False, seed=seed + 100)

    if is_attn:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda b: _collate_attn(b, charset))
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda b: _collate_attn(b, charset))
        model = create_neural_recognizer(arch_name, num_classes).to(device)
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda b: _collate_ctc(b, charset))
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda b: _collate_ctc(b, charset))
        model = create_neural_recognizer(arch_name, num_classes + 1).to(device)
        loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)

    param_count = count_parameters(model)
    print(f"[{arch_name}] Model Parameters: {param_count:,} ({param_count * 4 / 1024 / 1024:.2f} MB)")
    print(f"[{arch_name}] Training set: {len(train_dataset)} augmented crops/epoch | Val set: {len(val_dataset)} clean crops")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    history = []
    best_exact = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        t_epoch_start = time.time()

        for batch in train_loader:
            optimizer.zero_grad()
            images = batch[0].to(device)

            if is_attn:
                targets = batch[1].to(device)
                logits = model(images, targets[:, :-1])
                loss = loss_fn(logits.reshape(-1, logits.size(-1)), targets[:, 1:].reshape(-1))
            else:
                targets, lengths, _ = batch[1].to(device), batch[2].to(device), batch[3]
                logits = model(images)
                input_lengths = torch.full((images.size(0),), logits.size(0), dtype=torch.long, device=device)
                loss = loss_fn(logits, targets, input_lengths, lengths)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.item())
            batches += 1

        scheduler.step()
        avg_train_loss = total_loss / max(1, batches)

        # Validation on held-out clean targets
        model.eval()
        val_correct = 0
        val_total = 0
        val_dist = 0
        val_chars = 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch[0].to(device)
                labels = batch[-1]

                if is_attn:
                    # Quick greedy argmax decoding for attention
                    logits = model(images, None, max_length=24).argmax(dim=-1).cpu().tolist()
                    preds = []
                    for seq in logits:
                        chars = []
                        for idx in seq:
                            if idx == num_classes + 1:  # EOS
                                break
                            if 0 <= idx < num_classes:
                                chars.append(charset[idx])
                        preds.append("".join(chars))
                else:
                    logits = model(images)
                    preds = ctc_greedy_decode(logits, charset)

                for p_str, g_str in zip(preds, labels):
                    val_correct += int(p_str == g_str)
                    val_total += 1
                    dist = _edit_distance(p_str, g_str)
                    val_dist += dist
                    val_chars += max(1, len(g_str))

        exact_match = val_correct / max(1, val_total)
        cer = val_dist / max(1, val_chars)
        epoch_sec = time.time() - t_epoch_start

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_exact": round(exact_match, 4),
            "val_cer": round(cer, 4),
            "epoch_sec": round(epoch_sec, 2),
        }
        history.append(epoch_record)
        print(f"  Epoch {epoch:02d}/{epochs:02d}: Loss={avg_train_loss:.4f} | Val Exact={exact_match * 100:.1f}% | CER={cer:.3f} | {epoch_sec:.1f}s", flush=True)

        if exact_match > best_exact:
            best_exact = exact_match

    training_time = time.time() - start_time
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / "checkpoint.pt"

    torch.save(
        {
            "architecture": arch_name,
            "state_dict": model.state_dict(),
            "charset": charset,
            "num_classes": num_classes,
            "param_count": param_count,
            "best_exact": best_exact,
            "final_exact": exact_match,
            "final_cer": cer,
            "history": history,
            "training_time_sec": training_time,
            "dataset_source": "clean_training_set_3066.csv",
        },
        ckpt_path,
    )
    (output_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"[{arch_name}] SAVED checkpoint to {ckpt_path} (Best Exact: {best_exact * 100:.1f}%, Time: {training_time:.1f}s)\n", flush=True)

    return {
        "architecture": arch_name,
        "checkpoint": str(ckpt_path),
        "param_count": param_count,
        "best_exact": best_exact,
        "final_exact": exact_match,
        "final_cer": cer,
        "training_time_sec": training_time,
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-architecture neural OCR training on disjoint clean dataset splits.")
    parser.add_argument("--arch", default="all", help="Architecture name or 'all'")
    parser.add_argument("--train-csv", default="data/splits/train.csv", help="Path to disjoint train CSV")
    parser.add_argument("--val-csv", default="data/splits/val.csv", help="Path to disjoint validation CSV")
    parser.add_argument("--output-root", default="weights", help="Root directory for weights")
    parser.add_argument("--epochs", type=int, default=8, help="Epochs per architecture")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--multiplier", type=int, default=3, help="On-the-fly augmentation multiplier")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load disjoint train and validation ground truth texts
    train_texts = load_clean_ground_truth_texts(ROOT / args.train_csv)
    val_texts = load_clean_ground_truth_texts(ROOT / args.val_csv)
    charset = charset_from_labels(train_texts + val_texts)

    print(f"Charset length: {len(charset)} unique characters")
    print(f"Disjoint Train targets: {len(train_texts)} | Disjoint Val targets: {len(val_texts)} | Test Set: strictly isolated in data/splits/test.csv")

    arch_list = [
        "svtr_tiny_ctc",
        "crnn_bilstm_ctc",
        "crnn_bigru_ctc",
        "mobilenetv3_ctc",
        "mobilenetv3_bilstm_ctc",
        "resnet18_bilstm_ctc",
        "crnn_bilstm_attn",
    ]

    selected = arch_list if args.arch == "all" else [args.arch]
    results = {}

    print(f"\n=== Launching Multi-Architecture Training: {len(selected)} Models on 3,063 Ground Truth ===")
    for arch in selected:
        out_d = Path(args.output_root) / arch
        summary = train_single_architecture(
            arch_name=arch,
            output_dir=out_d,
            train_texts=train_texts,
            val_texts=val_texts,
            charset=charset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            multiplier=args.multiplier,
            lr=args.lr,
            device_str=args.device,
            seed=args.seed,
        )
        results[arch] = summary

    summary_file = Path(args.output_root) / "training_summary.json"
    summary_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"=== ALL {len(selected)} ARCHITECTURES TRAINED SUCCESSFULLY ===")
    print(f"Summary written to {summary_file}")


if __name__ == "__main__":
    main()
