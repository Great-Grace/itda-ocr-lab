#!/usr/bin/env python3
"""Train a small SVTR-inspired CTC recognizer on synthetic date strings."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ocr_lab.modules.attention_svtr import SVTRTinyCTC, charset_from_labels, ctc_greedy_decode


def _fonts() -> list[str]:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    ]
    return [path for path in candidates if Path(path).exists()]


def _date_strings(count: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    formats = [
        lambda y, m, d: f"{y:04d}.{m:02d}.{d:02d}",
        lambda y, m, d: f"{y:04d}-{m:02d}-{d:02d}",
        lambda y, m, d: f"{y:04d}/{m:02d}/{d:02d}",
        lambda y, m, d: f"{y:04d}{m:02d}{d:02d}",
        lambda y, m, d: f"{y % 100:02d}.{m:02d}.{d:02d}",
        lambda y, m, d: f"{d:02d}.{m:02d}.{y:04d}",
    ]
    values: list[str] = []
    for _ in range(count):
        y, m, d = rng.randint(2020, 2028), rng.randint(1, 12), rng.randint(1, 28)
        values.append(rng.choice(formats)(y, m, d))
    return values


def _render(text: str, rng: random.Random, fonts: list[str]) -> torch.Tensor:
    font_path = rng.choice(fonts) if fonts else None
    font = ImageFont.truetype(font_path, rng.randint(24, 34)) if font_path else ImageFont.load_default()
    width = max(120, int(font.getlength(text)) + 24)
    image = Image.new("L", (width, 48), color=rng.randint(205, 255))
    draw = ImageDraw.Draw(image)
    draw.text((10, rng.randint(3, 10)), text, fill=rng.randint(0, 70), font=font)
    if rng.random() < 0.35:
        image = image.rotate(rng.uniform(-7, 7), resample=Image.Resampling.BILINEAR, expand=False, fillcolor=235)
    if rng.random() < 0.25:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.1, 0.7)))
    array = np.asarray(image, dtype=np.float32)
    noise = np.random.default_rng(rng.randint(0, 10_000_000)).normal(0, rng.uniform(0, 10), array.shape)
    array = np.clip(array + noise, 0, 255).astype(np.uint8)
    image = Image.fromarray(array, mode="L")
    image = ImageOps.pad(image, (320, 48), color=255, centering=(0, 0))
    return torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).unsqueeze(0)


class SyntheticDateDataset(Dataset[tuple[torch.Tensor, str]]):
    def __init__(self, labels: list[str], seed: int) -> None:
        self.labels, self.seed, self.fonts = labels, seed, _fonts()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        return _render(self.labels[index], random.Random(self.seed + index * 7919), self.fonts), self.labels[index]


def _collate(batch: list[tuple[torch.Tensor, str]], charset: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    images, labels = zip(*batch)
    targets = torch.tensor([charset.index(char) + 1 for label in labels for char in label], dtype=torch.long)
    lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)
    return torch.stack(images), targets, lengths


def _edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for i, char in enumerate(left, 1):
        current = [i]
        for j, other in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (char != other)))
        previous = current
    return previous[-1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-count", type=int, default=20000)
    parser.add_argument("--val-count", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    train_labels = _date_strings(args.train_count, args.seed)
    val_labels = _date_strings(args.val_count, args.seed + 1)
    charset = charset_from_labels(train_labels + val_labels)
    train_loader = DataLoader(SyntheticDateDataset(train_labels, args.seed), batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=lambda b: _collate(b, charset))
    val_loader = DataLoader(SyntheticDateDataset(val_labels, args.seed + 1), batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=lambda b: _collate(b, charset))
    model = SVTRTinyCTC(len(charset), d_model=args.d_model, heads=args.heads, layers=args.layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4, eps=1e-6)
    loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
    history = []
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train(); train_loss = 0.0
        for images, targets, lengths in train_loader:
            images, targets, lengths = images.to(device), targets.to(device), lengths.to(device)
            logits = model(images)
            input_lengths = torch.full((images.size(0),), logits.size(0), dtype=torch.long, device=device)
            loss = loss_fn(logits, targets, input_lengths, lengths)
            if not torch.isfinite(loss):
                print(json.dumps({"warning": "non_finite_loss", "epoch": epoch}), flush=True)
                break
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); optimizer.step(); train_loss += float(loss.item())
        model.eval(); correct = total = edit = chars = 0; examples = []
        with torch.no_grad():
            for images, _, labels in val_loader:
                predictions = ctc_greedy_decode(model(images.to(device)), charset)
                targets_text = [val_labels[index] for index in range(total, min(total + len(predictions), len(val_labels)))]
                correct += sum(pred == target for pred, target in zip(predictions, targets_text)); total += len(predictions)
                edit += sum(_edit_distance(pred, target) for pred, target in zip(predictions, targets_text)); chars += sum(max(1, len(target)) for target in targets_text)
                if len(examples) < 3:
                    examples.extend(list(zip(predictions, targets_text))[:3 - len(examples)])
        row = {"epoch": epoch, "train_loss": train_loss / max(1, len(train_loader)), "val_exact": correct / max(1, total), "val_cer": edit / max(1, chars), "examples": examples}
        history.append(row); print(json.dumps(row), flush=True)
        torch.save({"state_dict": model.state_dict(), "charset": charset, "model": "SVTRTinyCTC", "epoch": epoch}, output / f"checkpoint_epoch_{epoch:03d}.pt")
    torch.save({"state_dict": model.state_dict(), "charset": charset, "model": "SVTRTinyCTC"}, output / "checkpoint.pt")
    (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output / "model.json").write_text(json.dumps({"charset": charset, "architecture": "SVTRTinyCTC", "d_model": args.d_model, "heads": args.heads, "layers": args.layers, "input_shape": [1, 48, 320]}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
