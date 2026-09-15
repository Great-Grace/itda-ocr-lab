#!/usr/bin/env python3
"""Train CTC date recognizers from real, label-aligned product-image crops.

This runner intentionally rejects synthetic text rendering. Augmentation is
deterministic, train-only, and constrained to transformations that preserve
the sequence of date characters.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ocr_lab.modules.neural_recognizer_suite import create_neural_recognizer, ctc_greedy_decode


CHARSET = "0123456789"


def numeric_target(value: str) -> str | None:
    digits = "".join(char for char in value if char.isdigit())
    return digits if len(digits) == 8 else None


def edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for index, char in enumerate(left, 1):
        current = [index]
        for other_index, other in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[other_index] + 1, previous[other_index - 1] + (char != other)))
        previous = current
    return previous[-1]


@dataclass(frozen=True)
class CropRow:
    image_id: str
    image: str
    target: str


@dataclass(frozen=True)
class ArchitectureSpec:
    """One reproducible model point in the recognition screening matrix."""

    name: str
    base: str
    params: dict[str, int]


def load_rows(root: Path) -> list[CropRow]:
    rows: list[CropRow] = []
    with (root / "labels.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            target = numeric_target(row.get("label", ""))
            image = root / row.get("image", "")
            if target and image.is_file():
                rows.append(CropRow(str(row["image_id"]), str(row["image"]), target))
    if not rows:
        raise ValueError(f"No full-date crops found in {root}")
    if len({row.image_id for row in rows}) != len(rows):
        raise ValueError(f"Duplicate image IDs in {root}")
    return rows


def augment(image: Image.Image, rng: random.Random, image_size: tuple[int, int]) -> Image.Image:
    # Sequence-preserving transforms only: no flip, cutout, digit erasure, or
    # 90-degree rotation. Validation never calls this function.
    image = image.convert("L")
    if rng.random() < 0.65:
        image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.75, 1.35))
        image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.80, 1.15))
    if rng.random() < 0.35:
        image = image.rotate(rng.uniform(-4.0, 4.0), resample=Image.Resampling.BILINEAR, fillcolor=255)
    if rng.random() < 0.30:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.10, 0.55)))
    if rng.random() < 0.35:
        target_width, target_height = image_size
        width = rng.randint(int(target_width * 0.66), int(target_width * 0.95))
        image = image.resize((width, target_height), Image.Resampling.BILINEAR).resize(image_size, Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32)
    if rng.random() < 0.45:
        array += np.random.default_rng(rng.randrange(2**32)).normal(0, rng.uniform(1.0, 6.0), array.shape)
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode="L")


class RealCropDataset(Dataset):
    def __init__(self, root: Path, rows: list[CropRow], training: bool, seed: int, image_size: tuple[int, int]) -> None:
        self.root, self.rows, self.training, self.seed, self.epoch, self.image_size = root, rows, training, seed, 0, image_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        row = self.rows[index]
        image = Image.open(self.root / row.image).convert("L")
        if self.training:
            image = augment(image, random.Random(self.seed + self.epoch * 1_000_003 + index * 7_919), self.image_size)
        image = ImageOps.pad(image, self.image_size, color=255, centering=(0, 0))
        array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).unsqueeze(0), row.target


def collate(batch: list[tuple[torch.Tensor, str]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    images, labels = zip(*batch)
    targets = torch.tensor([int(char) + 1 for label in labels for char in label], dtype=torch.long)
    lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)
    return torch.stack(images), targets, lengths, list(labels)


def collate_attention(batch: list[tuple[torch.Tensor, str]]) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    images, labels = zip(*batch)
    bos, eos = len(CHARSET), len(CHARSET) + 1
    max_length = max(len(label) for label in labels) + 2
    targets = torch.full((len(labels), max_length), -100, dtype=torch.long)
    for row, label in enumerate(labels):
        values = [bos] + [int(char) for char in label] + [eos]
        targets[row, :len(values)] = torch.tensor(values, dtype=torch.long)
    return torch.stack(images), targets, list(labels)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, attention: bool = False) -> tuple[float, float]:
    model.eval()
    correct = distance = char_count = total = 0
    with torch.inference_mode():
        for batch in loader:
            images, labels = batch[0], batch[-1]
            if attention:
                logits = model(images.to(device), None, max_length=10).argmax(dim=-1).cpu().tolist()
                predictions = []
                for sequence in logits:
                    decoded_chars = []
                    for index in sequence:
                        if index == len(CHARSET) + 1:
                            break
                        if 0 <= index < len(CHARSET):
                            decoded_chars.append(CHARSET[index])
                    predictions.append("".join(decoded_chars))
            else:
                predictions = ctc_greedy_decode(model(images.to(device)), CHARSET)
            correct += sum(prediction == label for prediction, label in zip(predictions, labels))
            distance += sum(edit_distance(prediction, label) for prediction, label in zip(predictions, labels))
            char_count += sum(len(label) for label in labels)
            total += len(labels)
    return correct / max(1, total), distance / max(1, char_count)


def train_architecture(spec: ArchitectureSpec, train_root: Path, val_root: Path, train_rows: list[CropRow], val_rows: list[CropRow], output: Path, args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    attention = spec.base == "crnn_bilstm_attn"
    model = create_neural_recognizer(spec.base, len(CHARSET), **spec.params).to(device) if attention else create_neural_recognizer(spec.base, len(CHARSET) + 1, **spec.params).to(device)
    image_size = (args.image_width, args.image_height)
    train_data = RealCropDataset(train_root, train_rows, training=True, seed=args.seed, image_size=image_size)
    val_data = RealCropDataset(val_root, val_rows, training=False, seed=args.seed, image_size=image_size)
    generator = torch.Generator().manual_seed(args.seed)
    collate_fn = collate_attention if attention else collate
    loader_kwargs = {
        "num_workers": max(0, args.num_workers),
        "pin_memory": device.type == "cuda",
        "collate_fn": collate_fn,
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = max(2, args.prefetch_factor)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, generator=generator, **loader_kwargs)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
    # Exact match is the primary metric, but it is often zero during the
    # first stages of small-data OCR training.  Keep the lowest-CER checkpoint
    # among tied exact scores so ``best.pt`` is not silently frozen at epoch 1.
    best_exact, best_cer, history = -1.0, float("inf"), []
    output.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        train_data.set_epoch(epoch)
        model.train(); loss_sum = 0.0; batches = 0
        for batch in train_loader:
            images = batch[0].to(device, non_blocking=True)
            if attention:
                targets = batch[1].to(device, non_blocking=True)
                logits = model(images, targets[:, :-1])
                loss = nn.CrossEntropyLoss(ignore_index=-100)(logits.reshape(-1, logits.shape[-1]), targets[:, 1:].reshape(-1))
            else:
                targets, lengths = batch[1].to(device, non_blocking=True), batch[2].to(device, non_blocking=True)
                logits = model(images)
                input_lengths = torch.full((images.shape[0],), logits.shape[0], dtype=torch.long, device=device)
                loss = loss_fn(logits, targets, input_lengths, lengths)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss in {spec.name} epoch {epoch}")
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); optimizer.step()
            loss_sum += float(loss); batches += 1
        exact, cer = evaluate(model, val_loader, device, attention=attention)
        record = {"epoch": epoch, "train_loss": loss_sum / max(1, batches), "val_exact": exact, "val_cer": cer}
        history.append(record); print(json.dumps({"architecture": spec.name, "base_architecture": spec.base, **record}), flush=True)
        if exact > best_exact or (exact == best_exact and cer < best_cer):
            best_exact, best_cer = exact, cer
            torch.save({"architecture": spec.name, "base_architecture": spec.base, "model_params": spec.params, "input_size": image_size, "state_dict": model.state_dict(), "charset": CHARSET, "best_exact": best_exact, "epoch": epoch,
                        "best_cer": best_cer, "training_data": "real_aligned_crops", "augmentation": "train_only_sequence_preserving", "attention": attention}, output / "best.pt")
    (output / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"architecture": spec.name, "base_architecture": spec.base, "model_params": spec.params, "best_exact": best_exact, "best_cer": best_cer, "final_exact": history[-1]["val_exact"], "final_cer": history[-1]["val_cer"], "checkpoint": str(output / "best.pt")}


def load_architecture_specs(args: argparse.Namespace) -> list[ArchitectureSpec]:
    if not args.architecture_specs:
        return [ArchitectureSpec(name=name.strip(), base=name.strip(), params={}) for name in args.architectures.split(",") if name.strip()]
    raw = json.loads(Path(args.architecture_specs).read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("architecture specs must be a non-empty JSON list")
    specs: list[ArchitectureSpec] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("each architecture spec must be an object")
        name, base, params = item.get("name"), item.get("base"), item.get("params", {})
        if not isinstance(name, str) or not name or not isinstance(base, str) or not base:
            raise ValueError("each architecture spec needs non-empty name and base")
        if not isinstance(params, dict) or any(not isinstance(key, str) or not isinstance(value, int) for key, value in params.items()):
            raise ValueError(f"architecture spec {name} has invalid params")
        specs.append(ArchitectureSpec(name=name, base=base, params=params))
    if len({spec.name for spec in specs}) != len(specs):
        raise ValueError("architecture spec names must be unique")
    return specs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-crops", required=True); parser.add_argument("--val-crops", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--architectures", default="svtr_tiny_ctc,crnn_bilstm_ctc,crnn_bigru_ctc,mobilenetv3_ctc,mobilenetv3_bilstm_ctc,resnet18_bilstm_ctc")
    parser.add_argument("--architecture-specs", help="JSON list of {name, base, params}; overrides --architectures")
    parser.add_argument("--only", help="comma-separated architecture names to run from --architecture-specs")
    parser.add_argument("--epochs", type=int, default=20); parser.add_argument("--batch-size", type=int, default=64); parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0); parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--image-width", type=int, default=320); parser.add_argument("--image-height", type=int, default=48)
    parser.add_argument("--device", default="cuda"); parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.image_width < 64 or args.image_height < 16:
        raise ValueError("image dimensions are implausibly small")
    train_root, val_root, output = Path(args.train_crops), Path(args.val_crops), Path(args.output)
    train_rows, val_rows = load_rows(train_root), load_rows(val_root)
    overlap = {row.image_id for row in train_rows} & {row.image_id for row in val_rows}
    if overlap:
        raise SystemExit(f"Train/val crop leakage: {sorted(overlap)[:10]}")
    specs = load_architecture_specs(args)
    if args.only:
        wanted = {name.strip() for name in args.only.split(",") if name.strip()}
        specs = [spec for spec in specs if spec.name in wanted]
        if not specs:
            raise ValueError(f"--only did not match any architecture: {sorted(wanted)}")
    results = [train_architecture(spec, train_root, val_root, train_rows, val_rows, output / spec.name, args) for spec in specs]
    (output / "suite_summary.json").write_text(json.dumps({"train_crops": len(train_rows), "val_crops": len(val_rows), "architectures": results}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
