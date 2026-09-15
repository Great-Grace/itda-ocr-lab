#!/usr/bin/env python3
"""Train CRNN-CTC or CRNN-attention on the same synthetic date corpus."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ocr_lab.modules.attention_svtr import CRNNAttention, CRNNCTC, ctc_greedy_decode, charset_from_labels
from train_synthetic_svtr_tiny import SyntheticDateDataset, _collate, _date_strings


def _edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for i, char in enumerate(left, 1):
        current = [i]
        for j, other in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (char != other)))
        previous = current
    return previous[-1]


def _attention_collate(batch: list[tuple[torch.Tensor, str]], charset: str) -> tuple[torch.Tensor, torch.Tensor]:
    images, labels = zip(*batch)
    n = len(charset)
    targets = torch.full((len(labels), max(len(label) for label in labels) + 1), -100, dtype=torch.long)
    for row, label in enumerate(labels):
        targets[row, :len(label)] = torch.tensor([charset.index(char) for char in label])
        targets[row, len(label)] = n + 1
    return torch.stack(images), targets


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=["crnn_ctc", "crnn_attn"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-count", type=int, default=20000)
    parser.add_argument("--val-count", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    train_labels = _date_strings(args.train_count, args.seed)
    val_labels = _date_strings(args.val_count, args.seed + 1)
    charset = charset_from_labels(train_labels + val_labels)
    if args.architecture == "crnn_ctc":
        train_loader = DataLoader(SyntheticDateDataset(train_labels, args.seed), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=lambda b: _collate(b, charset))
        val_loader = DataLoader(SyntheticDateDataset(val_labels, args.seed + 1), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=lambda b: _collate(b, charset))
        model = CRNNCTC(len(charset) + 1).to(device)
        loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
    else:
        train_loader = DataLoader(SyntheticDateDataset(train_labels, args.seed), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=lambda b: _attention_collate(b, charset))
        val_loader = DataLoader(SyntheticDateDataset(val_labels, args.seed + 1), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=lambda b: _attention_collate(b, charset))
        model = CRNNAttention(len(charset)).to(device)
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4, eps=1e-6)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); running = 0.0
        for batch in train_loader:
            images = batch[0].to(device); optimizer.zero_grad()
            if args.architecture == "crnn_ctc":
                targets, lengths = batch[1].to(device), batch[2].to(device)
                logits = model(images); input_lengths = torch.full((images.size(0),), logits.size(0), dtype=torch.long, device=device)
                loss = loss_fn(logits, targets, input_lengths, lengths)
            else:
                targets = batch[1].to(device)
                logits = model(images, targets[:, :-1])
                loss = loss_fn(logits.reshape(-1, logits.size(-1)), targets[:, 1:].reshape(-1))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step(); running += float(loss.item())
        model.eval(); correct = total = edit = chars = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch[0].to(device)
                if args.architecture == "crnn_ctc":
                    predictions = ctc_greedy_decode(model(images), charset)
                else:
                    logits = model(images, None, max_length=16).argmax(dim=-1).cpu().tolist()
                    predictions = []
                    for seq in logits:
                        text_chars = []
                        for index in seq:
                            if index == len(charset) + 1:
                                break
                            if 0 <= index < len(charset):
                                text_chars.append(charset[index])
                        text = "".join(text_chars)
                        predictions.append(text)
                # Dataset order is deterministic because validation loader is not shuffled.
                targets_text = val_labels[total:total + len(predictions)]
                correct += sum(a == b for a, b in zip(predictions, targets_text)); total += len(predictions)
                edit += sum(_edit_distance(a, b) for a, b in zip(predictions, targets_text)); chars += sum(max(1, len(b)) for b in targets_text)
        row = {"epoch": epoch, "train_loss": running / max(1, len(train_loader)), "val_exact": correct / max(1, total), "val_cer": edit / max(1, chars)}
        history.append(row); print(json.dumps(row), flush=True)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "charset": charset, "architecture": args.architecture}, output / "checkpoint.pt")
    (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
