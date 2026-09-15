"""Pilot fine-tuning of SVTR-Tiny on weakly labeled real date crops."""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ocr_lab.modules.attention_svtr import SVTRTinyCTC, ctc_greedy_decode


class CropDataset(Dataset):
    def __init__(self, rows, root, charset, augment=False, target_field="raw_text"):
        self.rows, self.root, self.charset, self.augment, self.target_field = rows, root, charset, augment, target_field

    def __len__(self): return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = Image.open(self.root / row["image"]).convert("L")
        if self.augment and random.random() < 0.5:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 0.5)))
        array = np.asarray(image, dtype=np.float32) / 255.0
        target = row[self.target_field]
        return torch.from_numpy(array).unsqueeze(0), target


def collate(batch, charset):
    images, labels = zip(*batch)
    targets = torch.tensor([charset.index(c) + 1 for label in labels for c in label], dtype=torch.long)
    lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)
    return torch.stack(images), targets, lengths, list(labels)


def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1): current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crops", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--target-field", choices=["raw_text", "label"], default="raw_text")
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()
    root = Path(args.crops)
    rows = [row for row in csv.DictReader((root / "labels.csv").open()) if "None" not in row["label"]]
    rows.sort(key=lambda row: row["image_id"])
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    charset = checkpoint["charset"]
    rows = [row for row in rows if set(row[args.target_field]) <= set(charset)]
    split = max(1, int(len(rows) * 0.8))
    train_rows, val_rows = rows[:split], rows[split:]
    model = SVTRTinyCTC(len(charset)); model.load_state_dict(checkpoint["state_dict"]); model.train()
    train_loader = DataLoader(CropDataset(train_rows, root, charset, augment=True, target_field=args.target_field), batch_size=16, shuffle=True, num_workers=0, collate_fn=lambda b: collate(b, charset))
    val_loader = DataLoader(CropDataset(val_rows, root, charset, target_field=args.target_field), batch_size=16, shuffle=False, num_workers=0, collate_fn=lambda b: collate(b, charset))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4, eps=1e-6)
    loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
    history=[]
    for epoch in range(1,args.epochs+1):
        model.train(); loss_total=0
        for images, targets, lengths, _ in train_loader:
            logits=model(images); input_lengths=torch.full((images.size(0),),logits.size(0),dtype=torch.long)
            loss=loss_fn(logits,targets,input_lengths,lengths); optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),2.0); optimizer.step(); loss_total+=float(loss.item())
        model.eval(); correct=total=edit=chars=0
        with torch.no_grad():
            for images, _, _, labels in val_loader:
                pred=ctc_greedy_decode(model(images),charset); correct+=sum(a==b for a,b in zip(pred,labels)); total+=len(labels); edit+=sum(edit_distance(a,b) for a,b in zip(pred,labels)); chars+=sum(max(1,len(b)) for b in labels)
        row={"epoch":epoch,"train_loss":loss_total/max(1,len(train_loader)),"val_exact":correct/max(1,total),"val_cer":edit/max(1,chars)}; history.append(row); print(json.dumps(row),flush=True)
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True); torch.save({"state_dict":model.state_dict(),"charset":charset,"model":"SVTRTinyCTC_REAL_WEAK"},out/"checkpoint.pt"); (out/"history.json").write_text(json.dumps(history,indent=2))

if __name__ == "__main__": main()
