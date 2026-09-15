"""Small SVTR-inspired CTC recognizer used for research prototypes.

This is intentionally separate from the production PaddleOCR adapters. It is
an attention-based visual sequence model for synthetic date/number crops;
the checkpoint must be trained before it can be used in an OCR run.
"""

from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor, nn


class SVTRTinyCTC(nn.Module):
    """Compact local/global mixing recognizer with a CTC head.

    The model follows the SVTR family idea: a convolutional stem produces a
    2-D feature map, height is collapsed into a horizontal sequence, local
    depthwise mixing is applied, and a small Transformer encoder provides
    global context before CTC classification.
    """

    def __init__(self, num_classes: int, d_model: int = 96, heads: int = 4, layers: int = 2) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, d_model, 3, padding=1), nn.BatchNorm2d(d_model), nn.GELU(),
        )
        self.local_mix = nn.Conv1d(d_model, d_model, 5, padding=2, groups=d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads, dim_feedforward=d_model * 4,
            dropout=0.0, batch_first=True, norm_first=False, activation="gelu",
        )
        self.global_mix = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, images: Tensor) -> Tensor:
        features = self.stem(images)
        sequence = features.mean(dim=2).transpose(1, 2)
        sequence = sequence + self.local_mix(sequence.transpose(1, 2)).transpose(1, 2)
        sequence = self.global_mix(sequence)
        logits = self.classifier(sequence)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
        return logits.log_softmax(dim=-1).transpose(0, 1)


class CRNNCTC(nn.Module):
    """Compact CNN-BiLSTM-CTC control model from the CRNN family."""

    def __init__(self, num_classes: int, hidden: int = 96) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, hidden, 3, padding=1), nn.BatchNorm2d(hidden), nn.ReLU(),
        )
        self.sequence = nn.LSTM(hidden, hidden, num_layers=2, bidirectional=True)
        self.classifier = nn.Linear(hidden * 2, num_classes)

    def encode(self, images: Tensor) -> Tensor:
        features = self.cnn(images).mean(dim=2).permute(2, 0, 1)
        return self.sequence(features)[0]

    def forward(self, images: Tensor) -> Tensor:
        return self.classifier(self.encode(images)).log_softmax(dim=-1)


class CRNNAttention(nn.Module):
    """CRNN encoder with a compact additive-attention decoder.

    This is an accuracy upper-bound prototype: autoregressive decoding is
    intentionally explicit so its CPU latency can be measured against CTC.
    """

    def __init__(self, num_classes: int, hidden: int = 96, embedding: int = 64) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.bos = num_classes
        self.eos = num_classes + 1
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, hidden, 3, padding=1), nn.BatchNorm2d(hidden), nn.ReLU(),
        )
        self.encoder = nn.LSTM(hidden, hidden, num_layers=2, bidirectional=True)
        context = hidden * 2
        self.embedding = nn.Embedding(num_classes + 2, embedding)
        self.query = nn.Linear(embedding, context)
        self.key = nn.Linear(context, context)
        self.score = nn.Linear(context, 1)
        self.cell = nn.GRUCell(embedding + context, context)
        self.output = nn.Linear(context, num_classes + 2)

    def encode(self, images: Tensor) -> Tensor:
        features = self.cnn(images).mean(dim=2).permute(2, 0, 1)
        return self.encoder(features)[0].permute(1, 0, 2)

    def _step(self, memory: Tensor, state: Tensor, token: Tensor) -> tuple[Tensor, Tensor]:
        embedded = self.embedding(token)
        query = self.query(embedded).unsqueeze(1)
        keys = self.key(memory)
        weights = self.score(torch.tanh(keys + query)).squeeze(-1).softmax(dim=1)
        context = torch.bmm(weights.unsqueeze(1), memory).squeeze(1)
        state = self.cell(torch.cat([embedded, context], dim=1), state)
        return self.output(state), state

    def forward(self, images: Tensor, targets: Tensor | None = None, max_length: int = 16) -> Tensor:
        memory = self.encode(images)
        state = memory.new_zeros((images.size(0), memory.size(-1)))
        token = torch.full((images.size(0),), self.bos, dtype=torch.long, device=images.device)
        outputs = []
        steps = targets.size(1) if targets is not None else max_length
        for step in range(steps):
            logits, state = self._step(memory, state, token)
            outputs.append(logits)
            token = targets[:, step] if targets is not None else logits.argmax(dim=-1)
        return torch.stack(outputs, dim=1)


def ctc_greedy_decode(logits: Tensor, charset: str, blank: int = 0) -> list[str]:
    """Decode CTC logits into strings for quick research evaluation."""

    indices = logits.argmax(dim=-1).transpose(0, 1).detach().cpu().tolist()
    results: list[str] = []
    for sequence in indices:
        output: list[str] = []
        previous = blank
        for index in sequence:
            if index != blank and index != previous and 0 < index <= len(charset):
                output.append(charset[index - 1])
            previous = index
        results.append("".join(output))
    return results


def charset_from_labels(labels: Iterable[str]) -> str:
    return "".join(sorted(set("".join(labels))))
