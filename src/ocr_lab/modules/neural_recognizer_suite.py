"""Neural Recognizer Architecture Suite for OCR Date Recognition.

Provides a unified combinatorial suite of recognizer models:
- Backbones: SVTR-Tiny, CNN, MobileNetV3, ResNet18
- Sequence Modeling: BiLSTM, BiGRU, Transformer, None
- Prediction Heads: CTC, Attention
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import torch
from torch import Tensor, nn


def charset_from_labels(labels: Iterable[str]) -> str:
    """Build unique sorted character set string from an iterable of labels."""
    return "".join(sorted(set("".join(labels))))


def ctc_greedy_decode(logits: Tensor, charset: str, blank: int = 0) -> list[str]:
    """Greedy CTC decoder for sequence logits of shape [Time, Batch, Classes]."""
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


# -------------------------------------------------------------------------
# 1. SVTR-Tiny CTC (Vision Transformer + Local Depthwise Mixing)
# -------------------------------------------------------------------------
class SVTRTinyCTC(nn.Module):
    """Compact local/global mixing recognizer with a CTC head."""

    def __init__(self, num_classes: int, d_model: int = 96, heads: int = 4, layers: int = 2) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.d_model = d_model
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, d_model, 3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
        )
        self.local_mix = nn.Conv1d(d_model, d_model, 5, padding=2, groups=d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * 4,
            dropout=0.0,
            batch_first=True,
            norm_first=False,
            activation="gelu",
        )
        self.global_mix = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, images: Tensor) -> Tensor:
        # images: [B, 1, H, W]
        features = self.stem(images)
        sequence = features.mean(dim=2).transpose(1, 2)  # [B, W', D]
        sequence = sequence + self.local_mix(sequence.transpose(1, 2)).transpose(1, 2)
        sequence = self.global_mix(sequence)
        logits = self.classifier(sequence)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
        return logits.log_softmax(dim=-1).transpose(0, 1)  # [T, B, C]


# -------------------------------------------------------------------------
# 2. CRNN-BiLSTM-CTC & CRNN-BiGRU-CTC (CNN Backbone + Recurrent Sequence)
# -------------------------------------------------------------------------
class CRNN_RNN_CTC(nn.Module):
    """Standard CNN + BiLSTM or BiGRU + CTC Recognizer."""

    def __init__(self, num_classes: int, rnn_type: str = "lstm", hidden: int = 96, layers: int = 2) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.rnn_type = rnn_type.lower()
        if self.rnn_type == "gru":
            self.sequence: nn.Module | None = nn.GRU(hidden, hidden, num_layers=layers, bidirectional=True, batch_first=False)
            classifier_width = hidden * 2
        elif self.rnn_type == "none":
            self.sequence = None
            classifier_width = hidden
        else:
            self.sequence = nn.LSTM(hidden, hidden, num_layers=layers, bidirectional=True, batch_first=False)
            classifier_width = hidden * 2
        self.classifier = nn.Linear(classifier_width, num_classes)

    def encode(self, images: Tensor) -> Tensor:
        # features: [B, hidden, H', W'] -> collapse H' -> [W', B, hidden]
        features = self.cnn(images).mean(dim=2).permute(2, 0, 1)
        if self.sequence is None:
            return features
        out, _ = self.sequence(features)
        return out

    def forward(self, images: Tensor) -> Tensor:
        encoded = self.encode(images)
        return self.classifier(encoded).log_softmax(dim=-1)  # [T, B, C]


# -------------------------------------------------------------------------
# 3. CRNN-Attention (CNN + BiLSTM + Bahdanau Autoregressive Attention Decoder)
# -------------------------------------------------------------------------
class CRNNAttention(nn.Module):
    """CRNN encoder with additive-attention autoregressive decoder."""

    def __init__(self, num_classes: int, hidden: int = 96, embedding: int = 64, encoder_layers: int = 2) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.bos = num_classes
        self.eos = num_classes + 1
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.encoder = nn.LSTM(hidden, hidden, num_layers=encoder_layers, bidirectional=True, batch_first=False)
        context = hidden * 2
        self.embedding = nn.Embedding(num_classes + 2, embedding)
        self.query = nn.Linear(embedding, context)
        self.key = nn.Linear(context, context)
        self.score = nn.Linear(context, 1)
        self.cell = nn.GRUCell(embedding + context, context)
        self.output = nn.Linear(context, num_classes + 2)

    def encode(self, images: Tensor) -> Tensor:
        features = self.cnn(images).mean(dim=2).permute(2, 0, 1)
        return self.encoder(features)[0].permute(1, 0, 2)  # [B, T, context]

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


# -------------------------------------------------------------------------
# 4. MobileNetV3 Inverted Bottleneck Backbone (Ultra-lightweight Mobile OCR)
# -------------------------------------------------------------------------
class InvertedResidual(nn.Module):
    """MobileNetV3 Inverted Residual Block with Squeeze-and-Excitation."""

    def __init__(self, in_ch: int, hidden_ch: int, out_ch: int, stride: int = 1, use_se: bool = True) -> None:
        super().__init__()
        self.stride = stride
        self.use_res = stride == 1 and in_ch == out_ch

        layers = []
        if in_ch != hidden_ch:
            layers.extend([
                nn.Conv2d(in_ch, hidden_ch, 1, bias=False),
                nn.BatchNorm2d(hidden_ch),
                nn.Hardswish(inplace=True),
            ])
        layers.extend([
            nn.Conv2d(hidden_ch, hidden_ch, 3, stride=stride, padding=1, groups=hidden_ch, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.Hardswish(inplace=True),
        ])
        if use_se:
            layers.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(hidden_ch, max(8, hidden_ch // 4), 1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(max(8, hidden_ch // 4), hidden_ch, 1),
                    nn.Hardsigmoid(inplace=True),
                )
            )
        layers.extend([
            nn.Conv2d(hidden_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV3Recognizer(nn.Module):
    """MobileNetV3 Recognizer with optional BiLSTM and CTC head."""

    def __init__(self, num_classes: int, use_rnn: bool = True, hidden: int = 96, rnn_layers: int = 2) -> None:
        super().__init__()
        self.use_rnn = use_rnn
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.Hardswish(inplace=True),
            InvertedResidual(16, 32, 24, stride=1),
            InvertedResidual(24, 64, 40, stride=2),
            InvertedResidual(40, 96, 40, stride=1),
            InvertedResidual(40, 120, hidden, stride=1),
            nn.Conv2d(hidden, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.Hardswish(inplace=True),
        )
        if use_rnn:
            self.sequence = nn.LSTM(hidden, hidden, num_layers=rnn_layers, bidirectional=True, batch_first=False)
            self.classifier = nn.Linear(hidden * 2, num_classes)
        else:
            self.sequence = None
            self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, images: Tensor) -> Tensor:
        features = self.stem(images).mean(dim=2).permute(2, 0, 1)  # [T, B, hidden]
        if self.use_rnn and self.sequence is not None:
            features, _ = self.sequence(features)
        logits = self.classifier(features)
        return logits.log_softmax(dim=-1)


# -------------------------------------------------------------------------
# 5. ResNet18 Recognizer (Deep Residual Backbone + BiLSTM + CTC)
# -------------------------------------------------------------------------
class BasicBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.downsample = None

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)


class ResNet18Recognizer(nn.Module):
    """ResNet-18 feature extractor with BiLSTM and CTC head."""

    def __init__(self, num_classes: int, hidden: int = 128, rnn_layers: int = 2) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            BasicBlock(32, 32, stride=1),
            BasicBlock(32, 64, stride=(2, 2)),
            BasicBlock(64, 64, stride=1),
            BasicBlock(64, hidden, stride=(2, 1)),
            BasicBlock(hidden, hidden, stride=1),
        )
        self.sequence = nn.LSTM(hidden, hidden, num_layers=rnn_layers, bidirectional=True, batch_first=False)
        self.classifier = nn.Linear(hidden * 2, num_classes)

    def forward(self, images: Tensor) -> Tensor:
        features = self.stem(images).mean(dim=2).permute(2, 0, 1)  # [T, B, hidden]
        features, _ = self.sequence(features)
        logits = self.classifier(features)
        return logits.log_softmax(dim=-1)


# -------------------------------------------------------------------------
# Architecture Factory Function
# -------------------------------------------------------------------------
def create_neural_recognizer(arch_name: str, num_classes: int, **params: int) -> nn.Module:
    """Create a recognizer model given an architecture identifier."""
    name = arch_name.lower().strip()
    if name == "svtr_tiny_ctc":
        return SVTRTinyCTC(num_classes, **params)
    elif name == "crnn_bilstm_ctc":
        return CRNN_RNN_CTC(num_classes, rnn_type="lstm", **params)
    elif name == "crnn_bigru_ctc":
        return CRNN_RNN_CTC(num_classes, rnn_type="gru", **params)
    elif name == "cnn_ctc":
        return CRNN_RNN_CTC(num_classes, rnn_type="none", **params)
    elif name == "crnn_bilstm_attn":
        return CRNNAttention(num_classes, **params)
    elif name == "mobilenetv3_ctc":
        return MobileNetV3Recognizer(num_classes, use_rnn=False, **params)
    elif name == "mobilenetv3_bilstm_ctc":
        return MobileNetV3Recognizer(num_classes, use_rnn=True, **params)
    elif name == "resnet18_bilstm_ctc":
        return ResNet18Recognizer(num_classes, **params)
    else:
        raise ValueError(f"Unknown neural recognizer architecture: {arch_name}")
