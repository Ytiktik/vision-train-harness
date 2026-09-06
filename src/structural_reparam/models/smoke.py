"""Minimal models used by end-to-end smoke experiments."""

from __future__ import annotations

import torch
from torch import nn


class ConstantLogitClassifier(nn.Module):
    """Classifier with one trainable logits vector, independent of the input."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(num_classes))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.logits.unsqueeze(0).expand(images.shape[0], -1)
