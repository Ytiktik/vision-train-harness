"""Tiny deterministic dataset for end-to-end training smoke tests."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset


def build_loaders(
    batch_size: int = 1,
    num_workers: int = 0,
    num_classes: int = 10,
    image_shape: tuple[int, int, int] = (3, 32, 32),
    seed: int = 0,
) -> tuple[DataLoader, DataLoader]:
    generator = torch.Generator().manual_seed(seed)
    images = torch.randn((1, *image_shape), generator=generator)
    targets = torch.zeros(1, dtype=torch.long)
    if num_classes < 1:
        raise ValueError("num_classes must be at least 1")

    dataset = TensorDataset(images, targets)
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    return DataLoader(dataset, **loader_kwargs), DataLoader(dataset, **loader_kwargs)
