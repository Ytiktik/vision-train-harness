"""SVHN (Street View House Numbers) dataset construction.

10 digit classes, 32x32 native, ~73k train / ~26k test. No resizing — the
native size matches CIFAR. We use the standard ``train`` split (not ``extra``)
to keep the dataset-size axis comparable to CIFAR10/100.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


SVHN_MEAN = (0.4377, 0.4438, 0.4728)
SVHN_STD = (0.1980, 0.2010, 0.1970)


def build_loaders(
    data_dir: str | Path,
    batch_size: int = 128,
    num_workers: int = 4,
    download: bool = True,
) -> tuple[DataLoader, DataLoader]:
    data_dir = Path(data_dir)

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(SVHN_MEAN, SVHN_STD),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(SVHN_MEAN, SVHN_STD),
        ]
    )

    train_set = datasets.SVHN(
        data_dir, split="train", transform=train_transform, download=download
    )
    test_set = datasets.SVHN(
        data_dir, split="test", transform=test_transform, download=download
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=4 if num_workers > 0 else None,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=4 if num_workers > 0 else None,
    )
    return train_loader, test_loader
