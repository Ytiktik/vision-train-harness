"""STL-10 dataset construction.

10 classes, 96x96 native, only 5000 labeled training images. We use only the
labeled splits (``train`` and ``test``) and ignore the ~100k unlabeled split,
since the probe is about supervised trajectory dynamics under low-data.

Inputs are resized to ``image_size`` (default 64) for arch parity with the
existing CifarRepVGG runs.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


STL10_MEAN = (0.4467, 0.4398, 0.4066)
STL10_STD = (0.2603, 0.2566, 0.2713)


def build_loaders(
    data_dir: str | Path,
    image_size: int = 64,
    batch_size: int = 128,
    num_workers: int = 4,
    download: bool = True,
    augment: bool = True,
    persistent_workers: bool = False,
    batch_mode: str = "shuffled",
) -> tuple[DataLoader, DataLoader]:
    """``batch_mode`` accepts only ``'shuffled'`` here: the class-concentrated
    sampler of the CIFAR loaders is not implemented for STL-10, and silently
    ignoring the argument would make a config lie about what it ran."""
    if batch_mode != "shuffled":
        raise ValueError(
            f"stl10.build_loaders supports batch_mode='shuffled' only, got {batch_mode!r}."
        )
    data_dir = Path(data_dir)

    aug_ops = (
        [
            transforms.RandomCrop(image_size, padding=image_size // 8),
            transforms.RandomHorizontalFlip(),
        ]
        if augment
        else []
    )
    train_transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            *aug_ops,
            transforms.ToTensor(),
            transforms.Normalize(STL10_MEAN, STL10_STD),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(STL10_MEAN, STL10_STD),
        ]
    )

    train_set = datasets.STL10(
        data_dir, split="train", transform=train_transform, download=download
    )
    test_set = datasets.STL10(
        data_dir, split="test", transform=test_transform, download=download
    )

    use_persistent = bool(persistent_workers) and num_workers > 0
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=use_persistent,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=use_persistent,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    return train_loader, test_loader
