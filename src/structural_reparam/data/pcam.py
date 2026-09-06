"""PatchCamelyon (PCam) dataset construction.

Binary histopathology classification: each 96x96 RGB patch is labelled by
whether its center 32x32 region contains tumor tissue (1) or not (0). Roughly
262k train / 32k val / 32k test patches. A "Goldilocks" benchmark — clearly
harder than a CIFAR-10/100 superclass split, but not ImageNet-hard; the ceiling
sits below 100% (real label ambiguity).

Wraps ``torchvision.datasets.PCAM``, which stores the gzipped HDF5 files under
``<data_dir>/pcam/`` and requires ``h5py``. First use with ``download=True``
fetches the files from the original Google Drive mirror (can be rate-limited).

Because the label is determined by the center 32x32 region, augmentation is
restricted to flips/90-degree rotations — never random crops, which could move
the decisive region out of frame.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# Channel statistics for PCam (RGB), as commonly used in the histopathology
# literature; computed over the train split.
PCAM_MEAN = (0.7008, 0.5384, 0.6916)
PCAM_STD = (0.2350, 0.2774, 0.2128)

NATIVE_SIZE = 96


def _build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    resize: list = []
    if image_size != NATIVE_SIZE:
        # Center-preserving resize (no crop) so the decisive center region stays in frame.
        resize = [transforms.Resize(image_size)]

    train_transform = transforms.Compose(
        [
            *resize,
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(PCAM_MEAN, PCAM_STD),
        ]
    )
    test_transform = transforms.Compose(
        [
            *resize,
            transforms.ToTensor(),
            transforms.Normalize(PCAM_MEAN, PCAM_STD),
        ]
    )
    return train_transform, test_transform


def build_loaders(
    data_dir: str | Path,
    image_size: int = NATIVE_SIZE,
    batch_size: int = 128,
    num_workers: int = 4,
    download: bool = True,
    persistent_workers: bool = False,
    batch_mode: str = "shuffled",
    test_split: str = "test",
) -> tuple[DataLoader, DataLoader]:
    if batch_mode != "shuffled":
        # PCam is binary by construction; class-concentrated sampling has no
        # meaning here (and the mechanistic probe always requests "shuffled").
        raise ValueError(
            f"pcam.build_loaders only supports batch_mode='shuffled', got {batch_mode!r}."
        )

    data_dir = Path(data_dir)
    train_transform, test_transform = _build_transforms(image_size)

    train_set = datasets.PCAM(data_dir, split="train", transform=train_transform, download=download)
    test_set = datasets.PCAM(data_dir, split=test_split, transform=test_transform, download=download)

    use_persistent = persistent_workers and num_workers > 0
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
