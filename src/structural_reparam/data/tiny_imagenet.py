"""Tiny-ImageNet (200 classes, 64x64) dataset construction.

Tiny-ImageNet is not available via torchvision. Download and extract once:

    cd data && \
        wget http://cs231n.stanford.edu/tiny-imagenet-200.zip && \
        unzip tiny-imagenet-200.zip

Expected layout under ``data_dir`` (default ``data/tiny-imagenet-200``):
    train/<wnid>/images/*.JPEG
    val/images/*.JPEG + val_annotations.txt
    wnids.txt, words.txt

The val split ships as a flat folder; this loader restructures it into
``val/<wnid>/*.JPEG`` on first use so ``ImageFolder`` can read it.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


TINY_IMAGENET_MEAN = (0.4802, 0.4481, 0.3975)
TINY_IMAGENET_STD = (0.2770, 0.2691, 0.2821)


def _ensure_val_imagefolder(val_dir: Path) -> None:
    """Rearrange Tiny-ImageNet val/ into ImageFolder layout if not already done."""
    annotations = val_dir / "val_annotations.txt"
    flat_images = val_dir / "images"
    if not annotations.exists() or not flat_images.exists():
        return

    with annotations.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            filename, wnid = parts[0], parts[1]
            src = flat_images / filename
            if not src.exists():
                continue
            class_dir = val_dir / wnid
            class_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), class_dir / filename)

    if flat_images.exists() and not any(flat_images.iterdir()):
        flat_images.rmdir()
    annotations.unlink(missing_ok=True)


def _subset_by_classes(
    dataset: datasets.ImageFolder, keep_class_indices: set[int]
) -> Subset:
    """Restrict an ImageFolder to samples whose class index is in ``keep_class_indices``.

    Targets are remapped so the kept classes use contiguous indices starting at 0
    (preserving the original sorted order), making the subset usable with a model
    head sized to ``len(keep_class_indices)``.
    """
    sorted_kept = sorted(keep_class_indices)
    remap = {old: new for new, old in enumerate(sorted_kept)}
    indices = [i for i, (_, t) in enumerate(dataset.samples) if t in keep_class_indices]
    dataset.samples = [(p, remap[t]) for p, t in dataset.samples if t in keep_class_indices]
    dataset.targets = [t for _, t in dataset.samples]
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    dataset.classes = [idx_to_class[old] for old in sorted_kept]
    dataset.class_to_idx = {c: i for i, c in enumerate(dataset.classes)}
    return Subset(dataset, list(range(len(dataset.samples))))


def build_loaders(
    data_dir: str | Path,
    batch_size: int = 128,
    num_workers: int = 4,
    num_classes: int | None = None,
    keep_classes: list[int] | None = None,
    persistent_workers: bool | None = None,
    batch_mode: str = "shuffled",
) -> tuple[DataLoader, DataLoader]:
    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    if not train_dir.is_dir():
        raise FileNotFoundError(
            f"Tiny-ImageNet train split not found at {train_dir}. "
            "Download tiny-imagenet-200.zip from cs231n.stanford.edu and extract it."
        )
    _ensure_val_imagefolder(val_dir)

    if persistent_workers is None:
        persistent_workers = num_workers > 0

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(64, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(TINY_IMAGENET_MEAN, TINY_IMAGENET_STD),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(TINY_IMAGENET_MEAN, TINY_IMAGENET_STD),
        ]
    )

    train_set = datasets.ImageFolder(train_dir, transform=train_transform)
    test_set = datasets.ImageFolder(val_dir, transform=test_transform)

    if num_classes is not None:
        if keep_classes is not None:
            raise ValueError("Pass either num_classes or keep_classes, not both.")
        if num_classes <= 0 or num_classes > len(train_set.classes):
            raise ValueError(
                f"num_classes={num_classes} must be in [1, {len(train_set.classes)}]"
            )
        # Take the first ``num_classes`` wnids in sorted order for determinism.
        keep_names = set(train_set.classes[:num_classes])
        train_keep = {train_set.class_to_idx[n] for n in keep_names}
        test_keep = {test_set.class_to_idx[n] for n in keep_names}
        train_set = _subset_by_classes(train_set, train_keep)
        test_set = _subset_by_classes(test_set, test_keep)
    elif keep_classes is not None:
        train_keep = set(keep_classes)
        test_keep = set(keep_classes)
        train_set = _subset_by_classes(train_set, train_keep)
        test_set = _subset_by_classes(test_set, test_keep)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=persistent_workers,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=persistent_workers,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    return train_loader, test_loader


def build_train_eval_loader(
    data_dir: str | Path,
    batch_size: int = 256,
    num_workers: int = 4,
    num_classes: int | None = None,
    keep_classes: list[int] | None = None,
) -> DataLoader:
    """No-augmentation, no-shuffle loader over the Tiny-ImageNet *train* split."""
    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(TINY_IMAGENET_MEAN, TINY_IMAGENET_STD),
        ]
    )
    train_set = datasets.ImageFolder(train_dir, transform=transform)
    if num_classes is not None:
        if keep_classes is not None:
            raise ValueError("Pass either num_classes or keep_classes, not both.")
        if num_classes <= 0 or num_classes > len(train_set.classes):
            raise ValueError(
                f"num_classes={num_classes} must be in [1, {len(train_set.classes)}]"
            )
        keep_names = set(train_set.classes[:num_classes])
        train_set = _subset_by_classes(train_set, {train_set.class_to_idx[n] for n in keep_names})
    elif keep_classes is not None:
        train_set = _subset_by_classes(train_set, set(keep_classes))
    return DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )
