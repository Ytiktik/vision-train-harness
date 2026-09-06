"""Full ImageNet-1k (ILSVRC2012) dataset construction.

ImageNet is far too large to auto-download here; it is baked onto the Thunder
snapshot during bootstrap (see ``scripts/prefetch_imagenet.py``, which pulls the
Kaggle ``imagenet-object-localization-challenge`` release and arranges it into
the ImageFolder layout below). ``build_loaders`` therefore expects the data to
already exist on disk (``download=False``).

Expected layout under ``data_dir`` (default ``data/imagenet``):
    train/<wnid>/*.JPEG
    val/<wnid>/*.JPEG
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


LOGGER = logging.getLogger(__name__)

# The ImageNet snapshot is served over a nydus lazy-loading overlay: files are
# fetched from network storage on first access. Under many concurrent DataLoader
# workers a read of a not-yet-materialized file can transiently fail (ENOENT /
# truncated read) even though the file exists. A single such failure would crash
# a multi-day run, so reads retry with a short backoff before giving up.
_READ_RETRIES = 6
_READ_BACKOFF_SEC = 0.25


def _robust_loader(path: str):
    from torchvision.datasets.folder import default_loader

    last_exc: Exception | None = None
    for attempt in range(_READ_RETRIES):
        try:
            return default_loader(path)
        except (FileNotFoundError, OSError) as exc:
            last_exc = exc
            time.sleep(_READ_BACKOFF_SEC * (attempt + 1))
    LOGGER.error("Failed to read %s after %d retries: %s", path, _READ_RETRIES, last_exc)
    raise last_exc


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_loaders(
    data_dir: str | Path,
    image_size: int = 224,
    batch_size: int = 256,
    num_workers: int = 8,
    download: bool = False,
    persistent_workers: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """Standard ImageNet-1k ImageFolder train/val loaders.

    ``download`` is accepted for parity with the other dataset builders but must
    be False — ImageNet is baked onto the snapshot, never fetched at train time.
    """
    data_dir = Path(data_dir)
    if download:
        raise ValueError(
            "ImageNet cannot be auto-downloaded at train time; bake it onto the "
            "snapshot with scripts/prefetch_imagenet.py (download=False here)."
        )

    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    if not train_dir.is_dir() or not val_dir.is_dir():
        raise FileNotFoundError(
            f"ImageNet splits not found under {data_dir} (expected train/ and val/). "
            "Build the ImageNet snapshot first: "
            "python scripts/bootstrap_thunder_snapshot.py --imagenet"
        )

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize(int(image_size * 256 / 224)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    train_set = datasets.ImageFolder(train_dir, transform=train_transform, loader=_robust_loader)
    test_set = datasets.ImageFolder(val_dir, transform=test_transform, loader=_robust_loader)

    use_workers = num_workers > 0
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=4 if use_workers else None,
        persistent_workers=persistent_workers and use_workers,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=4 if use_workers else None,
        persistent_workers=persistent_workers and use_workers,
    )
    return train_loader, test_loader
