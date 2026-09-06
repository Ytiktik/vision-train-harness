"""Imagenette (10 easily-separable ImageNet classes) dataset construction.

Auto-downloads the fast.ai imagenette2-160 release on first use if the data
directory is missing.

Expected layout under ``data_dir`` (default ``data/imagenette2-160``):
    train/<wnid>/*.JPEG
    val/<wnid>/*.JPEG
"""

from __future__ import annotations

import tarfile
import urllib.request
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"


def _ensure_dataset(data_dir: Path, url: str) -> None:
    """Download + extract the fast.ai imagenette/imagewoof tarball if missing.

    The tarball expands to a top-level directory whose name matches its stem
    (e.g. ``imagenette2-160/``). We extract into ``data_dir.parent`` so the
    resulting tree lands at ``data_dir`` itself.
    """
    if (data_dir / "train").is_dir() and (data_dir / "val").is_dir():
        return

    parent = data_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    archive = parent / Path(url).name

    if not archive.exists():
        print(f"[imagenette] downloading {url} -> {archive}")
        urllib.request.urlretrieve(url, archive)

    print(f"[imagenette] extracting {archive} -> {parent}")
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(parent)

    if not (data_dir / "train").is_dir() or not (data_dir / "val").is_dir():
        raise RuntimeError(
            f"Extraction of {archive} did not produce expected layout at {data_dir}"
        )


def build_loaders(
    data_dir: str | Path,
    image_size: int = 64,
    batch_size: int = 128,
    num_workers: int = 4,
    download: bool = True,
    download_url: str = IMAGENETTE_URL,
) -> tuple[DataLoader, DataLoader]:
    data_dir = Path(data_dir)
    if download:
        _ensure_dataset(data_dir, download_url)

    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    if not train_dir.is_dir() or not val_dir.is_dir():
        raise FileNotFoundError(
            f"Imagenette splits not found under {data_dir} (download=False)."
        )

    train_transform = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.15)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.15)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    train_set = datasets.ImageFolder(train_dir, transform=train_transform)
    test_set = datasets.ImageFolder(val_dir, transform=test_transform)

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
