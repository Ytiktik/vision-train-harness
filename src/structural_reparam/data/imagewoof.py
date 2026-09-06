"""Imagewoof (10 fine-grained ImageNet dog-breed classes) dataset construction.

Auto-downloads the fast.ai imagewoof2-160 release on first use if the data
directory is missing. Same on-disk layout as Imagenette, so this loader wraps
``structural_reparam.data.imagenette.build_loaders`` with a different default
download URL.
"""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader

from . import imagenette


IMAGEWOOF_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2-160.tgz"


def build_loaders(
    data_dir: str | Path,
    image_size: int = 64,
    batch_size: int = 128,
    num_workers: int = 4,
    download: bool = True,
    download_url: str = IMAGEWOOF_URL,
) -> tuple[DataLoader, DataLoader]:
    return imagenette.build_loaders(
        data_dir=data_dir,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        download=download,
        download_url=download_url,
    )
