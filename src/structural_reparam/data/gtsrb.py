"""GTSRB dataset construction (German traffic signs).

43 classes, variable native size, 26 640 train images. Images are resized to
32x32 for arch parity with the CIFAR cells; augmentation is the CIFAR recipe
minus the horizontal flip, which is wrong for signs.
"""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

GTSRB_MEAN = (0.3403, 0.3121, 0.3214)
GTSRB_STD = (0.2724, 0.2608, 0.2669)


def build_loaders(
    data_dir: str | Path,
    batch_size: int = 128,
    num_workers: int = 4,
    download: bool = True,
    augment: bool = True,
    persistent_workers: bool = False,
    batch_mode: str = "shuffled",
) -> tuple[DataLoader, DataLoader]:
    if batch_mode != "shuffled":
        raise ValueError(
            f"gtsrb.build_loaders supports batch_mode='shuffled' only, got {batch_mode!r}."
        )
    data_dir = Path(data_dir)
    resize = [transforms.Resize((32, 32))]
    aug_ops = [transforms.RandomCrop(32, padding=4)] if augment else []
    norm = [transforms.ToTensor(), transforms.Normalize(GTSRB_MEAN, GTSRB_STD)]
    train_ds = datasets.GTSRB(str(data_dir), split="train", download=download,
                              transform=transforms.Compose(resize + aug_ops + norm))
    test_ds = datasets.GTSRB(str(data_dir), split="test", download=download,
                             transform=transforms.Compose(resize + norm))
    train = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                       num_workers=num_workers, pin_memory=True,
                       persistent_workers=persistent_workers and num_workers > 0)
    test = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True,
                      persistent_workers=persistent_workers and num_workers > 0)
    return train, test
