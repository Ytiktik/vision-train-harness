"""Flowers-102 dataset construction, with the roles of the official splits swapped.

102 classes. The official train split has only 1 020 images (10 per class),
far too small to train on from scratch, so we TRAIN on the official test split
(6 149 images) and EVALUATE on the official train+val splits (2 040 images).
The swap is stated here and in every config that uses this loader. Images are
resized to 32x32 for arch parity with the CIFAR cells; augmentation is
crop + flip. Label loading needs scipy (torchvision reads the .mat files).
"""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import ConcatDataset, DataLoader
from torchvision import datasets, transforms

FLOWERS_MEAN = (0.4359, 0.3760, 0.2855)
FLOWERS_STD = (0.2768, 0.2231, 0.2480)


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
            f"flowers102.build_loaders supports batch_mode='shuffled' only, got {batch_mode!r}."
        )
    root = str(Path(data_dir))
    resize = [transforms.Resize((32, 32))]
    aug_ops = ([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
               if augment else [])
    norm = [transforms.ToTensor(), transforms.Normalize(FLOWERS_MEAN, FLOWERS_STD)]
    train_tf = transforms.Compose(resize + aug_ops + norm)
    test_tf = transforms.Compose(resize + norm)
    train_ds = datasets.Flowers102(root, split="test", download=download, transform=train_tf)
    test_ds = ConcatDataset([
        datasets.Flowers102(root, split="train", download=download, transform=test_tf),
        datasets.Flowers102(root, split="val", download=download, transform=test_tf),
    ])
    train = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                       num_workers=num_workers, pin_memory=True,
                       persistent_workers=persistent_workers and num_workers > 0)
    test = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True,
                      persistent_workers=persistent_workers and num_workers > 0)
    return train, test
