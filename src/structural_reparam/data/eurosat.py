"""EuroSAT dataset construction (Sentinel-2 RGB satellite tiles).

10 classes, 27 000 images at 64x64 native, no canonical split. We carve a
deterministic 80/20 split (21 600 train / 5 400 test) with a fixed torch
generator, resize to 32x32 for arch parity with the CIFAR cells, and use the
CIFAR augmentation (crop + flip; tile orientation is arbitrary, so flips are
legitimate).
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

EUROSAT_MEAN = (0.3438, 0.3802, 0.4080)
EUROSAT_STD = (0.1967, 0.1309, 0.1095)
SPLIT_SEED = 0
N_TRAIN = 21600


class _Transformed(Dataset):
    def __init__(self, base: Dataset, transform) -> None:
        self.base, self.transform = base, transform

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i):
        x, y = self.base[i]
        return self.transform(x), y


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
            f"eurosat.build_loaders supports batch_mode='shuffled' only, got {batch_mode!r}."
        )
    base = datasets.EuroSAT(str(Path(data_dir)), download=download)
    tr_base, te_base = torch.utils.data.random_split(
        base, [N_TRAIN, len(base) - N_TRAIN],
        generator=torch.Generator().manual_seed(SPLIT_SEED))
    resize = [transforms.Resize((32, 32))]
    aug_ops = ([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
               if augment else [])
    norm = [transforms.ToTensor(), transforms.Normalize(EUROSAT_MEAN, EUROSAT_STD)]
    train_ds = _Transformed(tr_base, transforms.Compose(resize + aug_ops + norm))
    test_ds = _Transformed(te_base, transforms.Compose(resize + norm))
    train = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                       num_workers=num_workers, pin_memory=True,
                       persistent_workers=persistent_workers and num_workers > 0)
    test = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True,
                      persistent_workers=persistent_workers and num_workers > 0)
    return train, test
