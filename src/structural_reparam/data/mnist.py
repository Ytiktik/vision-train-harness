"""MNIST dataset construction."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from structural_reparam.data.cifar10 import ClassConcentratedBatchSampler, _filter_to_classes


class _SubtractPerPixelMean:
    """Subtract a precomputed per-pixel mean from a (C, H, W) tensor."""

    def __init__(self, mean: torch.Tensor) -> None:
        self.mean = mean  # (1, H, W) broadcastable against (C, H, W)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x - self.mean


class _DividePerPixelStd:
    """Divide by per-pixel std only (no mean subtraction) from a (C, H, W) tensor."""

    def __init__(self, std: torch.Tensor) -> None:
        self.std = std  # (1, H, W)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x / self.std


class _StandardizePerPixel:
    """Subtract per-pixel mean AND divide by per-pixel std from a (C, H, W) tensor."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.mean = mean  # (1, H, W)
        self.std = std    # (1, H, W)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


def build_loaders(
    data_dir: str | Path,
    batch_size: int = 128,
    num_workers: int = 4,
    download: bool = True,
    persistent_workers: bool = False,
    batch_mode: str = "shuffled",
    sampler_seed: int = 0,
    classes_per_batch: int | None = None,
    mean_normalize: str = "scalar",
    keep_classes: list[int] | None = None,
) -> tuple[DataLoader, DataLoader]:
    data_dir = Path(data_dir)

    if mean_normalize == "scalar":
        normalize = transforms.Normalize((0.1307,), (0.3081,))
        train_transform = transforms.Compose([transforms.ToTensor(), normalize])
        test_transform = transforms.Compose([transforms.ToTensor(), normalize])
    elif mean_normalize == "per_pixel":
        # Per-pixel mean over the training set, then subtract — no std division.
        # Leaves first-layer pre-BN activations zero-mean in expectation, which
        # pairs with norm='batch_zeromean' (BN that skips mean subtraction).
        raw = datasets.MNIST(data_dir, train=True, download=download).data
        per_pixel_mean = (raw.float() / 255.0).mean(dim=0).unsqueeze(0)  # (1, 28, 28)
        subtract = _SubtractPerPixelMean(per_pixel_mean)
        train_transform = transforms.Compose([transforms.ToTensor(), subtract])
        test_transform = transforms.Compose([transforms.ToTensor(), subtract])
    elif mean_normalize == "per_pixel_var":
        # Per-pixel std only — no mean subtraction. Pairs with a norm that
        # handles mean (e.g. mean_weight_norm or batch_zeromean).
        raw = datasets.MNIST(data_dir, train=True, download=download).data
        pixels = raw.float() / 255.0
        per_pixel_std = pixels.std(dim=0).unsqueeze(0).clamp(min=1e-5)  # (1, H, W)
        train_transform = transforms.Compose([transforms.ToTensor(), _DividePerPixelStd(per_pixel_std)])
        test_transform = transforms.Compose([transforms.ToTensor(), _DividePerPixelStd(per_pixel_std)])
    elif mean_normalize == "per_pixel_mean_var":
        # Per-pixel mean AND std — full standardisation from the dataset.
        # Pairs with norm='weight_norm' to give γ·ŵ·(x−μ)/σ_x.
        raw = datasets.MNIST(data_dir, train=True, download=download).data
        pixels = raw.float() / 255.0
        per_pixel_mean = pixels.mean(dim=0).unsqueeze(0)
        per_pixel_std = pixels.std(dim=0).unsqueeze(0).clamp(min=1e-5)
        standardize = _StandardizePerPixel(per_pixel_mean, per_pixel_std)
        train_transform = transforms.Compose([transforms.ToTensor(), standardize])
        test_transform = transforms.Compose([transforms.ToTensor(), standardize])
    else:
        raise ValueError(
            f"Unknown mean_normalize={mean_normalize!r}. "
            "Expected 'scalar', 'per_pixel', 'per_pixel_var', or 'per_pixel_mean_var'."
        )

    train_set = datasets.MNIST(data_dir, train=True, transform=train_transform, download=download)
    test_set = datasets.MNIST(data_dir, train=False, transform=test_transform, download=download)

    if keep_classes is not None:
        train_set = _filter_to_classes(train_set, keep_classes)
        test_set = _filter_to_classes(test_set, keep_classes)

    use_persistent = persistent_workers and num_workers > 0

    if batch_mode == "shuffled":
        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=use_persistent,
            prefetch_factor=4 if num_workers > 0 else None,
        )
    elif batch_mode == "class_concentrated":
        if classes_per_batch is None:
            raise ValueError("batch_mode='class_concentrated' requires classes_per_batch.")
        k = batch_size // int(classes_per_batch)
        if k < 1:
            raise ValueError(
                f"batch_size={batch_size} too small for classes_per_batch={classes_per_batch} (k={k})."
            )
        targets = train_set.targets.tolist()
        train_loader = DataLoader(
            train_set,
            batch_sampler=ClassConcentratedBatchSampler(
                targets,
                k_per_class=k,
                classes_per_batch=int(classes_per_batch),
                seed=sampler_seed,
            ),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=use_persistent,
            prefetch_factor=4 if num_workers > 0 else None,
        )
    else:
        raise ValueError(f"Unknown batch_mode: {batch_mode!r}. Expected 'shuffled' or 'class_concentrated'.")

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
