"""CIFAR-10 with an injected pointwise colour-dot cue at controllable reliability.

Two cues at two difficulties, for studying RepVGG two-timescale / shortcut dynamics:

  * **Easy / pointwise cue** — a small uniform colour patch stamped at a fixed
    location. Each class owns a distinct saturated hue, so the patch is linearly
    separable in input space and readable by a position-free (1x1 / identity)
    operator. With probability ``p`` the patch encodes the *true* class; with
    probability ``1 - p`` it encodes a uniformly random class (actively
    misleading). A dot-only predictor therefore tops out near ``p + (1-p)/K``.

  * **Hard / spatial cue** — the underlying CIFAR-10 image, carrying the full
    label, which only the slow 3x3 + depth path can decode.

The training target is always the *true* CIFAR-10 class; the dot is just a
``p``-reliable input feature. Closing the gap above ``p + (1-p)/K`` forces the
model to use the image, so a model that over-trusts the dot is caught by a
*shortcut trap*.

The per-sample dot assignment is keyed deterministically by example index, so
the ``ablate`` views ("none" / "dot" / "image") of the same split stay aligned
and can be compared example-for-example (used by the cue-attribution probe).
"""

from __future__ import annotations

import colorsys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def _make_palette(num_classes: int) -> torch.Tensor:
    """``(num_classes, 3)`` of maximally-distinct, fully-saturated RGB colours in [0, 1]."""
    colors = [colorsys.hsv_to_rgb(i / num_classes, 1.0, 1.0) for i in range(num_classes)]
    return torch.tensor(colors, dtype=torch.float32)


class DotCueDataset(Dataset):
    """Wraps a [0, 1] image dataset, stamps the colour-dot cue, then normalises."""

    def __init__(
        self,
        base: Dataset,
        num_classes: int,
        p: float,
        dot_size: int,
        dot_loc: tuple[int, int],
        dot_contrast: float,
        ablate: str,
        dot_seed: int,
    ) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}")
        if ablate not in ("none", "dot", "image"):
            raise ValueError(f"ablate must be 'none', 'dot', or 'image', got {ablate!r}")
        if not 0.0 <= dot_contrast <= 1.0:
            raise ValueError(f"dot_contrast must be in [0, 1], got {dot_contrast}")
        self.base = base
        self.num_classes = int(num_classes)
        self.p = float(p)
        self.dot_size = int(dot_size)
        self.dot_loc = (int(dot_loc[0]), int(dot_loc[1]))
        self.dot_contrast = float(dot_contrast)
        self.ablate = ablate
        self.dot_seed = int(dot_seed)

        palette = _make_palette(self.num_classes)  # (K, 3) in [0, 1]
        neutral = torch.tensor(CIFAR_MEAN, dtype=torch.float32)  # fill that normalises to ~0
        # Blend toward neutral by (1 - contrast) so the cue strength is tunable.
        self.colors = neutral + self.dot_contrast * (palette - neutral)  # (K, 3)
        self.neutral = neutral
        self.normalize = transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
        # CIFAR-10 targets are plain python ints via torchvision; expose for samplers.
        self.targets = getattr(base, "targets", None)

    def __len__(self) -> int:
        return len(self.base)

    def _dot_class(self, idx: int, true_label: int) -> int:
        g = torch.Generator().manual_seed(self.dot_seed * 1_000_003 + idx)
        reliable = torch.rand(1, generator=g).item() < self.p
        if reliable:
            return true_label
        return int(torch.randint(0, self.num_classes, (1,), generator=g).item())

    def __getitem__(self, idx: int):
        x, label = self.base[idx]  # x: float tensor (3, H, W) in [0, 1]
        x = x.clone()
        y0, x0 = self.dot_loc
        s = self.dot_size

        if self.ablate == "dot":
            # Isolate the dot: wipe the image to neutral, keep the patch.
            x[:] = self.neutral.view(3, 1, 1)
        if self.ablate == "image":
            # Isolate the image: overwrite the patch region with neutral (no dot).
            x[:, y0 : y0 + s, x0 : x0 + s] = self.neutral.view(3, 1, 1)
        else:
            dot_class = self._dot_class(idx, int(label))
            x[:, y0 : y0 + s, x0 : x0 + s] = self.colors[dot_class].view(3, 1, 1)

        return self.normalize(x), label


def build_loaders(
    data_dir: str | Path,
    batch_size: int = 128,
    num_workers: int = 4,
    download: bool = True,
    persistent_workers: bool = False,
    num_classes: int = 10,
    p: float = 0.85,
    dot_size: int = 3,
    dot_loc: tuple[int, int] = (0, 0),
    dot_contrast: float = 1.0,
    ablate: str = "none",
    dot_seed: int = 0,
    augment: bool = True,
    batch_mode: str = "shuffled",
) -> tuple[DataLoader, DataLoader]:
    # ``batch_mode`` is accepted for parity with the other datasets / the
    # mechanistic probe's loader rebuild; only the default "shuffled" is supported.
    if batch_mode != "shuffled":
        raise ValueError(f"dot_cue_cifar only supports batch_mode='shuffled', got {batch_mode!r}")
    data_dir = Path(data_dir)

    train_pre = (
        [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
        if augment
        else []
    )
    # ToTensor only (-> [0, 1]); normalisation happens after the dot is stamped.
    train_tf = transforms.Compose([*train_pre, transforms.ToTensor()])
    test_tf = transforms.Compose([transforms.ToTensor()])

    train_base = datasets.CIFAR10(data_dir, train=True, transform=train_tf, download=download)
    test_base = datasets.CIFAR10(data_dir, train=False, transform=test_tf, download=download)

    dot_kwargs = dict(
        num_classes=num_classes,
        p=p,
        dot_size=dot_size,
        dot_loc=dot_loc,
        dot_contrast=dot_contrast,
        ablate=ablate,
        dot_seed=dot_seed,
    )
    train_set = DotCueDataset(train_base, **dot_kwargs)
    test_set = DotCueDataset(test_base, **dot_kwargs)

    use_persistent = persistent_workers and num_workers > 0
    loader_kwargs = dict(
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=use_persistent,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, **loader_kwargs)
    return train_loader, test_loader
