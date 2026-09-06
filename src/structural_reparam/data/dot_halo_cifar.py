"""CIFAR-10 with an OOD colour dot inside a noise halo, at a random location.

Two cues at two difficulties, designed so the 3x3 path is genuinely *slower* than
the 1x1/identity path on the easy cue (no depth required):

  * **Easy / pointwise cue** — a single OOD-coloured pixel (the "dot") whose
    colour direction encodes the class (p-reliable). It sits inside a small
    **noise halo**: the patch around it is high-variance noise, so a 3x3 conv
    centred on the dot reads dot + 8 noise taps (it must suppress 8 noisy taps
    and fights ~9x background variance), while a 1x1 / identity reads the dot
    cleanly. The dot's position *within* the halo and the halo's position in the
    image are both randomised per example, forcing a translation-invariant,
    value-based detector (no positional memorisation).

  * **Hard / spatial cue** — the underlying CIFAR-10 image, carrying the full
    label, readable only by the slow 3x3 + depth path.

Target is always the true CIFAR-10 class. Closing the gap above the dot-only
ceiling forces the image cue, so a model that over-trusts the dot is caught by a
shortcut trap. Placement / noise / dot-class are keyed deterministically by
example index so the ablate views ("none"/"dot"/"image") stay aligned.

ablate semantics:
  none   — CIFAR + noise halo + class-coloured dot   (full input)
  image  — CIFAR + noise halo, NO informative dot     (slow-cue competence)
  dot    — neutral background + noise halo + dot       (fast-cue competence)
"""

from __future__ import annotations

import colorsys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)

# Standard CIFAR-100 fine (0-99) -> coarse superclass (0-19) mapping.
CIFAR100_COARSE_MAP = [
    4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
    3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
    6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
    0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
    5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
    16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
    10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
    2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
    16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
    18, 1, 2, 15, 6, 0, 17, 8, 14, 13,
]


def _class_directions(num_classes: int, encoding: str = "hue") -> torch.Tensor:
    """``(num_classes, 3)`` distinct unit RGB directions encoding the class.

    ``hue``   — hue-spaced saturated colours (fine for ~10 classes; adjacent
                hues collide for many classes).
    ``spread``— fixed-seed random unit vectors over the full RGB sphere, far
                better separated when there are many classes (e.g. CIFAR-100,
                where 100 hues are nearly collinear).
    """
    if encoding == "hue":
        rgb = torch.tensor(
            [colorsys.hsv_to_rgb(i / num_classes, 1.0, 1.0) for i in range(num_classes)],
            dtype=torch.float32,
        )
        dirs = rgb - rgb.mean(dim=0, keepdim=True)
    elif encoding == "spread":
        # Farthest-point sampling on the unit sphere: greedily pick directions
        # that maximise the minimum pairwise separation. Plain random sampling
        # leaves near-duplicate directions (two classes with identical dots);
        # this gives near-optimal packing (~0.93 closest-pair cos at K=100).
        g = torch.Generator().manual_seed(1234)
        pool = torch.randn(20000, 3, generator=g)
        pool = pool / pool.norm(dim=1, keepdim=True).clamp_min(1e-6)
        chosen = [0]
        max_cos = (pool @ pool[0]).clamp(-1.0, 1.0)
        for _ in range(num_classes - 1):
            nxt = int(max_cos.argmin())
            chosen.append(nxt)
            max_cos = torch.maximum(max_cos, (pool @ pool[nxt]).clamp(-1.0, 1.0))
        dirs = pool[chosen]
    else:
        raise ValueError(f"encoding must be 'hue' or 'spread', got {encoding!r}")
    return dirs / dirs.norm(dim=1, keepdim=True).clamp_min(1e-6)


class DotHaloDataset(Dataset):
    def __init__(
        self,
        base: Dataset,
        num_classes: int,
        p: float,
        patch_size: int,
        dot_inset: int,
        noise_sigma: float,
        ood_scale: float,
        ablate: str,
        dot_seed: int,
        mean: tuple[float, float, float] = CIFAR_MEAN,
        std: tuple[float, float, float] = CIFAR_STD,
        dot_encoding: str = "hue",
        label_map: torch.Tensor | None = None,
    ) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}")
        if ablate not in ("none", "dot", "image"):
            raise ValueError(f"ablate must be 'none', 'dot', or 'image', got {ablate!r}")
        if patch_size - 2 * dot_inset < 1:
            raise ValueError(
                f"patch_size ({patch_size}) too small for dot_inset ({dot_inset}); "
                "need patch_size - 2*dot_inset >= 1 so the dot's neighbours stay inside the halo"
            )
        self.base = base
        self.num_classes = int(num_classes)
        self.p = float(p)
        self.patch_size = int(patch_size)
        self.dot_inset = int(dot_inset)
        self.noise_sigma = float(noise_sigma)
        self.ood_scale = float(ood_scale)
        self.ablate = ablate
        self.dot_seed = int(dot_seed)
        self.label_map = label_map

        self.normalize = transforms.Normalize(mean, std)
        # Dot colours live in *normalised* space at OOD magnitude, so they are
        # linearly separable from both CIFAR pixels and the noise halo (~unit std).
        self.dot_values = _class_directions(self.num_classes, dot_encoding) * self.ood_scale  # (K, 3)
        self.targets = getattr(base, "targets", None)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        x01, label = self.base[idx]  # (3, H, W) in [0, 1]
        if self.label_map is not None:
            label = int(self.label_map[int(label)])  # fine -> coarse superclass
        xb = self.normalize(x01.clone())
        _, H, W = xb.shape
        ps = self.patch_size

        g = torch.Generator().manual_seed(self.dot_seed * 1_000_003 + idx)
        ri = lambda lo, hi: int(torch.randint(lo, hi, (1,), generator=g).item())
        py, px = ri(0, H - ps + 1), ri(0, W - ps + 1)
        noise = torch.randn(3, ps, ps, generator=g) * self.noise_sigma
        lo, hi = self.dot_inset, ps - self.dot_inset
        dy, dx = ri(lo, hi), ri(lo, hi)
        reliable = torch.rand(1, generator=g).item() < self.p
        dot_class = int(label) if reliable else ri(0, self.num_classes)

        if self.ablate == "dot":
            xb = torch.zeros_like(xb)  # neutral background isolates the dot
        xb[:, py : py + ps, px : px + ps] = noise
        if self.ablate != "image":
            xb[:, py + dy, px + dx] = self.dot_values[dot_class]
        return xb, label


def build_loaders(
    data_dir: str | Path,
    batch_size: int = 128,
    num_workers: int = 4,
    download: bool = True,
    persistent_workers: bool = False,
    num_classes: int = 10,
    p: float = 0.85,
    patch_size: int = 5,
    dot_inset: int = 1,
    noise_sigma: float = 1.5,
    ood_scale: float = 10.0,
    ablate: str = "none",
    dot_seed: int = 0,
    augment: bool = True,
    batch_mode: str = "shuffled",
    cifar100: bool = False,
    dot_encoding: str = "hue",
    coarse: bool = False,
) -> tuple[DataLoader, DataLoader]:
    if batch_mode != "shuffled":
        raise ValueError(f"dot_halo_cifar only supports batch_mode='shuffled', got {batch_mode!r}")
    if coarse and not cifar100:
        raise ValueError("coarse=True (20 superclasses) requires cifar100=True")
    data_dir = Path(data_dir)

    ds_cls = datasets.CIFAR100 if cifar100 else datasets.CIFAR10
    mean, std = (CIFAR100_MEAN, CIFAR100_STD) if cifar100 else (CIFAR_MEAN, CIFAR_STD)
    label_map = torch.tensor(CIFAR100_COARSE_MAP, dtype=torch.long) if coarse else None

    train_pre = (
        [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
        if augment
        else []
    )
    train_tf = transforms.Compose([*train_pre, transforms.ToTensor()])
    test_tf = transforms.Compose([transforms.ToTensor()])

    train_base = ds_cls(data_dir, train=True, transform=train_tf, download=download)
    test_base = ds_cls(data_dir, train=False, transform=test_tf, download=download)

    kw = dict(
        num_classes=num_classes, p=p, patch_size=patch_size, dot_inset=dot_inset,
        noise_sigma=noise_sigma, ood_scale=ood_scale, ablate=ablate, dot_seed=dot_seed,
        mean=mean, std=std, dot_encoding=dot_encoding, label_map=label_map,
    )
    train_set = DotHaloDataset(train_base, **kw)
    test_set = DotHaloDataset(test_base, **kw)

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
