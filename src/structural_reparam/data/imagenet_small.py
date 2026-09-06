"""Downsampled ImageNet (ImageNet32 / ImageNet64) from baked pixel arrays.

The data is a single unsigned 8-bit array per split per resolution, built once by
``scripts/prefetch_imagenet_small.py`` and baked into the Thunder snapshot. Nothing
here decodes a JPEG: at 32 pixels the whole training set is 3.67GiB of raw pixels,
which fits on the GPU with room to spare, so the fastest correct pipeline is to put
the array on the device once and do the augmentation there in batch.

That matters more than it looks. The models this thesis trains are small — the
default cell is a depth-5 stack of 32, 64 and 128 channels — so a step at batch 128
costs a couple of milliseconds, while ImageNet32 has 10,009 steps per epoch against
CIFAR-100's 390. An ImageFolder pipeline decoding 1.28 million JPEGs per epoch would
leave the GPU idle essentially the whole time, and Thunder's prototyping GPU is
remote, with host-to-device bandwidth around 1GB/s, so even streaming decoded
batches costs more per epoch than the one-off transfer done here.

The augmentation is the paper's CIFAR recipe — random crop with padding, then a
horizontal flip — expressed as batched tensor indexing. Padding is ``resolution //
8``, which is the familiar 4 pixels at 32 and 8 at 64.

Layout under ``data_dir`` (default ``data/imagenet_small``):
    train_x32.npy / train_x64.npy   (N, 3, R, R) uint8
    val_x32.npy   / val_x64.npy     (N, 3, R, R) uint8
    train_y.npy   / val_y.npy       (N,) int16, labels 0..999
    wnids.json, meta.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from structural_reparam.data.cifar10 import _identity_collate


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_SUBDIR = "imagenet_small"


# --------------------------------------------------------------------------------------
# The dataset
# --------------------------------------------------------------------------------------

class ArrayImageDataset(torch.utils.data.Dataset):
    """A resident uint8 pixel array with batched augmentation on its own device.

    ``__getitems__`` is the batched entry point: the DataLoader hands it the whole
    index list for a batch and passes the result straight to ``_identity_collate``,
    so one advanced index plus one augmentation call replaces N ``__getitem__``
    calls and a default collate. This mirrors ``_CachedTensorDataset`` in
    ``cifar10.py``; the difference is that the randomness lives here, on the batch,
    rather than being frozen into the cache — which is why this dataset may be
    augmented and that one may not.
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        device: str | None,
        augment: bool,
        mean: tuple[float, ...] = IMAGENET_MEAN,
        std: tuple[float, ...] = IMAGENET_STD,
        pad: int | None = None,
    ) -> None:
        self.x = torch.from_numpy(np.ascontiguousarray(x))
        self.y = torch.from_numpy(np.ascontiguousarray(y).astype(np.int64))
        if device is not None:
            self.x = self.x.to(device)
            self.y = self.y.to(device)
        self.device = self.x.device
        self.augment = bool(augment)
        self.resolution = int(self.x.shape[-1])
        self.pad = self.resolution // 8 if pad is None else int(pad)
        self._mean = torch.tensor(mean, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(std, device=self.device).view(1, 3, 1, 1)
        self.targets: list[int] = self.y.tolist()

    def __len__(self) -> int:
        return int(self.x.shape[0])

    # -- the augmentation ---------------------------------------------------------

    def _random_crop(self, batch: torch.Tensor) -> torch.Tensor:
        """Pad by ``self.pad`` on every side and take a random resolution-sized crop
        per sample, as one gather rather than a Python loop."""
        p, r = self.pad, self.resolution
        if p == 0:
            return batch
        padded = F.pad(batch, (p, p, p, p))
        n = batch.shape[0]
        dev = batch.device
        off_y = torch.randint(0, 2 * p + 1, (n,), device=dev)
        off_x = torch.randint(0, 2 * p + 1, (n,), device=dev)
        span = torch.arange(r, device=dev)
        rows = (off_y[:, None] + span[None, :])[:, None, :, None]   # (n,1,r,1)
        cols = (off_x[:, None] + span[None, :])[:, None, None, :]   # (n,1,1,r)
        bidx = torch.arange(n, device=dev)[:, None, None, None]
        cidx = torch.arange(3, device=dev)[None, :, None, None]
        return padded[bidx, cidx, rows, cols]

    def _random_flip(self, batch: torch.Tensor) -> torch.Tensor:
        flip = torch.rand(batch.shape[0], device=batch.device) < 0.5
        return torch.where(flip[:, None, None, None], batch.flip(-1), batch)

    def transform(self, batch_u8: torch.Tensor) -> torch.Tensor:
        """uint8 (n,3,r,r) -> normalized float32, augmented if this split is."""
        out = batch_u8
        if self.augment:
            out = self._random_crop(out)
            out = self._random_flip(out)
        out = out.to(torch.float32).div_(255.0)
        return out.sub_(self._mean).div_(self._std)

    # -- the DataLoader entry points ----------------------------------------------

    def __getitems__(self, indices: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        idx = torch.as_tensor(indices, device=self.device)
        return self.transform(self.x[idx]), self.y[idx]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        x, y = self.__getitems__([idx])
        return x[0], int(y[0])


# --------------------------------------------------------------------------------------
# Loading the arrays
# --------------------------------------------------------------------------------------

def _resolve_dir(data_dir: str | Path) -> Path:
    data_dir = Path(data_dir)
    # Tolerate being pointed at the parent data root rather than the subdirectory.
    if not (data_dir / "meta.json").exists() and (data_dir / DEFAULT_SUBDIR).is_dir():
        data_dir = data_dir / DEFAULT_SUBDIR
    return data_dir


def read_meta(data_dir: str | Path) -> dict:
    """The build's metadata, including the resize convention the arrays were made
    under — quote it in provenance rather than assuming it."""
    return json.loads((_resolve_dir(data_dir) / "meta.json").read_text())


def _load_split(data_dir: Path, split: str, resolution: int, mmap: bool):
    x_path = data_dir / f"{split}_x{resolution}.npy"
    y_path = data_dir / f"{split}_y.npy"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            f"Downsampled ImageNet not found under {data_dir} (missing {x_path.name} "
            f"or {y_path.name}). Bake it onto the snapshot first: "
            "python scripts/bootstrap_thunder_snapshot.py --imagenet-small"
        )
    x = np.load(x_path, mmap_mode="r" if mmap else None)
    y = np.load(y_path)
    return x, y


def _apply_subsets(x, y, class_subset, subset_per_class):
    """Class filter (labels remapped to 0..N-1) and a deterministic first-k-per-class
    train subset, in that order — the same order and the same determinism as
    ``cifar100.build_loaders``, so paired comparisons stay paired."""
    if class_subset is not None:
        keep = sorted({int(c) for c in class_subset})
        remap = {c: i for i, c in enumerate(keep)}
        mask = np.isin(y, keep)
        idx = np.nonzero(mask)[0]
        x, y = x[idx], np.array([remap[int(t)] for t in y[idx]], dtype=np.int16)
    if subset_per_class is not None:
        if subset_per_class < 1:
            raise ValueError(f"subset_per_class must be >= 1, got {subset_per_class}")
        counts: dict[int, int] = {}
        keep_idx = []
        for i, t in enumerate(y):
            t = int(t)
            if counts.get(t, 0) < subset_per_class:
                counts[t] = counts.get(t, 0) + 1
                keep_idx.append(i)
        idx = np.asarray(keep_idx)
        x, y = x[idx], y[idx]
    return np.ascontiguousarray(x), np.ascontiguousarray(y)


# --------------------------------------------------------------------------------------
# The builders
# --------------------------------------------------------------------------------------

def build_loaders(
    data_dir: str | Path,
    resolution: int = 32,
    batch_size: int = 128,
    num_workers: int = 0,
    persistent_workers: bool = False,
    download: bool = False,
    augment: bool = True,
    cache_device: str | None = "cuda",
    class_subset: list[int] | None = None,
    subset_per_class: int | None = None,
    drop_last: bool = False,
    mmap: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """Train and validation loaders over the baked downsampled-ImageNet arrays.

    ``cache_device`` is where the pixel array lives and where the augmentation runs;
    ``"cuda"`` is the point of this module, and ``None`` keeps it in host memory as a
    fallback for a GPU too small to hold it (15GiB at 64 pixels). ``download`` exists
    only for parity with the other dataset builders and must be False — the arrays are
    baked onto the snapshot, never fetched at train time.
    """
    if download:
        raise ValueError(
            "Downsampled ImageNet cannot be auto-downloaded at train time; bake it "
            "with scripts/prefetch_imagenet_small.py (download=False here)."
        )
    if cache_device == "cuda" and not torch.cuda.is_available():
        cache_device = None
    if num_workers:
        # The array and the augmentation are already on the accelerator; workers
        # would each copy the array and add IPC for no gain.
        raise ValueError(
            "imagenet_small runs in-process (num_workers=0): the pixel array is "
            "resident and the augmentation is batched on its device."
        )

    data_dir = _resolve_dir(data_dir)
    train_x, train_y = _load_split(data_dir, "train", resolution, mmap)
    val_x, val_y = _load_split(data_dir, "val", resolution, mmap)
    train_x, train_y = _apply_subsets(train_x, train_y, class_subset, subset_per_class)
    val_x, val_y = _apply_subsets(val_x, val_y, class_subset, None)

    train_set = ArrayImageDataset(train_x, train_y, device=cache_device, augment=augment)
    val_set = ArrayImageDataset(val_x, val_y, device=cache_device, augment=False)

    def _loader(ds: ArrayImageDataset, shuffle: bool, drop: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            # Resident batches must not be pinned and need no host->device staging.
            pin_memory=False,
            persistent_workers=False,
            collate_fn=_identity_collate,
            drop_last=drop,
        )

    return _loader(train_set, True, drop_last), _loader(val_set, False, False)


def build_train_eval_loader(
    data_dir: str | Path,
    resolution: int = 32,
    batch_size: int = 256,
    num_workers: int = 0,
    persistent_workers: bool = False,
    download: bool = False,
    cache_device: str | None = "cuda",
    class_subset: list[int] | None = None,
    subset_per_class: int | None = None,
    mmap: bool = False,
) -> DataLoader:
    """No-augmentation, no-shuffle pass over the *train* split, for scoring a fixed
    snapshot in eval mode without the train-pass confounds (augmentation, BatchNorm
    train-mode statistics, weights moving between batches)."""
    train_loader, _ = build_loaders(
        data_dir,
        resolution=resolution,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        download=download,
        augment=False,
        cache_device=cache_device,
        class_subset=class_subset,
        subset_per_class=subset_per_class,
        mmap=mmap,
    )
    return DataLoader(
        train_loader.dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=_identity_collate,
    )
