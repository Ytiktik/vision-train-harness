"""CIFAR-10 dataset construction."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Sampler, Subset
from torchvision import datasets, transforms


class _AddGaussianNoise:
    def __init__(self, sigma: float, seed: int = 0) -> None:
        self.sigma = float(sigma)
        self._generator = torch.Generator().manual_seed(int(seed))

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        noise = torch.empty_like(tensor).normal_(mean=0.0, std=self.sigma, generator=self._generator)
        return tensor + noise


def _build_corruption(corruption: dict[str, Any], seed: int) -> tuple[list, list]:
    """Return (pre_totensor_transforms, post_normalize_transforms) for the corruption spec."""
    ctype = corruption["type"]
    if ctype == "gaussian_noise":
        sigma = float(corruption.get("severity", 0.1))
        return [], [_AddGaussianNoise(sigma=sigma, seed=seed)]
    if ctype == "gaussian_blur":
        kernel = int(corruption.get("kernel", 3))
        sigma = float(corruption.get("sigma", 1.0))
        return [transforms.GaussianBlur(kernel_size=kernel, sigma=sigma)], []
    raise ValueError(f"Unknown corruption type: {ctype!r}")


def _subset_per_class_indices(targets: list[int], n_per_class: int, seed: int) -> list[int]:
    targets_t = torch.as_tensor(targets)
    classes = torch.unique(targets_t).tolist()
    generator = torch.Generator().manual_seed(int(seed))
    kept: list[int] = []
    for c in classes:
        idx = (targets_t == c).nonzero(as_tuple=True)[0]
        perm = torch.randperm(idx.numel(), generator=generator)
        kept.extend(idx[perm[:n_per_class]].tolist())
    kept.sort()
    return kept


def _flip_labels(targets: list[int], fraction: float, num_classes: int, seed: int) -> list[int]:
    n = len(targets)
    n_flip = int(round(fraction * n))
    generator = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(n, generator=generator)
    flip_idx = perm[:n_flip].tolist()
    new_targets = list(targets)
    for i in flip_idx:
        original = new_targets[i]
        # Sample a class != original uniformly.
        offset = int(torch.randint(1, num_classes, (1,), generator=generator).item())
        new_targets[i] = (original + offset) % num_classes
    return new_targets


def _dataset_targets(dataset: torch.utils.data.Dataset) -> list[int]:
    """Integer targets of a dataset, following Subset indirection."""
    if isinstance(dataset, Subset):
        base = _dataset_targets(dataset.dataset)
        return [base[i] for i in dataset.indices]
    raw_targets = dataset.targets  # type: ignore[attr-defined]
    if isinstance(raw_targets, torch.Tensor):
        raw_targets = raw_targets.tolist()
    return list(raw_targets)


class _RelabelSubset(torch.utils.data.Dataset):
    """Subset of a dataset restricted to `keep_classes`, with labels remapped to 0..N-1."""

    def __init__(self, dataset: torch.utils.data.Dataset, indices: list[int], remap: dict[int, int]) -> None:
        self.dataset = dataset
        self.indices = indices
        self.remap = remap
        raw_targets = _dataset_targets(dataset)
        self.targets: list[int] = [remap[raw_targets[i]] for i in indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple:
        x, y = self.dataset[self.indices[idx]]
        return x, self.remap[int(y)]


def _filter_to_classes(
    dataset: torch.utils.data.Dataset, keep_classes: list[int]
) -> "_RelabelSubset":
    """Filter dataset to keep_classes and remap labels to 0..N-1 (sorted order)."""
    keep_set = set(keep_classes)
    remap = {c: i for i, c in enumerate(sorted(keep_classes))}
    raw_targets = _dataset_targets(dataset)
    indices = [i for i, t in enumerate(raw_targets) if t in keep_set]
    return _RelabelSubset(dataset, indices, remap)


class ClassConcentratedBatchSampler(Sampler[list[int]]):
    """Each batch contains exactly k_per_class examples from each of m distinct classes.

    Guaranteed distinct classes per batch: a per-class pointer tracks which chunk to
    serve next; at each step m classes are sampled without replacement from those that
    still have remaining chunks. Effective batch size = k_per_class * classes_per_batch.
    """

    def __init__(
        self,
        targets: list[int],
        k_per_class: int,
        classes_per_batch: int,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        if k_per_class < 1:
            raise ValueError(f"k_per_class must be >= 1, got {k_per_class}")
        if classes_per_batch < 1:
            raise ValueError(f"classes_per_batch must be >= 1, got {classes_per_batch}")
        self.targets = list(targets)
        self.k = int(k_per_class)
        self.m = int(classes_per_batch)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0

        targets_t = torch.as_tensor(self.targets)
        self.classes = sorted(torch.unique(targets_t).tolist())
        self.indices_by_class = {
            int(c): (targets_t == c).nonzero(as_tuple=True)[0].tolist()
            for c in self.classes
        }

    def __iter__(self) -> Iterator[list[int]]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1

        # Build per-class chunk lists with independently shuffled examples.
        class_chunks: dict[int, list[list[int]]] = {}
        for c, indices in self.indices_by_class.items():
            perm = torch.randperm(len(indices), generator=generator).tolist()
            shuffled = [indices[i] for i in perm]
            chunks = []
            for start in range(0, len(shuffled), self.k):
                chunk = shuffled[start : start + self.k]
                if len(chunk) == self.k or not self.drop_last:
                    chunks.append(chunk)
            class_chunks[c] = chunks

        # Per-class pointers. At each step pick m distinct classes at random.
        ptrs = {c: 0 for c in self.classes}
        while True:
            available = [c for c in self.classes if ptrs[c] < len(class_chunks[c])]
            if len(available) < self.m:
                break
            chosen_idx = torch.randperm(len(available), generator=generator)[: self.m].tolist()
            batch: list[int] = []
            for i in chosen_idx:
                c = available[i]
                batch.extend(class_chunks[c][ptrs[c]])
                ptrs[c] += 1
            yield batch

    def __len__(self) -> int:
        total_chunks = 0
        for indices in self.indices_by_class.values():
            full, remainder = divmod(len(indices), self.k)
            total_chunks += full
            if remainder and not self.drop_last:
                total_chunks += 1
        return total_chunks // self.m


class _CachedTensorDataset(torch.utils.data.Dataset):
    """Decoded-once tensor copy of a deterministic (augmentation-free) dataset.

    Only valid when the transform has no randomness; ``build_loaders`` enforces
    that by rejecting ``cache_tensors`` unless ``augment=False``. This matters at
    full batch: DataLoader parallelizes across *batches*, so a one-batch-per-epoch
    loader re-decodes the whole split in a single worker every epoch (the other
    ``num_workers`` sit idle) and the run is dataloader-bound rather than
    GPU-bound.
    """

    def __init__(self, dataset: torch.utils.data.Dataset, device: str | None = None) -> None:
        loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0)
        xs, ys = [], []
        for x, y in loader:
            xs.append(x)
            ys.append(y)
        self.x = torch.cat(xs)
        self.y = torch.cat(ys)
        self.targets: list[int] = self.y.tolist()
        if device is not None:
            # GPU-resident cache: at full batch the per-epoch host->device copy of
            # the (identical) split dominates the epoch ~100x over the GD step
            # itself; keeping the tensors on-device leaves only the permutation.
            self.x = self.x.to(device)
            self.y = self.y.to(device)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> tuple:
        return self.x[idx], int(self.y[idx])

    def __getitems__(self, indices: list[int]) -> tuple:
        # Batched fetch: DataLoader hands the whole index list here and passes the
        # result straight to collate_fn, so one advanced-index replaces N Python
        # __getitem__ calls plus a 10k-tensor default_collate stack.
        idx = torch.as_tensor(indices, device=self.x.device)
        return self.x[idx], self.y[idx]


def _identity_collate(batch: Any) -> Any:
    """Pass through the already-stacked (x, y) from _CachedTensorDataset.__getitems__."""
    return batch


def build_loaders(
    data_dir: str | Path,
    batch_size: int = 128,
    num_workers: int = 4,
    download: bool = True,
    subset_per_class: int | None = None,
    corruption: dict[str, Any] | None = None,
    label_noise: float | None = None,
    noise_seed: int = 0,
    persistent_workers: bool = False,
    batch_mode: str = "shuffled",
    sampler_seed: int = 0,
    classes_per_batch: int | None = None,
    keep_classes: list[int] | None = None,
    augment: bool = True,
    cache_tensors: bool = False,
    cache_device: str | None = None,
) -> tuple[DataLoader, DataLoader]:
    data_dir = Path(data_dir)
    if cache_device is not None and not cache_tensors:
        raise ValueError("cache_device requires cache_tensors=True.")
    if cache_tensors and augment:
        raise ValueError(
            "cache_tensors=True requires augment=False: caching a random transform "
            "would silently freeze one augmentation draw for the whole run."
        )

    pre_train, post_train = ([], [])
    pre_test, post_test = ([], [])
    if corruption is not None:
        pre_train, post_train = _build_corruption(corruption, seed=noise_seed)
        pre_test, post_test = _build_corruption(corruption, seed=noise_seed + 1)

    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))

    aug_ops = (
        [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
        if augment
        else []
    )
    train_transform = transforms.Compose(
        [
            *aug_ops,
            *pre_train,
            transforms.ToTensor(),
            normalize,
            *post_train,
        ]
    )
    test_transform = transforms.Compose(
        [
            *pre_test,
            transforms.ToTensor(),
            normalize,
            *post_test,
        ]
    )

    train_set = datasets.CIFAR10(data_dir, train=True, transform=train_transform, download=download)
    test_set = datasets.CIFAR10(data_dir, train=False, transform=test_transform, download=download)

    if label_noise is not None and label_noise > 0:
        if not 0 <= label_noise < 1:
            raise ValueError(f"label_noise must be in [0, 1), got {label_noise}")
        train_set.targets = _flip_labels(
            train_set.targets, fraction=label_noise, num_classes=10, seed=noise_seed
        )

    if subset_per_class is not None:
        idx = _subset_per_class_indices(train_set.targets, subset_per_class, seed=noise_seed)
        train_set = Subset(train_set, idx)

    if keep_classes is not None:
        train_set = _filter_to_classes(train_set, keep_classes)
        test_set = _filter_to_classes(test_set, keep_classes)

    if cache_tensors:
        # Cache after subset/keep_classes so only the kept examples are decoded.
        # Worker processes would each copy the cache and add IPC for no gain, so
        # the cached path runs in-process.
        train_set = _CachedTensorDataset(train_set, device=cache_device)
        num_workers = 0
        persistent_workers = False

    # Batches that already live on an accelerator must not be pinned (and need
    # no host->device staging).
    pin_train = torch.cuda.is_available() and cache_device is None
    use_persistent = persistent_workers and num_workers > 0
    if batch_mode == "shuffled":
        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_train,
            persistent_workers=use_persistent,
            prefetch_factor=4 if num_workers > 0 else None,
            collate_fn=_identity_collate if cache_tensors else None,
        )
    elif batch_mode == "class_concentrated":
        if classes_per_batch is None:
            raise ValueError(
                "batch_mode='class_concentrated' requires classes_per_batch."
            )
        k = batch_size // int(classes_per_batch)
        if k < 1:
            raise ValueError(
                f"batch_size={batch_size} too small for classes_per_batch={classes_per_batch} (k={k})."
            )
        targets = train_set.targets if not isinstance(train_set, Subset) else [train_set.dataset.targets[i] for i in train_set.indices]
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
        raise ValueError(
            f"Unknown batch_mode: {batch_mode!r}. Expected 'shuffled' or 'class_concentrated'."
        )
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


def build_train_eval_loader(
    data_dir: str | Path,
    batch_size: int = 256,
    num_workers: int = 4,
    persistent_workers: bool = False,
    download: bool = True,
    keep_classes: list[int] | None = None,
    subset_per_class: int | None = None,
    noise_seed: int = 0,
) -> DataLoader:
    """No-augmentation, no-shuffle loader over the CIFAR-10 *train* split.

    subset_per_class/noise_seed must match build_loaders so the same subset is
    reproduced (selection is deterministic in the seed).
    """
    data_dir = Path(data_dir)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    train_set = datasets.CIFAR10(data_dir, train=True, transform=transform, download=download)
    if subset_per_class is not None:
        idx = _subset_per_class_indices(train_set.targets, subset_per_class, seed=noise_seed)
        train_set = Subset(train_set, idx)
    if keep_classes is not None:
        train_set = _filter_to_classes(train_set, keep_classes)
    return DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )
