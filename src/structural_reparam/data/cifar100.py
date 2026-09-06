"""CIFAR-100 dataset construction."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Sampler
from torchvision import datasets, transforms


CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)

CIFAR100_NUM_SUPERCLASSES = 20

# Standard CIFAR-100 fine→coarse mapping: fine class id (0..99) → super class id (0..19).
# Values match the `coarse_labels` array shipped in the official `cifar-100-python` pickle.
CIFAR100_FINE_TO_COARSE: tuple[int, ...] = (
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
)


class _ClassSubsetDataset(torch.utils.data.Dataset):
    """CIFAR-100 filtered to a subset of classes with labels remapped to 0..N-1."""

    def __init__(
        self,
        base: torch.utils.data.Dataset,
        class_subset: list[int],
    ) -> None:
        keep = sorted(set(class_subset))
        self._label_map = {orig: new for new, orig in enumerate(keep)}
        self._indices = [
            i for i, t in enumerate(base.targets) if t in self._label_map
        ]
        self._base = base

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int):
        img, label = self._base[self._indices[idx]]
        return img, self._label_map[label]

    @property
    def targets(self) -> list[int]:
        return [self._label_map[self._base.targets[i]] for i in self._indices]


class _PerClassSubsetDataset(torch.utils.data.Dataset):
    """First-k-per-class deterministic train subset; labels unchanged.

    Selection is the first k occurrences of each class in the base dataset's
    fixed storage order, so every arm/seed of an experiment sees the identical
    subset (paired comparisons stay paired)."""

    def __init__(self, base: torch.utils.data.Dataset, per_class: int) -> None:
        if per_class < 1:
            raise ValueError(f"subset_per_class must be >= 1, got {per_class}")
        counts: dict[int, int] = {}
        self._indices = []
        for i, t in enumerate(base.targets):
            t = int(t)
            if counts.get(t, 0) < per_class:
                counts[t] = counts.get(t, 0) + 1
                self._indices.append(i)
        self._base = base

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int):
        return self._base[self._indices[idx]]

    @property
    def targets(self) -> list[int]:
        return [int(self._base.targets[i]) for i in self._indices]


class _LabelNoiseDataset(torch.utils.data.Dataset):
    """Deterministic label noise applied on top of any train dataset.

    mode='uniform': flip to a uniformly random wrong class (heat-like,
    incoherent conflict). mode='within_superclass': flip to a random SIBLING
    fine class of the same CIFAR-100 superclass (coherent, confusable-pair
    conflict — requires the full 100-class label space). The flip pattern
    depends only on (fraction, mode, seed) and the base dataset's target
    order — NOT on the experiment seed — so every arm of a paired comparison
    trains against the identical corrupted labels."""

    def __init__(self, base: torch.utils.data.Dataset, fraction: float,
                 num_classes: int, seed: int, mode: str = "uniform") -> None:
        from structural_reparam.data.cifar10 import _flip_labels
        if not 0 <= fraction < 1:
            raise ValueError(f"label_noise must be in [0, 1), got {fraction}")
        self._base = base
        if mode == "uniform":
            self._targets = _flip_labels([int(x) for x in base.targets], fraction,
                                         num_classes, seed)
        elif mode == "within_superclass":
            if num_classes != 100:
                raise ValueError("within_superclass noise needs the full "
                                 "100-class label space (no class_subset)")
            targets = [int(x) for x in base.targets]
            siblings = {c: [f for f in range(100)
                            if CIFAR100_FINE_TO_COARSE[f] == CIFAR100_FINE_TO_COARSE[c]
                            and f != c] for c in range(100)}
            n = len(targets)
            n_flip = int(round(fraction * n))
            g = torch.Generator().manual_seed(int(seed))
            perm = torch.randperm(n, generator=g)
            new_targets = list(targets)
            for i in perm[:n_flip].tolist():
                sibs = siblings[new_targets[i]]
                j = int(torch.randint(0, len(sibs), (1,), generator=g).item())
                new_targets[i] = sibs[j]
            self._targets = new_targets
        else:
            raise ValueError(f"unknown label_noise_mode {mode!r}")

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: int):
        img, _ = self._base[idx]
        return img, self._targets[idx]

    @property
    def targets(self) -> list[int]:
        return list(self._targets)


class _SuperClassBinaryDataset(torch.utils.data.Dataset):
    """CIFAR-100 filtered to two superclasses with labels remapped to 0/1."""

    def __init__(
        self,
        base: torch.utils.data.Dataset,
        superclass_pair: list[int],
    ) -> None:
        pair = list(superclass_pair)
        if len(pair) != 2 or len(set(pair)) != 2:
            raise ValueError(f"superclass_pair must contain two distinct superclass ids, got {pair}")
        for c in pair:
            if c < 0 or c >= CIFAR100_NUM_SUPERCLASSES:
                raise ValueError(
                    f"superclass id {c} must be in [0, {CIFAR100_NUM_SUPERCLASSES - 1}]"
                )
        self._label_map = {int(pair[0]): 0, int(pair[1]): 1}
        self._base = base
        self._indices = [
            i
            for i, fine_label in enumerate(base.targets)
            if CIFAR100_FINE_TO_COARSE[int(fine_label)] in self._label_map
        ]

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int):
        img, fine_label = self._base[self._indices[idx]]
        coarse_label = CIFAR100_FINE_TO_COARSE[int(fine_label)]
        return img, self._label_map[coarse_label]


    @property
    def targets(self) -> list[int]:
        return [
            self._label_map[CIFAR100_FINE_TO_COARSE[int(self._base.targets[i])]]
            for i in self._indices
        ]


class ClassBlockedBatchSampler(Sampler[list[int]]):
    """Yield batches containing examples from a single class."""

    def __init__(
        self,
        targets: list[int],
        batch_size: int,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self.targets = list(targets)
        self.batch_size = int(batch_size)
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

        class_order = torch.randperm(len(self.classes), generator=generator).tolist()
        for class_pos in class_order:
            c = self.classes[class_pos]
            indices = self.indices_by_class[c]
            perm = torch.randperm(len(indices), generator=generator).tolist()
            shuffled = [indices[i] for i in perm]
            for start in range(0, len(shuffled), self.batch_size):
                batch = shuffled[start : start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    yield batch

    def __len__(self) -> int:
        total = 0
        for indices in self.indices_by_class.values():
            full, remainder = divmod(len(indices), self.batch_size)
            total += full
            if remainder and not self.drop_last:
                total += 1
        return total


class ClassConcentratedBatchSampler(Sampler[list[int]]):
    """Each batch contains k_per_class examples from each of classes_per_batch classes.

    All available examples are split into chunks of k_per_class per class, the chunks
    are globally shuffled, then grouped m at a time to form batches.  This gives
    intra-batch gradient conflict (multiple classes present) while concentrating the
    signal far more than a fully-shuffled loader.  Effective batch size = k * m.
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

    def _make_chunks(self, generator: torch.Generator) -> list[list[int]]:
        chunks = []
        for c, indices in self.indices_by_class.items():
            perm = torch.randperm(len(indices), generator=generator).tolist()
            shuffled = [indices[i] for i in perm]
            for start in range(0, len(shuffled), self.k):
                chunk = shuffled[start : start + self.k]
                if len(chunk) == self.k or not self.drop_last:
                    chunks.append(chunk)
        return chunks

    def __iter__(self) -> Iterator[list[int]]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1

        chunks = self._make_chunks(generator)
        perm = torch.randperm(len(chunks), generator=generator).tolist()
        chunks = [chunks[i] for i in perm]

        for start in range(0, len(chunks) - self.m + 1, self.m):
            batch: list[int] = []
            for chunk in chunks[start : start + self.m]:
                batch.extend(chunk)
            yield batch

    def __len__(self) -> int:
        total_chunks = 0
        for indices in self.indices_by_class.values():
            full, remainder = divmod(len(indices), self.k)
            total_chunks += full
            if remainder and not self.drop_last:
                total_chunks += 1
        return total_chunks // self.m


def build_loaders(
    data_dir: str | Path,
    batch_size: int = 128,
    num_workers: int = 4,
    persistent_workers: bool = True,
    download: bool = True,
    batch_mode: str = "shuffled",
    sampler_seed: int = 0,
    class_subset: list[int] | None = None,
    superclass_pair: list[int] | None = None,
    subset_per_class: int | None = None,
    label_noise: float | None = None,
    label_noise_mode: str = "uniform",
    noise_seed: int = 777,
    k_per_class: int | None = None,
    classes_per_batch: int | None = None,
    augment: bool = True,
    cache_tensors: bool = False,
    cache_device: str | None = None,
) -> tuple[DataLoader, DataLoader]:
    if cache_device is not None and not cache_tensors:
        raise ValueError("cache_device requires cache_tensors=True.")
    if cache_tensors and augment:
        raise ValueError(
            "cache_tensors=True requires augment=False: caching a random transform "
            "would silently freeze one augmentation draw for the whole run."
        )
    if cache_tensors and batch_mode != "shuffled":
        raise ValueError("cache_tensors=True is only supported with batch_mode='shuffled'.")
    if batch_mode == "class_blocked":
        raise ValueError(
            "batch_mode='class_blocked' is deprecated (2026-07-08) and disabled; use "
            "'shuffled' or 'class_concentrated'. ClassBlockedBatchSampler is kept for "
            "possible future use."
        )
    data_dir = Path(data_dir)
    aug_ops = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()] if augment else []
    train_transform = transforms.Compose(
        [
            *aug_ops,
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )

    train_set = datasets.CIFAR100(data_dir, train=True, transform=train_transform, download=download)
    test_set = datasets.CIFAR100(data_dir, train=False, transform=test_transform, download=download)
    if class_subset is not None and superclass_pair is not None:
        raise ValueError("Pass either class_subset or superclass_pair, not both.")
    if class_subset is not None:
        train_set = _ClassSubsetDataset(train_set, class_subset)
        test_set = _ClassSubsetDataset(test_set, class_subset)
    elif superclass_pair is not None:
        train_set = _SuperClassBinaryDataset(train_set, superclass_pair)
        test_set = _SuperClassBinaryDataset(test_set, superclass_pair)
    if subset_per_class is not None:
        # Train-set only: shrinks the fitting problem, leaves eval sets intact.
        train_set = _PerClassSubsetDataset(train_set, subset_per_class)
    if label_noise is not None and label_noise > 0:
        # After subsetting: the noisy cell uses exactly the clean cell's images.
        n_cls = len(class_subset) if class_subset is not None else 100
        train_set = _LabelNoiseDataset(train_set, label_noise, n_cls, noise_seed,
                                       mode=label_noise_mode)
    if cache_tensors:
        # Cache after class filtering so only the kept examples are decoded.
        # Worker processes would each copy the cache and add IPC for no gain, so
        # the cached path runs in-process. (Mirrors cifar10.build_loaders.)
        from structural_reparam.data.cifar10 import _CachedTensorDataset

        train_set = _CachedTensorDataset(train_set, device=cache_device)
        num_workers = 0
        persistent_workers = False
    # Batches that already live on an accelerator must not be pinned (and need
    # no host->device staging).
    pin_train = torch.cuda.is_available() and cache_device is None
    _persistent = persistent_workers and num_workers > 0
    if batch_mode == "shuffled":
        from structural_reparam.data.cifar10 import _identity_collate

        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_train,
            persistent_workers=_persistent,
            prefetch_factor=4 if num_workers > 0 else None,
            collate_fn=_identity_collate if cache_tensors else None,
        )
    elif batch_mode == "class_blocked":
        train_loader = DataLoader(
            train_set,
            batch_sampler=ClassBlockedBatchSampler(
                train_set.targets,
                batch_size=batch_size,
                seed=sampler_seed,
            ),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=_persistent,
            prefetch_factor=4 if num_workers > 0 else None,
        )
    elif batch_mode == "class_concentrated":
        if k_per_class is None or classes_per_batch is None:
            raise ValueError(
                "batch_mode='class_concentrated' requires k_per_class and classes_per_batch."
            )
        train_loader = DataLoader(
            train_set,
            batch_sampler=ClassConcentratedBatchSampler(
                train_set.targets,
                k_per_class=int(k_per_class),
                classes_per_batch=int(classes_per_batch),
                seed=sampler_seed,
            ),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=_persistent,
            prefetch_factor=4 if num_workers > 0 else None,
        )
    else:
        raise ValueError(
            f"Unknown batch_mode: {batch_mode!r}. "
            "Expected 'shuffled' or 'class_concentrated' ('class_blocked' is deprecated)."
        )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=_persistent,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    return train_loader, test_loader


def build_train_eval_loader(
    data_dir: str | Path,
    batch_size: int = 256,
    num_workers: int = 4,
    persistent_workers: bool = True,
    download: bool = True,
    class_subset: list[int] | None = None,
    superclass_pair: list[int] | None = None,
    subset_per_class: int | None = None,
    label_noise: float | None = None,
    label_noise_mode: str = "uniform",
    noise_seed: int = 777,
) -> DataLoader:
    """No-augmentation, no-shuffle loader over the CIFAR-100 *train* split.

    Used to compute per-class metrics on a fixed model snapshot in eval mode,
    avoiding the train-pass confounds (data aug, BN train-mode stats, weights
    that change between batches).
    """
    data_dir = Path(data_dir)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )
    train_set = datasets.CIFAR100(data_dir, train=True, transform=transform, download=download)
    if class_subset is not None and superclass_pair is not None:
        raise ValueError("Pass either class_subset or superclass_pair, not both.")
    if class_subset is not None:
        train_set = _ClassSubsetDataset(train_set, class_subset)
    elif superclass_pair is not None:
        train_set = _SuperClassBinaryDataset(train_set, superclass_pair)
    # Same order as build_loaders, so this pass covers exactly the training
    # images and scores them against exactly the (possibly corrupted) labels
    # the run trained on. Both args were accepted and ignored before 2026-08-05.
    if subset_per_class is not None:
        train_set = _PerClassSubsetDataset(train_set, subset_per_class)
    if label_noise is not None and label_noise > 0:
        n_cls = len(class_subset) if class_subset is not None else 100
        train_set = _LabelNoiseDataset(train_set, label_noise, n_cls, noise_seed,
                                       mode=label_noise_mode)
    return DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )
