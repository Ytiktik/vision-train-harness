"""Full-resolution ImageNet-1k from indexed record shards.

The pixels are the original ILSVRC2012 JPEG bytes, copied verbatim by
``scripts/prefetch_imagenet_full.py`` into a few hundred ~1GiB shards with a global
index. This module is the reader. It is a drop-in alternative to
``structural_reparam.data.imagenet``, which reads the same images as 1.28 million
loose files; the transforms and the label order are identical, so the two are
interchangeable at equal fidelity.

What the layout buys. Thunder serves the ImageNet snapshot over a nydus lazy-loading
overlay — files arrive from network storage on first access, and ``imagenet.py``
carries a retry wrapper because reads transiently fail under many workers. Replacing
1.28M small-file opens per epoch with reads inside a few hundred large files removes
the per-file metadata cost, which is the part of that problem the loader cannot
retry its way out of.

What it keeps that a tar-shard pipeline would not: an index means O(1) access by
example id, so the sampler stays an ordinary global permutation and a probe can ask
for image i, rather than a shard shuffle plus a reservoir buffer.

Layout under ``data_dir`` (default ``data/imagenet_full``), written by the builder:
    train-00000.bin ...            raw concatenated JPEG bytes
    train_index.npy                shard/offset/length/label per image
    val-*.bin, val_index.npy, wnids.json, meta.json
"""

from __future__ import annotations

import io
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms


LOGGER = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_SUBDIR = "imagenet_full"


# --------------------------------------------------------------------------------------
# The dataset
# --------------------------------------------------------------------------------------

class RecordShardDataset(torch.utils.data.Dataset):
    """Random-access reader over the record shards.

    File handles are opened lazily and cached per process, so DataLoader workers each
    get their own after the fork; the dataset itself pickles as just the index and the
    paths. That is the reason handles are not opened in ``__init__``.
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str,
        transform=None,
        class_subset: list[int] | None = None,
    ) -> None:
        self.dir = Path(data_dir)
        self.split = split
        self.transform = transform
        index = np.load(self.dir / f"{split}_index.npy")
        self.shards = sorted(self.dir.glob(f"{split}-*.bin"))
        if not self.shards:
            raise FileNotFoundError(
                f"No {split} record shards under {self.dir}. Build the image first: "
                "python scripts/prefetch_imagenet_full.py --data-dir <dir>"
            )
        if len(index) and int(index["shard"].max()) >= len(self.shards):
            raise RuntimeError(
                f"{split} index references shard {int(index['shard'].max())} but only "
                f"{len(self.shards)} shards are present."
            )
        labels = index["label"].astype(np.int64)
        if class_subset is not None:
            keep = sorted({int(c) for c in class_subset})
            remap = {c: i for i, c in enumerate(keep)}
            sel = np.nonzero(np.isin(labels, keep))[0]
            index, labels = index[sel], np.array([remap[int(t)] for t in labels[sel]],
                                                dtype=np.int64)
        self.index = index
        self.labels = labels
        self.targets: list[int] = labels.tolist()
        self._handles: dict[int, object] = {}

    def __len__(self) -> int:
        return int(len(self.index))

    def _handle(self, shard: int):
        h = self._handles.get(shard)
        if h is None:
            h = open(self.shards[shard], "rb", buffering=0)
            self._handles[shard] = h
        return h

    def read_bytes(self, idx: int) -> bytes:
        """The original JPEG bytes of example ``idx``, exactly as ILSVRC2012 shipped."""
        row = self.index[idx]
        h = self._handle(int(row["shard"]))
        h.seek(int(row["offset"]))
        blob = h.read(int(row["length"]))
        if len(blob) != int(row["length"]):
            raise RuntimeError(
                f"short read for {self.split} example {idx}: got {len(blob)} of "
                f"{int(row['length'])} bytes from shard {int(row['shard'])}"
            )
        return blob

    def __getitem__(self, idx: int):
        from PIL import Image

        image = Image.open(io.BytesIO(self.read_bytes(idx))).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(self.labels[idx])

    def __getitems__(self, indices: list[int]) -> list:
        """Fetch a whole batch, reading the shards in offset order.

        The DataLoader hands the batch's index list here, so the reads can be issued
        in the order they lie on disk instead of the order the sampler drew them. The
        batch's contents and their positions are unchanged — the results are put back
        in the requested order before returning — so this is an I/O reordering and
        nothing else, and the tensors are identical to the per-item path.

        It matters here because a global permutation over 143GiB of shards means every
        batch touches many shards at random offsets, and the snapshot is served over a
        lazy-loading overlay where locality is the whole game.
        """
        rows = self.index[indices]
        order = np.lexsort((rows["offset"], rows["shard"]))
        out: list = [None] * len(indices)
        for k in order:
            k = int(k)
            out[k] = self[indices[k]]
        return out

    def prefetch_hint(self, indices) -> None:
        """Tell the kernel which byte ranges are wanted next (POSIX_FADV_WILLNEED).

        Advisory only: it changes no bytes and never fails the read if unsupported.
        """
        if not hasattr(os, "posix_fadvise"):
            return
        rows = self.index[indices]
        for row in rows[np.lexsort((rows["offset"], rows["shard"]))]:
            try:
                os.posix_fadvise(self._handle(int(row["shard"])).fileno(),
                                 int(row["offset"]), int(row["length"]),
                                 os.POSIX_FADV_WILLNEED)
            except OSError:
                return

    def warm_cache(self, block_bytes: int = 1 << 24, log_every: int = 32) -> float:
        """Read every shard sequentially once, to pull it into the page cache.

        This is the one-time cost that pays for the whole layout on Thunder. The
        snapshot arrives over a lazy-loading overlay, so the first touch of any byte
        fetches it from network storage; doing that as ~140 large sequential reads
        instead of 1.28 million random small ones is both far faster and the only way
        the 240GB-RAM instance's page cache ends up holding the set. Returns the bytes
        per second achieved, so a caller can log it.

        Reads and discards: it changes nothing on disk and nothing about the data.
        """
        import time

        total = 0
        started = time.time()
        for i, path in enumerate(self.shards):
            with open(path, "rb", buffering=0) as f:
                if hasattr(os, "posix_fadvise"):
                    try:
                        os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_SEQUENTIAL)
                    except OSError:
                        pass
                while True:
                    block = f.read(block_bytes)
                    if not block:
                        break
                    total += len(block)
            if log_every and (i + 1) % log_every == 0:
                rate = total / max(time.time() - started, 1e-9)
                LOGGER.info("warm_cache: %d/%d shards, %.1f GiB, %.0f MB/s",
                            i + 1, len(self.shards), total / 2**30, rate / 1e6)
        elapsed = max(time.time() - started, 1e-9)
        LOGGER.info("warm_cache: %s done, %.1f GiB in %.0fs (%.0f MB/s)",
                    self.split, total / 2**30, elapsed, total / elapsed / 1e6)
        return total / elapsed

    def __getstate__(self):
        # Never pickle open file handles to the workers.
        state = self.__dict__.copy()
        state["_handles"] = {}
        return state


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def _resolve_dir(data_dir: str | Path) -> Path:
    data_dir = Path(data_dir)
    if not (data_dir / "meta.json").exists() and (data_dir / DEFAULT_SUBDIR).is_dir():
        data_dir = data_dir / DEFAULT_SUBDIR
    return data_dir


def read_meta(data_dir: str | Path) -> dict:
    """The build's metadata. ``pixels_modified`` is False for this image: quote that in
    provenance rather than assuming the bytes are original."""
    return json.loads((_resolve_dir(data_dir) / "meta.json").read_text())


def build_transforms(image_size: int = 224, augment: bool = True):
    """The standard ImageNet pipeline, identical to ``imagenet.py``'s."""
    train = transforms.Compose(
        ([transforms.RandomResizedCrop(image_size), transforms.RandomHorizontalFlip()]
         if augment else
         [transforms.Resize(int(image_size * 256 / 224)), transforms.CenterCrop(image_size)])
        + [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    )
    test = transforms.Compose([
        transforms.Resize(int(image_size * 256 / 224)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train, test


# --------------------------------------------------------------------------------------
# The builders
# --------------------------------------------------------------------------------------

def default_workers() -> int:
    """One worker per core, less two left for the main process and the copy engine.

    The production A100 instances this image runs on come with 30 vCPUs for 2 GPUs
    and 16 for 1 (imagenet_full.vcpus_by_gpus), and full-resolution JPEG decoding is
    the binding constraint, so leaving cores idle is the main way to be slow here.
    """
    return max(1, min(32, (os.cpu_count() or 8) - 2))


def build_loaders(
    data_dir: str | Path,
    image_size: int = 224,
    batch_size: int = 256,
    num_workers: int | None = None,
    persistent_workers: bool = True,
    download: bool = False,
    augment: bool = True,
    class_subset: list[int] | None = None,
    drop_last: bool = True,
    warm: bool = False,
    prefetch_factor: int = 6,
) -> tuple[DataLoader, DataLoader]:
    """Train and validation loaders over the record shards.

    ``num_workers`` defaults to one per core less two (see ``default_workers``).
    ``warm`` reads every shard sequentially once before returning, which on Thunder's
    lazy-loading overlay is what turns the first epoch from a network-bound crawl into
    a page-cache read; it costs one pass over ~143GiB and changes nothing about the
    data. Leave it off when the caller only wants a few batches.

    ``download`` exists for parity with the other dataset builders and must be False:
    the shards are baked onto the snapshot, never fetched at train time.
    """
    if download:
        raise ValueError(
            "ImageNet cannot be auto-downloaded at train time; bake it with "
            "scripts/prefetch_imagenet_full.py (download=False here)."
        )
    data_dir = _resolve_dir(data_dir)
    train_tf, test_tf = build_transforms(image_size, augment)
    train_set = RecordShardDataset(data_dir, "train", train_tf, class_subset)
    val_set = RecordShardDataset(data_dir, "val", test_tf, class_subset)

    if warm:
        train_set.warm_cache()
        val_set.warm_cache()

    if num_workers is None:
        num_workers = default_workers()
    use_workers = num_workers > 0
    common = dict(
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=prefetch_factor if use_workers else None,
        persistent_workers=persistent_workers and use_workers,
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              drop_last=drop_last, **common)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, **common)
    return train_loader, val_loader


def build_train_eval_loader(
    data_dir: str | Path,
    image_size: int = 224,
    batch_size: int = 256,
    num_workers: int | None = None,
    persistent_workers: bool = True,
    download: bool = False,
    class_subset: list[int] | None = None,
) -> DataLoader:
    """No-augmentation, no-shuffle pass over the train split, for scoring a fixed
    snapshot in eval mode without the train-pass confounds."""
    if download:
        raise ValueError("download=False required; the shards are baked onto the snapshot.")
    data_dir = _resolve_dir(data_dir)
    _, test_tf = build_transforms(image_size, augment=False)
    dataset = RecordShardDataset(data_dir, "train", test_tf, class_subset)
    if num_workers is None:
        num_workers = default_workers()
    use_workers = num_workers > 0
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=4 if use_workers else None,
        persistent_workers=persistent_workers and use_workers,
    )
