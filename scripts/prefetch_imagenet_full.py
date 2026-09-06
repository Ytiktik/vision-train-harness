#!/usr/bin/env python3
"""Build the full-resolution ImageNet-1k image as indexed record shards.

Same source and transport as ``prefetch_imagenet_small.py``: the Kaggle
``imagenet-object-localization-challenge`` competition ships ILSVRC2012 as a ~166GB
zip, and we stream it rather than storing it. The difference is what comes out.
``prefetch_imagenet_small.py`` decodes and resizes; this one does NOT touch the
pixels at all — each JPEG's bytes are copied verbatim into a record shard, so the
result is bit-exact with the original ILSVRC2012 files and with every published
recipe measured on them.

Why shards and not an ImageFolder tree. The Thunder ImageNet snapshot is served over
a nydus lazy-loading overlay: files are fetched from network storage on first access,
and ``structural_reparam.data.imagenet.py`` carries a retry wrapper because reads
transiently fail under many workers. 1.28 million small-file opens per epoch against
that is the throughput problem. A few hundred ~1GB files replace the per-file
metadata lookups with reads inside a handful of large objects.

Why an index and not tar shards. A global index gives O(1) random access by example
id, so a probe can ask for image i and a loader can take a true global permutation
each epoch, rather than the shard-shuffle-plus-buffer approximation a WebDataset-style
pipeline forces. It also needs no runtime dependency: reading is a seek and a read.

Format under ``--data-dir`` (recorded in meta.json):
    train-00000.bin ...            raw concatenated JPEG bytes, ~1GB each
    val-00000.bin ...
    train_index.npy / val_index.npy   structured array, one row per image:
        shard  uint16   which .bin file
        offset uint64   byte offset of the JPEG within that shard
        length uint32   its length in bytes
        label  uint16   class id, wnids sorted lexicographically -> 0..999
    wnids.json, meta.json

Labels follow torchvision's ImageFolder convention so they agree with any
ImageFolder reading of the same data.

    KAGGLE_API_TOKEN=... python scripts/prefetch_imagenet_full.py \
        --data-dir /home/ubuntu/data/imagenet_full
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


DATA_PREFIX = "ILSVRC/Data/CLS-LOC"
TRAIN_PREFIX = f"{DATA_PREFIX}/train/"
VAL_PREFIX = f"{DATA_PREFIX}/val/"
VAL_SOLUTION = "LOC_val_solution.csv"

N_TRAIN = 1_281_167
N_VAL = 50_000
N_CLASSES = 1000

FORMAT_VERSION = 1
SHARD_BYTES = 1 << 30           # ~1GiB per shard
VALIDATE_EVERY = 2000           # decode 1 image in N to prove the bytes survived

INDEX_DTYPE = np.dtype([
    ("shard", "<u2"), ("offset", "<u8"), ("length", "<u4"), ("label", "<u2"),
])


# --------------------------------------------------------------------------------------
# Shard writing
# --------------------------------------------------------------------------------------

class ShardWriter:
    """Append raw bytes to a rolling series of ~1GiB files, returning placements."""

    def __init__(self, out_dir: Path, split: str, shard_bytes: int = SHARD_BYTES) -> None:
        self.dir = out_dir
        self.split = split
        self.shard_bytes = int(shard_bytes)
        self.index = 0
        self.offset = 0
        self.handle = None
        self._open()

    def _path(self, i: int) -> Path:
        return self.dir / f"{self.split}-{i:05d}.bin"

    def _open(self) -> None:
        self.handle = open(self._path(self.index), "wb", buffering=1 << 22)
        self.offset = 0

    def write(self, data: bytes) -> tuple[int, int, int]:
        if self.offset and self.offset + len(data) > self.shard_bytes:
            self.close()
            self.index += 1
            self._open()
        start = self.offset
        self.handle.write(data)
        self.offset += len(data)
        return self.index, start, len(data)

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None

    @property
    def shard_count(self) -> int:
        return self.index + 1


# --------------------------------------------------------------------------------------
# The Kaggle stream
# --------------------------------------------------------------------------------------

def download_url(competition: str) -> str:
    return f"https://www.kaggle.com/api/v1/competitions/data/download-all/{competition}"


def require_tools() -> None:
    if shutil.which("curl") is None:
        raise SystemExit("[in-full] curl not found (sudo apt-get install -y curl).")
    try:
        import stream_unzip  # noqa: F401
    except ImportError as exc:
        raise SystemExit("[in-full] pip install stream-unzip") from exc


def _curl(url: str, token: str) -> subprocess.Popen:
    cmd = ["curl", "-sS", "-L", "--retry", "5", "--retry-delay", "15",
           "-H", f"Authorization: Bearer {token}", url]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=1 << 20)


def _stdout_chunks(proc: subprocess.Popen, size: int = 1 << 20):
    while True:
        chunk = proc.stdout.read(size)
        if not chunk:
            return
        yield chunk


def looks_populated(out_dir: Path) -> bool:
    try:
        meta = json.loads((out_dir / "meta.json").read_text())
        return bool(meta.get("complete"))
    except Exception:  # noqa: BLE001
        return False


# --------------------------------------------------------------------------------------
# The pass over the archive
# --------------------------------------------------------------------------------------

def stream_build(competition: str, token: str, out_dir: Path, limit: int | None,
                 shard_bytes: int) -> None:
    from stream_unzip import stream_unzip

    writers = {"train": ShardWriter(out_dir, "train", shard_bytes),
               "val": ShardWriter(out_dir, "val", shard_bytes)}
    rows = {"train": [], "val": []}
    wnid_of_index: list[str] = []
    val_ids: list[str] = []
    val_csv: bytes | None = None
    n_seen = {"train": 0, "val": 0}
    n_bad = 0
    checked = 0

    proc = _curl(download_url(competition), token)
    started = time.time()
    last_report = started

    for name_b, _size, entry_chunks in stream_unzip(_stdout_chunks(proc)):
        name = name_b.decode("utf-8", "replace")
        if name == VAL_SOLUTION:
            val_csv = b"".join(entry_chunks)
            continue
        is_train = name.startswith(TRAIN_PREFIX)
        is_val = name.startswith(VAL_PREFIX)
        if not (is_train or is_val) or name.endswith("/"):
            for _ in entry_chunks:      # the stream only advances as we read
                pass
            continue
        split = "train" if is_train else "val"
        if limit and n_seen[split] >= limit:
            for _ in entry_chunks:
                pass
            continue
        data = b"".join(entry_chunks)

        # Sample-validate: prove the copied bytes are a decodable image, without
        # paying a decode on all 1.33M of them.
        if n_seen[split] % VALIDATE_EVERY == 0:
            try:
                from PIL import Image
                Image.open(io.BytesIO(data)).convert("RGB")
                checked += 1
            except Exception:  # noqa: BLE001
                n_bad += 1

        shard, offset, length = writers[split].write(data)
        rows[split].append((shard, offset, length, 0))
        if is_train:
            wnid_of_index.append(name[len(TRAIN_PREFIX):].split("/", 1)[0])
        else:
            val_ids.append(Path(name).stem)
        n_seen[split] += 1

        now = time.time()
        if now - last_report >= 60:
            done = n_seen["train"] + n_seen["val"]
            written = sum(w.index * shard_bytes + w.offset for w in writers.values())
            print(f"[in-full] {done} images ({n_seen['train']} train, {n_seen['val']} val), "
                  f"{written / 2**30:.1f} GiB written, {done / max(now - started, 1e-9):.0f} img/s, "
                  f"{checked} spot-checked, {n_bad} undecodable", flush=True)
            last_report = now

    for w in writers.values():
        w.close()
    rc = proc.wait()
    if rc != 0:
        raise SystemExit(f"[in-full] curl exited {rc}; archive stream incomplete.")

    _finish(out_dir, rows, writers, wnid_of_index, val_ids, val_csv, n_seen, n_bad,
            checked, limit, shard_bytes, elapsed=time.time() - started)


def _finish(out_dir, rows, writers, wnid_of_index, val_ids, val_csv, n_seen, n_bad,
            checked, limit, shard_bytes, elapsed) -> None:
    wnids = sorted(set(wnid_of_index))
    if not limit and len(wnids) != N_CLASSES:
        raise SystemExit(f"[in-full] saw {len(wnids)} wnids, expected {N_CLASSES}.")
    label_of_wnid = {w: i for i, w in enumerate(wnids)}

    if val_csv is None:
        raise SystemExit(f"[in-full] {VAL_SOLUTION} never appeared in the stream.")
    wnid_of_id = {}
    for row in csv.DictReader(io.StringIO(val_csv.decode("utf-8"))):
        wnid_of_id[row["ImageId"]] = row["PredictionString"].split()[0]
    missing = [i for i in val_ids if i not in wnid_of_id]
    if missing:
        raise SystemExit(f"[in-full] {len(missing)} val images absent from {VAL_SOLUTION}.")

    labels = {"train": [label_of_wnid[w] for w in wnid_of_index],
              "val": [label_of_wnid[wnid_of_id[i]] for i in val_ids]}
    for split in ("train", "val"):
        arr = np.array(rows[split], dtype=INDEX_DTYPE)
        arr["label"] = np.asarray(labels[split], dtype=np.uint16)
        np.save(out_dir / f"{split}_index.npy", arr)

    (out_dir / "wnids.json").write_text(json.dumps(wnids, indent=0))
    payload = sum(p.stat().st_size for p in out_dir.glob("*.bin"))
    meta = {
        "complete": True,
        "format": "indexed record shards, original JPEG bytes verbatim",
        "format_version": FORMAT_VERSION,
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "kaggle imagenet-object-localization-challenge (ILSVRC2012)",
        "pixels_modified": False,
        "shard_bytes": int(shard_bytes),
        "train_shards": writers["train"].shard_count,
        "val_shards": writers["val"].shard_count,
        "label_order": "wnids sorted lexicographically -> 0..N-1 (ImageFolder convention)",
        "num_classes": len(wnids),
        "n_train": n_seen["train"],
        "n_val": n_seen["val"],
        "payload_bytes": int(payload),
        "spot_checked": checked,
        "spot_check_failures": n_bad,
        "limit": limit,
        "build_seconds": round(elapsed),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[in-full] done: {n_seen['train']} train, {n_seen['val']} val, {len(wnids)} classes, "
          f"{payload / 2**30:.1f} GiB in {writers['train'].shard_count}+"
          f"{writers['val'].shard_count} shards, {checked} spot-checked with {n_bad} failures, "
          f"{elapsed / 60:.1f} min", flush=True)


# --------------------------------------------------------------------------------------

def verify(out_dir: Path) -> None:
    """Re-open the built image and check it, as a separate step so the build script
    needs no inline Python and a snapshot is never taken over a bad index."""
    meta = json.loads((out_dir / "meta.json").read_text())
    print("[in-full] meta: " + json.dumps(meta, indent=2), flush=True)
    if not meta.get("complete"):
        raise SystemExit("[in-full] meta.json says the build is not complete.")
    from PIL import Image

    for split in ("train", "val"):
        idx = np.load(out_dir / f"{split}_index.npy")
        shards = sorted(out_dir.glob(f"{split}-*.bin"))
        sizes = [p.stat().st_size for p in shards]
        print(f"[in-full] {split}: {len(idx)} rows, {len(shards)} shards, "
              f"{sum(sizes) / 2**30:.1f} GiB, labels {idx['label'].min()}..{idx['label'].max()}",
              flush=True)
        if len(idx) == 0 or not shards:
            raise SystemExit(f"[in-full] {split} is empty.")
        if int(idx["shard"].max()) >= len(shards):
            raise SystemExit(f"[in-full] {split} index references a missing shard.")
        # Every row must lie inside its shard, and a sample must actually decode.
        ends = idx["offset"].astype(np.uint64) + idx["length"].astype(np.uint64)
        for s in range(len(shards)):
            sel = idx["shard"] == s
            if sel.any() and int(ends[sel].max()) > sizes[s]:
                raise SystemExit(f"[in-full] {split} shard {s} index runs past end of file.")
        rng = np.random.default_rng(0)
        for i in rng.choice(len(idx), size=min(200, len(idx)), replace=False):
            r = idx[int(i)]
            with open(shards[int(r["shard"])], "rb") as f:
                f.seek(int(r["offset"]))
                blob = f.read(int(r["length"]))
            Image.open(io.BytesIO(blob)).convert("RGB")
    print("[in-full] verify ok", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--competition", default="imagenet-object-localization-challenge")
    ap.add_argument("--shard-bytes", type=int, default=SHARD_BYTES)
    ap.add_argument("--limit", type=int, default=None, help="images per split (smoke test).")
    ap.add_argument("--verify", action="store_true", help="check an existing build and exit.")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    out_dir = a.data_dir
    if a.verify:
        verify(out_dir)
        return
    if not a.force and looks_populated(out_dir):
        print(f"[in-full] {out_dir} already complete; skipping.", flush=True)
        return

    token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    if not token:
        raise SystemExit("[in-full] KAGGLE_API_TOKEN not set in the environment.")
    require_tools()

    out_dir.mkdir(parents=True, exist_ok=True)
    # ILSVRC2012 train+val is about 145GB of JPEG; require real headroom on top,
    # because the shards are written while the archive is still streaming.
    need_gb = 165.0
    free_gb = shutil.disk_usage(out_dir).free / 2**30
    print(f"[in-full] need ~{need_gb:.0f}GiB of payload, free {free_gb:.0f}GiB on {out_dir}",
          flush=True)
    if not a.limit and free_gb < need_gb:
        raise SystemExit(
            f"[in-full] insufficient disk: {free_gb:.0f}GiB free, need ~{need_gb:.0f}GiB. "
            "Raise imagenet_full.primary_disk_gb in configs/thunder.yaml."
        )

    stream_build(a.competition, token, out_dir, a.limit, a.shard_bytes)


if __name__ == "__main__":
    sys.exit(main())
