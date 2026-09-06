# Runbook for the training machine

You are running experiments on a workstation that nobody can reach from outside.
Code arrives here by `git pull` from https://github.com/Ytiktik/vision-train-harness
and results leave by Weights and Biases. Nothing else is a channel. This file is
the whole brief; it is also published as `CLAUDE.md` in that repo.

## What this is

Each run trains one *arm*. An arm is a network in which chosen 3×3 convolutions
carry two identical branches, each with its own normalization and a shared scale,
which fold back into a single convolution at inference. The comparison is always
an arm against the `single` — the same network with one branch everywhere — at
the **same seed**. That paired difference is the result; an arm without its own
single is not half a result, it is nothing.

Two networks, both at the canonical ImageNet widths 64/128/256/512 with a 3×3
stride-2 stem and 1000 classes:

- **ResNet-10**, one BasicBlock per stage, nine 3×3 convolutions, 5.41M parameters
- **ResNet-18**, two per stage, seventeen convolutions, 11.69M parameters

Four arms each: `single`, `pair_first` (the stem alone), `pair_last` (the final
3×3 convolution alone), `pair_both`.

## The cell — never change any of it

224 pixels, 40-epoch cosine from 0.1 at batch 256, SGD momentum 0.9, kernel-only
weight decay 1e-4, random resized crop and horizontal flip for training, resize
256 and centre crop 224 for validation, full float32. No mixed precision, no
`torch.compile`, no channels-last, no nvJPEG or PIL draft decoding — the last two
change decoded pixel values, and this project's claim is that the pixels are
ILSVRC2012's own.

Four runs already exist on other hardware under exactly this cell. Every number
you produce is compared against them, so a change here does not produce a
different measurement, it produces a worthless one. If something seems to require
a change, stop and report rather than adapting.

## Setup, once

```
python -m venv .venv && . .venv/bin/activate      # or use the existing ml-env
pip install torch torchvision numpy pyyaml wandb pillow
wandb login <key from the operator>
```

Do **not** install `webdataset` or `timm`; nothing uses them. Check the GPU knows
itself before anything else — a Blackwell card must report capability `(12, 0)`
and needs a cu128 wheel or newer:

```
python -c "import torch;print(torch.__version__,torch.version.cuda,torch.cuda.get_device_capability(0))"
```

## The data, and the one way it can silently ruin everything

The loader reads **indexed record shards**, not WebDataset:

```
train-00000.bin ...              raw concatenated JPEG bytes, ~1 GiB each
train_index.npy / val_index.npy  one row per image:
    shard uint16, offset uint64, length uint32, label uint16
wnids.json, meta.json
```

`scripts/prefetch_imagenet_full.py` is the reference implementation of that format
and of the label convention. Symlink whatever you build to `data/imagenet_full`.

**Labels are the silent failure.** The reader takes `index["label"]` as final: no
wnid mapping, no validation. Our existing runs used **wnids sorted
lexicographically to 0..999** (torchvision's ImageFolder order, so 0 is
`n01440764` tench, 999 `n15075141` toilet tissue). If a conversion assigns a
different order, every run trains, converges and reports a plausible accuracy
against permuted labels, and **no accuracy check can catch it**, because the
validation set is permuted identically. So: emit `wnids.json`, assert 1000 classes
in sorted order, assert against the source's own class list if it ships one, and
decode a dozen images and eyeball their classes before committing 150 GiB.

Write `meta.json` last, with `complete: true`, so an interrupted build cannot be
mistaken for a finished one.

## Running

```
PYTHONPATH=src python scripts/run_queue.py --queue queues/wave1_replication.txt --check
tmux new -s runs 'PYTHONPATH=src python scripts/run_queue.py --queue queues/wave1_replication.txt'
```

`--check` verifies the GPU, the dataset, the credentials and every config in
seconds and refuses to start if any is wrong; run it first, every time. The runner
then works through the queue one config at a time and retries a failure up to
three times. Retries are cheap: each run checkpoints every epoch — model,
optimizer, scheduler, epoch and generator state — uploads it to W&B, and
re-attaches to its own run, so a kill, a reboot or a crash costs one epoch. Never
delete `outputs/`; that is where the local resume state lives.

**Order and dependencies.** `queues/wave1_replication.txt` first: eight runs,
seeds 43 and 44 of `single` and `pair_both`. Then
`queues/wave2_decomposition.txt`, whose **first four runs (seed 42) can start
immediately** because their singles already exist elsewhere — the first two alone
give the ResNet-10 split. The seed-43 and seed-44 entries in that file depend on
wave 1's singles; do not start them before wave 1 finishes.

## What to report back, and when

- The full `--check` output, once. It carries the GPU, core count, torch and CUDA
  versions, shard count and commit, and it is how the epoch time gets predicted.
- The measured epoch rate after three or four epochs, as the **gap between epoch
  timestamps**, not runtime divided by epoch count — the first epoch carries
  startup and the difference is 30 percent.
- Any failure, with the last 50 lines of `outputs/<run>/run.log`.
- Nothing per epoch. One message per finished run or per queue is right.

## Sanity numbers from the runs that already exist

Seed 42, this exact cell, on an A100. Yours should land near these; a large
departure means the setup differs, not that the science changed.

| | ResNet-10 | ResNet-18 |
| --- | --- | --- |
| `single` train / val top-1 | 60.98 / 63.43 | 68.52 / 69.47 |
| `pair_both` train gain | +0.80 | +0.49 |
| stem: open of 64, angle, alignment | 53 (83 %), 159°, 1.000 | 51 (80 %), 159°, 1.000 |
| last conv kernel cosine | 1.00000 | 1.00000 |
| stem μ_eff | 0.761 | 0.761 |

ResNet-18's single reaching about 69.5 is the strongest single check that the
whole pipeline is right: it is within three tenths of torchvision's published
69.76 for the same architecture.

## Things not to do

Do not edit the configs, the recipe or the model code. Do not add arms, seeds or
architectures on your own initiative — the campaign's order is decided in
`directives/heavy_runs.md` in the private repo and the arms are gated on each
other. Do not re-run a config that finished. Do not train anything to "check the
setup"; `--check` and `pytest tests -q` are the checks, both seconds long. If the
data conversion, the environment or a result looks wrong, stop and report; a
wrong run costs six hours and a wrong *cell* costs the comparison.
