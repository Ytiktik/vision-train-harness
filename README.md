# vision-train-harness

Training harness for convolutional vision experiments on ImageNet-scale data.
A snapshot, pulled onto training machines rather than developed in; it carries no
credentials and no history, and is replaced wholesale when the code changes.

## Requirements

Python 3.11+, and `torch torchvision numpy pyyaml wandb pillow`. Nothing else —
in particular neither `webdataset` nor `timm` is used.

The GPU needs a torch build that knows it. Check before anything else:

    python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_capability(0))"

A Blackwell card reports capability `(12, 0)` and needs a cu128 wheel or newer.
The CUDA *toolkit* version on the machine is irrelevant — pip torch ships its own
runtime; only the driver matters.

## Setup, once

    python -m venv .venv && . .venv/bin/activate
    pip install torch torchvision numpy pyyaml wandb pillow
    wandb login                        # writes ~/.netrc, outside this repo
    ln -s /where/the/data/is data/imagenet_full

The dataset directory must hold record shards and their index: `train-*.bin`,
`val-*.bin`, `train_index.npy`, `val_index.npy`, `meta.json`. The reader is
`src/structural_reparam/data/imagenet_records.py`; it gives O(1) access by example
id, so the sampler stays a true global permutation over the whole training set.

## Running

One config path per line in a queue file, then:

    PYTHONPATH=src python scripts/run_queue.py --queue queue.txt --check
    tmux new -s runs 'PYTHONPATH=src python scripts/run_queue.py --queue queue.txt'

`--check` verifies the GPU, the dataset, the credentials and every config in
seconds and refuses to start if any is wrong. The runner then works through the
queue one config at a time and retries a failure up to three times. Retries are
cheap: each run checkpoints every epoch — model, optimizer, scheduler, epoch and
generator state — and uploads it, so an interruption costs one epoch rather than a
run, and the run re-attaches to its own record so its history stays continuous.

Configs are in `src/structural_reparam/agents/stage2/`, one per arm.

## Verifying the install

    PYTHONPATH=src python -m pytest tests -q

Fifty-five CPU tests in about twenty seconds, including an interrupted run
resuming to bit-identical parameters.
