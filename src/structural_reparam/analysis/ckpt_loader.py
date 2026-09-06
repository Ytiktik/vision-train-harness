"""Checkpoint fetch + architecture-inferring model builder for the ckpt100 cells.

Extracted (2026-08-19) from the archived agent
``agents/reparam_noise_floor_metric/exp_hetero_broad.py`` (``get_ckpt`` /
``build_generic``) so the preconditioning replays keep a base-package loader
that works for any depth/width cell without hard-coding the architecture.

Checkpoints are W&B artifacts named ``ckpt_{group}_{variant}_s{seed}`` in the
``claude-autonomous-reparam`` project; they are cached under ``CKPT_ROOT``
(default ``/tmp/nf_ckpts``) as ``{group}_{variant}_s{seed}/ckpt_ep{epoch}.pt``.

Imports of ``wandb`` and of the backbone model are deliberately lazy: this module
must not create an import cycle with ``reparam_sweeps100.lab`` (which imports
``structural_reparam.analysis.registry``), and ``wandb`` is an optional dev
dependency.
"""
from __future__ import annotations

import collections
import os
import re
from typing import Any

import torch

DEFAULT_ENTITY = "yoovi-t-tel-aviv-university"
DEFAULT_PROJECT = "claude-autonomous-reparam"


def fetch_checkpoint(
    group: str,
    variant: str,
    epoch: int,
    seed: int = 42,
    *,
    root: str | None = None,
    entity: str | None = DEFAULT_ENTITY,
    project: str = DEFAULT_PROJECT,
) -> dict[str, Any]:
    """Return the checkpoint dict for ``(group, variant, seed)`` at ``epoch``.

    Downloads the W&B artifact ``ckpt_{group}_{variant}_s{seed}:v0`` into the
    cache directory on first use. ``root`` defaults to ``$CKPT_ROOT`` or
    ``/tmp/nf_ckpts``. Pass ``entity=None`` to rely on the default W&B entity.
    """
    root = root or os.environ.get("CKPT_ROOT", "/tmp/nf_ckpts")
    name = f"ckpt_{group}_{variant}_s{seed}"
    cache_dir = os.path.join(root, f"{group}_{variant}_s{seed}")
    path = os.path.join(cache_dir, f"ckpt_ep{epoch}.pt")
    if not os.path.exists(path):
        import wandb  # lazy: optional dependency

        ref = f"{entity}/{project}/{name}:v0" if entity else f"{project}/{name}:v0"
        wandb.Api().artifact(ref, type="checkpoint").download(root=cache_dir)
    return torch.load(path, map_location="cpu", weights_only=False)


# Backwards-compatible alias for the archived agents' name.
get_ckpt = fetch_checkpoint


def build_generic(sd: dict[str, torch.Tensor]) -> torch.nn.Module:
    """Build a per-branch-BN ``LayerwiseRepVGGCifar`` matching a state dict.

    Infers ``stage_blocks``, ``stage_channels``, ``num_classes`` and ``num_3x3``
    from the parameter names and shapes, loads the weights with
    ``strict=False``, and returns the model in eval mode.
    """
    from structural_reparam.experiments.reparam_sweeps100.lab import (  # lazy: avoid cycle
        LayerwiseRepVGGCifar,
    )

    blocks: dict[int, set[int]] = collections.defaultdict(set)
    channels: dict[int, int] = {}
    for key in sd:
        m = re.match(r"stages\.(\d+)\.(\d+)\.", key)
        if m:
            blocks[int(m.group(1))].add(int(m.group(2)))
    for s in sorted(blocks):
        channels[s] = sd[f"stages.{s}.0.conv3_branches.0.0.weight"].shape[0]
    stage_blocks = [len(blocks[s]) for s in sorted(blocks)]
    stage_channels = [channels[s] for s in sorted(blocks)]
    num_classes = sd["fc.weight"].shape[0]
    branch_re = re.compile(r"stages\.0\.0\.conv3_branches\.(\d+)\.")
    num_3x3 = len({int(branch_re.match(k).group(1)) for k in sd if branch_re.match(k)})
    model = LayerwiseRepVGGCifar(
        num_classes=num_classes,
        width_mult=1.0,
        stage_channels=stage_channels,
        stage_blocks=stage_blocks,
        stage_strides=[1, 2, 2],
        num_3x3=num_3x3,
        norm="batch",
        use_1x1=False,
        use_identity=False,
        bn_position="per_branch",
    )
    model.load_state_dict(sd, strict=False)
    return model.eval()
