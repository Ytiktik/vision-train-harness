"""LayerNorm branches for the placed shared-scale model (added 2026-09-03).

The paper's mechanism (branch separation parking on the input's loud direction,
Section 3 of the framework paper) is derived for per-branch BatchNorm, whose sigma
is a population statistic of the branch output. This module swaps every branch's
BatchNorm for a per-branch LayerNorm so the same placed model, recipe and probes
can be run with a per-token normalizer, to test whether the branch difference
still collapses onto the loud input direction (the rank-one prediction of the
2026-09-03 to-do entry in the paper).

Two LayerNorm scopes are offered:

* ``scope="position"``: the transformer LayerNorm. Each spatial position is one
  token and the branch output is standardized over its channels at that position
  (mean and variance over the channel axis, per sample and per position).
* ``scope="image"``: GroupNorm with one group, the June 2026 ``reparam_norm_types``
  arm. The branch output is standardized per sample over channels and positions.

Both are affine-free: the block's shared per-channel gamma and per-branch betas are
unchanged. ``center=False`` gives the RMSNorm variant (no mean subtraction).

The module also tracks, as buffers, the BatchNorm-style per-channel running mean
and variance of its input, and stashes the per-channel batch variance on
``last_var`` exactly as ``BranchStats2d`` does. These statistics take no part in
the forward pass; they exist so the training-time pair-channel probe keeps working
unchanged and so the offline analysis can whiten with the sigma a BatchNorm would
have had. ``running_token_var`` is the running mean of the LayerNorm's own
per-token variance, the per-branch scalar the LayerNorm reading of the theory
normalizes by.

Note the gauge: with ``scope="position"`` and ``center=True`` the per-position
mean over channels is subtracted, so adding one common input direction to every
output channel of a branch (a rank-one matrix whose rows are all equal) changes
nothing about the block's output. That component of a branch kernel is invisible
to the loss and is shaped only by weight decay, and the offline analysis removes
it before testing the rank of the branch difference.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from structural_reparam.experiments.reparam_pinned_norm.lab import PlacedSharedScaleRepVGGCifar
from structural_reparam.experiments.reparam_shared_scale.lab import SharedScaleRepVGGBlock


class LayerNormStats2d(nn.Module):
    """Affine-free per-branch LayerNorm with BatchNorm-style bookkeeping.

    Forward: standardize the input per token (see the module docstring for the
    two scopes), in train and eval alike. Side effects in train mode: update the
    running per-channel mean and variance of the input (BatchNorm's statistics,
    for measurement only), update the running per-token variance, and stash the
    per-channel batch mean and variance on ``last_mu`` and ``last_var`` for the
    pair-channel probe.
    """

    def __init__(self, channels: int, eps: float = 1e-5, momentum: float = 0.1,
                 scope: str = "position", center: bool = True) -> None:
        super().__init__()
        if scope not in ("position", "image"):
            raise ValueError(f"scope must be 'position' or 'image', got {scope!r}")
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.scope = scope
        self.center = bool(center)
        self.register_buffer("running_mean", torch.zeros(channels))
        self.register_buffer("running_var", torch.ones(channels))
        self.register_buffer("running_token_var", torch.ones(()))
        self.last_mu: torch.Tensor | None = None
        self.last_var: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dims = (1,) if self.scope == "position" else (1, 2, 3)
        if self.center:
            mu = x.mean(dim=dims, keepdim=True)
            xc = x - mu
        else:
            xc = x
        var = xc.pow(2).mean(dim=dims, keepdim=True)
        if self.training:
            with torch.no_grad():
                m = self.momentum
                cmu = x.mean(dim=(0, 2, 3))
                cvar = x.var(dim=(0, 2, 3), unbiased=False)
                self.running_mean.mul_(1.0 - m).add_(m * cmu)
                self.running_var.mul_(1.0 - m).add_(m * cvar)
                self.running_token_var.mul_(1.0 - m).add_(m * var.mean())
                self.last_mu = cmu
                self.last_var = cvar
        return xc / torch.sqrt(var + self.eps)


class PlacedLayerNormPairRepVGGCifar(PlacedSharedScaleRepVGGCifar):
    """``PlacedSharedScaleRepVGGCifar`` with per-branch LayerNorm in the listed
    blocks (``ln_blocks``, network order; ``None`` means every block).

    Everything else is inherited: the shared per-channel gamma, the per-branch
    betas, the kernel init, ``block_branches`` placement, and the init knobs. The
    blocks are still ``SharedScaleRepVGGBlock`` instances, so the pair-channel
    probe, the checkpoint probe and the kernel-only decay builder see the same
    parameter names as the BatchNorm cell (``stages.<s>.<b>.convs.<j>.weight``,
    ``.gamma``, ``.betas.<j>``); only the ``stats`` submodules differ.
    """

    def __init__(self, *args, ln_blocks: Sequence[int] | None = None, ln_scope: str = "position",
                 ln_center: bool = True, ln_eps: float = 1e-5, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        blocks = [m for m in self.modules() if isinstance(m, SharedScaleRepVGGBlock)]
        chosen = set(range(len(blocks))) if ln_blocks is None else {int(b) for b in ln_blocks}
        for k, blk in enumerate(blocks):
            if k not in chosen:
                continue
            if blk.mode != "shared_gamma":
                raise ValueError("LayerNorm branches need mode='shared_gamma'")
            blk.stats = nn.ModuleList([
                LayerNormStats2d(blk.out_channels, eps=ln_eps, scope=ln_scope, center=ln_center)
                for _ in range(blk.num_branches)
            ])
        self.ln_blocks = sorted(chosen)
        self.ln_scope, self.ln_center = ln_scope, bool(ln_center)
