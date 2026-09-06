"""Single-branch MobileOne that *mimics* a k-branch block via init / LR tricks.

A train-time MobileOneBlock with ``num_conv_branches=k`` reparameterizes into a
single conv whose kernel is the SUM of the k branch kernels and whose bias is the
SUM of the k branch biases. Two consequences of that summation motivate this
experiment:

  * **Init.** k freshly Kaiming-initialized branches fuse to a kernel that is the
    sum of k i.i.d. Kaiming draws (variance k x a single branch). So a single
    branch initialized as ``sum of k Kaiming kernels`` matches the k-branch
    fused-kernel init.
  * **Bias dynamics.** The k branch BN biases all receive the same gradient
    (they are summed), so the fused bias B = sum b_i evolves as
    dB/dt = -eta * k * dL/dB under gradient flow — i.e. as a single bias with a
    k x learning rate. (See ``BiasBoostedSGD`` in ``structural_reparam.optim``.)
  * **Weight dynamics (ρ≈1).** When the branches are highly correlated (ρ→1, the
    only regime in which w₋ is negligible), the fused weight W = sum w_i evolves
    as W <- W - eta·(eγ/σ₊)·k²·( x − (1/σ₊²)(W·x)CₓW ) — i.e. a *plain* single-
    branch BN update (leading:curvature = 1:1) at a **k² x learning rate**. (k=2 ⇒
    4×.) This is a uniform LR boost, so ``BiasBoostedSGD``'s ``weight_lr_mult``
    handles it; no custom BN layer is needed.

Together they try to reproduce a k-branch block's behavior with a single branch.

This factory builds a vanilla :class:`MobileOne` (so it stays a faithful
reference) and only rewrites the 3x3 conv-branch kernels to the Kaiming-sum init
when ``kaiming_branch_sum_k > 1``. The scope — which branches count as "the 3x3
branch" — is shared with the optimizer via :func:`iter_target_conv_branches`.
"""

from __future__ import annotations

import math
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.init as init

from structural_reparam.models.mobileone import MobileOne, MobileOneBlock


def iter_target_conv_branches(model: nn.Module) -> Iterator[nn.Sequential]:
    """Yield the ``rbr_conv`` branches of every reparameterizable MobileOneBlock.

    These are exactly the branches that get summed under k-fold reparameterization
    — the ``num_conv_branches`` parallel convs of *both* the per-stage depthwise
    blocks (``kernel_size == 3``, ``groups > 1``) *and* the per-stage pointwise
    blocks (``kernel_size == 1``, ``groups == 1``). Apple's MobileOne gives every
    stage1-4 block ``num_conv_branches`` branches (see ``_make_stage``), so the
    init/LR mimic must cover both kinds, not just the 3x3 depthwise.

    Excluded:
      * the stem (``stage0``, ``in_channels == 3``), which is hardcoded to a
        single conv branch in the reference and so never k-folds;
      * ``rbr_scale`` (the 1x1 scale branch) and ``rbr_skip`` (the BN identity),
        which are always single branches, not part of the ``num_conv_branches``
        sum.

    Each yielded module is a ``Sequential(conv, bn)`` so callers can reach
    ``branch.conv`` / ``branch.bn`` (or index ``branch[0]`` / ``branch[1]``).
    """
    for module in model.modules():
        if not isinstance(module, MobileOneBlock):
            continue
        if getattr(module, "inference_mode", False):
            continue  # branches already fused away
        if module.in_channels == 3:
            continue  # stem: always single-branch in the reference, never k-folded
        for branch in module.rbr_conv:
            yield branch


def _reinit_kaiming_sum(conv: nn.Conv2d, k: int) -> None:
    """Set conv.weight to the sum of ``k`` independent Conv2d-default draws.

    Conv2d.reset_parameters initializes the weight with kaiming_uniform_(a=sqrt(5));
    summing k such draws reproduces the kernel that k freshly-initialized branches
    fuse to (variance k x a single branch).
    """
    with torch.no_grad():
        acc = torch.zeros_like(conv.weight)
        tmp = torch.empty_like(conv.weight)
        for _ in range(k):
            init.kaiming_uniform_(tmp, a=math.sqrt(5))
            acc += tmp
        conv.weight.copy_(acc)


def mobileone_branch_mimic(
    *, kaiming_branch_sum_k: int = 1, **mobileone_kwargs
) -> MobileOne:
    """Build a MobileOne, optionally with the Kaiming-sum 3x3-branch init.

    ``kaiming_branch_sum_k`` is the number of Kaiming draws summed into each 3x3
    depthwise conv-branch kernel (k=1 leaves the vanilla Conv2d init untouched, so
    the model is byte-for-byte a stock MobileOne). The other two branch-mimic
    factors are LR multipliers applied by ``BiasBoostedSGD`` (k× on the BN biases,
    k²× on the conv weights — see that optimizer), not model changes. All other
    kwargs pass straight through to :class:`MobileOne`.
    """
    if kaiming_branch_sum_k < 1:
        raise ValueError(
            f"kaiming_branch_sum_k must be >= 1, got {kaiming_branch_sum_k}"
        )
    model = MobileOne(**mobileone_kwargs)
    if kaiming_branch_sum_k > 1:
        for branch in iter_target_conv_branches(model):
            _reinit_kaiming_sum(branch.conv, kaiming_branch_sum_k)
    return model
