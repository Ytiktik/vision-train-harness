"""Optimizer builder for the conv-only weight-decay cells: SGD whose weight
decay is applied ONLY to weight matrices/kernels (ndim >= 2: conv kernels, fc
weight) and NOT to normalization affines or biases (ndim < 2: BN gamma/beta,
fc bias).

Purpose (2026-08-12 arc): the tied pair's persistent WD-regime advantage was
traced to the output-scale penalty structure (pair pays half the gamma penalty
at matched function) and the single's first-block dead channels were traced to
WD pruning gamma to zero. Removing decay from the scale channel entirely, while
keeping kernel regularization, switches that whole channel off — if the
dead-channel population and/or the pair's edge shrink, they were gamma-decay
sourced.
"""

from __future__ import annotations

import torch
from torch import nn


def build_conv_wd_sgd(
    params,
    model: nn.Module,
    lr: float,
    momentum: float = 0.0,
    weight_decay: float = 0.0,
    nesterov: bool = False,
    bn_gamma_lr_mult: float = 1.0,
    conv_lr_mult: float = 1.0,
    **_: object,
) -> torch.optim.Optimizer:
    """SGD with weight decay restricted to ndim>=2 parameters (conv/fc
    weights); norms' affine parameters and all biases get weight_decay=0.
    Optional lr multipliers: ``conv_lr_mult`` on the ndim>=2 (decayed) group,
    ``bn_gamma_lr_mult`` on 1-D norm scale parameters (names ending in
    ``.weight`` or ``wn_gamma``) — optimizer-side stand-ins for the pair's
    direction- and scale-lr multipliers."""
    decay, no_decay, gam = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2:
            decay.append(p)
        elif bn_gamma_lr_mult != 1.0 and (name.endswith(".weight") or name.endswith("wn_gamma")):
            gam.append(p)
        else:
            no_decay.append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay, "lr": lr * conv_lr_mult},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    if gam:
        groups.append({"params": gam, "weight_decay": 0.0, "lr": lr * bn_gamma_lr_mult})
    return torch.optim.SGD(groups, lr=lr, momentum=momentum, nesterov=nesterov)
