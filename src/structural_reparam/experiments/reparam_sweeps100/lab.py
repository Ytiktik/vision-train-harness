"""Editable autonomous experiment sandbox for structural reparameterization.

Claude may edit this file and ``configs/claude_autonomous.yaml`` during bounded
autonomous experiments. Keep new experimental models, layers, and helper
builders in this file. Do not edit the shared dataset loaders, submitters, or
generic trainer to test a local idea.

The training entrypoint is still the generic trainer: ``main()`` delegates to
``structural_reparam.deploy.train.main`` so configs can set
``job.entry_module: structural_reparam.claude_lab`` and darxio can run this file
through the existing submitter.
"""

from __future__ import annotations

from collections.abc import Sequence
import math
from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F


NormName = Literal[
    "batch",
    "batch_fixedmean",
    "batch_fixedvar",
    "batch_sgmean",
    "batch_sgvar",
    "batch_sgboth",
    "mean_only",
    "layer",
    "none",
]
ScaleInit = Literal["inv_sqrt_n", "inv_n", "ones"] | float
# A numeric branch_scale_init sets every branch's norm scale (or conv scale for
# post_sum) to that literal value — used for gamma-ladder arms (e.g. 1/(2*sqrt(2))).
#
# WHAT THE SCALE ACTUALLY DOES — read this before choosing one.
# The scale lands on a DIFFERENT object depending on ``bn_position``, so the same
# keyword gives a different init function in the two cases:
#
#   bn_position="post_sum"   — no per-branch norm, so the scale multiplies the CONV
#       weights and survives.  Branch outputs are independent, so their variances
#       add: the summed pre-norm activation matches the single-branch case when
#       scale = n**-0.5.  Here "inv_sqrt_n" is exactly the variance-preserving choice.
#
#   bn_position="per_branch" — each branch is BN'd on its own, and BN is INVARIANT
#       to the conv scale.  The conv factor is erased; only the BN affine gamma
#       (= scale) survives.  The n branches then sum to a fused kernel of gain
#           init_fused_gain = n_branches * scale
#       so the choices give:
#           "inv_sqrt_n" -> sqrt(n)   (n=2: 1.414)  NOT function-preserving
#           "ones"       -> n         (n=2: 2.0)    NOT function-preserving
#           "inv_n"      -> 1.0                     function-preserving vs single
#       A two-branch arm built with "inv_sqrt_n" therefore starts with a sqrt(2)
#       LARGER effective kernel than its single-branch reference, i.e. it is not a
#       function-preserving split of it.  Use "inv_n" when you want the fused
#       function (and hence the effective LR) matched to the single-branch arm.
#
# Use ``branch_gain_at_init()`` to get the number rather than re-deriving it.


def branch_gain_at_init(
    branch_scale_init: "ScaleInit", branch_count: int, bn_position: "BNPosition"
) -> float:
    """Init gain of the FUSED kernel relative to a single-branch reference.

    Returns 1.0 exactly when the construction is function-preserving.  See the
    ``ScaleInit`` comment above for why ``per_branch`` and ``post_sum`` differ.
    """
    scale = branch_scale_init_value(branch_scale_init, branch_count)
    if bn_position == "per_branch":
        # BN kills the conv scale; the n per-branch gammas add.
        return branch_count * scale
    # post_sum: independent branches, variances add.
    return (branch_count ** 0.5) * scale


def branch_scale_init_value(branch_scale_init: "ScaleInit", branch_count: int) -> float:
    """Resolve a ``branch_scale_init`` keyword to the literal scale it applies."""
    if isinstance(branch_scale_init, (int, float)):
        return float(branch_scale_init)
    if branch_scale_init == "inv_sqrt_n":
        return branch_count ** -0.5
    if branch_scale_init == "inv_n":
        return 1.0 / branch_count
    if branch_scale_init == "ones":
        return 1.0
    raise ValueError(
        "branch_scale_init must be 'inv_sqrt_n', 'inv_n', 'ones', or a number, "
        f"got {branch_scale_init!r}"
    )
BNPosition = Literal["per_branch", "post_sum", "weight_norm"]
StatMode = Literal["batch", "batch_nograd", "running"]


class PartialBatchNorm2d(nn.Module):
    """BatchNorm2d with per-statistic control over how the mean and variance
    are obtained, used to dissect the per-branch-BN two-branch advantage.

    ``mean_mode`` / ``var_mode`` each take:

    - ``"batch"``: the batch's own statistic, gradient flowing through it -- the
      usual BN behaviour, which couples examples within a batch and differs per
      parallel branch.
    - ``"batch_nograd"``: the batch's own statistic *value* but with the gradient
      stopped (``.detach()``). The forward pass is identical to standard BN
      (correct centering/scaling every step, training stays healthy); only the
      gradient coupling through that statistic is removed. This is the clean
      ablation of "batch coupling via gradients".
    - ``"running"``: the detached running EMA estimate -- removes the statistic's
      value-coupling to the current batch entirely (harsher; can mis-centre conv
      activations early in training).

    The running estimates are always updated in train mode and always used in
    eval mode, so eval matches standard BN. With both modes ``"batch"`` this is
    exactly ``nn.BatchNorm2d``.
    """

    def __init__(
        self,
        channels: int,
        mean_mode: StatMode = "batch",
        var_mode: StatMode = "batch",
        eps: float = 1e-5,
        momentum: float = 0.1,
    ) -> None:
        super().__init__()
        if mean_mode not in ("batch", "batch_nograd", "running"):
            raise ValueError(f"bad mean_mode {mean_mode!r}")
        if var_mode not in ("batch", "batch_nograd", "running"):
            raise ValueError(f"bad var_mode {var_mode!r}")
        self.mean_mode = mean_mode
        self.var_mode = var_mode
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.register_buffer("running_mean", torch.zeros(channels))
        self.register_buffer("running_var", torch.ones(channels))

    @staticmethod
    def _select(mode: StatMode, batch_stat: torch.Tensor, running_stat: torch.Tensor) -> torch.Tensor:
        if mode == "batch":
            return batch_stat
        if mode == "batch_nograd":
            return batch_stat.detach()
        return running_stat.detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            batch_mean = x.mean(dim=(0, 2, 3))
            batch_var = x.var(dim=(0, 2, 3), unbiased=False)
            with torch.no_grad():
                m = self.momentum
                self.running_mean.mul_(1.0 - m).add_(m * batch_mean)
                self.running_var.mul_(1.0 - m).add_(m * batch_var)
            mean = self._select(self.mean_mode, batch_mean, self.running_mean)
            var = self._select(self.var_mode, batch_var, self.running_var)
        else:
            mean = self.running_mean
            var = self.running_var
        xhat = (x - mean[None, :, None, None]) / torch.sqrt(var[None, :, None, None] + self.eps)
        return xhat * self.weight[None, :, None, None] + self.bias[None, :, None, None]


class MeanOnlyNorm2d(nn.Module):
    """Foldable **mean-only** norm: center with the (running) mean, NO variance
    divide, with learnable per-channel affine ``gamma, beta``.

        train:  out = gamma * (x - batch_mean) + beta     (running_mean updated)
        eval:   out = gamma * (x - running_mean) + beta

    This is the sigma-removal lever of the campaign: it handles the first-moment
    (centering) part of BN but leaves per-channel **magnitude** un-normalized, so
    the per-channel heterogeneity ``R`` the layer's input carries SURVIVES into
    the output (unlike BN, whose sigma-divide erases it). Folds to a per-channel
    diagonal affine at eval: ``A = diag(gamma)``, ``b = beta - gamma * mu``. The
    ``running_var`` buffer is kept (unused by the math) so BN-style tooling that
    reads it does not choke.
    """

    def __init__(self, channels: int, momentum: float = 0.1) -> None:
        super().__init__()
        self.eps = 0.0  # no variance term; kept so the BN-folder reads it safely
        self.momentum = float(momentum)
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.register_buffer("running_mean", torch.zeros(channels))
        self.register_buffer("running_var", torch.ones(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            batch_mean = x.mean(dim=(0, 2, 3))
            with torch.no_grad():
                m = self.momentum
                self.running_mean.mul_(1.0 - m).add_(m * batch_mean)
            mean = batch_mean
        else:
            mean = self.running_mean
        xhat = x - mean[None, :, None, None]
        return xhat * self.weight[None, :, None, None] + self.bias[None, :, None, None]


def _make_conv_norm(channels: int, norm: NormName) -> nn.Module:
    if norm == "batch":
        return nn.BatchNorm2d(channels)
    if norm == "mean_only":
        return MeanOnlyNorm2d(channels)
    if norm == "batch_fixedmean":
        return PartialBatchNorm2d(channels, mean_mode="running", var_mode="batch")
    if norm == "batch_fixedvar":
        return PartialBatchNorm2d(channels, mean_mode="batch", var_mode="running")
    if norm == "batch_sgmean":
        return PartialBatchNorm2d(channels, mean_mode="batch_nograd", var_mode="batch")
    if norm == "batch_sgvar":
        return PartialBatchNorm2d(channels, mean_mode="batch", var_mode="batch_nograd")
    if norm == "batch_sgboth":
        return PartialBatchNorm2d(channels, mean_mode="batch_nograd", var_mode="batch_nograd")
    if norm == "layer":
        return nn.GroupNorm(1, channels)
    if norm == "none":
        return nn.Identity()
    raise ValueError(f"Unsupported conv norm: {norm!r}")


def _make_mlp_norm(features: int, norm: NormName) -> nn.Module:
    if norm == "batch":
        return nn.BatchNorm1d(features)
    if norm == "layer":
        return nn.LayerNorm(features)
    if norm == "none":
        return nn.Identity()
    raise ValueError(f"Unsupported MLP norm: {norm!r}")


def _set_scale(module: nn.Module, value: float) -> None:
    weight = getattr(module, "weight", None)
    if isinstance(weight, nn.Parameter):
        with torch.no_grad():
            weight.fill_(value)


class ClaudeRepVGGBlock(nn.Module):
    """Small editable RepVGG-style block.

    This is intentionally simpler than the production RepVGG block. It gives
    Claude a readable starting point for structural-reparameterization ideas:
    several linear conv branches are summed during training and could be folded
    into one conv at inference if their norms are frozen or converted to affine
    constants.

    ``bn_position`` controls where normalization lives relative to the branch
    sum:

    - ``per_branch`` (default, classic RepVGG): each branch is ``conv -> norm``
      and the normed outputs are summed. The fold is only exact once each
      branch's BN stats are frozen/converted to affine constants.
    - ``post_sum``: every 3x3 branch is a bare bias-free conv, the convs are
      summed, and a *single* shared norm is applied to the sum:
      ``norm(sum_i conv_i(x))``. Here the fused inference weight is literally
      ``W = sum_i W_i``, so per-branch L2 decay (``sum_i ||W_i||^2``) and decay
      on the fused weight (``||sum_i W_i||^2``) can be compared cleanly. This is
      the mode used by the weight-decay reparameterization experiment.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        num_3x3: int = 1,
        use_1x1: bool = False,
        use_identity: bool = False,
        identity_use_bn: bool = True,
        norm: NormName = "batch",
        branch_scale_init: ScaleInit = "inv_sqrt_n",
        bn_position: BNPosition = "per_branch",
        branch_init_identical: bool = False,
        wn_norm_floor: float = 0.0,
        wn_gamma_mult: float = 1.0,
        wn_fix_norm: bool = False,
        branch_init_angle_deg: float | None = None,
        wn_gamma_match_angle: bool = False,
        wn_kernel_scale: float = 1.0,
        wn_kernel_scale_end: float | None = None,
        wn_kernel_scale_anneal_epochs: float = 0.0,
        wn_steps_per_epoch: int = 391,
        wn_gamma_lr_mult: float = 1.0,
        wn_gamma_lr_mult_end: float | None = None,
        kernel_init_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if num_3x3 < 1:
            raise ValueError(f"num_3x3 must be >= 1, got {num_3x3}")
        if not isinstance(branch_scale_init, (int, float)) and branch_scale_init not in (
            "inv_sqrt_n",
            "inv_n",
            "ones",
        ):
            raise ValueError(
                "branch_scale_init must be 'inv_sqrt_n', 'inv_n', 'ones', or a number, "
                f"got {branch_scale_init!r}"
            )
        if bn_position not in ("per_branch", "post_sum", "weight_norm"):
            raise ValueError(
                "bn_position must be 'per_branch', 'post_sum' or 'weight_norm', "
                f"got {bn_position!r}"
            )
        if bn_position == "weight_norm" and (use_1x1 or use_identity):
            raise ValueError("bn_position='weight_norm' supports 3x3 branches only")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.num_3x3 = num_3x3
        self.use_1x1 = bool(use_1x1)
        self.has_identity = bool(use_identity and stride == 1 and in_channels == out_channels)
        self.bn_position = bn_position
        self.branch_init_identical = bool(branch_init_identical)
        # weight_norm only: w_hat = w / max(||w||, wn_norm_floor). 0 = canonical WN
        # (no floor). A floor >0 makes weakly-used channels linear (w/floor -> silent)
        # instead of blowing up when kernel weight decay drives ||w|| -> 0.
        self.wn_norm_floor = float(wn_norm_floor)
        # weight_norm init/gauge controls (Q-experiments):
        #   wn_gamma_mult: gamma_0 = mult * kaiming norm (over/under-scaled init)
        #   branch_init_angle_deg: for a 2-branch block set the init angle theta_0 between
        #     w_1 and w_2 exactly (w_2 = cos t0 * w1_hat + sin t0 * z_hat, z_hat random _|_ w1_hat)
        #   wn_gamma_match_angle: divide gamma_0 by cos(theta_0/2) per channel so the pair's
        #     effective init scale gamma cos(theta/2) equals the single's gamma_0
        #   wn_fix_norm: project every kernel back to its init norm before each training
        #     forward (pure gauge under WN -> sigma constant, effective lr constant)
        self.wn_gamma_mult = float(wn_gamma_mult)
        self.wn_fix_norm = bool(wn_fix_norm)
        self.branch_init_angle_deg = None if branch_init_angle_deg is None else float(branch_init_angle_deg)
        self.wn_gamma_match_angle = bool(wn_gamma_match_angle)
        #   wn_kernel_scale: multiply the kernels by s after the gauge (w_hat unchanged, so the
        #     function is unchanged); under WN the direction learning rate is eta*gamma/||w||^2, so
        #     with wn_fix_norm this is a persistent direction-lr multiplier 1/s^2 (optimizer-side
        #     stand-in for the pair's 1/cos(theta/2) angular-rate advantage: s = sqrt(cos(theta0/2))).
        self.wn_kernel_scale = float(wn_kernel_scale)
        #   wn_kernel_scale_end / _anneal_epochs: with wn_fix_norm, anneal the kernel scale linearly
        #     (per training step) from wn_kernel_scale to wn_kernel_scale_end over that many epochs
        #     (wn_steps_per_epoch steps each) -> a decaying direction-lr multiplier, the single-branch
        #     stand-in for the pair's Q-locked closing (angular lr high while open, annealed as it aligns).
        self.wn_kernel_scale_end = None if wn_kernel_scale_end is None else float(wn_kernel_scale_end)
        self.wn_kernel_scale_anneal_epochs = float(wn_kernel_scale_anneal_epochs)
        self.wn_steps_per_epoch = int(wn_steps_per_epoch)
        self._wn_train_steps = 0
        #   wn_gamma_lr_mult: gamma is stored as gamma_raw with gamma = k * gamma_raw (k = sqrt(mult)),
        #     so plain SGD on gamma_raw is SGD on gamma with lr x mult (function unchanged); with
        #     wn_gamma_lr_mult_end the multiplier is annealed (same linear-per-step schedule as the
        #     kernel scale, gamma_raw rescaled so gamma is continuous). Single-branch stand-in for the
        #     pair's faster scale learning  g_dot ∝ [cos^2(theta/2) + (gamma^2/2sigma^2) sin^2(theta/2)].
        self.wn_gamma_lr_mult = float(wn_gamma_lr_mult)
        self.wn_gamma_lr_mult_end = None if wn_gamma_lr_mult_end is None else float(wn_gamma_lr_mult_end)
        self.wn_gamma_k = math.sqrt(self.wn_gamma_lr_mult)
        #   kernel_init_scale (per_branch BN blocks): multiply the 3x3 kernels by s at init. Under BN the
        #     kernel norm is a gauge (function unchanged) that sets the direction lr eta/||w||^2, so this is
        #     an initial direction-lr multiplier 1/s^2 that self-anneals as WD equilibrates the norm.
        self.kernel_init_scale = float(kernel_init_scale)

        if bn_position == "per_branch":
            self.conv3_branches = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
                        _make_conv_norm(out_channels, norm),
                    )
                    for _ in range(num_3x3)
                ]
            )
            self.conv1 = (
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, stride=stride, padding=0, bias=False),
                    _make_conv_norm(out_channels, norm),
                )
                if use_1x1
                else None
            )
            self.identity = (
                _make_conv_norm(out_channels, norm)
                if self.has_identity and identity_use_bn
                else nn.Identity()
            )
            self.post_norm = nn.Identity()
        elif bn_position == "weight_norm":
            # Per-branch WEIGHT normalisation, no activation norm at all:
            #   out = gamma * (1/N) sum_i conv(x, w_i / ||w_i||_c) + b
            # (||.||_c = Euclidean norm over in_ch x 3 x 3 per output channel c),
            # gamma/b shared across the N branches, per channel.  This is the isotropic
            # idealisation of the per-branch-BN pair (sigma_i == ||w_i|| exactly, no
            # input covariance), in the note's convention u = gamma/2 (w1_hat + w2_hat).
            # Init convention (branch_scale_init is IGNORED in this mode):
            #   * each kernel kaiming, then scaled by 1/sqrt(N)  (LR gauge: with the 1/N
            #     in the sum, the per-branch direction step (gamma/N)/||w_i||^2 equals the
            #     single's gamma/||w||^2; kernel WD preserves the gauge at equilibrium)
            #   * gamma_c = kaiming norm (= sqrt(N) * mean_i ||w_{i,c}||)  so a single or a
            #     TIED pair equals the plain kaiming conv at init; an independent pair has
            #     the same gamma and magnitude cos(theta/2) of it (the note's setup).
            # Folds exactly to one conv at inference.
            self.conv3_branches = nn.ModuleList(
                [
                    nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
                    for _ in range(num_3x3)
                ]
            )
            self.conv1 = None
            self.identity = nn.Identity()
            self.post_norm = nn.Identity()
            self.wn_gamma = nn.Parameter(torch.ones(out_channels))
            self.wn_bias = nn.Parameter(torch.zeros(out_channels))
        else:  # post_sum: bare convs summed, one shared norm on the sum
            self.conv3_branches = nn.ModuleList(
                [
                    nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
                    for _ in range(num_3x3)
                ]
            )
            self.conv1 = (
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, padding=0, bias=False)
                if use_1x1
                else None
            )
            self.identity = nn.Identity()  # raw skip add; no per-branch params
            self.post_norm = _make_conv_norm(out_channels, norm)
        self.activation = nn.ReLU(inplace=True)

        self.reset_parameters(branch_scale_init)

    def _conv3_weights(self) -> list[torch.Tensor]:
        """Conv weight tensors of the 3x3 branches, handling both BN positions."""
        if self.bn_position == "per_branch":
            return [branch[0].weight for branch in self.conv3_branches]
        return [branch.weight for branch in self.conv3_branches]

    def fused_3x3_l2(self) -> torch.Tensor:
        """Squared L2 norm of the fused 3x3 weight ``||sum_i W_i||^2``.

        Only well defined as the literal inference weight in ``post_sum`` mode;
        in ``per_branch`` mode it is a useful proxy that ignores BN scaling.
        Only the 3x3 branches are folded (the experiment leaves 1x1/identity
        off, so they are not part of the fused-weight decay control).
        """
        weights = self._conv3_weights()
        fused = weights[0]
        for w in weights[1:]:
            fused = fused + w
        return fused.pow(2).sum()

    def split_3x3_l2(self) -> torch.Tensor:
        """Per-branch squared L2 of the 3x3 branches ``sum_i ||W_i||^2``.

        This is what per-branch optimizer weight decay penalizes. For ``N``
        summed branches it equals, at the balanced optimum ``W_i = W/N``, an
        effective decay of ``1/N`` of the fused weight energy ``||W||^2``.
        """
        weights = self._conv3_weights()
        total = weights[0].pow(2).sum()
        for w in weights[1:]:
            total = total + w.pow(2).sum()
        return total

    def _tie_branch_init(self) -> None:
        """Copy the first 3x3 branch's conv weights into all the others.

        Identical-init branches receive identical inputs and gradients, so under
        gradient descent they remain identical forever (perfect symmetry) -- a
        2-branch block behaves like one scaled branch. Used to test whether the
        per-branch multi-branch advantage requires branch *diversity* (symmetry
        breaking) rather than merely having several normalized paths.
        """
        if self.num_3x3 < 2:
            return
        weights = self._conv3_weights()
        with torch.no_grad():
            for w in weights[1:]:
                w.copy_(weights[0])

    def reset_parameters(self, branch_scale_init: ScaleInit) -> None:
        branch_count = self.num_3x3 + int(self.conv1 is not None) + int(self.has_identity)
        scale = branch_scale_init_value(branch_scale_init, branch_count)
        # Recorded so probes/tests can assert the construction instead of re-deriving
        # it: 1.0 == function-preserving vs a single-branch reference.
        self.init_fused_gain = branch_gain_at_init(
            branch_scale_init, branch_count, self.bn_position
        )
        if self.bn_position == "per_branch":
            for branch in self.conv3_branches:
                nn.init.kaiming_normal_(branch[0].weight, mode="fan_in", nonlinearity="relu")
                if self.kernel_init_scale != 1.0:
                    with torch.no_grad():
                        branch[0].weight.mul_(self.kernel_init_scale)
                _set_scale(branch[1], scale)
            if self.conv1 is not None:
                nn.init.kaiming_normal_(self.conv1[0].weight, mode="fan_in", nonlinearity="relu")
                _set_scale(self.conv1[1], scale)
            if self.has_identity:
                _set_scale(self.identity, scale)
        elif self.bn_position == "weight_norm":
            n_br = self.num_3x3
            for conv in self.conv3_branches:
                nn.init.kaiming_normal_(conv.weight, mode="fan_in", nonlinearity="relu")
            if self.branch_init_identical:
                self._tie_branch_init()
            with torch.no_grad():
                if (self.branch_init_angle_deg is not None and n_br == 2
                        and not self.branch_init_identical):
                    # exact init angle theta_0 between the two kernels, per out channel
                    t0 = math.radians(self.branch_init_angle_deg)
                    w1, w2 = self._conv3_weights()
                    f1, f2 = w1.flatten(1), w2.flatten(1)
                    u1 = f1 / f1.norm(dim=1, keepdim=True)
                    z = f2 - (f2 * u1).sum(1, keepdim=True) * u1
                    z = z / z.norm(dim=1, keepdim=True).clamp_min(1e-12)
                    f2_new = f1.norm(dim=1, keepdim=True) * (math.cos(t0) * u1 + math.sin(t0) * z)
                    w2.copy_(f2_new.view_as(w2))
                # gamma_c = kaiming norm of the (unscaled) kernel(s)
                norms = torch.stack(
                    [w.flatten(1).norm(dim=1) for w in self._conv3_weights()]
                ).mean(0)
                self.wn_gamma.copy_(norms * self.wn_gamma_mult)
                if self.wn_gamma_match_angle and n_br == 2:
                    w1, w2 = [w.flatten(1) for w in self._conv3_weights()]
                    cos_t = (w1 * w2).sum(1) / (w1.norm(dim=1) * w2.norm(dim=1)).clamp_min(1e-12)
                    cos_half = torch.sqrt((0.5 * (1.0 + cos_t)).clamp_min(1e-6))
                    self.wn_gamma.div_(cos_half)
                self.wn_bias.zero_()
                # 1/sqrt(N) kernel gauge (see the constructor comment), times wn_kernel_scale
                for w in self._conv3_weights():
                    w.mul_(self.wn_kernel_scale / math.sqrt(n_br))
                self.wn_init_norms = torch.stack(
                    [w.detach().flatten(1).norm(dim=1).clone() for w in self._conv3_weights()]
                )  # [N, out]
                if self.wn_gamma_k != 1.0:
                    self.wn_gamma.div_(self.wn_gamma_k)   # stored raw; effective gamma = k * raw
            self.init_fused_gain = 1.0
            return
        else:
            # post_sum: no per-branch norm, so the conv scale SURVIVES. Fold it into
            # the conv weights; with "inv_sqrt_n" the independent branches' variances
            # add back to the single-branch magnitude. (Under per_branch the conv
            # scale is instead erased by BN — see the ScaleInit comment at the top.)
            for conv in self.conv3_branches:
                nn.init.kaiming_normal_(conv.weight, mode="fan_in", nonlinearity="relu")
                with torch.no_grad():
                    conv.weight.mul_(scale)
            if self.conv1 is not None:
                nn.init.kaiming_normal_(self.conv1.weight, mode="fan_in", nonlinearity="relu")
                with torch.no_grad():
                    self.conv1.weight.mul_(scale)

        if self.branch_init_identical:
            self._tie_branch_init()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bn_position == "per_branch":
            out = sum(branch(x) for branch in self.conv3_branches)
            if self.conv1 is not None:
                out = out + self.conv1(x)
            if self.has_identity:
                out = out + self.identity(x)
            return self.activation(out)
        if self.bn_position == "weight_norm":
            if self.wn_fix_norm and self.training:
                with torch.no_grad():
                    smul = 1.0
                    frac = 0.0
                    if self.wn_kernel_scale_anneal_epochs > 0:
                        frac = min(1.0, self._wn_train_steps / (self.wn_kernel_scale_anneal_epochs * self.wn_steps_per_epoch))
                    if self.wn_kernel_scale_end is not None and self.wn_kernel_scale_anneal_epochs > 0:
                        s_now = self.wn_kernel_scale + (self.wn_kernel_scale_end - self.wn_kernel_scale) * frac
                        smul = s_now / self.wn_kernel_scale
                    if self.wn_gamma_lr_mult_end is not None and self.wn_kernel_scale_anneal_epochs > 0:
                        k_new = math.sqrt(self.wn_gamma_lr_mult + (self.wn_gamma_lr_mult_end - self.wn_gamma_lr_mult) * frac)
                        if k_new != self.wn_gamma_k:
                            self.wn_gamma.mul_(self.wn_gamma_k / k_new)   # keep effective gamma continuous
                            self.wn_gamma_k = k_new
                    self._wn_train_steps += 1
                    for i, conv in enumerate(self.conv3_branches):
                        n = conv.weight.flatten(1).norm(dim=1).clamp_min(1e-12)
                        tgt = self.wn_init_norms[i].to(conv.weight.device) * smul
                        conv.weight.mul_((tgt / n).view(-1, 1, 1, 1))
            out = None
            for conv in self.conv3_branches:
                w = conv.weight
                # canonical weight norm (no eps): w_hat = w/||w||. NOTE: under kernel
                # weight decay a channel with ~zero loss gradient (unused) has no
                # norm equilibrium and decays as exp(-lambda*T); at lambda=5e-4,
                # constant lr 0.1, 100 ep this reached ||w||~1e-6..1e-8 in 5-6/256
                # last-block channels and 2 of 9 runs diverged (eps=1e-5 did NOT
                # prevent it). Keep lambda*T small (lambda<=1e-4 for 100 ep) or add
                # a guard if reusing this block at heavier decay.
                w_hat = w / w.flatten(1).norm(dim=1).clamp_min(
                    max(self.wn_norm_floor, 1e-12)).view(-1, 1, 1, 1)
                y = F.conv2d(x, w_hat, None, self.stride, 1)
                out = y if out is None else out + y
            gam = self.wn_gamma * self.wn_gamma_k if self.wn_gamma_k != 1.0 else self.wn_gamma
            out = out * (gam / self.num_3x3).view(1, -1, 1, 1) + self.wn_bias.view(1, -1, 1, 1)
            return self.activation(out)
        # post_sum: sum bare convs, then one shared norm
        out = sum(conv(x) for conv in self.conv3_branches)
        if self.conv1 is not None:
            out = out + self.conv1(x)
        if self.has_identity:
            out = out + x
        return self.activation(self.post_norm(out))


class ClaudeRepVGGCifar(nn.Module):
    """Small CIFAR classifier built from editable RepVGG-style blocks."""

    def __init__(
        self,
        num_classes: int = 10,
        width_mult: float = 1.0,
        stage_channels: Sequence[int] = (32, 64, 128),
        stage_blocks: Sequence[int] = (1, 1, 1),
        stage_strides: Sequence[int] = (1, 2, 2),
        num_3x3: int = 1,
        use_1x1: bool = False,
        use_identity: bool = False,
        identity_use_bn: bool = True,
        norm: NormName = "batch",
        branch_scale_init: ScaleInit = "inv_sqrt_n",
        bn_position: BNPosition = "per_branch",
        branch_init_identical: bool = False,
        wn_norm_floor: float = 0.0,
        wn_gamma_mult: float = 1.0,
        wn_fix_norm: bool = False,
        branch_init_angle_deg: float | None = None,
        wn_gamma_match_angle: bool = False,
        wn_kernel_scale: float = 1.0,
        wn_kernel_scale_end: float | None = None,
        wn_kernel_scale_anneal_epochs: float = 0.0,
        wn_steps_per_epoch: int = 391,
        wn_gamma_lr_mult: float = 1.0,
        wn_gamma_lr_mult_end: float | None = None,
        kernel_init_scale: float = 1.0,
        branch_decay_mode: Literal["split", "fused"] = "split",
        branch_decay_lambda: float = 0.0,
        base_decay_lambda: float = 0.0,
    ) -> None:
        super().__init__()
        if not (len(stage_channels) == len(stage_blocks) == len(stage_strides)):
            raise ValueError("stage_channels, stage_blocks, and stage_strides must match")
        if branch_decay_mode not in ("split", "fused"):
            raise ValueError(
                f"branch_decay_mode must be 'split' or 'fused', got {branch_decay_mode!r}"
            )
        self.branch_decay_mode = branch_decay_mode
        self.branch_decay_lambda = float(branch_decay_lambda)
        self.base_decay_lambda = float(base_decay_lambda)

        channels = [max(1, int(c * width_mult)) for c in stage_channels]
        in_channels = 3
        stages: list[nn.Module] = []
        for out_channels, blocks, stride in zip(channels, stage_blocks, stage_strides):
            layers: list[nn.Module] = []
            for block_idx in range(int(blocks)):
                block_stride = int(stride) if block_idx == 0 else 1
                block_in = in_channels if block_idx == 0 else out_channels
                layers.append(
                    ClaudeRepVGGBlock(
                        block_in,
                        out_channels,
                        stride=block_stride,
                        num_3x3=num_3x3,
                        use_1x1=use_1x1,
                        use_identity=use_identity,
                        identity_use_bn=identity_use_bn,
                        norm=norm,
                        branch_scale_init=branch_scale_init,
                        bn_position=bn_position,
                        branch_init_identical=branch_init_identical,
                        wn_norm_floor=wn_norm_floor,
                        wn_gamma_mult=wn_gamma_mult,
                        wn_fix_norm=wn_fix_norm,
                        branch_init_angle_deg=branch_init_angle_deg,
                        wn_gamma_match_angle=wn_gamma_match_angle,
                        wn_kernel_scale=wn_kernel_scale,
                        wn_kernel_scale_end=wn_kernel_scale_end,
                        wn_kernel_scale_anneal_epochs=wn_kernel_scale_anneal_epochs,
                        wn_steps_per_epoch=wn_steps_per_epoch,
                        wn_gamma_lr_mult=wn_gamma_lr_mult,
                        wn_gamma_lr_mult_end=wn_gamma_lr_mult_end,
                        kernel_init_scale=kernel_init_scale,
                    )
                )
            stages.append(nn.Sequential(*layers))
            in_channels = out_channels

        self.stages = nn.ModuleList(stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for stage in self.stages:
            x = stage(x)
        return self.fc(self.pool(x).flatten(1))

    def custom_l2(self) -> torch.Tensor:
        """Full weight-decay energy ``U``, owned entirely by the model.

        The generic trainer adds ``train.custom_l2 * model.custom_l2()`` to the
        loss. Run with ``train.custom_l2: 1.0`` and the optimizer's own
        ``weight_decay: 0`` so this method is the *only* source of L2 decay.
        Then every variant decays the non-branch parameters (fc, BN affine)
        identically and differs only in how the foldable 3x3 branches are
        penalized -- removing the fc/BN decay confound that plain
        ``weight_decay`` rescaling would introduce.

        - ``base_decay_lambda`` applies ``(lambda/2)||p||^2`` to every parameter
          that is not a 3x3 branch conv weight. Autograd yields the gradient
          ``lambda * p`` that PyTorch SGD's ``weight_decay`` adds before
          momentum, so it matches native weight decay.
        - ``split`` mode penalizes the branches as ``sum_i ||W_i||^2``
          (per-branch decay): for ``N`` summed branches an effective decay of
          ``lambda/N`` on the fused weight ``W = sum_i W_i``.
        - ``fused`` mode penalizes ``||sum_i W_i||^2`` -- decay applied directly
          to the literal fused inference weight at ``branch_decay_lambda``.
        """
        zero = next(self.parameters()).new_zeros(())

        branch_ids: set[int] = set()
        branch_term = zero
        for module in self.modules():
            if isinstance(module, ClaudeRepVGGBlock):
                for w in module._conv3_weights():
                    branch_ids.add(id(w))
                term = (
                    module.fused_3x3_l2()
                    if self.branch_decay_mode == "fused"
                    else module.split_3x3_l2()
                )
                branch_term = branch_term + term

        base_term = zero
        if self.base_decay_lambda > 0.0:
            for p in self.parameters():
                if id(p) in branch_ids:
                    continue
                base_term = base_term + p.pow(2).sum()

        return (
            0.5 * self.branch_decay_lambda * branch_term
            + 0.5 * self.base_decay_lambda * base_term
        )

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class ChannelScaleInjector(nn.Module):
    """Fixed (non-learned) per-channel diagonal rescale — the causal R-knob.

    Holds a frozen per-channel scale buffer ``s`` whose log is i.i.d.
    ``N(0, r_log_spread**2)``, then RMS-normalized so ``mean(s**2) == 1``. With
    ``r_log_spread == 0`` it is exactly the identity (R == 1, the null control);
    larger ``r_log_spread`` redistributes magnitude ACROSS channels without
    changing the overall feature energy, directly setting the per-channel
    heterogeneity ``R`` that the downstream norm sees.

    Because ``s`` is a fixed diagonal applied right before the next block's norm,
    it is benign and identical at train/eval. BN's sigma-divide cancels it
    per-channel (so BN is the built-in null control — insensitive to the knob),
    while mean-only keeps the magnitude and therefore FEELS it.
    """

    def __init__(self, channels: int, r_log_spread: float = 0.0, seed: int = 0) -> None:
        super().__init__()
        self.r_log_spread = float(r_log_spread)
        if self.r_log_spread <= 0.0:
            scale = torch.ones(channels)
        else:
            g = torch.Generator().manual_seed(int(seed))
            log_s = torch.randn(channels, generator=g) * self.r_log_spread
            scale = torch.exp(log_s)
            scale = scale / scale.pow(2).mean().clamp(min=1e-12).sqrt()  # RMS-normalize
        self.register_buffer("scale", scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale[None, :, None, None]


class HeteroRepVGGCifar(ClaudeRepVGGCifar):
    """``ClaudeRepVGGCifar`` plus the causal R-knob: a frozen per-channel rescale
    injected after stage ``inject_after_stage`` (default 0, the post-stem feature
    map), redistributing per-channel magnitude with spread ``r_log_spread``.

    ``r_log_spread == 0`` reproduces the parent exactly (R == 1). All branch
    structure, norms, and folding are inherited unchanged; the injector is a fixed
    diagonal and does not alter inference capacity.
    """

    def __init__(
        self,
        *args,
        r_log_spread: float = 0.0,
        inject_after_stage: int = 0,
        knob_seed: int = 0,
        stage_channels: Sequence[int] = (32, 64, 128),
        width_mult: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(
            *args, stage_channels=stage_channels, width_mult=width_mult, **kwargs
        )
        self.inject_after_stage = int(inject_after_stage)
        channels = [max(1, int(c * width_mult)) for c in stage_channels]
        inj_channels = channels[self.inject_after_stage]
        self.injector = ChannelScaleInjector(
            inj_channels, r_log_spread=r_log_spread, seed=knob_seed
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i == self.inject_after_stage:
                x = self.injector(x)
        return self.fc(self.pool(x).flatten(1))


class LayerwiseRepVGGCifar(nn.Module):
    """RepVGG-CIFAR where ``norm`` and ``num_3x3`` may be set PER BLOCK, so a
    structural treatment (mean-only norm, or a 2× 3×3 two-branch) can be applied to
    a chosen SUBSET of layers — e.g. only the naturally high-R early layers — with
    the rest left as plain BN single-branch.

    Each of ``norm`` / ``num_3x3`` is either a scalar (broadcast to every block) or a
    list of length == total block count (used block-by-block, in stage→block order).
    This is the INJECTION-FREE knob: it varies nothing but which layers get which
    foldable structure, testing whether the gain localizes to the high-R layers.
    Every block still folds to one inference conv.
    """

    def __init__(
        self,
        num_classes: int = 10,
        width_mult: float = 1.0,
        stage_channels: Sequence[int] = (64, 128, 256),
        stage_blocks: Sequence[int] = (3, 3, 3),
        stage_strides: Sequence[int] = (1, 2, 2),
        num_3x3: "int | Sequence[int]" = 1,
        norm: "NormName | Sequence[NormName]" = "batch",
        use_1x1: bool = False,
        use_identity: bool = False,
        identity_use_bn: bool = True,
        branch_scale_init: ScaleInit = "inv_sqrt_n",
        bn_position: BNPosition = "per_branch",
        branch_init_identical: bool = False,
        wn_norm_floor: float = 0.0,
        wn_gamma_mult: float = 1.0,
        wn_fix_norm: bool = False,
        branch_init_angle_deg: float | None = None,
        wn_gamma_match_angle: bool = False,
        wn_kernel_scale: float = 1.0,
        wn_kernel_scale_end: float | None = None,
        wn_kernel_scale_anneal_epochs: float = 0.0,
        wn_steps_per_epoch: int = 391,
        wn_gamma_lr_mult: float = 1.0,
        wn_gamma_lr_mult_end: float | None = None,
        kernel_init_scale: float = 1.0,
        branch_decay_lambda: float = 0.0,
        base_decay_lambda: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__()
        n_blocks = sum(int(b) for b in stage_blocks)
        norms = list(norm) if isinstance(norm, (list, tuple)) else [norm] * n_blocks
        n3s = list(num_3x3) if isinstance(num_3x3, (list, tuple)) else [int(num_3x3)] * n_blocks
        if len(norms) != n_blocks or len(n3s) != n_blocks:
            raise ValueError(
                f"per-block norm/num_3x3 must have len == #blocks ({n_blocks}); "
                f"got len(norm)={len(norms)}, len(num_3x3)={len(n3s)}"
            )
        self.block_norms = norms
        self.block_num_3x3 = [int(v) for v in n3s]
        channels = [max(1, int(c * width_mult)) for c in stage_channels]
        in_channels = 3
        stages: list[nn.Module] = []
        bi = 0
        for out_channels, blocks, stride in zip(channels, stage_blocks, stage_strides):
            layers: list[nn.Module] = []
            for block_idx in range(int(blocks)):
                block_stride = int(stride) if block_idx == 0 else 1
                block_in = in_channels if block_idx == 0 else out_channels
                layers.append(
                    ClaudeRepVGGBlock(
                        block_in,
                        out_channels,
                        stride=block_stride,
                        num_3x3=int(n3s[bi]),
                        use_1x1=use_1x1,
                        use_identity=use_identity,
                        identity_use_bn=identity_use_bn,
                        norm=norms[bi],
                        branch_scale_init=branch_scale_init,
                        bn_position=bn_position,
                        branch_init_identical=branch_init_identical,
                        wn_norm_floor=wn_norm_floor,
                        wn_gamma_mult=wn_gamma_mult,
                        wn_fix_norm=wn_fix_norm,
                        branch_init_angle_deg=branch_init_angle_deg,
                        wn_gamma_match_angle=wn_gamma_match_angle,
                        wn_kernel_scale=wn_kernel_scale,
                        wn_kernel_scale_end=wn_kernel_scale_end,
                        wn_kernel_scale_anneal_epochs=wn_kernel_scale_anneal_epochs,
                        wn_steps_per_epoch=wn_steps_per_epoch,
                        wn_gamma_lr_mult=wn_gamma_lr_mult,
                        wn_gamma_lr_mult_end=wn_gamma_lr_mult_end,
                        kernel_init_scale=kernel_init_scale,
                    )
                )
                bi += 1
            stages.append(nn.Sequential(*layers))
            in_channels = out_channels

        self.stages = nn.ModuleList(stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels[-1], num_classes)
        # decay attrs kept for custom_l2 tooling compatibility (unused at WD=0)
        self.branch_decay_mode = "split"
        self.branch_decay_lambda = float(branch_decay_lambda)
        self.base_decay_lambda = float(base_decay_lambda)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for stage in self.stages:
            x = stage(x)
        return self.fc(self.pool(x).flatten(1))

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# --- chan_hetero probe: measure the per-channel magnitude heterogeneity R ------
from structural_reparam.analysis.registry import ProbeContext, register_probe  # noqa: E402

_NORM_TYPES = (nn.BatchNorm2d, MeanOnlyNorm2d, PartialBatchNorm2d)


@register_probe("chan_hetero")
class ChannelHeteroProbe:
    """Per-epoch per-channel magnitude-heterogeneity probe — the CIFAR-side ``R``.

    Registers forward-pre-hooks on every conv-norm module and, over training-mode
    forwards within the epoch, accumulates per-channel mean/mean-square of each
    norm's INPUT (the magnitude the sigma-divide acts on). At ``epoch_stats`` it
    derives per-channel ``sigma_c`` and logs, per layer and as a network summary:

      chan_hetero/layerN/R        — p95(sigma_c)/p5(sigma_c) across channels
      chan_hetero/layerN/cv       — std(sigma_c)/mean(sigma_c) across channels
      chan_hetero/R_net           — median layer R (the headline scalar)
      chan_hetero/R_net_mean      — mean layer R

    No extra forward pass and no dataloader needed; accumulators reset each epoch.
    """

    def __init__(self, model: nn.Module) -> None:
        self.layers = [m for m in model.modules() if isinstance(m, _NORM_TYPES)]
        self._acc: dict[int, dict[str, torch.Tensor | float]] = {}
        self._handles = []
        for idx, layer in enumerate(self.layers):
            self._handles.append(
                layer.register_forward_pre_hook(self._make_hook(idx))
            )

    @classmethod
    def from_context(cls, ctx: ProbeContext) -> "ChannelHeteroProbe":
        return cls(model=ctx.model)

    def _make_hook(self, idx: int):
        def hook(module: nn.Module, inputs):
            if not module.training:
                return
            x = inputs[0].detach().double()
            # pooled per-channel raw moments over (N, H, W), summed across batches
            s = x.sum(dim=(0, 2, 3))
            sq = x.pow(2).sum(dim=(0, 2, 3))
            c3 = x.pow(3).sum(dim=(0, 2, 3))
            c4 = x.pow(4).sum(dim=(0, 2, 3))
            n = float(x.shape[0] * x.shape[2] * x.shape[3])
            # per-BATCH per-channel sigma (for batch-to-batch sigma instability)
            bmean = s / n
            bvar = (sq / n - bmean.pow(2)).clamp(min=0.0)
            bsig = bvar.sqrt()
            # PER-EXAMPLE conditional heterogeneity: within EACH image, per-channel
            # spatial std s_{n,c}, then cv/R ACROSS channels for that image, summed
            # over images (captures the within-sample heterogeneity the dataset-pooled
            # cv washes out — the marginally-symmetric / conditional structure).
            xm = x.mean(dim=(2, 3))                       # (N, C) per-image per-channel mean
            xsq = x.pow(2).mean(dim=(2, 3))               # (N, C)
            sp = (xsq - xm.pow(2)).clamp(min=0.0).sqrt()  # (N, C) per-image per-channel std
            mu_c = sp.mean(dim=1)                         # (N,)
            sd_c = sp.std(dim=1, unbiased=False)          # (N,)
            cv_pe = (sd_c / mu_c.clamp(min=1e-8))         # (N,)
            qs = torch.quantile(sp, sp.new_tensor([0.05, 0.95]), dim=1)  # (2, N)
            R_pe = qs[1] / qs[0].clamp(min=1e-8)          # (N,)
            cvpe = cv_pe.sum(); Rpe = R_pe.sum(); npe = float(x.shape[0])
            a = self._acc.get(idx)
            if a is None:
                self._acc[idx] = {"s": s, "sq": sq, "c3": c3, "c4": c4, "n": n,
                                  "bsig": bsig, "bsig2": bsig.pow(2), "nb": 1,
                                  "cvpe": cvpe, "Rpe": Rpe, "npe": npe}
            else:
                a["s"] += s; a["sq"] += sq; a["c3"] += c3; a["c4"] += c4; a["n"] += n
                a["bsig"] += bsig; a["bsig2"] += bsig.pow(2); a["nb"] += 1
                a["cvpe"] += cvpe; a["Rpe"] += Rpe; a["npe"] += npe
        return hook

    @staticmethod
    def _percentile(t: torch.Tensor, q: float) -> float:
        return torch.quantile(t, q).item()

    def epoch_stats(self, epoch: int = 0) -> dict[str, float]:
        """Candidate SINGLE-BRANCH metrics (the predictors of the two-branch gain).

        Per layer:
          R       — p95(sigma_c)/p5(sigma_c) across channels (cross-channel heterogeneity)
          cv      — std(sigma_c)/mean(sigma_c) across channels
          signoise— mean over channels of CV-across-batches of the per-batch sigma_c
                    (batch-to-batch sigma instability = the noise BN's divide injects;
                     the synthetic mechanism's direct analogue)
          kurt    — mean over channels of excess kurtosis of the activations
        Network aggregates: *_net (median/mean across layers) + R_max.
        """
        stats: dict[str, float] = {}
        layer_R: list[float] = []
        layer_sn: list[float] = []
        layer_ku: list[float] = []
        layer_cv: list[float] = []
        layer_cv_pe: list[float] = []
        layer_R_pe: list[float] = []
        for idx in range(len(self.layers)):
            a = self._acc.get(idx)
            if a is None or a["n"] <= 1 or a["nb"] < 2:
                continue
            n = a["n"]
            M1 = a["s"] / n; M2 = a["sq"] / n; M3 = a["c3"] / n; M4 = a["c4"] / n
            var = (M2 - M1.pow(2)).clamp(min=0.0)
            sigma = var.sqrt()
            keep = sigma > 1e-8
            if keep.sum() < 2:
                continue
            sg = sigma[keep]
            R = self._percentile(sg, 0.95) / max(self._percentile(sg, 0.05), 1e-8)
            cv = (sg.std(unbiased=False) / sg.mean().clamp(min=1e-8)).item()
            # excess kurtosis (central moments from raw)
            mu2 = (M2 - M1.pow(2)).clamp(min=1e-12)
            mu4 = M4 - 4 * M1 * M3 + 6 * M1.pow(2) * M2 - 3 * M1.pow(4)
            kurt = ((mu4 / mu2.pow(2) - 3.0)[keep]).mean().item()
            # batch-to-batch sigma instability: CV across batches of batch-sigma
            nb = a["nb"]
            bmean = a["bsig"] / nb
            bvar = (a["bsig2"] / nb - bmean.pow(2)).clamp(min=0.0)
            bcv = (bvar.sqrt() / bmean.clamp(min=1e-8))[keep]
            signoise = bcv.mean().item()
            stats[f"chan_hetero/layer{idx}/R"] = R
            stats[f"chan_hetero/layer{idx}/cv"] = cv
            stats[f"chan_hetero/layer{idx}/signoise"] = signoise
            stats[f"chan_hetero/layer{idx}/kurt"] = kurt
            layer_R.append(R); layer_sn.append(signoise); layer_ku.append(kurt)
            layer_cv.append(cv)
            # per-example (within-image, cross-channel) cv and R, averaged over images
            if a.get("npe", 0) > 0:
                cv_pe = (a["cvpe"] / a["npe"]).item() if hasattr(a["cvpe"], "item") else a["cvpe"] / a["npe"]
                R_pe = (a["Rpe"] / a["npe"]).item() if hasattr(a["Rpe"], "item") else a["Rpe"] / a["npe"]
                stats[f"chan_hetero/layer{idx}/cv_perex"] = cv_pe
                stats[f"chan_hetero/layer{idx}/R_perex"] = R_pe
                layer_cv_pe.append(cv_pe); layer_R_pe.append(R_pe)
        if layer_R:
            rt = torch.tensor(layer_R)
            stats["chan_hetero/R_net"] = rt.median().item()
            stats["chan_hetero/R_net_mean"] = rt.mean().item()
            stats["chan_hetero/R_max"] = rt.max().item()
            cvt = torch.tensor(layer_cv)
            stats["chan_hetero/cv_net"] = cvt.median().item()
            stats["chan_hetero/cv_net_mean"] = cvt.mean().item()
        if layer_cv_pe:
            cpe = torch.tensor(layer_cv_pe); rpe = torch.tensor(layer_R_pe)
            stats["chan_hetero/cv_net_perex"] = cpe.median().item()
            stats["chan_hetero/cv_net_perex_mean"] = cpe.mean().item()
            stats["chan_hetero/R_net_perex"] = rpe.median().item()
            stats["chan_hetero/signoise_net"] = torch.tensor(layer_sn).mean().item()
            stats["chan_hetero/kurt_net"] = torch.tensor(layer_ku).mean().item()
        self._acc = {}  # reset for next epoch
        return stats

    def close(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []


@register_probe("weight_norm")
class WeightNormProbe:
    """Per-epoch conv weight-norm probe — logs ||W|| (the quantity that grows
    monotonically under BN without weight decay, since BN makes the loss
    scale-invariant in W so gradients are orthogonal to W).

    Logged each epoch:
      weight_norm/total           — sqrt(sum of squared L2 norms of all conv weights)
      weight_norm/conv_mean       — mean per-conv ||W||
      weight_norm/convN           — per-conv ||W|| (3x3 branch convs, in order)
      weight_norm/fc              — classifier weight ||W||
    Read-only on parameters; no hooks, no extra forward.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    @classmethod
    def from_context(cls, ctx: ProbeContext) -> "WeightNormProbe":
        return cls(model=ctx.model)

    def epoch_stats(self, epoch: int = 0) -> dict[str, float]:
        stats: dict[str, float] = {}
        sq_total = 0.0
        per_conv: list[float] = []
        idx = 0
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                wn = m.weight.detach().float().norm().item()
                stats[f"weight_norm/conv{idx}"] = wn
                per_conv.append(wn)
                sq_total += wn * wn
                idx += 1
            elif isinstance(m, nn.Linear):
                stats["weight_norm/fc"] = m.weight.detach().float().norm().item()
        if per_conv:
            stats["weight_norm/total"] = float(sq_total ** 0.5)
            stats["weight_norm/conv_mean"] = sum(per_conv) / len(per_conv)
        return stats

    def close(self) -> None:
        pass


class ClaudeMLPBlock(nn.Module):
    """Editable branchable MLP block.

    Each branch is Linear -> norm. Branch outputs are summed, scaled by a shared
    gamma, shifted by a shared bias, then passed through ReLU. This mirrors the
    core branch-vs-single questions without convolutional structure.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_branches: int = 1,
        norm: NormName = "batch",
        branch_scale_init: ScaleInit = "inv_sqrt_n",
    ) -> None:
        super().__init__()
        if num_branches < 1:
            raise ValueError(f"num_branches must be >= 1, got {num_branches}")
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(in_features, out_features, bias=False),
                    _make_mlp_norm(out_features, norm),
                )
                for _ in range(num_branches)
            ]
        )
        gamma = num_branches ** -0.5 if branch_scale_init == "inv_sqrt_n" else 1.0
        self.gamma = nn.Parameter(torch.full((out_features,), float(gamma)))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.activation = nn.ReLU(inplace=True)
        for branch in self.branches:
            nn.init.kaiming_normal_(branch[0].weight, mode="fan_in", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = sum(branch(x) for branch in self.branches)
        out = out * self.gamma + self.bias
        return self.activation(out)


class ClaudeMLPCifar(nn.Module):
    """Simple CIFAR MLP using editable branchable blocks."""

    def __init__(
        self,
        num_classes: int = 10,
        input_shape: Sequence[int] = (3, 32, 32),
        hidden_dim: int = 256,
        depth: int = 2,
        num_branches: int = 1,
        norm: NormName = "batch",
        branch_scale_init: ScaleInit = "inv_sqrt_n",
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        input_dim = math.prod(int(v) for v in input_shape)
        layers: list[nn.Module] = [nn.Flatten()]
        in_features = input_dim
        for _ in range(depth):
            layers.append(
                ClaudeMLPBlock(
                    in_features,
                    hidden_dim,
                    num_branches=num_branches,
                    norm=norm,
                    branch_scale_init=branch_scale_init,
                )
            )
            in_features = hidden_dim
        layers.append(nn.Linear(hidden_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_lr_boosted_sgd(
    params,
    model: nn.Module,
    lr: float,
    momentum: float = 0.0,
    weight_decay: float = 0.0,
    nesterov: bool = False,
    conv3_weight_mult: float = 1.0,
    gamma_mult: float = 1.0,
    bias_mult: float = 1.0,
    other_mult: float = 1.0,
    **_: object,
) -> torch.optim.Optimizer:
    """SGD with per-parameter-group learning-rate multipliers, used to test the
    rho=1 limit hypothesis: that summed normalized branches are equivalent to a
    single branch with boosted learning rates.

    The branch derivation (centered data, gamma/sigma tied across branches) says
    two summed per-branch-normalized branches act, in the aligned (rho=1) limit,
    like one branch with the *bias* learning rate scaled by N (=2) and the
    *conv weight* learning rate boosted, while the *scale* (gamma) learning rate
    is unchanged. This builder lets a single-branch model mimic that by scaling
    the LR of each parameter class:

    - ``conv3_weight_mult`` -> the 3x3 branch conv weights,
    - ``gamma_mult``        -> norm scale (BN/GN ``weight``),
    - ``bias_mult``         -> norm shift (BN/GN ``bias``),
    - ``other_mult``        -> everything else (fc weight/bias).

    Param groups give *exact* per-group learning rates (correct under momentum),
    so a single branch at ``(conv3_weight_mult, bias_mult)`` can be compared
    cleanly against a real two-branch model at all multipliers = 1.
    """
    conv3_ids: set[int] = set()
    gamma_ids: set[int] = set()
    beta_ids: set[int] = set()
    for module in model.modules():
        if isinstance(module, ClaudeRepVGGBlock):
            for w in module._conv3_weights():
                conv3_ids.add(id(w))
            if module.bn_position == "per_branch":
                norms = [branch[1] for branch in module.conv3_branches]
            else:
                norms = [module.post_norm]
            for nm in norms:
                w = getattr(nm, "weight", None)
                b = getattr(nm, "bias", None)
                if isinstance(w, nn.Parameter):
                    gamma_ids.add(id(w))
                if isinstance(b, nn.Parameter):
                    beta_ids.add(id(b))

    buckets: dict[str, list[nn.Parameter]] = {"conv3": [], "gamma": [], "beta": [], "other": []}
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if id(p) in conv3_ids:
            buckets["conv3"].append(p)
        elif id(p) in gamma_ids:
            buckets["gamma"].append(p)
        elif id(p) in beta_ids:
            buckets["beta"].append(p)
        else:
            buckets["other"].append(p)

    mults = {
        "conv3": conv3_weight_mult,
        "gamma": gamma_mult,
        "beta": bias_mult,
        "other": other_mult,
    }
    groups = [
        {"params": ps, "lr": lr * mults[name]}
        for name, ps in buckets.items()
        if ps
    ]
    return torch.optim.SGD(
        groups, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov
    )


def main() -> None:
    from structural_reparam.deploy.train import main as train_main

    train_main()


if __name__ == "__main__":
    main()
