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


NormName = Literal[
    "batch",
    "batch_nobias",
    "batch_fixedmean",
    "batch_fixedvar",
    "batch_sgmean",
    "batch_sgvar",
    "batch_sgboth",
    "mean_only",
    "layer",
    "none",
]
ScaleInit = Literal["inv_sqrt_n", "ones"]
BNPosition = Literal["per_branch", "post_sum"]
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


class BatchNormGammaOnly2d(nn.Module):
    """Standard BatchNorm2d (batch mean/var, EMA buffers, learnable per-channel
    ``gamma``) but with **no learnable beta** — the per-branch bias is dropped.

        train:  out = gamma * (x - batch_mean) / sqrt(batch_var + eps)
        eval:   out = gamma * (x - running_mean) / sqrt(running_var + eps)

    This is the single-bias campaign's lever. The published ``indep_2`` two-branch
    block summed two full ``nn.BatchNorm2d`` branches, so it carried two learnable
    biases ``beta_1 + beta_2`` (paper eq with per-branch ``b_i``). Both betas see
    the same gradient, so summing them hands the bias an N× effective-LR boost the
    paper's Methodology claims to have dropped. Replacing each branch's BN with
    this gamma-only variant and adding ONE shared bias after the branch sum (see
    ``ClaudeRepVGGBlock(single_bias=True)``) realizes the paper's intended single-
    bias form ``sum_i gamma_i (w_i x - mu_i)/sigma_i + b``. Still exactly foldable:
    each branch folds to a conv with a per-channel constant offset, and those
    offsets plus the shared bias collapse to one inference bias.

    A ``bias`` buffer of zeros is registered (not a Parameter) so BN-style folding
    tooling that reads ``.bias`` does not choke.
    """

    def __init__(self, channels: int, eps: float = 1e-5, momentum: float = 0.1) -> None:
        super().__init__()
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.weight = nn.Parameter(torch.ones(channels))
        self.register_buffer("bias", torch.zeros(channels))
        self.register_buffer("running_mean", torch.zeros(channels))
        self.register_buffer("running_var", torch.ones(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            batch_mean = x.mean(dim=(0, 2, 3))
            batch_var = x.var(dim=(0, 2, 3), unbiased=False)
            with torch.no_grad():
                m = self.momentum
                self.running_mean.mul_(1.0 - m).add_(m * batch_mean)
                self.running_var.mul_(1.0 - m).add_(m * batch_var)
            mean, var = batch_mean, batch_var
        else:
            mean, var = self.running_mean, self.running_var
        xhat = (x - mean[None, :, None, None]) / torch.sqrt(var[None, :, None, None] + self.eps)
        return xhat * self.weight[None, :, None, None]


class BranchStats2d(nn.Module):
    """Affine-free per-branch normalization stats for the shared-scale blocks.

    Computes and EXPOSES the per-channel mean ``mu`` and variance ``var`` of its
    input (batch stats with gradient in train, frozen running stats in eval),
    updates EMA buffers, and returns the standardized input
    ``(x - mu)/sqrt(var + eps)``. It carries NO learnable affine -- the *shared*
    scale ``gamma`` and the per-branch biases live in the owning block. The last
    per-channel ``mu`` and ``var`` are stashed on ``.last_mu`` / ``.last_var`` so
    the block can build the shared-sigma combination (Exp2) without recomputing
    stats. Added to the chan_hetero probe's ``_NORM_TYPES`` so the probe hooks it
    and measures the per-channel sigma of the conv output the divide acts on.
    """

    def __init__(self, channels: int, eps: float = 1e-5, momentum: float = 0.1) -> None:
        super().__init__()
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.register_buffer("running_mean", torch.zeros(channels))
        self.register_buffer("running_var", torch.ones(channels))
        # filled each forward (per-channel vectors); used by the owning block
        self.last_mu: torch.Tensor | None = None
        self.last_var: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            mu = x.mean(dim=(0, 2, 3))
            var = x.var(dim=(0, 2, 3), unbiased=False)
            with torch.no_grad():
                m = self.momentum
                self.running_mean.mul_(1.0 - m).add_(m * mu)
                self.running_var.mul_(1.0 - m).add_(m * var)
        else:
            mu = self.running_mean
            var = self.running_var
        self.last_mu = mu
        self.last_var = var
        return (x - mu[None, :, None, None]) / torch.sqrt(var[None, :, None, None] + self.eps)


SharedScaleMode = Literal["shared_gamma", "shared_gamma_sigma"]


class SharedScaleRepVGGBlock(nn.Module):
    """Two-branch (N-branch) RepVGG block that collapses the per-branch BN scales
    ``gamma_i`` to a SINGLE shared per-channel ``gamma`` (the paper's
    ``gamma_1 = gamma_2`` simplification, Reparam-2 lines 61-69), while KEEPING the
    two per-branch biases ``beta_i`` (so the only change from the published
    ``indep_2`` block is the scale sharing).

    ``mode='shared_gamma'`` (Exp1, shared scale, per-branch sigma)::

        gamma * ( (w_1 x - mu_1)/sigma_1 + (w_2 x - mu_2)/sigma_2 ) + beta_1 + beta_2

    ``mode='shared_gamma_sigma'`` (Exp2, shared scale AND shared sigma)::

        gamma * ( (w_1 x - mu_1) + (w_2 x - mu_2) ) / sqrt((sigma_1^2 + sigma_2^2)/2)
          + beta_1 + beta_2

    The shared sigma is the RMS of the per-branch stds (matches the validated
    reference ``base_models.sum_conv_block.SqrtHalfSumSqSigmaConvBlock``; equals the
    common sigma exactly when sigma_1 == sigma_2). Both modes are EXACTLY foldable
    at eval (mu_i, sigma_i frozen): the block collapses to one 3x3 conv + one bias
    (verified by ``fused_conv_bias`` at scaffold time). ``gamma`` is initialised to
    ``num_branches ** -0.5`` so the output variance ~ 1 at init for uncorrelated
    branches -- matching ``indep_2``'s ``inv_sqrt_n`` per-branch-gamma init.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        num_branches: int = 2,
        mode: SharedScaleMode = "shared_gamma",
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if num_branches < 1:
            raise ValueError(f"num_branches must be >= 1, got {num_branches}")
        if mode not in ("shared_gamma", "shared_gamma_sigma"):
            raise ValueError(f"bad mode {mode!r}")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.num_branches = num_branches
        self.mode = mode
        self.eps = float(eps)
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
                for _ in range(num_branches)
            ]
        )
        self.stats = nn.ModuleList(
            [BranchStats2d(out_channels, eps=eps) for _ in range(num_branches)]
        )
        self.gamma = nn.Parameter(torch.full((out_channels,), float(num_branches) ** -0.5))
        # KEEP one bias per branch (matches indep_2's two betas, the user's choice).
        self.betas = nn.ParameterList(
            [nn.Parameter(torch.zeros(out_channels)) for _ in range(num_branches)]
        )
        self.activation = nn.ReLU(inplace=True)
        for conv in self.convs:
            nn.init.kaiming_normal_(conv.weight, mode="fan_in", nonlinearity="relu")

    def _bias_sum(self) -> torch.Tensor:
        b = self.betas[0]
        for beta in self.betas[1:]:
            b = b + beta
        return b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "shared_gamma":
            zsum = None
            for conv, st in zip(self.convs, self.stats):
                z = st(conv(x))
                zsum = z if zsum is None else zsum + z
            out = self.gamma[None, :, None, None] * zsum
        else:  # shared_gamma_sigma
            csum = None
            var_sum = None
            for conv, st in zip(self.convs, self.stats):
                a = conv(x)
                st(a)  # updates EMA + stashes last_mu/last_var; fires the probe hook
                mu = st.last_mu
                var = st.last_var
                c = a - mu[None, :, None, None]
                csum = c if csum is None else csum + c
                var_sum = var if var_sum is None else var_sum + var
            sigma_shared = torch.sqrt(
                (var_sum / self.num_branches).clamp(min=self.eps ** 2)
            )
            out = self.gamma[None, :, None, None] * (csum / sigma_shared[None, :, None, None])
        out = out + self._bias_sum()[None, :, None, None]
        return self.activation(out)

    @torch.no_grad()
    def fused_conv_bias(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the folded inference ``(W_eff, bias_eff)`` using frozen stats.

        Only valid after the running stats are populated (call after >=1 train
        forward). ``W_eff`` has shape (out, in, 3, 3); ``bias_eff`` shape (out,).
        """
        g = self.gamma  # (out,)
        beta = self._bias_sum()  # (out,)
        if self.mode == "shared_gamma":
            w_eff = None
            bias = beta.clone()
            for conv, st in zip(self.convs, self.stats):
                s = torch.sqrt(st.running_var + self.eps)  # (out,)
                w_i = conv.weight * (g / s)[:, None, None, None]
                w_eff = w_i if w_eff is None else w_eff + w_i
                bias = bias - g * st.running_mean / s
        else:
            var_sum = None
            mu_sum = None
            w_sum = None
            for conv, st in zip(self.convs, self.stats):
                var_sum = st.running_var if var_sum is None else var_sum + st.running_var
                mu_sum = st.running_mean if mu_sum is None else mu_sum + st.running_mean
                w_sum = conv.weight if w_sum is None else w_sum + conv.weight
            sigma_shared = torch.sqrt((var_sum / self.num_branches).clamp(min=self.eps ** 2))
            w_eff = w_sum * (g / sigma_shared)[:, None, None, None]
            bias = beta - g * mu_sum / sigma_shared
        return w_eff, bias


class SharedScaleRepVGGCifar(nn.Module):
    """RepVGG-CIFAR stack of ``SharedScaleRepVGGBlock``s -- the shared-scale
    analogue of ``LayerwiseRepVGGCifar`` used for the ``indep_2`` / ``single_base``
    depth sweeps, with the SAME backbone (stage_channels [64,128,256], strides
    [1,2,2]) so the shared-scale variants are paired-comparable to the published
    ``indep_2`` gain at matched depth/seed.
    """

    def __init__(
        self,
        num_classes: int = 100,
        width_mult: float = 1.0,
        stage_channels: Sequence[int] = (64, 128, 256),
        stage_blocks: Sequence[int] = (1, 1, 1),
        stage_strides: Sequence[int] = (1, 2, 2),
        num_branches: int = 2,
        mode: SharedScaleMode = "shared_gamma",
        eps: float = 1e-5,
        **_: object,
    ) -> None:
        super().__init__()
        if not (len(stage_channels) == len(stage_blocks) == len(stage_strides)):
            raise ValueError("stage_channels, stage_blocks, and stage_strides must match")
        self.mode = mode
        self.num_branches = int(num_branches)
        channels = [max(1, int(c * width_mult)) for c in stage_channels]
        in_channels = 3
        stages: list[nn.Module] = []
        for out_channels, blocks, stride in zip(channels, stage_blocks, stage_strides):
            layers: list[nn.Module] = []
            for block_idx in range(int(blocks)):
                block_stride = int(stride) if block_idx == 0 else 1
                block_in = in_channels if block_idx == 0 else out_channels
                layers.append(
                    SharedScaleRepVGGBlock(
                        block_in,
                        out_channels,
                        stride=block_stride,
                        num_branches=self.num_branches,
                        mode=mode,
                        eps=eps,
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

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def _make_conv_norm(channels: int, norm: NormName) -> nn.Module:
    if norm == "batch":
        return nn.BatchNorm2d(channels)
    if norm == "batch_nobias":
        return BatchNormGammaOnly2d(channels)
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
        single_bias: bool = False,
    ) -> None:
        super().__init__()
        if num_3x3 < 1:
            raise ValueError(f"num_3x3 must be >= 1, got {num_3x3}")
        if single_bias:
            if bn_position != "per_branch":
                raise ValueError("single_bias is only defined for bn_position='per_branch'")
            if norm != "batch":
                raise ValueError(
                    "single_bias replaces per-branch BN with gamma-only BN, so it "
                    f"requires norm='batch'; got norm={norm!r}"
                )
            # Drop the per-branch beta; the block carries ONE shared bias instead.
            norm = "batch_nobias"
        self.single_bias = bool(single_bias)
        if branch_scale_init not in ("inv_sqrt_n", "ones"):
            raise ValueError(
                "branch_scale_init must be 'inv_sqrt_n' or 'ones', "
                f"got {branch_scale_init!r}"
            )
        if bn_position not in ("per_branch", "post_sum"):
            raise ValueError(
                f"bn_position must be 'per_branch' or 'post_sum', got {bn_position!r}"
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.num_3x3 = num_3x3
        self.use_1x1 = bool(use_1x1)
        self.has_identity = bool(use_identity and stride == 1 and in_channels == out_channels)
        self.bn_position = bn_position
        self.branch_init_identical = bool(branch_init_identical)

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
        # One shared per-channel bias added AFTER the branch sum (paper eq :19),
        # replacing the N per-branch BN betas of the published indep_2 block.
        self.shared_bias = (
            nn.Parameter(torch.zeros(out_channels)) if self.single_bias else None
        )

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
        scale = branch_count ** -0.5 if branch_scale_init == "inv_sqrt_n" else 1.0
        if self.bn_position == "per_branch":
            for branch in self.conv3_branches:
                nn.init.kaiming_normal_(branch[0].weight, mode="fan_in", nonlinearity="relu")
                _set_scale(branch[1], scale)
            if self.conv1 is not None:
                nn.init.kaiming_normal_(self.conv1[0].weight, mode="fan_in", nonlinearity="relu")
                _set_scale(self.conv1[1], scale)
            if self.has_identity:
                _set_scale(self.identity, scale)
        else:
            # No per-branch norm to carry the scale; fold it into the conv weights
            # so the summed init magnitude matches the single-branch case.
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
            if self.shared_bias is not None:
                out = out + self.shared_bias[None, :, None, None]
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
        single_bias: bool = False,
        branch_decay_lambda: float = 0.0,
        base_decay_lambda: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.single_bias = bool(single_bias)
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
                        single_bias=self.single_bias,
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
    """SGD with per-parameter-group LR multipliers — the LR-boost control (b2w4).

    Ported verbatim from the reparam_hetero_bridge lab. The symmetric two-branch
    derivation (paper :70-71) says two summed per-branch-BN branches act, in the
    aligned (rho=1) limit, like ONE branch with the conv-weight LR boosted ~4x and
    the bias/scale LR ~2x. This builder lets a single-branch BN net mimic that:

    - ``conv3_weight_mult`` -> the 3x3 branch conv weights,
    - ``gamma_mult``        -> norm scale (BN ``weight``),
    - ``bias_mult``         -> norm shift (BN ``bias``),
    - ``other_mult``        -> everything else (fc weight/bias).

    Param groups give exact per-group LRs (correct under momentum). For this
    campaign's b2w4 control we use conv3=4, gamma=2, bias=2 — the LR-equivalent
    of the standard two-bias two-branch.
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


# --- chan_hetero probe: measure the per-channel magnitude heterogeneity R ------
from structural_reparam.analysis.registry import ProbeContext, register_probe  # noqa: E402

_NORM_TYPES = (nn.BatchNorm2d, MeanOnlyNorm2d, PartialBatchNorm2d, BatchNormGammaOnly2d, BranchStats2d)


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


GateMask = Literal["identity", "chan_half", "hadamard"]
DenomMode = Literal["quarter", "symmetric"]


class SigmaGateRepVGGBlock(nn.Module):
    """Single-conv block whose normalization denominator is gated by a fixed,
    PARAMETER-FREE sign-antisymmetric transform of the SAME conv weight.

    Motivation: the shared-sigma two-branch block (reparam_shared_scale Exp2)
    decomposes exactly into one signal weight u = W_1 + W_2 and a "phantom"
    difference d = (W_1 - W_2)/2 that contributes NOTHING to the forward signal
    (numerator is pure u*x) and only inflates the sigma-divide:
        out = gamma (u*x - mu_u) / sqrt(sigma_u^2/4 + sigma_d^2) + beta.
    The true d is an equal-magnitude, decorrelated, antisymmetric (w_-) partner
    of u. With a SINGLE conv we cannot store an independent d, so we GENERATE it
    parameter-free as d = s ⊙ u for a fixed balanced ±1 sign mask over the conv
    weight's input-tap dims (the w-antisymmetric image of u, decorrelated from u
    when the mask is balanced, and co-moving with u as u trains).

    sigma_d = std(d*x) enters ONLY the denominator; the signal path stays a
    single conv u. Trainable params == a single conv + BN affine (gamma, beta) ==
    `single_base`. The sign mask is a buffer (0 params). EXACTLY foldable at eval
    (sigma_u, sigma_d, mu_u frozen): W_eff = (gamma/c)·u, b_eff = beta - gamma·mu_u/c,
    with c = sqrt(sigma_u^2/4 + sigma_d^2) a per-output-channel constant -- so the
    gate vanishes into a per-channel rescale of the single inference kernel.

    Masks (s over the (in, kh, kw) tap dims, broadcast across out channels):
      identity : s ≡ +1 → d = u → sigma_d = sigma_u → constant rescale. NULL CONTROL
                 (must reproduce single_base up to a fixed scale BN absorbs).
      chan_half: +1 on the first half of input channels, −1 on the second →
                 d*x = (group-A contribution) − (group-B contribution); the gate
                 is the cross-group disagreement, sigma_d^2 − sigma_u^2 = −4·cov(A,B).
                 The faithful single-conv echo of "branch disagreement". HEADLINE.
      hadamard : balanced alternating ±1 over the flattened taps (maximally
                 decorrelated, full-input).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        gate_mask: GateMask = "chan_half",
        denom: "DenomMode" = "quarter",
        eps: float = 1e-5,
        momentum: float = 0.1,
    ) -> None:
        super().__init__()
        if gate_mask not in ("identity", "chan_half", "hadamard"):
            raise ValueError(f"bad gate_mask {gate_mask!r}")
        if denom not in ("quarter", "symmetric"):
            raise ValueError(f"bad denom {denom!r}")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.gate_mask = gate_mask
        # denom: how signal/gate variances combine into the divide.
        #   quarter   -> sqrt(sigma_u^2/4 + sigma_d^2)   (exact reduction of Exp2;
        #                u is the SUM, each branch ~ u/2, so the 1/4)
        #   symmetric -> sqrt((sigma_u^2 + sigma_d^2)/2) (RMS of signal & gate paths,
        #                Exp2's branch-RMS form with (u,d) as the two "branches")
        self.denom = denom
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        # gamma init targets output variance ~ 1 at init (sigma_d ~ sigma_u):
        # quarter c ~ sqrt(1.25) sigma_u -> gamma = sqrt(1.25); symmetric c ~ sigma_u -> gamma = 1.
        gamma0 = 1.25 ** 0.5 if denom == "quarter" else 1.0
        self.gamma = nn.Parameter(torch.full((out_channels,), gamma0))
        self.beta = nn.Parameter(torch.zeros(out_channels))
        self.register_buffer("running_mean_u", torch.zeros(out_channels))
        self.register_buffer("running_var_u", torch.ones(out_channels))
        self.register_buffer("running_var_d", torch.ones(out_channels))
        self.activation = nn.ReLU(inplace=True)
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_in", nonlinearity="relu")
        self.register_buffer("sign", self._build_sign())

    def _build_sign(self) -> torch.Tensor:
        out, cin, kh, kw = self.conv.weight.shape
        s = torch.ones(out, cin, kh, kw)
        if self.gate_mask == "identity":
            return s
        if self.gate_mask == "chan_half":
            half = cin // 2
            if half >= 1:
                s[:, half:, :, :] = -1.0
            return s
        # hadamard: balanced alternating ±1 over flattened (in, kh, kw), shared across out
        L = cin * kh * kw
        pat = torch.ones(L)
        pat[1::2] = -1.0
        return pat.view(1, cin, kh, kw).expand(out, cin, kh, kw).contiguous()

    @staticmethod
    def _stats(y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return y.mean(dim=(0, 2, 3)), y.var(dim=(0, 2, 3), unbiased=False)

    def _denom(self, var_u: torch.Tensor, var_d: torch.Tensor) -> torch.Tensor:
        if self.denom == "quarter":
            return torch.sqrt(var_u / 4.0 + var_d + self.eps)
        return torch.sqrt((var_u + var_d) / 2.0 + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u_out = self.conv(x)  # signal path (folds)
        d_out = nn.functional.conv2d(
            x, self.sign * self.conv.weight, None, self.stride, 1
        )  # parameter-free gate: d*x = s ⊙ u applied to x
        if self.training:
            mu_u, var_u = self._stats(u_out)
            _, var_d = self._stats(d_out)
            with torch.no_grad():
                m = self.momentum
                self.running_mean_u.mul_(1 - m).add_(m * mu_u.detach())
                self.running_var_u.mul_(1 - m).add_(m * var_u.detach())
                self.running_var_d.mul_(1 - m).add_(m * var_d.detach())
        else:
            mu_u, var_u, var_d = self.running_mean_u, self.running_var_u, self.running_var_d
        c = self._denom(var_u, var_d)
        out = self.gamma[None, :, None, None] * (
            u_out - mu_u[None, :, None, None]
        ) / c[None, :, None, None] + self.beta[None, :, None, None]
        return self.activation(out)

    @torch.no_grad()
    def fused_conv_bias(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Folded inference (W_eff, b_eff) from frozen stats (valid after >=1 train fwd)."""
        c = self._denom(self.running_var_u, self.running_var_d)
        w_eff = self.conv.weight * (self.gamma / c)[:, None, None, None]
        b_eff = self.beta - self.gamma * self.running_mean_u / c
        return w_eff, b_eff


class SigmaGateRepVGGCifar(nn.Module):
    """RepVGG-CIFAR stack of single-conv ``SigmaGateRepVGGBlock``s, same backbone as
    the indep_2/single_base/shared_scale depth sweeps (stage_channels [64,128,256],
    strides [1,2,2]) so it pairs against them. Trainable params == single_base."""

    def __init__(
        self,
        num_classes: int = 100,
        width_mult: float = 1.0,
        stage_channels: Sequence[int] = (64, 128, 256),
        stage_blocks: Sequence[int] = (1, 1, 1),
        stage_strides: Sequence[int] = (1, 2, 2),
        gate_mask: GateMask = "chan_half",
        denom: DenomMode = "quarter",
        eps: float = 1e-5,
        **_: object,
    ) -> None:
        super().__init__()
        if not (len(stage_channels) == len(stage_blocks) == len(stage_strides)):
            raise ValueError("stage_channels, stage_blocks, and stage_strides must match")
        self.gate_mask = gate_mask
        self.denom = denom
        channels = [max(1, int(c * width_mult)) for c in stage_channels]
        in_channels = 3
        stages: list[nn.Module] = []
        for out_channels, blocks, stride in zip(channels, stage_blocks, stage_strides):
            layers: list[nn.Module] = []
            for block_idx in range(int(blocks)):
                block_stride = int(stride) if block_idx == 0 else 1
                block_in = in_channels if block_idx == 0 else out_channels
                layers.append(
                    SigmaGateRepVGGBlock(
                        block_in, out_channels, stride=block_stride,
                        gate_mask=gate_mask, denom=denom, eps=eps,
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

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def main() -> None:
    from structural_reparam.deploy.train import main as train_main

    train_main()


if __name__ == "__main__":
    main()
