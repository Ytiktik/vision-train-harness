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
    "batch_fixedmean",
    "batch_fixedvar",
    "batch_sgmean",
    "batch_sgvar",
    "batch_sgboth",
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


def _make_conv_norm(channels: int, norm: NormName) -> nn.Module:
    if norm == "batch":
        return nn.BatchNorm2d(channels)
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
    ) -> None:
        super().__init__()
        if num_3x3 < 1:
            raise ValueError(f"num_3x3 must be >= 1, got {num_3x3}")
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
