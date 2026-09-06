"""RepVGG-style block: per-branch BN-style normalisation (mean+var) + scalar scale + bias.

Like BatchNorm, each branch is normalised by the current batch mean and std during training:
    out = Σᵢ γᵢ · (convᵢ(x) - μᵢ) / σᵢ + b

μᵢ and σᵢ are computed from the current batch with no gradient.
Running mean/var EMAs are maintained for inference and for folding into the fused conv.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from structural_reparam.base_models.custom_batchnorm import RunningVarNormNoBias

from structural_reparam.base_models.repvgg_common import RepVGGBlockBase, max_fusion_error
from structural_reparam.base_models.init_kernels import init_3x3, is_learnable

_EPS = 1e-5
_MOMENTUM = 0.1


class RepVGGBlockScalarBias(RepVGGBlockBase):
    """γᵢ·(convᵢ(x)-μᵢ)/σᵢ per branch, summed, plus a learned per-channel bias. No BN."""

    _fuse_delete_attrs = ("conv3_branches", "branch_scales", "bias")
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        num_3x3: int = 1,
        init_strategies: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.deployed = False

        if num_3x3 < 1:
            raise ValueError(f"num_3x3 must be >= 1, got {num_3x3}")
        if init_strategies is None:
            init_strategies = ["kaiming"] * num_3x3
        if len(init_strategies) != num_3x3:
            raise ValueError(
                f"init_strategies length {len(init_strategies)} != num_3x3 {num_3x3}"
            )
        for s in init_strategies:
            if not is_learnable(s):
                raise ValueError(f"Unknown learnable 3x3 init strategy: {s!r}")
        self.num_3x3 = num_3x3

        self.conv3_branches = nn.ModuleList(
            [
                nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
                RunningVarNormNoBias(out_channels, momentum=_MOMENTUM, eps=_EPS),
                )
                for _ in range(num_3x3)
            ]
        )
        for branch, strategy in zip(self.conv3_branches, init_strategies):
            with torch.no_grad():
                branch[0].weight.copy_(init_3x3(strategy, out_channels, in_channels))

        self.branch_scales = nn.ParameterList(
            [nn.Parameter(torch.ones(1)) for _ in range(num_3x3)]
        )

        # per-channel bias replaces BN affine — initialized to zero
        self.bias = nn.Parameter(torch.zeros(out_channels))

        self.act = nn.ReLU(inplace=True)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deployed:
            return self.act(self.fused_conv(x))

        out = self.branch_scales[0] * self.conv3_branches[0](x)
        for idx, (branch, scale) in enumerate(
            zip(self.conv3_branches[1:], self.branch_scales[1:]), start=1
        ):
            out = out + scale * branch(x)
        out = out + self.bias.reshape(1, -1, 1, 1)
        return self.act(out)

    def equivalent_kernel_and_bias(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Fold per-branch running normalisation into each branch kernel, then sum."""
        if self.deployed:
            return self.fused_conv.weight, self.fused_conv.bias  # type: ignore[return-value]
        kernel = None
        for branch, scale in zip(self.conv3_branches, self.branch_scales):
            conv, norm = branch[0], branch[1]
            k = norm.fold(conv.weight) * scale  # [out_c, in_c, 3, 3]
            kernel = k if kernel is None else kernel + k
        return kernel, self.bias.clone()

