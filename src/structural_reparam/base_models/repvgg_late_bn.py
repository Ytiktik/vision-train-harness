"""RepVGG-style block with a single BatchNorm after summing all branches (no per-branch BN)."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from structural_reparam.base_models.repvgg_common import RepVGGBlockBase, max_fusion_error
from structural_reparam.base_models.init_kernels import (
    fixed_3x3,
    init_3x3,
    is_learnable,
)


class FixedFilterBranchLateBN(nn.Module):
    """Frozen depthwise 3x3 + learned 1x1, no BN (BN lives in the parent block)."""

    def __init__(self, name: str, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        depthwise_kernel = fixed_3x3(name).repeat(in_channels, 1, 1, 1)
        self.register_buffer("depthwise_kernel", depthwise_kernel, persistent=True)

        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(self.pointwise.weight, mode="fan_in", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.conv2d(
            x,
            self.depthwise_kernel,
            bias=None,
            stride=self.stride,
            padding=1,
            groups=self.in_channels,
        )
        return self.pointwise(y)

    def equivalent_kernel(self) -> torch.Tensor:
        pw = self.pointwise.weight.view(self.out_channels, self.in_channels, 1, 1)
        dw = self.depthwise_kernel.view(self.in_channels, 1, 3, 3).squeeze(1)
        return pw * dw.unsqueeze(0)


class RepVGGBlockLateBN(RepVGGBlockBase):
    """RepVGG block where all branches are bare convs summed first, then a single BN is applied."""

    _fuse_delete_attrs = ("conv3_branches", "conv1", "fixed_branches", "bn")
    _fuse_final_bn = True
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        use_1x1: bool = True,
        use_identity: bool = True,
        num_3x3: int = 1,
        init_strategies: Sequence[str] | None = None,
        fixed_filters: Sequence[str] | None = None,
        track_distance: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.use_1x1 = use_1x1
        self.deployed = False

        if num_3x3 < 1:
            raise ValueError(f"num_3x3 must be >= 1, got {num_3x3}")
        if init_strategies is None:
            init_strategies = ["kaiming"] * num_3x3
        if len(init_strategies) != num_3x3:
            raise ValueError(
                f"init_strategies length {len(init_strategies)} does not match num_3x3 {num_3x3}"
            )
        for s in init_strategies:
            if not is_learnable(s):
                raise ValueError(f"Unknown learnable 3x3 init strategy: {s!r}")
        self.init_strategies = list(init_strategies)
        self.num_3x3 = num_3x3

        self.conv3_branches = nn.ModuleList(
            [
                nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
                for _ in range(num_3x3)
            ]
        )
        for branch, strategy in zip(self.conv3_branches, self.init_strategies):
            kernel = init_3x3(strategy, out_channels, in_channels)
            with torch.no_grad():
                branch.weight.copy_(kernel)

        self.conv1 = (
            nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=False)
            if use_1x1
            else None
        )
        if self.conv1 is not None:
            nn.init.kaiming_normal_(self.conv1.weight, mode="fan_in", nonlinearity="relu")

        self.has_identity = use_identity and stride == 1 and in_channels == out_channels

        fixed_filters = list(fixed_filters or [])
        self.fixed_filter_names = fixed_filters
        self.fixed_branches = nn.ModuleDict(
            {
                name: FixedFilterBranchLateBN(name, in_channels, out_channels, stride)
                for name in fixed_filters
            }
        )

        self.bn = nn.BatchNorm2d(out_channels)

        self.track_distance = track_distance and num_3x3 >= 2
        self._cos_sim_sq: torch.Tensor | None = None

        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deployed:
            return self.act(self.fused_conv(x))

        out = self.conv3_branches[0](x)
        first_branch_out = out if self.track_distance else None
        second_branch_out: torch.Tensor | None = None
        for idx, branch in enumerate(self.conv3_branches[1:], start=1):
            extra = branch(x)
            if idx == 1 and self.track_distance:
                second_branch_out = extra
            out = out + extra
        if self.conv1 is not None:
            out = out + self.conv1(x)
        if self.has_identity:
            out = out + x
        for branch in self.fixed_branches.values():
            out = out + branch(x)

        if self.track_distance and first_branch_out is not None and second_branch_out is not None:
            a = first_branch_out.flatten(1)
            b = second_branch_out.flatten(1)
            cos = F.cosine_similarity(a, b, dim=1, eps=1e-8)
            self._cos_sim_sq = (cos**2).mean()
        else:
            self._cos_sim_sq = None

        return self.act(self.bn(out))

    def equivalent_kernel(self) -> torch.Tensor:
        if self.deployed:
            return self.fused_conv.weight

        kernel = self.conv3_branches[0].weight.clone()
        for branch in self.conv3_branches[1:]:
            kernel = kernel + branch.weight
        if self.conv1 is not None:
            kernel = kernel + F.pad(self.conv1.weight, [1, 1, 1, 1])
        if self.has_identity:
            kernel = kernel + self._identity_kernel()
        for branch in self.fixed_branches.values():
            kernel = kernel + branch.equivalent_kernel()
        return kernel

    def _identity_kernel(self) -> torch.Tensor:
        ref = self.conv3_branches[0].weight
        eye = torch.eye(self.in_channels, device=ref.device, dtype=ref.dtype)
        return F.pad(eye.reshape(self.in_channels, self.in_channels, 1, 1), [1, 1, 1, 1])
