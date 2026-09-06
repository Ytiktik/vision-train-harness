"""RepVGG block: per-branch BN without affine params, single affine BN after sum."""

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


class FixedFilterBranchBNNoAffine(nn.Module):
    """Frozen depthwise 3x3 + learned 1x1 + BN(affine=False)."""

    def __init__(self, name: str, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        depthwise_kernel = fixed_3x3(name).repeat(in_channels, 1, 1, 1)
        self.register_buffer("depthwise_kernel", depthwise_kernel, persistent=True)

        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(self.pointwise.weight, mode="fan_in", nonlinearity="relu")

        self.bn = nn.BatchNorm2d(out_channels, affine=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.conv2d(
            x,
            self.depthwise_kernel,
            bias=None,
            stride=self.stride,
            padding=1,
            groups=self.in_channels,
        )
        return self.bn(self.pointwise(y))

    def equivalent_kernel_bias(self) -> tuple[torch.Tensor, torch.Tensor]:
        pw = self.pointwise.weight.view(self.out_channels, self.in_channels, 1, 1)
        dw = self.depthwise_kernel.view(self.in_channels, 1, 3, 3).squeeze(1)
        kernel = pw * dw.unsqueeze(0)
        scale = 1.0 / torch.sqrt(self.bn.running_var + self.bn.eps)
        return kernel * scale[:, None, None, None], -self.bn.running_mean * scale


class RepVGGBlockBNNoAffine(RepVGGBlockBase):
    """Per-branch BN(affine=False) normalises each branch; one affine BN follows the sum.

    Isolates whether the *learnable affine params* in per-branch BN matter, vs
    just the normalisation itself.
    """

    _fuse_delete_attrs = ("conv3_branches", "branch_bns", "conv1", "conv1_bn", "identity_bn", "fixed_branches", "bn")
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

        self.branch_bns = nn.ModuleList(
            [nn.BatchNorm2d(out_channels, affine=False) for _ in range(num_3x3)]
        )

        if use_1x1:
            self.conv1 = nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=False)
            nn.init.kaiming_normal_(self.conv1.weight, mode="fan_in", nonlinearity="relu")
            self.conv1_bn = nn.BatchNorm2d(out_channels, affine=False)
        else:
            self.conv1 = None
            self.conv1_bn = None

        self.has_identity = use_identity and stride == 1 and in_channels == out_channels
        self.identity_bn = nn.BatchNorm2d(out_channels, affine=False) if self.has_identity else None

        fixed_filters = list(fixed_filters or [])
        self.fixed_filter_names = fixed_filters
        self.fixed_branches = nn.ModuleDict(
            {
                name: FixedFilterBranchBNNoAffine(name, in_channels, out_channels, stride)
                for name in fixed_filters
            }
        )

        # single affine BN after summing all normalised branches
        self.bn = nn.BatchNorm2d(out_channels)

        self.track_distance = track_distance and num_3x3 >= 2
        self._cos_sim_sq: torch.Tensor | None = None

        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deployed:
            return self.act(self.fused_conv(x))

        out = self.branch_bns[0](self.conv3_branches[0](x))
        first_branch_out = out if self.track_distance else None
        second_branch_out: torch.Tensor | None = None
        for idx, (branch, bn) in enumerate(
            zip(self.conv3_branches[1:], self.branch_bns[1:]), start=1
        ):
            extra = bn(branch(x))
            if idx == 1 and self.track_distance:
                second_branch_out = extra
            out = out + extra
        if self.conv1 is not None:
            out = out + self.conv1_bn(self.conv1(x))
        if self.has_identity:
            out = out + self.identity_bn(x)
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

    def equivalent_kernel_bias(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Sum of all branch kernels/biases after folding no-affine BN."""
        if self.deployed:
            return self.fused_conv.weight, self.fused_conv.bias

        kernel, bias = self._fold_no_affine_bn(self.conv3_branches[0], self.branch_bns[0])
        for branch, bn in zip(self.conv3_branches[1:], self.branch_bns[1:]):
            k, b = self._fold_no_affine_bn(branch, bn)
            kernel = kernel + k
            bias = bias + b
        if self.conv1 is not None:
            k1, b1 = self._fold_no_affine_bn(self.conv1, self.conv1_bn, pad=True)
            kernel = kernel + k1
            bias = bias + b1
        if self.has_identity:
            ki, bi = self._fold_identity_no_affine_bn()
            kernel = kernel + ki
            bias = bias + bi
        for branch in self.fixed_branches.values():
            kf, bf = branch.equivalent_kernel_bias()
            kernel = kernel + kf
            bias = bias + bf
        return kernel, bias

    def _fold_no_affine_bn(
        self,
        conv: nn.Conv2d,
        bn: nn.BatchNorm2d,
        pad: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scale = 1.0 / torch.sqrt(bn.running_var + bn.eps)
        kernel = conv.weight if not pad else F.pad(conv.weight, [1, 1, 1, 1])
        return kernel * scale[:, None, None, None], -bn.running_mean * scale

    def _fold_identity_no_affine_bn(self) -> tuple[torch.Tensor, torch.Tensor]:
        bn = self.identity_bn
        scale = 1.0 / torch.sqrt(bn.running_var + bn.eps)
        eye = torch.eye(self.in_channels, device=bn.running_var.device, dtype=bn.running_var.dtype)
        kernel = F.pad(eye.reshape(self.in_channels, self.in_channels, 1, 1), [1, 1, 1, 1])
        return kernel * scale[:, None, None, None], -bn.running_mean * scale
