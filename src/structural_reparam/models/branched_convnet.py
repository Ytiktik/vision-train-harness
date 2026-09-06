"""Two-branch conv net for σ_tot vs σ_avg normalisation comparison."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from structural_reparam.base_models.sum_conv_block import (
    AlphaCorrSigmaConvBlock,
    AvgSigmaConvBlock,
    AvgSigmaSqrt2ConvBlock,
    SharedScaleConvBlock,
    SingleBranchConvBlock,
    SingletonSharedScaleConvBlock,
    SingletonSqrtHalfSumSqSigmaConvBlock,
    SqrtHalfSumSqSigmaConvBlock,
    SumBNConvBlock,
)


DEFAULT_STAGE_CHANNELS = [64, 64, 128, 256, 512]
DEFAULT_STAGE_BLOCKS = [1, 1, 2, 2, 1]
DEFAULT_STAGE_STRIDES = [1, 1, 2, 2, 2]


class BranchedConvNet(nn.Module):
    """CIFAR-scale two-branch conv net with configurable normalisation.

    norm='sum_bn':          BN(conv₁+conv₂) — single BN over the sum.
    norm='avg_sigma':       γ·(centered sum)/σ_avg — mean-sigma normalisation.
    norm='avg_sigma_sqrt2': γ·(centered sum)/(σ_avg·√2) — approx σ_tot when ρ=0.
    norm='shared_scale_bn': γ·(BN_noaffine(h₁)+BN_noaffine(h₂)) — shared scale over per-branch BNs.
    norm='single_branch':   single conv → BN — standard single-branch baseline.
    """

    def __init__(
        self,
        num_classes: int = 10,
        norm: str = "sum_bn",
        stage_channels: list[int] | None = None,
        stage_blocks: list[int] | None = None,
        stage_strides: list[int] | None = None,
        sigma_alpha_corr: float = 0.0,
    ) -> None:
        super().__init__()
        valid_norms = (
            "sum_bn", "avg_sigma", "avg_sigma_sqrt2", "shared_scale_bn",
            "single_branch", "alpha_corr_sigma", "sqrt_half_sum_sq_sigma",
            "two_branch", "singleton_sqrt_half_sum_sq_sigma",
        )
        if norm not in valid_norms:
            raise ValueError(f"norm must be one of {valid_norms}, got {norm!r}")

        stage_channels = stage_channels or DEFAULT_STAGE_CHANNELS
        stage_blocks = stage_blocks or DEFAULT_STAGE_BLOCKS
        stage_strides = stage_strides or DEFAULT_STAGE_STRIDES

        in_channels = 3
        stages: list[nn.Module] = []

        for out_channels, num_blocks, stride in zip(stage_channels, stage_blocks, stage_strides):
            blocks: list[nn.Module] = []
            for block_idx in range(num_blocks):
                block_stride = stride if block_idx == 0 else 1
                block_in = in_channels if block_idx == 0 else out_channels
                if norm == "sum_bn":
                    blocks.append(SumBNConvBlock(block_in, out_channels, block_stride))
                elif norm == "avg_sigma":
                    blocks.append(AvgSigmaConvBlock(block_in, out_channels, block_stride))
                elif norm == "avg_sigma_sqrt2":
                    blocks.append(AvgSigmaSqrt2ConvBlock(block_in, out_channels, block_stride))
                elif norm == "shared_scale_bn":
                    blocks.append(SharedScaleConvBlock(block_in, out_channels, block_stride))
                elif norm == "alpha_corr_sigma":
                    blocks.append(AlphaCorrSigmaConvBlock(block_in, out_channels, block_stride, alpha=sigma_alpha_corr))
                elif norm == "sqrt_half_sum_sq_sigma":
                    blocks.append(SqrtHalfSumSqSigmaConvBlock(block_in, out_channels, block_stride))
                elif norm == "two_branch":
                    blocks.append(SingletonSharedScaleConvBlock(block_in, out_channels, block_stride))
                elif norm == "singleton_sqrt_half_sum_sq_sigma":
                    blocks.append(SingletonSqrtHalfSumSqSigmaConvBlock(block_in, out_channels, block_stride))
                else:
                    blocks.append(SingleBranchConvBlock(block_in, out_channels, block_stride))
            stages.append(nn.Sequential(*blocks))
            in_channels = out_channels

        self.stages = nn.ModuleList(stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(stage_channels[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for stage in self.stages:
            x = stage(x)
        return self.fc(self.pool(x).flatten(1))

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
