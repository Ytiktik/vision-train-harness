"""CIFAR-scale RepVGG model used as the baseline research control."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from structural_reparam.base_models.repvgg import RepVGGBlock


DEFAULT_STAGE_CHANNELS = [64, 64, 128, 256, 512]
DEFAULT_STAGE_BLOCKS = [1, 2, 2, 2, 1]
DEFAULT_STAGE_STRIDES = [1, 1, 2, 2, 2]


class CifarRepVGG(nn.Module):
    """Small RepVGG-style classifier for CIFAR experiments."""

    def __init__(
        self,
        num_classes: int = 10,
        width_mult: float = 1.0,
        stage_channels: list[int] | None = None,
        stage_blocks: list[int] | None = None,
        stage_strides: list[int] | None = None,
        use_1x1: bool = True,
        use_identity: bool = True,
        num_3x3: int = 1,
        kernel_size: int = 3,
        init_strategies: Sequence[str] | None = None,
        conv3_branch_norms: Sequence[str] | None = None,
        fixed_filters: Sequence[str] | None = None,
        track_distance: bool = False,
        bn_use_bias: bool = True,
        bn_sum_norm: str = "global_bn",
        freeze_branches: list[int] | None = None,
        frozen_branch_scale: float = 1.0,
        freeze_identity: bool = False,
        frozen_identity_scale: float = 1.0,
        identity_use_bn: bool = True,
        freeze_bn_scale: bool = False,
        branch_init: str = "inv_sqrt_n",
        kaiming_branch_sum_k: int = 1,
        pool: str = "avg",
    ) -> None:
        super().__init__()
        stage_channels = stage_channels or DEFAULT_STAGE_CHANNELS
        stage_blocks = stage_blocks or DEFAULT_STAGE_BLOCKS
        stage_strides = stage_strides or DEFAULT_STAGE_STRIDES

        channels = [max(1, int(c * width_mult)) for c in stage_channels]
        in_channels = 3
        stages = []

        for out_channels, num_blocks, stride in zip(channels, stage_blocks, stage_strides):
            blocks = []
            for block_idx in range(num_blocks):
                block_stride = stride if block_idx == 0 else 1
                block_in = in_channels if block_idx == 0 else out_channels
                blocks.append(
                    RepVGGBlock(
                        block_in,
                        out_channels,
                        stride=block_stride,
                        use_1x1=use_1x1,
                        use_identity=use_identity,
                        num_3x3=num_3x3,
                        kernel_size=kernel_size,
                        init_strategies=init_strategies,
                        conv3_branch_norms=conv3_branch_norms,
                        fixed_filters=fixed_filters,
                        track_distance=track_distance,
                        bn_use_bias=bn_use_bias,
                        bn_sum_norm=bn_sum_norm,
                        freeze_branches=freeze_branches,
                        frozen_branch_scale=frozen_branch_scale,
                        freeze_identity=freeze_identity,
                        frozen_identity_scale=frozen_identity_scale,
                        identity_use_bn=identity_use_bn,
                        freeze_bn_scale=freeze_bn_scale,
                        branch_init=branch_init,
                        kaiming_branch_sum_k=kaiming_branch_sum_k,
                    )
                )
            stages.append(nn.Sequential(*blocks))
            in_channels = out_channels

        self.stages = nn.ModuleList(stages)
        if pool == "avg":
            self.pool: nn.Module = nn.AdaptiveAvgPool2d(1)
        elif pool == "max":
            self.pool = nn.AdaptiveMaxPool2d(1)
        else:
            raise ValueError(f"pool must be 'avg' or 'max', got {pool!r}")
        self.fc = nn.Linear(channels[-1], num_classes)

    def forward(self, x):
        for stage in self.stages:
            x = stage(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)

    def fuse(self) -> "CifarRepVGG":
        self.eval()
        for module in self.modules():
            if isinstance(module, RepVGGBlock):
                module.fuse()
        return self

    def param_count(self) -> int:
        return sum(param.numel() for param in self.parameters())

    def custom_l2(self) -> torch.Tensor:
        """Sum of per-block RepVGG custom L2 penalties (Ding et al. 2021, Sec 4.2)."""
        terms = [
            module.custom_l2()
            for module in self.modules()
            if isinstance(module, RepVGGBlock)
        ]
        if not terms:
            return torch.zeros((), device=self.fc.weight.device)
        return torch.stack(terms).sum()

    def distance_loss(self) -> torch.Tensor:
        """Sum of per-block branch-distance penalties from the most recent forward."""
        terms = [
            module.distance_loss()
            for module in self.modules()
            if isinstance(module, RepVGGBlock) and module.track_distance
        ]
        if not terms:
            return torch.zeros((), device=self.fc.weight.device)
        return torch.stack(terms).sum()
