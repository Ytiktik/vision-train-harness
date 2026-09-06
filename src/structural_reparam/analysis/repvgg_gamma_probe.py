"""Per-epoch BN gamma diagnostics for RepVGG freeze-scale experiments.

Logs mean / abs-min / max of the BN gamma vector for each 3×3 branch in
every RepVGGBlock, keyed as ``repvgg_gamma/S<stage>B<block>_branch<n>_gamma_{mean,min,max}``.
"""

from __future__ import annotations

import torch
from torch import nn

from structural_reparam.analysis.registry import ProbeContext, register_probe
from structural_reparam.base_models.repvgg import RepVGGBlock


@register_probe("repvgg_gamma")
class RepVGGGammaProbe:
    def __init__(self, model: nn.Module) -> None:
        self.blocks: list[tuple[str, RepVGGBlock]] = []
        for name, module in model.named_modules():
            if isinstance(module, RepVGGBlock):
                self.blocks.append((name, module))

    @classmethod
    def from_context(cls, ctx: ProbeContext) -> "RepVGGGammaProbe":
        return cls(model=ctx.model)

    def close(self) -> None:
        return None

    def epoch_stats(self, epoch: int = 0) -> dict[str, float]:
        out: dict[str, float] = {}
        for block_name, block in self.blocks:
            prefix = f"repvgg_gamma/{block_name}"
            for i, branch in enumerate(block.conv3_branches):
                bn: nn.BatchNorm2d = branch[1]
                gamma = bn.weight.detach()
                out[f"{prefix}_branch{i + 1}_gamma_mean"] = gamma.mean().item()
                out[f"{prefix}_branch{i + 1}_gamma_min"] = gamma.abs().min().item()
                out[f"{prefix}_branch{i + 1}_gamma_max"] = gamma.max().item()
            if block.identity_bn is not None:
                gamma = block.identity_bn.weight.detach()
                out[f"{prefix}_identity_gamma_mean"] = gamma.mean().item()
                out[f"{prefix}_identity_gamma_min"] = gamma.abs().min().item()
                out[f"{prefix}_identity_gamma_max"] = gamma.max().item()
        return out
