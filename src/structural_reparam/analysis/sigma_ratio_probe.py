"""Per-epoch σ_tot / σ_avg ratio probe for centered-sigma denominator experiments.

Reads running buffers from CenteredSigmaTotBranchLinear,
CenteredAvgSigmaBranchLinear, and BridgedSigmaBranchLinear — no extra
forward pass required.

Logs per layer (zero-indexed):
  sigma_ratio/layerN/sigma_tot   — mean running σ_tot across features
  sigma_ratio/layerN/sigma_avg   — mean running (σ₁+σ₂)/2 across features
  sigma_ratio/layerN/ratio       — σ_tot / σ_avg  (≈ √(2(1+ρ)) per theory)
  sigma_ratio/layerN/sigma_eff   — (BridgedSigma only) mean σ_eff used as denominator
"""

from __future__ import annotations

import torch
from torch import nn

from structural_reparam.analysis.registry import ProbeContext, register_probe
from structural_reparam.models.mlp import (
    BridgedSigmaBranchLinear,
    CenteredAvgSigmaBranchLinear,
    CenteredSigmaTotBranchLinear,
)

_TRACKED_TYPES = (CenteredSigmaTotBranchLinear, CenteredAvgSigmaBranchLinear, BridgedSigmaBranchLinear)


@register_probe("sigma_ratio")
class SigmaRatioProbe:
    def __init__(self, model: nn.Module) -> None:
        self.layers: list[nn.Module] = [
            m for m in model.modules() if isinstance(m, _TRACKED_TYPES)
        ]

    @classmethod
    def from_context(cls, ctx: ProbeContext) -> "SigmaRatioProbe":
        return cls(model=ctx.model)

    def epoch_stats(self, epoch: int = 0) -> dict[str, float]:
        stats: dict[str, float] = {}
        for i, layer in enumerate(self.layers):
            prefix = f"sigma_ratio/layer{i}"
            sigma_tot = layer.running_sigma_tot
            sigma_avg = (layer.running_sigma1 + layer.running_sigma2) * 0.5

            ratio = sigma_tot / sigma_avg.clamp(min=1e-8)
            stats[f"{prefix}/sigma_tot"] = sigma_tot.mean().item()
            stats[f"{prefix}/sigma_avg"] = sigma_avg.mean().item()
            stats[f"{prefix}/ratio"] = ratio.mean().item()

            if isinstance(layer, BridgedSigmaBranchLinear):
                sigma_eff = (
                    sigma_tot.pow(2) - layer.beta * (sigma_tot.pow(2) - sigma_avg.pow(2))
                ).clamp(min=1e-10).sqrt()
                stats[f"{prefix}/sigma_eff"] = sigma_eff.mean().item()
                stats[f"{prefix}/beta"] = layer.beta

        return stats

    def close(self) -> None:
        return None
