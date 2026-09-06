"""Center:surround ratio of each RepVGGBlock's fused-equivalent kernel.

Folds every block down to its single deploy-time 3x3 kernel and reports how
center-dominant it is:

  kernel_ratio/block{i}/rho           — center_rms / surround_rms
  kernel_ratio/block{i}/center_rms    — RMS of the center tap over (out, in)
  kernel_ratio/block{i}/surround_rms  — RMS of the surround ring over (out, in, 8)
  kernel_ratio/rho_mean               — mean rho across blocks

A pointwise cue (the colour dot) pushes the optimal kernel toward a large center
tap; watching ``rho`` rise early relative to the surround visualises the
fast-pathway dot-grab that the cue-attribution probe measures behaviourally.
"""

from __future__ import annotations

import torch
from torch import nn

from structural_reparam.analysis.registry import ProbeContext, register_probe
from structural_reparam.base_models.repvgg import RepVGGBlock


@register_probe("kernel_ratio")
class KernelRatioProbe:
    def __init__(self, model: nn.Module) -> None:
        self.blocks: list[tuple[str, RepVGGBlock]] = [
            (name, m) for name, m in model.named_modules() if isinstance(m, RepVGGBlock)
        ]

    @classmethod
    def from_context(cls, ctx: ProbeContext) -> "KernelRatioProbe":
        return cls(model=ctx.model)

    def close(self) -> None:
        return None

    @torch.no_grad()
    def epoch_stats(self, epoch: int = 0) -> dict[str, float]:
        out: dict[str, float] = {}
        rhos: list[float] = []
        for i, (_name, block) in enumerate(self.blocks):
            kernel = block.equivalent_kernel()  # (out, in, kh, kw)
            kh, kw = kernel.shape[-2:]
            cy, cx = kh // 2, kw // 2

            center = kernel[..., cy, cx]
            center_rms = center.pow(2).mean().sqrt().item()

            mask = torch.ones(kh, kw, dtype=torch.bool, device=kernel.device)
            mask[cy, cx] = False
            surround = kernel[..., mask]
            surround_rms = surround.pow(2).mean().sqrt().item()

            rho = center_rms / surround_rms if surround_rms > 0 else float("inf")
            out[f"kernel_ratio/block{i}/rho"] = rho
            out[f"kernel_ratio/block{i}/center_rms"] = center_rms
            out[f"kernel_ratio/block{i}/surround_rms"] = surround_rms
            if rho != float("inf"):
                rhos.append(rho)

        if rhos:
            out["kernel_ratio/rho_mean"] = sum(rhos) / len(rhos)
        return out
