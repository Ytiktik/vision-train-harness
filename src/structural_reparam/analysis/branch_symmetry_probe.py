"""Weight-space branch-symmetry probe.

Logs, per epoch, the largest absolute elementwise gap between branch 0 and every
other branch across all branched layers — parameters AND buffers (conv weights,
BN affine, BN running stats), so a tie that breaks anywhere is caught:

  - ``sym/max_branch_param_gap``: max over layers/branches/tensors of
    ``|theta_i - theta_0|.max()``. Exactly 0.0 means the branches are bitwise
    identical.

Built for degenerate (zeta=0 / ``branch_init_identical``) arms, where identical
init + deterministic training should keep branches tied forever; the probe turns
that "should" into a per-epoch logged verification.

Unlike the mechanistic probe this is a pure weight-space READ: no forward pass,
no data loader, no RNG consumption, no BN running-stat perturbation — enabling
it cannot change the run's dynamics or its eval metrics.
"""

from __future__ import annotations

import torch
from torch import nn

from structural_reparam.analysis.registry import ProbeContext, register_probe


@register_probe("branch_symmetry")
class BranchSymmetryProbe:
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    @classmethod
    def from_context(cls, ctx: ProbeContext) -> "BranchSymmetryProbe":
        return cls(model=ctx.model)

    def epoch_stats(self, epoch: int = 0) -> dict[str, float]:
        gap = 0.0
        with torch.no_grad():
            for m in self.model.modules():
                branches = getattr(m, "conv3_branches", None)
                if branches is None or len(branches) < 2:
                    continue
                tensor_lists = [
                    list(b.parameters()) + list(b.buffers()) for b in branches
                ]
                ref = tensor_lists[0]
                for other in tensor_lists[1:]:
                    for t0, ti in zip(ref, other):
                        if t0.shape != ti.shape:
                            continue
                        gap = max(gap, (ti.float() - t0.float()).abs().max().item())
        return {"sym/max_branch_param_gap": gap}

    def close(self) -> None:
        return None
