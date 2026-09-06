"""Fusion-equivalence gate for the `indep2` arm of the stage1 campaign.

The directive requires that the fused inference output match the training-mode
output to numerical tolerance for every pair arm. That is a property of the model
class and its wiring, not of a particular training run, so it is checked once here
rather than paid for on every run.

For ``bn_position: per_branch`` each branch is a bias-free 3x3 convolution
followed by its own BatchNorm, and the block sums the normalized outputs. In eval
mode BatchNorm is affine, so the whole sum folds into a single convolution:

    W = Σ_i (γ_i / σ_i) · W_i
    b = Σ_i (β_i − γ_i · μ_i / σ_i)

with σ_i = sqrt(running_var_i + eps). This script builds a model, runs a few
training batches so the running statistics are away from their initial values —
otherwise the fold would be tested against an identity and prove nothing — then
compares the summed branch output against the single folded convolution.

Usage:  python -m structural_reparam.agents.stage1_riga.check_fusion
"""

from __future__ import annotations

import torch
import torch.nn as nn

from structural_reparam.experiments.reparam_sweeps100.lab import LayerwiseRepVGGCifar

TOLERANCE = 1e-5


def main() -> int:
    torch.manual_seed(0)
    model = LayerwiseRepVGGCifar(
        num_classes=100, width_mult=1.0, stage_channels=[32, 64, 128],
        stage_blocks=[1, 1, 1], stage_strides=[1, 2, 2], num_3x3=2,
        bn_position="per_branch", branch_scale_init="inv_sqrt_n",
        branch_init_identical=False,
    )
    model.train()
    for _ in range(3):
        model(torch.randn(16, 3, 32, 32))
    model.eval()

    blocks = [m for m in model.modules() if type(m).__name__ == "ClaudeRepVGGBlock"]
    worst = 0.0
    for k, blk in enumerate(blocks):
        convs, bns = [], []
        for branch in blk.conv3_branches:
            convs.append([m for m in branch.modules() if isinstance(m, nn.Conv2d)][0])
            bns.append([m for m in branch.modules() if isinstance(m, nn.BatchNorm2d)][0])

        weight = torch.zeros_like(convs[0].weight)
        bias = torch.zeros(convs[0].out_channels)
        for conv, bn in zip(convs, bns):
            scale = bn.weight / (bn.running_var + bn.eps).sqrt()
            weight = weight + conv.weight * scale.view(-1, 1, 1, 1)
            conv_bias = conv.bias if conv.bias is not None else torch.zeros_like(bias)
            bias = bias + bn.bias + scale * (conv_bias - bn.running_mean)

        x = torch.randn(4, convs[0].in_channels, 16, 16)
        with torch.no_grad():
            reference = sum(bn(conv(x)) for conv, bn in zip(convs, bns))
            fused = nn.functional.conv2d(
                x, weight, bias, stride=convs[0].stride, padding=convs[0].padding
            )
        err = (reference - fused).abs().max().item() / reference.abs().max().item()
        worst = max(worst, err)
        print(f"  block {k}: max relative error {err:.3e}")

    ok = worst < TOLERANCE
    print(f"\nFusion equivalence: worst relative error {worst:.3e}, tolerance "
          f"{TOLERANCE:g} -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
