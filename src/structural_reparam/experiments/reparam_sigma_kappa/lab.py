"""kappa-strengthened shared-sigma two-branch block.

The shared-scale+sigma pair (reparam_shared_scale Exp2) is exactly

    gamma * (u x - mu_u) / sqrt(sigma_u^2/4 + sigma_d^2) + beta_1 + beta_2,
    u = w_1 + w_2,  d = (w_1 - w_2)/2,  sigma_d = std(d x),

via the per-batch identity (var_1 + var_2)/2 = var_u/4 + var_d. This block
generalizes it with a fixed multiplier on the difference-variance term:

    denom = sqrt(sigma_u^2/4 + kappa * sigma_d^2).

kappa = 1 reproduces the shared_gamma_sigma block EXACTLY. kappa > 1 is the
"strengthening" arm: since the numerator cancels d identically, kappa
multiplies BOTH the forward shrinkage (the meter dose) and d's only gradient
path (the engine dose) -- deliberately UNCOMPENSATED (no 1/kappa LR twin), per
the experiment design. gamma is initialised to num_branches**-0.5 for every
kappa (deliberately NOT renormalized against the kappa-inflated denominator).

Both kernels stay free and trained; only the denominator weighting changes.
Exactly foldable at eval (frozen stats): W_eff = (gamma/c) * (w_1 + w_2),
b_eff = beta_1 + beta_2 - gamma * mu_u / c, with c the per-channel denom.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

# Importing registers the weight_norm probe this experiment's configs enable.
import structural_reparam.experiments.reparam_sigma_gate.lab  # noqa: F401


class KappaSigmaRepVGGBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        kappa: float = 1.0,
        eps: float = 1e-5,
        momentum: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kappa = float(kappa)
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
                for _ in range(2)
            ]
        )
        # Matches SharedScaleRepVGGBlock: gamma init num_branches**-0.5, one
        # beta per branch. NOT rescaled for kappa != 1.
        self.gamma = nn.Parameter(torch.full((out_channels,), 2.0 ** -0.5))
        self.betas = nn.ParameterList(
            [nn.Parameter(torch.zeros(out_channels)) for _ in range(2)]
        )
        self.register_buffer("running_mean_u", torch.zeros(out_channels))
        self.register_buffer("running_var_u", torch.ones(out_channels))
        self.register_buffer("running_var_d", torch.ones(out_channels))
        self.activation = nn.ReLU(inplace=True)
        for conv in self.convs:
            nn.init.kaiming_normal_(conv.weight, mode="fan_in", nonlinearity="relu")

    def _denom(self, var_u: torch.Tensor, var_d: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(var_u / 4.0 + self.kappa * var_d + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a1 = self.convs[0](x)
        a2 = self.convs[1](x)
        u = a1 + a2
        d = 0.5 * (a1 - a2)
        if self.training:
            mu_u = u.mean(dim=(0, 2, 3))
            var_u = u.var(dim=(0, 2, 3), unbiased=False)
            var_d = d.var(dim=(0, 2, 3), unbiased=False)
            with torch.no_grad():
                m = self.momentum
                self.running_mean_u.mul_(1 - m).add_(m * mu_u.detach())
                self.running_var_u.mul_(1 - m).add_(m * var_u.detach())
                self.running_var_d.mul_(1 - m).add_(m * var_d.detach())
        else:
            mu_u, var_u, var_d = self.running_mean_u, self.running_var_u, self.running_var_d
        c = self._denom(var_u, var_d)
        out = self.gamma[None, :, None, None] * (
            u - mu_u[None, :, None, None]
        ) / c[None, :, None, None]
        out = out + (self.betas[0] + self.betas[1])[None, :, None, None]
        return self.activation(out)

    @torch.no_grad()
    def fused_conv_bias(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Folded inference (W_eff, b_eff) from frozen stats."""
        c = self._denom(self.running_var_u, self.running_var_d)
        w_eff = (self.convs[0].weight + self.convs[1].weight) * (self.gamma / c)[:, None, None, None]
        b_eff = self.betas[0] + self.betas[1] - self.gamma * self.running_mean_u / c
        return w_eff, b_eff


class KappaSigmaRepVGGCifar(nn.Module):
    """RepVGG-CIFAR stack of ``KappaSigmaRepVGGBlock``s, same backbone as the
    single/indep_2/shared-scale depth sweeps (stage_channels [64,128,256],
    strides [1,2,2]) so it pairs against them per seed."""

    def __init__(
        self,
        num_classes: int = 100,
        width_mult: float = 1.0,
        stage_channels: Sequence[int] = (64, 128, 256),
        stage_blocks: Sequence[int] = (1, 1, 1),
        stage_strides: Sequence[int] = (1, 2, 2),
        kappa: float = 1.0,
        eps: float = 1e-5,
        **_: object,
    ) -> None:
        super().__init__()
        if not (len(stage_channels) == len(stage_blocks) == len(stage_strides)):
            raise ValueError("stage_channels, stage_blocks, and stage_strides must match")
        self.kappa = float(kappa)
        channels = [max(1, int(c * width_mult)) for c in stage_channels]
        in_channels = 3
        stages: list[nn.Module] = []
        for out_channels, blocks, stride in zip(channels, stage_blocks, stage_strides):
            layers: list[nn.Module] = []
            for block_idx in range(int(blocks)):
                block_stride = int(stride) if block_idx == 0 else 1
                block_in = in_channels if block_idx == 0 else out_channels
                layers.append(
                    KappaSigmaRepVGGBlock(
                        block_in, out_channels, stride=block_stride,
                        kappa=self.kappa, eps=eps,
                    )
                )
            stages.append(nn.Sequential(*layers))
            in_channels = out_channels
        self.stages = nn.ModuleList(stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for stage in self.stages:
            x = stage(x)
        return self.fc(self.pool(x).flatten(1))

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def main() -> None:
    from structural_reparam.deploy.train import main as train_main

    train_main()


if __name__ == "__main__":
    main()
