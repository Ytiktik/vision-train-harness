"""Custom batch normalisation variants for zero-mean branch sums."""

from __future__ import annotations

import torch
from torch import nn


class SqrtkBias(nn.Module):
    """Fixed 1/sqrt(K) scale + learnable per-channel bias, no normalisation.

    Assumes the input is already zero-mean. The scale is a constant set at
    construction time; only the bias is learned. Fully decoupled gradients
    across branches (no shared statistics).
    """

    def __init__(self, num_features: int, inv_sqrt_k: float) -> None:
        super().__init__()
        self._inv_sqrt_k = inv_sqrt_k
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self._inv_sqrt_k + self.bias.view(1, -1, 1, 1)

    def fold(self, kernel: torch.Tensor, bias: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return kernel * self._inv_sqrt_k, bias * self._inv_sqrt_k + self.bias


class RunningVarNorm(nn.Module):
    """Per-channel affine normalisation by a running variance estimate only.

    Assumes the input is already zero-mean (e.g. sum of centered BN outputs),
    so no mean subtraction is performed. Running variance is updated via EMA
    during training but is detached from the backward graph, giving fully
    decoupled gradients across branches.
    """

    def __init__(self, num_features: int, momentum: float = 0.1, eps: float = 1e-5) -> None:
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            with torch.no_grad():
                batch_var = x.var(dim=[0, 2, 3], unbiased=False)
                self.running_var.mul_(1 - self.momentum).add_(self.momentum * batch_var)
        scale = self.weight / torch.sqrt(self.running_var + self.eps)
        return x * scale.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)

    def fold(self, kernel: torch.Tensor, bias: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale = self.weight / torch.sqrt(self.running_var + self.eps)
        return kernel * scale[:, None, None, None], bias * scale + self.bias



class RunningVarNormNoBias(nn.Module):
    """Per-channel affine normalisation by a running variance estimate only.

    Assumes the input is already zero-mean (e.g. sum of centered BN outputs),
    so no mean subtraction is performed. Running variance is updated via EMA
    during training but is detached from the backward graph, giving fully
    decoupled gradients across branches.
    """

    def __init__(self, num_features: int, momentum: float = 0.1, eps: float = 1e-5) -> None:
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            with torch.no_grad():
                batch_var = x.var(dim=[0, 2, 3], unbiased=False)
                self.running_var.mul_(1 - self.momentum).add_(self.momentum * batch_var)
            scale = self.weight / torch.sqrt(batch_var + self.eps)
        else:
            scale = self.weight / torch.sqrt(self.running_var + self.eps)
        return x * scale.view(1, -1, 1, 1)

    def fold(self, kernel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale = self.weight / torch.sqrt(self.running_var + self.eps)
        return kernel * scale[:, None, None, None]


class ZeroMeanBatchNorm(nn.Module):
    """BatchNorm with mean fixed to zero.

    Assumes the input is already zero-mean (e.g. sum of centered BN outputs),
    so variance is estimated as E[x²] rather than E[(x-μ)²] and no mean is
    subtracted. Variance gradient flows normally (unlike RunningVarNorm),
    so branches remain coupled through the variance cross-term but not the
    mean cross-term. Running variance is updated via EMA for eval/deployment.
    """

    def __init__(self, num_features: int, momentum: float = 0.1, eps: float = 1e-5) -> None:
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # E[x²] == Var[x] when E[x]=0; avoids mean-subtraction cross-terms
            batch_var = x.pow(2).mean(dim=[0, 2, 3])
            with torch.no_grad():
                self.running_var.mul_(1 - self.momentum).add_(self.momentum * batch_var.detach())
        else:
            batch_var = self.running_var
        scale = self.weight / torch.sqrt(batch_var + self.eps)
        return x * scale.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)

    def fold(self, kernel: torch.Tensor, bias: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale = self.weight / torch.sqrt(self.running_var + self.eps)
        return kernel * scale[:, None, None, None], bias * scale + self.bias


class WeightNorm(nn.Module):
    """Per-channel affine scaling by the effective kernel norm.

    Assumes the input has already been centered upstream, e.g. branch-local
    BatchNorm has produced ``w x - w E[x]``.  Unlike BatchNorm variants, the
    denominator is not an activation statistic: it is the L2 norm of the
    effective kernel for each output channel.
    """

    def __init__(self, num_features: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        scale = self._scale(kernel)
        return x * scale.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)

    def fold(self, kernel: torch.Tensor, bias: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale = self._scale(kernel)
        return kernel * scale[:, None, None, None], bias * scale + self.bias

    def _scale(self, kernel: torch.Tensor) -> torch.Tensor:
        kernel_norm = torch.sqrt(kernel.pow(2).sum(dim=[1, 2, 3]) + self.eps)
        return self.weight / kernel_norm


class NoScaleBias(nn.Module):
    """Learnable per-channel bias only, no scale and no normalisation.

    Pure additive transform: y = x + β. Fully decoupled gradients across
    branches (no shared statistics, no scale).
    """

    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.bias.view(1, -1, 1, 1)

    def fold(self, kernel: torch.Tensor, bias: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return kernel, bias + self.bias


class LearnableScaleBias(nn.Module):
    """Learnable per-channel scale and bias, no normalisation statistics.

    Pure affine transform: y = γ·x + β, where both γ and β are learned.
    γ is initialised to the value passed at construction. No mean subtraction,
    no variance estimation — fully decoupled gradients.
    """

    def __init__(self, num_features: int, inv_sqrt_k: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.full((num_features,), inv_sqrt_k))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)

    def fold(self, kernel: torch.Tensor, bias: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return kernel * self.weight[:, None, None, None], bias * self.weight + self.bias


def fold_bn(
    bn: nn.BatchNorm2d,
    kernel: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fold a BatchNorm2d into a preceding conv kernel and bias."""
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    return kernel * scale[:, None, None, None], scale * (bias - bn.running_mean) + bn.bias
