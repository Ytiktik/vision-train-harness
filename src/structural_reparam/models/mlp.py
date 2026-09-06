"""Simple MLP baseline with configurable width, depth, and branch count."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class ZeroMeanBatchNorm1d(nn.BatchNorm1d):
    """BatchNorm1d that does NOT subtract any mean — only variance-normalizes.

    Output: γ · x / sqrt(Var(x) + eps) + β. running_mean is held at 0 (unused
    in forward); running_var tracks variance as usual. Subclasses BatchNorm1d
    so isinstance checks for nn.BatchNorm1d still succeed.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            dims = (0,) if x.dim() == 2 else (0, 2)
            var = x.var(dim=dims, unbiased=False)
            if self.track_running_stats and self.running_var is not None:
                with torch.no_grad():
                    if self.num_batches_tracked is not None:
                        self.num_batches_tracked.add_(1)
                    m = self.momentum if self.momentum is not None else (
                        1.0 / float(self.num_batches_tracked.item())
                    )
                    self.running_var.mul_(1 - m).add_(var.detach() * m)
        else:
            var = self.running_var if self.running_var is not None else x.var(
                dim=(0,) if x.dim() == 2 else (0, 2), unbiased=False
            )
        inv_std = (var + self.eps).rsqrt()
        if x.dim() == 3:
            inv_std = inv_std.unsqueeze(-1)
        out = x * inv_std
        if self.weight is not None:
            w = self.weight.unsqueeze(-1) if x.dim() == 3 else self.weight
            out = out * w
        if self.bias is not None:
            b = self.bias.unsqueeze(-1) if x.dim() == 3 else self.bias
            out = out + b
        return out


class WeightNorm1d(nn.Module):
    """Affine scaling by each linear row norm instead of activation variance.

    Receives a reference to the paired Linear layer's weight at construction so
    forward only needs the activations — same interface as BatchNorm1d.
    """

    def __init__(self, num_features: int, linear_weight: torch.Tensor, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self._linear_weight = linear_weight  # reference to paired Linear.weight
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.weight / torch.sqrt(self._linear_weight.pow(2).sum(dim=1) + self.eps)
        return x * scale + self.bias


class MeanWeightNorm1d(nn.Module):
    """Batch mean subtraction + weight-norm scaling, no variance normalisation.

    Output: γ · (y − μ) / ||w|| + β  (the missing cell: subtract μ, divide by ||w||).
    Uses batch mean during training and a running EMA at eval, same as BatchNorm1d.
    """

    def __init__(self, num_features: int, linear_weight: torch.Tensor,
                 momentum: float = 0.1, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self._linear_weight = linear_weight
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            mu = x.mean(dim=0)
            with torch.no_grad():
                self.num_batches_tracked += 1
                self.running_mean.mul_(1 - self.momentum).add_(mu.detach() * self.momentum)
        else:
            mu = self.running_mean
        row_norms = torch.sqrt(self._linear_weight.pow(2).sum(dim=1) + self.eps)
        return (x - mu) * (self.weight / row_norms) + self.bias


class SharedScaleNBranchLinear(nn.Module):
    """N BatchNorm branches sharing a single per-channel scale γ.

    Each branch has its own Linear + BatchNorm(affine=False) so each branch
    maintains independent μ/σ statistics. A single γ then scales the sum:

        Output: γ · Σᵢ BN_norm(wᵢx) + bias

    γ is initialised to 1/√N so output variance matches a standard single-branch
    BN layer at init (BN-normalised branches are ~uncorrelated at random init).
    """

    def __init__(self, in_features: int, out_features: int,
                 num_branches: int = 1, norm: str = "batch") -> None:
        super().__init__()
        if norm in ("weight_norm", "mean_weight_norm"):
            raise ValueError(
                f"norm={norm!r} is incompatible with SharedScaleNBranchLinear: "
                "per-branch weight-norm scaling conflicts with a shared γ."
            )

        def _make_branch_norm():
            if norm == "batch_zeromean":
                return ZeroMeanBatchNorm1d(out_features, affine=False)
            if norm == "layer":
                return nn.LayerNorm(out_features, elementwise_affine=False)
            if norm == "batch":
                return nn.BatchNorm1d(out_features, affine=False)
            raise ValueError(f"Unsupported norm {norm!r} for SharedScaleNBranchLinear.")

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, out_features, bias=False),
                _make_branch_norm(),
            )
            for _ in range(num_branches)
        ])
        self.gamma = nn.Parameter(torch.full((out_features,), num_branches ** -0.5))
        self.bias = nn.Parameter(torch.zeros(out_features))
        for branch in self.branches:
            nn.init.kaiming_normal_(branch[0].weight, mode="fan_in", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = sum(branch(x) for branch in self.branches)
        return self.gamma * out + self.bias


class SumBatchNormBranchLinear(nn.Module):
    """Two branches summed BEFORE a single BatchNorm.

    Output: γ · BN(w₁x + w₂x) + bias
          = γ · ((w₁+w₂)x − μ_B) / σ_B((w₁+w₂)x) + bias

    Contrast with BranchedLinear(norm='batch') which applies BN per-branch
    then sums. Here the normalisation sees the fused direction directly.
    γ is initialised to 1/√2 to match the two-branch convention.
    """

    def __init__(self, in_features: int, out_features: int,
                 freeze_norm_scale: bool = False, eps: float = 1e-5) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features, bias=False)
        self.linear2 = nn.Linear(in_features, out_features, bias=False)
        self.bn = nn.BatchNorm1d(out_features, eps=eps)
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.kaiming_normal_(self.linear1.weight, mode="fan_in", nonlinearity="relu")
        nn.init.kaiming_normal_(self.linear2.weight, mode="fan_in", nonlinearity="relu")
        nn.init.constant_(self.bn.weight, 2 ** -0.5)
        nn.init.zeros_(self.bn.bias)
        self.bn.bias.requires_grad_(False)
        if freeze_norm_scale:
            self.bn.weight.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.linear1(x) + self.linear2(x)) + self.bias


class InvSigmaSumBranchLinear(nn.Module):
    """Two branches: scale the sum by (1/σ₁ + 1/σ₂).

    Output: γ ⊙ (1/σ₁ + 1/σ₂) ⊙ (w₁x + w₂x) + bias

    σᵢ = std_B(wᵢx) computed per-batch during training; running EMA of
    1/σᵢ used at eval. γ is initialised to 1/√2.
    """

    def __init__(self, in_features: int, out_features: int,
                 freeze_norm_scale: bool = False,
                 momentum: float = 0.1, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.linear1 = nn.Linear(in_features, out_features, bias=False)
        self.linear2 = nn.Linear(in_features, out_features, bias=False)
        self.gamma = nn.Parameter(torch.full((out_features,), 2 ** -0.5))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.register_buffer("running_inv_sigma1", torch.ones(out_features))
        self.register_buffer("running_inv_sigma2", torch.ones(out_features))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))
        nn.init.kaiming_normal_(self.linear1.weight, mode="fan_in", nonlinearity="relu")
        nn.init.kaiming_normal_(self.linear2.weight, mode="fan_in", nonlinearity="relu")
        if freeze_norm_scale:
            self.gamma.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.linear1(x)  # (N, out)
        h2 = self.linear2(x)
        if self.training:
            inv_s1 = h1.std(dim=0, unbiased=False).clamp(min=self.eps).reciprocal()
            inv_s2 = h2.std(dim=0, unbiased=False).clamp(min=self.eps).reciprocal()
            with torch.no_grad():
                self.num_batches_tracked += 1
                m = self.momentum
                self.running_inv_sigma1.mul_(1 - m).add_(inv_s1.detach() * m)
                self.running_inv_sigma2.mul_(1 - m).add_(inv_s2.detach() * m)
        else:
            inv_s1 = self.running_inv_sigma1
            inv_s2 = self.running_inv_sigma2
        return self.gamma * (inv_s1 + inv_s2) * (h1 + h2) + self.bias


class JointSigmaInvBranchLinear(nn.Module):
    """Two branches: center+scale sum by stop-gradient batch stats, scale by 1/σ.

    Output: γ ⊙ (w₁x + w₂x − sg(μ)) / sg(σ) + bias
    Running EMA of mean/var used at eval. γ initialised to 1/√2.
    """

    def __init__(self, in_features: int, out_features: int,
                 freeze_norm_scale: bool = False,
                 momentum: float = 0.1, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.linear1 = nn.Linear(in_features, out_features, bias=False)
        self.linear2 = nn.Linear(in_features, out_features, bias=False)
        self.gamma = nn.Parameter(torch.full((out_features,), 2 ** -0.5))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.register_buffer("running_mean", torch.zeros(out_features))
        self.register_buffer("running_var", torch.ones(out_features))
        nn.init.kaiming_normal_(self.linear1.weight, mode="fan_in", nonlinearity="relu")
        nn.init.kaiming_normal_(self.linear2.weight, mode="fan_in", nonlinearity="relu")
        if freeze_norm_scale:
            self.gamma.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.linear1(x)
        h2 = self.linear2(x)
        self._branch_outs = (h1, h2)
        h = h1 + h2
        if self.training:
            mu = h.mean(dim=0).detach()
            sigma = h.std(dim=0, unbiased=False).clamp(min=self.eps).detach()
            with torch.no_grad():
                self.running_mean.mul_(1 - self.momentum).add_(mu * self.momentum)
                self.running_var.mul_(1 - self.momentum).add_(sigma.pow(2) * self.momentum)
        else:
            mu = self.running_mean
            sigma = self.running_var.clamp(min=self.eps ** 2).sqrt()
        return self.gamma * (h - mu) / sigma + self.bias


class JointSigmaHalfInvBranchLinear(nn.Module):
    """Two branches: center+scale sum by stop-gradient batch stats, scale by 1/(2σ).

    Output: γ ⊙ (w₁x + w₂x − sg(μ)) / (2·sg(σ)) + bias
    Running EMA of mean/var used at eval. γ initialised to 1/√2.
    """

    def __init__(self, in_features: int, out_features: int,
                 freeze_norm_scale: bool = False,
                 momentum: float = 0.1, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.linear1 = nn.Linear(in_features, out_features, bias=False)
        self.linear2 = nn.Linear(in_features, out_features, bias=False)
        self.gamma = nn.Parameter(torch.full((out_features,), 2 ** -0.5))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.register_buffer("running_mean", torch.zeros(out_features))
        self.register_buffer("running_var", torch.ones(out_features))
        nn.init.kaiming_normal_(self.linear1.weight, mode="fan_in", nonlinearity="relu")
        nn.init.kaiming_normal_(self.linear2.weight, mode="fan_in", nonlinearity="relu")
        if freeze_norm_scale:
            self.gamma.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.linear1(x)
        h2 = self.linear2(x)
        self._branch_outs = (h1, h2)
        h = h1 + h2
        if self.training:
            mu = h.mean(dim=0).detach()
            sigma = h.std(dim=0, unbiased=False).clamp(min=self.eps).detach()
            with torch.no_grad():
                self.running_mean.mul_(1 - self.momentum).add_(mu * self.momentum)
                self.running_var.mul_(1 - self.momentum).add_(sigma.pow(2) * self.momentum)
        else:
            mu = self.running_mean
            sigma = self.running_var.clamp(min=self.eps ** 2).sqrt()
        return self.gamma * (h - mu) / (2.0 * sigma) + self.bias


class JointSigmaCorrInvBranchLinear(nn.Module):
    """Two branches: center sum, scale by 1/(σ·(2−ρ)), all stop-gradient.

    Output: γ ⊙ (w₁x + w₂x − sg(μ)) / (sg(σ)·sg(2−ρ)) + bias
    ρ = per-feature corr(w₁x, w₂x). Running EMA of mean/var/rho at eval. γ initialised to 1/√2.
    """

    def __init__(self, in_features: int, out_features: int,
                 freeze_norm_scale: bool = False,
                 momentum: float = 0.1, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.linear1 = nn.Linear(in_features, out_features, bias=False)
        self.linear2 = nn.Linear(in_features, out_features, bias=False)
        self.gamma = nn.Parameter(torch.full((out_features,), 2 ** -0.5))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.register_buffer("running_mean", torch.zeros(out_features))
        self.register_buffer("running_var", torch.ones(out_features))
        self.register_buffer("running_rho", torch.zeros(out_features))
        nn.init.kaiming_normal_(self.linear1.weight, mode="fan_in", nonlinearity="relu")
        nn.init.kaiming_normal_(self.linear2.weight, mode="fan_in", nonlinearity="relu")
        if freeze_norm_scale:
            self.gamma.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.linear1(x)
        h2 = self.linear2(x)
        self._branch_outs = (h1, h2)
        h = h1 + h2
        if self.training:
            mu = h.mean(dim=0).detach()
            sigma = h.std(dim=0, unbiased=False).clamp(min=self.eps).detach()
            s1 = h1.std(dim=0, unbiased=False).clamp(min=self.eps).detach()
            s2 = h2.std(dim=0, unbiased=False).clamp(min=self.eps).detach()
            h1c = (h1 - h1.mean(dim=0, keepdim=True)).detach()
            h2c = (h2 - h2.mean(dim=0, keepdim=True)).detach()
            rho = (h1c * h2c).mean(dim=0) / (s1 * s2).clamp(min=self.eps)
            with torch.no_grad():
                self.running_mean.mul_(1 - self.momentum).add_(mu * self.momentum)
                self.running_var.mul_(1 - self.momentum).add_(sigma.pow(2) * self.momentum)
                self.running_rho.mul_(1 - self.momentum).add_(rho * self.momentum)
        else:
            mu = self.running_mean
            sigma = self.running_var.clamp(min=self.eps ** 2).sqrt()
            rho = self.running_rho
        denom = sigma * (2.0 - rho).clamp(min=self.eps)
        return self.gamma * (h - mu) / denom + self.bias


class SGScaleBranchLinear(nn.Module):
    """Two branches: (hᵢ−μᵢ)/‖wᵢ‖ scaled by sg(‖wᵢ‖_row/σᵢ) per branch.

    Forward: γ ⊙ Σᵢ [ (wᵢx − μᵢ) / ‖wᵢ‖_row · sg(‖wᵢ‖_row/σᵢ) ] + bias

    Gradient flows through (wᵢx − μᵢ)/‖wᵢ‖ (angular + radial via 1/‖w‖ term)
    and through μᵢ. The ‖wᵢ‖/σᵢ scale is fully stop-gradiented.
    At eval: reduces to (h − running_mean) / running_sigma (‖w‖ cancels).
    γ initialised to 1/√2.
    """

    def __init__(self, in_features: int, out_features: int,
                 freeze_norm_scale: bool = False,
                 momentum: float = 0.1, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.linear1 = nn.Linear(in_features, out_features, bias=False)
        self.linear2 = nn.Linear(in_features, out_features, bias=False)
        self.gamma = nn.Parameter(torch.full((out_features,), 2 ** -0.5))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.register_buffer("running_var1", torch.ones(out_features))
        self.register_buffer("running_var2", torch.ones(out_features))
        self.register_buffer("running_mean1", torch.zeros(out_features))
        self.register_buffer("running_mean2", torch.zeros(out_features))
        nn.init.kaiming_normal_(self.linear1.weight, mode="fan_in", nonlinearity="relu")
        nn.init.kaiming_normal_(self.linear2.weight, mode="fan_in", nonlinearity="relu")
        if freeze_norm_scale:
            self.gamma.requires_grad_(False)

    def _branch_forward(
        self,
        h: torch.Tensor,
        linear: nn.Linear,
        running_mean: torch.Tensor,
        running_var: torch.Tensor,
    ) -> torch.Tensor:
        if self.training:
            mu = h.mean(dim=0)
            sigma = h.std(dim=0, unbiased=False).clamp(min=self.eps)
            w_norm = linear.weight.norm(dim=1)
            scale = (w_norm / sigma).detach()
            with torch.no_grad():
                running_mean.mul_(1 - self.momentum).add_(mu.detach() * self.momentum)
                running_var.mul_(1 - self.momentum).add_(sigma.pow(2).detach() * self.momentum)
        else:
            mu = running_mean
            w_norm = linear.weight.norm(dim=1)
            sigma = running_var.clamp(min=self.eps ** 2).sqrt()
            scale = w_norm / sigma
        return (h - mu) / w_norm.clamp(min=self.eps) * scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.linear1(x)
        h2 = self.linear2(x)
        self._branch_outs = (h1, h2)
        out1 = self._branch_forward(h1, self.linear1, self.running_mean1, self.running_var1)
        out2 = self._branch_forward(h2, self.linear2, self.running_mean2, self.running_var2)
        return self.gamma * (out1 + out2) + self.bias


class SingleSigmaBranchLinear(nn.Module):
    """Two branches centered per-branch, both divided by σ₁ = std_B(w₁x).

    Forward: γ ⊙ ((w₁x − μ₁) + (w₂x − μ₂)) / σ₁ + bias

    Full gradients through μ₁, μ₂ and σ₁. Running EMA of means/var₁ at eval.
    γ initialised to 1/√2.
    """

    def __init__(self, in_features: int, out_features: int,
                 freeze_norm_scale: bool = False,
                 momentum: float = 0.1, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.linear1 = nn.Linear(in_features, out_features, bias=False)
        self.linear2 = nn.Linear(in_features, out_features, bias=False)
        self.gamma = nn.Parameter(torch.full((out_features,), 2 ** -0.5))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.register_buffer("running_mean1", torch.zeros(out_features))
        self.register_buffer("running_mean2", torch.zeros(out_features))
        self.register_buffer("running_var1", torch.ones(out_features))
        nn.init.kaiming_normal_(self.linear1.weight, mode="fan_in", nonlinearity="relu")
        nn.init.kaiming_normal_(self.linear2.weight, mode="fan_in", nonlinearity="relu")
        if freeze_norm_scale:
            self.gamma.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.linear1(x)
        h2 = self.linear2(x)
        self._branch_outs = (h1, h2)
        if self.training:
            mu1 = h1.mean(dim=0)
            mu2 = h2.mean(dim=0)
            sigma1 = h1.std(dim=0, unbiased=False).clamp(min=self.eps)
            with torch.no_grad():
                self.running_mean1.mul_(1 - self.momentum).add_(mu1.detach() * self.momentum)
                self.running_mean2.mul_(1 - self.momentum).add_(mu2.detach() * self.momentum)
                self.running_var1.mul_(1 - self.momentum).add_(sigma1.detach().pow(2) * self.momentum)
        else:
            mu1 = self.running_mean1
            mu2 = self.running_mean2
            sigma1 = self.running_var1.clamp(min=self.eps ** 2).sqrt()
        return self.gamma * ((h1 - mu1) + (h2 - mu2)) / sigma1 + self.bias


class AvgSigmaInvBranchLinear(nn.Module):
    """Two branches scaled by 1/σ where σ = (σ₁+σ₂)/2 (stop-gradient).

    Output: γ ⊙ (1/σ) ⊙ (w₁x + w₂x) + bias
    Running EMA of avg_sigma used at eval. γ initialised to 1/√2.
    """

    def __init__(self, in_features: int, out_features: int,
                 freeze_norm_scale: bool = False,
                 momentum: float = 0.1, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.linear1 = nn.Linear(in_features, out_features, bias=False)
        self.linear2 = nn.Linear(in_features, out_features, bias=False)
        self.gamma = nn.Parameter(torch.full((out_features,), 2 ** -0.5))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.register_buffer("running_avg_sigma", torch.ones(out_features))
        nn.init.kaiming_normal_(self.linear1.weight, mode="fan_in", nonlinearity="relu")
        nn.init.kaiming_normal_(self.linear2.weight, mode="fan_in", nonlinearity="relu")
        if freeze_norm_scale:
            self.gamma.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.linear1(x)
        h2 = self.linear2(x)
        self._branch_outs = (h1, h2)
        if self.training:
            s1 = h1.std(dim=0, unbiased=False).clamp(min=self.eps).detach()
            s2 = h2.std(dim=0, unbiased=False).clamp(min=self.eps).detach()
            avg_sigma = (s1 + s2) * 0.5
            with torch.no_grad():
                self.running_avg_sigma.mul_(1 - self.momentum).add_(avg_sigma * self.momentum)
        else:
            avg_sigma = self.running_avg_sigma.clamp(min=self.eps)
        return self.gamma * (h1 + h2) / avg_sigma + self.bias


class AvgSigmaCorrInvBranchLinear(nn.Module):
    """Two branches scaled by 1/(σ·(2+ρ)) where σ=(σ₁+σ₂)/2, ρ=per-feature corr (stop-gradient).

    Output: γ ⊙ (1/(σ·(2+ρ))) ⊙ (w₁x + w₂x) + bias
    Running EMA of avg_sigma and rho used at eval. γ initialised to 1/√2.
    """

    def __init__(self, in_features: int, out_features: int,
                 freeze_norm_scale: bool = False,
                 momentum: float = 0.1, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.linear1 = nn.Linear(in_features, out_features, bias=False)
        self.linear2 = nn.Linear(in_features, out_features, bias=False)
        self.gamma = nn.Parameter(torch.full((out_features,), 2 ** -0.5))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.register_buffer("running_avg_sigma", torch.ones(out_features))
        self.register_buffer("running_rho", torch.zeros(out_features))
        nn.init.kaiming_normal_(self.linear1.weight, mode="fan_in", nonlinearity="relu")
        nn.init.kaiming_normal_(self.linear2.weight, mode="fan_in", nonlinearity="relu")
        if freeze_norm_scale:
            self.gamma.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.linear1(x)
        h2 = self.linear2(x)
        self._branch_outs = (h1, h2)
        if self.training:
            s1 = h1.std(dim=0, unbiased=False).clamp(min=self.eps).detach()
            s2 = h2.std(dim=0, unbiased=False).clamp(min=self.eps).detach()
            avg_sigma = (s1 + s2) * 0.5
            h1c = (h1 - h1.mean(dim=0, keepdim=True)).detach()
            h2c = (h2 - h2.mean(dim=0, keepdim=True)).detach()
            rho = (h1c * h2c).mean(dim=0) / (s1 * s2).clamp(min=self.eps)
            with torch.no_grad():
                self.running_avg_sigma.mul_(1 - self.momentum).add_(avg_sigma * self.momentum)
                self.running_rho.mul_(1 - self.momentum).add_(rho * self.momentum)
        else:
            avg_sigma = self.running_avg_sigma.clamp(min=self.eps)
            rho = self.running_rho
        denom = avg_sigma * (2.0 + rho).clamp(min=self.eps)
        return self.gamma * (h1 + h2) / denom + self.bias


class CenteredSigmaTotBranchLinear(nn.Module):
    """Two branches centered per-branch, divided by σ_tot = std(h₁+h₂). Logs σ_avg too.

    Forward: γ ⊙ ((h₁−μ₁) + (h₂−μ₂)) / σ_tot + bias
    Gradients flow through μ and σ_tot (no stop-gradient), matching standard BN behaviour.
    Also tracks running σ_avg = (σ₁+σ₂)/2 for comparison logging.
    γ initialised to 1 (output variance ≈ 1 at init, equivalent to single-branch BN).
    """

    def __init__(self, in_features: int, out_features: int,
                 freeze_norm_scale: bool = False,
                 momentum: float = 0.1, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.linear1 = nn.Linear(in_features, out_features, bias=False)
        self.linear2 = nn.Linear(in_features, out_features, bias=False)
        self.gamma = nn.Parameter(torch.ones(out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.register_buffer("running_mean1", torch.zeros(out_features))
        self.register_buffer("running_mean2", torch.zeros(out_features))
        self.register_buffer("running_sigma_tot", torch.ones(out_features))
        self.register_buffer("running_sigma1", torch.ones(out_features))
        self.register_buffer("running_sigma2", torch.ones(out_features))
        nn.init.kaiming_normal_(self.linear1.weight, mode="fan_in", nonlinearity="relu")
        nn.init.kaiming_normal_(self.linear2.weight, mode="fan_in", nonlinearity="relu")
        if freeze_norm_scale:
            self.gamma.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.linear1(x)
        h2 = self.linear2(x)
        self._branch_outs = (h1, h2)
        h = h1 + h2
        if self.training:
            mu1 = h1.mean(dim=0)
            mu2 = h2.mean(dim=0)
            sigma_tot = h.std(dim=0, unbiased=False).clamp(min=self.eps)
            with torch.no_grad():
                s1 = h1.std(dim=0, unbiased=False).clamp(min=self.eps)
                s2 = h2.std(dim=0, unbiased=False).clamp(min=self.eps)
                m = self.momentum
                self.running_mean1.mul_(1 - m).add_(mu1 * m)
                self.running_mean2.mul_(1 - m).add_(mu2 * m)
                self.running_sigma_tot.mul_(1 - m).add_(sigma_tot * m)
                self.running_sigma1.mul_(1 - m).add_(s1 * m)
                self.running_sigma2.mul_(1 - m).add_(s2 * m)
        else:
            mu1 = self.running_mean1
            mu2 = self.running_mean2
            sigma_tot = self.running_sigma_tot.clamp(min=self.eps)
        return self.gamma * ((h1 - mu1) + (h2 - mu2)) / sigma_tot + self.bias


class CenteredAvgSigmaBranchLinear(nn.Module):
    """Two branches centered per-branch, divided by σ_avg = (σ₁+σ₂)/2. Logs σ_tot too.

    Forward: γ ⊙ ((h₁−μ₁) + (h₂−μ₂)) / σ_avg + bias
    Gradients flow through μ and σ_avg (no stop-gradient), matching standard BN behaviour.
    Also tracks running σ_tot = std(h₁+h₂) for comparison logging.
    γ initialised to 1/√2 (output variance ≈ 1 at init when ρ≈0).
    """

    def __init__(self, in_features: int, out_features: int,
                 freeze_norm_scale: bool = False,
                 momentum: float = 0.1, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.linear1 = nn.Linear(in_features, out_features, bias=False)
        self.linear2 = nn.Linear(in_features, out_features, bias=False)
        self.gamma = nn.Parameter(torch.full((out_features,), 2 ** -0.5))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.register_buffer("running_mean1", torch.zeros(out_features))
        self.register_buffer("running_mean2", torch.zeros(out_features))
        self.register_buffer("running_sigma1", torch.ones(out_features))
        self.register_buffer("running_sigma2", torch.ones(out_features))
        self.register_buffer("running_sigma_tot", torch.ones(out_features))
        nn.init.kaiming_normal_(self.linear1.weight, mode="fan_in", nonlinearity="relu")
        nn.init.kaiming_normal_(self.linear2.weight, mode="fan_in", nonlinearity="relu")
        if freeze_norm_scale:
            self.gamma.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.linear1(x)
        h2 = self.linear2(x)
        self._branch_outs = (h1, h2)
        h = h1 + h2
        if self.training:
            mu1 = h1.mean(dim=0)
            mu2 = h2.mean(dim=0)
            s1 = h1.std(dim=0, unbiased=False).clamp(min=self.eps)
            s2 = h2.std(dim=0, unbiased=False).clamp(min=self.eps)
            sigma_avg = (s1 + s2) * 0.5
            with torch.no_grad():
                sigma_tot = h.std(dim=0, unbiased=False).clamp(min=self.eps)
                m = self.momentum
                self.running_mean1.mul_(1 - m).add_(mu1 * m)
                self.running_mean2.mul_(1 - m).add_(mu2 * m)
                self.running_sigma1.mul_(1 - m).add_(s1 * m)
                self.running_sigma2.mul_(1 - m).add_(s2 * m)
                self.running_sigma_tot.mul_(1 - m).add_(sigma_tot * m)
        else:
            mu1 = self.running_mean1
            mu2 = self.running_mean2
            sigma_avg = ((self.running_sigma1 + self.running_sigma2) * 0.5).clamp(min=self.eps)
        return self.gamma * ((h1 - mu1) + (h2 - mu2)) / sigma_avg + self.bias


class CenteredAvgSigmaSqrt2BranchLinear(nn.Module):
    """Two branches centered per-branch, divided by σ_avg·√2 = (σ₁+σ₂)/√2.

    Forward: γ ⊙ ((h₁−μ₁) + (h₂−μ₂)) / (σ_avg·√2) + bias
    σ_avg·√2 approximates σ_tot when branches are uncorrelated (ρ=0).
    γ initialised to 1 (output variance ≈ 1 at init when ρ≈0).
    """

    def __init__(self, in_features: int, out_features: int,
                 freeze_norm_scale: bool = False,
                 momentum: float = 0.1, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.linear1 = nn.Linear(in_features, out_features, bias=False)
        self.linear2 = nn.Linear(in_features, out_features, bias=False)
        self.gamma = nn.Parameter(torch.ones(out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.register_buffer("running_mean1", torch.zeros(out_features))
        self.register_buffer("running_mean2", torch.zeros(out_features))
        self.register_buffer("running_sigma1", torch.ones(out_features))
        self.register_buffer("running_sigma2", torch.ones(out_features))
        nn.init.kaiming_normal_(self.linear1.weight, mode="fan_in", nonlinearity="relu")
        nn.init.kaiming_normal_(self.linear2.weight, mode="fan_in", nonlinearity="relu")
        if freeze_norm_scale:
            self.gamma.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.linear1(x)
        h2 = self.linear2(x)
        self._branch_outs = (h1, h2)
        if self.training:
            mu1 = h1.mean(dim=0)
            mu2 = h2.mean(dim=0)
            s1 = h1.std(dim=0, unbiased=False).clamp(min=self.eps)
            s2 = h2.std(dim=0, unbiased=False).clamp(min=self.eps)
            sigma_eff = (s1 + s2) * 0.5 * math.sqrt(2)
            with torch.no_grad():
                m = self.momentum
                self.running_mean1.mul_(1 - m).add_(mu1 * m)
                self.running_mean2.mul_(1 - m).add_(mu2 * m)
                self.running_sigma1.mul_(1 - m).add_(s1 * m)
                self.running_sigma2.mul_(1 - m).add_(s2 * m)
        else:
            mu1 = self.running_mean1
            mu2 = self.running_mean2
            sigma_eff = ((self.running_sigma1 + self.running_sigma2) * 0.5 * math.sqrt(2)).clamp(min=self.eps)
        return self.gamma * ((h1 - mu1) + (h2 - mu2)) / sigma_eff + self.bias


class WNSumBNBranchLinear(nn.Module):
    """Two weight-normed branches summed then divided by stop-gradient batch stats.

    Forward: γ ⊙ (ŵ₁x + ŵ₂x − sg(μ_B)) / sg(σ_B) + bias
    ŵᵢ = wᵢ/‖wᵢ‖; gradient flows through ŵᵢ (tangential) but not through σ_B or μ_B.
    Running EMA of mean/var used at eval time. γ initialised to 1/√2.
    """

    def __init__(self, in_features: int, out_features: int,
                 freeze_norm_scale: bool = False,
                 momentum: float = 0.1, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.linear1 = nn.Linear(in_features, out_features, bias=False)
        self.linear2 = nn.Linear(in_features, out_features, bias=False)
        self.gamma = nn.Parameter(torch.full((out_features,), 2 ** -0.5))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.register_buffer("running_mean", torch.zeros(out_features))
        self.register_buffer("running_var", torch.ones(out_features))
        nn.init.kaiming_normal_(self.linear1.weight, mode="fan_in", nonlinearity="relu")
        nn.init.kaiming_normal_(self.linear2.weight, mode="fan_in", nonlinearity="relu")
        if freeze_norm_scale:
            self.gamma.requires_grad_(False)

    @staticmethod
    def _unit(w: torch.Tensor) -> torch.Tensor:
        return w / w.norm(dim=1, keepdim=True).clamp_min(1e-12)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = F.linear(x, self._unit(self.linear1.weight))
        h2 = F.linear(x, self._unit(self.linear2.weight))
        self._branch_outs = (h1, h2)  # read by mechanistic probe
        h = h1 + h2
        if self.training:
            mu = h.mean(dim=0).detach()
            sigma = h.std(dim=0, unbiased=False).clamp(min=self.eps).detach()
            with torch.no_grad():
                self.running_mean.mul_(1 - self.momentum).add_(mu * self.momentum)
                self.running_var.mul_(1 - self.momentum).add_(sigma.pow(2) * self.momentum)
        else:
            mu = self.running_mean
            sigma = self.running_var.clamp(min=self.eps ** 2).sqrt()
        return self.gamma * (h - mu) / sigma + self.bias


class WNInvSigmaSumBranchLinear(nn.Module):
    """Two weight-normed branches scaled by stop-gradient (1/σ₁ + 1/σ₂).

    Forward: γ ⊙ (sg(1/σ₁) + sg(1/σ₂)) ⊙ (ŵ₁x + ŵ₂x) + bias
    ŵᵢ = wᵢ/‖wᵢ‖; gradient flows through ŵᵢ (tangential) but not through the σᵢ
    scale factors. Running EMA of 1/σᵢ used at eval. γ initialised to 1/√2.
    """

    def __init__(self, in_features: int, out_features: int,
                 freeze_norm_scale: bool = False,
                 momentum: float = 0.1, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.linear1 = nn.Linear(in_features, out_features, bias=False)
        self.linear2 = nn.Linear(in_features, out_features, bias=False)
        self.gamma = nn.Parameter(torch.full((out_features,), 2 ** -0.5))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.register_buffer("running_inv_sigma1", torch.ones(out_features))
        self.register_buffer("running_inv_sigma2", torch.ones(out_features))
        self.register_buffer("running_mean", torch.zeros(out_features))
        nn.init.kaiming_normal_(self.linear1.weight, mode="fan_in", nonlinearity="relu")
        nn.init.kaiming_normal_(self.linear2.weight, mode="fan_in", nonlinearity="relu")
        if freeze_norm_scale:
            self.gamma.requires_grad_(False)

    @staticmethod
    def _unit(w: torch.Tensor) -> torch.Tensor:
        return w / w.norm(dim=1, keepdim=True).clamp_min(1e-12)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = F.linear(x, self._unit(self.linear1.weight))
        h2 = F.linear(x, self._unit(self.linear2.weight))
        self._branch_outs = (h1, h2)  # read by mechanistic probe
        h = h1 + h2
        if self.training:
            inv_s1 = h1.std(dim=0, unbiased=False).clamp(min=self.eps).reciprocal().detach()
            inv_s2 = h2.std(dim=0, unbiased=False).clamp(min=self.eps).reciprocal().detach()
            mu = h.mean(dim=0).detach()
            with torch.no_grad():
                self.running_inv_sigma1.mul_(1 - self.momentum).add_(inv_s1 * self.momentum)
                self.running_inv_sigma2.mul_(1 - self.momentum).add_(inv_s2 * self.momentum)
                self.running_mean.mul_(1 - self.momentum).add_(mu * self.momentum)
        else:
            inv_s1 = self.running_inv_sigma1
            inv_s2 = self.running_inv_sigma2
            mu = self.running_mean
        return self.gamma * (inv_s1 + inv_s2) * (h - mu) + self.bias


class _WeakProjBNFunc(torch.autograd.Function):
    """BN forward with a modified backward that scales the variance-direction correction.

    Standard BN backward:  ∂L/∂xᵢ ∝ N·δᵢ − Σδ − ŷᵢ·Σ(δ·ŷ)
    Modified:               ∂L/∂xᵢ ∝ N·δᵢ − Σδ − (1−α)·ŷᵢ·Σ(δ·ŷ)

    α=0 → standard BN (variance direction fully projected out).
    α=1 → only mean-centering correction (variance direction fully allowed).
    """

    @staticmethod
    def forward(ctx, x, gamma, running_mean, running_var,
                training, momentum, eps, alpha):
        if training:
            mu = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)
            with torch.no_grad():
                n = x.shape[0]
                running_var_update = var.detach() * (n / (n - 1)) if n > 1 else var.detach()
                running_mean.mul_(1 - momentum).add_(mu * momentum)
                running_var.mul_(1 - momentum).add_(running_var_update * momentum)
        else:
            mu = running_mean
            var = running_var
        sigma = (var + eps).sqrt()
        x_hat = (x - mu) / sigma
        y = gamma * x_hat
        ctx.save_for_backward(x_hat, gamma, sigma)
        ctx.alpha = alpha
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x_hat, gamma, sigma = ctx.saved_tensors
        alpha = ctx.alpha
        N = x_hat.shape[0]
        d_gamma = (grad_output * x_hat).sum(dim=0)
        d_x_hat = grad_output * gamma
        sum_d = d_x_hat.sum(dim=0)
        sum_d_x = (d_x_hat * x_hat).sum(dim=0)
        d_x = (N * d_x_hat - sum_d - (1.0 - alpha) * x_hat * sum_d_x) / (N * sigma)
        return d_x, d_gamma, None, None, None, None, None, None


class WeakProjBatchNorm1d(nn.Module):
    """BatchNorm1d with weakened variance-direction gradient projection.

    Identical forward to BatchNorm1d. In the backward, the term that projects
    out the variance direction (ŷ·Σ(δ·ŷ)) is scaled by (1−α), so α=0 is
    standard BN and α=1 allows full gradient flow in the variance direction.
    """

    def __init__(self, num_features: int, alpha: float = 0.0,
                 momentum: float = 0.1, eps: float = 1e-5) -> None:
        super().__init__()
        self.alpha = alpha
        self.momentum = momentum
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _WeakProjBNFunc.apply(
            x, self.weight,
            self.running_mean, self.running_var,
            self.training, self.momentum, self.eps, self.alpha,
        )


def _make_norm(norm: str, num_features: int, weak_proj_alpha: float = 0.0) -> nn.Module:
    if norm == "layer":
        return nn.LayerNorm(num_features)
    if norm == "batch":
        return nn.BatchNorm1d(num_features)
    if norm == "batch_zeromean":
        return ZeroMeanBatchNorm1d(num_features)
    if norm == "weak_proj_bn":
        return WeakProjBatchNorm1d(num_features, alpha=weak_proj_alpha)
    raise ValueError(
        f"Unsupported norm: {norm!r}. Expected batch, batch_zeromean, layer, or weak_proj_bn. "
        "For weight_norm, use BranchedLinear which wires the linear weight reference."
    )


class BridgedSigmaBranchLinear(nn.Module):
    """Two branches centered per-branch, divided by σ_eff that interpolates σ_tot↔σ_avg.

    σ_eff² = σ_tot² − β·(σ_tot² − σ_avg²)

    β=0 recovers σ_eff = σ_tot  (single-BN-equivalent, like CenteredSigmaTotBranchLinear).
    β=1 recovers σ_eff = σ_avg  (two-branch-equivalent, like CenteredAvgSigmaBranchLinear).
    The correction β·(σ_tot²−σ_avg²) is stop-gradiented; gradients flow through σ_tot² only.

    γ initialised to sqrt((2−β)/2) so output variance ≈ 1 at init (ρ≈0).
    """

    def __init__(self, in_features: int, out_features: int,
                 beta: float = 0.0,
                 freeze_norm_scale: bool = False,
                 momentum: float = 0.1, eps: float = 1e-5) -> None:
        super().__init__()
        if not (0.0 <= beta <= 1.0):
            raise ValueError(f"beta must be in [0, 1], got {beta}")
        self.beta = beta
        self.eps = eps
        self.momentum = momentum
        self.linear1 = nn.Linear(in_features, out_features, bias=False)
        self.linear2 = nn.Linear(in_features, out_features, bias=False)
        gamma_init = math.sqrt((2.0 - beta) / 2.0)
        self.gamma = nn.Parameter(torch.full((out_features,), gamma_init))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.register_buffer("running_mean1", torch.zeros(out_features))
        self.register_buffer("running_mean2", torch.zeros(out_features))
        self.register_buffer("running_sigma_tot", torch.ones(out_features))
        self.register_buffer("running_sigma1", torch.ones(out_features))
        self.register_buffer("running_sigma2", torch.ones(out_features))
        nn.init.kaiming_normal_(self.linear1.weight, mode="fan_in", nonlinearity="relu")
        nn.init.kaiming_normal_(self.linear2.weight, mode="fan_in", nonlinearity="relu")
        if freeze_norm_scale:
            self.gamma.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.linear1(x)
        h2 = self.linear2(x)
        self._branch_outs = (h1, h2)
        h = h1 + h2
        if self.training:
            mu1 = h1.mean(dim=0)
            mu2 = h2.mean(dim=0)
            s1 = h1.std(dim=0, unbiased=False).clamp(min=self.eps)
            s2 = h2.std(dim=0, unbiased=False).clamp(min=self.eps)
            sigma_tot = h.std(dim=0, unbiased=False).clamp(min=self.eps)
            sigma_avg = (s1 + s2) * 0.5
            correction = (self.beta * (sigma_tot.pow(2) - sigma_avg.pow(2))).detach()
            sigma_eff = (sigma_tot.pow(2) - correction).clamp(min=self.eps ** 2).sqrt()
            with torch.no_grad():
                m = self.momentum
                self.running_mean1.mul_(1 - m).add_(mu1 * m)
                self.running_mean2.mul_(1 - m).add_(mu2 * m)
                self.running_sigma_tot.mul_(1 - m).add_(sigma_tot * m)
                self.running_sigma1.mul_(1 - m).add_(s1 * m)
                self.running_sigma2.mul_(1 - m).add_(s2 * m)
        else:
            mu1 = self.running_mean1
            mu2 = self.running_mean2
            sigma_tot = self.running_sigma_tot.clamp(min=self.eps)
            sigma_avg = ((self.running_sigma1 + self.running_sigma2) * 0.5).clamp(min=self.eps)
            correction = self.beta * (sigma_tot.pow(2) - sigma_avg.pow(2))
            sigma_eff = (sigma_tot.pow(2) - correction).clamp(min=self.eps ** 2).sqrt()
        return self.gamma * ((h1 - mu1) + (h2 - mu2)) / sigma_eff + self.bias


class AlphaCorrSigmaBranchLinear(nn.Module):
    """Two branches centered per-branch, divided by σ_eff = √(σ₁²+σ₂²+2αρσ₁σ₂).

    α= 0 → σ_eff = √(σ₁²+σ₂²) (uncorrelated sum std, σ_avg).
    α=+1 → σ_eff = std(h₁+h₂) = σ_tot exactly.
    α=-1 → σ_eff = std(h₁−h₂) = σ_anti exactly.

    When α=-1 and ρ→1 the expression inside the sqrt approaches 0; the
    clamp at eps² prevents division by zero.

    Full gradients flow through μ, σ₁, σ₂, and cov(h₁,h₂) in the denominator.
    γ initialised to 1 so output variance ≈ 1 at init (ρ≈0, σ₁≈σ₂).
    """

    def __init__(self, in_features: int, out_features: int,
                 alpha: float = 0.0,
                 freeze_norm_scale: bool = False,
                 momentum: float = 0.1, eps: float = 1e-5) -> None:
        super().__init__()
        if not (-10.0 <= alpha <= 10.0):
            raise ValueError(f"alpha must be in [-10, 10], got {alpha}")
        self.alpha = alpha
        self.eps = eps
        self.momentum = momentum
        self.linear1 = nn.Linear(in_features, out_features, bias=False)
        self.linear2 = nn.Linear(in_features, out_features, bias=False)
        self.gamma = nn.Parameter(torch.ones(out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.register_buffer("running_mean1", torch.zeros(out_features))
        self.register_buffer("running_mean2", torch.zeros(out_features))
        self.register_buffer("running_sigma1", torch.ones(out_features))
        self.register_buffer("running_sigma2", torch.ones(out_features))
        self.register_buffer("running_rho", torch.zeros(out_features))
        nn.init.kaiming_normal_(self.linear1.weight, mode="fan_in", nonlinearity="relu")
        nn.init.kaiming_normal_(self.linear2.weight, mode="fan_in", nonlinearity="relu")
        if freeze_norm_scale:
            self.gamma.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.linear1(x)
        h2 = self.linear2(x)
        self._branch_outs = (h1, h2)
        if self.training:
            mu1 = h1.mean(dim=0)
            mu2 = h2.mean(dim=0)
            h1c = h1 - mu1
            h2c = h2 - mu2
            s1 = h1.std(dim=0, unbiased=False).clamp(min=self.eps)
            s2 = h2.std(dim=0, unbiased=False).clamp(min=self.eps)
            cov = (h1c * h2c).mean(dim=0)
            sigma_eff = (s1.pow(2) + s2.pow(2) + 2 * self.alpha * cov).clamp(
                min=self.eps ** 2).sqrt()
            with torch.no_grad():
                rho = cov / (s1 * s2).clamp(min=self.eps)
                m = self.momentum
                self.running_mean1.mul_(1 - m).add_(mu1 * m)
                self.running_mean2.mul_(1 - m).add_(mu2 * m)
                self.running_sigma1.mul_(1 - m).add_(s1 * m)
                self.running_sigma2.mul_(1 - m).add_(s2 * m)
                self.running_rho.mul_(1 - m).add_(rho * m)
                sigma_tot = (h1c + h2c).std(dim=0, unbiased=False).clamp(min=self.eps)
                self._probe_data = {
                    "sigma_eff": sigma_eff.detach().mean().item(),
                    "sigma1": s1.mean().item(),
                    "sigma2": s2.mean().item(),
                    "sigma_tot": sigma_tot.mean().item(),
                    "rho": rho.mean().item(),
                }
        else:
            mu1 = self.running_mean1
            mu2 = self.running_mean2
            h1c = h1 - mu1
            h2c = h2 - mu2
            s1 = self.running_sigma1.clamp(min=self.eps)
            s2 = self.running_sigma2.clamp(min=self.eps)
            rho = self.running_rho
            sigma_eff = (s1.pow(2) + s2.pow(2) + 2 * self.alpha * rho * s1 * s2).clamp(
                min=self.eps ** 2).sqrt()
        return self.gamma * (h1c + h2c) / sigma_eff + self.bias


class SqrtHalfSumSqSigmaBranchLinear(nn.Module):
    """Two branches centered per-branch, divided by √((σ₁²+σ₂²)/2).

    This is the RMS of the two individual branch sigmas.
    Compare with AlphaCorrSigma(α=0) which uses √(σ₁²+σ₂²) — a factor of √2 larger.

    γ initialised to 1/√2 so output variance ≈ 1 at init (ρ≈0, σ₁≈σ₂).
    """

    def __init__(self, in_features: int, out_features: int,
                 freeze_norm_scale: bool = False,
                 momentum: float = 0.1, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.linear1 = nn.Linear(in_features, out_features, bias=False)
        self.linear2 = nn.Linear(in_features, out_features, bias=False)
        self.gamma = nn.Parameter(torch.full((out_features,), 2 ** -0.5))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.register_buffer("running_mean1", torch.zeros(out_features))
        self.register_buffer("running_mean2", torch.zeros(out_features))
        self.register_buffer("running_sigma1", torch.ones(out_features))
        self.register_buffer("running_sigma2", torch.ones(out_features))
        nn.init.kaiming_normal_(self.linear1.weight, mode="fan_in", nonlinearity="relu")
        nn.init.kaiming_normal_(self.linear2.weight, mode="fan_in", nonlinearity="relu")
        if freeze_norm_scale:
            self.gamma.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.linear1(x)
        h2 = self.linear2(x)
        self._branch_outs = (h1, h2)
        if self.training:
            mu1 = h1.mean(dim=0)
            mu2 = h2.mean(dim=0)
            h1c = h1 - mu1
            h2c = h2 - mu2
            s1 = h1.std(dim=0, unbiased=False).clamp(min=self.eps)
            s2 = h2.std(dim=0, unbiased=False).clamp(min=self.eps)
            sigma_eff = ((s1.pow(2) + s2.pow(2)) * 0.5).clamp(min=self.eps ** 2).sqrt()
            with torch.no_grad():
                m = self.momentum
                self.running_mean1.mul_(1 - m).add_(mu1 * m)
                self.running_mean2.mul_(1 - m).add_(mu2 * m)
                self.running_sigma1.mul_(1 - m).add_(s1 * m)
                self.running_sigma2.mul_(1 - m).add_(s2 * m)
                cov = (h1c * h2c).mean(dim=0)
                rho = cov / (s1 * s2).clamp(min=self.eps)
                sigma_tot = (h1c + h2c).std(dim=0, unbiased=False).clamp(min=self.eps)
                self._probe_data = {
                    "sigma_eff": sigma_eff.detach().mean().item(),
                    "sigma1": s1.mean().item(),
                    "sigma2": s2.mean().item(),
                    "sigma_tot": sigma_tot.mean().item(),
                    "rho": rho.mean().item(),
                }
        else:
            mu1 = self.running_mean1
            mu2 = self.running_mean2
            h1c = h1 - mu1
            h2c = h2 - mu2
            s1 = self.running_sigma1.clamp(min=self.eps)
            s2 = self.running_sigma2.clamp(min=self.eps)
            sigma_eff = ((s1.pow(2) + s2.pow(2)) * 0.5).clamp(min=self.eps ** 2).sqrt()
        return self.gamma * (h1c + h2c) / sigma_eff + self.bias


class TwoBranchLinear(nn.Module):
    """Two BN branches, each with its own γ, sharing one bias.

    Forward: γ₁ ⊙ BN₁(h₁) + γ₂ ⊙ BN₂(h₂) + bias
    γᵢ initialised to 1/√2 (output variance ≈ 1 at init, uncorrelated branches).
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, out_features, bias=False),
                nn.BatchNorm1d(out_features, affine=False),
            )
            for _ in range(2)
        ])
        self.gammas = nn.ParameterList([
            nn.Parameter(torch.full((out_features,), 2 ** -0.5))
            for _ in range(2)
        ])
        self.bias = nn.Parameter(torch.zeros(out_features))
        for branch in self.branches:
            nn.init.kaiming_normal_(branch[0].weight, mode="fan_in", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return sum(g * branch(x) for g, branch in zip(self.gammas, self.branches)) + self.bias


class SingletonSqrtHalfSumSqSigmaBranchLinear(nn.Module):
    """Two branches centered per-branch, each with its own γ, divided by shared σ_eff=√((σ₁²+σ₂²)/2).

    Forward: γ₁ ⊙ h₁c/σ_eff + γ₂ ⊙ h₂c/σ_eff + bias
    γᵢ initialised to 1/√2.
    """

    def __init__(self, in_features: int, out_features: int,
                 momentum: float = 0.1, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.linear1 = nn.Linear(in_features, out_features, bias=False)
        self.linear2 = nn.Linear(in_features, out_features, bias=False)
        self.gamma1 = nn.Parameter(torch.full((out_features,), 2 ** -0.5))
        self.gamma2 = nn.Parameter(torch.full((out_features,), 2 ** -0.5))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.register_buffer("running_mean1", torch.zeros(out_features))
        self.register_buffer("running_mean2", torch.zeros(out_features))
        self.register_buffer("running_sigma1", torch.ones(out_features))
        self.register_buffer("running_sigma2", torch.ones(out_features))
        nn.init.kaiming_normal_(self.linear1.weight, mode="fan_in", nonlinearity="relu")
        nn.init.kaiming_normal_(self.linear2.weight, mode="fan_in", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.linear1(x)
        h2 = self.linear2(x)
        self._branch_outs = (h1, h2)
        if self.training:
            mu1 = h1.mean(dim=0)
            mu2 = h2.mean(dim=0)
            h1c = h1 - mu1
            h2c = h2 - mu2
            s1 = h1.std(dim=0, unbiased=False).clamp(min=self.eps)
            s2 = h2.std(dim=0, unbiased=False).clamp(min=self.eps)
            sigma_eff = ((s1.pow(2) + s2.pow(2)) * 0.5).clamp(min=self.eps ** 2).sqrt()
            with torch.no_grad():
                m = self.momentum
                self.running_mean1.mul_(1 - m).add_(mu1 * m)
                self.running_mean2.mul_(1 - m).add_(mu2 * m)
                self.running_sigma1.mul_(1 - m).add_(s1 * m)
                self.running_sigma2.mul_(1 - m).add_(s2 * m)
                cov = (h1c * h2c).mean(dim=0)
                rho = cov / (s1 * s2).clamp(min=self.eps)
                sigma_tot = (h1c + h2c).std(dim=0, unbiased=False).clamp(min=self.eps)
                self._probe_data = {
                    "sigma_eff": sigma_eff.detach().mean().item(),
                    "sigma1": s1.mean().item(),
                    "sigma2": s2.mean().item(),
                    "sigma_tot": sigma_tot.mean().item(),
                    "rho": rho.mean().item(),
                }
        else:
            mu1 = self.running_mean1
            mu2 = self.running_mean2
            h1c = h1 - mu1
            h2c = h2 - mu2
            s1 = self.running_sigma1.clamp(min=self.eps)
            s2 = self.running_sigma2.clamp(min=self.eps)
            sigma_eff = ((s1.pow(2) + s2.pow(2)) * 0.5).clamp(min=self.eps ** 2).sqrt()
        return self.gamma1 * h1c / sigma_eff + self.gamma2 * h2c / sigma_eff + self.bias


class BranchedLinear(nn.Module):
    """Linear layer with N parallel branches.

    Each branch is Linear(bias=False) → norm layer with beta frozen to 0.
    Branch norm gammas are initialised to 1/sqrt(N). A single shared bias is
    added after summing all branch outputs, matching the RepVGG convention.
    Supports norm="batch" (BatchNorm1d), norm="batch_zeromean",
    norm="weight_norm", or norm="layer" (LayerNorm).
    """

    def __init__(self, in_features: int, out_features: int, num_branches: int = 1,
                 init_strategies: list[str] | None = None, norm: str = "batch",
                 branch2_init_angle_deg: float | None = None,
                 freeze_norm_scale: bool = False,
                 weak_proj_alpha: float = 0.0) -> None:
        super().__init__()
        if init_strategies is None:
            init_strategies = ["kaiming"] * num_branches
        if len(init_strategies) != num_branches:
            raise ValueError(f"init_strategies length {len(init_strategies)} != num_branches {num_branches}")

        branches = []
        for _ in range(num_branches):
            linear = nn.Linear(in_features, out_features, bias=False)
            if norm == "weight_norm":
                norm_layer = WeightNorm1d(out_features, linear.weight)
            elif norm == "mean_weight_norm":
                norm_layer = MeanWeightNorm1d(out_features, linear.weight)
            else:
                norm_layer = _make_norm(norm, out_features, weak_proj_alpha=weak_proj_alpha)
            branches.append(nn.Sequential(linear, norm_layer))
        self.branches = nn.ModuleList(branches)
        self.bias = nn.Parameter(torch.zeros(out_features))

        for branch, strategy in zip(self.branches, init_strategies):
            linear, norm_layer = branch[0], branch[1]
            nn.init.kaiming_normal_(linear.weight, mode="fan_in", nonlinearity="relu")
            if strategy == "kaiming_plus_identity":
                with torch.no_grad():
                    n = min(out_features, in_features)
                    linear.weight[:n, :n] += torch.eye(n)
            elif strategy == "double_kaiming" or strategy.startswith("kaiming_x"):
                # "double_kaiming" == "kaiming_x2"; "kaiming_xN" adds N-1 extra kaiming
                # samples on top of the base kaiming above, so the row stays Gaussian
                # but with variance N · kaiming variance (sum of N iid kaimings).
                n_extra = 1 if strategy == "double_kaiming" else int(strategy.split("_x")[1]) - 1
                with torch.no_grad():
                    for _ in range(n_extra):
                        extra = torch.empty_like(linear.weight)
                        nn.init.kaiming_normal_(extra, mode="fan_in", nonlinearity="relu")
                        linear.weight.add_(extra)
                        num_branches += 1  # for correct BN init below
                        
            inv_sqrt_n = num_branches ** -0.5
            nn.init.constant_(norm_layer.weight, inv_sqrt_n)
            if hasattr(norm_layer, 'bias') and norm_layer.bias is not None:
                nn.init.zeros_(norm_layer.bias)
                norm_layer.bias.requires_grad_(False)
            if freeze_norm_scale:
                norm_layer.weight.requires_grad_(False)

        if branch2_init_angle_deg is not None and num_branches >= 2:
            self._init_branch2_angle(branch2_init_angle_deg)

    def _init_branch2_angle(self, angle_deg: float) -> None:
        """Rotate branch 1's weight rows to be exactly `angle_deg` degrees from branch 0's rows.

        Each row of branch 1 is replaced by a vector with the same norm as the
        corresponding branch 0 row, lying in the plane spanned by that row and
        branch 1's kaiming direction, at the specified angle.
        """
        theta = math.radians(angle_deg)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        w0 = self.branches[0][0].weight.data  # (out, in)
        w1 = self.branches[1][0].weight.data

        norms0 = w0.norm(dim=1, keepdim=True).clamp(min=1e-8)
        w0_hat = w0 / norms0

        # Gram-Schmidt: component of w1 perpendicular to w0
        proj = (w1 * w0_hat).sum(dim=1, keepdim=True) * w0_hat
        perp = w1 - proj
        perp_norms = perp.norm(dim=1, keepdim=True)

        # Degenerate rows: w1 parallel to w0 → resample a random perpendicular
        degenerate = perp_norms < 1e-8
        if degenerate.any():
            rand = torch.randn_like(w1)
            rand_proj = (rand * w0_hat).sum(dim=1, keepdim=True) * w0_hat
            rand_perp = rand - rand_proj
            perp = torch.where(degenerate.expand_as(perp), rand_perp, perp)
            perp_norms = perp.norm(dim=1, keepdim=True).clamp(min=1e-8)

        perp_hat = perp / perp_norms
        with torch.no_grad():
            self.branches[1][0].weight.copy_(norms0 * (cos_t * w0_hat + sin_t * perp_hat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._branch_forward(self.branches[0], x)
        for branch in self.branches[1:]:
            out = out + self._branch_forward(branch, x)
        return out + self.bias

    def _branch_forward(self, branch: nn.Sequential, x: torch.Tensor) -> torch.Tensor:
        return branch(x)


class MLP(nn.Module):
    """Fully-connected classifier with BatchNorm, ReLU, and optional multi-branch layers.

    Args:
        in_features: Flattened input size (e.g. 3*32*32 for CIFAR).
        num_classes: Number of output logits.
        width: Number of hidden units per layer.
        depth: Number of hidden layers (excluding the final linear layer).
        num_branches: Parallel branches per hidden layer (1 = standard single branch).
        branch_layers: Hidden-layer indices (0-based) that use num_branches; all
            other layers stay single-branch. None (default) applies num_branches
            to every hidden layer. Negative indices count from the end.
    """

    def __init__(
        self,
        in_features: int = 3 * 32 * 32,
        num_classes: int = 10,
        width: int = 512,
        depth: int = 3,
        num_branches: int = 1,
        branch_layers: list[int] | None = None,
        init_strategies: list[str] | None = None,
        freeze_branches: list[int] | None = None,
        frozen_branch_scale: float = 1.0,
        norm: str = "batch",
        branch2_init_angle_deg: float | None = None,
        freeze_norm_scale: bool = False,
        freeze_fc: bool = False,
        use_shared_scale: bool = False,
        sigma_bridge_beta: float = 0.0,
        sigma_alpha_corr: float = 0.0,
        sigma_subtract_corr: float = 0.0,
        weak_proj_alpha: float = 0.0,
    ) -> None:
        super().__init__()

        branch_layer_set = None
        if branch_layers is not None:
            branch_layer_set = {idx % depth for idx in branch_layers}

        layers: list[nn.Module] = []
        current_in = in_features
        for layer_idx in range(depth):
            # When branch_layers is given, only the listed hidden layers are
            # multi-branch; all others collapse to a single branch.
            layer_branches = num_branches
            if branch_layer_set is not None and layer_idx not in branch_layer_set:
                layer_branches = 1
            if norm == "sum_bn":
                layers.append(SumBatchNormBranchLinear(
                    current_in, width, freeze_norm_scale=freeze_norm_scale))
            elif norm == "inv_sigma_sum":
                layers.append(InvSigmaSumBranchLinear(
                    current_in, width, freeze_norm_scale=freeze_norm_scale))
            elif norm == "single_sigma":
                layers.append(SingleSigmaBranchLinear(
                    current_in, width, freeze_norm_scale=freeze_norm_scale))
            elif norm == "sg_scale":
                layers.append(SGScaleBranchLinear(
                    current_in, width, freeze_norm_scale=freeze_norm_scale))
            elif norm == "joint_sigma_inv":
                layers.append(JointSigmaInvBranchLinear(
                    current_in, width, freeze_norm_scale=freeze_norm_scale))
            elif norm == "joint_sigma_half_inv":
                layers.append(JointSigmaHalfInvBranchLinear(
                    current_in, width, freeze_norm_scale=freeze_norm_scale))
            elif norm == "joint_sigma_corr_inv":
                layers.append(JointSigmaCorrInvBranchLinear(
                    current_in, width, freeze_norm_scale=freeze_norm_scale))
            elif norm == "avg_sigma_inv":
                layers.append(AvgSigmaInvBranchLinear(
                    current_in, width, freeze_norm_scale=freeze_norm_scale))
            elif norm == "avg_sigma_corr_inv":
                layers.append(AvgSigmaCorrInvBranchLinear(
                    current_in, width, freeze_norm_scale=freeze_norm_scale))
            elif norm == "wn_sum_bn":
                layers.append(WNSumBNBranchLinear(
                    current_in, width, freeze_norm_scale=freeze_norm_scale))
            elif norm == "wn_inv_sigma_sum":
                layers.append(WNInvSigmaSumBranchLinear(
                    current_in, width, freeze_norm_scale=freeze_norm_scale))
            elif norm == "centered_sigma_tot":
                layers.append(CenteredSigmaTotBranchLinear(
                    current_in, width, freeze_norm_scale=freeze_norm_scale))
            elif norm == "centered_avg_sigma":
                layers.append(CenteredAvgSigmaBranchLinear(
                    current_in, width, freeze_norm_scale=freeze_norm_scale))
            elif norm == "centered_avg_sigma_sqrt2":
                layers.append(CenteredAvgSigmaSqrt2BranchLinear(
                    current_in, width, freeze_norm_scale=freeze_norm_scale))
            elif norm == "shared_scale_bn":
                layers.append(SharedScaleNBranchLinear(current_in, width, 2, norm="batch"))
            elif norm == "single_branch":
                layers.append(BranchedLinear(
                    current_in, width, 1, norm="batch",
                    freeze_norm_scale=freeze_norm_scale))
            elif norm == "bridged_sigma":
                layers.append(BridgedSigmaBranchLinear(
                    current_in, width, beta=sigma_bridge_beta,
                    freeze_norm_scale=freeze_norm_scale))
            elif norm == "alpha_corr_sigma":
                layers.append(AlphaCorrSigmaBranchLinear(
                    current_in, width, alpha=sigma_alpha_corr,
                    freeze_norm_scale=freeze_norm_scale))
            elif norm == "sqrt_half_sum_sq_sigma":
                layers.append(SqrtHalfSumSqSigmaBranchLinear(
                    current_in, width, freeze_norm_scale=freeze_norm_scale))
            elif norm == "two_branch":
                layers.append(TwoBranchLinear(current_in, width))
            elif norm == "singleton_sqrt_half_sum_sq_sigma":
                layers.append(SingletonSqrtHalfSumSqSigmaBranchLinear(current_in, width))
            elif use_shared_scale:
                layers.append(SharedScaleNBranchLinear(current_in, width, layer_branches, norm=norm))
            else:
                layers.append(BranchedLinear(
                    current_in, width, layer_branches, init_strategies,
                    norm=norm, branch2_init_angle_deg=branch2_init_angle_deg,
                    freeze_norm_scale=freeze_norm_scale,
                    weak_proj_alpha=weak_proj_alpha,
                ))
            layers.append(nn.ReLU(inplace=True))
            current_in = width

        self.hidden = nn.Sequential(*layers)
        self.fc = nn.Linear(current_in, num_classes)

        if freeze_fc:
            for p in self.fc.parameters():
                p.requires_grad_(False)

        # Freeze specified branches and apply scale to their BN gammas.
        if freeze_branches:
            for module in self.hidden:
                if not isinstance(module, BranchedLinear):
                    continue
                for idx in freeze_branches:
                    if idx < 0 or idx >= len(module.branches):
                        raise ValueError(
                            f"freeze_branches index {idx} out of range for "
                            f"num_branches={len(module.branches)}"
                        )
                    bn = module.branches[idx][1]
                    with torch.no_grad():
                        bn.weight.mul_(frozen_branch_scale)
                    for p in module.branches[idx].parameters():
                        p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(1)
        x = self.hidden(x)
        return self.fc(x)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
