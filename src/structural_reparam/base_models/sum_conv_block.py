"""Two-branch conv blocks: sum-then-BN vs. mean-sigma normalisation."""

from __future__ import annotations

import math

import torch
from torch import nn


class SumBNConvBlock(nn.Module):
    """Two 3×3 branches summed before a single BatchNorm2d.

    Forward: BN(conv₁(x) + conv₂(x)) → ReLU
    Equivalent to single-BN over the summed direction.
    γ initialised to 1; output variance ≈ 1 at init.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps)
        self.act = nn.ReLU(inplace=True)
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_in", nonlinearity="relu")
        nn.init.kaiming_normal_(self.conv2.weight, mode="fan_in", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv1(x) + self.conv2(x)))


class AvgSigmaConvBlock(nn.Module):
    """Two 3×3 branches: center per-branch, divided by σ_avg = (σ₁+σ₂)/2.

    Forward: γ ⊙ ((h₁−μ₁) + (h₂−μ₂)) / σ_avg → ReLU
    Stats computed over (N, H, W) per channel. Running EMA at eval.
    Gradients flow through μ and σ_avg (no stop-gradient), matching standard BN.
    γ initialised to 1/√2 (output variance ≈ 1 when ρ≈0 at init).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        momentum: float = 0.1,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.gamma = nn.Parameter(torch.full((out_channels,), 2 ** -0.5))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.register_buffer("running_mean1", torch.zeros(out_channels))
        self.register_buffer("running_mean2", torch.zeros(out_channels))
        self.register_buffer("running_sigma1", torch.ones(out_channels))
        self.register_buffer("running_sigma2", torch.ones(out_channels))
        self.act = nn.ReLU(inplace=True)
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_in", nonlinearity="relu")
        nn.init.kaiming_normal_(self.conv2.weight, mode="fan_in", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.conv1(x)  # (N, C, H, W)
        h2 = self.conv2(x)
        if self.training:
            mu1 = h1.mean(dim=(0, 2, 3))
            mu2 = h2.mean(dim=(0, 2, 3))
            s1 = h1.std(dim=(0, 2, 3), unbiased=False).clamp(min=self.eps)
            s2 = h2.std(dim=(0, 2, 3), unbiased=False).clamp(min=self.eps)
            sigma_avg = (s1 + s2) * 0.5
            with torch.no_grad():
                m = self.momentum
                self.running_mean1.mul_(1 - m).add_(mu1.detach() * m)
                self.running_mean2.mul_(1 - m).add_(mu2.detach() * m)
                self.running_sigma1.mul_(1 - m).add_(s1.detach() * m)
                self.running_sigma2.mul_(1 - m).add_(s2.detach() * m)
        else:
            mu1 = self.running_mean1
            mu2 = self.running_mean2
            sigma_avg = ((self.running_sigma1 + self.running_sigma2) * 0.5).clamp(min=self.eps)

        g = self.gamma[:, None, None]
        b = self.bias[:, None, None]
        sg = sigma_avg[:, None, None]
        h1c = h1 - mu1[:, None, None]
        h2c = h2 - mu2[:, None, None]
        return self.act(g * (h1c + h2c) / sg + b)


class AvgSigmaSqrt2ConvBlock(nn.Module):
    """Two 3×3 branches: center per-branch, divided by σ_avg·√2 = (σ₁+σ₂)/√2.

    σ_avg·√2 approximates σ_tot when branches are uncorrelated (ρ=0).
    Forward: γ ⊙ ((h₁−μ₁) + (h₂−μ₂)) / (σ_avg·√2) → ReLU
    γ initialised to 1 (output variance ≈ 1 when ρ≈0 at init).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        momentum: float = 0.1,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.gamma = nn.Parameter(torch.ones(out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.register_buffer("running_mean1", torch.zeros(out_channels))
        self.register_buffer("running_mean2", torch.zeros(out_channels))
        self.register_buffer("running_sigma1", torch.ones(out_channels))
        self.register_buffer("running_sigma2", torch.ones(out_channels))
        self.act = nn.ReLU(inplace=True)
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_in", nonlinearity="relu")
        nn.init.kaiming_normal_(self.conv2.weight, mode="fan_in", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.conv1(x)
        h2 = self.conv2(x)
        if self.training:
            mu1 = h1.mean(dim=(0, 2, 3))
            mu2 = h2.mean(dim=(0, 2, 3))
            s1 = h1.std(dim=(0, 2, 3), unbiased=False).clamp(min=self.eps)
            s2 = h2.std(dim=(0, 2, 3), unbiased=False).clamp(min=self.eps)
            sigma_eff = (s1 + s2) * 0.5 * math.sqrt(2)
            with torch.no_grad():
                m = self.momentum
                self.running_mean1.mul_(1 - m).add_(mu1.detach() * m)
                self.running_mean2.mul_(1 - m).add_(mu2.detach() * m)
                self.running_sigma1.mul_(1 - m).add_(s1.detach() * m)
                self.running_sigma2.mul_(1 - m).add_(s2.detach() * m)
        else:
            mu1 = self.running_mean1
            mu2 = self.running_mean2
            sigma_eff = ((self.running_sigma1 + self.running_sigma2) * 0.5 * math.sqrt(2)).clamp(min=self.eps)

        g = self.gamma[:, None, None]
        b = self.bias[:, None, None]
        se = sigma_eff[:, None, None]
        h1c = h1 - mu1[:, None, None]
        h2c = h2 - mu2[:, None, None]
        return self.act(g * (h1c + h2c) / se + b)


class SharedScaleConvBlock(nn.Module):
    """Two 3×3 branches: per-branch BN2d (no affine) summed with shared γ.

    Forward: γ ⊙ (BN_noaffine(conv₁(x)) + BN_noaffine(conv₂(x))) + bias → ReLU
    Each branch normalises independently; one shared scale and bias after summing.
    γ initialised to 1/√2 (output variance ≈ 1 at init, uncorrelated branches).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels, affine=False, eps=eps),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels, affine=False, eps=eps),
        )
        self.gamma = nn.Parameter(torch.full((out_channels,), 2 ** -0.5))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.act = nn.ReLU(inplace=True)
        nn.init.kaiming_normal_(self.branch1[0].weight, mode="fan_in", nonlinearity="relu")
        nn.init.kaiming_normal_(self.branch2[0].weight, mode="fan_in", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.gamma[:, None, None]
        b = self.bias[:, None, None]
        return self.act(g * (self.branch1(x) + self.branch2(x)) + b)


class AlphaCorrSigmaConvBlock(nn.Module):
    """Two 3×3 branches centered per-branch, divided by √(σ₁²+σ₂²+2αρσ₁σ₂).

    α= 0 → σ_avg = √(σ₁²+σ₂²).
    α=+1 → σ_tot = std(h₁+h₂).
    α=-1 → σ_anti = std(h₁−h₂).
    Singularity at ρ→1 for α<0 handled by clamp at eps².
    Stats computed over (N, H, W) per channel.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        alpha: float = 0.0,
        momentum: float = 0.1,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if not (-10.0 <= alpha <= 10.0):
            raise ValueError(f"alpha must be in [-10, 10], got {alpha}")
        self.alpha = alpha
        self.eps = eps
        self.momentum = momentum
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.gamma = nn.Parameter(torch.ones(out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.register_buffer("running_mean1", torch.zeros(out_channels))
        self.register_buffer("running_mean2", torch.zeros(out_channels))
        self.register_buffer("running_sigma1", torch.ones(out_channels))
        self.register_buffer("running_sigma2", torch.ones(out_channels))
        self.register_buffer("running_rho", torch.zeros(out_channels))
        self.act = nn.ReLU(inplace=True)
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_in", nonlinearity="relu")
        nn.init.kaiming_normal_(self.conv2.weight, mode="fan_in", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.conv1(x)
        h2 = self.conv2(x)
        self._branch_outs = (h1, h2)
        if self.training:
            mu1 = h1.mean(dim=(0, 2, 3))
            mu2 = h2.mean(dim=(0, 2, 3))
            h1c = h1 - mu1[:, None, None]
            h2c = h2 - mu2[:, None, None]
            s1 = h1.std(dim=(0, 2, 3), unbiased=False).clamp(min=self.eps)
            s2 = h2.std(dim=(0, 2, 3), unbiased=False).clamp(min=self.eps)
            cov = (h1c * h2c).mean(dim=(0, 2, 3))
            sigma_eff = (s1.pow(2) + s2.pow(2) + 2 * self.alpha * cov).clamp(
                min=self.eps ** 2).sqrt()
            with torch.no_grad():
                rho = cov / (s1 * s2).clamp(min=self.eps)
                m = self.momentum
                self.running_mean1.mul_(1 - m).add_(mu1.detach() * m)
                self.running_mean2.mul_(1 - m).add_(mu2.detach() * m)
                self.running_sigma1.mul_(1 - m).add_(s1.detach() * m)
                self.running_sigma2.mul_(1 - m).add_(s2.detach() * m)
                self.running_rho.mul_(1 - m).add_(rho.detach() * m)
                sigma_tot = (h1c + h2c).std(dim=(0, 2, 3), unbiased=False).clamp(min=self.eps)
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
            h1c = h1 - mu1[:, None, None]
            h2c = h2 - mu2[:, None, None]
            s1 = self.running_sigma1.clamp(min=self.eps)
            s2 = self.running_sigma2.clamp(min=self.eps)
            rho = self.running_rho
            sigma_eff = (s1.pow(2) + s2.pow(2) + 2 * self.alpha * rho * s1 * s2).clamp(
                min=self.eps ** 2).sqrt()

        g = self.gamma[:, None, None]
        b = self.bias[:, None, None]
        se = sigma_eff[:, None, None]
        return self.act(g * (h1c + h2c) / se + b)


class SqrtHalfSumSqSigmaConvBlock(nn.Module):
    """Two 3×3 branches centered per-branch, divided by √((σ₁²+σ₂²)/2).

    RMS of the two branch sigmas — a factor √2 smaller than √(σ₁²+σ₂²).
    γ initialised to 1/√2 so output variance ≈ 1 at init (ρ≈0, σ₁≈σ₂).
    Stats computed over (N, H, W) per channel.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        momentum: float = 0.1,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.gamma = nn.Parameter(torch.full((out_channels,), 2 ** -0.5))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.register_buffer("running_mean1", torch.zeros(out_channels))
        self.register_buffer("running_mean2", torch.zeros(out_channels))
        self.register_buffer("running_sigma1", torch.ones(out_channels))
        self.register_buffer("running_sigma2", torch.ones(out_channels))
        self.act = nn.ReLU(inplace=True)
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_in", nonlinearity="relu")
        nn.init.kaiming_normal_(self.conv2.weight, mode="fan_in", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.conv1(x)
        h2 = self.conv2(x)
        self._branch_outs = (h1, h2)
        if self.training:
            mu1 = h1.mean(dim=(0, 2, 3))
            mu2 = h2.mean(dim=(0, 2, 3))
            h1c = h1 - mu1[:, None, None]
            h2c = h2 - mu2[:, None, None]
            s1 = h1.std(dim=(0, 2, 3), unbiased=False).clamp(min=self.eps)
            s2 = h2.std(dim=(0, 2, 3), unbiased=False).clamp(min=self.eps)
            sigma_eff = ((s1.pow(2) + s2.pow(2)) * 0.5).clamp(min=self.eps ** 2).sqrt()
            with torch.no_grad():
                m = self.momentum
                self.running_mean1.mul_(1 - m).add_(mu1.detach() * m)
                self.running_mean2.mul_(1 - m).add_(mu2.detach() * m)
                self.running_sigma1.mul_(1 - m).add_(s1.detach() * m)
                self.running_sigma2.mul_(1 - m).add_(s2.detach() * m)
                cov = (h1c * h2c).mean(dim=(0, 2, 3))
                rho = cov / (s1 * s2).clamp(min=self.eps)
                sigma_tot = (h1c + h2c).std(dim=(0, 2, 3), unbiased=False).clamp(min=self.eps)
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
            h1c = h1 - mu1[:, None, None]
            h2c = h2 - mu2[:, None, None]
            s1 = self.running_sigma1.clamp(min=self.eps)
            s2 = self.running_sigma2.clamp(min=self.eps)
            sigma_eff = ((s1.pow(2) + s2.pow(2)) * 0.5).clamp(min=self.eps ** 2).sqrt()

        g = self.gamma[:, None, None]
        b = self.bias[:, None, None]
        se = sigma_eff[:, None, None]
        return self.act(g * (h1c + h2c) / se + b)


class SingletonSharedScaleConvBlock(nn.Module):
    """Two 3×3 branches: per-branch BN2d (no affine), each with its own γ, sharing one bias.

    Forward: γ₁ ⊙ BN₁(conv₁(x)) + γ₂ ⊙ BN₂(conv₂(x)) + bias → ReLU
    γᵢ initialised to 1/√2 (output variance ≈ 1 at init, uncorrelated branches).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels, affine=False, eps=eps),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels, affine=False, eps=eps),
        )
        self.gamma1 = nn.Parameter(torch.full((out_channels,), 2 ** -0.5))
        self.gamma2 = nn.Parameter(torch.full((out_channels,), 2 ** -0.5))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.act = nn.ReLU(inplace=True)
        nn.init.kaiming_normal_(self.branch1[0].weight, mode="fan_in", nonlinearity="relu")
        nn.init.kaiming_normal_(self.branch2[0].weight, mode="fan_in", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g1 = self.gamma1[:, None, None]
        g2 = self.gamma2[:, None, None]
        b = self.bias[:, None, None]
        return self.act(g1 * self.branch1(x) + g2 * self.branch2(x) + b)


class SingletonSqrtHalfSumSqSigmaConvBlock(nn.Module):
    """Two branches centered per-branch, each with its own γ, divided by shared σ_eff=√((σ₁²+σ₂²)/2).

    Forward: γ₁ ⊙ h₁c/σ_eff + γ₂ ⊙ h₂c/σ_eff + bias → ReLU
    γᵢ initialised to 1/√2. Stats computed over (N, H, W) per channel.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        momentum: float = 0.1,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.gamma1 = nn.Parameter(torch.full((out_channels,), 2 ** -0.5))
        self.gamma2 = nn.Parameter(torch.full((out_channels,), 2 ** -0.5))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.register_buffer("running_mean1", torch.zeros(out_channels))
        self.register_buffer("running_mean2", torch.zeros(out_channels))
        self.register_buffer("running_sigma1", torch.ones(out_channels))
        self.register_buffer("running_sigma2", torch.ones(out_channels))
        self.act = nn.ReLU(inplace=True)
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_in", nonlinearity="relu")
        nn.init.kaiming_normal_(self.conv2.weight, mode="fan_in", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.conv1(x)
        h2 = self.conv2(x)
        self._branch_outs = (h1, h2)
        if self.training:
            mu1 = h1.mean(dim=(0, 2, 3))
            mu2 = h2.mean(dim=(0, 2, 3))
            h1c = h1 - mu1[:, None, None]
            h2c = h2 - mu2[:, None, None]
            s1 = h1.std(dim=(0, 2, 3), unbiased=False).clamp(min=self.eps)
            s2 = h2.std(dim=(0, 2, 3), unbiased=False).clamp(min=self.eps)
            sigma_eff = ((s1.pow(2) + s2.pow(2)) * 0.5).clamp(min=self.eps ** 2).sqrt()
            with torch.no_grad():
                m = self.momentum
                self.running_mean1.mul_(1 - m).add_(mu1.detach() * m)
                self.running_mean2.mul_(1 - m).add_(mu2.detach() * m)
                self.running_sigma1.mul_(1 - m).add_(s1.detach() * m)
                self.running_sigma2.mul_(1 - m).add_(s2.detach() * m)
                cov = (h1c * h2c).mean(dim=(0, 2, 3))
                rho = cov / (s1 * s2).clamp(min=self.eps)
                sigma_tot = (h1c + h2c).std(dim=(0, 2, 3), unbiased=False).clamp(min=self.eps)
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
            h1c = h1 - mu1[:, None, None]
            h2c = h2 - mu2[:, None, None]
            s1 = self.running_sigma1.clamp(min=self.eps)
            s2 = self.running_sigma2.clamp(min=self.eps)
            sigma_eff = ((s1.pow(2) + s2.pow(2)) * 0.5).clamp(min=self.eps ** 2).sqrt()

        g1 = self.gamma1[:, None, None]
        g2 = self.gamma2[:, None, None]
        b = self.bias[:, None, None]
        se = sigma_eff[:, None, None]
        return self.act(g1 * h1c / se + g2 * h2c / se + b)


class SingleBranchConvBlock(nn.Module):
    """Single 3×3 conv → BatchNorm2d → ReLU.

    Standard single-branch baseline for comparison with two-branch variants.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps)
        self.act = nn.ReLU(inplace=True)
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_in", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))
