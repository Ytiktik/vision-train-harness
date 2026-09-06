"""Kernel initializers for parallel 3x3 branches and frozen 3x3 filters.

Two families:

* Learnable 3x3 branches (E1 / E2): build a random (out_c, in_c, 3, 3) kernel
  drawn from a structured subspace.
* Fixed 3x3 filters (E4): hand-crafted depthwise (1, 1, 3, 3) signal-processing
  kernels used as frozen buffers inside ``FixedFilterBranch``.

All learnable initializers are scaled to the Kaiming fan-in std
``sqrt(2 / (in_c * 3 * 3))`` so swapping the strategy doesn't change the output
variance at init.
"""

from __future__ import annotations

import math

import torch
from torch import nn


_LEARNABLE_3x3 = {"kaiming", "kaiming_plus_identity", "rank1", "circulant", "symmetric", "antisymmetric"}
_FIXED_3x3 = {"gaussian", "highpass", "lowpass"}


def is_learnable(strategy: str) -> bool:
    return strategy in _LEARNABLE_3x3


def is_fixed(strategy: str) -> bool:
    return strategy in _FIXED_3x3


def _kaiming_std(in_channels: int, kernel_size: int = 3) -> float:
    return math.sqrt(2.0 / (in_channels * kernel_size * kernel_size))


def init_3x3(strategy: str, out_channels: int, in_channels: int) -> torch.Tensor:
    """Build a (out_c, in_c, 3, 3) kernel under the given structural prior."""
    if strategy == "kaiming":
        kernel = torch.empty(out_channels, in_channels, 3, 3)
        nn.init.kaiming_normal_(kernel, mode="fan_in", nonlinearity="relu")
        return kernel

    if strategy == "kaiming_plus_identity":
        kernel = torch.empty(out_channels, in_channels, 3, 3)
        nn.init.kaiming_normal_(kernel, mode="fan_in", nonlinearity="relu")
        if out_channels == in_channels:
            idx = torch.arange(out_channels)
            kernel[idx, idx, 1, 1] += 1.0
        return kernel

    std = _kaiming_std(in_channels)

    if strategy == "rank1":
        u = torch.randn(out_channels, in_channels, 3, 1)
        v = torch.randn(out_channels, in_channels, 1, 3)
        kernel = u * v
    elif strategy == "circulant":
        gen = torch.randn(out_channels, in_channels, 3)
        kernel = torch.stack(
            [
                gen,
                torch.roll(gen, shifts=1, dims=-1),
                torch.roll(gen, shifts=2, dims=-1),
            ],
            dim=-2,
        )
    elif strategy == "symmetric":
        raw = torch.randn(out_channels, in_channels, 3, 3)
        kernel = 0.5 * (raw + raw.transpose(-1, -2))
    elif strategy == "antisymmetric":
        raw = torch.randn(out_channels, in_channels, 3, 3)
        kernel = 0.5 * (raw - raw.transpose(-1, -2))
    else:
        raise ValueError(f"Unknown 3x3 init strategy: {strategy!r}")

    flat = kernel.reshape(out_channels, -1)
    norms = flat.norm(dim=1, keepdim=True).clamp_min(1e-8)
    target = std * math.sqrt(in_channels * 3 * 3)
    flat = flat * (target / norms)
    return flat.reshape(out_channels, in_channels, 3, 3)


def fixed_3x3(name: str) -> torch.Tensor:
    """Return a (1, 1, 3, 3) frozen depthwise kernel for a named filter."""
    if name == "gaussian":
        kernel = torch.tensor(
            [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]
        ) / 16.0
    elif name == "highpass":
        kernel = torch.tensor(
            [[-1.0, -1.0, -1.0], [-1.0, 8.0, -1.0], [-1.0, -1.0, -1.0]]
        )
    elif name == "lowpass":
        kernel = torch.full((3, 3), 1.0 / 9.0)
    else:
        raise ValueError(f"Unknown fixed 3x3 filter: {name!r}")
    return kernel.reshape(1, 1, 3, 3)
