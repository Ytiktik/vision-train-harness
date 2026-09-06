"""Shared helpers for RepVGG-style foldable blocks."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class RepVGGBlockBase(nn.Module):
    """Common behavior for RepVGG block variants.

    Variants differ in how they fold branches, but all expose one of the
    equivalent-kernel methods used below.
    """

    _fuse_delete_attrs: tuple[str, ...] = ()
    _fuse_final_bn: bool = False
    _fuse_final_bn_attr: str = "bn"

    def custom_l2(self) -> torch.Tensor:
        """Squared Frobenius norm of the equivalent kernel."""
        kernel = self._custom_l2_kernel()
        return (kernel ** 2).sum()

    def distance_loss(self) -> torch.Tensor:
        """Cosine-sim^2 between tracked branches, or zero before tracking runs."""
        cos_sim_sq = getattr(self, "_cos_sim_sq", None)
        if cos_sim_sq is not None:
            return cos_sim_sq
        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        return torch.zeros((), device=device)

    def _custom_l2_kernel(self) -> torch.Tensor:
        kernel, _ = self._equivalent_kernel_bias_for_fuse()
        return kernel

    def fuse(self):
        if getattr(self, "deployed", False):
            return self

        kernel, bias = self._fuse_kernel_bias()
        self.fused_conv = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=3,
            stride=self.stride,
            padding=1,
            bias=True,
        )
        self.fused_conv.weight.data.copy_(kernel.detach())
        self.fused_conv.bias.data.copy_(bias.detach())

        for name in self._fuse_delete_attrs:
            if hasattr(self, name):
                delattr(self, name)
        self.deployed = True
        return self

    def _fuse_kernel_bias(self) -> tuple[torch.Tensor, torch.Tensor]:
        kernel, bias = self._equivalent_kernel_bias_for_fuse()
        if self._fuse_final_bn:
            kernel, bias = self._fold_final_bn(kernel, bias)
        return kernel, bias

    def _equivalent_kernel_bias_for_fuse(self) -> tuple[torch.Tensor, torch.Tensor]:
        if hasattr(self, "equivalent_kernel_bias"):
            return self.equivalent_kernel_bias()
        if hasattr(self, "equivalent_kernel_and_bias"):
            return self.equivalent_kernel_and_bias()
        if hasattr(self, "equivalent_kernel"):
            kernel = self.equivalent_kernel()
            bias = torch.zeros(self.out_channels, device=kernel.device, dtype=kernel.dtype)
            return kernel, bias
        raise NotImplementedError(
            f"{type(self).__name__} must define an equivalent-kernel method for fusion"
        )

    def _fold_final_bn(
        self, kernel: torch.Tensor, bias: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bn = getattr(self, self._fuse_final_bn_attr)
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        kernel = kernel * scale[:, None, None, None]
        bias = bn.bias + scale * (bias - bn.running_mean)
        return kernel, bias


@torch.no_grad()
def max_fusion_error(block: Any, x: torch.Tensor) -> float:
    """Return max absolute output difference before and after fusion."""
    block.eval()
    before = block(x)
    block.fuse()
    after = block(x)
    return (before - after).abs().max().item()
