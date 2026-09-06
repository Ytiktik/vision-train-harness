"""Optimizer factories for config-driven experiments."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn


class NormalizedGD(torch.optim.Optimizer):
    """Gradient descent with global L2 gradient normalization and no memory.

    Update rule: θ ← θ - lr * g / ||g||₂

    The full gradient vector across all parameters is concatenated, its L2
    norm computed, and every parameter update is scaled by that single scalar.
    No momentum or second-moment accumulation — purely memoryless.
    """

    def __init__(self, params: Iterable, lr: float = 0.1) -> None:
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Compute global L2 norm across all parameters
        global_norm = torch.zeros((), dtype=torch.float32)
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    global_norm += p.grad.detach().pow(2).sum()
        global_norm = global_norm.sqrt().clamp_min(1e-8)

        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data.add_(p.grad.detach() / global_norm, alpha=-lr)

        return loss


class GammaDecoupledSGD(torch.optim.SGD):
    """SGD where each linear/conv weight gradient is divided by its paired BN gamma.

    For every (Linear, BatchNorm1d) or (Conv2d, BatchNorm2d) sequential pair
    found in the model, w.grad[o] is scaled by angular_lr_mult / |γ[o]| per
    output channel before the SGD step. The gradient of γ itself is left
    unchanged.

    angular_lr_mult=1 recovers pure gamma-decoupled SGD. Values >1 speed up
    angular learning relative to gamma; values <1 slow it down.
    """

    def __init__(self, params, model: nn.Module, lr: float, momentum: float = 0.9,
                 weight_decay: float = 0.0, angular_lr_mult: float = 1.0,
                 alternating: bool = False) -> None:
        super().__init__(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        self.angular_lr_mult = angular_lr_mult
        self.alternating = alternating
        self._alt_step = 0
        self._pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
        for module in model.modules():
            if not (isinstance(module, nn.Sequential) and len(module) == 2):
                continue
            if (isinstance(module[0], nn.Conv2d)
                    and isinstance(module[1], nn.BatchNorm2d)
                    and module[1].weight is not None):
                self._pairs.append((module[0].weight, module[1].weight))
            elif (isinstance(module[0], nn.Linear)
                    and isinstance(module[1], (nn.BatchNorm1d, nn.LayerNorm))
                    and module[1].weight is not None):
                self._pairs.append((module[0].weight, module[1].weight))

    def step(self, closure=None):
        if self.alternating:
            self._alt_step += 1
            update_w = (self._alt_step % 2 == 1)
            for w, gamma in self._pairs:
                if update_w:
                    if w.grad is not None:
                        scale = gamma.detach().pow(2).clamp_min(1e-8)
                        w.grad.div_(scale.view(-1, 1) if w.grad.dim() == 2 else scale.view(-1, 1, 1, 1))
                        if self.angular_lr_mult != 1.0:
                            w.grad.mul_(self.angular_lr_mult)
                    if gamma.grad is not None:
                        gamma.grad.zero_()
                else:
                    if w.grad is not None:
                        w.grad.zero_()
        else:
            for w, gamma in self._pairs:
                if w.grad is None:
                    continue
                scale = gamma.detach().pow(2).clamp_min(1e-8)
                w.grad.div_(scale.view(-1, 1) if w.grad.dim() == 2 else scale.view(-1, 1, 1, 1))
                if self.angular_lr_mult != 1.0:
                    w.grad.mul_(self.angular_lr_mult)
        return super().step(closure)


class URebaseSGD(torch.optim.SGD):
    """Lift–step–project SGD on u = γ·w for (Linear, BN) pairs.

    For every (Linear, BatchNorm1d) or (Conv2d, BatchNorm2d) sequential pair,
    the linear/conv weight w and the BN affine γ are treated as a joint
    reparametrization of the effective post-BN weight u = γ·w/σ. Per step,
    per output channel o:

        Δw = -lr · α · g_w · (1/γ² if normalize_by_gamma2 else 1)
        Δγ = -lr · β · g_γ              (α = angular_lr_mult, β = gamma_lr_mult)
        u  = γ·w + Δw·γ + Δγ·w          (drops ΔγΔw second-order term)
        γ' = <u, w_old>                  (≈ γ + Δγ; first-order, no drift)
        w' = u / ||u||                  (||w'_o|| = 1)

    The knobs `angular_lr_mult` (α) and `gamma_lr_mult` (β) independently
    scale the w-side and γ-side steps; setting α=β recovers a plain base-lr
    scaling. `normalize_by_gamma2` toggles the γ⁻² scaling on g_w that
    matches the previous GammaDecoupledSGD-v2 setup.

    Note: γ' = ||u|| would also rebase, but ||u||² = (γ+Δγ)² + γ²||Δw||²
    (cross term zero because BN backward gives <g_w, w> = 0 per row), so
    ||u|| ≥ |γ + Δγ| strictly whenever Δw ≠ 0 — every tangential w-step
    leaks ½γ||Δw||² into γ, causing a positive secular drift that destabilizes
    training over thousands of steps. Projecting onto w_old eliminates that
    drift and recovers exact first-order γ dynamics.

    σ is not handled here — BN's backward already folds it into g_w and g_γ.
    The constructor performs an initial rebase so ||w_o|| = 1 from the start;
    this is forward-equivalent since (c·γ)·(w/c) = γ·w.

    Non-paired params are stepped by the inherited SGD with the configured
    momentum and weight_decay. Paired w gets no momentum/WD (its norm is held
    at 1 by the rebase). Paired γ gets weight_decay applied as a simple
    multiplicative shrink γ ← γ·(1 - lr·wd) after rebase — needed for
    stability since otherwise γ has no constraint and can drift unboundedly.
    Momentum is not applied to γ to keep the lift–step–project semantics clean.
    """

    def __init__(self, params, model: nn.Module, lr: float, momentum: float = 0.0,
                 weight_decay: float = 0.0, angular_lr_mult: float = 1.0,
                 gamma_lr_mult: float = 1.0,
                 normalize_by_gamma2: bool = False, eps_floor: float = 1e-12) -> None:
        super().__init__(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        self.angular_lr_mult = angular_lr_mult
        self.gamma_lr_mult = gamma_lr_mult
        self.normalize_by_gamma2 = normalize_by_gamma2
        self.eps_floor = eps_floor

        from structural_reparam.models.mlp import WeightNorm1d as _WeightNorm1d

        self._pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._wn_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
        for module in model.modules():
            if not (isinstance(module, nn.Sequential) and len(module) == 2):
                continue
            head, bn = module[0], module[1]
            if not isinstance(head, (nn.Linear, nn.Conv2d)) or bn.weight is None:
                continue
            if not head.weight.requires_grad:
                continue
            if isinstance(bn, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
                if not bn.weight.requires_grad:
                    continue
                self._pairs.append((head.weight, bn.weight))
            elif isinstance(bn, _WeightNorm1d):
                # Include even when gamma is frozen: w still needs the rebase to
                # stay on the unit sphere and prevent radial momentum accumulation.
                self._wn_pairs.append((head.weight, bn.weight))

        with torch.no_grad():
            for w, gamma in self._pairs:
                self._rebase_inplace(w, gamma)
            for w, _gamma in self._wn_pairs:
                # WeightNorm computes gamma*ŵ·x — normalising w leaves the output
                # unchanged. The BN-style rebase (gamma *= ||w||) would change it
                # by ||w||² since WeightNorm already divides by ||w|| internally.
                view = (-1,) + (1,) * (w.dim() - 1)
                norms = self._row_norm(w).clamp_min(self.eps_floor)
                w.div_(norms.view(view))

    @staticmethod
    def _row_norm(w: torch.Tensor) -> torch.Tensor:
        """L2 norm of each output-channel row of w (shape [out, ...] → [out])."""
        return w.reshape(w.shape[0], -1).norm(dim=1)

    def _rebase_inplace(self, w: torch.Tensor, gamma: torch.Tensor) -> None:
        view = (-1,) + (1,) * (w.dim() - 1)
        norms = self._row_norm(w).clamp_min(self.eps_floor)
        w.div_(norms.view(view))
        gamma.mul_(norms)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        group = self.param_groups[0]
        lr = group["lr"]
        wd = group["weight_decay"]
        lr_w = lr * self.angular_lr_mult
        lr_g = lr * self.gamma_lr_mult
        gamma_shrink = 1.0 - lr * wd

        with torch.no_grad():
            for pairs, wn_mode in ((self._pairs, False), (self._wn_pairs, True)):
                for w, gamma in pairs:
                    if w.grad is None and gamma.grad is None:
                        continue
                    g_w = w.grad if w.grad is not None else torch.zeros_like(w)
                    g_g = gamma.grad if gamma.grad is not None else torch.zeros_like(gamma)

                    view = (-1,) + (1,) * (w.dim() - 1)
                    gamma_v = gamma.view(view)
                    dg_v = (-lr_g * g_g).view(view)
                    if self.normalize_by_gamma2:
                        inv_g2 = gamma.detach().pow(2).clamp_min(self.eps_floor).reciprocal()
                        dw = (-lr_w * inv_g2).view(view) * g_w
                    else:
                        dw = -lr_w * g_w

                    u = gamma_v * w + gamma_v * dw + dg_v * w

                    # γ_new = <u, w_old> (projection onto old direction): first-order
                    # in (Δw, Δγ); equals γ + Δγ when <g_w, w> = 0 (BN/WN scale-invariant).
                    gamma_new = (u * w).reshape(u.shape[0], -1).sum(dim=1)
                    norms = self._row_norm(u).clamp_min(self.eps_floor)
                    w.data.copy_(u / norms.view(view))
                    if not wn_mode or gamma.requires_grad:
                        gamma.data.copy_(gamma_new)
                        if gamma_shrink != 1.0:
                            gamma.data.mul_(gamma_shrink)

                    # Prevent parent SGD from re-stepping these params (or applying WD).
                    w.grad = None
                    gamma.grad = None

        super().step(closure=None)
        return loss


def _is_conv_bn_branch(module: nn.Module) -> bool:
    return (
        isinstance(module, nn.Sequential)
        and len(module) == 2
        and isinstance(module[0], nn.Conv2d)
        and isinstance(module[1], nn.BatchNorm2d)
    )


def _iter_repvgg_conv3_branches(model: nn.Module) -> Iterable[nn.Sequential]:
    """Yield dense RepVGG-style 3x3 conv-BN branches."""
    for module in model.modules():
        branches = getattr(module, "conv3_branches", None)
        if branches is None:
            continue
        for branch in branches:
            if _is_conv_bn_branch(branch):
                yield branch


def _iter_mobileone_depthwise_3x3_branches(model: nn.Module) -> Iterable[nn.Sequential]:
    """Yield MobileOne's legacy depthwise-only 3x3 conv-BN branch scope.

    Kept for reproducing pre-fix runs: it boosts ONLY the 3x3 depthwise
    ``rbr_conv`` branches and silently skips every 1x1 pointwise block, even
    though those carry ``num_conv_branches`` branches too. Use
    ``mobileone_conv_branches`` (or ``auto``) for the faithful full scope.
    """
    for module in model.modules():
        branches = getattr(module, "rbr_conv", None)
        if branches is None or getattr(module, "inference_mode", False):
            continue
        if getattr(module, "kernel_size", None) == 3 and getattr(module, "groups", 1) > 1:
            for branch in branches:
                if _is_conv_bn_branch(branch):
                    yield branch


def _iter_mobileone_conv_branches(model: nn.Module) -> Iterable[nn.Sequential]:
    """Yield every reparameterizable MobileOne ``rbr_conv`` branch (3x3 + 1x1).

    Shares its definition with the Kaiming-sum init via
    ``iter_target_conv_branches`` so init and LR boost target the same branch
    set: the ``num_conv_branches`` convs of every non-stem depthwise and
    pointwise block.
    """
    from structural_reparam.models.mobileone_branch_mimic import (
        iter_target_conv_branches,
    )

    for branch in iter_target_conv_branches(model):
        if _is_conv_bn_branch(branch):
            yield branch


def _iter_all_conv_bn_branches(model: nn.Module) -> Iterable[nn.Sequential]:
    for module in model.modules():
        if _is_conv_bn_branch(module):
            yield module


def iter_boosted_conv_bn_branches(
    model: nn.Module, scope: str = "auto"
) -> Iterable[nn.Sequential]:
    """Yield conv-BN branches whose fused params should get LR boosts.

    Scopes:
      * auto: RepVGG conv3_branches plus the full MobileOne conv-branch scope.
      * repvgg_3x3: modules exposed through conv3_branches.
      * mobileone_conv_branches: every non-stem MobileOne rbr_conv branch
        (3x3 depthwise AND 1x1 pointwise) — the faithful k-fold scope.
      * mobileone_depthwise_3x3: legacy depthwise-only scope (pre-fix repro).
      * all_conv_bn: every Sequential(Conv2d, BatchNorm2d) pair.
    """
    if scope == "auto":
        yield from _iter_repvgg_conv3_branches(model)
        yield from _iter_mobileone_conv_branches(model)
    elif scope == "repvgg_3x3":
        yield from _iter_repvgg_conv3_branches(model)
    elif scope == "mobileone_conv_branches":
        yield from _iter_mobileone_conv_branches(model)
    elif scope == "mobileone_depthwise_3x3":
        yield from _iter_mobileone_depthwise_3x3_branches(model)
    elif scope == "all_conv_bn":
        yield from _iter_all_conv_bn_branches(model)
    else:
        raise ValueError(
            "target_branch_scope must be one of 'auto', 'repvgg_3x3', "
            "'mobileone_conv_branches', 'mobileone_depthwise_3x3', or "
            f"'all_conv_bn', got {scope!r}"
        )


class BiasBoostedSGD(torch.optim.SGD):
    """SGD with branch-mimic LR boosts on reparameterized conv branches.

    A k-branch Conv-BN block fuses to a single conv whose bias is the sum of the
    k branch BN biases and whose kernel is the sum of the k branch kernels. Under
    gradient flow, the highly-correlated branch regime motivates effective LR
    boosts on the fused parameters: k x for BN bias and k^2 x for conv weight.

    This optimizer mimics both with a single branch: selected BN biases and conv
    weights each go in their own param group at lr * mult; every other
    parameter keeps the base lr. target_branch_scope='auto' targets dense
    RepVGG conv3_branches and the full MobileOne conv-branch scope (every
    non-stem rbr_conv, 3x3 depthwise AND 1x1 pointwise — matching the
    Kaiming-sum init). Use mobileone_depthwise_3x3 for the legacy depthwise-only
    scope, or all_conv_bn to boost every Conv-BN pair.

    Both multipliers default to 1.0 (no-op: plain SGD split across equal-LR
    groups), so a config can point every variant here and only the boosted
    variants override. The cosine scheduler anneals each group from its own base
    LR, so the ratios hold for all of training.
    """

    def __init__(self, params, model: nn.Module, lr: float, momentum: float = 0.0,
                 weight_decay: float = 0.0, bias_lr_mult: float = 1.0,
                 weight_lr_mult: float = 1.0,
                 target_branch_scope: str = "auto") -> None:
        bias_ids: set[int] = set()
        biases: list[nn.Parameter] = []
        weight_ids: set[int] = set()
        weights: list[nn.Parameter] = []
        seen_branches: set[int] = set()
        for branch in iter_boosted_conv_bn_branches(model, target_branch_scope):
            if id(branch) in seen_branches:
                continue
            seen_branches.add(id(branch))
            conv = branch[0]
            bn = branch[1]
            bias = bn.bias
            if bias is not None and bias.requires_grad and id(bias) not in bias_ids:
                bias_ids.add(id(bias))
                biases.append(bias)
            weight = conv.weight
            if weight is not None and weight.requires_grad and id(weight) not in weight_ids:
                weight_ids.add(id(weight))
                weights.append(weight)

        if not biases and not weights:
            raise ValueError(
                "BiasBoostedSGD found no conv-BN branch params to boost; "
                f"target_branch_scope={target_branch_scope!r}. Expected a "
                "train-time RepVGG-style model or MobileOne-style model."
            )

        boosted_ids = bias_ids | weight_ids
        rest = [p for p in model.parameters() if id(p) not in boosted_ids]

        # `params` (the flat model.parameters()) is intentionally ignored — the
        # groups below cover exactly the same tensors, just split by LR.
        groups = [
            {"params": rest, "lr": lr},
            {"params": biases, "lr": lr * bias_lr_mult},
            {"params": weights, "lr": lr * weight_lr_mult},
        ]
        super().__init__(groups, lr=lr, momentum=momentum, weight_decay=weight_decay)
        self.bias_lr_mult = bias_lr_mult
        self.weight_lr_mult = weight_lr_mult
        self.target_branch_scope = target_branch_scope


def sgd_kernel_only_weight_decay(
    params: Iterable[nn.Parameter],
    model: nn.Module,
    lr: float,
    momentum: float = 0.0,
    weight_decay: float = 0.0,
    nesterov: bool = False,
) -> torch.optim.SGD:
    """SGD that applies weight decay only to conv/Linear weights (the kernels).

    Matches the RepVGG recipe (Ding et al. 2021, Sec. 4.1): "weight decay of
    1e-4 on the kernels of conv and fully-connected layers." BatchNorm affine
    params (gamma, beta) and every bias go in a no-decay group. The flat
    ``params`` arg is ignored — the two groups below cover exactly the same
    tensors, split by whether weight decay applies (mirrors BiasBoostedSGD).
    """
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    seen: set[int] = set()
    for module in model.modules():
        for name, p in module.named_parameters(recurse=False):
            if not p.requires_grad or id(p) in seen:
                continue
            seen.add(id(p))
            if isinstance(module, (nn.Conv2d, nn.Linear)) and name == "weight":
                decay.append(p)
            else:
                no_decay.append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.SGD(groups, lr=lr, momentum=momentum, nesterov=nesterov)


_OPTIMIZERS = {
    "sgd": torch.optim.SGD,
    "adam": torch.optim.Adam,
    "normalized_gd": NormalizedGD,
}


def build_optimizer(
    params: Iterable[nn.Parameter], name: str = "sgd", **kwargs: Any
) -> torch.optim.Optimizer:
    optimizer_cls = _OPTIMIZERS.get(name.lower())
    if optimizer_cls is None:
        supported = ", ".join(sorted(_OPTIMIZERS))
        raise ValueError(f"Unsupported optimizer: {name}. Supported: {supported}")
    return optimizer_cls(params, **kwargs)
