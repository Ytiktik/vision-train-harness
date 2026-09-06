"""Per-epoch mechanistic diagnostics for BN-MLP stall analysis.

Computes, at the end of each epoch on a class-balanced batch:

  - ``fc.weight`` / ``fc.bias`` gradient norms (smoking gun for the BN-erases-
    class-info stall: ``fc.weight.grad`` collapses toward 0 when hidden
    activations have lost per-class structure).
  - Per-branch ``Linear.weight`` gradient norm and parameter norm.
  - Per-branch BN ``gamma`` summary (mean / min / max) and two-branch gamma
    scale ratio — gamma collapsing to zero indicates a dead branch.
  - Dead-ReLU fraction at the hidden layer (units that never fire on the
    balanced batch in eval mode).
  - Per-layer input batch variance: mean feature-wise variance across examples
    in the balanced batch, measured at each ``BranchedLinear`` input and at
    the final ``fc`` input.
  - Class-feature separation = between-class variance / within-class variance
    of the fc-input features on the balanced batch. Goes to ~0 when the model
    has no class-discriminative features.

The probe runs one extra forward+backward per epoch on a balanced batch; it
snapshots and restores ``p.grad`` so it does not pollute the next training
step's gradient state.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn.grad import conv2d_weight
from torch.utils.data import DataLoader

from structural_reparam.analysis.registry import ProbeContext, register_probe
from structural_reparam.base_models.repvgg import RepVGGBlock
from structural_reparam.models.mobileone import MobileOneBlock
from structural_reparam.base_models.sum_conv_block import (
    AlphaCorrSigmaConvBlock,
    SharedScaleConvBlock,
    SingleBranchConvBlock,
    SingletonSharedScaleConvBlock,
    SingletonSqrtHalfSumSqSigmaConvBlock,
    SqrtHalfSumSqSigmaConvBlock,
)
from structural_reparam.models.mlp import (
    AlphaCorrSigmaBranchLinear,
    AvgSigmaCorrInvBranchLinear,
    AvgSigmaInvBranchLinear,
    BranchedLinear,
    BridgedSigmaBranchLinear,
    CenteredAvgSigmaBranchLinear,
    CenteredSigmaTotBranchLinear,
    JointSigmaCorrInvBranchLinear,
    JointSigmaHalfInvBranchLinear,
    JointSigmaInvBranchLinear,
    SGScaleBranchLinear,
    SharedScaleNBranchLinear,
    SingleSigmaBranchLinear,
    SingletonSqrtHalfSumSqSigmaBranchLinear,
    SqrtHalfSumSqSigmaBranchLinear,
    TwoBranchLinear,
    WNInvSigmaSumBranchLinear,
    WNSumBNBranchLinear,
)

_BRANCH_TYPES = (
    BranchedLinear,
    SharedScaleNBranchLinear,
    TwoBranchLinear,
    SingletonSqrtHalfSumSqSigmaBranchLinear,
    WNSumBNBranchLinear,
    WNInvSigmaSumBranchLinear,
    AvgSigmaInvBranchLinear,
    AvgSigmaCorrInvBranchLinear,
    JointSigmaInvBranchLinear,
    JointSigmaHalfInvBranchLinear,
    JointSigmaCorrInvBranchLinear,
    SGScaleBranchLinear,
    SingleSigmaBranchLinear,
    CenteredSigmaTotBranchLinear,
    CenteredAvgSigmaBranchLinear,
    BridgedSigmaBranchLinear,
    AlphaCorrSigmaBranchLinear,
    SqrtHalfSumSqSigmaBranchLinear,
    AlphaCorrSigmaConvBlock,
    SingleBranchConvBlock,
    SharedScaleConvBlock,
    SingletonSharedScaleConvBlock,
    SingletonSqrtHalfSumSqSigmaConvBlock,
    SqrtHalfSumSqSigmaConvBlock,
    RepVGGBlock,
    MobileOneBlock,
)


@register_probe("mechanistic")
class MechanisticProbe:
    def __init__(
        self,
        model: nn.Module,
        balanced_loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
        mobileone_conv_scope: str = "kernel3",
    ) -> None:
        self.model = model
        self.balanced_loader = balanced_loader
        self._balanced_iter = iter(balanced_loader)
        self.criterion = criterion
        self.device = device

        if mobileone_conv_scope not in ("kernel3", "all"):
            raise ValueError(
                "mobileone_conv_scope must be 'kernel3' (probe rbr_conv only in the "
                f"3x3 depthwise blocks) or 'all' (rbr_conv in every block), got "
                f"{mobileone_conv_scope!r}"
            )
        self.mobileone_conv_scope = mobileone_conv_scope

        self.branched_layers = [
            m
            for m in model.modules()
            if isinstance(m, _BRANCH_TYPES)
            and not (
                isinstance(m, MobileOneBlock)
                and mobileone_conv_scope == "kernel3"
                and m.kernel_size != 3
            )
        ]
        # MobileOne names its classifier head `linear`; the MLP/conv models use
        # `fc`. Resolve whichever final nn.Linear is present.
        head = getattr(model, "fc", None)
        if not isinstance(head, nn.Linear):
            head = getattr(model, "linear", None)
        if not isinstance(head, nn.Linear):
            raise ValueError("MechanisticProbe expects model.fc or model.linear to be nn.Linear.")
        self.fc: nn.Linear = head

    @classmethod
    def from_context(cls, ctx: ProbeContext) -> "MechanisticProbe":
        from structural_reparam.deploy.train import import_target, resolve_repo_path

        balanced_args = dict(ctx.config["dataset"].get("args", {}))
        # Apply the variant's dataset override (e.g. keep_classes) so the
        # balanced loader matches the model's num_classes — otherwise binary
        # variants build a 10-class loader and the probe's loss indexes out of
        # range. Mirrors build_loaders() in deploy/train.py.
        balanced_args.update(ctx.variant.get("dataset", {}).get("args", {}))
        balanced_args["batch_mode"] = "shuffled"
        balanced_args["num_workers"] = 0
        balanced_args["persistent_workers"] = False
        balanced_args.pop("classes_per_batch", None)
        balanced_args.pop("k_per_class", None)
        if "data_dir" in balanced_args:
            balanced_args["data_dir"] = resolve_repo_path(balanced_args["data_dir"])
        balanced_loader, _ = import_target(ctx.config["dataset"]["target"])(**balanced_args)
        return cls(
            model=ctx.model,
            balanced_loader=balanced_loader,
            criterion=ctx.criterion,
            device=ctx.device,
            mobileone_conv_scope=ctx.probe_config.get("mobileone_conv_scope", "kernel3"),
        )

    def close(self) -> None:
        return None

    def _next_balanced_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            return next(self._balanced_iter)
        except StopIteration:
            self._balanced_iter = iter(self.balanced_loader)
            return next(self._balanced_iter)

    def epoch_stats(self, epoch: int = 0) -> dict[str, float]:
        device = self.device
        x, y = self._next_balanced_batch()
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # Snapshot caller's gradients so we can restore after our backward.
        grad_snap: list[tuple[nn.Parameter, torch.Tensor | None]] = [
            (p, p.grad.detach().clone() if p.grad is not None else None)
            for p in self.model.parameters()
        ]

        was_training = self.model.training
        self.model.train()
        for p in self.model.parameters():
            p.grad = None

        # Capture per-branch pre-norm activations during the train-mode forward.
        branch_h: dict[tuple[int, int], torch.Tensor] = {}
        branch_sigma: dict[tuple[int, int], float] = {}
        branch_sigma_vec: dict[tuple[int, int], torch.Tensor] = {}
        branch_x_in: dict[tuple[int, int], torch.Tensor] = {}

        def make_sigma_hook(layer_idx: int, branch_idx: int):
            def hook(module: nn.Module, args: tuple) -> None:
                h_in = args[0].detach()
                stat_dims = (0,) if h_in.dim() == 2 else (0, 2, 3)
                sv = h_in.std(dim=stat_dims, unbiased=False)
                branch_h[(layer_idx, branch_idx)] = h_in
                branch_sigma[(layer_idx, branch_idx)] = sv.mean().item()
                branch_sigma_vec[(layer_idx, branch_idx)] = sv
            return hook

        def make_x_in_hook(layer_idx: int, branch_idx: int):
            def hook(module: nn.Module, args: tuple) -> None:
                branch_x_in[(layer_idx, branch_idx)] = args[0].detach()
            return hook

        sigma_handles = []
        for layer_idx, layer in enumerate(self.branched_layers):
            if isinstance(layer, (BranchedLinear, SharedScaleNBranchLinear, TwoBranchLinear)):
                for i, branch in enumerate(layer.branches):
                    sigma_handles.append(
                        branch[1].register_forward_pre_hook(make_sigma_hook(layer_idx, i))
                    )
                    sigma_handles.append(
                        branch[0].register_forward_pre_hook(make_x_in_hook(layer_idx, i))
                    )
            elif isinstance(layer, (SharedScaleConvBlock, SingletonSharedScaleConvBlock)):
                # branch1/branch2 are Sequential([Conv2d, BN2d]); hook BN pre-input.
                sigma_handles.append(
                    layer.branch1[1].register_forward_pre_hook(make_sigma_hook(layer_idx, 0))
                )
                sigma_handles.append(
                    layer.branch2[1].register_forward_pre_hook(make_sigma_hook(layer_idx, 1))
                )
                sigma_handles.append(
                    layer.branch1[0].register_forward_pre_hook(make_x_in_hook(layer_idx, 0))
                )
                sigma_handles.append(
                    layer.branch2[0].register_forward_pre_hook(make_x_in_hook(layer_idx, 1))
                )
            elif isinstance(layer, RepVGGBlock):
                # conv3_branches[i] is Sequential([Conv2d, BN2d]); hook BN pre-input
                # for σ and conv pre-input for x_in.
                for i, branch in enumerate(layer.conv3_branches):
                    sigma_handles.append(
                        branch[1].register_forward_pre_hook(make_sigma_hook(layer_idx, i))
                    )
                    sigma_handles.append(
                        branch[0].register_forward_pre_hook(make_x_in_hook(layer_idx, i))
                    )
                # 1x1 (conv1) and identity (identity_bn) auxiliary branches, when
                # present, carry their own BN — probe them like the 3x3 branches.
                if layer.conv1 is not None:
                    sigma_handles.append(
                        layer.conv1[1].register_forward_pre_hook(make_sigma_hook(layer_idx, "1x1"))
                    )
                    sigma_handles.append(
                        layer.conv1[0].register_forward_pre_hook(make_x_in_hook(layer_idx, "1x1"))
                    )
                if layer.identity_bn is not None:
                    sigma_handles.append(
                        layer.identity_bn.register_forward_pre_hook(
                            make_sigma_hook(layer_idx, "identity")
                        )
                    )
            elif isinstance(layer, MobileOneBlock):
                # rbr_conv[i] is Sequential([Conv2d, BN2d]); hook BN pre-input for σ
                # and conv pre-input for x_in.
                for i, branch in enumerate(layer.rbr_conv):
                    sigma_handles.append(
                        branch[1].register_forward_pre_hook(make_sigma_hook(layer_idx, i))
                    )
                    sigma_handles.append(
                        branch[0].register_forward_pre_hook(make_x_in_hook(layer_idx, i))
                    )
                # 1x1 scale (rbr_scale, a Conv-BN) and identity (rbr_skip, a bare
                # BN) branches, when present, carry their own BN — probe them like
                # the main conv branches.
                if layer.rbr_scale is not None:
                    sigma_handles.append(
                        layer.rbr_scale.bn.register_forward_pre_hook(
                            make_sigma_hook(layer_idx, "1x1")
                        )
                    )
                    sigma_handles.append(
                        layer.rbr_scale.conv.register_forward_pre_hook(
                            make_x_in_hook(layer_idx, "1x1")
                        )
                    )
                if layer.rbr_skip is not None:
                    sigma_handles.append(
                        layer.rbr_skip.register_forward_pre_hook(
                            make_sigma_hook(layer_idx, "identity")
                        )
                    )
            elif isinstance(layer, SingleBranchConvBlock):
                sigma_handles.append(
                    layer.bn.register_forward_pre_hook(make_sigma_hook(layer_idx, 0))
                )
                sigma_handles.append(
                    layer.conv.register_forward_pre_hook(make_x_in_hook(layer_idx, 0))
                )
            else:
                # Modules that store _branch_outs after each forward (set in mlp.py).
                def make_wn_hook(lidx: int):
                    def hook(module: nn.Module, args: tuple, output: torch.Tensor) -> None:
                        for bidx, h in enumerate(module._branch_outs):
                            h_d = h.detach()
                            stat_dims = (0,) if h_d.dim() == 2 else (0, 2, 3)
                            sv = h_d.std(dim=stat_dims, unbiased=False)
                            branch_h[(lidx, bidx)] = h_d
                            branch_sigma[(lidx, bidx)] = sv.mean().item()
                            branch_sigma_vec[(lidx, bidx)] = sv
                    return hook
                sigma_handles.append(layer.register_forward_hook(make_wn_hook(layer_idx)))
                # Capture linear inputs so BN-correction metrics work for these types too.
                if hasattr(layer, "linear1") and hasattr(layer, "linear2"):
                    sigma_handles.append(
                        layer.linear1.register_forward_pre_hook(make_x_in_hook(layer_idx, 0))
                    )
                    sigma_handles.append(
                        layer.linear2.register_forward_pre_hook(make_x_in_hook(layer_idx, 1))
                    )

        # Capture ∂L/∂y at each branched layer output (for w_- gradient alignment).
        grad_outputs: dict[int, torch.Tensor] = {}

        def make_grad_hook(layer_idx: int):
            def hook(module: nn.Module, grad_input: tuple, grad_output: tuple) -> None:
                if grad_output[0] is not None:
                    grad_outputs[layer_idx] = grad_output[0].detach()
            return hook

        grad_handles = [
            layer.register_full_backward_hook(make_grad_hook(layer_idx))
            for layer_idx, layer in enumerate(self.branched_layers)
        ]

        # register_full_backward_hook wraps layer outputs in views; inplace ReLU
        # would then modify those views, which PyTorch forbids. Disable inplace
        # temporarily for this forward-backward only.
        relu_modules = [m for m in self.model.modules() if isinstance(m, nn.ReLU) and m.inplace]
        for m in relu_modules:
            m.inplace = False

        logits = self.model(x)
        loss = self.criterion(logits, y)
        loss.backward()

        for m in relu_modules:
            m.inplace = True

        for handle in sigma_handles + grad_handles:
            handle.remove()

        fc_w_grad = (
            self.fc.weight.grad.detach().norm().item()
            if self.fc.weight.grad is not None else 0.0
        )
        fc_b_grad = (
            self.fc.bias.grad.detach().norm().item()
            if self.fc.bias is not None and self.fc.bias.grad is not None
            else 0.0
        )

        out: dict[str, float] = {
            "mech/fc_weight_grad_norm": fc_w_grad,
            "mech/fc_bias_grad_norm": fc_b_grad,
        }
        for layer_idx, layer in enumerate(self.branched_layers):
            if isinstance(layer, SharedScaleNBranchLinear):
                gamma_param = layer.gamma
                branch_linears = [(branch[0], gamma_param) for branch in layer.branches]
            elif isinstance(layer, TwoBranchLinear):
                branch_linears = [(branch[0], g) for branch, g in zip(layer.branches, layer.gammas)]
            elif isinstance(layer, BranchedLinear):
                branch_linears = [(branch[0], branch[1].weight) for branch in layer.branches]
            elif isinstance(layer, SharedScaleConvBlock):
                branch_linears = [
                    (layer.branch1[0], layer.gamma),
                    (layer.branch2[0], layer.gamma),
                ]
            elif isinstance(layer, SingletonSharedScaleConvBlock):
                branch_linears = [
                    (layer.branch1[0], layer.gamma1),
                    (layer.branch2[0], layer.gamma2),
                ]
            elif isinstance(layer, (SingletonSqrtHalfSumSqSigmaBranchLinear,)):
                branch_linears = [(layer.linear1, layer.gamma1), (layer.linear2, layer.gamma2)]
            elif isinstance(layer, (SingletonSqrtHalfSumSqSigmaConvBlock,)):
                branch_linears = [(layer.conv1, layer.gamma1), (layer.conv2, layer.gamma2)]
            elif isinstance(layer, RepVGGBlock):
                branch_linears = [(branch[0], branch[1].weight) for branch in layer.conv3_branches]
            elif isinstance(layer, MobileOneBlock):
                branch_linears = [(branch[0], branch[1].weight) for branch in layer.rbr_conv]
            elif isinstance(layer, SingleBranchConvBlock):
                branch_linears = [(layer.conv, layer.bn.weight)]
            else:
                gamma_param = layer.bn.weight if hasattr(layer, "bn") else layer.gamma
                if hasattr(layer, "conv1"):
                    branch_linears = [(layer.conv1, gamma_param), (layer.conv2, gamma_param)]
                else:
                    branch_linears = [(layer.linear1, gamma_param), (layer.linear2, gamma_param)]

            for i, (lin, gamma_param) in enumerate(branch_linears):
                key_prefix = f"mech/L{layer_idx}_branch{i+1}"
                g = lin.weight.grad
                out[f"{key_prefix}_W_grad_norm"] = (
                    g.detach().norm().item() if g is not None else 0.0
                )
                out[f"{key_prefix}_W_norm"] = lin.weight.detach().norm().item()
                gamma = gamma_param.detach()
                out[f"{key_prefix}_gamma_mean"] = gamma.mean().item()
                out[f"{key_prefix}_gamma_min"] = gamma.abs().min().item()
                out[f"{key_prefix}_gamma_max"] = gamma.max().item()

            # ── 1x1 + identity auxiliary branches (RepVGG / MobileOne) ─────────
            # When present these carry their own BN (and, for the 1x1, a conv).
            # Collect the same W/γ/σ diagnostics as the 3x3 branches so a dead or
            # collapsed 1x1 / identity branch is visible too. branch_sigma is keyed
            # by the string branch ids "1x1"/"identity" set during hook registration.
            conv1x1 = identity_bn = None
            if isinstance(layer, RepVGGBlock):
                conv1x1 = layer.conv1            # Sequential([Conv2d, BN]) or None
                identity_bn = layer.identity_bn  # BatchNorm2d or None
            elif isinstance(layer, MobileOneBlock):
                conv1x1 = layer.rbr_scale        # Sequential([conv, bn]) or None
                identity_bn = layer.rbr_skip     # BatchNorm2d or None

            if conv1x1 is not None:
                cp = f"mech/L{layer_idx}_conv1x1"
                conv_w, bn_1x1 = conv1x1[0], conv1x1[1]
                gw = conv_w.weight.grad
                out[f"{cp}_W_grad_norm"] = gw.detach().norm().item() if gw is not None else 0.0
                out[f"{cp}_W_norm"] = conv_w.weight.detach().norm().item()
                gamma = bn_1x1.weight.detach()
                out[f"{cp}_gamma_mean"] = gamma.mean().item()
                out[f"{cp}_gamma_min"] = gamma.abs().min().item()
                out[f"{cp}_gamma_max"] = gamma.max().item()
                gg = bn_1x1.weight.grad
                out[f"{cp}_gamma_grad_norm"] = gg.detach().norm().item() if gg is not None else 0.0
                s = branch_sigma.get((layer_idx, "1x1"))
                if s is not None:
                    out[f"{cp}_sigma"] = s

            if identity_bn is not None:
                ip = f"mech/L{layer_idx}_identity"
                gamma = identity_bn.weight.detach()
                out[f"{ip}_gamma_mean"] = gamma.mean().item()
                out[f"{ip}_gamma_min"] = gamma.abs().min().item()
                out[f"{ip}_gamma_max"] = gamma.max().item()
                gg = identity_bn.weight.grad
                out[f"{ip}_gamma_grad_norm"] = gg.detach().norm().item() if gg is not None else 0.0
                s = branch_sigma.get((layer_idx, "identity"))
                if s is not None:
                    out[f"{ip}_sigma"] = s

            if len(branch_linears) == 2:
                g1 = branch_linears[0][1].detach()
                g2 = branch_linears[1][1].detach()
                # Per-element ratio, then mean: mean(|γ₁ᵢ|/|γ₂ᵢ|). Differs from
                # mean(|γ₁|)/mean(|γ₂|) — captures per-feature scale imbalance
                # that cancels out in a ratio of means.
                out[f"mech/corr/L{layer_idx}_gamma_ratio"] = (
                    g1.abs() / g2.abs().clamp(min=1e-8)
                ).mean().item()
                # Signed per-branch γ means logged next to ρ so the corr panel
                # shows γ₁, γ₂, ρ together. The relative sign of γ₁,γ₂ vs the
                # sign of ρ is gauge-dependent; only γ₁γ₂ρ is meaningful.
                out[f"mech/corr/L{layer_idx}_gamma1"] = g1.mean().item()
                out[f"mech/corr/L{layer_idx}_gamma2"] = g2.mean().item()

            # W1/W2 cosine similarity (flattened weight vectors).
            if len(branch_linears) == 2:
                w1 = branch_linears[0][0].weight.detach().flatten()
                w2 = branch_linears[1][0].weight.detach().flatten()
                cos_sim = (w1 @ w2) / (w1.norm() * w2.norm()).clamp(min=1e-8)
                out[f"mech/corr/L{layer_idx}_W_cos_sim"] = cos_sim.item()

            # Sigma diagnostics: prefer _probe_data when available (gives true σ_eff
            # with α applied, plus σ₁, σ₂, σ₊=std(h₁+h₂), ρ=corr(h₁,h₂)).
            # Fall back to hook-captured branch sigmas for other layer types.
            lprefix = f"mech/L{layer_idx}"
            pd = getattr(layer, "_probe_data", None)
            if pd is not None:
                out[f"{lprefix}_sigma_eff"] = pd["sigma_eff"]
                out[f"{lprefix}_sigma1"] = pd["sigma1"]
                out[f"{lprefix}_sigma2"] = pd["sigma2"]
                if pd["sigma2"] > 1e-8:
                    out[f"mech/corr/L{layer_idx}_sigma_ratio"] = pd["sigma1"] / pd["sigma2"]
                out[f"{lprefix}_sigma_plus"] = pd["sigma_tot"]
                out[f"mech/corr/L{layer_idx}_rho"] = pd["rho"]
            else:
                s1 = branch_sigma.get((layer_idx, 0))
                s2 = branch_sigma.get((layer_idx, 1))
                if s1 is not None:
                    out[f"{lprefix}_sigma1"] = s1
                if s2 is not None:
                    out[f"{lprefix}_sigma2"] = s2
                if s1 is not None and s2 is not None and s2 > 1e-8:
                    out[f"mech/corr/L{layer_idx}_sigma_ratio"] = s1 / s2

            # δ/σ², R_ratio, w_minus_grad_cos: MLP-only (require 2-D activations).
            sv0 = branch_sigma_vec.get((layer_idx, 0))
            sv1 = branch_sigma_vec.get((layer_idx, 1))
            h0 = branch_h.get((layer_idx, 0))
            h1_b = branch_h.get((layer_idx, 1))
            is_mlp = (h0 is not None and h0.dim() == 2)
            if sv0 is not None and sv1 is not None and is_mlp:
                sigma_avg = (sv0 + sv1) / 2
                delta = (sv0 - sv1).abs()
                wm = delta / sigma_avg.pow(2).clamp(min=1e-8)
                out[f"mech/L{layer_idx}_wm_factor_mean"] = wm.mean().item()
                out[f"mech/L{layer_idx}_wm_factor_abs_mean"] = wm.abs().mean().item()

            if sv0 is not None and sv1 is not None and h0 is not None and h1_b is not None and is_mlp:
                sigma_avg_v = (sv0 + sv1) / 2
                delta_v = (sv0 - sv1).abs()
                h_plus = h0 + h1_b
                h_minus = h0 - h1_b
                c_plus = h_plus - h_plus.mean(dim=0)
                c_minus = h_minus - h_minus.mean(dim=0)
                w_plus_mag = (c_plus.abs() / sigma_avg_v.clamp(min=1e-8)).mean()
                wm_scale = delta_v / sigma_avg_v.pow(2).clamp(min=1e-8)
                w_minus_term = wm_scale * c_minus
                w_minus_mag = w_minus_term.abs().mean()
                R = w_minus_mag / w_plus_mag.clamp(min=1e-8)
                out[f"mech/L{layer_idx}_R_ratio"] = R.item()
                out[f"mech/L{layer_idx}_w_plus_mag"] = w_plus_mag.item()
                out[f"mech/L{layer_idx}_w_minus_mag"] = w_minus_mag.item()

                grad_out = grad_outputs.get(layer_idx)
                if grad_out is not None:
                    wm_flat = w_minus_term.flatten()
                    g_flat = grad_out.flatten()
                    wm_norm = wm_flat.norm().clamp(min=1e-8)
                    g_norm = g_flat.norm().clamp(min=1e-8)
                    out[f"mech/L{layer_idx}_w_minus_grad_cos"] = (
                        (wm_flat * g_flat).sum() / (wm_norm * g_norm)
                    ).item()

            # BN correction term T_i = (z̄_i / σ_i³) * Cₓwᵢ, where Cₓwᵢ = (1/N) Xᵀzᵢ.
            # Measures the σ-gradient part of the BN weight update that differs per branch.
            if is_mlp and len(branch_linears) == 2:
                corrections: list[torch.Tensor] = []
                for i, (lin, _) in enumerate(branch_linears):
                    z_i = branch_h.get((layer_idx, i))
                    x_i = branch_x_in.get((layer_idx, i))
                    if z_i is None or x_i is None:
                        continue
                    # z_i: (N, d_out) — BN input; x_i: (N, d_in) — linear input
                    sigma_i = z_i.std(dim=0, unbiased=False).clamp(min=1e-8)  # (d_out,)
                    z_bar = z_i.mean(dim=0)                                   # (d_out,)
                    Cxw = (z_i.T @ x_i) / z_i.shape[0]                       # (d_out, d_in)
                    T_i = (z_bar / sigma_i.pow(3))[:, None] * Cxw             # (d_out, d_in)
                    corrections.append(T_i)

                    T_norm = T_i.norm().item()
                    out[f"mech/L{layer_idx}_branch{i+1}_bn_correction_norm"] = T_norm
                    g = lin.weight.grad
                    if g is not None:
                        g_norm = g.detach().norm().clamp(min=1e-8).item()
                        out[f"mech/L{layer_idx}_branch{i+1}_bn_correction_frac"] = T_norm / g_norm
                    w_flat = lin.weight.detach().flatten()
                    T_flat = T_i.flatten()
                    T_fnorm = T_flat.norm().clamp(min=1e-8)
                    out[f"mech/L{layer_idx}_branch{i+1}_bn_correction_w_cos"] = (
                        (T_flat @ w_flat) / (T_fnorm * w_flat.norm().clamp(min=1e-8))
                    ).item()

                if len(corrections) == 2:
                    T1_flat = corrections[0].flatten()
                    T2_flat = corrections[1].flatten()
                    out[f"mech/L{layer_idx}_bn_correction_cos_sim"] = (
                        (T1_flat @ T2_flat)
                        / (T1_flat.norm().clamp(min=1e-8) * T2_flat.norm().clamp(min=1e-8))
                    ).item()

            # ── w₊ / w₋ curvature-correction regime check ──────────────────────
            # The fused-branch update is
            #   w₊ ← w₊ − η(eγ/σ)·( 2x − 1/(2σ²)[(w₊·x)Cₓw₊ + (w₋·x)Cₓw₋] ).
            # The single-branch approximation drops (w₋·x)Cₓw₋ when w₋·x ≪ 1. We
            # measure each correction term's magnitude (and its ratio) directly.
            #
            # With z_i = wᵢ·x the captured BN-input (branch_h), z± = z₁±z₂ gives
            # (w±·x), and Cₓw± is the σ-gradient direction. No explicit Cₓ or
            # upstream grad is needed — the common eγ/σ factor cancels in every
            # ratio below. The per-sample RMS of the scalar (w±·x) is σ± = std(z±),
            # so the RMS magnitude of the per-channel vector (w±·x)Cₓw± is σ±·‖Cₓw±‖.
            #   • Linear (2-D):  Cₓw± = (1/N)·z̃±ᵀX  (X = centered input patches = rows).
            #   • Conv   (4-D):  Cₓw± lives in patch space; it equals the conv
            #     weight-gradient using the centered pre-activation z̃± as the
            #     upstream gradient, normalised by the patch count M = N·Hₒ·Wₒ.
            #     (handles grouped/depthwise convs, e.g. MobileOne's 3×3 branches.)
            x0 = branch_x_in.get((layer_idx, 0))
            have_io = h0 is not None and h1_b is not None and x0 is not None
            sig_plus = sig_minus = cxw_plus_norm = cxw_minus_norm = None
            tr_Cx = None
            if have_io and h0.dim() == 2:
                N = h0.shape[0]
                xc = x0 - x0.mean(dim=0)
                z_plus, z_minus = h0 + h1_b, h0 - h1_b
                zc_plus = z_plus - z_plus.mean(dim=0)
                zc_minus = z_minus - z_minus.mean(dim=0)
                sig_plus = zc_plus.std(dim=0, unbiased=False)          # (d_out,) = σ₊
                sig_minus = zc_minus.std(dim=0, unbiased=False)        # (d_out,) = σ₋
                cxw_plus_norm = ((zc_plus.T @ xc) / N).norm(dim=1)     # ‖Cₓw₊‖ per channel
                cxw_minus_norm = ((zc_minus.T @ xc) / N).norm(dim=1)
                tr_Cx = xc.pow(2).sum(dim=1).mean().clamp(min=1e-12).sqrt()  # √tr(Cₓ)
            elif have_io and h0.dim() == 4 and isinstance(lin, nn.Conv2d):
                conv0 = branch_linears[0][0]
                N = h0.shape[0]
                xc = x0 - x0.mean(dim=(0, 2, 3), keepdim=True)
                z_plus, z_minus = h0 + h1_b, h0 - h1_b
                zc_plus = z_plus - z_plus.mean(dim=(0, 2, 3), keepdim=True)
                zc_minus = z_minus - z_minus.mean(dim=(0, 2, 3), keepdim=True)
                sig_plus = zc_plus.std(dim=(0, 2, 3), unbiased=False)  # (C_out,) = σ₊
                sig_minus = zc_minus.std(dim=(0, 2, 3), unbiased=False)
                M = N * z_plus.shape[2] * z_plus.shape[3]
                ws, st, pd_, dl, gr = (
                    conv0.weight.shape, conv0.stride, conv0.padding,
                    conv0.dilation, conv0.groups,
                )
                Cxw_plus = conv2d_weight(xc, ws, zc_plus, st, pd_, dl, gr) / M
                Cxw_minus = conv2d_weight(xc, ws, zc_minus, st, pd_, dl, gr) / M
                cxw_plus_norm = Cxw_plus.flatten(1).norm(dim=1)        # over (C_in/g,kh,kw)
                cxw_minus_norm = Cxw_minus.flatten(1).norm(dim=1)
                # tr(Cₓ) in patch space is not cheap here → skip the 2x comparison.

            if sig_plus is not None:
                sig2 = ((sv0 + sv1) / 2).pow(2).clamp(min=1e-8)        # σ² per channel

                # Per-channel RMS magnitude of each correction term, with the
                # derivation's 1/(2σ²) prefactor: term± = σ±·‖Cₓw±‖ / (2σ²).
                term_plus = sig_plus * cxw_plus_norm / (2 * sig2)
                term_minus = sig_minus * cxw_minus_norm / (2 * sig2)
                out[f"mech/L{layer_idx}_wplus_corr_norm"] = term_plus.mean().item()
                out[f"mech/L{layer_idx}_wminus_corr_norm"] = term_minus.mean().item()

                # KEY regime number: ‖(w₋·x)Cₓw₋‖ / ‖(w₊·x)Cₓw₊‖. ≪1 ⇒ dropping
                # the w₋ term is justified (the 1/(2σ²) prefactor cancels here).
                out[f"mech/corr/L{layer_idx}_wminus_corr_ratio"] = (
                    term_minus.norm() / term_plus.norm().clamp(min=1e-8)
                ).item()

                # The literal "w₋·x ≪ 1" scale and its dimensionless form σ₋/σ₊.
                out[f"mech/L{layer_idx}_sigma_minus"] = sig_minus.mean().item()
                out[f"mech/corr/L{layer_idx}_sigma_minus_ratio"] = (
                    sig_minus / sig_plus.clamp(min=1e-8)
                ).mean().item()

                # Does the curvature correction matter at all vs the leading 2x?
                # Per channel ‖term₊‖ / ‖2x‖, with ‖2x‖_rms = 2·√tr(Cₓ). (Linear only.)
                if tr_Cx is not None:
                    out[f"mech/L{layer_idx}_wplus_curv_strength"] = (
                        term_plus.mean() / (2 * tr_Cx)
                    ).item()

            # Per-feature/channel branch correlation and sigma_plus — skip if _probe_data
            # already has ρ/sigma_tot, but always compute sigma_plus as a fallback so it
            # is logged for all layer types (SharedScaleNBranchLinear etc.).
            if h0 is not None and h1_b is not None:
                stat_dims = (0,) if h0.dim() == 2 else (0, 2, 3)
                h0c = h0 - h0.mean(dim=stat_dims, keepdim=True)
                h1c = h1_b - h1_b.mean(dim=stat_dims, keepdim=True)
                if pd is None:
                    cov = (h0c * h1c).mean(dim=stat_dims)
                    s0 = h0.std(dim=stat_dims, unbiased=False).clamp(min=1e-8)
                    s1 = h1_b.std(dim=stat_dims, unbiased=False).clamp(min=1e-8)
                    out[f"mech/corr/L{layer_idx}_rho"] = (cov / (s0 * s1)).mean().item()
                if f"{lprefix}_sigma_plus" not in out:
                    out[f"{lprefix}_sigma_plus"] = (
                        (h0c + h1c).std(dim=stat_dims, unbiased=False).mean().item()
                    )

            # Gauge-invariant correlation: signed_ρ = sign(γ₁γ₂)·ρ. The raw sign
            # of ρ flips under the (γ₂,w₂)→(−γ₂,−w₂) gauge, but γ₁γ₂ρ does not,
            # so this isolates whether the branches reinforce (>0) or cancel (<0).
            rho_key = f"mech/corr/L{layer_idx}_rho"
            g1_key = f"mech/corr/L{layer_idx}_gamma1"
            g2_key = f"mech/corr/L{layer_idx}_gamma2"
            if rho_key in out and g1_key in out and g2_key in out:
                gsign = math.copysign(1.0, out[g1_key] * out[g2_key])
                out[f"mech/corr/L{layer_idx}_signed_rho"] = gsign * out[rho_key]

            # W_+/W_- norms: ||W₁+W₂||_F and ||W₁−W₂||_F (raw, pre-gamma).
            if isinstance(layer, SharedScaleNBranchLinear) and len(layer.branches) == 2:
                W1 = layer.branches[0][0].weight.detach()
                W2 = layer.branches[1][0].weight.detach()
                out[f"mech/L{layer_idx}_W_plus_norm"] = (W1 + W2).norm().item()
                out[f"mech/L{layer_idx}_W_minus_norm"] = (W1 - W2).norm().item()
            elif isinstance(layer, TwoBranchLinear) and len(layer.branches) == 2:
                W1 = layer.branches[0][0].weight.detach()
                W2 = layer.branches[1][0].weight.detach()
                g1 = layer.gammas[0].detach()[:, None]
                g2 = layer.gammas[1].detach()[:, None]
                W_plus = g1 * W1 + g2 * W2
                W_minus = g2 * W2 - g1 * W1
                out[f"mech/L{layer_idx}_W_plus_norm"] = W_plus.norm().item()
                out[f"mech/L{layer_idx}_W_minus_norm"] = W_minus.norm().item()
            elif isinstance(layer, BranchedLinear) and len(layer.branches) == 2:
                W1 = layer.branches[0][0].weight.detach()
                W2 = layer.branches[1][0].weight.detach()
                g1 = layer.branches[0][1].weight.detach()[:, None]
                g2 = layer.branches[1][1].weight.detach()[:, None]
                W_plus = g1 * W1 + g2 * W2
                W_minus = g2 * W2 - g1 * W1
                out[f"mech/L{layer_idx}_W_plus_norm"] = W_plus.norm().item()
                out[f"mech/L{layer_idx}_W_minus_norm"] = W_minus.norm().item()
            elif isinstance(layer, SharedScaleConvBlock):
                W1 = layer.branch1[0].weight.detach()
                W2 = layer.branch2[0].weight.detach()
                out[f"mech/L{layer_idx}_W_plus_norm"] = (W1 + W2).norm().item()
                out[f"mech/L{layer_idx}_W_minus_norm"] = (W1 - W2).norm().item()
            elif isinstance(layer, SingletonSharedScaleConvBlock):
                W1 = layer.branch1[0].weight.detach()
                W2 = layer.branch2[0].weight.detach()
                g1 = layer.gamma1.detach()[:, None, None, None]
                g2 = layer.gamma2.detach()[:, None, None, None]
                W_plus = g1 * W1 + g2 * W2
                W_minus = g2 * W2 - g1 * W1
                out[f"mech/L{layer_idx}_W_plus_norm"] = W_plus.norm().item()
                out[f"mech/L{layer_idx}_W_minus_norm"] = W_minus.norm().item()
            elif isinstance(layer, MobileOneBlock) and len(layer.rbr_conv) == 2:
                W1 = layer.rbr_conv[0][0].weight.detach()
                W2 = layer.rbr_conv[1][0].weight.detach()
                out[f"mech/L{layer_idx}_W_plus_norm"] = (W1 + W2).norm().item()
                out[f"mech/L{layer_idx}_W_minus_norm"] = (W1 - W2).norm().item()
            elif hasattr(layer, "linear1") and hasattr(layer, "linear2"):
                W1 = layer.linear1.weight.detach()
                W2 = layer.linear2.weight.detach()
                out[f"mech/L{layer_idx}_W_plus_norm"] = (W1 + W2).norm().item()
                out[f"mech/L{layer_idx}_W_minus_norm"] = (W1 - W2).norm().item()
            elif hasattr(layer, "conv1") and hasattr(layer, "conv2"):
                W1 = layer.conv1.weight.detach()
                W2 = layer.conv2.weight.detach()
                out[f"mech/L{layer_idx}_W_plus_norm"] = (W1 + W2).norm().item()
                out[f"mech/L{layer_idx}_W_minus_norm"] = (W1 - W2).norm().item()

        # Activation-side diagnostics: layer input variance, dead ReLU fraction,
        # and per-class feature separation. Capture tensors via hooks in eval mode.
        self.model.eval()
        captured: dict[str, torch.Tensor] = {}
        layer_inputs: dict[int, torch.Tensor] = {}

        def layer_input_hook(layer_idx: int):
            def hook(module, args):
                layer_inputs[layer_idx] = args[0].detach()
            return hook

        def fc_hook(module, args, output):
            captured["h"] = args[0].detach()

        layer_handles = [
            layer.register_forward_pre_hook(layer_input_hook(layer_idx))
            for layer_idx, layer in enumerate(self.branched_layers)
        ]
        h_handle = self.fc.register_forward_hook(fc_hook)
        with torch.no_grad():
            _ = self.model(x)
        h_handle.remove()
        for handle in layer_handles:
            handle.remove()

        for layer_idx in range(len(self.branched_layers)):
            layer_x = layer_inputs[layer_idx].flatten(1)
            out[f"mech/L{layer_idx}_input_batch_var_mean"] = (
                layer_x.var(dim=0, unbiased=False).mean().item()
            )

        h = captured["h"]  # (N, hidden)
        out["mech/fc_input_batch_var_mean"] = (
            h.flatten(1).var(dim=0, unbiased=False).mean().item()
        )

        dead_relu_frac = (h <= 0).all(dim=0).float().mean().item()

        classes = torch.unique(y)
        class_means: list[torch.Tensor] = []
        within_var_acc = torch.zeros((), device=device)
        count_acc = 0
        for c in classes:
            mask = y == c
            n_c = int(mask.sum().item())
            if n_c < 2:
                continue
            h_c = h[mask]
            class_means.append(h_c.mean(dim=0))
            within_var_acc += h_c.var(dim=0, unbiased=False).mean() * n_c
            count_acc += n_c
        if class_means and count_acc > 0:
            within_var = (within_var_acc / count_acc).item()
            between_var = torch.stack(class_means).var(dim=0, unbiased=False).mean().item()
            separation = between_var / (within_var + 1e-12)
        else:
            separation = float("nan")
        out["mech/dead_relu_frac"] = dead_relu_frac
        out["mech/class_feature_separation"] = separation

        # Restore caller's grad state and training mode.
        for p, g in grad_snap:
            if g is None:
                p.grad = None
            else:
                if p.grad is None:
                    p.grad = g.clone()
                else:
                    p.grad.copy_(g)
        if was_training:
            self.model.train()
        else:
            self.model.eval()

        return out
