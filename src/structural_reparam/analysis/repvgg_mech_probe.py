"""Per-epoch weight diagnostics for CifarRepVGG skip-connection ablations.

Tracks at the end of each epoch:
  - BN gamma (γ) statistics across all blocks, broken out by branch type
    (3x3 conv branch, identity branch).
  - ||w||² (squared Frobenius norm) of the 3x3 conv weights per block,
    averaged across all blocks.
  - Pre-BN activation σ for the 1x1 and identity branches: the per-channel
    std of each BN's input on a balanced batch, averaged over the block's
    channels then over blocks. σ is an *activation* statistic (not a weight),
    so this requires one extra forward pass; it is done in eval mode under
    ``no_grad`` so it never pollutes the training gradient state.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

from structural_reparam.analysis.registry import ProbeContext, register_probe
from structural_reparam.base_models.repvgg import RepVGGBlock
from structural_reparam.models.cifar_repvgg import CifarRepVGG


@register_probe("repvgg_mech")
class RepVGGMechProbe:
    def __init__(
        self,
        model: CifarRepVGG,
        balanced_loader: DataLoader,
        device: torch.device,
    ) -> None:
        if not isinstance(model, CifarRepVGG):
            raise ValueError("RepVGGMechProbe requires a CifarRepVGG model.")
        self.model = model
        self.balanced_loader = balanced_loader
        self._balanced_iter = iter(balanced_loader)
        self.device = device

    @classmethod
    def from_context(cls, ctx: ProbeContext) -> "RepVGGMechProbe":
        from structural_reparam.deploy.train import import_target, resolve_repo_path

        # Build a plain shuffled loader for the σ forward pass. Mirrors the
        # mechanistic probe: apply the variant's dataset override so the loader
        # matches the model, and force single-worker / non-persistent so the
        # extra per-epoch pass stays cheap.
        balanced_args = dict(ctx.config["dataset"].get("args", {}))
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
            model=ctx.model,  # type: ignore[arg-type]
            balanced_loader=balanced_loader,
            device=ctx.device,
        )

    def close(self) -> None:
        return None

    def _next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            return next(self._balanced_iter)
        except StopIteration:
            self._balanced_iter = iter(self.balanced_loader)
            return next(self._balanced_iter)

    def epoch_stats(self, epoch: int = 0) -> dict[str, float]:
        blocks = [m for m in self.model.modules() if isinstance(m, RepVGGBlock)]

        conv_gammas: list[float] = []
        identity_gammas: list[float] = []
        conv1_gammas: list[float] = []
        conv_w_sq_norms: list[float] = []
        conv1_w_sq_norms: list[float] = []

        for block in blocks:
            # 3x3 branch: BN gamma and ||w||²
            for branch in block.conv3_branches:
                conv, bn = branch[0], branch[1]
                conv_gammas.append(bn.weight.detach().abs().mean().item())
                conv_w_sq_norms.append(conv.weight.detach().pow(2).sum().item())

            # 1x1 branch: BN gamma and ||w||² (the branch this ablation isolates;
            # its gamma magnitude relative to the identity gamma shows how much
            # the learned 1x1 residual path carries beyond a bare skip).
            if block.conv1 is not None:
                conv1, conv1_bn = block.conv1[0], block.conv1[1]
                conv1_gammas.append(conv1_bn.weight.detach().abs().mean().item())
                conv1_w_sq_norms.append(conv1.weight.detach().pow(2).sum().item())

            # Identity branch BN gamma (only when BN exists)
            if block.identity_bn is not None:
                identity_gammas.append(
                    block.identity_bn.weight.detach().abs().mean().item()
                )

        # ── Pre-BN activation σ for the 1x1 and identity branches ──────────────
        # σ is the per-channel std of each BN's *input* on a real batch; capture
        # it via forward-pre-hooks (matching the mechanistic probe convention of
        # hooking the BN input). One eval/no_grad forward pass, hooks removed
        # after, so the training gradient state is untouched.
        conv1_sigmas, identity_sigmas = self._collect_branch_sigmas(blocks)

        out: dict[str, float] = {}

        if conv_gammas:
            t = torch.tensor(conv_gammas)
            out["repvgg_mech/conv_gamma_mean"] = t.mean().item()
            out["repvgg_mech/conv_gamma_std"] = t.std().item()

        if conv1_gammas:
            t = torch.tensor(conv1_gammas)
            out["repvgg_mech/conv1_gamma_mean"] = t.mean().item()
            out["repvgg_mech/conv1_gamma_std"] = t.std().item()

        if identity_gammas:
            t = torch.tensor(identity_gammas)
            out["repvgg_mech/identity_gamma_mean"] = t.mean().item()
            out["repvgg_mech/identity_gamma_std"] = t.std().item()

        if conv_w_sq_norms:
            t = torch.tensor(conv_w_sq_norms)
            out["repvgg_mech/conv_w_sq_norm_mean"] = t.mean().item()
            out["repvgg_mech/conv_w_sq_norm_std"] = t.std().item()

        if conv1_w_sq_norms:
            t = torch.tensor(conv1_w_sq_norms)
            out["repvgg_mech/conv1_w_sq_norm_mean"] = t.mean().item()
            out["repvgg_mech/conv1_w_sq_norm_std"] = t.std().item()

        if conv1_sigmas:
            t = torch.tensor(conv1_sigmas)
            out["repvgg_mech/conv1_sigma_mean"] = t.mean().item()
            out["repvgg_mech/conv1_sigma_std"] = t.std().item()

        if identity_sigmas:
            t = torch.tensor(identity_sigmas)
            out["repvgg_mech/identity_sigma_mean"] = t.mean().item()
            out["repvgg_mech/identity_sigma_std"] = t.std().item()

        return out

    def _collect_branch_sigmas(
        self, blocks: list[RepVGGBlock]
    ) -> tuple[list[float], list[float]]:
        captured: dict[tuple[int, str], float] = {}

        def make_hook(key: tuple[int, str]):
            def hook(module: nn.Module, args: tuple) -> None:
                h = args[0].detach()
                stat_dims = (0,) if h.dim() == 2 else (0, 2, 3)
                captured[key] = h.std(dim=stat_dims, unbiased=False).mean().item()
            return hook

        handles = []
        for i, block in enumerate(blocks):
            if block.conv1 is not None:
                handles.append(
                    block.conv1[1].register_forward_pre_hook(make_hook((i, "1x1")))
                )
            if block.identity_bn is not None:
                handles.append(
                    block.identity_bn.register_forward_pre_hook(make_hook((i, "identity")))
                )

        if not handles:
            return [], []

        x, _ = self._next_batch()
        x = x.to(self.device, non_blocking=True)
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                self.model(x)
        finally:
            if was_training:
                self.model.train()
            for h in handles:
                h.remove()

        conv1_sigmas = [captured[(i, "1x1")] for i in range(len(blocks)) if (i, "1x1") in captured]
        identity_sigmas = [
            captured[(i, "identity")] for i in range(len(blocks)) if (i, "identity") in captured
        ]
        return conv1_sigmas, identity_sigmas
