"""Split-rescue on the REAL model: continue a trained single-branch checkpoint
as a function-preserving two-branch split (Minimal Synthetic Model, § Splitting
rescue, ported from LN branches to the paper's per-branch-BN RepVGG blocks):

    gamma*BN(Wx)+beta -> gamma/2*BN((W+zeta)x)+beta/2 + gamma/2*BN((W-zeta)x)+beta/2

Conv weights need no rescaling (BN is 0-homogeneous in W); per-branch BN
running stats are copied from the source branch, so at ``split_eps == 0`` the
forward pass (train and eval) is exactly the single-branch model's — and with
deterministic kernels the two branches then receive bitwise-identical
gradients and stay identical forever (the invariant-manifold control).

Entry module: delegates to the generic trainer with one addition — a
``cosine_tail`` scheduler that reproduces epochs ``start+1..total`` of a
T_max=total cosine schedule, so a 25-epoch continuation from ep75 sees exactly
the LR the 100-epoch reference runs saw over epochs 76-100. (The optimizer's
momentum buffers necessarily restart at zero — the checkpoints hold model
state only.)
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import torch
from torch import nn

from structural_reparam.experiments.reparam_sweeps100.lab import (
    ClaudeRepVGGBlock,
    LayerwiseRepVGGCifar,
)
from structural_reparam.analysis.registry import ProbeContext, register_probe

LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[4]

WANDB_PROJECT = "yoovi-t-tel-aviv-university/claude-autonomous-reparam"


def _fetch_checkpoint(artifact: str, filename: str, cache_dir: Path) -> Path:
    """Return the local path of ``filename`` from a W&B checkpoint artifact,
    downloading into ``cache_dir`` only if it is not already cached."""
    path = cache_dir / filename
    if path.exists():
        return path
    import wandb

    api = wandb.Api()
    art = api.artifact(f"{WANDB_PROJECT}/{artifact}", type="checkpoint")
    cache_dir.mkdir(parents=True, exist_ok=True)
    root = Path(art.download(root=str(cache_dir)))
    got = root / filename
    if not got.exists():
        raise FileNotFoundError(
            f"{filename} not in artifact {artifact}: "
            f"{sorted(p.name for p in root.iterdir())}"
        )
    return got


def _load_checkpoint_state(
    ckpt_group: str, ckpt_variant: str, ckpt_epoch: int, ckpt_seed: "int | str", ckpt_cache: str
) -> tuple[dict[str, torch.Tensor], int]:
    """Fetch + validate a source checkpoint; returns (state_dict, resolved seed)."""
    seed = torch.initial_seed() if ckpt_seed == "auto" else int(ckpt_seed)
    artifact = f"ckpt_{ckpt_group}_{ckpt_variant}_s{seed}:latest"
    cache = REPO_ROOT / ckpt_cache / f"{ckpt_group}_{ckpt_variant}_s{seed}"
    ckpt_path = _fetch_checkpoint(artifact, f"ckpt_ep{ckpt_epoch}.pt", cache)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if int(ck.get("seed", -1)) != seed or int(ck.get("epoch", -1)) != int(ckpt_epoch):
        raise ValueError(
            f"checkpoint mismatch: wanted seed={seed} epoch={ckpt_epoch}, "
            f"got seed={ck.get('seed')} epoch={ck.get('epoch')} ({ckpt_path})"
        )
    return ck["state_dict"], seed


class ContinueRepVGG(LayerwiseRepVGGCifar):
    """Continue a checkpoint AS-IS (no split): strict-load into an identically
    shaped model. The pure single-branch / pure two-branch anchors of the
    const-lr comparison — same schedule, no reparameterization event."""

    def __init__(
        self,
        *,
        ckpt_group: str,
        ckpt_variant: str,
        ckpt_epoch: int,
        ckpt_seed: "int | str" = "auto",
        ckpt_cache: str = "outputs/ckpt_cache",
        **model_args,
    ) -> None:
        super().__init__(**model_args)
        src_sd, seed = _load_checkpoint_state(
            ckpt_group, ckpt_variant, ckpt_epoch, ckpt_seed, ckpt_cache
        )
        self.load_state_dict(src_sd, strict=True)
        LOGGER.info(
            "ContinueRepVGG: resumed %s ep%s seed%s unchanged", ckpt_variant, ckpt_epoch, seed
        )


def build_arm(mode: str = "split", **args):
    """Config-level dispatcher: one model.target for mixed split/continue
    variants (the generic trainer fixes model.target per config)."""
    if mode == "split":
        return SplitRescueRepVGG(**args)
    if mode == "continue":
        return ContinueRepVGG(**args)
    raise ValueError(f"unknown arm mode {mode!r}")


class SplitRescueRepVGG(LayerwiseRepVGGCifar):
    """Two-branch ``LayerwiseRepVGGCifar`` initialized as the function-preserving
    split of a trained SINGLE-branch checkpoint (see module docstring).

    ``zeta`` is an isotropic direction with ``||zeta||_F = split_eps*||W||_F``
    per conv, drawn from a deterministic per-(seed, conv-index) generator.
    ``ckpt_seed: "auto"`` resolves to ``torch.initial_seed()`` — the generic
    trainer seeds each run with its sweep seed right before building the model,
    so every seed's run picks up its own source checkpoint.
    """

    def __init__(
        self,
        *,
        ckpt_group: str,
        ckpt_variant: str,
        ckpt_epoch: int,
        split_eps: float = 0.0,
        conv_scale: float = 1.0,
        ckpt_seed: "int | str" = "auto",
        ckpt_cache: str = "outputs/ckpt_cache",
        **model_args,
    ) -> None:
        super().__init__(**model_args)
        if any(n != 2 for n in self.block_num_3x3):
            raise ValueError(
                f"SplitRescueRepVGG needs num_3x3 == 2 everywhere, got {self.block_num_3x3}"
            )
        src_sd, seed = _load_checkpoint_state(
            ckpt_group, ckpt_variant, ckpt_epoch, ckpt_seed, ckpt_cache
        )
        self.load_split_state(
            src_sd, split_eps=float(split_eps), zeta_seed=seed,
            conv_scale=float(conv_scale),
        )
        LOGGER.info(
            "SplitRescueRepVGG: split init from %s ep%s seed%s, split_eps=%g, conv_scale=%g",
            ckpt_variant, ckpt_epoch, seed, split_eps, conv_scale,
        )

    def load_split_state(
        self,
        src_sd: dict[str, torch.Tensor],
        split_eps: float,
        zeta_seed: int,
        conv_scale: float = 1.0,
    ) -> None:
        """Map a single-branch state_dict onto this two-branch model:
        conv ``W -> conv_scale*(W+zeta), conv_scale*(W-zeta)``, BN affine
        ``gamma,beta -> gamma/2,beta/2`` per copy, BN running stats copied to
        both branches (mean scaled by ``conv_scale``, var by ``conv_scale**2``).

        ``conv_scale`` is a function-inert kernel-norm gauge (BN is
        0-homogeneous in W up to eps) that sets the branches' effective conv
        learning rate: with per-branch effective scale gamma/2, the pair's
        function-space kernel speed relative to the parent is
        ``2*(1/2)^2/conv_scale^2``. conv_scale=1 (default, all runs before
        2026-08-12) gives the historical x1/2 kernel cooling; 1/sqrt(2) makes
        the conv channel LR-neutral (the per-branch trainable affines still run
        at x2 — removing that too needs a fixed branch prefactor, not a gauge).
        """
        new_sd: dict[str, torch.Tensor] = {}
        conv_idx = 0
        for k, v in src_sd.items():
            if ".conv3_branches.0.0.weight" in k:
                w = v
                if split_eps > 0.0:
                    g = torch.Generator().manual_seed(zeta_seed * 1009 + conv_idx)
                    d = torch.randn(w.shape, generator=g, dtype=w.dtype)
                    zeta = d * (split_eps * w.norm() / d.norm().clamp(min=1e-12))
                else:
                    zeta = torch.zeros_like(w)
                conv_idx += 1
                new_sd[k] = (w + zeta) * conv_scale
                new_sd[k.replace(".conv3_branches.0.", ".conv3_branches.1.")] = (w - zeta) * conv_scale
            elif ".conv3_branches.0.1." in k:
                # per-branch BN: halve the affine; copy the running stats,
                # rescaled to match the conv_scale'd pre-BN activations
                for bi in (0, 1):
                    nk = k.replace(".conv3_branches.0.", f".conv3_branches.{bi}.")
                    if k.endswith(".1.weight") or k.endswith(".1.bias"):
                        new_sd[nk] = v * 0.5
                    elif k.endswith(".running_mean"):
                        new_sd[nk] = v * conv_scale
                    elif k.endswith(".running_var"):
                        new_sd[nk] = v * conv_scale**2
                    else:
                        new_sd[nk] = v.clone()
            else:
                new_sd[k] = v.clone()
        self.load_state_dict(new_sd, strict=True)


@register_probe("branch_div")
class BranchDivergenceProbe:
    """Per-epoch branch-divergence probe — the D-mode growth signal of the
    split-rescue experiment.

    Per two-branch block:
      branch_div/blockN/rel        — ||W1-W2||_F / ||W1+W2||_F  (2||D||/2||W||)
      branch_div/blockN/cos        — cosine(W1, W2)
      branch_div/blockN/gamma_rel  — same ratio for the per-branch BN gammas
    Network aggregates: net_rel_median, net_rel_mean.
    """

    def __init__(self, model: nn.Module) -> None:
        self.blocks = [
            m for m in model.modules()
            if isinstance(m, ClaudeRepVGGBlock) and m.num_3x3 == 2
        ]

    @classmethod
    def from_context(cls, ctx: ProbeContext) -> "BranchDivergenceProbe":
        return cls(model=ctx.model)

    def epoch_stats(self, epoch: int = 0) -> dict[str, float]:
        stats: dict[str, float] = {}
        rels: list[float] = []
        for i, block in enumerate(self.blocks):
            w1, w2 = (w.detach().float() for w in block._conv3_weights())
            rel = ((w1 - w2).norm() / (w1 + w2).norm().clamp(min=1e-12)).item()
            cos = torch.nn.functional.cosine_similarity(
                w1.flatten(), w2.flatten(), dim=0
            ).item()
            if block.bn_position == "per_branch":
                g1 = block.conv3_branches[0][1].weight.detach().float()
                g2 = block.conv3_branches[1][1].weight.detach().float()
                gamma_rel = ((g1 - g2).norm() / (g1 + g2).norm().clamp(min=1e-12)).item()
            else:  # shared scale (post_sum / weight_norm): no per-branch gamma
                gamma_rel = 0.0
            stats[f"branch_div/block{i}/rel"] = rel
            stats[f"branch_div/block{i}/cos"] = cos
            stats[f"branch_div/block{i}/gamma_rel"] = gamma_rel
            rels.append(rel)
        if rels:
            t = torch.tensor(rels)
            stats["branch_div/net_rel_median"] = t.median().item()
            stats["branch_div/net_rel_mean"] = t.mean().item()
        return stats

    def close(self) -> None:
        pass


class GatedPairSGD(torch.optim.SGD):
    """SGD whose branch-difference mode is frozen after a set number of steps —
    the CIFAR analogue of the minimal model's ``close_at(t)`` gate (§ *How much
    of the benefit is escape?*). While the gate is open this is exactly SGD
    (weight decay is materialized into the grads here, with the optimizer's
    native wd kept at 0, so open-gate steps match native SGD's math). After
    ``close_after_steps`` steps, each two-branch pair (conv W1/W2, BN gamma,
    BN beta) receives the MEAN of its two (wd-inclusive) gradients, and the
    momentum buffers are averaged once at the flip — both branches then take
    identical steps forever: D is frozen at its current value, not erased."""

    def __init__(self, params, pairs, close_after_steps, weight_decay=0.0, **kw):
        super().__init__(params, weight_decay=0.0, **kw)
        self._pairs = pairs
        self._wd = float(weight_decay)
        self._close_after = int(close_after_steps)
        self._step_count = 0
        self._flipped = False

    @torch.no_grad()
    def _flip_momentum(self) -> None:
        for p1, p2 in self._pairs:
            s1, s2 = self.state.get(p1), self.state.get(p2)
            b1 = s1.get("momentum_buffer") if s1 else None
            b2 = s2.get("momentum_buffer") if s2 else None
            if b1 is not None and b2 is not None:
                m = (b1 + b2).mul_(0.5)
                b1.copy_(m)
                b2.copy_(m)

    @torch.no_grad()
    def step(self, closure=None):
        if self._wd:
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is not None:
                        p.grad.add_(p, alpha=self._wd)
        if self._step_count >= self._close_after:
            if not self._flipped:
                self._flip_momentum()
                self._flipped = True
            for p1, p2 in self._pairs:
                if p1.grad is None or p2.grad is None:
                    continue
                g = (p1.grad + p2.grad).mul_(0.5)
                p1.grad.copy_(g)
                p2.grad.copy_(g)
        self._step_count += 1
        return super().step(closure)


def build_gated_sgd(
    params,
    model: nn.Module,
    lr: float,
    momentum: float = 0.0,
    weight_decay: float = 0.0,
    nesterov: bool = False,
    close_after_epochs: float = 0.0,
    steps_per_epoch: int = 391,
    **_: object,
) -> torch.optim.Optimizer:
    """Optimizer target for close_at experiments: gate closes after
    ``close_after_epochs`` (CIFAR-100 b128 -> 391 steps/epoch)."""
    pairs: list[tuple[nn.Parameter, nn.Parameter]] = []
    for m in model.modules():
        if isinstance(m, ClaudeRepVGGBlock) and m.num_3x3 == 2:
            b1, b2 = m.conv3_branches
            pairs.append((b1[0].weight, b2[0].weight))
            pairs.append((b1[1].weight, b2[1].weight))
            pairs.append((b1[1].bias, b2[1].bias))
    if not pairs:
        raise ValueError("build_gated_sgd: no two-branch blocks found to gate")
    return GatedPairSGD(
        params,
        pairs=pairs,
        close_after_steps=round(float(close_after_epochs) * int(steps_per_epoch)),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=nesterov,
    )


_ORIG_BUILD_SCHEDULER = None


def _build_scheduler_with_tail(config, optimizer):
    train_config = config["train"]
    if train_config.get("scheduler") == "cosine_tail":
        total = int(train_config["scheduler_total_epochs"])
        start = int(train_config["scheduler_start_epoch"])

        def factor(last_epoch: int) -> float:
            # During continuation epoch j (1-based) the LR must equal the
            # original run's during epoch start+j: lr0*(1+cos(pi*(start+j-1)/total))/2.
            # LambdaLR is queried with last_epoch = j-1.
            return 0.5 * (1.0 + math.cos(math.pi * (start + last_epoch) / total))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    return _ORIG_BUILD_SCHEDULER(config, optimizer)


def main() -> None:
    global _ORIG_BUILD_SCHEDULER
    from structural_reparam.deploy import train as train_mod

    _ORIG_BUILD_SCHEDULER = train_mod.build_scheduler
    train_mod.build_scheduler = _build_scheduler_with_tail
    train_mod.main()


if __name__ == "__main__":
    main()
