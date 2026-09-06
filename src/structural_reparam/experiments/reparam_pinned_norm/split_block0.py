"""Infinitesimal block-0 split of a trained single, continued under the
constant-rate kernel-decay recipe.

Takes a single checkpoint (one-branch blocks, ``single_d3_wdg``), splits block 0
into a shared-gamma pair: w1 = w + zeta*|w|*d, w2 = w - zeta*|w|*d with d a
per-channel random unit direction (zeta 0 = exact collapsed control), gamma/2,
running stats copied to both branches, one trainable bias.  Continues training
to --total-epochs with SGD momentum 0.9, constant lr, kernel-only weight decay
(same grouping as build_kernel_wd_sgd), fresh momentum buffers.

Logs one JSON line per epoch: running train accuracy, block-0 gamma quantiles
(relative to the split-time value), Euclidean and Sigma-whitened branch cosine
(median, fraction negative, against the committed patch covariance), branch
kernel norms.  Usage (GPU box):

  python -m structural_reparam.experiments.reparam_pinned_norm.split_block0 \
    --ckpt outputs/.../checkpoints/single_d3_wdg_seed42/ckpt_ep20.pt \
    --start-epoch 20 --total-epochs 100 --zeta 0.01 --seed 42 \
    --data-dir data/cifar100 --out outputs/wdconv_gauge_c100_d3/split0/split_e20_z001_s42.jsonl
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

import importlib
from structural_reparam.experiments.reparam_pinned_norm.lab import (
    PlacedSharedScaleRepVGGCifar,
    build_kernel_wd_sgd,
)


def split_single_sd(sd: dict, zeta: float, gen: torch.Generator, gauged: bool = False) -> dict:
    """Map a single ([1,1,1]) state dict onto the [2,1,1] pair model.

    ``gauged``: scale each branch kernel by 1/sqrt(2) (running_var by 1/2,
    running_mean by 1/sqrt(2)) -- function-preserving under BN, puts the branch
    norms exactly at the weight-decay equilibrium for the halved gamma and
    matches the single's per-branch angular step (the learning-rate-clean
    sqrt(2) gauge).  Combine with a gamma lr multiplier 0.25 on block 0 so the
    shared gamma (which collects both branches' gradient) mirrors the single's
    gamma trajectory."""
    out = {}
    for k, v in sd.items():
        out[k] = v.clone()
    w = sd["stages.0.0.convs.0.weight"]
    C = w.shape[0]
    d = torch.randn(w.shape, generator=gen, dtype=w.dtype)
    d = d / d.flatten(1).norm(dim=1).clamp_min(1e-12).view(C, 1, 1, 1)
    r = w.flatten(1).norm(dim=1).view(C, 1, 1, 1)
    s = 2.0 ** -0.5 if gauged else 1.0
    out["stages.0.0.convs.0.weight"] = s * (w + zeta * r * d)
    out["stages.0.0.convs.1.weight"] = s * (w - zeta * r * d)
    out["stages.0.0.gamma"] = sd["stages.0.0.gamma"] / 2.0
    for nm in ("running_mean", "running_var", "num_batches_tracked"):
        key = f"stages.0.0.stats.0.{nm}"
        if key in sd:
            fac = 1.0
            if gauged and nm == "running_mean":
                fac = s
            if gauged and nm == "running_var":
                fac = s * s
            out[key] = sd[key].clone() * fac if sd[key].is_floating_point() else sd[key].clone()
            out[f"stages.0.0.stats.1.{nm}"] = out[key].clone()
    out["stages.0.0.betas.0"] = sd["stages.0.0.betas.0"].clone()
    out["stages.0.0.betas.1"] = torch.zeros_like(sd["stages.0.0.betas.0"])
    return out


@torch.no_grad()
def top_eigvec(cov: torch.Tensor) -> torch.Tensor:
    """Unit top eigenvector of the patch covariance -- the flat/brightness mode."""
    ev, V = torch.linalg.eigh(cov.double())
    e = V[:, int(ev.argmax())].float()
    return e if float(e.sum()) >= 0 else -e


@torch.no_grad()
def damp_top_grad(weights, e: torch.Tensor, c: float, renorm: bool, wd_e: float = 0.0) -> None:
    """Multiply each block-0 kernel gradient's component along ``e`` by ``c``.

    ``e`` is Sigma's top eigenvector, i.e. exactly the direction the open pair
    deletes from its own tangent space by parking on it.  c=1 is a no-op, c=0
    freezes that direction.  ``renorm`` restores the per-channel gradient norm
    so the arm is purely directional and carries no step-size change.  ``wd_e``
    (the weight decay) subtracts the decay's own pull along ``e``, so the
    kernel's brightness component is frozen rather than decaying away.
    """
    for w in weights:
        if w.grad is None:
            continue
        g = w.grad.view(w.shape[0], -1)
        n0 = g.norm(dim=1, keepdim=True) if renorm else None
        g.add_(torch.outer((g @ e) * (c - 1.0), e))
        if wd_e:
            g.add_(torch.outer((w.detach().view(w.shape[0], -1) @ e) * (-wd_e), e))
        if renorm:
            g.mul_(n0 / g.norm(dim=1, keepdim=True).clamp_min(1e-12))


@torch.no_grad()
def block0_stats(model: nn.Module, cov: torch.Tensor, g_ref: torch.Tensor) -> dict:
    blk = model.stages[0][0]
    e_top = top_eigvec(cov)
    a_top = {f"a_top_b{i}": float(((cv.weight.detach().flatten(1).float().cpu() @ e_top)
                                   / cv.weight.detach().flatten(1).float().cpu().norm(dim=1).clamp_min(1e-12))
                                  .abs().median())
             for i, cv in enumerate(blk.convs)}
    if len(blk.convs) < 2:
        g = blk.gamma.detach().abs().float().cpu()
        rel = g / g_ref.clamp_min(1e-12)
        q = lambda t, p: float(torch.quantile(t, p))
        return {"g_rel_p10": q(rel, 0.10), "g_rel_p50": q(rel, 0.50), "g_rel_p90": q(rel, 0.90), **a_top}
    g = blk.gamma.detach().abs().float().cpu()
    a = blk.convs[0].weight.detach().flatten(1).float().cpu()
    b = blk.convs[1].weight.detach().flatten(1).float().cpu()
    cos = (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1)).clamp_min(1e-12)
    Sa, Sb = a @ cov, b @ cov
    cw = (Sa * b).sum(1) / ((Sa * a).sum(1).clamp_min(1e-12).sqrt() * (Sb * b).sum(1).clamp_min(1e-12).sqrt())
    rel = g / g_ref.clamp_min(1e-12)
    q = lambda t, p: float(torch.quantile(t, p))
    return {
        "g_rel_p10": q(rel, 0.10), "g_rel_p50": q(rel, 0.50), "g_rel_p90": q(rel, 0.90),
        "cos_p50": q(cos, 0.5), "cos_frac_neg": float((cos < 0).float().mean()),
        "cosw_p50": q(cw, 0.5), "cosw_p10": q(cw, 0.1), "cosw_frac_neg": float((cw < 0).float().mean()),
        "n1_p50": q(a.norm(dim=1), 0.5), "n2_p50": q(b.norm(dim=1), 0.5), **a_top,
    }



@torch.no_grad()
def block0_arrays(model: nn.Module, cov: torch.Tensor) -> dict:
    """Per-channel block-0 arrays (lists of length C): |gamma|, per-branch kernel
    norm, per-branch running sigma (sqrt(running_var + eps)), per-branch
    covariance-proxy sigma (sqrt(w^T Cov w) against the committed CIFAR-100
    patch covariance), and for pairs the Euclidean and Sigma-whitened branch
    cosines.  Keys are prefixed ``pc_``."""
    blk = model.stages[0][0]
    out = {"pc_gamma": blk.gamma.detach().abs().float().cpu().tolist()}
    ws = []
    for b, conv in enumerate(blk.convs):
        w = conv.weight.detach().flatten(1).float().cpu()
        ws.append(w)
        out[f"pc_norm_b{b}"] = w.norm(dim=1).tolist()
        out[f"pc_sigma_cov_b{b}"] = torch.einsum("cd,de,ce->c", w, cov, w).clamp_min(0).sqrt().tolist()
        st = blk.stats[b]
        out[f"pc_run_sigma_b{b}"] = (st.running_var.detach().float().cpu() + st.eps).sqrt().tolist()
    if len(ws) >= 2:
        a, b2 = ws[0], ws[1]
        out["pc_cos"] = ((a * b2).sum(1) / (a.norm(dim=1) * b2.norm(dim=1)).clamp_min(1e-12)).tolist()
        Sa, Sb = a @ cov, b2 @ cov
        cw = (Sa * b2).sum(1) / ((Sa * a).sum(1).clamp_min(1e-12).sqrt() * (Sb * b2).sum(1).clamp_min(1e-12).sqrt())
        out["pc_cosw"] = cw.tolist()
    return out


@torch.no_grad()
def save_block0(model: nn.Module, path: Path) -> None:
    """Dump block 0's tensors (gamma, every branch kernel, running stats) so the
    branch geometry can be recomputed offline against the patch covariance."""
    blk = model.stages[0][0]
    out = {"gamma": blk.gamma.detach().float().cpu()}
    for b, conv in enumerate(blk.convs):
        out[f"w{b}"] = conv.weight.detach().float().cpu()
        st = blk.stats[b]
        out[f"running_var{b}"] = st.running_var.detach().float().cpu()
        out[f"running_mean{b}"] = st.running_mean.detach().float().cpu()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--start-epoch", type=int, required=True)
    ap.add_argument("--total-epochs", type=int, default=100)
    ap.add_argument("--zeta", type=float, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--data-dir", default="data/cifar100")
    ap.add_argument("--dataset", default="cifar100", help="module under structural_reparam.data providing build_loaders")
    ap.add_argument("--num-classes", type=int, default=100)
    ap.add_argument("--cov", type=Path, default=None, help="patch covariance .pt (defaults to the committed CIFAR-100 one)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gauged", action="store_true")
    ap.add_argument("--no-split", action="store_true", help="control: continue the unsplit single under this loop")
    ap.add_argument("--per-channel", action="store_true", help="log per-channel block-0 arrays each epoch")
    ap.add_argument("--block0-kernel-lr-mult", type=float, default=1.0,
                    help="multiply the learning rate of block 0's KERNELS only (gamma and the biases keep the "
                         "global rate). At the decay wall this multiplies the tangential rate by its square root, "
                         "since r scales as sqrt(eta): m=4 is a twice-hotter block-0 direction, permanently.")
    ap.add_argument("--block0-damp-top", type=float, default=1.0,
                    help="retained fraction of block 0's KERNEL gradient along the patch covariance's top "
                         "eigenvector (the flat/brightness mode). 1.0 is a no-op; 0.0 freezes that direction. "
                         "This is the direction the open pair deletes from its own tangent space.")
    ap.add_argument("--damp-renorm", action="store_true",
                    help="rescale the damped gradient back to its per-channel norm, so the arm is purely "
                         "directional and carries no reduction in step size")
    ap.add_argument("--damp-freeze-decay", action="store_true",
                    help="also cancel weight decay along the damped direction, so the kernel's brightness "
                         "component is frozen rather than decaying away")
    ap.add_argument("--save-block0", type=Path, default=None, help="directory for per-epoch block-0 tensor dumps")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    dev = torch.device(a.device)

    ck = torch.load(a.ckpt, map_location="cpu")
    sd = ck["state_dict"] if "state_dict" in ck else ck
    gen = torch.Generator().manual_seed(a.seed * 1000 + a.start_epoch)
    if a.no_split:
        model = PlacedSharedScaleRepVGGCifar(
            num_classes=a.num_classes, stage_blocks=[1, 1, 1], block_branches=[1, 1, 1],
            mode="shared_gamma")
        model.load_state_dict(sd, strict=True)
    else:
        pair_sd = split_single_sd(sd, a.zeta, gen, gauged=a.gauged)
        model = PlacedSharedScaleRepVGGCifar(
            num_classes=a.num_classes, stage_blocks=[1, 1, 1], block_branches=[2, 1, 1],
            mode="shared_gamma", single_beta_blocks=[0])
        model.load_state_dict(pair_sd, strict=True)
    model.to(dev).train()
    g_ref = model.stages[0][0].gamma.detach().abs().float().cpu().clone()
    cov_path = a.cov or Path(os.path.dirname(__file__)) / "cifar100_patch_cov.pt"
    cov = torch.load(cov_path, map_location="cpu").float()

    damp_e = damp_ws = None
    if a.block0_damp_top != 1.0:
        damp_e = top_eigvec(cov).to(dev)
        damp_ws = [cv.weight for cv in model.stages[0][0].convs]
        damp_wd = a.weight_decay if a.damp_freeze_decay else 0.0

    over = {"stages.0.0": {"gamma": 0.25}} if a.gauged else None
    lr_over = ({"stages.0.0.convs": a.lr * a.block0_kernel_lr_mult}
               if a.block0_kernel_lr_mult != 1.0 else None)
    opt = build_kernel_wd_sgd(model.parameters(), model, lr=a.lr, momentum=a.momentum,
                              weight_decay=a.weight_decay, affine_overrides=over, lr_overrides=lr_over)
    build_loaders = importlib.import_module(f"structural_reparam.data.{a.dataset}").build_loaders
    # loaders differ in signature (stl10 has no batch_mode); pass only what each accepts
    want = {"data_dir": a.data_dir, "batch_size": a.batch_size, "num_workers": a.num_workers,
            "batch_mode": "shuffled", "download": False}
    accepted = set(inspect.signature(build_loaders).parameters)
    loaders = build_loaders(**{k: v for k, v in want.items() if k in accepted})
    train_loader = loaders[0] if isinstance(loaders, (tuple, list)) else loaders

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("a") as fo:
        rec0 = {"epoch": a.start_epoch, "zeta": a.zeta, "seed": a.seed, "gauged": a.gauged,
                "k_lr_mult": a.block0_kernel_lr_mult, "no_split": a.no_split,
                "damp_top": a.block0_damp_top, "damp_renorm": a.damp_renorm,
                "damp_freeze_decay": a.damp_freeze_decay, "train_accuracy": None}
        rec0.update(block0_stats(model, cov, g_ref))
        if a.per_channel:
            rec0.update(block0_arrays(model, cov))
        fo.write(json.dumps(rec0) + "\n"); fo.flush()
        if a.save_block0 is not None:
            save_block0(model, a.save_block0 / f"{a.out.stem}_ep{a.start_epoch}.pt")
        for ep in range(a.start_epoch + 1, a.total_epochs + 1):
            correct = total = 0
            for x, y in train_loader:
                x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                out = model(x)
                loss = F.cross_entropy(out, y)
                loss.backward()
                if damp_e is not None:
                    damp_top_grad(damp_ws, damp_e, a.block0_damp_top, a.damp_renorm, damp_wd)
                opt.step()
                correct += int((out.argmax(1) == y).sum()); total += y.numel()
            rec = {"epoch": ep, "zeta": a.zeta, "seed": a.seed, "no_split": a.no_split,
                   "damp_top": a.block0_damp_top, "train_accuracy": correct / max(total, 1)}
            rec.update(block0_stats(model, cov, g_ref))
            if a.per_channel:
                rec.update(block0_arrays(model, cov))
            fo.write(json.dumps(rec) + "\n"); fo.flush()
            if a.save_block0 is not None:
                save_block0(model, a.save_block0 / f"{a.out.stem}_ep{ep}.pt")
            print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
