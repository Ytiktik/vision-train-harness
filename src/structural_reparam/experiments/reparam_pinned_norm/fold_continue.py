"""Fold-and-continue test for the last-block-only pair.

For each seed: load the last-only pair's epoch-100 checkpoint, fold its last
block into a one-branch block that computes the same function (per channel:
kernel direction ∝ w1/σ1 + w2/σ2 renormalized to ‖w1‖, γ_s = γ·std(z1+z2),
β_s = β1+β2, BatchNorm running stats calibrated on training batches), check the
fold on a fixed batch, then continue training three arms from their epoch-100
states with the pinned-norm optimizer at the same learning rate:
  folded  : the folded single (starts at the pair's function),
  pair    : the last-only pair itself (control: does it hold?),
  single  : the plain single from its own epoch-100 checkpoint (control: flat).
Logs per epoch: running train loss/accuracy, last-block gamma median, output
scale median, and writes JSON lines + a final checkpoint per arm.
"""
from __future__ import annotations

import argparse, json, re
from pathlib import Path

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

from structural_reparam.data.cifar100 import build_loaders
from structural_reparam.deploy.train import run_epoch
from structural_reparam.experiments.reparam_pinned_norm.hessian_lambda_max import build_from_sd, fixed_batch
from structural_reparam.experiments.reparam_pinned_norm.lab import build_pinned_norm_sgd, pinned_params, PinnedNormSGD


def build_opt_lastlr(model, lr, last_lr, last, momentum=0.9):
    """Pinned-norm SGD with a separate learning rate for the last block's parameters."""
    pin = pinned_params(model); pin_ids = {id(p) for p in pin}
    pinned = []
    with torch.no_grad():
        for p in pin:
            pinned.append((p, p.view(p.shape[0], -1).norm(dim=1, keepdim=True).clone()))
    named = list(model.named_parameters())
    is_last = lambda n: n.startswith(last + ".")
    groups = [
        {"params": [p for n, p in named if id(p) in pin_ids and not is_last(n)], "lr": lr, "weight_decay": 0.0},
        {"params": [p for n, p in named if id(p) in pin_ids and is_last(n)], "lr": last_lr, "weight_decay": 0.0},
        {"params": [p for n, p in named if id(p) not in pin_ids and not is_last(n)], "lr": lr, "weight_decay": 0.0},
        {"params": [p for n, p in named if id(p) not in pin_ids and is_last(n)], "lr": last_lr, "weight_decay": 0.0},
    ]
    return PinnedNormSGD(groups, pinned, lr=lr, momentum=momentum)


def last_block_stats(model, last):
    blk = dict(model.named_modules())[last]
    g = blk.gamma.detach().abs()
    ws = [c.weight.detach().flatten(1) for c in blk.convs]
    if len(ws) == 1:
        return float(g.median()), float(g.median()), 1.0
    cos = (ws[0] * ws[1]).sum(1) / (ws[0].norm(dim=1) * ws[1].norm(dim=1))
    geff = g * (2 + 2 * cos).clamp(min=0).sqrt()
    return float(g.median()), float(geff.median()), float(cos.median())


@torch.no_grad()
def fold_pair(pair_sd: dict, last: str, pair_model, calib_batches, dev, radius=None) -> dict:
    """Return a one-branch state dict computing the pair's function."""
    p = last + "."
    w1, w2 = pair_sd[p + "convs.0.weight"].double(), pair_sd[p + "convs.1.weight"].double()
    s1 = pair_sd[p + "stats.0.running_var"].double().sqrt(); s2 = pair_sd[p + "stats.1.running_var"].double().sqrt()
    C = w1.shape[0]
    weff = w1 / s1.view(C, 1, 1, 1) + w2 / s2.view(C, 1, 1, 1)
    r = w1.flatten(1).norm(dim=1) if radius is None else torch.as_tensor(radius).double().to(w1.device).expand(C).clone()
    ws = weff * (r / weff.flatten(1).norm(dim=1)).view(C, 1, 1, 1)
    # std of z1+z2 per channel from the pair in train mode on calibration batches
    blk = dict(pair_model.named_modules())[last]
    zs = {}
    hooks = [st.register_forward_hook(lambda m, i, o, k=k: zs.__setitem__(k, o.detach())) for k, st in enumerate(blk.stats)]
    var_sum = torch.zeros(C, dtype=torch.float64, device=dev); n = 0
    pair_model.train()
    for x, _ in calib_batches:
        pair_model(x.to(dev)); zsum = (zs[0] + zs[1]).double()
        var_sum += zsum.var(dim=(0, 2, 3)); n += 1
    for h in hooks: h.remove()
    sig_eff = (var_sum / n).sqrt().cpu()
    sd = {}
    for k, v in pair_sd.items():
        if not k.startswith(p): sd[k] = v.clone()
    sd[p + "gamma"] = (pair_sd[p + "gamma"].double() * sig_eff).float()
    sd[p + "convs.0.weight"] = ws.float()
    sd[p + "betas.0"] = (pair_sd[p + "betas.0"] + pair_sd[p + "betas.1"]).clone()
    sd[p + "stats.0.running_mean"] = torch.zeros(C); sd[p + "stats.0.running_var"] = torch.ones(C)
    if p + "stats.0.num_batches_tracked" in pair_sd: sd[p + "stats.0.num_batches_tracked"] = torch.tensor(0)
    return sd, sig_eff


@torch.no_grad()
def calibrate_bn(model, batches, dev):
    """Set every BranchStats2d running mean/var to the average batch statistics
    over the calibration batches (the module has no reset/cumulative mode)."""
    stats = [m for m in model.modules() if hasattr(m, "running_var") and hasattr(m, "last_mu")]
    saved = {m: m.momentum for m in stats}
    for m in stats: m.momentum = 0.0
    acc = {m: [torch.zeros_like(m.running_mean, dtype=torch.float64), torch.zeros_like(m.running_var, dtype=torch.float64)] for m in stats}
    model.train(); n = 0
    for x, _ in batches:
        model(x.to(dev)); n += 1
        for m in stats:
            acc[m][0] += m.last_mu.double(); acc[m][1] += m.last_var.double()
    for m in stats:
        m.running_mean.copy_((acc[m][0] / n).to(m.running_mean.dtype)); m.running_var.copy_((acc[m][1] / n).to(m.running_var.dtype))
        m.momentum = saved[m]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-root", type=Path, required=True); ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--pair", default="pair_last_d3_pinned"); ap.add_argument("--single", default="single_d3_pinned")
    ap.add_argument("--epoch", type=int, default=100); ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--arms", nargs="+", default=["folded", "pair", "single"]); ap.add_argument("--data-dir", default="data/cifar100")
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--device", default="cuda"); ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--last-lr", type=float, default=None, help="learning rate for the last block only (others use --lr)")
    a = ap.parse_args(); dev = torch.device(a.device); torch.manual_seed(a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    train_loader, test_loader = build_loaders(a.data_dir, batch_size=128, num_workers=a.num_workers, persistent_workers=True, download=False)
    calib = [b for _, b in zip(range(20), train_loader)]
    xb, yb = fixed_batch(a.data_dir, 1024, dev)
    pair_sd = torch.load(a.ckpt_root / f"{a.pair}_seed{a.seed}" / f"ckpt_ep{a.epoch}.pt", map_location="cpu")["state_dict"]
    single_sd = torch.load(a.ckpt_root / f"{a.single}_seed{a.seed}" / f"ckpt_ep{a.epoch}.pt", map_location="cpu")["state_dict"]
    pair_model, last = build_from_sd(pair_sd); pair_model.to(dev)
    folded_sd, sig_eff = fold_pair(pair_sd, last, pair_model, calib, dev)
    folded_model, _ = build_from_sd(folded_sd); folded_model.to(dev)
    calibrate_bn(folded_model, calib, dev)
    single_model, _ = build_from_sd(single_sd); single_model.to(dev)
    # fold check: train-mode loss on the fixed batch, and eval-mode outputs agreement
    def bl(m):
        m.train()
        with torch.no_grad(): return float(F.cross_entropy(m(xb), yb))
    def ev(m):
        m.eval()
        with torch.no_grad(): return m(xb)
    lp, lf, ls = bl(pair_model), bl(folded_model), bl(single_model)
    op, of = ev(pair_model), ev(folded_model)
    agree = float((op.argmax(1) == of.argmax(1)).float().mean()); rel = float((op - of).norm() / op.norm())
    chk = dict(seed=a.seed, loss_pair=lp, loss_folded=lf, loss_single=ls, eval_argmax_agree=agree, eval_logit_rel_diff=rel,
               sig_eff_med=float(sig_eff.median()), gamma_folded_med=float(folded_sd[last + ".gamma"].abs().median()), gamma_pair_med=float(pair_sd[last + ".gamma"].abs().median()))
    print("FOLD CHECK", json.dumps(chk), flush=True)
    (a.out / f"fold_check_s{a.seed}.json").write_text(json.dumps(chk))
    models = {"folded": folded_model, "pair": pair_model, "single": single_model}
    crit = nn.CrossEntropyLoss()
    for arm in a.arms:
        m = models[arm]; m.to(dev)
        opt = build_opt_lastlr(m, a.lr, a.last_lr, last) if a.last_lr is not None else build_pinned_norm_sgd(m.parameters(), model=m, lr=a.lr, momentum=0.9, weight_decay=0.0)
        log = a.out / f"continue_{arm}_s{a.seed}.jsonl"
        with log.open("w") as fo:
            g0, ge0, c0 = last_block_stats(m, last)
            fo.write(json.dumps(dict(arm=arm, seed=a.seed, epoch=0, gamma_med=g0, output_med=ge0, cos_med=c0, batch_loss=bl(m))) + "\n")
            for ep in range(1, a.epochs + 1):
                m.train()
                st = run_epoch(m, train_loader, crit, dev, optimizer=opt)
                g, ge, c = last_block_stats(m, last)
                rec = dict(arm=arm, seed=a.seed, epoch=ep, train_loss=st["loss"], train_accuracy=st["accuracy"], gamma_med=g, output_med=ge, cos_med=c)
                if ep % 5 == 0:
                    m.eval(); te = run_epoch(m, test_loader, crit, dev, optimizer=None); rec.update(test_loss=te["loss"], test_accuracy=te["accuracy"])
                print(json.dumps(rec), flush=True); fo.write(json.dumps(rec) + "\n"); fo.flush()
        torch.save({"state_dict": m.state_dict(), "arm": arm, "seed": a.seed, "epochs": a.epochs}, a.out / f"continue_{arm}_s{a.seed}_final.pt")


if __name__ == "__main__":
    main()
