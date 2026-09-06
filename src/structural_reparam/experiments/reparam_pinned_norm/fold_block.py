"""Block-wise fold / unfold test at the same function (2026-08-19).

--block B (e.g. stages.0.0): from a pair checkpoint (default the full shared-γ
pair) build 'foldedB' = the pair with block B folded into one kernel (same
function); from the single checkpoint build 'unfoldedB' = the single with block
B written as an exact collapsed pair (same function). For every arm: full-set
equivalence, per-group sharpness (whole network, each block, block B kernels
tangential, all γ, fc), then continuation at --lr with per-step logging in the
first epoch of block B's kernel angular step and γ, then per-epoch.
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, torchvision, torchvision.transforms as T
from structural_reparam.data.cifar100 import build_loaders
from structural_reparam.experiments.reparam_pinned_norm.hessian_lambda_max import build_from_sd, lambda_max, fixed_batch
from structural_reparam.experiments.reparam_pinned_norm.fold_continue import fold_pair, calibrate_bn
from structural_reparam.experiments.reparam_pinned_norm.fold_sharpness import evaluate, unfold_single, MEAN, STD
from structural_reparam.experiments.reparam_pinned_norm.lab import build_pinned_norm_sgd, build_kernel_wd_sgd
from structural_reparam.experiments.reparam_pinned_norm.fold_continue import build_opt_lastlr

def block_stats(model, blk):
    """median |gamma|, median output scale, median branch cosine, median kernel norm per branch, median running sigma per branch"""
    b = dict(model.named_modules())[blk]; g = b.gamma.detach().abs()
    ws = [c.weight.detach().flatten(1) for c in b.convs]
    norms = [float(w.norm(dim=1).median()) for w in ws]; sigs = [float((st.running_var + st.eps).sqrt().median()) for st in b.stats]
    if len(ws) == 1: return float(g.median()), float(g.median()), 1.0, norms, sigs
    cos = (ws[0] * ws[1]).sum(1) / (ws[0].norm(dim=1) * ws[1].norm(dim=1))
    return float(g.median()), float((g * (2 + 2 * cos).clamp(min=0).sqrt()).median()), float(cos.median()), norms, sigs

def kernel_dirs(model, blk):
    return [c.weight.detach().flatten(1).clone() for c in dict(model.named_modules())[blk].convs]

def sharpness(model, blk, x, y, iters):
    model.train()
    for m in model.modules():
        if hasattr(m, "momentum"): m.momentum = 0.0
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    blocks = sorted({m_.group(1) for n, _ in named for m_ in [re.match(r"(stages\.\d+\.\d+)\.", n)] if m_}, key=lambda b: tuple(int(t) for t in b.split(".")[1:]))
    bk = [p for n, p in named if n.startswith(blk + ".convs.")]
    def proj(vs, ks=bk):
        for vv, w in zip(vs, ks):
            W = w.detach().view(w.shape[0], -1); what = W / W.norm(dim=1, keepdim=True); V = vv.view(vv.shape[0], -1); V.sub_((V * what).sum(1, keepdim=True) * what)
    groups = [("all", [p for _, p in named], None)] + [(b, [p for n, p in named if n.startswith(b + ".")], None) for b in blocks]
    groups += [("block_tan", bk, proj), ("gammas", [p for n, p in named if n.endswith(".gamma")], None), ("fc", [p for n, p in named if n.startswith("fc.")], None)]
    return {name: lambda_max(model, params, x, y, iters, project=pr)[0] for name, params, pr in groups}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-root", type=Path, required=True); ap.add_argument("--seed", type=int, required=True); ap.add_argument("--block", default="stages.0.0")
    ap.add_argument("--pair", default="shared_gamma_d3_pinned"); ap.add_argument("--single", default="single_d3_pinned"); ap.add_argument("--epoch", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.1); ap.add_argument("--epochs", type=int, default=3); ap.add_argument("--arms", nargs="+", default=["pair", "folded", "single", "unfolded"])
    ap.add_argument("--data-dir", default="data/cifar100"); ap.add_argument("--hess-iters", type=int, default=30); ap.add_argument("--hess-batch", type=int, default=1024); ap.add_argument("--probe-every", type=int, default=20)
    ap.add_argument("--skip-eval", action="store_true"); ap.add_argument("--skip-hess", action="store_true")
    ap.add_argument("--optimizer", choices=["pinned", "wd"], default="pinned", help="pinned: PinnedNormSGD (no decay); wd: plain SGD with kernel-only decay --weight-decay")
    ap.add_argument("--weight-decay", type=float, default=5e-4); ap.add_argument("--pair-wd-scale", type=float, default=1.0)
    ap.add_argument("--block-lr", type=float, default=None, help="learning rate for block B's parameters in the continuation (default: --lr)")
    ap.add_argument("--fold-norm", default="pair", help="kernel norm of the folded single: 'pair' (the pair's branch-0 norm, per channel), 'single' (the single checkpoint's norm at block B, per channel), or a float")
    ap.add_argument("--unfold-norm-scale", type=float, default=1.0, help="multiply the unfolded pair's branch kernels by this (1/sqrt2 = gauged unfold); running stats rescaled accordingly")
    ap.add_argument("--max-steps", type=int, default=None, help="cap the number of continuation steps per arm (gates)")
    ap.add_argument("--single-beta", action="store_true", help="build pair blocks with one trainable bias (as the wdconv_gauge runs)")
    ap.add_argument("--tag", default="", help="suffix for the output file name")
    ap.add_argument("--block-wd-scale", type=float, default=None, help="multiply the decay of block B's kernels in the continuation (wd optimizer only), e.g. 0.25 = lowered wall")
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--device", default="cuda"); ap.add_argument("--num-workers", type=int, default=2)
    a = ap.parse_args(); dev = torch.device(a.device); torch.manual_seed(a.seed); a.out.mkdir(parents=True, exist_ok=True); blk = a.block
    train_loader, test_loader = build_loaders(a.data_dir, batch_size=128, num_workers=a.num_workers, persistent_workers=True, download=False)
    train_plain = torch.utils.data.DataLoader(torchvision.datasets.CIFAR100(a.data_dir, train=True, download=False, transform=T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])), batch_size=500, shuffle=False, num_workers=a.num_workers)
    calib = [b for _, b in zip(range(20), train_loader)]; xb, yb = fixed_batch(a.data_dir, a.hess_batch, dev)
    pair_sd = torch.load(a.ckpt_root / f"{a.pair}_seed{a.seed}" / f"ckpt_ep{a.epoch}.pt", map_location="cpu")["state_dict"]
    single_sd = torch.load(a.ckpt_root / f"{a.single}_seed{a.seed}" / f"ckpt_ep{a.epoch}.pt", map_location="cpu")["state_dict"]
    pair_model, _ = build_from_sd(pair_sd, single_beta=a.single_beta); pair_model.to(dev)
    if a.fold_norm == "pair": radius = None
    elif a.fold_norm == "single": radius = single_sd[blk + ".convs.0.weight"].flatten(1).norm(dim=1)
    else: radius = torch.full((pair_sd[blk + ".convs.0.weight"].shape[0],), float(a.fold_norm))
    folded_sd, sig_eff = fold_pair(pair_sd, blk, pair_model, calib, dev, radius=radius)
    folded_model, _ = build_from_sd(folded_sd, single_beta=a.single_beta); folded_model.to(dev); calibrate_bn(folded_model, calib, dev)
    single_model, _ = build_from_sd(single_sd, single_beta=a.single_beta); single_model.to(dev)
    unf_model, _ = build_from_sd(unfold_single(single_sd, blk, norm_scale=a.unfold_norm_scale), single_beta=a.single_beta); unf_model.to(dev)
    models = {"pair": pair_model, "folded": folded_model, "single": single_model, "unfolded": unf_model}
    log = (a.out / f"fold_block_{blk}_s{a.seed}{a.tag}.jsonl").open("w")
    def emit(d): d.update(seed=a.seed, block=blk); print(json.dumps(d), flush=True); log.write(json.dumps(d) + "\n"); log.flush()
    emit(dict(kind="fold", sig_eff_med=float(sig_eff.median()), sig_eff_p10=float(np.percentile(sig_eff.numpy(), 10)), sig_eff_p90=float(np.percentile(sig_eff.numpy(), 90)),
              fold_norm=a.fold_norm, unfold_norm_scale=a.unfold_norm_scale, optimizer=a.optimizer, weight_decay=a.weight_decay, pair_wd_scale=a.pair_wd_scale, lr=a.lr, block_lr=a.block_lr, block_wd_scale=a.block_wd_scale, pair=a.pair, single=a.single, epoch=a.epoch,
              **{f"{n}_norm_med": block_stats(m, blk)[3] for n, m in models.items()}, **{f"{n}_sigma_med": block_stats(m, blk)[4] for n, m in models.items()}))
    if not a.skip_eval:
        for name, m in models.items():
            trl, tra = evaluate(m, train_plain, dev, True); tel, tea = evaluate(m, test_loader, dev, False)
            emit(dict(kind="equivalence", arm=name, train_mode_full_train_loss=trl, train_mode_full_train_acc=tra, eval_mode_test_loss=tel, eval_mode_test_acc=tea))
    if not a.skip_hess:
        for name, m in models.items(): emit(dict(kind="sharpness", arm=name, **sharpness(m, blk, xb, yb, a.hess_iters)))
    crit = nn.CrossEntropyLoss()
    for arm in a.arms:
        m = models[arm]; m.train()
        for mod in m.modules():
            if hasattr(mod, "momentum"): mod.momentum = 0.1
        if a.optimizer == "wd":
            opt = build_kernel_wd_sgd(m.parameters(), model=m, lr=a.lr, momentum=0.9, weight_decay=a.weight_decay, pair_wd_scale=a.pair_wd_scale,
                                      lr_overrides=({blk: a.block_lr} if a.block_lr is not None else None),
                                      wd_overrides=({blk: a.block_wd_scale} if a.block_wd_scale is not None else None))
        elif a.block_lr is not None:
            opt = build_opt_lastlr(m, a.lr, a.block_lr, blk)
        else:
            opt = build_pinned_norm_sgd(m.parameters(), model=m, lr=a.lr, momentum=0.9, weight_decay=0.0)
        prev = kernel_dirs(m, blk); step = 0
        def probe():
            m.train()
            with torch.no_grad(): o = m(xb)
            return float(F.cross_entropy(o, yb)), float((o.argmax(1) == yb).float().mean())
        pl, pa = probe(); g0, ge0, c0, n0, s0 = block_stats(m, blk)
        emit(dict(kind="step", arm=arm, epoch=0, step=0, probe_loss=pl, probe_acc=pa, gamma_med=g0, output_med=ge0, cos_med=c0, norm_med=n0, sigma_med=s0))
        stop = False
        for ep in range(1, a.epochs + 1):
            m.train(); tl = 0.0; tc = 0; tn = 0
            for x, y in train_loader:
                x, y = x.to(dev), y.to(dev); opt.zero_grad(set_to_none=True); out = m(x); loss = crit(out, y); loss.backward(); opt.step(); step += 1
                bl = loss.item(); tl += bl * y.numel(); tc += (out.argmax(1) == y).sum().item(); tn += y.numel()
                if ep == 1:
                    cur = kernel_dirs(m, blk)
                    ang = float(np.mean([torch.rad2deg(torch.acos(((c * p).sum(1) / (c.norm(dim=1) * p.norm(dim=1))).clamp(-1, 1))).mean().item() for c, p in zip(cur, prev)])); prev = cur
                    g, ge, c, nm, sm = block_stats(m, blk); rec = dict(kind="step", arm=arm, epoch=ep, step=step, batch_loss=bl, gamma_med=g, output_med=ge, cos_med=c, ang_step_deg=ang, norm_med=nm, sigma_med=sm)
                    if step % a.probe_every == 0: pl, pa = probe(); rec.update(probe_loss=pl, probe_acc=pa)
                    emit(rec)
                if a.max_steps is not None and step >= a.max_steps: stop = True; break
            g, ge, c, nm, sm = block_stats(m, blk); pl, pa = probe()
            emit(dict(kind="epoch", arm=arm, epoch=ep, train_loss=tl / max(tn, 1), train_accuracy=tc / max(tn, 1), gamma_med=g, output_med=ge, cos_med=c, norm_med=nm, sigma_med=sm, probe_loss=pl, probe_acc=pa))
            if stop: break
    log.close()

if __name__ == "__main__":
    main()
