"""Apples-to-apples last-block test on the fold (user, 2026-08-19).

For each seed: build the last-only pair (epoch-100 checkpoint), its folded
single (same function), and the plain single (its own epoch-100 checkpoint).
  1. equivalence: train-mode loss/accuracy on the FULL training set (no
     augmentation) and eval-mode loss/accuracy on the test set, for all three;
  2. sharpness at the same function: top Hessian eigenvalue per parameter group
     (whole network, each block, last block, last-block kernels tangential, all
     gammas) on a fixed 1024-image train-mode batch, for all three;
  3. relaxation: continue each arm at the given lr with the pinned optimizer,
     logging EVERY step of the first epoch (batch loss, batch acc, last-block
     gamma median, mean angular step of the last-block kernels since the previous
     step, and every `--probe-every` steps the fixed-batch train-mode loss/acc),
     then per-epoch for the remaining epochs.
Outputs JSON lines under --out.
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, torchvision, torchvision.transforms as T
from structural_reparam.data.cifar100 import build_loaders
from structural_reparam.experiments.reparam_pinned_norm.hessian_lambda_max import build_from_sd, lambda_max, fixed_batch
from structural_reparam.experiments.reparam_pinned_norm.fold_continue import fold_pair, calibrate_bn, last_block_stats, build_opt_lastlr
from structural_reparam.experiments.reparam_pinned_norm.lab import build_pinned_norm_sgd

MEAN, STD = (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)

@torch.no_grad()
def evaluate(model, loader, dev, train_mode):
    model.train(train_mode)
    for m in model.modules():
        if hasattr(m, "momentum"): m.momentum = 0.0  # never touch running stats here
    tot = 0; loss = 0.0; corr = 0
    for x, y in loader:
        x, y = x.to(dev), y.to(dev); out = model(x)
        loss += F.cross_entropy(out, y, reduction="sum").item(); corr += (out.argmax(1) == y).sum().item(); tot += y.numel()
    return loss / tot, corr / tot

def sharpness(model, last, x, y, iters):
    model.train()
    for m in model.modules():
        if hasattr(m, "momentum"): m.momentum = 0.0
    named = list(model.named_parameters())
    blocks = sorted({m_.group(1) for n, _ in named for m_ in [re.match(r"(stages\.\d+\.\d+)\.", n)] if m_}, key=lambda b: tuple(int(t) for t in b.split(".")[1:]))
    lastk = [p for n, p in named if n.startswith(last + ".convs.")]
    def proj(vs, ks=lastk):
        for vv, w in zip(vs, ks):
            W = w.detach().view(w.shape[0], -1); what = W / W.norm(dim=1, keepdim=True); V = vv.view(vv.shape[0], -1); V.sub_((V * what).sum(1, keepdim=True) * what)
    groups = [("all", [p for _, p in named], None)] + [(b, [p for n, p in named if n.startswith(b + ".")], None) for b in blocks]
    groups += [("last_tan", lastk, proj), ("gammas", [p for n, p in named if n.endswith(".gamma")], None), ("fc", [p for n, p in named if n.startswith("fc.")], None)]
    return {name: lambda_max(model, params, x, y, iters, project=pr)[0] for name, params, pr in groups}

@torch.no_grad()
def unfold_single(single_sd: dict, last: str, norm_scale: float = 1.0) -> dict:
    """Exact collapsed pair computing the single's function: w1 = w2 = w,
    shared gamma = gamma_s / 2, beta1 = beta_s, beta2 = 0, stats copied."""
    p = last + "."; sd = {}
    for k, v in single_sd.items():
        if not k.startswith(p): sd[k] = v.clone()
    for i in (0, 1):
        sd[p + f"convs.{i}.weight"] = single_sd[p + "convs.0.weight"].clone() * float(norm_scale)
        sd[p + f"stats.{i}.running_mean"] = single_sd[p + "stats.0.running_mean"].clone() * float(norm_scale)
        sd[p + f"stats.{i}.running_var"] = single_sd[p + "stats.0.running_var"].clone() * float(norm_scale) ** 2
        if p + "stats.0.num_batches_tracked" in single_sd: sd[p + f"stats.{i}.num_batches_tracked"] = single_sd[p + "stats.0.num_batches_tracked"].clone()
    sd[p + "gamma"] = single_sd[p + "gamma"] / 2
    sd[p + "betas.0"] = single_sd[p + "betas.0"].clone(); sd[p + "betas.1"] = torch.zeros_like(single_sd[p + "betas.0"])
    return sd


def kernel_dirs(model, last):
    blk = dict(model.named_modules())[last]
    return [c.weight.detach().flatten(1).clone() for c in blk.convs]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-root", type=Path, required=True); ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--pair", default="pair_last_d3_pinned"); ap.add_argument("--single", default="single_d3_pinned"); ap.add_argument("--epoch", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.1); ap.add_argument("--last-lr", type=float, default=None); ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--arms", nargs="+", default=["folded", "pair", "single"]); ap.add_argument("--data-dir", default="data/cifar100")
    ap.add_argument("--hess-iters", type=int, default=40); ap.add_argument("--hess-batch", type=int, default=1024); ap.add_argument("--probe-every", type=int, default=20)
    ap.add_argument("--skip-eval", action="store_true"); ap.add_argument("--skip-hess", action="store_true")
    ap.add_argument("--unfold", action="store_true", help="also build 'unfolded': the single's last block as an exact collapsed pair")
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--device", default="cuda"); ap.add_argument("--num-workers", type=int, default=2)
    a = ap.parse_args(); dev = torch.device(a.device); torch.manual_seed(a.seed); a.out.mkdir(parents=True, exist_ok=True)
    train_loader, test_loader = build_loaders(a.data_dir, batch_size=128, num_workers=a.num_workers, persistent_workers=True, download=False)
    plain_tf = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
    train_plain = torch.utils.data.DataLoader(torchvision.datasets.CIFAR100(a.data_dir, train=True, download=False, transform=plain_tf), batch_size=500, shuffle=False, num_workers=a.num_workers)
    calib = [b for _, b in zip(range(20), train_loader)]
    xb, yb = fixed_batch(a.data_dir, a.hess_batch, dev)
    pair_sd = torch.load(a.ckpt_root / f"{a.pair}_seed{a.seed}" / f"ckpt_ep{a.epoch}.pt", map_location="cpu")["state_dict"]
    single_sd = torch.load(a.ckpt_root / f"{a.single}_seed{a.seed}" / f"ckpt_ep{a.epoch}.pt", map_location="cpu")["state_dict"]
    pair_model, last = build_from_sd(pair_sd); pair_model.to(dev)
    folded_sd, sig_eff = fold_pair(pair_sd, last, pair_model, calib, dev)
    folded_model, _ = build_from_sd(folded_sd); folded_model.to(dev); calibrate_bn(folded_model, calib, dev)
    single_model, _ = build_from_sd(single_sd); single_model.to(dev)
    models = {"pair": pair_model, "folded": folded_model, "single": single_model}
    if a.unfold:
        unf_model, _ = build_from_sd(unfold_single(single_sd, last)); unf_model.to(dev); models["unfolded"] = unf_model
    log = (a.out / f"fold_sharpness_s{a.seed}.jsonl").open("w")
    def emit(d): d.update(seed=a.seed); print(json.dumps(d), flush=True); log.write(json.dumps(d) + "\n"); log.flush()
    if not a.skip_eval:
        for name, m in models.items():
            trl, tra = evaluate(m, train_plain, dev, train_mode=True)
            tel, tea = evaluate(m, test_loader, dev, train_mode=False)
            emit(dict(kind="equivalence", arm=name, train_mode_full_train_loss=trl, train_mode_full_train_acc=tra, eval_mode_test_loss=tel, eval_mode_test_acc=tea))
    if not a.skip_hess:
        for name, m in models.items():
            emit(dict(kind="sharpness", arm=name, **sharpness(m, last, xb, yb, a.hess_iters)))
    crit = nn.CrossEntropyLoss()
    for arm in a.arms:
        m = models[arm]; m.train()
        for mod in m.modules():
            if hasattr(mod, "momentum"): mod.momentum = 0.1
        opt = build_opt_lastlr(m, a.lr, a.last_lr, last) if a.last_lr is not None else build_pinned_norm_sgd(m.parameters(), model=m, lr=a.lr, momentum=0.9, weight_decay=0.0)
        prev = kernel_dirs(m, last); step = 0
        def probe():
            m.train()
            with torch.no_grad(): o = m(xb)
            return float(F.cross_entropy(o, yb)), float((o.argmax(1) == yb).float().mean())
        pl, pa = probe(); g0, ge0, c0 = last_block_stats(m, last)
        emit(dict(kind="step", arm=arm, epoch=0, step=0, probe_loss=pl, probe_acc=pa, gamma_med=g0, output_med=ge0, cos_med=c0))
        for ep in range(1, a.epochs + 1):
            m.train(); tl = 0.0; tc = 0; tn = 0
            for x, y in train_loader:
                x, y = x.to(dev), y.to(dev); opt.zero_grad(set_to_none=True); out = m(x); loss = crit(out, y); loss.backward(); opt.step(); step += 1
                bl = loss.item(); ba = (out.argmax(1) == y).float().mean().item(); tl += bl * y.numel(); tc += (out.argmax(1) == y).sum().item(); tn += y.numel()
                if ep == 1:
                    cur = kernel_dirs(m, last)
                    ang = float(np.mean([torch.rad2deg(torch.acos(((c * p).sum(1) / (c.norm(dim=1) * p.norm(dim=1))).clamp(-1, 1))).mean().item() for c, p in zip(cur, prev)]))
                    prev = cur; g, ge, c = last_block_stats(m, last)
                    rec = dict(kind="step", arm=arm, epoch=ep, step=step, batch_loss=bl, batch_acc=ba, gamma_med=g, output_med=ge, cos_med=c, ang_step_deg=ang)
                    if step % a.probe_every == 0:
                        pl, pa = probe(); rec.update(probe_loss=pl, probe_acc=pa)
                    emit(rec)
            g, ge, c = last_block_stats(m, last); pl, pa = probe()
            emit(dict(kind="epoch", arm=arm, epoch=ep, train_loss=tl / tn, train_accuracy=tc / tn, gamma_med=g, output_med=ge, cos_med=c, probe_loss=pl, probe_acc=pa))
    log.close()

if __name__ == "__main__":
    main()
