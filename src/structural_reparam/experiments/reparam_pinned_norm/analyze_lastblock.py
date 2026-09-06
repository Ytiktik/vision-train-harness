"""Last-block conditioning readout for the pinned-norm cell.

Reads the checkpoint state dicts written by the checkpoint probe
(``<ckpt_root>/<variant>_seed<seed>/ckpt_ep<E>.pt``) for the one-branch
(single) and shared-gamma pair arms and prints, per epoch, the per-channel
median and max over the last block of: gamma (per branch), sigma (running
std), kernel norm, kernel cosine (pair), effective output scale
g_eff = |gamma| sqrt(2+2cos), the direction-mode curvature in sigma units
(single gamma^2/sigma^2; pair gamma^2 (1/s1^2 + 1/s2^2), i.e. the branch
sum), the radial factor (1 single; 2+2cos pair) and kappa = direction/radial.
Also the paired train-accuracy gain from the run JSON lines if a log dir is
given.  Usage:
  python -m structural_reparam.experiments.reparam_pinned_norm.analyze_lastblock \
      --ckpt-root outputs/pinned_c100_d3_wd0/checkpoints --seeds 42 43 44 \
      --epochs 0 5 10 20 30 50 75 100 --last-block stages.2.0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def block_stats(sd: dict, blk: str) -> dict[str, np.ndarray]:
    p = blk + "."
    g = sd[p + "gamma"].double()
    ws = [sd[k].double().flatten(1) for k in sorted(sd) if k.startswith(p + "convs.") and k.endswith(".weight")]
    sig = [sd[k].double().sqrt() for k in sorted(sd) if k.startswith(p + "stats.") and k.endswith("running_var")]
    out: dict[str, np.ndarray] = {"gamma": g.abs(), "wn": ws[0].norm(dim=1), "sig1": sig[0]}
    if len(ws) == 1:
        out.update(geff=g.abs(), T=g**2 / sig[0]**2, radial=torch.ones_like(g))
    else:
        cos = (ws[0] * ws[1]).sum(1) / (ws[0].norm(dim=1) * ws[1].norm(dim=1))
        out.update(cos=cos, sig2=sig[1], geff=g.abs() * (2 + 2 * cos).clamp(min=0).sqrt(),
                   T=g**2 * (1 / sig[0]**2 + 1 / sig[1]**2), radial=2 + 2 * cos)
        out["Tbranch"] = g**2 / sig[0]**2
    out["kappa"] = out["T"] / out["radial"]
    return {k: v.numpy() for k, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-root", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, nargs="+", default=[0, 5, 10, 20, 30, 50, 75, 100])
    ap.add_argument("--last-block", default="stages.2.0")
    ap.add_argument("--single", default="single_d3_pinned")
    ap.add_argument("--pair", default="shared_gamma_d3_pinned")
    ap.add_argument("--log-dir", type=Path, default=None, help="dir with pinned_s<seed>.log JSON lines for the accuracy gain")
    a = ap.parse_args()

    def agg(vals):
        return f"{np.mean(vals):.2f}±{np.std(vals):.2f}" if len(vals) > 1 else f"{vals[0]:.2f}"

    print(f"last block {a.last_block}; mean over seeds {a.seeds} of per-seed median | max")
    hdr = "| epoch | stat | single γ | single σ | single γ²/σ² | pair γ | pair σ₁ | pair cos | pair g_eff | pair per-branch γ²/σ² | pair Σ | pair radial 2+2cos | κ_pair | κ_single |"
    print(hdr); print("|---" * 14 + "|")
    for ep in a.epochs:
        S, P = [], []
        for s in a.seeds:
            fs = a.ckpt_root / f"{a.single}_seed{s}" / f"ckpt_ep{ep}.pt"
            fp = a.ckpt_root / f"{a.pair}_seed{s}" / f"ckpt_ep{ep}.pt"
            if not (fs.exists() and fp.exists()):
                continue
            S.append(block_stats(torch.load(fs, map_location="cpu")["state_dict"], a.last_block))
            P.append(block_stats(torch.load(fp, map_location="cpu")["state_dict"], a.last_block))
        if not S:
            continue
        for stat, f in (("med", np.median), ("max", np.max)):
            row = [str(ep), stat, agg([f(x["gamma"]) for x in S]), agg([f(x["sig1"]) for x in S]), agg([f(x["T"]) for x in S]),
                   agg([f(x["gamma"]) for x in P]), agg([f(x["sig1"]) for x in P]), agg([f(x["cos"]) for x in P]), agg([f(x["geff"]) for x in P]),
                   agg([f(x["Tbranch"]) for x in P]), agg([f(x["T"]) for x in P]), agg([f(x["radial"]) for x in P]), agg([f(x["kappa"]) for x in P]), agg([f(x["kappa"]) for x in S])]
            print("| " + " | ".join(row) + " |")
        print(f"   ep{ep}: pair min cos per seed {[round(float(x['cos'].min()),3) for x in P]}; kernel norm single {[round(float(np.median(x['wn'])),3) for x in S]} pair {[round(float(np.median(x['wn'])),3) for x in P]}")

    if a.log_dir:
        print("\nrunning train accuracy per epoch (from JSON lines); gain = pair - single, per seed")
        gains = {}
        for s in a.seeds:
            f = a.log_dir / f"pinned_s{s}.log"
            if not f.exists():
                continue
            rows = [json.loads(l) for l in f.read_text().splitlines() if l.startswith('{"epoch"')]
            acc = {}
            for r in rows:
                acc.setdefault(r["variant"], {})[r["epoch"]] = (r["train_accuracy"], r["train_loss"], r.get("test_accuracy"))
            for ep in a.epochs:
                if ep in acc.get(a.single, {}) and ep in acc.get(a.pair, {}):
                    gains.setdefault(ep, []).append(100 * (acc[a.pair][ep][0] - acc[a.single][ep][0]))
            last = max(acc.get(a.single, {0: 0}))
            if a.single in acc and a.pair in acc:
                print(f"  seed {s}: last common epoch {min(max(acc[a.single]), max(acc[a.pair]))}: single train {100*acc[a.single][max(acc[a.single])][0]:.2f} pair train {100*acc[a.pair][max(acc[a.pair])][0]:.2f}; test {100*(acc[a.single][max(acc[a.single])][2] or 0):.2f} vs {100*(acc[a.pair][max(acc[a.pair])][2] or 0):.2f}")
        for ep, g in sorted(gains.items()):
            print(f"  epoch {ep}: gain {np.mean(g):+.2f} ± {np.std(g)/np.sqrt(len(g)) if len(g)>1 else 0:.2f} points (n={len(g)}; per seed {[round(x,2) for x in g]})")


if __name__ == "__main__":
    main()
