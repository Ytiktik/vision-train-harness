"""Table of the annealed affine-ablation family (W&B group cos_c100_wdconv): paired train-accuracy
gain (points, mean +- sem over seeds) against single_d3_wdconv at the given epochs, for every
variant in the group matching the family. Usage: python -m ...affine_report [--epochs 10 50 80 100] [--csv out.csv]"""
from __future__ import annotations
import argparse, csv, sys
import numpy as np

FAMILY = ["shared_gamma_d3_wdconv", "shared_gamma_d3_wdconv_collapsed", "single_g4_d3_wdconv", "single_g4b2_d3_wdconv",
          "pairfirst_affrest_d3_wdconv", "single_lr02_d3_wdconv", "single_lr04_d3_wdconv", "single_b2_d3_wdconv",
          "single_g4b2_first_d3_wdconv", "single_g4b2_mid_d3_wdconv", "single_g4b2_last_d3_wdconv", "single_g4b2_rest_d3_wdconv",
          "single_g4_first_d3_wdconv", "single_g4_mid_d3_wdconv", "single_g4_last_d3_wdconv",
          "single_b2_first_d3_wdconv", "single_b2_mid_d3_wdconv", "single_b2_last_d3_wdconv",
          "pair_first_collapsed_d3_wdconv", "pair_mid_collapsed_d3_wdconv", "pair_last_collapsed_d3_wdconv"]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, nargs="+", default=[10, 30, 50, 70, 85, 100]); ap.add_argument("--csv", default=None)
    a = ap.parse_args()
    import wandb
    api = wandb.Api(timeout=120)
    runs = list(api.runs("claude-autonomous-reparam", filters={"group": "cos_c100_wdconv"}))
    H = {}
    for r in runs:
        v = (r.config.get("variant") or {}).get("name"); s = r.config.get("seed")
        if v not in FAMILY and v != "single_d3_wdconv": continue
        H[(v, s)] = {int(x["epoch"]): (x["train_accuracy"], x.get("test_accuracy")) for x in r.scan_history(keys=["epoch", "train_accuracy", "test_accuracy"]) if x.get("train_accuracy") is not None}
    seeds = sorted({s for v, s in H if v == "single_d3_wdconv"})
    rows = []
    print(f"{'variant':34s} " + " ".join(f"{'ep'+str(e):>16s}" for e in a.epochs) + "   train/test@100  n")
    for v in FAMILY:
        if not any((v, s) in H for s in seeds): continue
        cells = []
        for e in a.epochs:
            g = [100 * (H[(v, s)][e][0] - H[("single_d3_wdconv", s)][e][0]) for s in seeds if (v, s) in H and e in H[(v, s)] and e in H[("single_d3_wdconv", s)]]
            cells.append(f"{np.mean(g):+.2f}+-{(np.std(g, ddof=1)/np.sqrt(len(g)) if len(g) > 1 else 0):.2f}" if g else "   (pending)   ")
            rows.append({"variant": v, "epoch": e, "gain": np.mean(g) if g else None, "sem": (np.std(g, ddof=1)/np.sqrt(len(g)) if len(g) > 1 else None), "n": len(g)})
        tr = [100 * H[(v, s)][100][0] for s in seeds if (v, s) in H and 100 in H[(v, s)]]
        te = [100 * H[(v, s)][100][1] for s in seeds if (v, s) in H and 100 in H[(v, s)] and H[(v, s)][100][1] is not None]
        n = len([s for s in seeds if (v, s) in H])
        print(f"{v:34s} " + " ".join(f"{c:>16s}" for c in cells) + f"   {np.mean(tr) if tr else float('nan'):5.2f}/{np.mean(te) if te else float('nan'):5.2f}  {n}")
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["variant", "epoch", "gain", "sem", "n"]); w.writeheader(); w.writerows(rows)

if __name__ == "__main__":
    main()
