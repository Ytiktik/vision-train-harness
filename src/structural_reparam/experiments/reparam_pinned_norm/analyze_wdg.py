"""Read the decayed gauge cell (W&B group wdconv_gauge_c100_d3) from Weights & Biases:
per-epoch pair_channel probe scalars and accuracies for every arm and seed.

  python -m structural_reparam.experiments.reparam_pinned_norm.analyze_wdg --out DIR [--group G] [--epochs 0 1 2 3 5 10 20 30 50 100]

Writes DIR/history.csv (all runs, all epochs, all scalars), prints (a) the growth-regime
table per arm (median over channels, mean over seeds) for the pair block and the
single's same block: branch norm^2, |gamma| (per branch and total), norm^2/gamma,
summed gamma^2/||w||^2 (kernel units), summed gamma^2/sigma_run^2 and gamma^2/sigma_batch^2,
output scale, cosine; (b) train/test accuracy per arm with paired gains against the single.
"""
from __future__ import annotations
import argparse, csv, sys
from collections import defaultdict
from pathlib import Path
import numpy as np


def fetch(group: str, project: str = "claude-autonomous-reparam"):
    import wandb
    api = wandb.Api(timeout=120)
    runs = [r for r in api.runs(project, filters={"group": group})]
    rows = []
    for r in runs:
        v = (r.config.get("variant") or {}).get("name") or r.name.split("/")[-1]
        seed = r.config.get("seed")
        for rec in r.scan_history():
            if "epoch" not in rec and "_step" not in rec:
                continue
            d = {k: val for k, val in rec.items() if isinstance(val, (int, float)) and not k.startswith("_")}
            d["epoch"] = rec.get("epoch", rec.get("_step"))
            d["variant"] = v; d["seed"] = seed; d["state"] = r.state
            rows.append(d)
    return rows


def block_of(variant: str) -> int:
    return 0 if "first" in variant else 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--group", default="wdconv_gauge_c100_d3")
    ap.add_argument("--epochs", type=int, nargs="+", default=[0, 1, 2, 3, 5, 10, 20, 30, 50, 75, 100, 150])
    ap.add_argument("--csv", type=Path, default=None, help="read this history.csv instead of W&B")
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    if a.csv:
        rows = list(csv.DictReader(a.csv.open()))
        for r in rows:
            for k, v in list(r.items()):
                if k not in ("variant", "state"):
                    try: r[k] = float(v) if v not in ("", None) else float("nan")
                    except ValueError: pass
    else:
        rows = fetch(a.group)
        keys = sorted({k for r in rows for k in r})
        with (a.out / "history.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
        print(f"{len(rows)} rows from {len({(r['variant'], r['seed']) for r in rows})} runs -> {a.out/'history.csv'}")
    by = defaultdict(dict)   # (variant, seed) -> epoch -> row
    for r in rows:
        if r.get("epoch") is None or (isinstance(r["epoch"], float) and np.isnan(r["epoch"])): continue
        by[(r["variant"], int(r["seed"]))][int(r["epoch"])] = r
    variants = sorted({v for v, _ in by})
    def agg(variant, epoch, key):
        vals = [by[(variant, s)][epoch].get(key, float("nan")) for (v, s) in by if v == variant and epoch in by[(v, s)]]
        vals = [x for x in vals if x is not None and not (isinstance(x, float) and np.isnan(x))]
        return (float(np.mean(vals)), len(vals)) if vals else (float("nan"), 0)
    # (a) growth-regime table
    print("\n== growth regime: medians over channels, mean over seeds; block = the pair block of the arm (block 2 for last, block 0 for first); single shown at both blocks ==")
    cols = [("norm_b0_p50", "norm b0"), ("norm_b1_p50", "norm b1"), ("gamma_p50", "|gamma|"), ("output_p50", "output"), ("n2_over_gamma_p50", "n2/gamma"),
            ("mag_k_sum_p50", "sum g2/n2"), ("mag_s_sum_p50", "sum g2/sig_run2"), ("mag_b_sum_p50", "sum g2/sig_batch2"), ("cos_p50", "cos"), ("run_sigma_b0_p50", "sig_run b0"), ("batch_sigma_b0_p50", "sig_batch b0")]
    for variant in variants:
        blocks = [0, 2] if "single" in variant else [block_of(variant)]
        for b in blocks:
            print(f"\n-- {variant}  block {b}")
            print("epoch  n  " + "  ".join(f"{lab:>16s}" for _, lab in cols))
            for e in a.epochs:
                vals = [agg(variant, e, f"pair_channel/block{b}/{k}") for k, _ in cols]
                if vals[0][1] == 0: continue
                print(f"{e:5d} {vals[0][1]:2d}  " + "  ".join(f"{v:16.3f}" for v, _ in vals))
    # (b) accuracies and paired gains vs single
    print("\n== train / test accuracy (percent, mean over seeds) and paired gain against single_d3_wdg (mean +- s.e.m. over seeds) ==")
    for e in a.epochs:
        line = []
        for variant in variants:
            tr, n = agg(variant, e, "train_accuracy")
            if n == 0: continue
            te, _ = agg(variant, e, "test_accuracy")
            gains = [100 * (by[(variant, s)][e]["train_accuracy"] - by[("single_d3_wdg", s)][e]["train_accuracy"]) for (v, s) in by if v == variant and ("single_d3_wdg", s) in by and e in by[(v, s)] and e in by[("single_d3_wdg", s)] and "train_accuracy" in by[(v, s)][e] and "train_accuracy" in by[("single_d3_wdg", s)][e]]
            g = f"{np.mean(gains):+.2f}+-{np.std(gains, ddof=1)/np.sqrt(len(gains)) if len(gains) > 1 else 0:.2f}(n={len(gains)})" if gains else "  n/a"
            line.append(f"{variant:28s} train {100*tr:6.2f} test {100*te:6.2f} gain {g}")
        if line: print(f"epoch {e}:\n  " + "\n  ".join(line))


if __name__ == "__main__":
    main()
