"""Figure: running train accuracy per epoch for the gauge tests in the pinned-norm
cell (single vs block-only shared-gamma pairs: ungauged open init, gauged open
init, gauged collapsed init), one panel per block (block 0, last block).
Data: W&B project claude-autonomous-reparam, group pinned_c100_d3_wd0 (falls back
to local JSON-lines logs with --log-dir). Mean over seeds 42-44, shaded min-max.
  python -m structural_reparam.experiments.reparam_pinned_norm.plot_gauge_curves --out <png>
"""
from __future__ import annotations
import argparse, glob, json
from pathlib import Path
import numpy as np

PANELS = [
    ("Block 0 pair only", [("single_d3_pinned", "single"), ("pair_first_d3_pinned", "pair, ungauged (kernels at r), open init"),
                           ("pair_first_d3_pinned_gauged", "pair, gauged (r/√2), open init"), ("pair_first_d3_pinned_gauged_collapsed", "pair, gauged, collapsed init")]),
    ("Last block pair only", [("single_d3_pinned", "single"), ("pair_last_d3_pinned", "pair, ungauged (kernels at r), open init"),
                              ("pair_last_d3_pinned_gauged", "pair, gauged (r/√2), open init"), ("pair_last_d3_pinned_gauged_collapsed", "pair, gauged, collapsed init")]),
]
COLORS = {"single": "#4a3aa7", "ungauged": "#eb6834", "gauged": "#2a78d6", "collapsed": "#1baf7a"}
STYLES = {"single": "-", "ungauged": "-", "gauged": "--", "collapsed": ":"}

def key_of(variant):
    if variant == "single_d3_pinned": return "single"
    if variant.endswith("_gauged_collapsed"): return "collapsed"
    if variant.endswith("_gauged"): return "gauged"
    return "ungauged"

def from_wandb(variants, seeds, group="pinned_c100_d3_wd0", entity_project="yoovi-t-tel-aviv-university/claude-autonomous-reparam"):
    import wandb
    api = wandb.Api(); out = {}
    for r in api.runs(entity_project, filters={"group": group}):
        v = (r.config.get("variant") or {}).get("name") if isinstance(r.config.get("variant"), dict) else None
        v = v or r.name.split("/")[-2] if "/" in r.name else v
        name = r.name
        for cand in variants:
            if name.endswith(cand) or f"/{cand}/" in name or name == cand or (v == cand):
                seed = r.config.get("seed") or r.config.get("base_seed")
                h = r.history(keys=["epoch", "train_accuracy"], pandas=False)
                out.setdefault(cand, {})[int(seed)] = {int(x["epoch"]): float(x["train_accuracy"]) for x in h if x.get("train_accuracy") is not None}
    return out

def from_logs(log_dir):
    out = {}
    for f in glob.glob(str(Path(log_dir) / "*.log")):
        for l in open(f):
            if l.startswith('{"epoch"'):
                r = json.loads(l); out.setdefault(r["variant"], {}).setdefault(int(r["seed"]), {})[int(r["epoch"])] = float(r["train_accuracy"])
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--log-dir", default=None); ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    a = ap.parse_args()
    variants = sorted({v for _, arms in PANELS for v, _ in arms})
    data = from_logs(a.log_dir) if a.log_dir else from_wandb(variants, a.seeds)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (title, arms) in zip(axes, PANELS):
        for v, label in arms:
            runs = data.get(v, {}); k = key_of(v)
            if not runs: continue
            eps = sorted(set.intersection(*[set(d) for d in runs.values()]))
            if not eps: continue
            M = np.array([[100 * runs[s][e] for e in eps] for s in runs])
            ax.plot(eps, M.mean(0), STYLES[k], color=COLORS[k], lw=2, label=f"{label} (n={len(runs)})")
            ax.fill_between(eps, M.min(0), M.max(0), color=COLORS[k], alpha=0.15, lw=0)
            ax.annotate(f"{M.mean(0)[-1]:.1f}", (eps[-1], M.mean(0)[-1]), textcoords="offset points", xytext=(4, 0), fontsize=8, color="#333")
        ax.set_title(title); ax.set_xlabel("epoch"); ax.grid(alpha=0.25); ax.set_xlim(0, 105)
    axes[0].set_ylabel("running train-mode accuracy (%)")
    axes[1].legend(fontsize=8, loc="lower right", frameon=False)
    fig.suptitle("Pinned-norm cell (no weight decay, constant lr 0.1): single vs block-only shared-γ pair, mean over seeds (band = min-max)", fontsize=10)
    fig.tight_layout(); fig.savefig(a.out, dpi=160); print("saved", a.out)

if __name__ == "__main__":
    main()
