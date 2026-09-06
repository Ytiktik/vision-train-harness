"""The speed-up figures of sections 1.3 and 4.2.1: the two-branch gain over training.

Three figures, drawn to the conventions in `figstyle`:

* ``speedup_depth_c100_kwd.png`` and ``speedup_depth_c10_kwd.png``: the whole-network
  pair (``indep2``, the Figure 1 and 2 arm) minus its single, paired per seed, as a
  function of epoch, one curve per depth, train gain left and test gain right. The
  band is the standard error of the paired difference over seeds. Each curve ends
  with the single's absolute accuracy for that pane at epoch 100, in grey.
* ``speedup_first_block_kwd.png``: the first-block-only pair (``pair_first`` of the
  stage2 placement groups) minus its single, over training, at depths 3, 5 and 8 under
  constant learning rate (solid) and depths 3 and 5 under the cosine schedule (dashed).
  Train pane only: at constant rate the per-epoch test accuracy is a single un-annealed
  snapshot that swings by several points, so a test pane would show noise.

Data: W&B groups ``kwd_depth_c100`` and ``kwd_depth_c10`` (six seeds at CIFAR-100 depths
3, 5, 8, three elsewhere), ``stage2_constlr_c100_{d3_w1,d5_w1,d8_w1}`` and
``stage2_place_c100_{d3_w1,d5_w1}`` (three seeds, 42 to 44).

Usage: python -m structural_reparam.agents.stage1_plots.make_speedup_figs [--dry-run]
"""

from __future__ import annotations

import argparse
import collections
import math
import re
import statistics as st
from pathlib import Path

from structural_reparam.agents.stage1_plots import figstyle as fs

DEPTHS = {"kwd_depth_c100": (3, 5, 8, 10, 12, 15), "kwd_depth_c10": (3, 4, 5, 6, 8, 15)}
EPOCHS = list(range(1, 101))


def _api():
    import wandb
    return wandb.Api(timeout=60)


def _latest(runs):
    """Newest finished run per (arm, seed); a rerun replaces its predecessor."""
    out = {}
    for r in runs:
        if r.state != "finished":
            continue
        m = re.match(r"(.+)_s(\d+)$", r.name.split("/")[-1])
        if not m:
            continue
        key = (m.group(1), int(m.group(2)))
        if key not in out or str(r.created_at) > str(out[key].created_at):
            out[key] = r
    return out


def _curve(run):
    h = run.history(keys=["epoch", "train_accuracy", "test_accuracy"], pandas=False)
    c = {}
    for x in h:
        tr, te = _finite(x.get("train_accuracy")), _finite(x.get("test_accuracy"))
        if tr is None:
            continue
        c[int(x["epoch"])] = (fs.pct(tr), fs.pct(te) if te is not None else None)
    return c


def _finite(v):
    """W&B history hands back NaN as a float or as the string 'NaN'; both mean 'not logged'."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def paired_curves(api, group, single_arm, pair_arm):
    """Per-epoch paired gain (train, test) with standard errors, plus the single's endpoint."""
    runs = _latest(api.runs(fs.PROJECT, filters={"group": group}))
    seeds = sorted(s for (a, s) in runs if a == single_arm and (pair_arm, s) in runs)
    S = {s: _curve(runs[(single_arm, s)]) for s in seeds}
    P = {s: _curve(runs[(pair_arm, s)]) for s in seeds}
    tr, te = {}, {}
    for e in EPOCHS:
        d_tr = [P[s][e][0] - S[s][e][0] for s in seeds if e in P[s] and e in S[s]]
        d_te = [P[s][e][1] - S[s][e][1] for s in seeds
                if e in P[s] and e in S[s] and P[s][e][1] is not None and S[s][e][1] is not None]
        if d_tr:
            tr[e] = fs.mean_se(d_tr)[:2]
        if d_te:
            te[e] = fs.mean_se(d_te)[:2]
    base_tr = st.mean(S[s][100][0] for s in seeds if 100 in S[s])
    base_te_vals = [S[s][100][1] for s in seeds if 100 in S[s] and S[s][100][1] is not None]
    base_te = st.mean(base_te_vals) if base_te_vals else float("nan")
    return dict(train=tr, test=te, base_train=base_tr, base_test=base_te, n=len(seeds))


def _spread(ys, min_gap):
    """Nudge label positions apart so end-of-curve annotations never overprint."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    pos = [ys[i] for i in order]
    for k in range(1, len(pos)):
        if pos[k] - pos[k - 1] < min_gap:
            pos[k] = pos[k - 1] + min_gap
    out = [0.0] * len(ys)
    for k, i in enumerate(order):
        out[i] = pos[k]
    return out


def _draw_curves(ax, curves, field, base_key, colors, styles, annotate=True):
    ax.axhline(0.0, color="black", lw=1.0, zorder=1)
    ends, allys = [], []
    for (label, c), col, ls in zip(curves, colors, styles):
        xs = sorted(c[field])
        ys = [c[field][e][0] for e in xs]
        es = [c[field][e][1] for e in xs]
        allys += ys
        ax.plot(xs, ys, color=col, lw=1.5, ls=ls, label=label, zorder=3)
        ax.fill_between(xs, [y - s for y, s in zip(ys, es)], [y + s for y, s in zip(ys, es)],
                        color=col, alpha=0.16, lw=0, zorder=2)
        ends.append((xs[-1], ys[-1], c[base_key]) if xs else None)
    if annotate and ends:
        span = (max(allys) - min(allys)) or 1.0
        spread = _spread([e[1] for e in ends], 0.07 * span)
        for (x, _y, base), ypos in zip(ends, spread):
            ax.annotate(f"{base:.1f}", (x, ypos), textcoords="offset points",
                        xytext=(4, 0), ha="left", va="center", fontsize=fs.ANNOT_SIZE,
                        color="#444444")
    ax.set_xlim(0, 108)
    ax.set_xlabel("epoch", fontsize=fs.LABEL_SIZE)
    ax.tick_params(labelsize=fs.TICK_SIZE)
    ax.grid(alpha=fs.GRID_ALPHA, zorder=0)


def depth_figure(api, group, ds, fname, dry, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    depths = DEPTHS[group]
    curves = [(f"depth {d}", paired_curves(api, group, f"single_d{d}", f"indep2_d{d}")) for d in depths]
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / max(1, len(depths) - 1) * 0.9) for i in range(len(depths))]
    fig, axes = plt.subplots(1, 2, figsize=fs.FIGSIZE)
    for ax, field, base_key, pane in ((axes[0], "train", "base_train", "train"),
                                      (axes[1], "test", "base_test", "test")):
        _draw_curves(ax, curves, field, base_key, colors, ["-"] * len(curves))
        ax.set_ylabel(f"two-branch {pane}-accuracy gain (pp)", fontsize=fs.LABEL_SIZE)
    axes[0].legend(fontsize=7.5, frameon=False, ncol=2)
    fig.suptitle(f"Gain over training by depth, {ds}", fontsize=fs.TITLE_SIZE, y=1.02)
    fig.tight_layout()
    _save(fig, fname, dry, outdir)
    for label, c in curves:
        print(f"    {label}: train gain ep5 {c['train'][5][0]:+.2f} ± {c['train'][5][1]:.2f}, "
              f"ep15 {c['train'][15][0]:+.2f} ± {c['train'][15][1]:.2f}, ep25 {c['train'][25][0]:+.2f}, "
              f"ep100 {c['train'][100][0]:+.2f}; single train@100 {c['base_train']:.1f}; n={c['n']}")


def first_block_figure(api, fname, dry, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    spec = [("depth 3, constant rate", "stage2_constlr_c100_d3_w1", "-"),
            ("depth 5, constant rate", "stage2_constlr_c100_d5_w1", "-"),
            ("depth 8, constant rate", "stage2_constlr_c100_d8_w1", "-"),
            ("depth 3, cosine", "stage2_place_c100_d3_w1", "--"),
            ("depth 5, cosine", "stage2_place_c100_d5_w1", "--")]
    curves = [(lab, paired_curves(api, g, "single", "pair_first")) for lab, g, _ in spec]
    colors = [fs.COLORS[0], fs.COLORS[1], fs.COLORS[2], fs.COLORS[0], fs.COLORS[1]]
    fig, ax = plt.subplots(1, 1, figsize=(fs.FIGSIZE[0] * 0.55, fs.FIGSIZE[1]))
    _draw_curves(ax, curves, "train", "base_train", colors, [s for _, _, s in spec])
    ax.set_ylabel("first-block pair, train-accuracy gain (pp)", fontsize=fs.LABEL_SIZE)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - 0.55 * (hi - lo), hi)   # room for the legend below the curves
    ax.legend(fontsize=7.5, frameon=False, loc="lower left", ncol=2)
    fig.suptitle("First-block pair over training, CIFAR-100", fontsize=fs.TITLE_SIZE, y=1.02)
    fig.tight_layout()
    _save(fig, fname, dry, outdir)
    for label, c in curves:
        w = [c["train"][e][0] for e in range(81, 101) if e in c["train"]]
        print(f"    {label}: ep5 {c['train'][5][0]:+.2f}, ep25 {c['train'][25][0]:+.2f}, "
              f"ep100 {c['train'][100][0]:+.2f}, window 81-100 {st.mean(w):+.2f}; single@100 {c['base_train']:.1f}; n={c['n']}")


def _save(fig, fname, dry, outdir):
    import matplotlib.pyplot as plt
    p = Path(outdir) / fname
    if dry:
        print(f"  [dry-run] {p}")
    else:
        Path(outdir).mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=fs.DPI, bbox_inches="tight")
        print(f"  wrote {p}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--outdir", default=fs.VAULT)
    args = ap.parse_args()
    api = _api()
    depth_figure(api, "kwd_depth_c100", "CIFAR-100", "speedup_depth_c100_kwd.png", args.dry_run, args.outdir)
    depth_figure(api, "kwd_depth_c10", "CIFAR-10", "speedup_depth_c10_kwd.png", args.dry_run, args.outdir)
    first_block_figure(api, "speedup_first_block_kwd.png", args.dry_run, args.outdir)


if __name__ == "__main__":
    main()
