"""The synthesis figure at the head of section 4: a single-branch network with the two
interventions of the paper's two mechanisms — block 0's kernel gradient along v_max
cooled by the pair's own factor, and the last block's gamma x4, beta x2 learning-rate
multiplier — against the all-paired network, over training, in the default cell.

``synthesis_d5_c100_kwd.png``: paired per-seed train gain (left) and test gain (right)
as a function of epoch, each arm minus the paper's ``single`` of
``stage2_place_c100_d5_w1`` at the same seed; band = standard error over three seeds
(42 to 44). Curves: the paper's ``pair_all`` (every block paired, shared gamma) from
that same group; and from ``stage2_synth_c100_d5_w1`` the cooling alone, the boost
alone, and both together (solid at cooling factor 0.048, the measured coefficient of
the preconditioner along v_max, pair over single; dashed at 0.09, cos^2(theta/2)).
Every arm except ``pair_all`` has one branch everywhere. The grey number at the end of
each curve is the single's absolute accuracy for that pane at epoch 100.

Usage: python -m structural_reparam.agents.stage1_plots.make_synthesis_fig [--dry-run]
"""

from __future__ import annotations

import argparse
import statistics as st
from pathlib import Path

from structural_reparam.agents.stage1_plots import figstyle as fs
from structural_reparam.agents.stage1_plots.make_speedup_figs import (
    EPOCHS, _api, _curve, _draw_curves, _latest, _save)

REF_GROUP = "stage2_place_c100_d5_w1"
SYN_GROUP = "stage2_synth_c100_d5_w1"
SEEDS = (42, 43, 44)
FNAME = "synthesis_d5_c100_kwd.png"

ARMS = [  # (label, group, arm, colour, line style)
    ("every block paired (the pair)", REF_GROUP, "pair_all", "black", "-"),
    ("single, cooling ×0.048 + last-block γ×4 β×2", SYN_GROUP, "cool048_g4b2", fs.COLORS[1], "-"),
    ("single, cooling ×0.09 + last-block γ×4 β×2", SYN_GROUP, "cool090_g4b2", fs.COLORS[1], "--"),
    ("single, last-block γ×4 β×2 alone", SYN_GROUP, "g4b2_last", fs.COLORS[3], "-"),
    ("single, cooling ×0.048 alone", SYN_GROUP, "cool048_first", fs.COLORS[2], "-"),
    ("single, cooling ×0.09 alone", SYN_GROUP, "cool090_first", fs.COLORS[2], "--"),
]


def paired_cross(singles, arm_runs):
    """Per-epoch paired gain of arm_runs[s] minus singles[s] over the seeds both have."""
    seeds = sorted(s for s in singles if s in arm_runs)
    S = {s: _curve(singles[s]) for s in seeds}
    P = {s: _curve(arm_runs[s]) for s in seeds}
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
    bte = [S[s][100][1] for s in seeds if 100 in S[s] and S[s][100][1] is not None]
    return dict(train=tr, test=te, base_train=base_tr, base_test=st.mean(bte) if bte else float("nan"), n=len(seeds))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--outdir", default=fs.VAULT)
    args = ap.parse_args()
    api = _api()
    ref = _latest(api.runs(fs.PROJECT, filters={"group": REF_GROUP}))
    syn = _latest(api.runs(fs.PROJECT, filters={"group": SYN_GROUP}))
    singles = {s: ref[("single", s)] for s in SEEDS if ("single", s) in ref}
    curves, colors, styles = [], [], []
    for label, group, arm, col, ls in ARMS:
        pool = ref if group == REF_GROUP else syn
        runs = {s: pool[(arm, s)] for s in SEEDS if (arm, s) in pool}
        if not runs:
            print(f"  (no finished runs for {arm}; skipped)")
            continue
        curves.append((label, paired_cross(singles, runs)))
        colors.append(col)
        styles.append(ls)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=fs.FIGSIZE)
    for ax, field, base_key, pane in ((axes[0], "train", "base_train", "train"),
                                      (axes[1], "test", "base_test", "test")):
        _draw_curves(ax, curves, field, base_key, colors, styles)
        ax.set_ylabel(f"{pane}-accuracy gain over the single (pp)", fontsize=fs.LABEL_SIZE)
    lo, hi = axes[0].get_ylim()
    axes[0].set_ylim(lo - 0.42 * (hi - lo), hi)
    axes[0].legend(fontsize=6.8, frameon=False, loc="lower left")
    fig.suptitle("The pair rebuilt from a single branch, CIFAR-100 depth 5", fontsize=fs.TITLE_SIZE, y=1.02)
    fig.tight_layout()
    _save(fig, FNAME, args.dry_run, args.outdir)
    for label, c in curves:
        t = c["train"]
        w = [t[e][0] for e in range(81, 101) if e in t]
        print(f"    {label}: ep5 {t[5][0]:+.2f} ± {t[5][1]:.2f}, ep25 {t[25][0]:+.2f}, ep50 {t[50][0]:+.2f}, "
              f"ep100 {t[100][0]:+.2f} ± {t[100][1]:.2f}, window 81-100 {st.mean(w):+.2f}; "
              f"test@100 {c['test'][100][0]:+.2f} ± {c['test'][100][1]:.2f}; single train@100 {c['base_train']:.1f}; n={c['n']}")


if __name__ == "__main__":
    main()
