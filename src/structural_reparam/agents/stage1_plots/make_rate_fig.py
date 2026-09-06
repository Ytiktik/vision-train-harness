"""How fast the first block's open channels align and open.

Figure 15 samples epochs {0, 5, 25, ...} and the separation-axis alignment is
already at 0.995 by the first sample, so the paper never shows the rate. This
figure shows the first five epochs at optimizer-step resolution. Default source
(--source blockmeasure, since 2026-09-03): the per-step measurement module's derived
arrays of the default cell (outputs/blockmeasure_c100_d5_w1/derived/pair_all_seed*,
batch frame; whitened angle and |axis . v_max| at every step of epochs 1 to 5, open
and non-dead channels). --source stepwise reads the retired recordings of
scripts/analysis/block0_stepwise.py (--prefix d5all, or signed for the original
depth-3 constant-rate recordings).

Left pane: |separation axis . top eigendirection| per step, median over the
open channels of three seeds, band = 25th to 75th percentile. Right pane: the
whitened branch angle theta, same statistic. Open means theta >= 10 degrees at
that step. 391 steps per epoch.

Also prints the rate numbers: per-channel steps from alignment 0.5 to 0.9, and
the median angle at the start and end of the window.
"""
from __future__ import annotations

import argparse

import numpy as np

from structural_reparam.agents.stage1_plots import figstyle as fs

OPEN_RAD = np.deg2rad(10.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--outdir", default=str(fs.VAULT))
    ap.add_argument("--prefix", default="d5all")
    ap.add_argument("--source", default="blockmeasure", choices=["blockmeasure", "stepwise"])
    ap.add_argument("--derived-root", default="outputs/blockmeasure_c100_d5_w1/derived")
    a = ap.parse_args()

    als, ths = [], []
    cross = []
    for seed in (42, 43, 44):
        if a.source == "stepwise":
            d = np.load("outputs/block0_stepwise/%s_s%d.npz" % (a.prefix, seed))
            th = d["theta"]
            e2 = d["e2vec"].astype(np.float64)
            v = d["vtop"].astype(np.float64)
            al = np.abs(np.einsum("tcd,d->tc", e2, v))
            m = th >= OPEN_RAD
            th_deg = np.degrees(th)
        else:
            d = np.load("%s/pair_all_seed%d/derived_batch.npz" % (a.derived_root, seed))
            early = np.isin(d["epoch"], [1, 2, 3, 4, 5])
            al = d["alignment_pre"][early]
            th_deg = d["pre_theta_deg"][early]
            m = d["open_pre"][early] & ~d["dead"][early]
        als.append(np.where(m, al, np.nan))
        ths.append(np.where(m, th_deg, np.nan))
        T, C = al.shape
        for c in range(C):
            i5 = np.where(al[:, c] >= 0.5)[0]
            i9 = np.where(al[:, c] >= 0.9)[0]
            if len(i5) and len(i9) and i9[0] >= i5[0]:
                cross.append(int(i9[0]) - int(i5[0]))
    A = np.concatenate(als, axis=1)   # steps x (channels*seeds)
    TH = np.concatenate(ths, axis=1)
    steps = np.arange(A.shape[0])

    def bands(X):
        return (np.nanmedian(X, 1), np.nanquantile(X, 0.25, 1),
                np.nanquantile(X, 0.75, 1))

    am, alo, ahi = bands(A)
    tm, tlo, thi = bands(TH)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=fs.FIGSIZE, sharex=True)
    for ax, (m, lo, hi), ylab in zip(
            axes, [(am, alo, ahi), (tm, tlo, thi)],
            ["|separation axis · top eigendirection|", "branch angle θ (deg)"]):
        ax.fill_between(steps, lo, hi, alpha=0.20, color=fs.COLORS[0], zorder=2)
        ax.plot(steps, m, lw=1.4, color=fs.COLORS[0], zorder=3,
                label="median over open channels")
        for ep in range(1, 5):
            ax.axvline(391 * ep, color="0.8", lw=0.7, zorder=1)
        ax.set_xlabel("optimizer step (391 per epoch)", fontsize=fs.LABEL_SIZE)
        ax.set_ylabel(ylab, fontsize=fs.LABEL_SIZE)
        ax.grid(alpha=fs.GRID_ALPHA, zorder=0)
        ax.tick_params(labelsize=fs.TICK_SIZE)
    axes[0].set_ylim(0, 1.03)
    axes[0].legend(fontsize=7.5, frameon=False, loc="lower right")
    fig.suptitle("Alignment and angle of the first block's open channels, "
                 "first five epochs, CIFAR-100", fontsize=fs.TITLE_SIZE, y=1.0)
    fig.tight_layout()
    p = f"{a.outdir}/block0_rate_kwd.png"
    if a.dry_run:
        print(f"  [dry-run] {p}")
    else:
        fig.savefig(p, dpi=fs.DPI, bbox_inches="tight")
        print(f"  wrote {p}")
    plt.close(fig)

    cr = np.array(cross)
    print(f"  alignment 0.5 -> 0.9 per channel: median {np.median(cr):.0f} steps "
          f"p25 {np.quantile(cr, 0.25):.0f} p75 {np.quantile(cr, 0.75):.0f} "
          f"(n={len(cr)} channels; 391 steps per epoch)")
    print(f"  median alignment: step 0 {am[0]:.2f}, step 391 {am[391]:.2f}, "
          f"end {am[-1]:.2f}")
    print(f"  median angle: step 0 {tm[0]:.0f} deg, step 391 {tm[391]:.0f}, "
          f"end {tm[-1]:.0f}")


if __name__ == "__main__":
    main()
