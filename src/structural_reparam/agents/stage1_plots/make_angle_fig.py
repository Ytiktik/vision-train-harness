"""The branch-angle trajectory, per block, over training.

Left pane: the Σ-whitened branch angle, median over the OPEN channels only. Right
pane: how many of that block's channels are still open, on a log axis.

The right pane is a percentage and not a raw count because blocks differ in width
(32, 64 and 128 here), and on a count axis block 0 — which keeps the LARGEST share
of its channels open — plots lowest. The legend carries each block's channel count
so the percentage can be converted back. The axis is linear: the claim this figure
makes lives between 10 and 100 %, and a log axis squashes exactly that range to buy
resolution below 1 %, which the QUORUM rule below already makes unnecessary.

The left pane will not draw an open-channel median over a handful of channels. A
seed contributes to the solid line only while at least QUORUM of its channels are
open; once a block falls below that the line switches to a dotted median over EVERY
channel, which is the honest description of a closed block and goes to ~0°.

A channel is open when its branch angle is at least 10 degrees. Angles are Σ-whitened
(computed on p = Σ^½w/σ, with Σ re-estimated every epoch from 20 batches of that
block's own input), because the raw Euclidean angle and the whitened one disagree and
the whitened one is what the preconditioner sees.

Curves are the mean over three seeds of the `pair_all` arm, in which every block
carries a pair, so every block has an angle to report.

Usage: python -m structural_reparam.agents.stage1_plots.make_angle_fig [--dry-run]
"""

from __future__ import annotations

import argparse
import collections
import math

from structural_reparam.agents.stage1_plots import figstyle as fs

CELLS = {
    "stage2_place_c100_d5_w1": (5, "CIFAR-100, depth 5", "angle_d5_w1_kwd.png"),
    "stage2_place_c100_d3_w1": (3, "CIFAR-100, depth 3", "angle_d3_w1_kwd.png"),
}
ARM = "pair_all"

# Fewest open channels for which a median over them is reported as a trajectory.
# Below this the block is drawn as closed and the all-channel median takes over.
QUORUM = 8


def series(group, depth):
    """-> ({block: (epochs, angle_open, n_open, angle_all, n_total)}, n_runs).

    ``angle_open`` averages the seeds that have any open channel at all, and is
    NaN when none do. ``n_open`` is the seed-mean count, and it is what decides
    where the solid line stops: the same number the right pane plots, against the
    same threshold the right pane draws, so the two panes cannot disagree.
    ``angle_all`` is the median over EVERY channel and exists at every epoch; it
    continues the curve once the block has closed, where the open-only median is a
    median over one or two channels and pure noise.
    """
    import wandb
    api = wandb.Api(timeout=60)
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    n_runs = 0
    for r in api.runs(fs.PROJECT, filters={"group": group}):
        if r.state != "finished" or not r.name.split("/")[-1].startswith(ARM):
            continue
        n_runs += 1
        # NO keys= filter: scan_history drops any row missing a requested key, and
        # the probe emits no angle for a block with zero open channels, so filtering
        # would silently truncate every block's curve at the epoch the last block
        # finishes closing.
        for h in r.scan_history(page_size=1000):
            ep = h.get("epoch")
            if ep is None:
                continue
            for b in range(depth):
                nt = h.get(f"pair_channel/block{b}/whitened_n_total")
                if not nt:
                    continue
                a = h.get(f"pair_channel/block{b}/whitened_angle_open_median")
                no = h.get(f"pair_channel/block{b}/whitened_n_open") or 0
                # a is None exactly when no channel is open, so the open-only median is
                # undefined; cos_sigma_p50 is the all-channel median and still exists
                c50 = h.get(f"pair_channel/block{b}/cos_sigma_p50")
                a_all = (math.degrees(math.acos(max(-1.0, min(1.0, c50))))
                         if c50 is not None else None)
                acc[b][ep].append((a, no, a_all, nt))
    out = {}
    for b, byep in acc.items():
        eps = sorted(byep)
        ang, ang_all, nopen = [], [], []
        for e in eps:
            rows = byep[e]
            vals = [x[0] for x in rows if x[0] is not None]
            ang.append(sum(vals) / len(vals) if vals else float("nan"))
            av = [x[2] for x in rows if x[2] is not None]
            ang_all.append(sum(av) / len(av) if av else float("nan"))
            nopen.append(sum(x[1] for x in rows) / len(rows))
        out[b] = (eps, ang, nopen, ang_all, byep[eps[0]][0][3])
    return out, n_runs


def _switch(ang, nopen):
    """Index at which the block stops having a quorate open set, for good.

    Taken as one past the LAST quorate epoch rather than the first non-quorate one,
    so a block whose count wobbles across the threshold gets one style change
    instead of a flickering line.
    """
    last = None
    for i, (v, k) in enumerate(zip(ang, nopen)):
        if v == v and k >= QUORUM:
            last = i
    if last is None:
        return 0
    return last + 1 if last + 1 < len(ang) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--outdir", default=fs.VAULT)
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm

    for group, (depth, label, fname) in CELLS.items():
        s, n = series(group, depth)
        if not s:
            print(f"  {group}: no probe history, skipped")
            continue
        colors = [cm.viridis(i / max(1, depth - 1) * 0.88) for i in range(depth)]
        fig, axes = plt.subplots(1, 2, figsize=fs.FIGSIZE)
        for b in sorted(s):
            eps, ang, nopen, ang_all, nt = s[b]
            nm = "block 0 (first)" if b == 0 else \
                 f"block {b} (last)" if b == depth - 1 else f"block {b}"
            nm = f"{nm}, {int(nt)} ch"
            j = _switch(ang, nopen)
            hi = len(eps) if j is None else j
            axes[0].plot(eps[:hi], ang[:hi], lw=1.7, color=colors[b], label=nm)
            if j is not None:
                # from the switch on, the open set is too small for its median to be
                # a trajectory, so the all-channel median takes over; it falls to ~0°
                # and matches what the right pane shows
                k = max(0, j - 1)
                axes[0].plot(eps[k:], ang_all[k:], lw=1.4, ls=":", color=colors[b])
                axes[0].plot([eps[j]], [ang_all[j]], marker="X", ms=7,
                             color=colors[b], zorder=5)
                axes[0].annotate(f"fewer than {QUORUM} channels open",
                                 (eps[j], ang_all[j]), textcoords="offset points",
                                 xytext=(9, -13), fontsize=6.5, color=colors[b],
                                 ha="left", va="top",
                                 bbox=dict(fc="white", ec="none", pad=0.6,
                                           alpha=0.85))
            axes[1].plot(eps, [k / nt * 100 for k in nopen], lw=1.7,
                         color=colors[b], label=nm)
        axes[0].axhline(90, color="grey", lw=0.8, ls=":")
        axes[0].set_ylabel("Σ-whitened branch angle (degrees)\nmedian over open channels",
                           fontsize=fs.LABEL_SIZE)
        axes[0].set_ylim(-8, 182)
        axes[0].set_yticks([0, 45, 90, 135, 180])
        axes[1].set_ylim(-3, 103)
        axes[1].set_yticks([0, 25, 50, 75, 100])
        axes[1].set_ylabel("channels open (%)", fontsize=fs.LABEL_SIZE)
        for ax in axes:
            ax.set_xlabel("epoch", fontsize=fs.LABEL_SIZE)
            ax.tick_params(labelsize=fs.TICK_SIZE)
            ax.grid(alpha=fs.GRID_ALPHA)
        axes[0].legend(fontsize=7, frameon=False)
        fig.suptitle(f"Branch angle over training, {label}",
                     fontsize=fs.TITLE_SIZE, y=1.02)
        fig.tight_layout()
        p = f"{a.outdir}/{fname}"
        if a.dry_run:
            print(f"  [dry-run] {p}")
        else:
            fig.savefig(p, dpi=fs.DPI, bbox_inches="tight")
            print(f"  wrote {p}  ({n} runs)")
        plt.close(fig)
        for b in sorted(s):
            eps, ang, nopen, ang_all, nt = s[b]
            j = _switch(ang, nopen)
            tail = (f"open-median {ang[-1]:6.1f}°"
                    if ang[-1] == ang[-1] and nopen[-1] >= QUORUM
                    else f"all-channel median {ang_all[-1]:5.2f}°")
            sw = "never closes" if j is None else f"closes at epoch {eps[j]}"
            print(f"    block {b}: epoch {eps[0]} {ang[0]:6.1f}° ({nopen[0]:.1f}/{nt}"
                  f" open)   epoch {eps[-1]} {tail} ({nopen[-1]:.1f}/{nt} open)"
                  f"   {sw}")


if __name__ == "__main__":
    main()
