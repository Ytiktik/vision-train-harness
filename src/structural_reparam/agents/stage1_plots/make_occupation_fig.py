"""Occupation of the top eigendirection over training, ten settings.

One pane: F(t), the fraction of the currently-open block-0 channels (whitened
angle >= 10 deg, at least 8 open) whose separation axis is parked on the top
eigendirection (alignment >= 0.95) at optimizer step t, on a log time axis.
This is section 3.3's selection measured as an occupation in time, the clock of
section 4.2.5; the curves
replace any single crossing-time table, since every quantile is readable off
them. The half-crossing is marked per curve; the flat doses lie on the floor.

Data: the pair_channel probe artifacts of the constant-rate wdg cell —
`wdconv_gauge_{svhn,gtsrb,eurosat,flowers}_d3` (arm pair_first_d3_wdg) and
`wdconv_gauge_c100_zca_d3` (the four doses) and `wdconv_gauge_c100_damptop_d3`
(pair_c100, the unwhitened CIFAR-100 anchor since 2026-09-02; the ZCA group's
pair_raw, used until then, agrees with it) — step records every
20 steps for the first 3 epochs, epoch snapshots after, three seeds. Artifacts
are fetched to a local cache on first use.

Colours and markers match make_generality_fig.py: one shape per dataset, the
ZCA doses in one red family fading with the dose. Datasets solid, doses dashed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from structural_reparam.agents.stage1_plots import figstyle as fs

CACHE = Path("/tmp/nf_probes")
SEEDS = (42, 43, 44)
THRESHOLD = 0.95     # "parked": alignment at or above this
OPEN_DEG = 10.0      # a channel counts as open at this whitened angle
MIN_OPEN = 8         # a record needs at least this many open channels

# label, group, variant, steps per epoch, mu_eff, colour, dashed
# mu_eff = (lambda_max - lambda_next)/tr Sigma, the predictor of section 3.6
# (equation 64); the 50-batch seed-0 draw of scripts/analysis/quiet_band.py clock.
# The legend carried lambda_max/lambda_next until 2026-09-04.
SETTINGS = [
    ("sharpened α=0.30", "wdconv_gauge_c100_sharp_d3", "pair_sharp03", 391, 0.834, "#3f007d", True),
    ("GTSRB",       "wdconv_gauge_gtsrb_d3",    "pair_first_d3_wdg", 208, 0.745, "#8c564b", False),
    ("SVHN",        "wdconv_gauge_svhn_d3",     "pair_first_d3_wdg", 573, 0.708, "#2ca02c", False),
    ("EuroSAT",     "wdconv_gauge_eurosat_d3",  "pair_first_d3_wdg", 169, 0.681, "#9467bd", False),
    ("CIFAR-100",   "wdconv_gauge_c100_damptop_d3", "pair_c100",     391, 0.633, "#17becf", False),   # the Tables 12 and 13 unwhitened runs (was the ZCA group's pair_raw until 2026-09-02)
    ("Flowers-102", "wdconv_gauge_flowers_d3",  "pair_first_d3_wdg",  48, 0.554, "#1f77b4", False),
    ("ZCA ε=10",    "wdconv_gauge_c100_zca_d3", "pair_z10",          391, 0.317, "#67000d", True),
    ("ZCA ε=1",     "wdconv_gauge_c100_zca_d3", "pair_z1",           391, 0.112, "#a50f15", True),
    ("ZCA ε=0.1",   "wdconv_gauge_c100_zca_d3", "pair_z01",          391, 0.004, "#ef3b2c", True),
    ("ZCA ε=0.01",  "wdconv_gauge_c100_zca_d3", "pair_z001",         391, 0.006, "#fc9272", True),
]


def probe_npz(group: str, variant: str, seed: int) -> Path:
    name = f"pair_channel_{group}_{variant}_s{seed}"
    npz = CACHE / name / f"{variant}_seed{seed}.npz"
    if not npz.exists():
        import wandb
        wandb.Api(timeout=120).artifact(
            f"{fs.PROJECT}/{name}:latest", type="probe").download(root=str(npz.parent))
    return npz


def seed_curve(group: str, variant: str, seed: int, spe: int):
    """(times, F) for one run: occupation over the merged step+epoch timeline."""
    d = np.load(probe_npz(group, variant, seed))
    times, fracs = [], []
    for kind, tv in (("step", d["step_step"].astype(float)),
                     ("epoch", d["epoch_epoch"].astype(float) * spe)):
        al = d[f"{kind}_block0_align_vmax"]
        wa = d[f"{kind}_block0_wangle"]
        for r in range(al.shape[0]):
            op = wa[r] >= OPEN_DEG
            if op.sum() >= MIN_OPEN:
                times.append(tv[r])
                fracs.append(float((al[r][op] >= THRESHOLD).mean()))
    o = np.argsort(times)
    return np.array(times)[o], np.array(fracs)[o]


def crossing(times, fracs, level=0.5):
    i = np.where(fracs >= level)[0]
    return float(times[i[0]]) if len(i) else float("inf")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--outdir", default=str(fs.VAULT))
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.0))

    print(f"half-times (steps to F >= 0.5 at threshold {THRESHOLD}), per seed and median:")
    for label, group, variant, spe, mu, col, dashed in SETTINGS:
        curves = [seed_curve(group, variant, s, spe) for s in SEEDS]
        # one step-function grid per setting: the union of the seeds' times
        grid = np.unique(np.concatenate([t for t, _ in curves]))
        per_seed = []
        for t, f in curves:
            idx = np.searchsorted(t, grid, side="right") - 1
            per_seed.append(np.where(idx >= 0, f[np.clip(idx, 0, None)], np.nan))
        mean = np.nanmean(per_seed, 0)
        ls = (0, (4, 2)) if dashed else "-"
        for f in per_seed:
            ax.plot(grid, f, color=col, lw=0.6, alpha=0.25, ls=ls, zorder=2)
        ax.plot(grid, mean, color=col, lw=1.6, ls=ls, zorder=3,
                label=f"{label}  ({mu:.3f})")
        halves = [crossing(t, f) for t, f in curves]
        t50 = float(np.median(halves))
        if np.isfinite(t50):
            ax.plot([t50], [0.5], marker="o", ms=5, color=col,
                    markeredgecolor="white", markeredgewidth=0.6, zorder=4)
        show = ", ".join("never" if not np.isfinite(h) else f"{h:.0f}" for h in halves)
        print(f"  {label:12s} μ_eff {mu:5.3f}: {show}   median "
              f"{'never' if not np.isfinite(t50) else f'{t50:.0f}'}")

    ax.set_xscale("log")
    ax.set_xlim(18, 42000)
    ax.set_ylim(-0.02, 1.02)
    ax.axhline(0.5, color="0.75", lw=0.7, zorder=1)
    ax.set_xlabel("optimizer step (log scale)", fontsize=fs.LABEL_SIZE)
    ax.set_ylabel("fraction of open channels parked\n(alignment ≥ 0.95)",
                  fontsize=fs.LABEL_SIZE)
    ax.grid(alpha=fs.GRID_ALPHA, zorder=0)
    ax.tick_params(labelsize=fs.TICK_SIZE)
    ax.legend(fontsize=7, frameon=False, ncols=2, loc="upper left",
              title="setting  (μ_eff)", title_fontsize=7.5)
    ax.set_title("Occupation of the top eigendirection, ten settings",
                 fontsize=fs.TITLE_SIZE)
    fig.tight_layout()

    p = f"{a.outdir}/block0_occupation_kwd.png"
    if a.dry_run:
        print(f"  [dry-run] {p}")
    else:
        fig.savefig(p, dpi=fs.DPI, bbox_inches="tight")
        print(f"  wrote {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
