"""Generality of the open-pair geometry: ten settings against the predictor.

Two single-pane figures over one x-axis, mu_eff = (lambda_max - lambda_next)/tr
Sigma, the paper's predictor of selection (section 3.6, equation 64):

    generality_alignment_kwd.png  the separation axis's alignment with v_max
    generality_open_kwd.png       the open-channel fraction of block 0

Until 2026-09-04 this script drew one three-pane figure whose x-axis was the
runner-up ratio lambda_max/lambda_next on a log scale. The axis moved to mu_eff
that day, and the panes were split into separate figures the same day: the
resting-angle pane became redundant with the stability figure, which carries the
same numbers on the same axis, so it is not drawn here any more.

mu_eff values are the 50-batch seed-0 draw of scripts/analysis/quiet_band.py
clock, the same draw the tables use. The axis is linear, not logarithmic: mu_eff
is a bounded share of the input variance and the two flat doses belong at zero.

Each point is a block-0 pair, median over open channels at the final epoch,
mean over three seeds. Values are the ones tabulated in section 4.2.5; sources
per row are in DATA below.

    dataset rows: W&B groups wdconv_gauge_{svhn,gtsrb,eurosat,flowers}_d3
      (pair_channel probe summaries); CIFAR-100 from the epoch-100 checkpoints
      of wdconv_gauge_c100_d3 pair_first (the same constant-rate wdg cell,
      remeasured 2026-08-30).
    dose rows: W&B group wdconv_gauge_c100_zca_d3 (pair_channel summaries); the
      sharpening dose from wdconv_gauge_c100_sharp_d3 pair_sharp03 checkpoints via
      scripts/analysis/sharpen_readback.py.

The sharpening dose of Table 13, alpha = +0.30 of the same (Sigma_img + eps I)^alpha
family the ZCA doses use with alpha = -1/2, was added on 2026-09-06; it is the one
setting whose mu_eff sits above the datasets' range, and it is drawn as a square in a
purple hue so that it is not read as one of the red whitening doses.
"""
from __future__ import annotations

import argparse

import numpy as np

from structural_reparam.agents.stage1_plots import figstyle as fs

# Datasets carry their identity in the marker SHAPE (one shape each, so the
# figure survives colour-blind viewing); the ZCA doses are all squares in one
# hue, red fading dark-to-light with the dose.
#        label        mu_eff  angle  align  open_frac  marker  color


DATA = [
    # label, mu_eff, angle (p25, med, p75), alignment (p25, med, p75),
    #   open fraction, marker, colour — one covariance convention throughout:
    #   the augmented training stream (through the model's own transform for the doses).
    ("sharpened α=0.30", 0.834, (144, 150, 161), (0.996, 0.999, 1.000), (61, 61.3, 62), "s", "#3f007d"),
    ("Flowers-102", 0.554, (115, 130, 145), (0.963, 0.991, 0.997), (49, 49.3, 50), "o", "#1f77b4"),
    ("CIFAR-100",   0.633, (133, 146, 154), (0.993, 0.999, 1.000), (51, 51.3, 52), "^", "#17becf"),
    ("SVHN",        0.708, (139, 143, 149), (0.994, 0.998, 0.999), (59, 60.0, 61), "v", "#2ca02c"),
    ("EuroSAT",     0.681, (134, 143, 154), (0.992, 0.997, 0.999), (49, 50.3, 52), "D", "#9467bd"),
    ("GTSRB",       0.745, (147, 153, 158), (0.988, 0.996, 0.999), (58, 60.7, 62), "P", "#8c564b"),
    ("ZCA ε=10",    0.317, (107, 125, 139), (0.985, 0.996, 0.997), (43, 43.7, 45), "s", "#67000d"),
    ("ZCA ε=1",     0.112, (71,  107, 122), (0.979, 0.986, 0.990), (41, 41.7, 42), "s", "#a50f15"),
    ("ZCA ε=0.1",   0.004, (58,  77,  91),  (0.373, 0.422, 0.453), (27, 28.7, 31), "s", "#ef3b2c"),
    ("ZCA ε=0.01",  0.006, (36,  49,  64),  (0.299, 0.387, 0.575), (17, 18.0, 19), "s", "#fc9272"),
]


ISO_NULL = 0.13   # |axis . v_max| for a direction drawn isotropic in the whitened frame


def _axis(ax, fs):
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8])
    ax.set_xlim(-0.04, 0.88)
    ax.set_xlabel("μ_eff = (λ_max − λ_next)/tr Σ", fontsize=fs.LABEL_SIZE)
    ax.grid(alpha=fs.GRID_ALPHA, zorder=0)
    ax.tick_params(labelsize=fs.TICK_SIZE)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--outdir", default=str(fs.VAULT))
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for name, title, ylab, ylim, pick in (
        ("generality_alignment_kwd.png",
         "The predictor μ_eff sets the separation axis's alignment",
         "|separation axis · v_max| over open channels", (0, 1.05), "align"),
        ("generality_open_kwd.png",
         "The predictor μ_eff sets how many channels stay open",
         "open-channel fraction (of block 0's 64 channels)", (0, 1.05), "open"),
    ):
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        if pick == "align":
            ax.axhline(ISO_NULL, color="0.6", lw=1.0, ls=":", zorder=1,
                       label="isotropic random-direction null")
        for label, mu, ang, al, of, mark, col in DATA:
            edge = "#555555" if mark == "s" else "white"
            kw = dict(marker=mark, s=56, color=col, zorder=4,
                      edgecolors=edge, linewidths=0.7)
            if pick == "align":
                if al is None:
                    continue
                lo, med, hi = al
            else:
                olo, om, ohi = of
                lo, med, hi = olo / 64, om / 64, ohi / 64
            ax.errorbar(mu, med, yerr=[[med - lo], [hi - med]], color=col,
                        lw=1.3, capsize=3, zorder=3)
            ax.scatter(mu, med, label=label, **kw)
        _axis(ax, fs)
        ax.set_ylabel(ylab, fontsize=fs.LABEL_SIZE)
        ax.set_ylim(*ylim)
        ax.set_title(title, fontsize=fs.TITLE_SIZE)
        fig.legend(fontsize=8, frameon=False, loc="lower center",
                   bbox_to_anchor=(0.5, -0.14), ncol=4,
                   columnspacing=0.9, handletextpad=0.2)
        fig.tight_layout()
        p = f"{a.outdir}/{name}"
        if a.dry_run:
            print(f"  [dry-run] {p}")
        else:
            fig.savefig(p, dpi=fs.DPI, bbox_inches="tight")
            print(f"  wrote {p}")
        plt.close(fig)


if __name__ == "__main__":
    main()
