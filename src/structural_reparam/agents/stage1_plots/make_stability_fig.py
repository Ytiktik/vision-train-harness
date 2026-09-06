"""The resting angle against the paper's predictor, ten settings.

Horizontal axis: mu_eff = (lambda_max - lambda_next) / tr Sigma, the predictor of
section 3.6 (equation 64), one value per setting from the input's own block-seen
spectrum, on a linear axis. Vertical axis: the whitened branch angle of the pair's
block 0 at the final epoch, median over open channels with the 25th to 75th
percentile as a bar.

Until 2026-09-04 this figure ran on the measured lambda_minus/lambda_plus of the
trained vectors and carried the parameter-free stability boundary
cos^2(theta*/2) = lambda_plus/(lambda_minus - lambda_plus) as a curve. mu_eff is a
property of the input, not of the trained pair, so that curve is not a function of
this axis and is not drawn. The boundary comparison was to move to a table of its own,
which was never written and whose dangling references were removed on 2026-09-05, so the
only statement of it left in the paper is Appendix A.9's nine-setting pool. The
per-channel dot clouds went with the same edit (they collapsed to one vertical
strip per setting once every channel of a setting shared an x), and the unwhitened
"raw" anchor was dropped because on this axis it lands on the CIFAR-100 point.

Angles and quartiles are Tables 13 and 14's, so this figure and the left pane of
the geometry figure carry identical numbers. mu_eff values are the 50-batch seed-0
draw of scripts/analysis/quiet_band.py clock, the draw the tables use; the sharpening
dose's is the measured block-seen value of its own trained stream
(scripts/analysis/sharpen_readback.py).

The sharpening dose of Table 13, alpha = +0.30 of the same (Sigma_img + eps I)^alpha
family the ZCA doses use with alpha = -1/2, was added on 2026-09-06; it is the one
setting whose mu_eff sits above the datasets' range, and it is drawn as a square in a
purple hue so that it is not read as one of the red whitening doses.
"""
from __future__ import annotations

import argparse

from structural_reparam.agents.stage1_plots import figstyle as fs

#        label       mu_eff   angle (p25, med, p75)   marker  color
DATA = [
    ("sharpened α=0.30", 0.834, (144, 150, 161), "s", "#3f007d"),
    ("Flowers-102", 0.554, (115, 130, 145), "o", "#1f77b4"),
    ("CIFAR-100",   0.633, (133, 146, 154), "^", "#17becf"),
    ("SVHN",        0.708, (139, 143, 149), "D", "#2ca02c"),
    ("EuroSAT",     0.681, (134, 143, 154), "v", "#9467bd"),
    ("GTSRB",       0.745, (147, 153, 158), "P", "#111111"),
    ("ZCA ε=10",    0.317, (107, 125, 139), "s", "#7f0000"),
    ("ZCA ε=1",     0.112, (71, 107, 122),  "s", "#c62828"),
    ("ZCA ε=0.1",   0.004, (58, 77, 91),    "s", "#ef7070"),
    ("ZCA ε=0.01",  0.006, (36, 49, 64),    "s", "#f7b2b2"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--outdir", default=str(fs.VAULT))
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.0, 4.0))

    for label, mu, (alo, ang, ahi), mark, col in DATA:
        edge = "#555555" if mark == "s" else "white"
        ax.errorbar(mu, ang, yerr=[[ang - alo], [ahi - ang]], color=col,
                    lw=1.3, capsize=3, zorder=3)
        ax.scatter(mu, ang, marker=mark, s=56, color=col, zorder=4,
                   edgecolors=edge, linewidths=0.7, label=label)

    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8])
    ax.set_xlim(-0.04, 0.88)
    ax.set_xlabel("μ_eff = (λ_max − λ_next)/tr Σ", fontsize=fs.LABEL_SIZE)
    ax.set_ylabel("resting angle over open channels (deg)", fontsize=fs.LABEL_SIZE)
    ax.set_ylim(30, 175)
    ax.grid(alpha=fs.GRID_ALPHA, zorder=0)
    ax.tick_params(labelsize=fs.TICK_SIZE)
    ax.set_title("The resting angle against the predictor μ_eff",
                 fontsize=fs.TITLE_SIZE)
    fig.legend(fontsize=8, frameon=False, loc="lower center",
               bbox_to_anchor=(0.5, -0.14), ncol=4,
               columnspacing=0.9, handletextpad=0.2)
    fig.tight_layout()
    p = f"{a.outdir}/stability_angle_kwd.png"
    if a.dry_run:
        print(f"  [dry-run] {p}")
    else:
        fig.savefig(p, dpi=fs.DPI, bbox_inches="tight")
        print(f"  wrote {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
