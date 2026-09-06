"""Does the branch separation axis park on the top eigendirection of the input covariance?

One pane: the alignment of each channel's separation axis with the top eigendirection of
the input covariance, over training. The covariance's own spectrum used to be a second
pane, but it establishes a background fact rather than a result -- that a dominant
direction exists at all -- so it is printed to stdout and quoted in the caption instead.

Block 0, so Sigma is the covariance of the 3x3 image patches and needs no forward pass.
Everything is in the whitened convention p = Sigma^(1/2) w, where the effective covariance
is Sigma^2 and the top eigenvector is the same vector. The separation axis is the
normalised p_hat_1 - p_hat_2, restricted to OPEN channels because a closed pair's
difference vector is near zero and its direction is numerical noise.

The null is not quoted from another source: it is the same measurement on the epoch-0
checkpoint of the same runs, so it shares the covariance, the channels and the code.

Two sources. `--source checkpoints` (the original, group `stage2_place_c100_d3_w1`)
reads the kernels at the six checkpoint epochs against a fixed 50-batch estimate of
the augmented-stream patch covariance. `--source probe` (the default since
2026-09-01, group `stage2_place_c100_d3_w1_probe`: the same arm, recipe and seeds
rerun on darxio with the current pair-channel probe) reads the probe's per-channel
whitened alignment at EVERY epoch, computed inside the run against Sigma
re-estimated that epoch from 20 batches (the Figure 13 convention). In probe mode
the script also computes the checkpoint-based value on the same runs at the six
checkpoint epochs and prints the two side by side.

Usage: python -m structural_reparam.agents.stage1_plots.make_align_fig
           [--dry-run] [--source probe|checkpoints] [--group G]
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

from structural_reparam.agents.stage1_plots import figstyle as fs
from structural_reparam.analysis.ckpt_loader import fetch_checkpoint

EPOCHS = [0, 5, 25, 50, 75, 100]      # checkpoint epochs (the cross-check)
TABLE_EPOCHS = [0, 1, 2, 3, 5, 10, 25, 50, 75, 100]   # rows of the paper's alignment table
SEEDS = [42, 43, 44]
# fetched through the shared loader rather than a hand-populated cache directory,
# so the figure regenerates after /tmp is cleared
GROUP = "stage2_place_c100_d3_w1"            # the original placement runs (checkpoints)
PROBE_GROUP = "stage2_place_c100_d3_w1_probe"  # the 2026-09-01 rerun with the per-epoch probe
ARM = "pair_all"
OPEN = float(torch.cos(torch.deg2rad(torch.tensor(10.0))))


def collect_probe(group, arm=ARM, key="epoch_block0_align_vmax"):
    """-> (epochs, alignment over open channels per epoch pooled over seeds, open count per epoch).

    Reads the pair-channel probe artifact of each seed. The open mask is the probe's
    own whitened cosine (`cos_sigma`) at the 10-degree convention.
    """
    from structural_reparam.agents.stage1_plots.make_occupation_fig import probe_npz
    per_seed = []
    for seed in SEEDS:
        d = np.load(probe_npz(group, arm, seed))
        per_seed.append((d["epoch_epoch"], d[key], d["epoch_block0_cos_sigma"]))
    epochs = [int(e) for e in per_seed[0][0]]
    for e, _, _ in per_seed[1:]:
        assert [int(x) for x in e] == epochs, "probe artifacts disagree on the epoch grid"
    vals, nopen = [], []
    for i, _ in enumerate(epochs):
        A, no = [], 0
        for _, al, cos_sigma in per_seed:
            mask = cos_sigma[i] <= OPEN
            A.append(al[i][mask]); no += int(mask.sum())
        vals.append(torch.as_tensor(np.concatenate(A), dtype=torch.float64))
        nopen.append(no)
    return epochs, vals, nopen


def patch_cov(n_batches=50):
    sys.path.insert(0, "src")
    from structural_reparam.data.cifar100 import build_loaders
    torch.manual_seed(0)
    tr, _ = build_loaders(data_dir="data/cifar100", batch_size=128, num_workers=0,
                          persistent_workers=False, batch_mode="shuffled", download=False)
    cov, n = None, 0
    with torch.no_grad():
        for i, b in enumerate(tr):
            if i >= n_batches:
                break
            p = torch.nn.functional.unfold(b[0], 3, padding=1, stride=1)
            p = p.transpose(1, 2).reshape(-1, p.shape[1])
            p = p - p.mean(0, keepdim=True)
            c = p.T @ p / p.shape[0]
            cov = c if cov is None else cov + c
            n += 1
    return (cov / n).double()


def collect_checkpoints(group, arm=ARM):
    """-> (median, p25, p75, (open, total)) per checkpoint epoch, plus the spectrum."""
    S = patch_cov()
    ev, evec = torch.linalg.eigh(S)
    lam, V = ev.flip(0), evec.flip(1)
    e_top = V[:, 0]
    Shalf = V @ torch.diag(lam.clamp_min(0).sqrt()) @ V.T

    med, p10, p90, nopen = [], [], [], []
    for ep in EPOCHS:
        A = []
        no = tot = 0
        for seed in SEEDS:
            sd = fetch_checkpoint(group, arm if group != GROUP else f"{arm}_s{seed}", ep, seed=seed)
            sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
            w = [sd[k].flatten(1).double() for k in sorted(sd)
                 if k.startswith("stages.0.0.convs.") and k.endswith(".weight")]
            p1, p2 = w[0] @ Shalf, w[1] @ Shalf
            u1 = p1 / p1.norm(dim=1, keepdim=True)
            u2 = p2 / p2.norm(dim=1, keepdim=True)
            mask = (u1 * u2).sum(1) <= OPEN
            d = u1 - u2
            d = d / d.norm(dim=1, keepdim=True).clamp_min(1e-12)
            A.append((d @ e_top).abs()[mask])
            no += int(mask.sum()); tot += u1.shape[0]
        A = torch.cat(A)
        med.append(float(A.median())); p10.append(float(A.quantile(0.25)))
        p90.append(float(A.quantile(0.75))); nopen.append((no, tot))
    return med, p10, p90, nopen, lam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--outdir", default=fs.VAULT)
    ap.add_argument("--source", choices=["probe", "checkpoints"], default="probe")
    ap.add_argument("--group", default=None,
                    help=f"W&B group; default {PROBE_GROUP} for probe, {GROUP} for checkpoints")
    ap.add_argument("--dots", action="store_true",
                    help="mark and label the checkpoint epochs on the curve (off by default since 2026-09-03; the values go in the table)")
    a = ap.parse_args()
    group = a.group or (PROBE_GROUP if a.source == "probe" else GROUP)

    ck_med, ck_p10, ck_p90, ck_nopen, lam = collect_checkpoints(group)
    if a.source == "probe":
        epochs, vals, no = collect_probe(group)
        med = [float(v.median()) for v in vals]
        p10 = [float(v.quantile(0.25)) for v in vals]
        p90 = [float(v.quantile(0.75)) for v in vals]
        nopen = [(n, 3 * 32) for n in no]
    else:
        epochs, med, p10, p90, nopen = EPOCHS, ck_med, ck_p10, ck_p90, ck_nopen
    dense = len(epochs) > len(EPOCHS)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(5.4, 3.8))
    axes = [ax, ax]
    axes[1].fill_between(epochs, p10, p90, alpha=0.20, color=fs.COLORS[0], zorder=2)
    axes[1].plot(epochs, med, marker=None if dense else "o", lw=1.7, color=fs.COLORS[0], zorder=3,
                 label="trained, median over open channels")
    if dense and a.dots:
        axes[1].plot(EPOCHS, [med[epochs.index(e)] for e in EPOCHS], ls="none", marker="o",
                     ms=4, color=fs.COLORS[0], zorder=4)
    axes[1].axhline(med[0], color=fs.COLORS[1], lw=1.2, ls="--", zorder=2,
                    label=f"random-init null ({med[0]:.2f})")
    for x in (EPOCHS if a.dots else []):
        m = med[epochs.index(x)]
        axes[1].annotate(f"{m:.3f}", (x, m), textcoords="offset points",
                         xytext=(0, -11), ha="center", fontsize=7, color="#444444")
    axes[1].set_ylim(0, 1.03)
    axes[1].set_xlabel("epoch", fontsize=fs.LABEL_SIZE)
    axes[1].set_ylabel("|whitened separation axis · top eigenvector|", fontsize=fs.LABEL_SIZE)
    axes[1].legend(fontsize=7.5, frameon=False, loc="lower right")
    axes[1].grid(alpha=fs.GRID_ALPHA, zorder=0)
    ax.tick_params(labelsize=fs.TICK_SIZE)
    fig.suptitle("The separation axis parks on the loudest input direction",
                 fontsize=fs.TITLE_SIZE, y=1.00)
    fig.tight_layout()
    p = f"{a.outdir}/diffmode_alignment_kwd.png"
    if a.dry_run:
        print(f"  [dry-run] {p}")
    else:
        fig.savefig(p, dpi=fs.DPI, bbox_inches="tight")
        print(f"  wrote {p}")
    plt.close(fig)
    print(f"   spectrum: " + ", ".join(f"{float(x):.2f}" for x in lam[:5]))
    print(f"   group {group}, source {a.source} ({len(epochs)} epochs); "
          f"checkpoint-based value of the same runs beside it")
    for ep, cm, clo, (cno, ctot) in zip(EPOCHS, ck_med, ck_p10, ck_nopen):
        i = epochs.index(ep)
        print(f"   epoch {ep:>3d}: median {med[i]:.4f}  p25 {p10[i]:.4f}  open {nopen[i][0]}/{nopen[i][1]}"
              f"   | checkpoints: median {cm:.4f}  p25 {clo:.4f}  open {cno}/{ctot}")
    print("   markdown rows for the paper's alignment table (whitened half):")
    print("   | epoch | open channels | median | 25th percentile | 75th percentile |")
    for ep in TABLE_EPOCHS:
        if ep not in epochs:
            continue
        i = epochs.index(ep)
        print(f"   | {ep} | {nopen[i][0]} of {nopen[i][1]} | {med[i]:.3f} | {p10[i]:.3f} | {p90[i]:.3f} |")


if __name__ == "__main__":
    main()
