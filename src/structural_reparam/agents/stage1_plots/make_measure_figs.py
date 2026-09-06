"""The unwhitened measurement twins of the angle and alignment figures.

Two figures from the block-0 `pair_all` runs of the canonical depth-3 cell
(`stage2_place_c100_d3_w1`), computed on the unwhitened kernels with no whitening
anywhere in the quantity itself:

- `unwhitened_angle_d3_w1_kwd.png`: the unwhitened branch angle over training, the
  angle between the two kernels in the plain inner product. By default this is read
  at EVERY epoch (0 to 100) from the runs' training-time pair-channel probe artifact,
  which stores the per-channel Euclidean and whitened cosines each epoch; the
  whitened cosine gives the open mask. `--angle-source checkpoints` restores the
  older six-epoch version read from the checkpoint artifacts.
- `diffmode_alignment_unwhitened_kwd.png`: the alignment of the unwhitened
  separation axis, the normalized difference of the two unit kernels, with the top
  eigenvector of the input patch covariance. The original runs' probe artifacts
  predate the probe's alignment record (2026-08-29), so from the checkpoints this
  exists only at epochs 0, 5, 25, 50, 75 and 100. Since 2026-09-01 the default
  (`--align-source probe`) reads it at every epoch from the rerun group
  `stage2_place_c100_d3_w1_probe` (same arm, recipe and seeds, rerun on darxio with
  the probe's new `align_vmax_raw` record) and prints the checkpoint-based value of
  the same runs beside it at the six checkpoint epochs.

The open-channel mask is the whitened 10-degree convention shared with Figures 12
and 13, so Figures 12 to 16 describe the same channel population. For the per-epoch
angle figure the whitened cosine is the probe's own, against Sigma re-estimated each
epoch from 20 batches inside the run (the Figure 13 convention); for the checkpoint
figures it is against a fixed 50-batch estimate of the augmented stream. At the six
checkpoint epochs the two masks give the same open-channel median to 0.1 degree (the
script prints both side by side). e_top is the
top eigenvector of the augmented-stream patch covariance; its whitened and
unwhitened versions agree to 0.9995, so the alignment quantity itself uses no
whitening.

Usage: python -m structural_reparam.agents.stage1_plots.make_measure_figs
           [--dry-run] [--angle-source probe|checkpoints] [--align-source probe|checkpoints]
           [--angle-group G] [--align-group G]
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from structural_reparam.agents.stage1_plots import figstyle as fs
from structural_reparam.agents.stage1_plots.make_align_fig import patch_cov
from structural_reparam.agents.stage1_plots.make_occupation_fig import probe_npz
from structural_reparam.analysis.ckpt_loader import fetch_checkpoint

EPOCHS = [0, 5, 25, 50, 75, 100]      # checkpoint epochs (the cross-check)
TABLE_EPOCHS = [0, 1, 2, 3, 5, 10, 25, 50, 75, 100]   # rows of the paper's alignment table
SEEDS = [42, 43, 44]
GROUP = "stage2_place_c100_d3_w1"              # the original placement runs
PROBE_GROUP = "stage2_place_c100_d3_w1_probe"  # the 2026-09-01 rerun with the per-epoch probe
ARM = "pair_all"


def _variant(group, seed):
    # the original group named its variants per seed; the rerun runs one variant over three seeds
    return f"{ARM}_s{seed}" if group == GROUP else ARM
OPEN = float(torch.cos(torch.deg2rad(torch.tensor(10.0))))


def collect(group=GROUP):
    """Checkpoint-based: -> (angle per epoch, alignment per epoch) at EPOCHS."""
    S = patch_cov()
    ev, evec = torch.linalg.eigh(S)
    lam, V = ev.flip(0), evec.flip(1)
    e_top = V[:, 0]
    Shalf = V @ torch.diag(lam.clamp_min(0).sqrt()) @ V.T

    ang, ali = [], []
    for ep in EPOCHS:
        deg, al = [], []
        for seed in SEEDS:
            sd = fetch_checkpoint(group, _variant(group, seed), ep, seed=seed)
            sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
            w = [sd[k].flatten(1).double() for k in sorted(sd)
                 if k.startswith("stages.0.0.convs.") and k.endswith(".weight")]
            p1, p2 = w[0] @ Shalf, w[1] @ Shalf
            u1 = p1 / p1.norm(dim=1, keepdim=True)
            u2 = p2 / p2.norm(dim=1, keepdim=True)
            mask = (u1 * u2).sum(1) <= OPEN
            r1 = w[0] / w[0].norm(dim=1, keepdim=True)
            r2 = w[1] / w[1].norm(dim=1, keepdim=True)
            cos = (r1 * r2).sum(1).clamp(-1, 1)
            deg.append(torch.rad2deg(torch.acos(cos))[mask])
            d = r1 - r2
            d = d / d.norm(dim=1, keepdim=True).clamp_min(1e-12)
            al.append((d @ e_top).abs()[mask])
        ang.append(torch.cat(deg))
        ali.append(torch.cat(al))
    return ang, ali


def collect_probe(group, key):
    """Probe-based: -> (epochs, per-epoch values over open channels pooled over seeds, open count).

    Reads the per-epoch per-channel arrays of the pair-channel probe artifact of
    each seed (block 0). `key` is `cos` for the unwhitened angle (converted to
    degrees) or `align_vmax_raw` for the unwhitened axis alignment; the whitened
    cosine `cos_sigma`, against the probe's own per-epoch Sigma, gives the open mask.
    """
    per_seed = []
    for seed in SEEDS:
        d = np.load(probe_npz(group, _variant(group, seed), seed))
        per_seed.append((d["epoch_epoch"], d[f"epoch_block0_{key}"], d["epoch_block0_cos_sigma"]))
    epochs = [int(e) for e in per_seed[0][0]]
    for e, _, _ in per_seed[1:]:
        assert [int(x) for x in e] == epochs, "probe artifacts disagree on the epoch grid"
    vals, nopen = [], []
    for i, _ in enumerate(epochs):
        v, no = [], 0
        for _, arr, cos_sigma in per_seed:
            mask = cos_sigma[i] <= OPEN
            x = arr[i][mask]
            if key == "cos":
                x = np.degrees(np.arccos(np.clip(x, -1.0, 1.0)))
            v.append(x)
            no += int(mask.sum())
        vals.append(torch.as_tensor(np.concatenate(v), dtype=torch.float64))
        nopen.append(no)
    return epochs, vals, nopen


def one_fig(epochs, vals, ylabel, title, fname, ylim, dry, outdir, fmt="{:.3f}", annotate_at=None,
            dots=True):
    """`dots=False` draws the per-epoch line and band only: no markers and no value labels at
    `annotate_at` (the values then go in a table under the figure)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    med = [float(v.median()) for v in vals]
    p10 = [float(v.quantile(0.25)) for v in vals]
    p90 = [float(v.quantile(0.75)) for v in vals]
    annotate_at = list(epochs) if annotate_at is None else [e for e in annotate_at if e in epochs]
    dense = len(epochs) > len(annotate_at)
    fig, ax = plt.subplots(1, 1, figsize=(5.4, 3.8))
    ax.fill_between(epochs, p10, p90, alpha=0.20, color=fs.COLORS[0], zorder=2)
    ax.plot(epochs, med, marker=None if dense else "o", lw=1.7, color=fs.COLORS[0], zorder=3,
            label="trained, median over open channels")
    if dense and dots:
        ax.plot(annotate_at, [med[epochs.index(e)] for e in annotate_at], ls="none", marker="o",
                ms=4, color=fs.COLORS[0], zorder=4)
    ax.axhline(med[0], color=fs.COLORS[1], lw=1.2, ls="--", zorder=2,
               label=f"random-init null ({med[0]:.2f})")
    for e in (annotate_at if dots else []):
        m = med[epochs.index(e)]
        ax.annotate(fmt.format(m), (e, m), textcoords="offset points",
                    xytext=(0, -11), ha="center", fontsize=7, color="#444444")
    ax.set_ylim(*ylim)
    ax.set_xlabel("epoch", fontsize=fs.LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=fs.LABEL_SIZE)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    ax.grid(alpha=fs.GRID_ALPHA, zorder=0)
    ax.tick_params(labelsize=fs.TICK_SIZE)
    fig.suptitle(title, fontsize=fs.TITLE_SIZE, y=1.00)
    fig.tight_layout()
    p = f"{outdir}/{fname}"
    if dry:
        print(f"  [dry-run] {p}")
    else:
        fig.savefig(p, dpi=fs.DPI, bbox_inches="tight")
        print(f"  wrote {p}")
    plt.close(fig)
    return med, p10, p90


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--outdir", default=fs.VAULT)
    ap.add_argument("--angle-source", choices=["probe", "checkpoints"], default="probe",
                    help="unwhitened angle: per-epoch probe artifacts (default) or the six checkpoint epochs")
    ap.add_argument("--align-source", choices=["probe", "checkpoints"], default="probe",
                    help="unwhitened axis alignment: per-epoch probe artifacts of the rerun (default) or checkpoints")
    ap.add_argument("--angle-group", default=GROUP, help=f"group for the angle figure (default {GROUP})")
    ap.add_argument("--align-group", default=None,
                    help=f"group for the alignment figure (default {PROBE_GROUP} for probe, {GROUP} for checkpoints)")
    ap.add_argument("--no-ckpt-check", action="store_true",
                    help="skip the checkpoint-based cross-check (needs CIFAR-100 under data/cifar100 for Sigma)")
    a = ap.parse_args()
    align_group = a.align_group or (PROBE_GROUP if a.align_source == "probe" else GROUP)

    # checkpoint-based values: always computed, as the check printed beside the probe values
    if a.no_ckpt_check:
        assert a.angle_source == "probe" and a.align_source == "probe", "--no-ckpt-check needs the probe sources"
        ang_ck = ali_ck = None
    else:
        ang_ck, _ = collect(a.angle_group)
        ali_ck = collect(align_group)[1]
    if a.angle_source == "probe":
        ep_ang, ang, nopen_ang = collect_probe(a.angle_group, "cos")
    else:
        ep_ang, ang, nopen_ang = EPOCHS, ang_ck, [int(v.numel()) for v in ang_ck]
    if a.align_source == "probe":
        ep_ali, ali, nopen_ali = collect_probe(align_group, "align_vmax_raw")
    else:
        ep_ali, ali, nopen_ali = EPOCHS, ali_ck, [int(v.numel()) for v in ali_ck]

    m1, lo1, hi1 = one_fig(ep_ang, ang, "unwhitened branch angle (degrees)",
                           "The branch angle in the unwhitened metric",
                           "unwhitened_angle_d3_w1_kwd.png", (0, 185), a.dry_run, a.outdir,
                           fmt="{:.0f}°", annotate_at=EPOCHS, dots=False)
    m2, lo2, hi2 = one_fig(ep_ali, ali, "|unwhitened separation axis · top eigenvector|",
                           "The kernel difference points at the loudest input direction",
                           "diffmode_alignment_unwhitened_kwd.png", (0, 1.03), a.dry_run, a.outdir,
                           annotate_at=EPOCHS, dots=False)
    print(f"   unwhitened angle: {a.angle_source} of {a.angle_group} ({len(ep_ang)} epochs); "
          f"checkpoint value of the same group beside it")
    print("   markdown table for the paper (median and quartiles over open channels of three seeds):")
    print("   | epoch | open channels | median unwhitened angle | 25th percentile | 75th percentile |")
    print("   | --- | --- | --- | --- | --- |")
    for ep in EPOCHS:
        i = ep_ang.index(ep)
        print(f"   | {ep} | {nopen_ang[i]}/96 | {m1[i]:.1f}° | {lo1[i]:.1f}° | {hi1[i]:.1f}° |")
    for ep in EPOCHS:
        i = ep_ang.index(ep); ck = "skipped" if ang_ck is None else f"{float(ang_ck[EPOCHS.index(ep)].median()):5.1f}"
        print(f"   epoch {ep:>3d}: angle {m1[i]:6.1f} deg (p25 {lo1[i]:5.1f}, p75 {hi1[i]:5.1f}, "
              f"open {nopen_ang[i]}/96)   | checkpoints {ck}")
    print(f"   unwhitened alignment: {a.align_source} of {align_group} ({len(ep_ali)} epochs); "
          f"checkpoint value of the same group beside it")
    print("   markdown rows for the paper's alignment table (unwhitened half):")
    print("   | epoch | open channels | median | 25th percentile | 75th percentile |")
    for ep in TABLE_EPOCHS:
        if ep not in ep_ali:
            continue
        i = ep_ali.index(ep)
        print(f"   | {ep} | {nopen_ali[i]} of 96 | {m2[i]:.3f} | {lo2[i]:.3f} | {hi2[i]:.3f} |")
    for ep in EPOCHS:
        i = ep_ali.index(ep); ck = "skipped" if ali_ck is None else f"{float(ali_ck[EPOCHS.index(ep)].median()):.4f}"
        print(f"   epoch {ep:>3d}: alignment {m2[i]:.4f} (p25 {lo2[i]:.4f}, p75 {hi2[i]:.4f}, "
              f"open {nopen_ali[i]}/96)   | checkpoints {ck}")


if __name__ == "__main__":
    main()
