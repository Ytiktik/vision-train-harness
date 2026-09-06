"""How much of each block's gain survives losing the cosine tail.

Not a figure of the paper. It carried a figure number and a provenance row from
2026-09-01 until 2026-09-05 without ever being embedded, and both were removed then;
the numbers it draws are the constant-rate columns of Tables 4 and 5.

The first version was grouped bars over every block position at two depths. It failed
because the two panes had different y-scales and different cells, the middle blocks are
noise that crowded out the comparison, and bars show two levels rather than the CHANGE
between them, which is the entire claim.

This plots the change directly: the fraction of a block's cosine-schedule gain that
remains at constant learning rate. That is a ratio taken inside one cell, so the two
depths become comparable even though their cells are not, and every bar is annotated
with the two raw gains behind it so the normalisation hides nothing.

Middle blocks are omitted. At depth 5 they are +0.53, +0.15, +0.01 under cosine, so
their retention ratio is a small number divided by a smaller one and carries no
information.
"""

from __future__ import annotations

import argparse
import collections

from structural_reparam.agents.stage1_plots import figstyle as fs

CELLS = [
    # Both schedules of the depth-3 cell come from stage2 groups with the paper's pair
    # (shared gamma, per-branch beta). The constant-rate points used to be read from
    # the wdg mechanism cell, whose pairs keep one trainable beta; at the last block
    # that convention drops the bias channel (+0.80 on its own, Table 9) and gave the
    # +0.21 that the 2026-08-31 2x2 exposed as an arm mismatch. Repointed 2026-09-01.
    dict(label="depth 3\n64/128/256", cos=("stage2_place_c100_d3_w2", None),
         const=("stage2_constlr_c100_d3_w2", None)),
    dict(label="depth 5\n32/64/128", cos=("stage2_place_c100_d5_w1", None),
         const=("stage2_constlr_c100_d5_w1", None)),
]


def gains(group, remap):
    import wandb
    api = wandb.Api(timeout=60)
    c = collections.defaultdict(dict)
    for r in api.runs(fs.PROJECT, filters={"group": group}):
        if r.state != "finished":
            continue
        v = r.name.split("/")[-1]
        seed = r.name.split("/")[0].split("_")[-1]
        arm = remap.get(v) if remap else v.rsplit("_s", 1)[0]
        if arm is None:
            continue
        c[seed][arm] = fs.pct(r.summary.get("train_accuracy"))
    out = {}
    for a in ("pair_first", "pair_last"):
        xs = [v[a] - v["single"] for v in c.values() if a in v and "single" in v]
        if xs:
            out[a] = fs.mean_se(xs)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--outdir", default=fs.VAULT)
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rows = []
    for cell in CELLS:
        G, K = gains(*cell["cos"]), gains(*cell["const"])
        for arm, nice in (("pair_first", "first block"), ("pair_last", "last block")):
            if arm not in G or arm not in K:
                continue
            g, gse, _ = G[arm]
            k, kse, _ = K[arm]
            r = k / g
            # delta method on the ratio
            rse = abs(r) * ((kse / k) ** 2 + (gse / g) ** 2) ** 0.5 if k else gse / abs(g)
            rows.append((f"{nice}\n{cell['label']}", r * 100, rse * 100, g, k, arm))

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    x = np.arange(len(rows))
    cols = [fs.COLORS[0] if r[5] == "pair_first" else fs.COLORS[1] for r in rows]
    ax.bar(x, [r[1] for r in rows], 0.6, yerr=[r[2] for r in rows], capsize=4,
           color=cols, zorder=3)
    ax.axhline(0, color="black", lw=1.0, zorder=4)
    ax.axhline(100, color="grey", lw=0.9, ls=":", zorder=2)
    ax.text(len(rows) - 0.45, 103, "gain fully retained", fontsize=7,
            color="grey", ha="right")
    for i, (_lab, pctv, _se, g, k, _a) in enumerate(rows):
        ax.annotate(f"{g:+.2f} → {k:+.2f}", (i, max(pctv, 0)),
                    textcoords="offset points", xytext=(0, 14), ha="center",
                    fontsize=7, color="#444444")
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], fontsize=fs.TICK_SIZE)
    ax.set_ylabel("percent of the cosine-schedule gain\nretained at constant learning rate",
                  fontsize=fs.LABEL_SIZE)
    ax.tick_params(labelsize=fs.TICK_SIZE)
    ax.grid(alpha=fs.GRID_ALPHA, axis="y", zorder=0)
    ax.margins(y=0.22)
    fig.suptitle("What survives removing the cosine tail", fontsize=fs.TITLE_SIZE, y=1.0)
    fig.tight_layout()
    p = f"{a.outdir}/schedule_placement_kwd.png"
    if a.dry_run:
        print(f"  [dry-run] {p}")
    else:
        fig.savefig(p, dpi=fs.DPI, bbox_inches="tight")
        print(f"  wrote {p}")
    plt.close(fig)
    for lab, pctv, se, g, k, _ in rows:
        print(f"   {lab.replace(chr(10),' '):28s} {g:+.2f} -> {k:+.2f}  = {pctv:6.1f}% ± {se:.1f}")


if __name__ == "__main__":
    main()
