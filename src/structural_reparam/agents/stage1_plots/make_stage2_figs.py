"""Stage 2 figures: the per-block placement ladders and the schedule contrast.

Placement ladders are one figure per cell, two panes, x is the block index in
network order. The annotation over each point is the single-branch baseline for
that pane's metric, which is constant across a ladder — it is the same single in
every arm — so it appears once per point as a reminder of what the gain is
measured against.

The schedule contrast is reported as a table in the paper, not a figure: it has two
x-positions and two series, and four points do not make a line chart.
"""

from __future__ import annotations

import argparse
import collections

from structural_reparam.agents.stage1_plots import figstyle as fs

CELLS = {
    "stage2_place_c100_d3_w1": (3, "CIFAR-100, depth 3, 32/64/128", "placement_d3_w1_kwd.png"),
    "stage2_place_c100_d5_w1": (5, "CIFAR-100, depth 5, 32/64/128", "placement_d5_w1_kwd.png"),
    "stage2_place_c100_d3_w2": (3, "CIFAR-100, depth 3, 64/128/256", "placement_d3_w2_kwd.png"),
}


def fetch(group):
    import wandb
    api = wandb.Api(timeout=60)
    c = collections.defaultdict(dict)
    for r in api.runs(fs.PROJECT, filters={"group": group}):
        if r.state != "finished":
            continue
        seed = r.name.split("/")[0].split("_")[-1]
        arm = r.name.split("/")[-1].rsplit("_s", 1)[0]
        c[seed][arm] = (fs.pct(r.summary.get("train_accuracy")),
                        fs.pct(r.summary.get("test_accuracy")))
    return c


def ladder(c, depth):
    """-> stats keyed by block index, using each cell's own single."""
    names = {0: "pair_first", depth - 1: "pair_last"}
    for b in range(1, depth - 1):
        names[b] = f"pair_b{b}"
    cells = {}
    for b, nm in names.items():
        for seed, arms in c.items():
            if nm in arms and "single" in arms:
                cells[(b, seed)] = {"single": arms["single"], "arm": arms[nm]}
    return fs.paired(cells, "single", "arm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--outdir", default=fs.VAULT)
    a = ap.parse_args()

    for group, (depth, label, fname) in CELLS.items():
        c = fetch(group)
        s = ladder(c, depth)
        xs = sorted(s)
        labels = [("first" if b == 0 else "last" if b == depth - 1 else str(b)) for b in xs]
        fs.two_pane(s, xs, "block holding the two-branch pair (network order)",
                    f"Placement ladder, {label}", fname,
                    xticklabels=labels, dry=a.dry_run, outdir=a.outdir)
        print("    " + "  ".join(
            f"b{x} {s[x]['train']:+.2f}±{s[x]['train_se']:.2f}" for x in xs))


if __name__ == "__main__":
    main()
