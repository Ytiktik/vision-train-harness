"""Per-mechanism scaling: the first-block pair and the last-block pair, each alone, against
depth and against width, next to the whole-network gain of Table 2.

Two figures to the ``figstyle`` conventions (train gain left, test gain right):

* ``ladder_depth_c100_kwd.png``: horizontal axis depth at width 1.0 (32/64/128), cosine.
* ``ladder_width_c100_kwd.png``: horizontal axis width multiplier at depth 5.

Curves: the first-block pair alone (``pair_first``) and the last-block pair alone
(``pair_last``), each the paired per-seed difference from the placed-class ``single`` of the
same group and seed; their sum, seed by seed; and, in grey, Table 2's whole-network gain
(``indep2`` minus ``single`` of ``kwd_depth_c100`` / ``kwd_width_c100``, a different model
class with one scale per branch, drawn for reference only) and, in black, the shared-γ
all-paired network of Table 3 (``sharedg`` minus ``single_shared`` of ``kwd_transfer_c100``,
six seeds, depths 3, 5 and 8), the arm the two end terms should add up to. Error bars are the standard
error of the paired difference over seeds.

Run names come in two forms: the older stage2 groups name variants ``arm_s42`` with one seed
per config, the 2026-09-03 ladder groups name them ``arm`` with ``/seed_42`` appended by the
trainer (``num_seeds: 3``); the per-seed shards of the slow cells (depth 15, width 4) use
the older form. ``_runs`` handles both.

Usage: python -m structural_reparam.agents.stage1_plots.make_ladder_figs [--dry-run] [--print-only]
"""

from __future__ import annotations

import argparse
import math
import re
import statistics as st

from structural_reparam.agents.stage1_plots import figstyle as fs

DEPTH_CELLS = [(d, f"stage2_place_c100_d{d}_w1") for d in (3, 4, 5, 6, 8, 10, 12, 15)]
WIDTH_CELLS = [(0.25, "stage2_place_c100_d5_w025"), (0.5, "stage2_place_c100_d5_w05"),
               (1.0, "stage2_place_c100_d5_w1"), (2.0, "stage2_place_c100_d5_w2"), (4.0, "stage2_place_c100_d5_w4")]
REF_DEPTH = ("kwd_depth_c100", "single_d{x}", "indep2_d{x}")
REF_WIDTH = ("kwd_width_c100", "single_w{x}", "indep2_w{x}")
WIDTH_TAG = {0.25: "025", 0.5: "05", 1.0: "1", 2.0: "2", 4.0: "4"}


def _finite(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _runs(api, group):
    """{(arm, seed): run} over finished runs, newest per key, both naming forms."""
    out = {}
    for r in api.runs(fs.PROJECT, filters={"group": group}):
        if r.state != "finished":
            continue
        parts = r.name.split("/")
        seed = None
        if len(parts) >= 3 and parts[-1].startswith("seed_"):
            seed, arm = int(parts[-1][5:]), parts[-2]
        else:
            m = re.match(r"(.+)_s(\d+)$", parts[-1])
            if m:
                arm, seed = m.group(1), int(m.group(2))
            else:
                continue
        key = (arm, seed)
        if key not in out or str(r.created_at) > str(out[key].created_at):
            out[key] = r
    return out


def _paired(runs, single, arm, seeds=(42, 43, 44)):
    tr, te = [], []
    for s in seeds:
        a, b = runs.get((arm, s)), runs.get((single, s))
        if a is None or b is None:
            continue
        ta, tb = _finite(a.summary.get("train_accuracy")), _finite(b.summary.get("train_accuracy"))
        ea, eb = _finite(a.summary.get("test_accuracy")), _finite(b.summary.get("test_accuracy"))
        if ta is None or tb is None:
            continue
        tr.append(fs.pct(ta) - fs.pct(tb))
        te.append((fs.pct(ea) - fs.pct(eb)) if ea is not None and eb is not None else float("nan"))
    return tr, te


def cell_stats(api, group, ref, x):
    runs = _runs(api, group)
    out = {}
    # every seed the group holds for all three arms (three in most cells, six where the cell was extended)
    seeds = tuple(sorted(sd for (arm, sd) in runs if arm == "single" and ("pair_first", sd) in runs and ("pair_last", sd) in runs))
    for arm in ("pair_first", "pair_last"):
        out[arm] = _paired(runs, "single", arm, seeds=seeds)
    out["seeds"] = seeds
    n = min(len(out["pair_first"][0]), len(out["pair_last"][0]))
    out["sum"] = ([a + b for a, b in zip(out["pair_first"][0][:n], out["pair_last"][0][:n])],
                  [a + b for a, b in zip(out["pair_first"][1][:n], out["pair_last"][1][:n])])
    base = [fs.pct(_finite(runs[("single", sd)].summary.get("train_accuracy"))) for sd in seeds]
    out["base_train"] = st.mean(base) if base else float("nan")
    rg, rs, ra = ref
    if isinstance(x, float) and x == 1.0:   # Table 2 measures width 1.0 once, as depth 5 of the depth rig
        rg, rs, ra = REF_DEPTH
        tag = "5"
    else:
        tag = WIDTH_TAG[x] if isinstance(x, float) else str(x)
    rruns = _runs(api, rg)
    out["whole"] = _paired(rruns, rs.format(x=tag), ra.format(x=tag), seeds=(42, 43, 44, 45, 46, 47))
    # the shared-gamma all-paired network of Table 3 (kwd_transfer_c100, six seeds), the arm the
    # two end terms should add up to; exists at depths 3, 5 and 8 at width 1.0 only
    d = x if not isinstance(x, float) else (5 if x == 1.0 else None)
    if d in (3, 5, 8):
        truns = _runs(api, "kwd_transfer_c100")
        out["sharedg"] = _paired(truns, f"single_shared_d{d}", f"sharedg_d{d}", seeds=(42, 43, 44, 45, 46, 47))
    elif any(arm == "pair_all" for (arm, _s) in runs):
        # the cell's own all-paired arm against its own single (depth 4, added 2026-09-03)
        out["sharedg"] = _paired(runs, "single", "pair_all", seeds=seeds)
    else:
        out["sharedg"] = ([], [])
    return out


def _ms(vals):
    v = [x for x in vals if not (x is None or math.isnan(x))]
    if not v:
        return float("nan"), float("nan"), 0
    return st.mean(v), (st.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else 0.0), len(v)


def figure(cells, stats, xlabel, title, fname, dry, outdir, log_x=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=fs.FIGSIZE)
    series = [("first block alone", "pair_first", fs.COLORS[0], "-"), ("last block alone", "pair_last", fs.COLORS[1], "-"),
              ("first + last", "sum", fs.COLORS[2], "--"), ("every block paired, shared γ (Table 3)", "sharedg", "black", ":"),
              ("every block paired, one γ per branch (Table 2)", "whole", "#888888", ":")]
    for ax, idx, pane in ((axes[0], 0, "train"), (axes[1], 1, "test")):
        ax.axhline(0.0, color="black", lw=1.0, zorder=1)
        for label, key, col, ls in series:
            xs, ys, es = [], [], []
            for x, _g in cells:
                m, se, n = _ms(stats[x][key][idx])
                if n:
                    xs.append(x); ys.append(m); es.append(se)
            if xs:
                ax.errorbar(xs, ys, yerr=es, marker="o", ms=3.5, capsize=2.5, lw=1.5, ls=ls, color=col, label=label, zorder=3)
        if pane == "train":
            for x, _g in cells:
                b = stats[x]["base_train"]
                m, se, n = _ms(stats[x]["pair_last"][0])
                if n and not math.isnan(b):
                    ax.annotate(f"{b:.1f}", (x, m + se), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=fs.ANNOT_SIZE, color="#444444")
        if log_x:
            ax.set_xscale("log", base=2)
            ax.set_xticks([x for x, _ in cells]); ax.set_xticklabels([str(x) for x, _ in cells])
        else:
            ax.set_xticks([x for x, _ in cells])
        ax.set_xlabel(xlabel, fontsize=fs.LABEL_SIZE)
        ax.set_ylabel(f"{pane}-accuracy gain over the single (pp)", fontsize=fs.LABEL_SIZE)
        ax.tick_params(labelsize=fs.TICK_SIZE)
        ax.grid(alpha=fs.GRID_ALPHA, zorder=0)
        ax.margins(y=0.2)
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle(title, fontsize=fs.TITLE_SIZE, y=1.02)
    fig.tight_layout()
    from structural_reparam.agents.stage1_plots.make_speedup_figs import _save
    _save(fig, fname, dry, outdir)


def report(cells, stats, label):
    print(f"\n== {label}: paired per-seed train gain over the placed-class single, percentage points, mean ± s.e. (n); test in brackets")
    for x, g in cells:
        s = stats[x]
        row = []
        for key in ("pair_first", "pair_last", "sum", "sharedg", "whole"):
            m, se, n = _ms(s[key][0]); mt, _, _ = _ms(s[key][1])
            row.append(f"{key} {m:+.2f} ± {se:.2f} (n={n}) [{mt:+.2f}]" if n else f"{key} —")
        print(f"  {x!s:5s} single {s['base_train']:.1f} % (seeds {len(s['seeds'])})  " + "  ".join(row))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--print-only", action="store_true")
    ap.add_argument("--outdir", default=fs.VAULT)
    a = ap.parse_args()
    import wandb
    api = wandb.Api(timeout=60)
    ds = {d: cell_stats(api, g, REF_DEPTH, d) for d, g in DEPTH_CELLS}
    ws = {w: cell_stats(api, g, REF_WIDTH, w) for w, g in WIDTH_CELLS}
    report(DEPTH_CELLS, ds, "depth axis, width 1.0")
    report(WIDTH_CELLS, ws, "width axis, depth 5")
    if not a.print_only:
        figure(DEPTH_CELLS, ds, "depth (conv blocks)", "First- and last-block terms by depth, CIFAR-100", "ladder_depth_c100_kwd.png", a.dry_run, a.outdir)
        figure(WIDTH_CELLS, ws, "width multiplier (1.0 = 32/64/128)", "First- and last-block terms by width, CIFAR-100 depth 5", "ladder_width_c100_kwd.png", a.dry_run, a.outdir, log_x=True)


if __name__ == "__main__":
    main()
