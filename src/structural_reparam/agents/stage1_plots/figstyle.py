"""Shared figure conventions for the paper. Every plot script imports this.

The rules, and why each one is a rule:

1. **The title names only what distinguishes this rig**, plus the dataset — "Depth
   sweep, CIFAR-100". It never repeats the recipe. The recipe is stated once in the
   paper's methods and putting it in every title trains the reader to skip titles.
2. **Two panes, train on the left and test on the right**, sharing the x axis.
   Datasets get separate figures rather than separate panes, so a pane is always one
   curve of one quantity.
3. **Every point is annotated with the single-branch arm's absolute accuracy for
   that pane's metric** in small text — train accuracy over the train pane, test
   accuracy over the test pane. A gain of +2 points means something different
   against a 59 % baseline than against a 97 % one, and the figure should say which
   without a trip to the caption.
4. **A black line at zero**, so the sign of an effect is readable at a glance.
5. **Error bars are the standard error of the paired per-seed difference**, never
   the spread of the two arms measured separately.
6. Everything else — colours, fonts, grid, figure size, resolution — is fixed here
   so no two figures in the paper differ by accident.

Filenames keep the `_kwd` suffix so this set coexists with Reparameterization 4's
all-parameter-decay figures instead of overwriting them.
"""

from __future__ import annotations

import collections
import statistics as st
from pathlib import Path

VAULT = Path("/media/ytiktik/F23A44933A44572F/Users/USER/OneDrive/thesis/obs"
             "/Reparameterization Paper/figures")
PROJECT = "claude-autonomous-reparam"

FIGSIZE = (9.2, 3.8)
DPI = 160
TITLE_SIZE = 11
LABEL_SIZE = 9.5
TICK_SIZE = 9
ANNOT_SIZE = 6.5
GRID_ALPHA = 0.22
COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
MARKERS = ["o", "s", "^", "D"]


def pct(v):
    if v is None:
        return None
    v = float(v)
    return v * 100.0 if v <= 1.0 else v


def mean_se(xs):
    n = len(xs)
    if not n:
        return float("nan"), float("nan"), 0
    return sum(xs) / n, (st.stdev(xs) / n ** 0.5 if n > 1 else 0.0), n


def paired(cells, base_arm, arm):
    """cells: {(cell_key, seed): {arm_name: (train, test)}} -> per-cell statistics.

    Returns {cell: dict(train, train_se, test, test_se, n, base_train, base_test)}.
    A seed missing either arm contributes nothing, so a half-finished cell reports
    its true seed count rather than mixing arms.
    """
    diffs = collections.defaultdict(list)
    bases = collections.defaultdict(list)
    for (cell, _seed), arms in cells.items():
        if base_arm in arms and arm in arms:
            b, a = arms[base_arm], arms[arm]
            diffs[cell].append((a[0] - b[0], a[1] - b[1]))
            bases[cell].append(b)
    out = {}
    for cell, vals in diffs.items():
        tr, tr_se, n = mean_se([v[0] for v in vals])
        te, te_se, _ = mean_se([v[1] for v in vals])
        bt, _, _ = mean_se([b[0] for b in bases[cell]])
        bte, _, _ = mean_se([b[1] for b in bases[cell]])
        out[cell] = dict(train=tr, train_se=tr_se, test=te, test_se=te_se,
                         n=n, base_train=bt, base_test=bte)
    return out


def two_pane(stats, xs, xlabel, title, fname, xticklabels=None, dry=False,
             outdir=VAULT, annotate=True):
    """The standard figure: train gain left, test gain right, one curve each."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
    labels = xticklabels or [str(x) for x in xs]
    for ax, field, base_key, pane in (
            (axes[0], "train", "base_train", "train"),
            (axes[1], "test", "base_test", "test")):
        ys = [stats[x][field] for x in xs]
        es = [stats[x][field + "_se"] for x in xs]
        ax.axhline(0.0, color="black", lw=1.0, zorder=1)
        ax.errorbar(range(len(xs)), ys, yerr=es, marker=MARKERS[0], capsize=3,
                    lw=1.6, color=COLORS[0], zorder=3)
        if annotate:
            # anchor the label above the TOP OF THE ERROR BAR, not the point, or a
            # long whisker prints straight through the text
            for i, x in enumerate(xs):
                ax.annotate(f"{stats[x][base_key]:.1f}",
                            (i, ys[i] + es[i]), textcoords="offset points",
                            xytext=(0, 6), ha="center", fontsize=ANNOT_SIZE,
                            color="#444444")
            ax.margins(y=0.26)
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels(labels, fontsize=TICK_SIZE)
        ax.tick_params(labelsize=TICK_SIZE)
        ax.set_xlabel(xlabel, fontsize=LABEL_SIZE)
        ax.set_ylabel(f"two-branch {pane}-accuracy gain (pp)", fontsize=LABEL_SIZE)
        ax.grid(alpha=GRID_ALPHA, zorder=0)
    # one title, placed above the axes so it cannot collide with anything
    fig.suptitle(title, fontsize=TITLE_SIZE, y=1.02)
    fig.tight_layout()
    p = Path(outdir) / fname
    if dry:
        print(f"  [dry-run] {p}")
    else:
        Path(outdir).mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=DPI, bbox_inches="tight")
        print(f"  wrote {p}")
    plt.close(fig)


def multi_pane(series, xs, xlabel, title, fname, xticklabels=None, dry=False,
               outdir=VAULT, annotate_from=None):
    """Same two panes, but several labelled curves — for the transfer comparison.

    ``series`` is a list of (label, stats). ``annotate_from`` names which series
    supplies the single's absolute accuracy annotation, since with several curves
    only one baseline can sensibly be printed.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
    labels = xticklabels or [str(x) for x in xs]
    for ax, field, base_key, pane in (
            (axes[0], "train", "base_train", "train"),
            (axes[1], "test", "base_test", "test")):
        ax.axhline(0.0, color="black", lw=1.0, zorder=1)
        allys = []
        for i, (lab, stats) in enumerate(series):
            ys = [stats[x][field] for x in xs]
            allys += ys
            ax.errorbar(range(len(xs)), ys, yerr=[stats[x][field + "_se"] for x in xs],
                        marker=MARKERS[i % len(MARKERS)], capsize=3, lw=1.6,
                        color=COLORS[i % len(COLORS)], label=lab, zorder=3)
        if annotate_from is not None:
            stats = dict(series)[annotate_from]
            top = [max(st_[x][field] + st_[x][field + "_se"] for _l, st_ in series)
                   for x in xs]
            for i, x in enumerate(xs):
                ax.annotate(f"{stats[x][base_key]:.1f}", (i, top[i]),
                            textcoords="offset points", xytext=(0, 6),
                            ha="center", fontsize=ANNOT_SIZE, color="#444444")
            ax.margins(y=0.28)
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels(labels, fontsize=TICK_SIZE)
        ax.tick_params(labelsize=TICK_SIZE)
        ax.set_xlabel(xlabel, fontsize=LABEL_SIZE)
        ax.set_ylabel(f"two-branch {pane}-accuracy gain (pp)", fontsize=LABEL_SIZE)
        ax.grid(alpha=GRID_ALPHA, zorder=0)
    axes[0].legend(fontsize=7.5, frameon=False)
    fig.suptitle(title, fontsize=TITLE_SIZE, y=1.02)
    fig.tight_layout()
    p = Path(outdir) / fname
    if dry:
        print(f"  [dry-run] {p}")
    else:
        Path(outdir).mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=DPI, bbox_inches="tight")
        print(f"  wrote {p}")
    plt.close(fig)
