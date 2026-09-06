"""All section 1 and section 2 figures, drawn to the conventions in `figstyle`.

Every figure is two panes, train gain left and test gain right, one dataset per
figure, each point annotated with the single-branch arm's absolute accuracy for
that pane's metric. Titles name only what distinguishes the rig; the recipe lives
in the paper's methods, not in eight titles.

Usage: python -m structural_reparam.agents.stage1_plots.make_kwd_figs [--dry-run]
"""

from __future__ import annotations

import argparse
import collections

from structural_reparam.agents.stage1_plots import figstyle as fs

DEFAULT_CHANNELS = [32, 64, 128]


def fetch(group):
    import wandb
    api = wandb.Api(timeout=60)
    rows = []
    for r in api.runs(fs.PROJECT, filters={"group": group}):
        if r.state != "finished":
            continue
        cfg = r.config
        base = (cfg.get("model") or {}).get("args", {}) if isinstance(cfg.get("model"), dict) else {}
        var = cfg.get("variant") or {}
        a = {**base, **((var.get("args") or {}) if isinstance(var, dict) else {})}
        t = dict(cfg.get("train") or {})
        t.update((var.get("train") or {}) if isinstance(var, dict) else {})
        blocks = a.get("stage_blocks") or []
        chans = a.get("stage_channels") or [64, 128, 256]
        eff = [max(1, int(c * float(a.get("width_mult", 1.0)))) for c in chans]
        rows.append(dict(
            arm="indep2" if a.get("num_3x3") == 2 else "single",
            depth=sum(blocks) if blocks else None,
            width=round(eff[0] / DEFAULT_CHANNELS[0], 4),
            lr=float(t["lr"]) if t.get("lr") is not None else None,
            wd=float(t["weight_decay"]) if t.get("weight_decay") is not None else None,
            seed=cfg.get("seed", cfg.get("base_seed")),
            train=fs.pct(r.summary.get("train_accuracy")),
            test=fs.pct(r.summary.get("test_accuracy")),
        ))
    return rows


def cells(rows, keyfn):
    c = collections.defaultdict(dict)
    for r in rows:
        if r["train"] is None:
            continue
        c[(keyfn(r), r["seed"])][r["arm"]] = (r["train"], r["test"])
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--outdir", default=fs.VAULT)
    args = ap.parse_args()
    dry, out = args.dry_run, args.outdir

    depth_c100 = fetch("kwd_depth_c100")
    depth_c10 = fetch("kwd_depth_c10")

    # ---- depth: one figure per dataset -------------------------------------
    for rows, ds, fname in ((depth_c100, "CIFAR-100", "overparam_depth_c100_kwd.png"),
                            (depth_c10, "CIFAR-10", "overparam_depth_c10_kwd.png")):
        s = fs.paired(cells([r for r in rows if r["width"] == 1.0], lambda r: r["depth"]),
                      "single", "indep2")
        xs = sorted(s)
        fs.two_pane(s, xs, "depth (total blocks)", f"Depth sweep, {ds}", fname,
                    dry=dry, outdir=out)
        print("    " + "  ".join(
            f"d{x} {s[x]['train']:+.2f}/{s[x]['test']:+.2f} (single {s[x]['base_train']:.1f})"
            for x in xs))

    # the depth-5 CIFAR-100 cell is the shared anchor of the other three axes
    anchor = fs.paired(cells([r for r in depth_c100 if r["depth"] == 5 and r["width"] == 1.0],
                             lambda r: "a"), "single", "indep2").get("a")

    for group, keyfn, xlabel, title, fname, akey in (
            ("kwd_width_c100", lambda r: r["width"],
             "width multiplier  (1.0 = 32/64/128 channels)",
             "Width sweep, CIFAR-100, depth 5", "width_d5_c100_kwd.png", 1.0),
            ("kwd_lr_c100", lambda r: r["lr"], "learning rate",
             "Learning-rate sweep, CIFAR-100, depth 5", "lr_d5_c100_kwd.png", 0.1),
            ("kwd_reg_c100", lambda r: r["wd"], "weight decay on kernels",
             "Weight-decay sweep, CIFAR-100, depth 5", "wd_dose_d5_c100_kwd.png", 5e-4)):
        s = fs.paired(cells(fetch(group), keyfn), "single", "indep2")
        if anchor:
            s[akey] = anchor
        xs = sorted(x for x in s if x is not None)
        fs.two_pane(s, xs, xlabel, title, fname, dry=dry, outdir=out)
        print("    " + "  ".join(
            f"{x} {s[x]['train']:+.2f}/{s[x]['test']:+.2f} (single {s[x]['base_train']:.1f})"
            for x in xs))


if __name__ == "__main__":
    main()
