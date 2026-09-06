"""The shared-scale transfer figure, drawn to the conventions in `figstyle`.

Two curves against depth: the independent-scale gain from the depth rig and the
shared-scale gain from the transfer check. Each curve is paired against the single
of ITS OWN model class, because a difference taken across two implementations
would carry any implementation gap along with the branch effect. The annotation
reports the independent arm's baseline; the two classes agree on it to within 0.13
points at every depth, so one number describes both.
"""

from __future__ import annotations

import argparse
import collections

from structural_reparam.agents.stage1_plots import figstyle as fs

DEPTHS = [3, 5, 8]


def collect():
    import wandb
    api = wandb.Api(timeout=60)

    shared = collections.defaultdict(dict)
    for r in api.runs(fs.PROJECT, filters={"group": "kwd_transfer_c100"}):
        if r.state != "finished":
            continue
        v = r.name.split("/")[-1]
        d = int(v.split("_d")[1].split("_")[0])
        s = int(v.split("_s")[-1])
        arm = "sharedg" if v.startswith("sharedg") else "single_shared"
        shared[(d, s)][arm] = (fs.pct(r.summary.get("train_accuracy")),
                               fs.pct(r.summary.get("test_accuracy")))

    indep = collections.defaultdict(dict)
    for r in api.runs(fs.PROJECT, filters={"group": "kwd_depth_c100"}):
        if r.state != "finished":
            continue
        cfg = r.config
        a = {**(cfg.get("model") or {}).get("args", {}),
             **(cfg.get("variant") or {}).get("args", {})}
        d = sum(a["stage_blocks"])
        if d not in DEPTHS:
            continue
        indep[(d, cfg.get("seed", cfg.get("base_seed")))][
            "indep2" if a["num_3x3"] == 2 else "single"] = (
            fs.pct(r.summary.get("train_accuracy")), fs.pct(r.summary.get("test_accuracy")))
    return (fs.paired(indep, "single", "indep2"),
            fs.paired(shared, "single_shared", "sharedg"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--outdir", default=fs.VAULT)
    a = ap.parse_args()
    I, S = collect()
    xs = [d for d in DEPTHS if d in I and d in S]
    ind = "one scale per branch"
    fs.multi_pane([(ind, I), ("one scale shared per block", S)], xs,
                  "depth (total blocks)", "Shared versus independent scale, CIFAR-100",
                  "shared_gamma_transfer_c100_kwd.png", dry=a.dry_run,
                  outdir=a.outdir, annotate_from=ind)
    for d in xs:
        print(f"    d{d} independent {I[d]['train']:+.2f}/{I[d]['test']:+.2f}  "
              f"shared {S[d]['train']:+.2f}/{S[d]['test']:+.2f}  "
              f"(single {I[d]['base_train']:.1f})")


if __name__ == "__main__":
    main()
