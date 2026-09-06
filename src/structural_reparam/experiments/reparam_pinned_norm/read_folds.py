"""Summarize fold_block.py outputs: python -m ...read_folds --dir DIR [--tags _L_fs ...]
Per tag: the fold line (norms, sigma), equivalence (train-mode full-train loss/acc per arm),
sharpness per arm and group, and the continuation per arm: probe loss/acc and block stats at
step 0, end of epoch 1 .. N (mean over seeds), plus the first-epoch step trajectory of the probe loss."""
from __future__ import annotations
import argparse, json, glob, re
from collections import defaultdict
from pathlib import Path
import numpy as np

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", type=Path, required=True); ap.add_argument("--tags", nargs="*", default=None)
    ap.add_argument("--groups", nargs="*", default=["all", "block_tan", "stages.0.0", "stages.1.0", "stages.2.0", "gammas", "fc"])
    a = ap.parse_args()
    files = sorted(a.dir.glob("fold_block_*.jsonl"))
    by_tag = defaultdict(list)
    for f in files:
        m = re.match(r"fold_block_(stages\.\d\.\d)_s(\d+)(.*)\.jsonl", f.name)
        if not m: continue
        blk, seed, tag = m.groups()
        if a.tags and tag not in a.tags: continue
        by_tag[(blk, tag)].append((int(seed), [json.loads(l) for l in f.open() if l.strip()]))
    for (blk, tag), runs in sorted(by_tag.items()):
        print(f"\n=================== block {blk} tag {tag}  (seeds {[s for s,_ in runs]})")
        fl = [r for s, recs in runs for r in recs if r["kind"] == "fold"][0]
        print("fold:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in fl.items() if k not in ("kind", "seed", "block")})
        arms = sorted({r["arm"] for s, recs in runs for r in recs if "arm" in r}, key=lambda x: ["pair", "folded", "single", "unfolded"].index(x) if x in ("pair","folded","single","unfolded") else 9)
        eq = {arm: [r for s, recs in runs for r in recs if r["kind"] == "equivalence" and r["arm"] == arm] for arm in arms}
        if any(eq.values()):
            print("equivalence (train-mode full-train loss / acc; eval-mode test acc), mean over seeds:")
            for arm in arms:
                rs = eq[arm]
                if rs: print(f"  {arm:9s} loss {np.mean([r['train_mode_full_train_loss'] for r in rs]):.4f}  acc {100*np.mean([r['train_mode_full_train_acc'] for r in rs]):6.2f}  test {100*np.mean([r['eval_mode_test_acc'] for r in rs]):6.2f}")
        sh = {arm: [r for s, recs in runs for r in recs if r["kind"] == "sharpness" and r["arm"] == arm] for arm in arms}
        if any(sh.values()):
            print("sharpness lambda_max per group (mean over seeds):")
            print("  arm       " + " ".join(f"{g:>11s}" for g in a.groups))
            for arm in arms:
                rs = sh[arm]
                if rs: print(f"  {arm:9s} " + " ".join(f"{np.mean([r.get(g, float('nan')) for r in rs]):11.3f}" for g in a.groups))
        # continuation
        for arm in arms:
            ep = [r for s, recs in runs for r in recs if r["kind"] == "epoch" and r["arm"] == arm]
            st0 = [r for s, recs in runs for r in recs if r["kind"] == "step" and r["arm"] == arm and r["step"] == 0]
            if not ep and not st0: continue
            print(f"continuation arm {arm}: epoch | probe loss | probe acc | train acc | gamma med | output med | cos | norm med | sigma med")
            def line(e, rs):
                n = [np.mean([r["norm_med"][i] for r in rs]) for i in range(len(rs[0]["norm_med"]))]
                sg = [np.mean([r["sigma_med"][i] for r in rs]) for i in range(len(rs[0]["sigma_med"]))]
                ta = np.mean([r.get("train_accuracy", float("nan")) for r in rs])
                print(f"   {e:3d} | {np.mean([r['probe_loss'] for r in rs]):.4f} | {100*np.mean([r['probe_acc'] for r in rs]):6.2f} | {100*ta:6.2f} | {np.mean([r['gamma_med'] for r in rs]):.3f} | {np.mean([r['output_med'] for r in rs]):.3f} | {np.mean([r['cos_med'] for r in rs]):.3f} | {[round(x,3) for x in n]} | {[round(x,3) for x in sg]}")
            if st0: line(0, st0)
            for e in sorted({r["epoch"] for r in ep}):
                line(e, [r for r in ep if r["epoch"] == e])
            steps = [r for s, recs in runs for r in recs if r["kind"] == "step" and r["arm"] == arm and r["epoch"] == 1 and "probe_loss" in r]
            if steps:
                ss = sorted({r["step"] for r in steps})
                print("   first-epoch probe loss at steps " + ", ".join(f"{s}:{np.mean([r['probe_loss'] for r in steps if r['step']==s]):.3f}" for s in ss[:14]))
                print("   first-epoch branch norm (b0) at steps " + ", ".join(f"{s}:{np.mean([r['norm_med'][0] for r in steps if r['step']==s]):.3f}" for s in ss[:14]))
if __name__ == "__main__":
    main()
