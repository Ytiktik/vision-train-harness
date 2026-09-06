"""The classifier-cap table: does the last block's scale substitute for a capped fc?

Two single-branch arms differing only in whether ``fc.weight`` is weight-decayed.
If the loss wants a larger classifier throughout and decay refuses it, the block
nearest the output should be supplying the shortfall through its scale -- so freeing
the classifier should pull the LAST block's gamma down while leaving the first
block's alone.

Reported per epoch: the Frobenius norm of the classifier, and the median over
channels of the shared gamma at the first and last block. Aggregated as mean +/-
standard error over seeds. A final row gives the first-block shared gamma of the
pair arm (``--pair-arm``, default ``pair_all``, every block paired), the quantity
section 3.4 needs when it reads the late slowing of the branch angle as the
closing term fading with the growth rate of the scale.

The block naming is ``stages.{stage}.{block}.gamma``; first and last are inferred
from the state dict rather than hard-coded, so this works at any depth.

Usage:
  python -m structural_reparam.agents.stage1_plots.make_fccap_table \
      [--group stage2_place_c100_d3_w1] [--epochs 5,25,50,100]
"""

from __future__ import annotations

import argparse
import collections
import re
import statistics as st

import torch

from structural_reparam.analysis.ckpt_loader import fetch_checkpoint

ARMS = {"single": "single", "single_nofcwd": "decay removed from `fc.weight`"}


def _blocks(sd):
    """-> (first_key, last_key) for the gamma of the first and last block."""
    keys = [k for k in sd if re.fullmatch(r"stages\.\d+\.\d+\.gamma", k)]
    order = sorted(keys, key=lambda k: tuple(int(x) for x in k.split(".")[1:3]))
    return order[0], order[-1]


def read(group, seeds, epochs, pair_arm=None):
    """-> {(arm, quantity, epoch): [value per seed]}."""
    out = collections.defaultdict(list)
    arms = list(ARMS) + ([pair_arm] if pair_arm else [])
    for arm in arms:
        for seed in seeds:
            for ep in epochs:
                sd = fetch_checkpoint(group, f"{arm}_s{seed}", ep, seed=seed)
                sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
                first, last = _blocks(sd)
                out[(arm, "fc", ep)].append(float(sd["fc.weight"].double().norm()))
                out[(arm, "last", ep)].append(float(sd[last].double().median()))
                out[(arm, "first", ep)].append(float(sd[first].double().median()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="stage2_place_c100_d3_w1")
    ap.add_argument("--label", default="depth 3, 32/64/128")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--epochs", default="5,25,50,100")
    ap.add_argument("--pair-arm", default="pair_all",
                    help="pair arm for the first-block shared-gamma row; '' skips it")
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]
    epochs = [int(x) for x in a.epochs.split(",")]

    v = read(a.group, seeds, epochs, pair_arm=a.pair_arm or None)

    def cell(arm, q, ep, dp):
        xs = v[(arm, q, ep)]
        m = sum(xs) / len(xs)
        se = st.stdev(xs) / len(xs) ** 0.5 if len(xs) > 1 else 0.0
        return f"{m:.{dp}f} ± {se:.{dp}f}"

    rows = [("‖W_fc‖", "fc", 2), ("last-block γ", "last", 3), ("first-block γ", "first", 3)]
    print(f"| {a.label} | " + " | ".join(f"epoch {e}" for e in epochs) + " |")
    print("| --- |" + " --- |" * len(epochs))
    for name, q, dp in rows:
        for arm, arm_label in ARMS.items():
            lbl = f"{name}, {arm_label}"
            print(f"| {lbl} | " + " | ".join(cell(arm, q, e, dp) for e in epochs) + " |")
    if a.pair_arm:
        lbl = "first-block shared γ, pair (every block paired)"
        print(f"| {lbl} | " + " | ".join(cell(a.pair_arm, "first", e, 3) for e in epochs) + " |")


if __name__ == "__main__":
    main()
