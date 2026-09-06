"""Early validation for the reparam_ckpt_width_d5_wd0 campaign.

Gate A — checkpoint usability (GATING, same as reparam_ckpt_sweeps): for the
first finished run(s) in the ckpt100_*_width_d5_wd0 groups, download the
checkpoint artifact, strict-load the ep100 state_dict into
LayerwiseRepVGGCifar, evaluate on the real test set, and require the
recomputed test accuracy to match the run's logged final `test_accuracy`.

REF — WD5e-4 reference (INFORMATIONAL, non-gating): these WD=0 cells are NEW,
so there is no replication-parity gate. Instead print the same variant+seed's
final test accuracy from the WD5e-4 d5 width group (ckpt100_*_width_d5_wd5e4)
side by side — different WD, differences expected; plausible-range eyeball
only.

Usage:
  python -m structural_reparam.experiments.reparam_ckpt_width_d5_wd0.validate_first_runs \
      [--max-runs 3] [--project yoovi-t-tel-aviv-university/claude-autonomous-reparam]
"""

from __future__ import annotations

import argparse

import wandb

from structural_reparam.experiments.reparam_ckpt_sweeps.validate_first_runs import gate_a

GROUPS = ["ckpt100_c100_width_d5_wd0", "ckpt100_c10_width_d5_wd0"]
REF_MAP = {  # new WD0 group -> WD5e-4 d5 width group (reference only, NOT a parity twin)
    "ckpt100_c100_width_d5_wd0": "ckpt100_c100_width_d5_wd5e4",
    "ckpt100_c10_width_d5_wd0": "ckpt100_c10_width_d5_wd5e4",
}


def ref_line(run, api, project: str) -> str:
    group = run.config["logging"]["group"]
    variant = run.config["variant"]["name"]
    seed = run.config["seed"]
    new_final = float(run.summary.get("test_accuracy", float("nan")))
    ref = None
    for r in api.runs(project, filters={"group": REF_MAP[group]}):
        if r.config.get("variant", {}).get("name") == variant and r.config.get("seed") == seed:
            ref = float(r.summary.get("test_accuracy", float("nan")))
            break
    if ref is None:
        return f"{run.name}: no WD5e-4 reference run found (variant {variant} seed {seed})"
    return (
        f"{run.name}: final test_acc {new_final:.4f} @WD0 vs {ref:.4f} @WD5e-4 "
        f"(Δ={new_final - ref:+.4f}; different WD — eyeball only, non-gating)"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="yoovi-t-tel-aviv-university/claude-autonomous-reparam")
    ap.add_argument("--max-runs", type=int, default=3)
    args = ap.parse_args()

    api = wandb.Api()
    finished = []
    for group in GROUPS:
        for r in api.runs(args.project, filters={"group": group}):
            if r.state == "finished":
                finished.append(r)
    if not finished:
        print("NO FINISHED RUNS YET in any ckpt100_*_width_d5_wd0 group")
        return
    print(f"{len(finished)} finished run(s); validating up to {args.max_runs}\n")
    failures = 0
    for run in finished[: args.max_runs]:
        ok, msg = gate_a(run, api)
        print("GATE A:", msg)
        if not ok:
            failures += 1
        print("REF   :", ref_line(run, api, args.project), "\n")
    print("VERDICT:", "PASS" if failures == 0 else f"FAIL ({failures} gate-A failures)")


if __name__ == "__main__":
    main()
