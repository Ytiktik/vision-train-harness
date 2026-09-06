"""Campaign-level close-out validation for reparam_ckpt_sweeps (exit criteria).

1. ALL 84 runs: finished, checkpoint/validated==1, artifact committed with
   7 files (6 ckpts + manifest), all nonzero size, epochs {0,5,25,50,75,100}.
2. One sampled run per group: download artifact, strict-load ep100 into
   LayerwiseRepVGGCifar, CPU forward pass (reuses gate A from
   validate_first_runs).
3. Gain parity: per cell, indep_2 − single_base mean gain @ep100 (3 seeds)
   vs the same cell in the original sweeps100 group; report Δgain vs the
   paired-seed SE of the original.
"""

from __future__ import annotations

import math
from collections import defaultdict

import wandb

from structural_reparam.experiments.reparam_ckpt_sweeps.validate_first_runs import GROUP_MAP, gate_a

PROJECT = "yoovi-t-tel-aviv-university/claude-autonomous-reparam"
EXPECTED = {"ckpt100_c100_depth_wd5e4": 24, "ckpt100_c10_depth_wd5e4": 24,
            "ckpt100_c10_width_wd5e4": 18, "ckpt100_c100_width_wd5e4": 18}
EPOCHS = {0, 5, 25, 50, 75, 100}


def cell_of(variant: str) -> tuple[str, str] | None:
    # Original sweeps100 groups also carry single_meanonly_* — must be excluded,
    # not lumped into indep_2.
    if variant.startswith("single_base_"):
        arm = "single_base"
    elif variant.startswith("indep_2_"):
        arm = "indep_2"
    else:
        return None
    return arm, variant.split("_")[-1]  # e.g. ('indep_2', 'd3') / ('single_base','w025')


def gains(runs) -> dict[str, dict[int, float]]:
    accs: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    for r in runs:
        v = r.config.get("variant", {}).get("name", "")
        ac = cell_of(v)
        if ac is None:
            continue
        accs[ac][r.config.get("seed")] = float(r.summary.get("test_accuracy", float("nan")))
    out: dict[str, dict[int, float]] = {}
    cells = {c for (_, c) in accs}
    for c in cells:
        sb, i2 = accs.get(("single_base", c), {}), accs.get(("indep_2", c), {})
        out[c] = {s: i2[s] - sb[s] for s in sb if s in i2}
    return out


def main() -> None:
    api = wandb.Api()
    print("== 1) full 84-run artifact audit ==")
    bad = 0
    all_runs: dict[str, list] = {}
    for g, n in EXPECTED.items():
        runs = list(api.runs(PROJECT, filters={"group": g}))
        all_runs[g] = runs
        if len(runs) != n:
            print(f"  {g}: RUN COUNT {len(runs)} != {n}"); bad += 1
        for r in runs:
            probs = []
            if r.state != "finished":
                probs.append(f"state={r.state}")
            if r.summary.get("checkpoint/validated") != 1:
                probs.append("validated!=1")
            arts = [a for a in r.logged_artifacts() if a.type == "checkpoint"]
            if len(arts) != 1:
                probs.append(f"n_artifacts={len(arts)}")
            else:
                entries = arts[0].manifest.entries
                names = set(entries)
                eps = {int(x.split("ep")[1].split(".")[0]) for x in names if x.startswith("ckpt_ep")}
                if len(entries) != 7 or "manifest.json" not in names:
                    probs.append(f"files={sorted(names)}")
                if eps != EPOCHS:
                    probs.append(f"epochs={sorted(eps)}")
                if any(not (e.size or 0) > 0 for e in entries.values()):
                    probs.append("zero-size entry")
            if probs:
                print(f"  BAD {r.name}: {', '.join(probs)}"); bad += 1
    print(f"  audit: {'ALL CLEAN' if bad == 0 else f'{bad} problems'} over {sum(len(v) for v in all_runs.values())} runs")

    print("== 2) per-group sample download+load+forward (gate A) ==")
    gate_fail = 0
    for g, runs in all_runs.items():
        ok, msg = gate_a(runs[0], api)
        print(f"  {msg}")
        if not ok:
            gate_fail += 1

    print("== 3) gain parity vs sweeps100 (per cell, mean gain pp) ==")
    parity_flags = 0
    for g, runs in all_runs.items():
        orig = list(api.runs(PROJECT, filters={"group": GROUP_MAP[g]}))
        g_new, g_old = gains(runs), gains(orig)
        for cell in sorted(g_new):
            dn = list(g_new[cell].values())
            do = [g_old[cell][s] for s in g_new[cell] if cell in g_old and s in g_old[cell]]
            if not dn or not do:
                print(f"  {g} {cell}: no twin cell in original — skipped"); continue
            mn, mo = sum(dn) / len(dn), sum(do) / len(do)
            # paired-seed spread of the original gain as the noise scale
            so = (sum((x - mo) ** 2 for x in do) / max(len(do) - 1, 1)) ** 0.5
            z = abs(mn - mo) / so if so > 0 else float("inf")
            flag = "" if (z < 3 or abs(mn - mo) < 0.005) else "  <-- OUTSIDE"
            if flag:
                parity_flags += 1
            print(f"  {g} {cell}: gain new {mn*100:+.2f} vs orig {mo*100:+.2f} pp "
                  f"(orig sd {so*100:.2f}, z={z:.1f}){flag}")

    verdict = "PASS" if (bad == 0 and gate_fail == 0 and parity_flags == 0) else "FAIL"
    print(f"\nCAMPAIGN VALIDATION VERDICT: {verdict} "
          f"(audit bad={bad}, gateA fail={gate_fail}, parity flags={parity_flags})")


if __name__ == "__main__":
    main()
