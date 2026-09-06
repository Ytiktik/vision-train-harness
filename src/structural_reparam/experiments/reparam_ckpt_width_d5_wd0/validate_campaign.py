"""Campaign-level close-out validation for reparam_ckpt_width_d5_wd0 (exit criteria).

1. ALL 36 runs (2 groups x 18): finished, checkpoint/validated==1, artifact
   committed with 7 files (6 ckpts + manifest), all nonzero size, epochs
   {0,5,25,50,75,100}.
2. One sampled run per group: download artifact, strict-load ep100 into
   LayerwiseRepVGGCifar, CPU forward pass (reuses gate A from
   reparam_ckpt_sweeps.validate_first_runs).
3. Gain coherence (INFORMATIONAL, non-gating — these are NEW cells, no
   originals to be paired against): per width cell, indep_2 − single_base mean
   gain @ep100 (3 seeds) ± sd, printed alongside the w1.0 anchor = the wd0.0
   cell of the ckpt100_*_wd_d5 groups (which IS this sweep's width-1.0 point,
   deduped there). Gains must be finite for every cell.
"""

from __future__ import annotations

from collections import defaultdict

import wandb

from structural_reparam.experiments.reparam_ckpt_sweeps.validate_first_runs import gate_a

PROJECT = "yoovi-t-tel-aviv-university/claude-autonomous-reparam"
EXPECTED = {"ckpt100_c100_width_d5_wd0": 18, "ckpt100_c10_width_d5_wd0": 18}
ANCHOR = {  # w1.0 @d5/WD0 lives in the WD-axis groups (dedup convention)
    "ckpt100_c100_width_d5_wd0": "ckpt100_c100_wd_d5",
    "ckpt100_c10_width_d5_wd0": "ckpt100_c10_wd_d5",
}
ANCHOR_CELL = "wd0.0"
EPOCHS = {0, 5, 25, 50, 75, 100}


def cell_of(variant: str) -> tuple[str, str] | None:
    if variant.startswith("single_base_"):
        arm = "single_base"
    elif variant.startswith("indep_2_"):
        arm = "indep_2"
    else:
        return None
    return arm, variant.split("_")[-1]  # e.g. ('indep_2', 'w025') / ('indep_2', 'wd0.0')


def gains(runs, cells: set[str] | None = None) -> dict[str, dict[int, float]]:
    accs: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    for r in runs:
        ac = cell_of(r.config.get("variant", {}).get("name", ""))
        if ac is None or (cells is not None and ac[1] not in cells):
            continue
        accs[ac][r.config.get("seed")] = float(r.summary.get("test_accuracy", float("nan")))
    out: dict[str, dict[int, float]] = {}
    for c in {c for (_, c) in accs}:
        sb, i2 = accs.get(("single_base", c), {}), accs.get(("indep_2", c), {})
        out[c] = {s: i2[s] - sb[s] for s in sb if s in i2}
    return out


def mean_sd(xs: list[float]) -> tuple[float, float]:
    mu = sum(xs) / len(xs)
    sd = (sum((x - mu) ** 2 for x in xs) / max(len(xs) - 1, 1)) ** 0.5
    return mu, sd


def main() -> None:
    api = wandb.Api()
    print("== 1) full 36-run artifact audit ==")
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

    print("== 3) gain table @d5/WD0 (indep_2 − single_base, pp; w1 anchor from wd group) ==")
    nan_cells = 0
    for g, runs in all_runs.items():
        g_new = gains(runs)
        anchor_runs = list(api.runs(PROJECT, filters={"group": ANCHOR[g]}))
        g_anchor = gains(anchor_runs, cells={ANCHOR_CELL})
        for cell in sorted(g_new):
            ds = list(g_new[cell].values())
            if not ds or any(x != x for x in ds):
                print(f"  {g} {cell}: MISSING/NaN gains"); nan_cells += 1; continue
            mu, sd = mean_sd(ds)
            print(f"  {g} {cell}: gain {mu*100:+.2f} ± {sd*100:.2f} pp (n={len(ds)})")
        if g_anchor.get(ANCHOR_CELL):
            mu, sd = mean_sd(list(g_anchor[ANCHOR_CELL].values()))
            print(f"  {g} w1 (≡ {ANCHOR[g]} {ANCHOR_CELL}): gain {mu*100:+.2f} ± {sd*100:.2f} pp")

    verdict = "PASS" if (bad == 0 and gate_fail == 0 and nan_cells == 0) else "FAIL"
    print(f"\nCAMPAIGN VALIDATION VERDICT: {verdict} "
          f"(audit bad={bad}, gateA fail={gate_fail}, missing cells={nan_cells})")


if __name__ == "__main__":
    main()
