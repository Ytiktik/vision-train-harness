"""Report the stage 0 headroom scout and apply the campaign's decision rule.

Reads the W&B group ``kwd_scout`` and prints, for every cell of the depth by
width grid, the final train and test accuracy of the single-branch arm, then
applies the directive's rule:

    choose the largest width whose CIFAR-100 depth-3 train accuracy is at or
    below 75 percent and whose CIFAR-100 depth-8 train accuracy is not above
    97 percent, breaking ties in favour of width 1.0.

Every run must carry ``decay_group/gate_passed == 1``. A run missing that key was
built with an optimizer other than the campaign's checked kernel-only builder and
is reported as invalid rather than silently included.

Usage:  python -m structural_reparam.agents.stage1_scout.report_scout
"""

from __future__ import annotations

PROJECT = "claude-autonomous-reparam"
GROUP = "kwd_scout"
# The default cell, fixed by the user on 2026-08-22 and named width 1.0 here.
DEFAULT_STAGE_CHANNELS = [32, 64, 128]
SATURATION_CEILING = 97.0   # percent train accuracy above which a depth is saturated
KNEE_GAIN = 1.0             # a depth step worth less than this in train accuracy
                            # counts as past the saturation knee


def _pct(v):
    """Accuracies are logged as a fraction in some runs and percent in others."""
    if v is None:
        return None
    v = float(v)
    return v * 100.0 if v <= 1.0 else v


def collect():
    import wandb

    api = wandb.Api(timeout=60)
    rows = []
    for r in api.runs(PROJECT, filters={"group": GROUP}):
        cfg = r.config
        base = (cfg.get("model") or {}).get("args", {}) if isinstance(cfg.get("model"), dict) else {}
        # The trainer logs the base model args and the chosen variant separately.
        # The variant's args override the base ones, and it is the merged pair
        # that actually built the model, so depth and width must be read from the
        # merge. Reading the base alone reports every cell as the base depth.
        variant = cfg.get("variant") or {}
        margs = dict(base)
        margs.update((variant.get("args") or {}) if isinstance(variant, dict) else {})
        blocks = margs.get("stage_blocks")
        raw_width = margs.get("width_mult")
        # The user reset the width convention on 2026-08-22: the default cell is
        # 32, 64, 128 channels and is now called width 1.0. The scout ran under
        # the old base of 64, 128, 256, so labels are derived from the channels
        # the run actually built, not from the width_mult it recorded.
        chans = margs.get("stage_channels") or [64, 128, 256]
        eff = [max(1, int(c * float(raw_width))) for c in chans] if raw_width else None
        width = round(eff[0] / DEFAULT_STAGE_CHANNELS[0], 4) if eff else None
        ds = (cfg.get("dataset") or {}).get("name") if isinstance(cfg.get("dataset"), dict) else None

        # Two independent proofs that decay reached the kernels only. The
        # optimizer target in the config is the primary one: the checked builder
        # raises at construction if any parameter with fewer than two dimensions
        # sits in a decaying group, so a run that exists at all with that target
        # passed the assertion. The summary key is the secondary, explicit proof;
        # it is absent in the very first scout runs because the trainer builds the
        # optimizer before it calls wandb.init.
        topt = (cfg.get("train") or {}).get("optimizer") if isinstance(cfg.get("train"), dict) else None
        target = topt.get("target") if isinstance(topt, dict) else topt
        rows.append(dict(
            name=r.name,
            state=r.state,
            dataset=ds or ("cifar100" if margs.get("num_classes") == 100 else "cifar10"),
            depth=sum(blocks) if blocks else None,
            width=width,
            channels=eff,
            train=_pct(r.summary.get("train_accuracy")),
            test=_pct(r.summary.get("test_accuracy")),
            gate=r.summary.get("decay_group/gate_passed"),
            optimizer_target=target,
            url=r.url,
        ))
    return rows


CHECKED_BUILDER = "structural_reparam.agents.stage1_scout.lab.build_checked_kernel_only_sgd"


def gate_verdict(row):
    """"proven" if the run recorded the gate, "implied" if it merely used the
    checked builder (which cannot construct without passing), "INVALID" otherwise."""
    if row["gate"] == 1:
        return "proven"
    if row["optimizer_target"] == CHECKED_BUILDER:
        return "implied"
    return "INVALID"


def main():
    rows = collect()
    if not rows:
        print(f"No runs found in group {GROUP}.")
        return

    bad = [r for r in rows if gate_verdict(r) == "INVALID"]
    unfinished = [r for r in rows if r["state"] != "finished"]

    print(f"{len(rows)} run(s) in group {GROUP}; "
          f"{len(rows) - len(unfinished)} finished.\n")

    hdr = (f"{'dataset':9s} {'depth':>5s} {'width':>5s} {'channels':>14s} "
           f"{'train %':>8s} {'test %':>7s}  {'state':10s} decay gate")
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda r: (r["dataset"], r["width"] or 0, r["depth"] or 0)):
        tr = f"{r['train']:.2f}" if r["train"] is not None else "  --  "
        te = f"{r['test']:.2f}" if r["test"] is not None else "  --  "
        ch = ",".join(str(c) for c in r["channels"]) if r["channels"] else "?"
        print(f"{r['dataset']:9s} {str(r['depth']):>5s} {str(r['width']):>5s} "
              f"{ch:>14s} {tr:>8s} {te:>7s}  {r['state']:10s} {gate_verdict(r)}")

    if bad:
        print("\nDECAY-GROUP GATE NOT ESTABLISHED for these runs. They were not "
              "built with the campaign's checked optimizer, so they are invalid:")
        for r in bad:
            print(f"  {r['name']} (optimizer target {r['optimizer_target']}) {r['url']}")

    if unfinished:
        print(f"\n{len(unfinished)} run(s) not finished; the decision rule is "
              f"withheld until every cell is in.")
        return

    print("\nThe default cell and the depth ladder")
    print("-------------------------------------")
    print("The default cell is fixed: stage channels "
          f"{DEFAULT_STAGE_CHANNELS}, called width 1.0 in this campaign. The "
          "question left to the scout is how deep the ladder has to go.")

    for ds in ("cifar100", "cifar10"):
        cells = sorted([r for r in rows if r["dataset"] == ds and r["width"] == 1.0
                        and r["train"] is not None],
                       key=lambda r: r["depth"])
        if not cells:
            continue
        print(f"\n  {ds}, width 1.0, train accuracy against depth:")
        prev = None
        for r in cells:
            step = "" if prev is None else f"  (+{r['train'] - prev:.2f} over depth {prev_d})"
            sat = "  SATURATED" if r["train"] > SATURATION_CEILING else ""
            print(f"    depth {r['depth']}: {r['train']:.2f} %{step}{sat}")
            prev, prev_d = r["train"], r["depth"]
        knee = None
        for a, b in zip(cells, cells[1:]):
            if b["train"] - a["train"] < KNEE_GAIN:
                knee = b["depth"]
                break
        if knee is None:
            print(f"    No knee inside the scouted range: every depth step is still "
                  f"worth at least {KNEE_GAIN:g} point of train accuracy, so the "
                  f"ladder has to extend past depth {cells[-1]['depth']}.")
        else:
            print(f"    Saturation knee at depth {knee}; the ladder needs at least "
                  f"two depths beyond it.")


if __name__ == "__main__":
    main()
