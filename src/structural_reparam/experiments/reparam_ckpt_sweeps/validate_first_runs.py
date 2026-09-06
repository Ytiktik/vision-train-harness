"""Early validation for the reparam_ckpt_sweeps campaign (user-mandated gates).

Gate A — checkpoint usability: for the first finished run(s) in the ckpt100_*
groups, download the checkpoint artifact, strict-load the ep100 state_dict into
LayerwiseRepVGGCifar, evaluate on the real test set, and require the recomputed
test accuracy to match the run's logged final `test_accuracy`.

Gate B — replication parity: compare the run's train/test loss+accuracy history
at matching epochs against the SAME variant+seed run in the original
`sweeps100_*_wd5e4` group; report deltas against the original group's
across-seed noise. (Same seeds/recipe, but probe removal changes RNG
consumption, so expect seed-level closeness, not bit-identity.)

Usage:
  python -m structural_reparam.experiments.reparam_ckpt_sweeps.validate_first_runs \
      [--max-runs 3] [--project yoovi-t-tel-aviv-university/claude-autonomous-reparam]
"""

from __future__ import annotations

import argparse
import math
import tempfile
from pathlib import Path

import torch
import wandb

GROUP_MAP = {  # ckpt100 group -> original sweeps100 group
    "ckpt100_c100_depth_wd5e4": "sweeps100_c100_depth_wd5e4",
    "ckpt100_c100_width_wd5e4": "sweeps100_c100_width_wd5e4",
    "ckpt100_c10_depth_wd5e4": "sweeps100_c10_depth_wd5e4",
    "ckpt100_c10_width_wd5e4": "sweeps100_c10_width_wd5e4",
}
HIST_KEYS = ["epoch", "train_loss", "train_accuracy", "test_loss", "test_accuracy"]


def build_model(run_config: dict):
    from structural_reparam.experiments.reparam_sweeps100.lab import LayerwiseRepVGGCifar

    args = dict(run_config["model"]["args"])
    args.update(run_config["variant"].get("args", {}))
    return LayerwiseRepVGGCifar(**args), args


def build_test_loader(run_config: dict):
    from structural_reparam.deploy.train import import_target, resolve_repo_path

    ds = run_config["dataset"]
    args = dict(ds.get("args", {}))
    args["data_dir"] = resolve_repo_path(args["data_dir"])
    args["num_workers"] = 2
    args.pop("persistent_workers", None)
    _, test_loader = import_target(ds["target"])(**args)
    return test_loader


@torch.no_grad()
def evaluate(model, loader) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        out = model(x)
        correct += (out.argmax(1) == y).sum().item()
        total += y.numel()
    return correct / total


def gate_a(run, api) -> tuple[bool, str]:
    """Download artifact, load ep100 checkpoint, re-evaluate, compare."""
    if run.summary.get("checkpoint/validated") != 1:
        return False, f"{run.name}: checkpoint/validated != 1"
    arts = [a for a in run.logged_artifacts() if a.type == "checkpoint"]
    if not arts:
        return False, f"{run.name}: no checkpoint artifact"
    art = arts[0]
    with tempfile.TemporaryDirectory() as td:
        root = Path(art.download(root=td))
        files = sorted(p.name for p in root.iterdir())
        n_ckpt = len([f for f in files if f.startswith("ckpt_ep")])
        ck_path = root / "ckpt_ep100.pt"
        if not ck_path.exists():
            return False, f"{run.name}: ckpt_ep100.pt missing (files: {files})"
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        model, _ = build_model(run.config)
        model.load_state_dict(ck["state_dict"], strict=True)
        loader = build_test_loader(run.config)
        acc = evaluate(model, loader)
    logged = float(run.summary.get("test_accuracy", float("nan")))
    ok = abs(acc - logged) < 5e-3  # frozen BN stats + same transform: should be ~exact
    return ok, (
        f"{run.name}: artifact {art.name} ({n_ckpt} ckpts) | recomputed ep100 acc "
        f"{acc:.4f} vs logged {logged:.4f} (|Δ|={abs(acc - logged):.4f}) -> "
        f"{'OK' if ok else 'MISMATCH'}"
    )


def gate_b(run, api, project: str) -> str:
    """Metric-history parity vs the original sweeps100 run (same variant+seed)."""
    group = run.config["logging"]["group"]
    orig_group = GROUP_MAP[group]
    variant = run.config["variant"]["name"]
    seed = run.config["seed"]
    orig_runs = api.runs(project, filters={"group": orig_group})
    twin = None
    seed_final_accs = []
    for r in orig_runs:
        if r.config.get("variant", {}).get("name") == variant:
            seed_final_accs.append(float(r.summary.get("test_accuracy", float("nan"))))
            if r.config.get("seed") == seed:
                twin = r
    if twin is None:
        return f"{run.name}: no twin {orig_group}/{variant}/seed{seed} — parity SKIPPED"

    import pandas as pd

    new_h = run.history(keys=HIST_KEYS, pandas=True)
    old_h = twin.history(keys=HIST_KEYS, pandas=True)
    for df in (new_h, old_h):
        for k in HIST_KEYS:
            df[k] = pd.to_numeric(df[k], errors="coerce")
    merged = new_h.merge(old_h, on="epoch", suffixes=("_new", "_old"))
    lines = [f"{run.name} vs {orig_group} twin (n_common_epochs={len(merged)}):"]
    for k in HIST_KEYS[1:]:
        d = (merged[f"{k}_new"] - merged[f"{k}_old"]).abs().dropna()
        if len(d) == 0:
            lines.append(f"  {k}: no common numeric epochs")
            continue
        lines.append(f"  {k}: n={len(d)} mean|Δ|={d.mean():.4f} max|Δ|={d.max():.4f}")
    finals = [a for a in seed_final_accs if not math.isnan(a)]
    if len(finals) >= 2:
        mu = sum(finals) / len(finals)
        sd = (sum((a - mu) ** 2 for a in finals) / (len(finals) - 1)) ** 0.5
        new_final = float(run.summary.get("test_accuracy", float("nan")))
        z = abs(new_final - mu) / sd if sd > 0 else float("inf")
        lines.append(
            f"  final test_acc {new_final:.4f} vs orig {mu:.4f}±{sd:.4f} "
            f"(z={z:.2f}, {'within' if z < 3 else 'OUTSIDE'} 3-sd seed noise)"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="yoovi-t-tel-aviv-university/claude-autonomous-reparam")
    ap.add_argument("--max-runs", type=int, default=3)
    args = ap.parse_args()

    api = wandb.Api()
    finished = []
    for group in GROUP_MAP:
        for r in api.runs(args.project, filters={"group": group}):
            if r.state == "finished":
                finished.append(r)
    if not finished:
        print("NO FINISHED RUNS YET in any ckpt100_* group")
        return
    print(f"{len(finished)} finished run(s); validating up to {args.max_runs}\n")
    failures = 0
    for run in finished[: args.max_runs]:
        ok, msg = gate_a(run, api)
        print("GATE A:", msg)
        if not ok:
            failures += 1
        print("GATE B:", gate_b(run, api, args.project), "\n")
    print("VERDICT:", "PASS" if failures == 0 else f"FAIL ({failures} gate-A failures)")


if __name__ == "__main__":
    main()
