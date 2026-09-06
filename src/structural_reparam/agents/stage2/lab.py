"""Stage 2 entry module: the campaign's decay-group gate around the placed-class
optimizer, plus the import that registers this campaign's probe fork.

No model is defined here. Every arm uses
``structural_reparam.experiments.reparam_pinned_norm.lab.PlacedSharedScaleRepVGGCifar``
unchanged, and the optimizer is that module's ``build_kernel_wd_sgd`` wrapped in
the assertion the campaign requires of every run.

Importing this module also imports ``probe``, whose ``@register_probe`` decorator
puts ``pair_channel_open`` in the registry. The trainer builds the optimizer
before it builds probes, so naming this module as the optimizer target guarantees
the probe is registered by the time the probe config is read.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import torch
import torch.nn as nn

from structural_reparam.experiments.reparam_pinned_norm.lab import (  # noqa: F401
    PlacedSharedScaleRepVGGCifar,
    build_kernel_wd_sgd,
)

from structural_reparam.agents.stage2 import probe as _probe  # noqa: F401  (registers the probe)



def _install_lr_floor(optimizer: torch.optim.SGD, model: nn.Module, lr: float,
                      prefixes, hold_from_epoch: float, total_epochs: int,
                      suffix: str = ".gamma") -> dict[str, float]:
    """Stop cooling the named coordinates at ``hold_from_epoch``.

    The trainer builds one cosine over every parameter group, so a coordinate can
    only be run hotter, never cooled differently. This gives it a floor instead.
    Because a cosine from a peak decreases monotonically, "follow the schedule
    until epoch k and then hold" is exactly "never let this rate fall below the
    value the schedule takes at epoch k", which needs no epoch counter: a step
    hook that raises the rate back to the floor implements it.

    The floor is the schedule's own value at that epoch,
    ``lr * (1 + cos(pi * k / total)) / 2``, so k=0 holds the rate at ``lr`` for
    the whole run and k=total is the ordinary schedule.

    The targeted parameters are moved into a group of their own first. Without
    that they share a group with every other decay-free parameter -- each block's
    scale and every shift -- and flooring that group would hold all of them.
    Splitting before the gate below means the gate covers the new group too.
    """
    targets = [n for n, _ in model.named_parameters()
               if any(n == pre + suffix or n.startswith(pre + ".") and n.endswith(suffix)
                      for pre in prefixes)]
    if not targets:
        raise RuntimeError(f"lr floor: no parameter matches {list(prefixes)} with suffix {suffix!r}")
    by_name = dict(model.named_parameters())
    wanted = {id(by_name[n]) for n in targets}

    moved, home_wd, home_lr = [], None, None
    for group in optimizer.param_groups:
        keep = []
        for q in group["params"]:
            if id(q) in wanted:
                moved.append(q)
                home_wd, home_lr = float(group["weight_decay"]), float(group["lr"])
            else:
                keep.append(q)
        group["params"] = keep
    if len(moved) != len(targets):
        raise RuntimeError(f"lr floor: found {len(moved)} of {len(targets)} targeted parameters")
    optimizer.add_param_group({"params": moved, "weight_decay": home_wd, "lr": home_lr})

    k, total = float(hold_from_epoch), int(total_epochs)
    if not 0.0 <= k <= total:
        raise RuntimeError(f"lr floor: hold_from_epoch {k} outside [0, {total}]")
    floor = float(home_lr) * (1.0 + math.cos(math.pi * k / total)) / 2.0
    gi = len(optimizer.param_groups) - 1

    def _floor(opt, *_a, **_k):
        if opt.param_groups[gi]["lr"] < floor:
            opt.param_groups[gi]["lr"] = floor

    optimizer.register_step_pre_hook(_floor)
    print(f"[lr floor] {targets} moved to group {gi} (weight_decay={home_wd:g}, "
          f"lr={home_lr:g}); cooling stops at epoch {k:g} of {total}, floor={floor:.6g}",
          flush=True)
    return {"lr_floor/value": floor, "lr_floor/hold_from_epoch": k,
            "lr_floor/group": float(gi), "lr_floor/n_params": float(len(moved))}


def _decay_gammas(optimizer: torch.optim.SGD, model: nn.Module, prefixes,
                  weight_decay: float, suffix: str = ".gamma") -> tuple[set[int], dict[str, float]]:
    """Put weight decay on the shared scales under ``prefixes`` and nothing else.

    The campaign's recipe leaves every BatchNorm scale decay-free, which is what
    makes the last block's scale the one coordinate free to grow the logit
    magnitude (section 4.1). This arm removes that freedom for the named blocks
    only: their ``.gamma`` parameters are moved out of the decay-free group into a
    group of their own at ``weight_decay``, keeping the learning rate they already
    had (so an affine multiplier on the same block still applies). Every other
    ndim<2 parameter, the block shifts included, stays decay-free, and the gate
    below is told which tensors are exempt so it still rejects any other decay on
    an ndim<2 parameter. Returns the exempted ids and a summary for W&B.
    """
    targets = [n for n, _ in model.named_parameters()
               if any(n == pre + suffix or n.startswith(pre + ".") and n.endswith(suffix)
                      for pre in prefixes)]
    if not targets:
        raise RuntimeError(f"gamma decay: no parameter matches {list(prefixes)} with suffix {suffix!r}")
    by_name = dict(model.named_parameters())
    wanted = {id(by_name[n]) for n in targets}
    moved_by_lr: dict[float, list[nn.Parameter]] = {}
    for group in optimizer.param_groups:
        keep = []
        for q in group["params"]:
            if id(q) in wanted:
                moved_by_lr.setdefault(float(group["lr"]), []).append(q)
            else:
                keep.append(q)
        group["params"] = keep
    n_moved = sum(len(v) for v in moved_by_lr.values())
    if n_moved != len(targets):
        raise RuntimeError(f"gamma decay: found {n_moved} of {len(targets)} targeted parameters")
    optimizer.param_groups[:] = [g for g in optimizer.param_groups if g["params"]]
    for glr, ps in moved_by_lr.items():
        optimizer.add_param_group({"params": ps, "weight_decay": float(weight_decay), "lr": glr})
    print(f"[gamma decay] {targets} moved to their own group(s) at weight_decay={weight_decay:g} "
          f"(lr {sorted(moved_by_lr)}); every other ndim<2 parameter stays decay-free", flush=True)
    return wanted, {"gamma_wd/value": float(weight_decay), "gamma_wd/n_params": float(n_moved)}


def build_checked_kernel_wd_sgd(params: Iterable[nn.Parameter], model: nn.Module,
                                lr: float,
                                hold_lr_prefixes: Sequence[str] | None = None,
                                hold_from_epoch: float | None = None,
                                total_epochs: int | None = None,
                                gamma_wd_prefixes: Sequence[str] | None = None,
                                gamma_wd: float | None = None,
                                **kwargs) -> torch.optim.SGD:
    """``build_kernel_wd_sgd`` plus the campaign's decay-group assertion.

    ``build_kernel_wd_sgd`` keys its parameter groups by (weight decay, learning
    rate), so a run with per-block affine multipliers has several groups at the
    same decay. The gate does not care how they are partitioned; it checks the two
    invariants the campaign rests on:

    * every parameter with ``ndim < 2`` -- BatchNorm gamma, every block beta, every
      bias -- sits in a group whose weight decay is exactly zero, and
    * every trainable parameter is covered by exactly one group.

    ``gamma_wd_prefixes`` with ``gamma_wd`` is the one deliberate exception: the
    shared scales of the named blocks are decayed at ``gamma_wd`` (see
    ``_decay_gammas``), and the gate exempts exactly those tensors.

    Group sizes and learning rates are printed to the training log and written to
    the W&B run summary. Because the trainer builds the optimizer before calling
    ``wandb.init``, the summary write is deferred to a one-shot optimizer step
    hook, which fires once the run exists and then removes itself.
    """
    optimizer = build_kernel_wd_sgd(params, model=model, lr=lr, **kwargs)

    floor_summary: dict[str, float] = {}
    if hold_lr_prefixes:
        if hold_from_epoch is None or total_epochs is None:
            raise RuntimeError("hold_lr_prefixes needs hold_from_epoch and total_epochs")
        floor_summary = _install_lr_floor(optimizer, model, lr, hold_lr_prefixes,
                                          hold_from_epoch, total_epochs)

    exempt: set[int] = set()
    if gamma_wd_prefixes:
        if gamma_wd is None:
            raise RuntimeError("gamma_wd_prefixes needs gamma_wd")
        exempt, gamma_summary = _decay_gammas(optimizer, model, gamma_wd_prefixes, gamma_wd)
        floor_summary.update(gamma_summary)
    elif gamma_wd is not None:
        raise RuntimeError("gamma_wd needs gamma_wd_prefixes")

    name_of = {id(p): n for n, p in model.named_parameters()}
    seen: set[int] = set()
    summary: dict[str, float] = {}
    offenders: list[str] = []

    for gi, group in enumerate(optimizer.param_groups):
        wd = float(group["weight_decay"])
        glr = float(group.get("lr", lr))
        tensors = elements = 0
        for p in group["params"]:
            if id(p) in seen:
                offenders.append(f"{name_of.get(id(p), '<unnamed>')} is in more than one group")
            seen.add(id(p))
            tensors += 1
            elements += p.numel()
            if p.ndim < 2 and wd != 0.0 and id(p) not in exempt:
                offenders.append(
                    f"{name_of.get(id(p), '<unnamed>')} has ndim={p.ndim} but sits in a "
                    f"group with weight_decay={wd}")
        summary[f"decay_group/{gi}_weight_decay"] = wd
        summary[f"decay_group/{gi}_lr"] = glr
        summary[f"decay_group/{gi}_tensors"] = tensors
        summary[f"decay_group/{gi}_elements"] = elements
        print(f"[decay-group gate] group {gi}: weight_decay={wd:g} lr={glr:g} "
              f"tensors={tensors} elements={elements}", flush=True)

    missing = [n for n, p in model.named_parameters()
               if p.requires_grad and id(p) not in seen]
    if missing:
        offenders.append("parameters in no optimizer group: " + ", ".join(missing[:8]))
    if offenders:
        raise RuntimeError("kernel-only weight-decay gate FAILED:\n  " + "\n  ".join(offenders))

    summary.update(floor_summary)
    summary["decay_group/gate_passed"] = 1
    summary["decay_group/n_groups"] = len(optimizer.param_groups)
    print(f"[decay-group gate] PASSED: {len(seen)} tensors in "
          f"{len(optimizer.param_groups)} groups, every ndim<2 parameter decay-free"
          + (f" except the {len(exempt)} decayed gamma tensor(s) named above." if exempt else "."),
          flush=True)

    state = {"handle": None, "done": False}

    def _record_once(opt, *_a, **_k):  # pragma: no cover - needs a live run
        if state["done"]:
            return
        try:
            import wandb
            if wandb.run is None:
                return
            wandb.run.summary.update(summary)
            state["done"] = True
            if state["handle"] is not None:
                state["handle"].remove()
        except Exception:
            state["done"] = True
            if state["handle"] is not None:
                state["handle"].remove()

    try:
        state["handle"] = optimizer.register_step_post_hook(_record_once)
    except AttributeError:
        pass

    return optimizer
