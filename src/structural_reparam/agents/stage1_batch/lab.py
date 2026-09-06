"""Queued package 'stage1_batch' for the `stage1` campaign (kernel-only weight decay).

Nothing here defines a new model. Rig A reuses
``structural_reparam.experiments.reparam_sweeps100.lab.LayerwiseRepVGGCifar``
unchanged for both arms and only adds the campaign's decay-group verification
gate around the shared kernel-only-weight-decay optimizer builder.

The gate is copied from ``agents/stage1_scout/lab.py`` rather than imported, so
this directory stands alone once the scout's directory is archived.

The gate exists because this whole campaign was opened on the finding that the
paper's introductory phenomenology was measured with weight decay applied to
every parameter, including BatchNorm gamma/beta and every bias. Every run must
therefore prove, in its own W&B record, that decay reached the kernels only.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from structural_reparam.optim import sgd_kernel_only_weight_decay

# Re-exported so a config may point `model.target` at this module if that is ever
# more convenient than the experiments package. The class itself is untouched.
from structural_reparam.experiments.reparam_sweeps100.lab import (  # noqa: F401
    LayerwiseRepVGGCifar,
)


def build_checked_kernel_only_sgd(
    params,
    model: nn.Module,
    lr: float,
    momentum: float = 0.0,
    weight_decay: float = 0.0,
    nesterov: bool = False,
) -> torch.optim.SGD:
    """``sgd_kernel_only_weight_decay`` plus the campaign's decay-group assertion.

    Builds the optimizer with the shared helper, then verifies and records that

    * every parameter with ``ndim < 2`` (BatchNorm gamma and beta, every bias)
      sits in a param group whose ``weight_decay`` is exactly zero, and
    * every parameter of the model is covered by exactly one group.

    The per-group tensor and element counts are printed to stdout so they land in
    the training log, and are written to the W&B run summary when a run is live.
    A violation raises, so a mis-wired config fails at optimizer construction
    rather than producing a silently invalid data point.
    """
    optimizer = sgd_kernel_only_weight_decay(
        params, model=model, lr=lr, momentum=momentum,
        weight_decay=weight_decay, nesterov=nesterov,
    )

    name_of = {id(p): n for n, p in model.named_parameters()}
    seen: set[int] = set()
    summary: dict[str, float | int] = {}
    offenders: list[str] = []

    for gi, group in enumerate(optimizer.param_groups):
        wd = float(group["weight_decay"])
        tensors = 0
        elements = 0
        for p in group["params"]:
            if id(p) in seen:
                offenders.append(
                    f"{name_of.get(id(p), '<unnamed>')} appears in more than one group"
                )
            seen.add(id(p))
            tensors += 1
            elements += p.numel()
            if p.ndim < 2 and wd != 0.0:
                offenders.append(
                    f"{name_of.get(id(p), '<unnamed>')} has ndim={p.ndim} but sits in a "
                    f"group with weight_decay={wd}"
                )
        summary[f"decay_group/{gi}_weight_decay"] = wd
        summary[f"decay_group/{gi}_tensors"] = tensors
        summary[f"decay_group/{gi}_elements"] = elements
        print(
            f"[decay-group gate] group {gi}: weight_decay={wd:g} "
            f"tensors={tensors} elements={elements}",
            flush=True,
        )

    missing = [n for n, p in model.named_parameters()
               if p.requires_grad and id(p) not in seen]
    if missing:
        offenders.append(
            "parameters missing from every optimizer group: " + ", ".join(missing[:8])
        )

    if offenders:
        raise RuntimeError(
            "kernel-only weight-decay gate FAILED:\n  " + "\n  ".join(offenders)
        )

    summary["decay_group/gate_passed"] = 1
    print(
        f"[decay-group gate] PASSED: {len(seen)} parameter tensors covered, "
        f"every ndim<2 parameter is decay-free.",
        flush=True,
    )

    # The trainer builds the optimizer BEFORE it calls wandb.init, so writing the
    # summary here would be a silent no-op. Defer it to the first optimizer step,
    # by which time the run exists, and remove the hook once it has fired.
    state = {"handle": None, "done": False}

    def _record_once(opt, *_args, **_kwargs):  # pragma: no cover - needs a live run
        if state["done"]:
            return
        try:
            import wandb

            if wandb.run is None:
                return  # not up yet; try again on the next step
            wandb.run.summary.update(summary)
            state["done"] = True
            if state["handle"] is not None:
                state["handle"].remove()
        except Exception:
            state["done"] = True  # logging must never break training
            if state["handle"] is not None:
                state["handle"].remove()

    try:
        state["handle"] = optimizer.register_step_post_hook(_record_once)
    except AttributeError:  # very old torch; fall back to a best-effort write
        try:
            import wandb

            if wandb.run is not None:
                wandb.run.summary.update(summary)
        except Exception:
            pass

    return optimizer
