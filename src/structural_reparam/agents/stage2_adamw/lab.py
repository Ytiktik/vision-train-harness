"""Stage 2 AdamW entry module: the paper's default cell with AdamW in place of SGD.

The question this campaign answers is whether the two mechanisms of the paper
survive an adaptive optimizer. Nothing else changes: the model class, the arms,
the schedule, the batch size and the kernel-only weight decay are the default
cell's, and only the optimizer is swapped.

No model is defined here. Every arm uses
``structural_reparam.experiments.reparam_pinned_norm.lab.PlacedSharedScaleRepVGGCifar``
unchanged, exactly as the SGD campaign in ``agents/stage2`` does.

Two things need care when moving the recipe to AdamW.

* AdamW's weight decay is decoupled, so the shrink per step is ``lr * wd``
  rather than SGD's coupled ``lr * wd`` applied to the gradient. The grouping
  is the same one the paper's cell uses -- decay on conv and Linear weight
  tensors only, zero on every BatchNorm gamma, block beta and bias -- but the
  absolute size of the shrink follows the (much smaller) AdamW learning rate.
  That is stated where the results are reported; it is not something the
  grouping can fix.
* ``momentum`` and ``nesterov`` are in the trainer's optimizer-argument set and
  will be forwarded if a config sets them. They are accepted and ignored here
  so a config copied from the SGD campaign cannot silently mean something else.

Importing this module also imports ``agents.stage2.probe``, whose
``@register_probe`` decorator puts ``pair_channel_open`` in the registry. The
trainer builds the optimizer before it builds probes, so naming this module as
the optimizer target guarantees the probe is registered by the time the probe
config is read.
"""

from __future__ import annotations

import logging
from typing import Iterable, Sequence

import torch
import torch.nn as nn

from structural_reparam.experiments.reparam_pinned_norm.lab import (  # noqa: F401
    PlacedSharedScaleRepVGGCifar,
    _pair_kernel_ids,
)

from structural_reparam.agents.stage2 import probe as _probe  # noqa: F401  (registers the probe)

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parameter grouping. A transcription of ``build_kernel_wd_sgd``'s grouping with
# the optimizer class left open; ``tests/test_stage2_adamw.py`` asserts the two
# produce the same partition, so a change to the SGD side cannot drift silently.
# ---------------------------------------------------------------------------
def kernel_wd_param_groups(
    model: nn.Module,
    lr: float,
    weight_decay: float = 0.0,
    pair_wd_scale: float = 1.0,
    bn_gamma_lr_mult: float = 1.0,
    block_bias_lr_mult: float = 1.0,
    lr_overrides: dict[str, float] | None = None,
    wd_overrides: dict[str, float] | None = None,
    affine_overrides: dict[str, dict[str, float]] | None = None,
) -> list[dict]:
    """Kernel-only weight decay, keyed by (weight decay, learning rate).

    ``weight_decay`` applies to ndim>=2 weights only (conv kernels, ``fc.weight``);
    the kernels of blocks with >= 2 branches get ``weight_decay * pair_wd_scale``;
    gammas get ``lr * bn_gamma_lr_mult``; block biases (``.betas.<i>``) get
    ``lr * block_bias_lr_mult``; ``affine_overrides`` maps a module prefix to
    ``{"gamma": m_g, "beta": m_b}`` learning-rate multipliers under that prefix
    only, composing with the global ones; ``lr_overrides`` maps a prefix to an
    absolute learning rate; ``wd_overrides`` maps a prefix to a multiplier on the
    weight decay of the ndim>=2 weights under it.
    """
    pair_ids = _pair_kernel_ids(model)
    overrides = dict(lr_overrides or {})
    wd_over = dict(wd_overrides or {})
    groups: dict[tuple[float, float], list[nn.Parameter]] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        wd = float(weight_decay) if p.ndim >= 2 else 0.0
        if id(p) in pair_ids:
            wd *= float(pair_wd_scale)
        for prefix, scale in wd_over.items():
            if p.ndim >= 2 and (name == prefix or name.startswith(prefix + ".")):
                wd *= float(scale)
                break
        plr = float(lr)
        if name.endswith(".gamma") and bn_gamma_lr_mult != 1.0:
            plr *= float(bn_gamma_lr_mult)
        if ".betas." in name and block_bias_lr_mult != 1.0:
            plr *= float(block_bias_lr_mult)
        for prefix, mults in (affine_overrides or {}).items():
            if name == prefix or name.startswith(prefix + "."):
                if name.endswith(".gamma") and "gamma" in mults:
                    plr *= float(mults["gamma"])
                if ".betas." in name and "beta" in mults:
                    plr *= float(mults["beta"])
                break
        for prefix, olr in overrides.items():
            if name == prefix or name.startswith(prefix + "."):
                plr = float(olr)
                break
        groups.setdefault((wd, plr), []).append(p)
    return [{"params": ps, "weight_decay": wd, "lr": plr} for (wd, plr), ps in groups.items()]


# ---------------------------------------------------------------------------
# The builder, with the campaign's decay-group gate. Same two invariants the SGD
# gate checks: every ndim<2 parameter is decay-free, and every trainable
# parameter is covered exactly once.
# ---------------------------------------------------------------------------
def build_checked_kernel_wd_adamw(
    params: Iterable[nn.Parameter],
    model: nn.Module,
    lr: float,
    weight_decay: float = 0.0,
    betas: Sequence[float] = (0.9, 0.999),
    eps: float = 1e-8,
    amsgrad: bool = False,
    pair_wd_scale: float = 1.0,
    bn_gamma_lr_mult: float = 1.0,
    block_bias_lr_mult: float = 1.0,
    lr_overrides: dict[str, float] | None = None,
    wd_overrides: dict[str, float] | None = None,
    affine_overrides: dict[str, dict[str, float]] | None = None,
    momentum: float | None = None,
    nesterov: bool | None = None,
    **_: object,
) -> torch.optim.AdamW:
    """Decoupled-WD AdamW on the kernel-only groups, plus the campaign's gate.

    ``momentum`` and ``nesterov`` are accepted and ignored: the trainer forwards
    them whenever a config sets them, and AdamW has no such knobs. Passing one
    is a sign the config was copied from an SGD arm, so it is logged.
    """
    if momentum not in (None, 0.0) or nesterov:
        LOGGER.warning(
            "AdamW arm ignoring SGD-only optimizer args momentum=%r nesterov=%r; "
            "the AdamW momentum is betas=%r", momentum, nesterov, tuple(betas))

    param_groups = kernel_wd_param_groups(
        model,
        lr=lr,
        weight_decay=weight_decay,
        pair_wd_scale=pair_wd_scale,
        bn_gamma_lr_mult=bn_gamma_lr_mult,
        block_bias_lr_mult=block_bias_lr_mult,
        lr_overrides=lr_overrides,
        wd_overrides=wd_overrides,
        affine_overrides=affine_overrides,
    )
    optimizer = torch.optim.AdamW(
        param_groups, lr=lr, betas=tuple(betas), eps=eps, amsgrad=amsgrad
    )

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
            if p.ndim < 2 and wd != 0.0:
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

    summary["decay_group/gate_passed"] = 1
    summary["decay_group/n_groups"] = len(optimizer.param_groups)
    summary["decay_group/optimizer_is_adamw"] = 1
    print(f"[decay-group gate] PASSED (AdamW): {len(seen)} tensors in "
          f"{len(optimizer.param_groups)} groups, every ndim<2 parameter decay-free.",
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
