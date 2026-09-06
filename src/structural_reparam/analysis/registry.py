"""Generic registry for snapshot (epoch-end) analysis probes.

A probe class is registered with ``@register_probe("name")`` and must expose:

  - ``from_context(cls, ctx: ProbeContext) -> Probe`` classmethod, building the
    probe from a uniform context object that includes the model, optimizer,
    criterion, device, full config, variant dict, output dir, seed, and the
    probe's own sub-config.
  - ``epoch_stats(epoch: int) -> dict[str, float]``: returns metrics that are
    merged into the per-epoch wandb record. Keys typically prefixed with the
    probe name (e.g. ``"mech/..."``).
  - ``close() -> None``: cleanup (close files, remove hooks, etc.).

``train.py`` reads ``probes:`` from the experiment YAML, looks up each entry
by name, and builds it via ``from_context``. No per-probe wire-in code lives
in ``train.py`` — adding a new snapshot probe is a one-file change: declare
the class with ``@register_probe(...)`` and enable it in the YAML.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn


@dataclass
class ProbeContext:
    model: nn.Module
    optimizer: torch.optim.Optimizer
    criterion: nn.Module
    device: torch.device
    config: dict[str, Any]
    variant: dict[str, Any]
    output_dir: Path
    seed: int
    probe_config: dict[str, Any]


PROBE_REGISTRY: dict[str, type] = {}


def register_probe(name: str) -> Callable[[type], type]:
    def decorator(cls: type) -> type:
        cls.PROBE_NAME = name
        PROBE_REGISTRY[name] = cls
        return cls

    return decorator


def build_probes(ctx_factory: Callable[[str, dict[str, Any]], ProbeContext],
                 probes_config: dict[str, dict[str, Any]] | None) -> list:
    """Build the list of probes declared in the config.

    ``probes_config`` is the ``probes:`` block from YAML. ``ctx_factory(name,
    probe_config)`` returns a fully-populated :class:`ProbeContext` for each
    enabled probe. Probes whose entry is missing, ``None``, or has
    ``enabled: false`` are skipped.
    """
    probes: list = []
    if not probes_config:
        return probes
    for name, probe_cfg in probes_config.items():
        if probe_cfg is None:
            continue
        if not probe_cfg.get("enabled", True):
            continue
        if name not in PROBE_REGISTRY:
            raise ValueError(
                f"Unknown probe '{name}'. Registered probes: {sorted(PROBE_REGISTRY)}"
            )
        cls = PROBE_REGISTRY[name]
        ctx = ctx_factory(name, probe_cfg)
        probes.append(cls.from_context(ctx))
    return probes
