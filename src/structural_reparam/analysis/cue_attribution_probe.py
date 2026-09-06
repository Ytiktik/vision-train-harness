"""Cue-attribution probe for the dot-cue CIFAR task.

Each epoch, evaluates the model on three ablated views of the test split and
logs how much of its accuracy comes from the easy pointwise dot cue vs the hard
spatial image cue:

  cue/joint_acc       — full input (image + dot), matches normal test accuracy
  cue/dot_only_acc    — image wiped to neutral, dot kept   (fast-cue ceiling)
  cue/image_only_acc  — dot removed, image kept            (slow-cue ceiling)
  cue/dot_reliance    — joint_acc - image_only_acc  (accuracy the dot adds)
  cue/image_reliance  — joint_acc - dot_only_acc    (accuracy the image adds)

The shortcut trap shows up as a *low* ``cue/image_only_acc`` (the model never
learned the spatial cue) despite a healthy ``cue/joint_acc``; compare across
the plain vs reparam arms.

Builds its own loaders from the experiment's dataset target, mirroring
``MechanisticProbe`` — the dataset ``ablate`` arg selects each view.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

from structural_reparam.analysis.registry import ProbeContext, register_probe


@register_probe("cue_attribution")
class CueAttributionProbe:
    def __init__(
        self,
        model: nn.Module,
        loaders: dict[str, DataLoader],
        device: torch.device,
        max_batches: int | None = None,
    ) -> None:
        self.model = model
        self.loaders = loaders
        self.device = device
        self.max_batches = max_batches

    @classmethod
    def from_context(cls, ctx: ProbeContext) -> "CueAttributionProbe":
        from structural_reparam.deploy.train import import_target, resolve_repo_path

        base_args = dict(ctx.config["dataset"].get("args", {}))
        base_args.update(ctx.variant.get("dataset", {}).get("args", {}))
        base_args["num_workers"] = 0
        base_args["persistent_workers"] = False
        base_args["augment"] = False  # deterministic eval views
        if "data_dir" in base_args:
            base_args["data_dir"] = resolve_repo_path(base_args["data_dir"])

        target = import_target(ctx.config["dataset"]["target"])
        loaders: dict[str, DataLoader] = {}
        for view in ("none", "dot", "image"):
            args = dict(base_args)
            args["ablate"] = view
            _, test_loader = target(**args)
            loaders[view] = test_loader

        return cls(
            model=ctx.model,
            loaders=loaders,
            device=ctx.device,
            max_batches=ctx.probe_config.get("max_batches"),
        )

    def close(self) -> None:
        return None

    @torch.no_grad()
    def _accuracy(self, loader: DataLoader) -> float:
        correct = 0
        total = 0
        for i, (x, y) in enumerate(loader):
            if self.max_batches is not None and i >= self.max_batches:
                break
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            preds = self.model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.numel()
        return correct / total if total else 0.0

    def epoch_stats(self, epoch: int = 0) -> dict[str, float]:
        was_training = self.model.training
        self.model.eval()
        try:
            joint = self._accuracy(self.loaders["none"])
            dot_only = self._accuracy(self.loaders["dot"])
            image_only = self._accuracy(self.loaders["image"])
        finally:
            if was_training:
                self.model.train()

        return {
            "cue/joint_acc": joint,
            "cue/dot_only_acc": dot_only,
            "cue/image_only_acc": image_only,
            "cue/dot_reliance": joint - image_only,
            "cue/image_reliance": joint - dot_only,
        }
