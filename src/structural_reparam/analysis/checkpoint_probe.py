"""Checkpoint snapshot probe.

Saves the model ``state_dict`` at a configured list of epochs (epoch 0 = the
initialized model, before any optimizer step) and, at the end of the run, logs
all saved checkpoints as ONE W&B artifact per run, then validates the upload by
re-reading the committed artifact's file manifest.

Enable via the generic ``probes:`` block:

.. code-block:: yaml

    probes:
      checkpoint:
        enabled: true
        epochs: [0, 5, 25, 50, 75, 100]

Each checkpoint file holds ``{"epoch", "variant", "seed", "state_dict"}`` with
all tensors on CPU (BN running stats included), so any activation/weight metric
can be recomputed offline at any saved epoch. When W&B is disabled (e.g.
``--smoke``) the files are still written under the run's output dir and the
artifact/validation steps are skipped.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

import torch
from torch import nn

from structural_reparam.analysis.registry import ProbeContext, register_probe

LOGGER = logging.getLogger(__name__)

DEFAULT_EPOCHS = [0, 5, 25, 50, 75, 100]

# W&B artifact names allow alphanumerics, dashes, underscores and dots.
_NAME_SANITIZER = re.compile(r"[^A-Za-z0-9_.-]+")


def _sanitize(name: str) -> str:
    return _NAME_SANITIZER.sub("-", name)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@register_probe("checkpoint")
class CheckpointProbe:
    """Save state_dicts at configured epochs; upload + validate one W&B artifact."""

    def __init__(
        self,
        model: nn.Module,
        epochs: list[int],
        save_dir: Path,
        variant_name: str,
        seed: int,
        group: str,
        model_target: str,
        variant_args: dict[str, Any],
    ) -> None:
        self.model = model
        self.epochs = sorted(set(int(e) for e in epochs))
        self.save_dir = save_dir
        self.variant_name = variant_name
        self.seed = seed
        self.group = group
        self.model_target = model_target
        self.variant_args = variant_args
        self.saved: dict[int, Path] = {}
        self.save_dir.mkdir(parents=True, exist_ok=True)
        if 0 in self.epochs:
            # Probes are built after the model but before the first optimizer
            # step, so this is the true init checkpoint.
            self._save(0)

    @classmethod
    def from_context(cls, ctx: ProbeContext) -> "CheckpointProbe":
        variant_name = ctx.variant.get("name", "variant")
        experiment = ctx.config.get("experiment", {}).get("name", "experiment")
        group = ctx.config.get("logging", {}).get("group", experiment)
        return cls(
            model=ctx.model,
            epochs=ctx.probe_config.get("epochs", DEFAULT_EPOCHS),
            save_dir=ctx.output_dir / "checkpoints" / f"{variant_name}_seed{ctx.seed}",
            variant_name=variant_name,
            seed=ctx.seed,
            group=group,
            model_target=ctx.config.get("model", {}).get("target", ""),
            variant_args=ctx.variant.get("args", {}),
        )

    def _save(self, epoch: int) -> Path:
        path = self.save_dir / f"ckpt_ep{epoch}.pt"
        state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        torch.save(
            {
                "epoch": epoch,
                "variant": self.variant_name,
                "seed": self.seed,
                "state_dict": state,
            },
            path,
        )
        self.saved[epoch] = path
        LOGGER.info(
            "Checkpoint saved variant=%s seed=%s epoch=%s -> %s (%.2f MB)",
            self.variant_name,
            self.seed,
            epoch,
            path,
            path.stat().st_size / 1e6,
        )
        return path

    def epoch_stats(self, epoch: int) -> dict[str, float]:
        if epoch in self.epochs and epoch not in self.saved:
            self._save(epoch)
            return {"checkpoint/saved_epoch": float(epoch)}
        return {}

    # -- upload & validation -------------------------------------------------

    def _artifact_name(self) -> str:
        return _sanitize(f"ckpt_{self.group}_{self.variant_name}_s{self.seed}")

    def close(self) -> None:
        import wandb

        if not self.saved:
            LOGGER.warning("CheckpointProbe: nothing was saved (epochs=%s)", self.epochs)
            return
        if wandb.run is None:
            LOGGER.info(
                "CheckpointProbe: W&B disabled; %d checkpoints left at %s, no artifact",
                len(self.saved),
                self.save_dir,
            )
            return

        run = wandb.run
        manifest = {
            "group": self.group,
            "variant": self.variant_name,
            "variant_args": self.variant_args,
            "model_target": self.model_target,
            "seed": self.seed,
            "epochs_requested": self.epochs,
            "epochs_saved": sorted(self.saved),
            "files": {
                f"ckpt_ep{ep}.pt": {
                    "bytes": p.stat().st_size,
                    "sha256": _sha256(p),
                }
                for ep, p in sorted(self.saved.items())
            },
            "run_id": run.id,
            "torch_version": torch.__version__,
        }
        manifest_path = self.save_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))

        artifact = wandb.Artifact(
            name=self._artifact_name(),
            type="checkpoint",
            metadata={
                "group": self.group,
                "variant": self.variant_name,
                "seed": self.seed,
                "epochs_saved": sorted(self.saved),
            },
        )
        for _, path in sorted(self.saved.items()):
            artifact.add_file(str(path))
        artifact.add_file(str(manifest_path))
        run.log_artifact(artifact)

        expected_files = len(self.saved) + 1  # + manifest.json
        try:
            artifact.wait()  # blocks until the artifact is committed server-side
            entries = artifact.manifest.entries
            n_files = len(entries)
            bad = [name for name, e in entries.items() if not (e.size or 0) > 0]
        except Exception as exc:  # noqa: BLE001 — validation must fail loudly
            run.summary["checkpoint/validated"] = 0
            raise RuntimeError(f"Checkpoint artifact commit/validation errored: {exc}") from exc

        if n_files != expected_files or bad:
            run.summary["checkpoint/validated"] = 0
            raise RuntimeError(
                f"Checkpoint artifact validation FAILED: expected {expected_files} files, "
                f"committed {n_files}, zero-size entries: {bad}"
            )
        run.summary["checkpoint/validated"] = 1
        run.summary["checkpoint/n_files"] = n_files
        run.summary["checkpoint/artifact_name"] = artifact.name
        run.summary["checkpoint/epochs_saved"] = sorted(self.saved)
        LOGGER.info(
            "Checkpoint artifact validated: %s (%d files, epochs %s)",
            artifact.name,
            n_files,
            sorted(self.saved),
        )
