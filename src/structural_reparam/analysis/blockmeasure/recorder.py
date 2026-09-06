"""The recorder: a registry probe that stores the primitives of one two-branch block
at every selected optimizer step, from inside the project's trainer.

What it records, and why only that
----------------------------------
Every quantity of the theory is a function of a few raw objects at one moment: the
two kernels, gamma, the per-branch biases, the gradients of those, the optimizer's
momentum buffers for the same tensors, the learning rate in force, and the batch
covariance the normalizer saw. The recorder stores exactly those, per selected
step, and nothing derived: no angle, no whitened vector, no r. Every definition can
then change offline without a new training run, and every identity of
``measure.py`` can be checked against what actually happened.

How it attaches, without touching the trainer
---------------------------------------------
The trainer builds probes from the registry with a context that includes the
optimizer, and calls each probe once per epoch. A torch optimizer accepts step
hooks of its own, so the recorder registers a pre-step hook (copies the
parameters before the update) and a post-step hook (copies the gradients, which
are still attached, the momentum buffers that were just applied, the updated
parameters, the learning rates of the parameter groups, and the normalizers'
batch variances of this step's forward pass). The applied step is then exactly
``w_post = w_pre - lr * buffer`` (torch SGD without Nesterov: d = g + wd * w;
buffer = mu * buffer + d; w <- w - lr * buffer), which ``measure.check_applied_step``
verifies on every recorded step.

The per-step batch covariance of the block's input is recorded too (a forward
pre-hook keeps a reference to the input; the covariance is formed in the post-hook
and the reference released), so the identities that BatchNorm makes exact in its
own frame can be checked exactly, and the fixed-Sigma frame's error is measured
rather than assumed.

Epoch bookkeeping: the trainer's epochs are 1-based and ``epoch_stats(e)`` is
called after epoch ``e`` finishes, so the recorder starts at epoch 1 and moves to
``e + 1`` on each call. A step is recorded when the current epoch is in the
configured list (default 1 to 5 and every tenth epoch).

Sigma for the fixed frame is estimated once at construction with
``measure.estimate_patch_covariance`` under a forked random stream, so the run's
own shuffling and augmentation are untouched, and a second estimate at a different
seed is stored beside it as the estimator's own error bar.

File format
-----------
Under ``<output_dir>/blockmeasure/<variant>_seed<seed>/``:

  ep<E>.npz      one file per recorded epoch, arrays over the T steps of that epoch:
                 step [T] int, epoch [T] int, lr_w lr_gamma lr_beta wd_w [T],
                 w_pre w_post g_w buf_w [T, 2, C, D] (the model's own dtype),
                 gamma_pre gamma_post g_gamma buf_gamma [T, C],
                 beta_pre beta_post g_beta buf_beta [T, 2, C] (NaN where a beta is frozen),
                 batch_var [T, 2, C] (the normalizers' biased batch variance),
                 sigma_batch [T, D, D] float64 (if enabled).
  sigma.npz/json the fixed-frame covariance and its settings and hash.
  sigma_check.npz/json  the second estimate.
  manifest.json  provenance: group, variant, seed, block, parameter names, optimizer
                 hyperparameters, probe config, module version, torch version, the
                 conventions of ``measure.py``, and the per-file sizes and hashes.

``load_recording`` reads a directory back into one ``Recording`` with the epochs
concatenated in step order.

Registration
------------
The optimizer-target re-export at the bottom of this module is how the probe gets
registered before the trainer reads the ``probes:`` block, following the stage-2
campaign's pattern: name this module's ``build_checked_kernel_wd_sgd`` as the
config's optimizer target and the import happens in time. No shared module is edited.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from structural_reparam.analysis.registry import ProbeContext, register_probe
from structural_reparam.analysis.blockmeasure import BLOCKMEASURE_VERSION
from structural_reparam.analysis.blockmeasure import measure as M

LOGGER = logging.getLogger(__name__)

DEFAULT_EPOCHS: list[int] = [1, 2, 3, 4, 5] + list(range(10, 101, 10))
"""Recorded epochs in the trainer's 1-based numbering: every step of the first five
epochs (the climb, at full resolution) and of every tenth epoch after (the parked
regime and the late-training angle)."""

_NAME_SANITIZER = re.compile(r"[^A-Za-z0-9_.-]+")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _find_two_branch_blocks(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Every module that looks like a ``SharedScaleRepVGGBlock`` with two branches,
    in network order: it has ``convs`` (a ModuleList of length 2), ``stats`` and
    ``gamma``. Duck typing keeps the recorder independent of the model module."""
    out = []
    for name, m in model.named_modules():
        convs = getattr(m, "convs", None)
        if isinstance(convs, nn.ModuleList) and len(convs) == 2 and hasattr(m, "stats") and hasattr(m, "gamma"):
            out.append((name, m))
    return out


@register_probe("blockmeasure")
class BlockMeasureRecorder:
    """Record the primitives of one two-branch block at every selected step."""

    PROBE_NAME = "blockmeasure"

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, block_name: str, block: nn.Module,
                 epochs: list[int], sigma: M.CovarianceEstimate, sigma_check: M.CovarianceEstimate | None,
                 save_dir: Path, meta: dict[str, Any], record_batch_sigma: bool = True) -> None:
        self.model = model
        self.optimizer = optimizer
        self.block_name = block_name
        self.block = block
        self.epochs = sorted(set(int(e) for e in epochs))
        self.sigma = sigma
        self.sigma_check = sigma_check
        self.save_dir = Path(save_dir)
        self.meta = meta
        self.record_batch_sigma = bool(record_batch_sigma)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.w = [block.convs[0].weight, block.convs[1].weight]
        self.gamma = block.gamma
        self.betas = list(block.betas)
        self._group_of = {}
        for gi, group in enumerate(optimizer.param_groups):
            for p in group["params"]:
                self._group_of[id(p)] = gi
        for p in self.w + [self.gamma]:
            if id(p) not in self._group_of:
                raise RuntimeError(f"blockmeasure: parameter of block {block_name} is not in the optimizer")
        g0 = optimizer.param_groups[self._group_of[id(self.w[0])]]
        self.momentum = float(g0.get("momentum", 0.0))
        self.nesterov = bool(g0.get("nesterov", False))
        self.dampening = float(g0.get("dampening", 0.0))
        if self.nesterov or self.dampening != 0.0:
            raise NotImplementedError("blockmeasure assumes plain SGD momentum (no Nesterov, no dampening), "
                                      "which is what the applied-step identity uses")

        self._epoch = 1
        self._step = 0
        self._pending: dict[str, Any] | None = None
        self._captured_input: torch.Tensor | None = None
        self._rows: list[dict[str, Any]] = []
        self.files: dict[int, Path] = {}
        self.steps_total_recorded = 0

        self._h_pre = optimizer.register_step_pre_hook(self._pre_step)
        self._h_post = optimizer.register_step_post_hook(self._post_step)
        self._h_fwd = block.register_forward_pre_hook(self._capture_input) if self.record_batch_sigma else None

        self.sigma.save(self.save_dir / "sigma")
        if self.sigma_check is not None:
            self.sigma_check.save(self.save_dir / "sigma_check")
        self._write_manifest(final=False)

    # -- construction from the trainer's context ------------------------------

    @classmethod
    def from_context(cls, ctx: ProbeContext) -> "BlockMeasureRecorder":
        pc = dict(ctx.probe_config or {})
        block_index = int(pc.get("block", 0))
        blocks = _find_two_branch_blocks(ctx.model)
        if not blocks:
            raise RuntimeError("blockmeasure: the model has no two-branch block")
        if block_index != 0:
            raise NotImplementedError("blockmeasure v1 records block 0 only (its input covariance is the data's)")
        block_name, block = blocks[block_index]
        if getattr(block, "stride", 1) != 1:
            raise NotImplementedError("blockmeasure v1 assumes stride 1 at the recorded block")

        sig_cfg = dict(pc.get("sigma", {}))
        if "precomputed" in sig_cfg:
            sigma = M.CovarianceEstimate.load(sig_cfg["precomputed"])
            sigma_check = None
        else:
            n_batches = int(sig_cfg.get("n_batches", M.SIGMA_BATCHES))
            seed = int(sig_cfg.get("seed", M.SIGMA_SEED))
            sigma = M.estimate_patch_covariance(ctx.config.get("dataset"), n_batches=n_batches, seed=seed, device=ctx.device)
            check_batches = int(sig_cfg.get("crosscheck_batches", n_batches))
            sigma_check = M.estimate_patch_covariance(ctx.config.get("dataset"), n_batches=check_batches,
                                                      seed=seed + M.SIGMA_CROSSCHECK_SEED_OFFSET, device=ctx.device)
            LOGGER.info("blockmeasure: Sigma from %d batches (seed %d): gap %.2f, hash %s; cross-check |v_max . v_max'| = %.6f",
                        n_batches, seed, sigma.gap_ratio, sigma.sha256[:12], abs(float(sigma.v_max @ sigma_check.v_max)))

        variant_name = ctx.variant.get("name", "variant")
        experiment = ctx.config.get("experiment", {}).get("name", "experiment")
        group = ctx.config.get("logging", {}).get("group", experiment)
        train_cfg = ctx.config.get("train", {})
        meta = {
            "group": group, "variant": variant_name, "variant_args": ctx.variant.get("args", {}),
            "experiment": experiment, "seed": int(ctx.seed), "block_index": block_index, "block_name": block_name,
            "model_target": ctx.config.get("model", {}).get("target", ""),
            "parameter_names": {"w1": f"{block_name}.convs.0.weight", "w2": f"{block_name}.convs.1.weight",
                                "gamma": f"{block_name}.gamma", "betas": [f"{block_name}.betas.{j}" for j in range(len(block.betas))]},
            "normalizer_eps": float(getattr(block.stats[0], "eps", float("nan"))),
            "train": {k: train_cfg.get(k) for k in ("lr", "momentum", "weight_decay", "scheduler", "epochs", "amp", "max_grad_norm")},
            "probe_config": pc, "blockmeasure_version": BLOCKMEASURE_VERSION, "torch_version": torch.__version__,
            "device": str(ctx.device),
            "tf32": {"cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
                     "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32)},
            "conventions": M.CONVENTIONS,
        }
        return cls(model=ctx.model, optimizer=ctx.optimizer, block_name=block_name, block=block,
                   epochs=pc.get("epochs", DEFAULT_EPOCHS), sigma=sigma, sigma_check=sigma_check,
                   save_dir=ctx.output_dir / "blockmeasure" / f"{variant_name}_seed{ctx.seed}", meta=meta,
                   record_batch_sigma=pc.get("record_batch_sigma", True))

    # -- hooks ------------------------------------------------------------------

    @property
    def recording_now(self) -> bool:
        return self._epoch in self.epochs

    def _capture_input(self, module: nn.Module, inp: tuple) -> None:
        if module.training and self.recording_now:
            self._captured_input = inp[0].detach()

    def _pre_step(self, optimizer, args, kwargs) -> None:
        if not self.recording_now:
            return
        self._pending = {
            "w_pre": torch.stack([p.detach().flatten(1).clone() for p in self.w]),   # [2, C, D], the unfold order
            "gamma_pre": self.gamma.detach().clone(),
            "beta_pre": torch.stack([b.detach().clone() for b in self.betas]),
        }

    def _post_step(self, optimizer, args, kwargs) -> None:
        step = self._step
        self._step += 1
        if self._pending is None:
            self._captured_input = None
            return
        row = self._pending
        self._pending = None
        state = optimizer.state

        def flat(t):
            return t.detach().flatten(1).clone() if t.ndim > 1 else t.detach().clone()

        def grad_of(p):
            return flat(p.grad) if p.grad is not None else torch.full_like(flat(p), float("nan"))

        def buf_of(p):
            b = state.get(p, {}).get("momentum_buffer") if p in state else None
            return flat(b) if b is not None else torch.full_like(flat(p), float("nan"))

        row["w_post"] = torch.stack([flat(p) for p in self.w])
        row["g_w"] = torch.stack([grad_of(p) for p in self.w])
        row["buf_w"] = torch.stack([buf_of(p) for p in self.w])
        row["gamma_post"] = self.gamma.detach().clone()
        row["g_gamma"] = grad_of(self.gamma)
        row["buf_gamma"] = buf_of(self.gamma)
        row["beta_post"] = torch.stack([b.detach().clone() for b in self.betas])
        row["g_beta"] = torch.stack([grad_of(b) for b in self.betas])
        row["buf_beta"] = torch.stack([buf_of(b) for b in self.betas])
        row["batch_var"] = torch.stack([st.last_var.detach().clone() if st.last_var is not None
                                        else torch.full_like(self.gamma, float("nan")) for st in self.block.stats])
        groups = optimizer.param_groups
        gw = groups[self._group_of[id(self.w[0])]]
        row["lr_w"] = float(gw["lr"])
        row["wd_w"] = float(gw.get("weight_decay", 0.0))
        row["lr_gamma"] = float(groups[self._group_of[id(self.gamma)]]["lr"])
        beta_groups = [self._group_of.get(id(b)) for b in self.betas]
        row["lr_beta"] = float(groups[beta_groups[0]]["lr"]) if beta_groups[0] is not None else float("nan")
        row["step"] = step
        row["epoch"] = self._epoch
        if self.record_batch_sigma:
            if self._captured_input is None:
                row["sigma_batch"] = torch.full((self.w[0].shape[1] * 9,) * 2, float("nan"), dtype=torch.float64)
            else:
                conv = self.block.convs[0]
                row["sigma_batch"] = M.patch_covariance_of_batch(
                    self._captured_input, conv.kernel_size[0], conv.padding[0], conv.stride[0]).detach()
            self._captured_input = None
        self._rows.append(row)

    # -- per-epoch flush and scalars for W&B ------------------------------------

    def _flush(self, epoch: int) -> Path | None:
        if not self._rows:
            return None
        keys = list(self._rows[0].keys())
        arrays: dict[str, np.ndarray] = {}
        for k in keys:
            vals = [r[k] for r in self._rows]
            if isinstance(vals[0], torch.Tensor):
                arrays[k] = torch.stack(vals).cpu().numpy()      # the model's own dtype, unchanged
            else:
                arrays[k] = np.asarray(vals, dtype=np.float64 if isinstance(vals[0], float) else np.int64)
        path = self.save_dir / f"ep{epoch}.npz"
        np.savez(path, **arrays)
        self.files[epoch] = path
        self.steps_total_recorded += len(self._rows)
        n = len(self._rows)
        self._rows = []
        LOGGER.info("blockmeasure: epoch %d, %d steps -> %s (%.1f MB)", epoch, n, path, path.stat().st_size / 1e6)
        return path

    def _geometry_scalars(self, epoch: int) -> dict[str, float]:
        """The same numbers the offline pass will give for the epoch's end state,
        from ``measure.geometry`` on the current weights and the fixed Sigma."""
        st = M.BlockState(w1=self.w[0].detach().flatten(1).double().cpu().numpy(),
                          w2=self.w[1].detach().flatten(1).double().cpu().numpy(),
                          gamma=self.gamma.detach().double().cpu().numpy())
        g = M.geometry(st, self.sigma)
        op = M.open_mask(g)
        al = M.alignment(g)
        out = {"blockmeasure/n_open": float(op.sum()), "blockmeasure/n_parked": float(M.parked_mask(g).sum()),
               "blockmeasure/n_channels": float(op.size)}
        if op.any():
            out["blockmeasure/angle_open_p50"] = float(np.median(g.theta_deg[op]))
            out["blockmeasure/alignment_open_p50"] = float(np.median(al[op]))
            out["blockmeasure/lambda_plus_open_p50"] = float(np.median(g.lambda_plus[op]))
            out["blockmeasure/lambda_minus_open_p50"] = float(np.median(g.lambda_minus[op]))
        return out

    def epoch_stats(self, epoch: int) -> dict[str, float]:
        stats: dict[str, float] = {}
        if epoch in self.epochs:
            path = self._flush(epoch)
            stats["blockmeasure/epoch_recorded"] = float(epoch)
            stats["blockmeasure/steps_recorded_total"] = float(self.steps_total_recorded)
            if path is not None:
                stats["blockmeasure/mb_written"] = float(path.stat().st_size / 1e6)
        stats.update(self._geometry_scalars(epoch))
        self._epoch = epoch + 1
        return stats

    # -- manifest and artifact --------------------------------------------------

    def _write_manifest(self, final: bool) -> Path:
        manifest = dict(self.meta)
        manifest.update({
            "epochs_requested": self.epochs, "epochs_saved": sorted(self.files),
            "steps_recorded": self.steps_total_recorded, "record_batch_sigma": self.record_batch_sigma,
            "optimizer": {"momentum": self.momentum, "nesterov": self.nesterov, "dampening": self.dampening},
            "sigma": {"sha256": self.sigma.sha256, "settings": self.sigma.settings, "gap_ratio": self.sigma.gap_ratio},
            "sigma_check": None if self.sigma_check is None else {
                "sha256": self.sigma_check.sha256, "settings": self.sigma_check.settings,
                "gap_ratio": self.sigma_check.gap_ratio,
                "v_max_overlap_with_sigma": abs(float(self.sigma.v_max @ self.sigma_check.v_max))},
            "final": final,
        })
        if final:
            manifest["files"] = {p.name: {"bytes": p.stat().st_size, "sha256": _sha256(p)}
                                 for p in sorted(self.save_dir.glob("*.npz"))}
        path = self.save_dir / "manifest.json"
        path.write_text(json.dumps(manifest, indent=2, default=str))
        return path

    def _artifact_name(self) -> str:
        return _NAME_SANITIZER.sub("-", f"blockmeasure_{self.meta['group']}_{self.meta['variant']}_s{self.meta['seed']}")

    def close(self) -> None:
        for h in (self._h_pre, self._h_post, self._h_fwd):
            if h is not None:
                h.remove()
        if self._rows:   # a recorded epoch that never reached epoch_stats (interrupted run)
            self._flush(self._epoch)
        manifest_path = self._write_manifest(final=True)
        try:
            import wandb
        except ImportError:
            return
        if wandb.run is None:
            LOGGER.info("blockmeasure: W&B disabled; recording left at %s", self.save_dir)
            return
        run = wandb.run
        artifact = wandb.Artifact(name=self._artifact_name(), type="blockmeasure",
                                  metadata={"group": self.meta["group"], "variant": self.meta["variant"],
                                            "seed": self.meta["seed"], "epochs_saved": sorted(self.files)})
        files = sorted(self.save_dir.glob("*.npz")) + sorted(self.save_dir.glob("*.json"))
        for p in files:
            artifact.add_file(str(p))
        run.log_artifact(artifact)
        try:
            artifact.wait()
            entries = artifact.manifest.entries
            bad = [name for name, e in entries.items() if not (e.size or 0) > 0]
        except Exception as exc:  # noqa: BLE001
            run.summary["blockmeasure/validated"] = 0
            raise RuntimeError(f"blockmeasure artifact commit/validation errored: {exc}") from exc
        if len(entries) != len(files) or bad:
            run.summary["blockmeasure/validated"] = 0
            raise RuntimeError(f"blockmeasure artifact validation FAILED: expected {len(files)} files, "
                               f"committed {len(entries)}, zero-size entries: {bad}")
        run.summary["blockmeasure/validated"] = 1
        run.summary["blockmeasure/artifact_name"] = artifact.name
        run.summary["blockmeasure/steps_recorded"] = self.steps_total_recorded
        LOGGER.info("blockmeasure artifact validated: %s (%d files)", artifact.name, len(entries))


# ---------------------------------------------------------------------------
# Reading a recording back
# ---------------------------------------------------------------------------


@dataclass
class Recording:
    """A recording directory read back with the epochs concatenated in step order.

    ``arrays`` holds every per-step array (see the module docstring for names and
    shapes) as numpy, with the leading axis of length T over all recorded steps;
    ``epoch_of_row`` says which recorded epoch each row came from. ``consecutive``
    ``[T]`` is True where row t+1 is the optimizer step right after row t (same
    epoch, step + 1); differences across a gap are never dynamics.
    """

    directory: Path
    meta: dict[str, Any]
    sigma: M.CovarianceEstimate
    sigma_check: M.CovarianceEstimate | None
    arrays: dict[str, np.ndarray]
    consecutive: np.ndarray
    epochs: list[int] = field(default_factory=list)

    @property
    def n_steps(self) -> int:
        return int(self.arrays["step"].shape[0])

    @property
    def n_channels(self) -> int:
        return int(self.arrays["w_pre"].shape[2])

    @property
    def dim(self) -> int:
        return int(self.arrays["w_pre"].shape[3])

    def block_state(self, which: str = "pre") -> M.BlockState:
        """The kernels and gamma of every recorded step as one batched ``BlockState``
        (float64), ``which`` being "pre" or "post"."""
        w = self.arrays[f"w_{which}"].astype(np.float64)
        b = self.arrays[f"beta_{which}"].astype(np.float64)
        return M.BlockState(w1=w[:, 0], w2=w[:, 1], gamma=self.arrays[f"gamma_{which}"].astype(np.float64),
                            beta1=b[:, 0], beta2=b[:, 1])


def load_recording(directory: str | Path) -> Recording:
    """Read a recording directory. Verifies each file's hash against the manifest
    when the manifest is final, concatenates the epoch files in epoch order, and
    marks which consecutive rows are consecutive optimizer steps."""
    d = Path(directory)
    meta = json.loads((d / "manifest.json").read_text())
    if meta.get("final") and "files" in meta:
        for name, info in meta["files"].items():
            p = d / name
            if not p.exists():
                raise FileNotFoundError(f"{p} listed in the manifest is missing")
            if _sha256(p) != info["sha256"]:
                raise ValueError(f"{p} does not match its manifest hash")
    sigma = M.CovarianceEstimate.load(d / "sigma")
    sigma_check = M.CovarianceEstimate.load(d / "sigma_check") if (d / "sigma_check.npz").exists() else None
    files = sorted(d.glob("ep*.npz"), key=lambda p: int(p.stem[2:]))
    if not files:
        raise FileNotFoundError(f"no ep*.npz in {d}")
    parts = [dict(np.load(p)) for p in files]
    arrays = {k: np.concatenate([p[k] for p in parts], axis=0) for k in parts[0]}
    step, epoch = arrays["step"], arrays["epoch"]
    consecutive = np.zeros(step.shape[0], dtype=bool)
    consecutive[:-1] = (np.diff(step) == 1) & (epoch[1:] == epoch[:-1])
    return Recording(directory=d, meta=meta, sigma=sigma, sigma_check=sigma_check, arrays=arrays,
                     consecutive=consecutive, epochs=[int(p.stem[2:]) for p in files])


# ---------------------------------------------------------------------------
# Registration through the optimizer target (no shared module edited)
# ---------------------------------------------------------------------------

from structural_reparam.agents.stage2 import lab as _stage2_lab  # noqa: E402  (also registers pair_channel_open)


def build_checked_kernel_wd_sgd(params, model, lr, **kwargs):
    """Stage 2's ``build_checked_kernel_wd_sgd`` (the paper's optimizer with the
    decay-group gate), made safe to combine with the recorder's step hooks.

    Naming this function as a config's optimizer target imports this module, and
    with it the ``@register_probe`` above, before the trainer reads the ``probes:``
    block; that is how the probe is registered without editing any shared module.

    The one change: stage 2's builder registers a one-shot post-step hook that
    writes the group sizes to the W&B summary and then removes itself from inside
    the hook. torch iterates its hook dictionary while calling the hooks, so a hook
    that deletes itself is safe only when it is the last one in the dictionary,
    which it always was in stage 2's runs. The recorder registers its hooks after
    it, and the deletion then raises "OrderedDict mutated during iteration" at the
    first step (seen on 2026-09-02). Here every hook the builder registered is
    re-registered behind a wrapper that calls it once and afterwards does nothing,
    so nothing is ever deleted during iteration and the summary is still written.
    """
    optimizer = _stage2_lab.build_checked_kernel_wd_sgd(params, model=model, lr=lr, **kwargs)
    existing = list(optimizer._optimizer_step_post_hooks.values())
    optimizer._optimizer_step_post_hooks.clear()
    for hook in existing:
        state = {"done": False}

        def run_once(opt, args, kw, _hook=hook, _state=state):
            if _state["done"]:
                return
            _state["done"] = True
            _hook(opt, args, kw)   # its own handle.remove() is now a no-op

        optimizer.register_step_post_hook(run_once)
    return optimizer
