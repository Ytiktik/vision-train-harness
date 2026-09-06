"""Stage 2's fork of ``PairChannelProbe``.

Forked rather than edited in place: `reparam_pinned_norm` is a read-only
model-class dependency for this campaign. It is a *subclass*, not a copy, so the
measurement code cannot drift from the original -- only the four things this
campaign needs are overridden.

What differs from the original, and why:

1. ``cov_blocks`` defaults to ``[0]`` upstream, so every deeper block logs a NaN
   Sigma-whitened cosine. Here it covers **every block**. Stage 2 is about the
   last block, so a probe that only whitens block 0 measures the wrong end of the
   network.

2. Upstream estimates Sigma **once, at construction**. Block 0's input is the data
   and is stationary, so one estimate would do there, but a deeper block's
   post-ReLU input drifts as training proceeds and a stale Sigma silently
   mis-whitens it. Here Sigma is re-estimated on an epoch schedule.

3. Upstream's ``_input_cov`` spins up its own loader and runs a **full model
   forward per batch per block index**, so covering B blocks costs B passes. Here
   one sweep captures every block's input patches at once through pre-hooks on all
   blocks.

4. **Open-channel angle statistics.** Not every channel's pair opens. In a block
   where most channels have collapsed, a median over all channels is dominated by
   the closed ones and reads as "closed" even when the channels that did open sit
   at a wide angle. So the angle statistics are computed over the open channels
   only, and the open count is logged beside them -- a median of 150 degrees over
   4 open channels of 64 is a different claim from the same median over 60 of 64.
   Minimum, maximum, median and mean are all reported, not the median alone.

The openness threshold is a choice, not a fact, so it is a constructor argument,
it is logged into the run summary, and it must appear in every caption. The
inherited ``_save`` already writes the full per-channel cosine array for every
epoch into the run's artifact, so the threshold can be changed offline without
re-running anything.
"""

from __future__ import annotations

import logging

import torch

from structural_reparam.analysis.registry import ProbeContext, register_probe
from structural_reparam.experiments.reparam_pinned_norm.lab import PairChannelProbe

LOGGER = logging.getLogger(__name__)

OPEN_ANGLE_DEG = 10.0          # a channel is "open" at >= this branch angle


def _deg(cos: torch.Tensor) -> torch.Tensor:
    return torch.rad2deg(torch.arccos(cos.clamp(-1.0, 1.0)))


def _open_stats(cos: torch.Tensor, prefix: str, open_cos_max: float) -> dict:
    """Angle statistics over the open channels only, plus the open count.

    ``cos`` is one cosine per output channel. A channel is open when its angle is
    at least the configured threshold, i.e. its cosine is at or below
    ``open_cos_max``. NaN channels (no covariance available) are excluded from
    both the count and the statistics.
    """
    out: dict[str, float] = {}
    finite = cos[~torch.isnan(cos)]
    total = int(finite.numel())
    out[prefix + "n_total"] = float(total)
    if total == 0:
        return out
    mask = finite <= open_cos_max
    n_open = int(mask.sum())
    out[prefix + "n_open"] = float(n_open)
    out[prefix + "frac_open"] = float(n_open) / total
    if n_open == 0:
        return out
    ang = _deg(finite[mask])
    out[prefix + "angle_open_min"] = float(ang.min())
    out[prefix + "angle_open_max"] = float(ang.max())
    out[prefix + "angle_open_median"] = float(ang.median())
    out[prefix + "angle_open_mean"] = float(ang.mean())
    return out


@register_probe("pair_channel_open")
class OpenChannelPairProbe(PairChannelProbe):
    PROBE_NAME = "pair_channel_open"

    def __init__(self, model, out_path, variant, seed, group,
                 step_log_epochs: int = 3, step_every: int = 20,
                 input_cov_batches: int = 20, dataset_cfg=None,
                 cov_every_epochs: int = 1, open_angle_deg: float = OPEN_ANGLE_DEG,
                 **_ignored) -> None:
        self._cov_batches = int(input_cov_batches)
        self._cov_every = max(1, int(cov_every_epochs))
        self._dataset_cfg = dataset_cfg
        self.open_angle_deg = float(open_angle_deg)
        self.open_cos_max = float(torch.cos(torch.deg2rad(torch.tensor(self.open_angle_deg))))

        # cov_blocks=() so the parent skips its per-block, one-pass-each estimate;
        # the single sweep below replaces it and covers every block.
        super().__init__(model, out_path, variant, seed, group,
                         step_log_epochs=step_log_epochs, step_every=step_every,
                         cov_blocks=(), input_cov_batches=input_cov_batches,
                         dataset_cfg=dataset_cfg)

        self._refresh_cov()
        # The parent's epoch-0 record was taken with no covariance, so its
        # whitened cosine is NaN. Retake it now that Sigma exists.
        self.epoch_records.clear()
        self.step_records.clear()
        self._record(epoch=0, step=0, kind="epoch")
        self._log_epoch0()

    @classmethod
    def from_context(cls, ctx: ProbeContext) -> "OpenChannelPairProbe":
        variant = ctx.variant.get("name", "variant")
        experiment = ctx.config.get("experiment", {}).get("name", "experiment")
        group = ctx.config.get("logging", {}).get("group", experiment)
        out_dir = ctx.output_dir / "pair_channel_open"
        out_dir.mkdir(parents=True, exist_ok=True)
        pc = ctx.probe_config or {}
        return cls(ctx.model, out_dir / f"{variant}_seed{ctx.seed}.npz", variant, ctx.seed,
                   group,
                   step_log_epochs=pc.get("step_log_epochs", 3),
                   step_every=pc.get("step_every", 20),
                   input_cov_batches=pc.get("input_cov_batches", 20),
                   cov_every_epochs=pc.get("cov_every_epochs", 1),
                   open_angle_deg=pc.get("open_angle_deg", OPEN_ANGLE_DEG),
                   dataset_cfg=ctx.config.get("dataset"))

    # -- one sweep, every block ----------------------------------------------
    @torch.no_grad()
    def _refresh_cov(self) -> None:
        """Estimate the 3x3 input-patch covariance of EVERY block in one pass.

        Pre-hooks on all blocks capture each block's input during a single forward,
        so the cost is one sweep of ``input_cov_batches`` batches regardless of how
        many blocks there are, instead of one sweep per block.
        """
        cfg = self._dataset_cfg
        if cfg is None:
            LOGGER.warning("pair_channel_open: no dataset config; Sigma unavailable")
            return
        try:
            from structural_reparam.deploy.train import import_target

            args = dict(cfg.get("args", {}))
            args["num_workers"] = 0
            args["persistent_workers"] = False
            # The full-resolution ImageNet loader takes `warm`, which reads all
            # 143 GiB of record shards sequentially to fill the page cache. That is
            # worth doing once per run, in the trainer's own loader, and never
            # again: a covariance sweep that re-warmed on every refresh would spend
            # more time reading the dataset than training does. Only clear the flag
            # if this dataset has one, so the CIFAR builders keep their signatures.
            if "warm" in args:
                args["warm"] = False
            loaders = import_target(cfg["target"])(**args)
            loader = loaders[0] if isinstance(loaders, (tuple, list)) else loaders
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("pair_channel_open: loader for Sigma failed: %r", exc)
            return

        dev = next(self.model.parameters()).device
        captured: dict[int, torch.Tensor] = {}
        handles = []
        for bi, _, blk in self.blocks:
            def pre(mod, inp, _bi=bi):
                captured[_bi] = inp[0].detach()
            handles.append(blk.register_forward_pre_hook(pre))

        was_training = self.model.training
        self.model.eval()
        acc: dict[int, torch.Tensor] = {}
        n = 0
        try:
            for k, batch in enumerate(loader):
                if k >= self._cov_batches:
                    break
                captured.clear()
                self.model(batch[0].to(dev))
                for bi, _, blk in self.blocks:
                    xin = captured.get(bi)
                    if xin is None:
                        continue
                    p = torch.nn.functional.unfold(xin, 3, padding=1, stride=blk.stride)
                    p = p.transpose(1, 2).reshape(-1, p.shape[1])
                    p = p - p.mean(0, keepdim=True)
                    c = p.T @ p / p.shape[0]
                    acc[bi] = c if bi not in acc else acc[bi] + c
                n += 1
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("pair_channel_open: Sigma sweep failed: %r", exc)
        finally:
            for h in handles:
                h.remove()
            self.model.train(was_training)

        if n:
            self.cov = {bi: (c / n).cpu() for bi, c in acc.items()}
            # The parent builds Sigma^(1/2) and the top eigenvector (its
            # ``_cov_aux``) once, from ``cov_blocks`` at construction, and its
            # snapshot indexes ``_cov_aux[bi]`` for every paired block that has a
            # covariance. This sweep replaces ``self.cov`` on every refresh, so the
            # aux must be rebuilt here too, or a paired block raises KeyError at
            # the epoch-0 record (radial_boost_2x2 pair runs, 2026-08-31). Only
            # paired blocks consume it, so single-branch blocks are skipped.
            self._cov_aux = {}
            self._cov_spec = {}
            for bi, name, blk in self.blocks:
                if bi not in self.cov:
                    continue
                paired = len(getattr(blk, "convs", ())) >= 2
                # The spectrum is recorded for every paired block, whose
                # eigendecomposition is computed here anyway, and for block 0 in
                # every arm, whose covariance is the input data's and is 27x27 at a
                # 3x3 stem, so the extra decomposition costs nothing. A deep
                # unpaired block is skipped: its covariance can be thousands of
                # dimensions and nothing reads its spectrum.
                if not paired and bi != 0:
                    continue
                ev, V = torch.linalg.eigh(self.cov[bi].double())
                lam, V = ev.flip(0), V.flip(1)
                # The selection predictor of the paper's section 3.6,
                # mu_eff = (lambda_max - lambda_next) / tr Sigma: the loud
                # direction's margin over its runner-up as a share of the input
                # variance. Recorded beside the runner-up ratio, which section
                # 4.2.5 uses for the resting angle, so both predictions can be read
                # off the run without re-estimating the covariance offline.
                trace = float(lam.clamp_min(0).sum())
                self._cov_spec[bi] = (float(lam[0]), float(lam[1]) if lam.numel() > 1
                                      else float("nan"), trace)
                if not paired:
                    continue
                sh = (V @ torch.diag(lam.clamp_min(0).sqrt()) @ V.T).float()
                self._cov_aux[bi] = (sh, V[:, 0].float())

    # -- statistics -----------------------------------------------------------
    def _scalars(self, snap: dict) -> dict:
        s = super()._scalars(snap)
        s["pair_channel_open/open_angle_deg"] = self.open_angle_deg
        for bi, (lam_max, lam_next, trace) in getattr(self, "_cov_spec", {}).items():
            p = f"pair_channel/block{bi}/"
            s[p + "lambda_max"] = lam_max
            s[p + "lambda_next"] = lam_next
            s[p + "trace_sigma"] = trace
            if trace > 0:
                s[p + "mu_eff"] = (lam_max - lam_next) / trace
            if lam_next > 0:
                s[p + "runner_up_ratio"] = lam_max / lam_next
        for bi, d in snap.items():
            if int(d["nb"]) < 2:
                continue
            p = f"pair_channel/block{bi}/"
            s.update(_open_stats(d["cos"], p + "euclid_", self.open_cos_max))
            s.update(_open_stats(d["cos_sigma"], p + "whitened_", self.open_cos_max))
        return s

    def epoch_stats(self, epoch: int) -> dict:
        if int(epoch) % self._cov_every == 0:
            self._refresh_cov()
        return super().epoch_stats(epoch)
