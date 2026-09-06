"""Weight-norm pair experiment entry: the split-rescue trainer plus a per-epoch
``wn_pair`` probe for the preconditioning-hypothesis bookkeeping.

Block (``bn_position="weight_norm"``): ``out_c = gamma_c * (1/N) sum_i (w_{i,c}/||w_{i,c}||) * x + b_c``.
The note's conventions map onto it directly (u = gamma/2 (w1_hat + w2_hat) x):
    sigma_i   = ||w_{i,c}||                       (exact, no input covariance)
    gamma_note = gamma_c
    kappa     = gamma^2 / sigma^2,   x = kappa/2,   sigma^2 = mean_i ||w_i||^2
Single-branch blocks (N=1) get gamma, ||w1||, kappa only (no angle, no Q).
    Q_theta   = sin^2(theta/2) * exp(x)           (14a form; theta = angle(w1, w2))
    Q_w       = ||w_-||^2 * exp(x),  w_- = (w1 - w2)/2   (14b form)
    Q_pure    = sin^2(theta/2) * exp(gamma_note^2 / (2 sigma_0^2)),  sigma_0 = ||w|| at init
              (the flow invariant with the kernel norm frozen: gamma is decay-free and
              ||w_i|| is a pure gauge of this block, so sigma(t) shrinking under kernel
              decay is bookkeeping, not dynamics; the exponent moves only through gamma)
Per epoch and per two-branch block the probe logs channel median / p10 / p90 of
ln Q_theta, ln Q_w, cos theta, gamma_c, ||w1||, ||w2||, kappa, and the median
per-channel change of ln Q since init; the full per-channel arrays for every
epoch (incl. epoch 0 = init) go to ``<output_dir>/wn_pair/<variant>_seed<seed>.npz``.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from structural_reparam.analysis.registry import ProbeContext, register_probe
from structural_reparam.experiments.reparam_sweeps100.lab import ClaudeRepVGGBlock
from structural_reparam.experiments.reparam_split_rescue_cifar import lab as _rescue_lab

QUANTS = ("lnQ_theta", "lnQ_w", "lnQ_pure", "cos", "gamma", "bias", "w1_norm", "w2_norm", "kappa",
          "geff", "ang_step", "ang_init")
# geff     = effective scale of the block's filter: gamma (single) / gamma*cos(theta/2) (pair)
# ang_step = angle (rad) the EFFECTIVE direction (w_hat single; normalize(w1_hat+w2_hat) pair)
#            moved since the previous record (per epoch); ang_init = angle from its init direction


@register_probe("wn_pair")
class WNPairProbe:
    def __init__(self, model: nn.Module, save_path: Path | None) -> None:
        self.blocks = [
            m for m in model.modules()
            if isinstance(m, ClaudeRepVGGBlock)
            and m.bn_position == "weight_norm" and m.num_3x3 in (1, 2)
        ]
        self.save_path = save_path
        self.epochs: list[int] = []
        self.hist: dict[str, list[np.ndarray]] = {}   # "block{i}/quant" -> [epoch arrays]
        self.init_lnq: dict[int, np.ndarray] = {}
        self.init_sigma2: dict[int, np.ndarray] = {}
        self.init_dir: dict[int, torch.Tensor] = {}
        self.prev_dir: dict[int, torch.Tensor] = {}
        self._record(0)  # epoch 0 = init

    @classmethod
    def from_context(cls, ctx: ProbeContext) -> "WNPairProbe":
        variant_name = ctx.variant.get("name", "variant")
        d = ctx.output_dir / "wn_pair"
        d.mkdir(parents=True, exist_ok=True)
        return cls(model=ctx.model, save_path=d / f"{variant_name}_seed{ctx.seed}.npz")

    @torch.no_grad()
    def _block_quants(self, block: ClaudeRepVGGBlock, i: int) -> dict[str, np.ndarray]:
        ws = [w.detach().float().flatten(1) for w in block._conv3_weights()]
        gamma = block.wn_gamma.detach().float() * float(getattr(block, "wn_gamma_k", 1.0))
        bias = block.wn_bias.detach().float()
        if len(ws) == 1:
            n1 = ws[0].norm(dim=1)
            kappa = gamma ** 2 / (n1 ** 2).clamp_min(1e-12)
            d = ws[0] / n1.clamp_min(1e-12).unsqueeze(1)
            out = {"gamma": gamma, "bias": bias, "w1_norm": n1, "kappa": kappa, "geff": gamma.abs()}
            out.update(self._angles(i, d))
            return {k: v.cpu().numpy().astype(np.float64) for k, v in out.items()}
        w1, w2 = ws
        n1, n2 = w1.norm(dim=1), w2.norm(dim=1)
        cos = (w1 * w2).sum(1) / (n1 * n2).clamp_min(1e-12)
        sigma2 = 0.5 * (n1 ** 2 + n2 ** 2)
        if i not in self.init_sigma2:
            self.init_sigma2[i] = sigma2.clone()
        kappa = gamma ** 2 / sigma2.clamp_min(1e-12)
        x = 0.5 * kappa
        x_pure = 0.5 * gamma ** 2 / self.init_sigma2[i].clamp_min(1e-12)
        sin2_half = (0.5 * (1.0 - cos)).clamp_min(1e-30)
        wminus2 = 0.25 * (w1 - w2).pow(2).sum(1).clamp_min(1e-30)
        u1 = w1 / n1.clamp_min(1e-12).unsqueeze(1); u2 = w2 / n2.clamp_min(1e-12).unsqueeze(1)
        s = u1 + u2; sn = s.norm(dim=1)
        d = s / sn.clamp_min(1e-12).unsqueeze(1)
        cos_half = torch.sqrt((0.5 * (1.0 + cos)).clamp_min(0.0))
        out = {
            "lnQ_theta": torch.log(sin2_half) + x,
            "lnQ_w": torch.log(wminus2) + x,
            "lnQ_pure": torch.log(sin2_half) + x_pure,
            "cos": cos, "gamma": gamma, "bias": bias,
            "w1_norm": n1, "w2_norm": n2, "kappa": kappa,
            "geff": gamma.abs() * cos_half,
        }
        ang = self._angles(i, d)
        # effective direction undefined when the pair is (anti-)collapsed to ||u1+u2|| ~ 0
        bad = sn < 1e-3
        for k in ang: ang[k] = torch.where(bad, torch.full_like(ang[k], float("nan")), ang[k])
        out.update(ang)
        return {k: v.cpu().numpy().astype(np.float64) for k, v in out.items()}

    def _angles(self, i: int, d: torch.Tensor) -> dict[str, torch.Tensor]:
        """Angular step since the previous record and displacement from init, per channel."""
        d = d.detach()
        if i not in self.init_dir:
            self.init_dir[i] = d.clone(); self.prev_dir[i] = d.clone()
            z = torch.zeros(d.shape[0], device=d.device)
            return {"ang_step": z, "ang_init": z.clone()}
        step = torch.acos(((d * self.prev_dir[i]).sum(1)).clamp(-1.0, 1.0))
        init = torch.acos(((d * self.init_dir[i]).sum(1)).clamp(-1.0, 1.0))
        self.prev_dir[i] = d.clone()
        return {"ang_step": step, "ang_init": init}

    def _record(self, epoch: int) -> dict[str, float]:
        stats: dict[str, float] = {}
        self.epochs.append(epoch)
        for i, block in enumerate(self.blocks):
            q = self._block_quants(block, i)
            for k, arr in q.items():
                self.hist.setdefault(f"block{i}/{k}", []).append(arr)
            if "cos" not in q:  # single-branch block
                for k in ("gamma", "bias", "w1_norm", "kappa", "geff", "ang_step", "ang_init"):
                    stats[f"wn_pair/block{i}/{k}_med"] = float(np.median(q[k]))
                    stats[f"wn_pair/block{i}/{k}_p10"] = float(np.percentile(q[k], 10))
                    stats[f"wn_pair/block{i}/{k}_p90"] = float(np.percentile(q[k], 90))
                continue
            if i not in self.init_lnq:
                self.init_lnq[i] = {k: q[k].copy() for k in ("lnQ_theta", "lnQ_pure")}
            # tied branches: cos == 1 -> ln Q = -inf-ish; report as nan
            open_mask = q["cos"] < 1.0 - 1e-6
            for k in QUANTS:
                arr = q[k]
                if k.startswith("lnQ"):
                    arr = np.where(open_mask, arr, np.nan)
                if np.all(np.isnan(arr)):
                    continue
                stats[f"wn_pair/block{i}/{k}_med"] = float(np.nanmedian(arr))
                stats[f"wn_pair/block{i}/{k}_p10"] = float(np.nanpercentile(arr, 10))
                stats[f"wn_pair/block{i}/{k}_p90"] = float(np.nanpercentile(arr, 90))
            for k in ("lnQ_theta", "lnQ_pure"):
                dlnq = np.where(open_mask, q[k] - self.init_lnq[i][k], np.nan)
                if not np.all(np.isnan(dlnq)):
                    stats[f"wn_pair/block{i}/d{k}_med"] = float(np.nanmedian(dlnq))
                    stats[f"wn_pair/block{i}/d{k}_p10"] = float(np.nanpercentile(dlnq, 10))
                    stats[f"wn_pair/block{i}/d{k}_p90"] = float(np.nanpercentile(dlnq, 90))
            stats[f"wn_pair/block{i}/frac_open"] = float(open_mask.mean())
        return stats

    def epoch_stats(self, epoch: int = 0) -> dict[str, float]:
        stats = self._record(epoch)
        self._save()
        return stats

    def _save(self) -> None:
        if self.save_path is None:
            return
        arrays = {k: np.stack(v) for k, v in self.hist.items()}   # [n_epochs, channels]
        arrays["epochs"] = np.asarray(self.epochs)
        np.savez(self.save_path, **arrays)

    def close(self) -> None:
        self._save()


def main() -> None:
    _rescue_lab.main()


if __name__ == "__main__":
    main()
