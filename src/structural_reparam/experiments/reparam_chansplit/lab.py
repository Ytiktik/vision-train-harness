"""reparam_chansplit — per-CHANNEL warm split of block 0 of the single-branch d3
CIFAR-100 cell (constant lr 0.1, conv-only WD 5e-4), √2 gauge.

Test of the "gamma floor / preconditioning" story at channel resolution: at a
chosen epoch of the single-branch run, a SELECTED SUBSET of block-0 channels is
re-parameterised as an open two-branch pair (branch angle 90° in the input
metric) that is exactly function- AND learning-rate-neutral at the split
instant; the remaining channels stay single. Arms differ only in WHICH channels
are paired (lowest-|gamma| "down" set, highest-|gamma| "up" set, random set,
all, none). The none arm continued from the same checkpoint with the same data
order is the paired control.

Split of channel c (kernel w, BN gamma, beta; input-patch covariance Sigma):
    D  ⟂_Sigma w,  Dᵀ Sigma D = wᵀ Sigma w / 4          (90° in the Sigma metric)
    w1 = √2 (w/2 + D),  w2 = √2 (w/2 − D)                (√2 gauge)
    out_c = gamma_c · ((w1x − mu1) + (w2x − mu2)) / (√2 · rho_c · sigma_bar) + beta_c
    sigma_bar² = (sigma1² + sigma2²)/2,  rho_c = measured sigma_bar/sigma_w at split (≈1)
so that at the split  out_c == gamma_c (wx − mu)/sigma_w + beta_c  exactly, each kernel
copy has sensitivity 1/√2 (k·s² = 2·½ = 1), gamma has sensitivity 1, beta unchanged.
Momentum buffers are carried over (kernel copies get m_w/√2). Everything after the
split is angle dynamics: the pair's rate multipliers become 1/cos²(θ/2) (direction)
and cos²(θ/2) + (κ/2) sin²(θ/2) (scale) as θ moves.

Three sub-commands:
  make-ckpt : train the single from its epoch-0 checkpoint for --split-epoch epochs
              at constant lr and save model + optimizer state (the split source).
  run       : build an arm from that state and continue to --epochs; logs per-epoch
              train/test accuracy and block-0 per-channel gamma / beta / norms /
              branch angle; npz + W&B (group chansplit_c100_d3_wdconv_flat).
  dry-split : build the split only (no training) and print the per-channel
              feasibility / neutrality diagnostics.

Generalised split (2026-08-17, --theta-deg / --norm-mult; the original construction
is theta 90 with D only Sigma-orthogonal and its Euclidean norm unconstrained):
    D ⟂ w  AND  D ⟂_Sigma w,   Dᵀ Sigma D = tan²(θ₀/2) · wᵀ Sigma w / 4   (branch angle θ₀ in Sigma)
    ‖D‖² = ‖w‖² (r² − ½) / 2  so that  ‖w1‖ = ‖w2‖ = r ‖w‖          (Euclidean branch norm)
    out_c = gamma_c · (z1 + z2) / (2 cos(θ₀/2)) + beta_c   [branch_sigma block; = /√2 at 90°]
which is function-neutral and gamma-LR-neutral at ANY θ₀, gives each kernel copy
sensitivity 1/√2 at any θ₀, and with r = 1 leaves the Euclidean branch norms equal
to the parent's (k·s² = 1, and no weight-decay norm transient when the parent is
already at its WD equilibrium). Feasible iff the required Rayleigh quotient
tan²(θ₀/2)·(wᵀΣw/‖w‖²)/(r²−½) lies in the spectrum of Sigma restricted to the
doubly-orthogonal subspace (checked per channel; --strict-feasible aborts otherwise).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "src")
from structural_reparam.experiments.reparam_sweeps100.lab import LayerwiseRepVGGCifar  # noqa: E402
from structural_reparam.agents.reparam_split_wd0.lab import build_conv_wd_sgd  # noqa: E402
from structural_reparam.data.cifar100 import build_loaders  # noqa: E402

BLOCK = "stages.0.0"
WANDB_PROJECT = "claude-autonomous-reparam"
WANDB_GROUP = "chansplit_c100_d3_wdconv_flat"
ENTITY = "yoovi-t-tel-aviv-university"


def build_single(sd=None):
    mdl = LayerwiseRepVGGCifar(num_classes=100, width_mult=1.0, stage_channels=[64, 128, 256],
                               stage_blocks=[1, 1, 1], stage_strides=[1, 2, 2], num_3x3=1, norm="batch",
                               use_1x1=False, use_identity=False, bn_position="per_branch")
    if sd is not None:
        mdl.load_state_dict(sd, strict=True)
    return mdl


def fetch_ckpt(spec: str, epoch: int) -> str:
    if os.path.exists(spec):
        return spec
    import wandb
    art = wandb.Api(timeout=120).artifact(spec, type="checkpoint")
    d = art.download()
    fp = os.path.join(d, f"ckpt_ep{epoch}.pt")
    if not os.path.exists(fp):
        raise FileNotFoundError(f"{fp} (artifact dir has {os.listdir(d)})")
    return fp


# --------------------------------------------------------------------------- block
class ChanSplitBlock0(nn.Module):
    """Wraps block 0's (conv, bn, relu). Channels in ``sel`` are computed by the
    √2-gauge shared-gamma/shared-sigma pair (w1, w2); all others by the original
    single path. With ``sel`` empty this is exactly the original block."""

    def __init__(self, conv: nn.Conv2d, bn: nn.BatchNorm2d, sel: list[int],
                 w1: torch.Tensor | None = None, w2: torch.Tensor | None = None,
                 rho: torch.Tensor | None = None, momentum: float = 0.1, block: str = "shared_sigma",
                 theta_deg: float = 90.0):
        super().__init__()
        if block not in ("shared_sigma", "branch_sigma"):
            raise ValueError(block)
        self.block_mode = block  # shared_sigma: gamma*(c1+c2)/(sqrt2*rho*sigma_bar); branch_sigma: gamma*(z1+z2)*mix
        self.theta_deg = float(theta_deg)
        # branch_sigma mixing constant 1/(2 cos(theta0/2)): function-neutral at the split angle theta0 (1/sqrt2 at 90 deg)
        self.mix = 1.0 / (2.0 * math.cos(math.radians(theta_deg) / 2))
        self.conv = conv
        self.bn = bn
        self.eps = bn.eps
        self.momentum = momentum
        self.register_buffer("sel", torch.tensor(sorted(sel), dtype=torch.long))
        k = len(sel)
        if k:
            self.w1 = nn.Parameter(w1.clone())
            self.w2 = nn.Parameter(w2.clone())
            self.register_buffer("rho", rho.clone())
            self.register_buffer("rm1", torch.zeros(k)); self.register_buffer("rv1", torch.ones(k))
            self.register_buffer("rm2", torch.zeros(k)); self.register_buffer("rv2", torch.ones(k))
        self.last = {}

    @property
    def n_pair(self) -> int:
        return int(self.sel.numel())

    def _branch(self, x, w, rm, rv):
        a = F.conv2d(x, w, None, self.conv.stride, self.conv.padding)
        if self.training:
            mu = a.mean((0, 2, 3)); var = a.var((0, 2, 3), unbiased=False)
            with torch.no_grad():
                n = a.numel() / a.shape[1]
                rm.mul_(1 - self.momentum).add_(self.momentum * mu)
                rv.mul_(1 - self.momentum).add_(self.momentum * var * n / max(n - 1, 1))
        else:
            mu, var = rm, rv
        return a - mu[None, :, None, None], var, mu

    def forward(self, x):
        y = self.bn(self.conv(x))
        if self.n_pair:
            c1, v1, mu1 = self._branch(x, self.w1, self.rm1, self.rv1)
            c2, v2, mu2 = self._branch(x, self.w2, self.rm2, self.rv2)
            g = self.bn.weight[self.sel]; b = self.bn.bias[self.sel]
            if self.block_mode == "shared_sigma":
                sbar = torch.sqrt((v1 + v2) / 2 + self.eps)
                mix = (c1 + c2) / (math.sqrt(2) * self.rho * sbar)[None, :, None, None]
            else:  # branch_sigma: each branch BN-normalised, sum / sqrt2 (per-branch sigma free)
                z1 = c1 / torch.sqrt(v1 + self.eps)[None, :, None, None]
                z2 = c2 / torch.sqrt(v2 + self.eps)[None, :, None, None]
                mix = (z1 + z2) * self.mix
            out = g[None, :, None, None] * mix + b[None, :, None, None]
            y = y.index_copy(1, self.sel, out)
            if self.training:
                self.last = {"v1": v1.detach(), "v2": v2.detach()}
        return F.relu(y)


def install_block(model: nn.Module, block: ChanSplitBlock0) -> None:
    """Replace ``stages.0.0`` by ``block`` (module attribute path)."""
    parent = model.stages[0]
    parent[0] = block


# --------------------------------------------------------------------- statistics
@torch.no_grad()
def input_patch_cov(loader, n_batches: int, dev) -> tuple[torch.Tensor, torch.Tensor]:
    """Covariance (27x27) of 3x3 image patches (padding 1) — the metric BN'd conv
    outputs actually see (centered, since BN subtracts the channel mean). Returns
    (Sigma, mean)."""
    s = None; m = None; n = 0
    for i, (x, _) in enumerate(loader):
        if i >= n_batches:
            break
        p = F.unfold(x.to(dev), 3, padding=1)  # B x 27 x L
        p = p.transpose(1, 2).reshape(-1, p.shape[1]).double()
        s = p.T @ p if s is None else s + p.T @ p
        m = p.sum(0) if m is None else m + p.sum(0)
        n += p.shape[0]
    mean = m / n
    return (s / n - torch.outer(mean, mean)).float(), mean.float()


def make_split(W: torch.Tensor, sel: list[int], Sigma: torch.Tensor, gen: torch.Generator,
               theta_deg: float | None = None, norm_mult: float | None = None, norm_auto: bool = False,
               auto_margin: float = 1.02, gauge_norm_mult: float | None = None):
    """W: (C, in, 3, 3). Returns w1, w2 (k, in, 3, 3) = √2 (w/2 ± D).

    Legacy (theta_deg is None): D ⟂_Sigma w, Dᵀ Sigma D = wᵀ Sigma w / 4 (90° in the input
    metric), Euclidean norm of D unconstrained (random direction).

    Generalised (theta_deg given): D ⟂ w and D ⟂_Sigma w, Dᵀ Sigma D = tan²(θ/2) wᵀ Sigma w / 4
    (branch angle θ in Sigma) and ‖D‖² = ‖w‖² (r² − ½)/2 so ‖w1‖ = ‖w2‖ = r ‖w‖ (r = norm_mult).
    D is a random unit vector of the doubly-orthogonal subspace mixed with that subspace's
    top (or bottom) Sigma-eigenvector so that its Rayleigh quotient hits the required value.
    norm_auto: per channel, raise r_c above norm_mult to the smallest value that makes the
    angle feasible, r_c² = ½ + auto_margin · tan²(θ/2) (wᵀΣw/‖w‖²) / λ_max(Σ|doubly-orthogonal);
    channels feasible at the floor keep r_c = norm_mult.
    gauge_norm_mult: if given, replace the √2 prefactor by a per-channel c so that the branch
    Euclidean norm is exactly gauge_norm_mult · ‖w‖ (function-inert under per-branch BN; the
    D geometry — angle, Rayleigh quotient — is unchanged). E.g. 1/√2 with the r=1 D geometry
    gives c = 1, i.e. branches at the pair's observed weight-decay equilibrium norm.
    Returns (w1, w2, diag) in the generalised case, diag a per-channel dict of feasibility."""
    C = W.shape[0]; shp = W.shape[1:]
    Wf = W.flatten(1).double(); S = Sigma.double()
    w1 = []; w2 = []
    if theta_deg is None:
        for c in sel:
            w = Wf[c]
            z = torch.randn(w.numel(), generator=gen, dtype=torch.float64)
            Sw = S @ w
            d0 = z - (z @ Sw) / (w @ Sw) * w            # Sigma-orthogonal to w
            d = d0 * torch.sqrt((w @ Sw) / 4 / (d0 @ S @ d0))
            w1.append(math.sqrt(2) * (w / 2 + d)); w2.append(math.sqrt(2) * (w / 2 - d))
        w1 = torch.stack(w1).float().reshape(len(sel), *shp); w2 = torch.stack(w2).float().reshape(len(sel), *shp)
        return w1, w2
    r = float(norm_mult); t2 = math.tan(math.radians(theta_deg) / 2) ** 2
    if r * r <= 0.5:
        raise ValueError(f"norm_mult must exceed 1/sqrt2 (got {r}); ‖D‖² = ‖w‖²(r²−½)/2 must be positive")
    diag = {"rq_required": [], "rq_min": [], "rq_max": [], "feasible": [], "mix_frac": [], "d_norm2_over_w_norm2": [], "r_used": [], "gauge_c": []}
    for c in sel:
        w = Wf[c]; n = w.numel()
        Sw = S @ w
        # orthonormal basis of the doubly-orthogonal subspace {d : wᵀd = 0, wᵀΣd = 0}
        A = torch.stack([w, Sw], 1)                        # n x 2
        Qa, _ = torch.linalg.qr(A)                          # n x 2 orthonormal span of {w, Σw}
        P = torch.eye(n, dtype=torch.float64) - Qa @ Qa.T
        evals, evecs = torch.linalg.eigh(P)                 # basis of the complement = eigenvectors with eval 1
        B = evecs[:, evals > 0.5]                           # n x (n-2)
        Sb = B.T @ S @ B                                    # Sigma restricted to the subspace
        lam, U = torch.linalg.eigh(Sb)                      # ascending
        sig_norm2 = t2 * (w @ Sw) / 4                       # required Dᵀ Σ D
        rc = r
        if norm_auto:
            # smallest r_c (>= floor r) with rq <= lam_max: (r_c² − ½)/2 · ‖w‖² · lam_max >= sig_norm2
            r_min = math.sqrt(0.5 + auto_margin * float(sig_norm2 / ((w @ w) * lam[-1])) * 2)
            rc = max(r, r_min)
        d_norm2 = (w @ w) * (rc * rc - 0.5) / 2             # required ‖D‖²
        rq = sig_norm2 / d_norm2                            # required Rayleigh quotient
        feasible = bool(lam[0] - 1e-9 <= rq <= lam[-1] + 1e-9)
        z = torch.randn(B.shape[1], generator=gen, dtype=torch.float64)
        # target eigenvector to mix with: top if we need a higher quotient than z has, else bottom
        z = z / z.norm()
        rz = z @ Sb @ z
        e = U[:, -1] if rq > rz else U[:, 0]; le = lam[-1] if rq > rz else lam[0]
        zp = z - (z @ e) * e; zp = zp / zp.norm(); rzp = zp @ Sb @ zp   # zp ⟂ e, and Sb e = le e ⇒ no cross term
        # cos²φ · rzp + sin²φ · le = rq
        if abs(le - rzp) < 1e-12:
            cos2 = 1.0
        else:
            cos2 = float((le - rq) / (le - rzp))
        cos2c = min(max(cos2, 0.0), 1.0)
        dsub = math.sqrt(cos2c) * zp + math.sqrt(1 - cos2c) * e
        d = B @ dsub
        d = d / d.norm() * math.sqrt(d_norm2)
        # if infeasible, the Sigma-norm (hence the angle) is off; keep the Euclidean norm exact and record it
        c = math.sqrt(2)
        if gauge_norm_mult is not None:
            c = float(gauge_norm_mult * w.norm() / (w / 2 + d).norm())   # ‖w/2+d‖ == ‖w/2−d‖ since wᵀd = 0
        w1.append(c * (w / 2 + d)); w2.append(c * (w / 2 - d))
        diag["gauge_c"].append(c)
        diag["rq_required"].append(float(rq)); diag["rq_min"].append(float(lam[0])); diag["rq_max"].append(float(lam[-1]))
        diag["feasible"].append(feasible); diag["mix_frac"].append(1 - cos2c); diag["d_norm2_over_w_norm2"].append(float(d_norm2 / (w @ w))); diag["r_used"].append(float(rc))
    w1 = torch.stack(w1).float().reshape(len(sel), *shp); w2 = torch.stack(w2).float().reshape(len(sel), *shp)
    return w1, w2, diag


def split_diagnostics(W: torch.Tensor, sel: list[int], w1: torch.Tensor, w2: torch.Tensor, Sigma: torch.Tensor) -> dict:
    """Per-channel check of the split: Sigma-angle, Euclidean angle, branch Euclidean norms /
    parent, branch Sigma-norms / parent Sigma-norm, and the input variance per unit norm
    (sigma²/‖w‖²) of parent vs branches."""
    S = Sigma.double(); Wf = W.flatten(1).double()[sel]; f1 = w1.flatten(1).double(); f2 = w2.flatten(1).double()
    sn = lambda f: (f * (f @ S)).sum(1)
    cos_s = (f1 * (f2 @ S)).sum(1) / torch.sqrt(sn(f1) * sn(f2))
    cos_e = F.cosine_similarity(f1, f2, dim=1)
    pn2 = (Wf * Wf).sum(1); ps2 = sn(Wf)
    return {"cos_sigma": cos_s.numpy(), "cos_euclid": cos_e.numpy(),
            "branch_norm_over_parent": torch.sqrt(((f1 * f1).sum(1) + (f2 * f2).sum(1)) / 2 / pn2).numpy(),
            "branch_signorm_over_parent": torch.sqrt((sn(f1) + sn(f2)) / 2 / ps2).numpy(),
            "parent_var_per_norm2": (ps2 / pn2).numpy(),
            "branch_var_per_norm2": ((sn(f1) + sn(f2)) / ((f1 * f1).sum(1) + (f2 * f2).sum(1))).numpy()}


@torch.no_grad()
def calibrate_rho(model, block: ChanSplitBlock0, loader, n_batches: int, dev) -> tuple[torch.Tensor, dict]:
    """Measure sigma_bar/sigma_w per selected channel over n_batches (train mode
    batch stats, averaged) and the function-match error at the split."""
    model.train()
    ratios = []; errs = []; m1s = []; m2s = []; v1s = []; v2s = []
    conv, bn, sel = block.conv, block.bn, block.sel
    for i, (x, _) in enumerate(loader):
        if i >= n_batches:
            break
        x = x.to(dev)
        a = conv(x)[:, sel]
        vw = a.var((0, 2, 3), unbiased=False)
        a1 = F.conv2d(x, block.w1, None, conv.stride, conv.padding); a2 = F.conv2d(x, block.w2, None, conv.stride, conv.padding)
        v1 = a1.var((0, 2, 3), unbiased=False); v2 = a2.var((0, 2, 3), unbiased=False)
        m1s.append(a1.mean((0, 2, 3))); m2s.append(a2.mean((0, 2, 3))); v1s.append(v1); v2s.append(v2)
        ratios.append(torch.sqrt(((v1 + v2) / 2 + bn.eps) / (vw + bn.eps)))
        # function match with rho=1 (pre-ReLU), relative to the single's output std
        g = bn.weight[sel]; b = bn.bias[sel]
        ys = g[None, :, None, None] * (a - a.mean((0, 2, 3))[None, :, None, None]) / torch.sqrt(vw + bn.eps)[None, :, None, None] + b[None, :, None, None]
        c1 = a1 - a1.mean((0, 2, 3))[None, :, None, None]; c2 = a2 - a2.mean((0, 2, 3))[None, :, None, None]
        if block.block_mode == "shared_sigma":
            yp = g[None, :, None, None] * (c1 + c2) / (math.sqrt(2) * torch.sqrt((v1 + v2) / 2 + bn.eps))[None, :, None, None] + b[None, :, None, None]
        else:
            yp = g[None, :, None, None] * (c1 / torch.sqrt(v1 + bn.eps)[None, :, None, None] + c2 / torch.sqrt(v2 + bn.eps)[None, :, None, None]) * block.mix + b[None, :, None, None]
        errs.append(((yp - ys).pow(2).mean((0, 2, 3)) / ys.var((0, 2, 3))).sqrt())
    rho = torch.stack(ratios).mean(0)
    if block.block_mode != "shared_sigma":
        rho = torch.ones_like(rho)  # per-branch BN normalises each branch itself; no calibration factor
    err = torch.stack(errs).mean(0)
    # initialise the pair's running stats from the calibration batches (eval-mode use)
    block.rm1.copy_(torch.stack(m1s).mean(0)); block.rv1.copy_(torch.stack(v1s).mean(0))
    block.rm2.copy_(torch.stack(m2s).mean(0)); block.rv2.copy_(torch.stack(v2s).mean(0))
    return rho, {"rho_mean": rho.mean().item(), "rho_min": rho.min().item(), "rho_max": rho.max().item(),
                 "match_rel_rms_err_rho1": err.mean().item(), "match_rel_rms_err_rho1_max": err.max().item()}


# ---------------------------------------------------------------------- selection
def select_channels(gamma: torch.Tensor, beta: torch.Tensor, arm: str, k: int, seed: int,
                    oracle_gamma: torch.Tensor | None = None) -> list[int]:
    ga = gamma.abs()
    order = torch.argsort(ga)  # ascending
    if arm in ("downoracle", "uporacle"):
        if oracle_gamma is None:
            raise ValueError("oracle arms need --sel-from (npz with stages.0.0/gamma_end)")
        oo = torch.argsort(oracle_gamma.abs())
        return oo[:k].tolist() if arm == "downoracle" else oo[-k:].tolist()
    if arm in ("downkappa", "upkappa"):
        # oracle_gamma here carries the single's endpoint kappa = gamma^2 / running_var (conditioning statistic)
        if oracle_gamma is None:
            raise ValueError("kappa arms need --sel-from (npz with stages.0.0/gamma_end and rvar_end)")
        oo = torch.argsort(oracle_gamma)
        return oo[:k].tolist() if arm == "downkappa" else oo[-k:].tolist()
    if arm == "none":
        return []
    if arm == "all":
        return list(range(gamma.numel()))
    if arm == "down":
        return order[:k].tolist()
    if arm == "up":
        return order[-k:].tolist()
    if arm.startswith("rand"):
        g = torch.Generator().manual_seed(seed * 1000 + 7)
        return torch.randperm(gamma.numel(), generator=g)[:k].tolist()
    if arm == "downbeta":  # most-closed ReLUs: lowest beta/|gamma|
        return torch.argsort(beta / ga)[:k].tolist()
    raise ValueError(arm)


# ----------------------------------------------------------------------- training
@torch.no_grad()
def evaluate(model, loader, dev, max_batches=None):
    model.eval()
    correct = 0; n = 0; loss = 0.0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(dev), y.to(dev)
        out = model(x)
        loss += F.cross_entropy(out, y, reduction="sum").item()
        correct += (out.argmax(1) == y).sum().item(); n += y.numel()
    model.train()
    return correct / n, loss / n


def train_epochs(model, opt, train_loader, dev, n_epochs, on_epoch_end=None, tag="", on_step=None):
    t0 = time.time(); gstep = 0
    for ep in range(n_epochs):
        model.train()
        run_loss = 0.0; correct = 0; n = 0
        for x, y in train_loader:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            opt.step()
            gstep += 1
            if on_step is not None:
                on_step(ep, gstep, loss.item())
            run_loss += loss.item() * y.numel(); correct += (out.argmax(1) == y).sum().item(); n += y.numel()
        stats = {"run_loss": run_loss / n, "run_acc": correct / n}
        if on_epoch_end is not None:
            on_epoch_end(ep, stats)
        print(f"[{tag}] epoch {ep+1}/{n_epochs} run_loss {stats['run_loss']:.3f} run_acc {stats['run_acc']:.4f} ({time.time()-t0:.0f}s)", flush=True)


def loaders(a):
    return build_loaders(a.data_dir, batch_size=a.batch_size, num_workers=a.num_workers,
                         persistent_workers=a.num_workers > 0, batch_mode="shuffled", sampler_seed=a.seed)


def cmd_make_ckpt(a):
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device(a.device)
    sd = torch.load(fetch_ckpt(a.ckpt, 0), map_location="cpu", weights_only=False)["state_dict"]
    model = build_single(sd).to(dev).train()
    opt = build_conv_wd_sgd(model.parameters(), model, lr=a.lr, momentum=a.momentum, weight_decay=a.wd)
    train_loader, test_loader = loaders(a)
    hist = []
    def cb(ep, st):
        if (ep + 1) % 5 == 0 or ep + 1 == a.split_epoch:
            te, tl = evaluate(model, test_loader, dev); st.update(test_acc=te, test_loss=tl)
        hist.append(dict(epoch=ep + 1, **st))
    if a.split_epoch > 0:
        train_epochs(model, opt, train_loader, dev, a.split_epoch, cb, tag=f"make-ckpt s{a.seed}")
    else:
        # epoch-0 source: the init itself; populate BN running stats with one train-mode pass (no step)
        model.train()
        with torch.no_grad():
            for i, (x, _) in enumerate(train_loader):
                model(x.to(dev))
                if i >= 20:
                    break
        te, tl = evaluate(model, test_loader, dev); hist.append(dict(epoch=0, run_loss=float("nan"), run_acc=float("nan"), test_acc=te, test_loss=tl))
    tr_acc, tr_loss = evaluate(model, train_loader, dev)
    os.makedirs(a.out_dir, exist_ok=True)
    fp = os.path.join(a.out_dir, f"single_d3_wdconv_flat_s{a.seed}_ep{a.split_epoch}.pt")
    torch.save({"epoch": a.split_epoch, "seed": a.seed, "state_dict": model.state_dict(), "optimizer": opt.state_dict(),
                "history": hist, "train_eval_acc": tr_acc, "train_eval_loss": tr_loss, "args": vars(a)}, fp)
    print("saved", fp, "train_eval_acc %.4f" % tr_acc, flush=True)
    if a.wandb:
        import wandb
        run = wandb.init(project=WANDB_PROJECT, group=WANDB_GROUP, name=f"make_ckpt_s{a.seed}_ep{a.split_epoch}", job_type="ckpt", config=vars(a))
        art = wandb.Artifact(f"chansplit_src_single_d3_wdconv_flat_s{a.seed}_ep{a.split_epoch}", type="checkpoint")
        art.add_file(fp); run.log_artifact(art)
        run.summary.update({"train_eval_acc": tr_acc, "final_test_acc": hist[-1].get("test_acc")}); run.finish()


def cmd_run(a):
    dev = torch.device(a.device)
    src = torch.load(a.src, map_location="cpu", weights_only=False)
    split_epoch = int(src["epoch"])
    model = build_single(src["state_dict"]).to(dev).train()
    blk = model.stages[0][0]
    conv, bn = blk.conv3_branches[0][0], blk.conv3_branches[0][1]
    C = conv.weight.shape[0]
    torch.manual_seed(a.seed + 1000); np.random.seed(a.seed + 1000)  # identical data order across arms
    train_loader, test_loader = loaders(a)

    oracle = None
    if a.sel_from:
        ref = np.load(a.sel_from)
        og = ref[f"{BLOCK}/gamma_end"]
        oracle = torch.tensor(og[-1])  # the single's endpoint gamma (its "fate")
        if a.arm in ("downkappa", "upkappa"):
            oracle = torch.tensor(og[-1] ** 2 / ref[f"{BLOCK}/rvar_end"][-1])  # kappa = gamma^2 / sigma^2 at the single's endpoint
    sel = sorted(select_channels(bn.weight.detach().cpu(), bn.bias.detach().cpu(), a.arm, a.k, a.seed, oracle))  # sorted: block.sel order == w1/w2 row order
    info = {"arm": a.arm, "k": len(sel), "sel": sel, "split_epoch": split_epoch, "block": a.block}
    if oracle is not None:
        info["oracle_stat_sel"] = oracle[sel].tolist() if sel else []
        info["oracle_stat_median"] = float(oracle.median())
    g0 = bn.weight.detach().cpu(); b0 = bn.bias.detach().cpu()
    info["sel_gamma_abs"] = g0.abs()[sel].tolist() if sel else []
    info["sel_beta_over_gamma"] = (b0 / g0.abs())[sel].tolist() if sel else []
    info["block_gamma_abs_median"] = g0.abs().median().item()
    Sigma, _ = input_patch_cov(train_loader, a.cov_batches, dev)  # always: keeps the data-order RNG identical across arms
    if sel:
        gen = torch.Generator().manual_seed(a.seed * 100 + a.dir_seed)
        theta = getattr(a, "theta_deg", None)
        gauge_c = None
        if theta is None:
            w1, w2 = make_split(conv.weight.detach().cpu(), sel, Sigma.cpu(), gen)
        else:
            if a.block != "branch_sigma":
                raise ValueError("--theta-deg / --norm-mult are only defined for --block branch_sigma")
            w1, w2, sdiag = make_split(conv.weight.detach().cpu(), sel, Sigma.cpu(), gen, theta_deg=theta, norm_mult=a.norm_mult, norm_auto=a.norm_auto,
                                       gauge_norm_mult=a.gauge_norm_mult)
            nf = int(sum(1 for f in sdiag["feasible"] if not f))
            gauge_c = torch.tensor(sdiag["gauge_c"], dtype=torch.float32)
            info.update(theta_deg=theta, norm_mult=a.norm_mult, norm_auto=a.norm_auto, n_infeasible=nf,
                        gauge_norm_mult=a.gauge_norm_mult, gauge_c_median=float(gauge_c.median()), gauge_c_min=float(gauge_c.min()), gauge_c_max=float(gauge_c.max()),
                        r_used=sdiag["r_used"], r_used_median=float(np.median(sdiag["r_used"])), r_used_max=float(np.max(sdiag["r_used"])),
                        n_above_floor=int(sum(1 for x in sdiag["r_used"] if x > a.norm_mult * 1.0001)),
                        rq_required_median=float(np.median(sdiag["rq_required"])), rq_max_median=float(np.median(sdiag["rq_max"])))
            if nf and getattr(a, "strict_feasible", False):
                raise RuntimeError(f"{nf}/{len(sel)} channels cannot reach theta={theta} at norm_mult={a.norm_mult}: {sdiag}")
            sd_ = split_diagnostics(conv.weight.detach().cpu(), sel, w1, w2, Sigma.cpu())
            info.update({f"split_{k}_median": float(np.median(v)) for k, v in sd_.items()})
        block = ChanSplitBlock0(conv, bn, sel, w1.to(dev), w2.to(dev), torch.ones(len(sel), device=dev), block=a.block,
                                theta_deg=(90.0 if theta is None else theta)).to(dev)
        assert block.sel.tolist() == sel
        rho, cal = calibrate_rho(model, block, train_loader, a.calib_batches, dev)
        block.rho.copy_(rho)
        info.update(cal)
        if cal["match_rel_rms_err_rho1_max"] > 0.1:
            raise RuntimeError(f"function match failed at split: {cal}")
        # empirical angle in the Sigma metric and Euclidean
        S = Sigma.cpu().double()
        with torch.no_grad():
            f1 = w1.flatten(1).double(); f2 = w2.flatten(1).double()
            info["cos_sigma_init"] = ((f1 * (f2 @ S)).sum(1) / torch.sqrt((f1 * (f1 @ S)).sum(1) * (f2 * (f2 @ S)).sum(1))).mean().item()
            info["cos_euclid_init"] = F.cosine_similarity(f1, f2, dim=1).mean().item()
        # zero the retired single kernels of the paired channels (their outputs are masked)
        with torch.no_grad():
            conv.weight[block.sel] = 0.0
    else:
        gauge_c = None
        block = ChanSplitBlock0(conv, bn, [], block=a.block, theta_deg=(getattr(a, "theta_deg", None) or 90.0)).to(dev)
        for i, _ in enumerate(train_loader):  # consume the same loader iterator as calibrate_rho does
            if i >= a.calib_batches - 1:
                break
    install_block(model, block)
    print("arm", a.arm, "sel", sel, json.dumps({k: v for k, v in info.items() if k not in ("sel", "sel_gamma_abs", "sel_beta_over_gamma")}), flush=True)

    # optimizer: same param order as the source for the single params, then the pair kernels
    opt = build_conv_wd_sgd(model.parameters(), model, lr=a.lr, momentum=a.momentum, weight_decay=a.wd)
    if a.carry_momentum:
        src_opt = src["optimizer"]
        # map source momentum buffers by parameter identity: rebuild the source param list order
        src_model = build_single(src["state_dict"])
        src_params = [p for p in src_model.parameters()]
        src_state = src_opt["state"]
        # source groups: decay (ndim>=2) then no_decay, in model.parameters() order
        src_dec = [i for i, p in enumerate(src_params) if p.ndim >= 2]; src_nod = [i for i, p in enumerate(src_params) if p.ndim < 2]
        src_order = src_dec + src_nod  # index j in optimizer state <-> src_params[src_order[j]]
        name_of = {id(p): n for n, p in src_model.named_parameters()}
        buf_by_name = {}
        for j, pi in enumerate(src_order):
            st = src_state.get(j, {})
            if "momentum_buffer" in st and st["momentum_buffer"] is not None:
                buf_by_name[name_of[id(src_params[pi])]] = st["momentum_buffer"]
        new_named = dict(model.named_parameters())
        alias = {"stages.0.0.conv.weight": "stages.0.0.conv3_branches.0.0.weight",
                 "stages.0.0.bn.weight": "stages.0.0.conv3_branches.0.1.weight",
                 "stages.0.0.bn.bias": "stages.0.0.conv3_branches.0.1.bias"}
        new_params_order = [p for g in opt.param_groups for p in g["params"]]
        applied = 0
        for j, p in enumerate(new_params_order):
            nm = [n for n, q in new_named.items() if q is p][0]
            src_nm = alias.get(nm, nm)
            if src_nm in buf_by_name:
                buf = buf_by_name[src_nm].clone().to(dev)
                if nm == "stages.0.0.conv.weight" and sel:
                    buf[block.sel] = 0.0
                opt.state[p] = {"momentum_buffer": buf}; applied += 1
            elif nm in ("stages.0.0.w1", "stages.0.0.w2") and "stages.0.0.conv3_branches.0.0.weight" in buf_by_name:
                mb = buf_by_name["stages.0.0.conv3_branches.0.0.weight"].to(dev)[block.sel]
                if gauge_c is not None:   # gradient w.r.t. each copy = (1/c) x parent gradient
                    mb = mb / gauge_c.to(dev)[:, None, None, None]
                else:
                    mb = mb / math.sqrt(2)
                opt.state[p] = {"momentum_buffer": mb.clone()}; applied += 1
        # note: source names are conv3_branches.0.{0,1}.*; the wrapped block exposes them as conv.* / bn.*
        info["momentum_buffers_applied"] = applied
    print("momentum buffers applied:", info.get("momentum_buffers_applied"), flush=True)

    # logging containers
    n_more = a.epochs - split_epoch
    hist = []
    KEYS = ("gamma", "beta", "wnorm_single", "rvar_single", "w1norm", "w2norm", "cos_euclid", "cos_sigma", "v1", "v2", "sig1", "sig2")
    per = {k: [] for k in KEYS}
    dense = {k: [] for k in KEYS}; dense["step"] = []; dense["loss"] = []
    Sd = Sigma.to(dev).double()

    def snapshot(store=per):
        with torch.no_grad():
            store["gamma"].append(bn.weight.detach().cpu().numpy().copy()); store["beta"].append(bn.bias.detach().cpu().numpy().copy())
            store["wnorm_single"].append(conv.weight.detach().flatten(1).norm(dim=1).cpu().numpy())
            store["rvar_single"].append(bn.running_var.detach().cpu().numpy().copy())
            if sel:
                f1 = block.w1.detach().flatten(1); f2 = block.w2.detach().flatten(1)
                store["w1norm"].append(f1.norm(dim=1).cpu().numpy()); store["w2norm"].append(f2.norm(dim=1).cpu().numpy())
                store["cos_euclid"].append(F.cosine_similarity(f1, f2, dim=1).cpu().numpy())
                d1 = f1.double(); d2 = f2.double()
                store["cos_sigma"].append(((d1 * (d2 @ Sd)).sum(1) / torch.sqrt((d1 * (d1 @ Sd)).sum(1) * (d2 * (d2 @ Sd)).sum(1))).cpu().numpy())
                store["v1"].append(block.rv1.cpu().numpy().copy()); store["v2"].append(block.rv2.cpu().numpy().copy())
                # Sigma-norms of the branch kernels (= the BN sigma the invariant uses, from the fixed split-time Sigma)
                store["sig1"].append(torch.sqrt((d1 * (d1 @ Sd)).sum(1)).cpu().numpy()); store["sig2"].append(torch.sqrt((d2 * (d2 @ Sd)).sum(1)).cpu().numpy())

    def on_step(ep, gstep, loss):
        if a.dense_steps and ep < a.dense_epochs and gstep % a.dense_steps == 0:
            snapshot(dense); dense["step"].append(gstep); dense["loss"].append(loss)

    wb = None
    if a.wandb:
        import wandb
        wb = wandb.init(project=WANDB_PROJECT, group=WANDB_GROUP, name=f"{a.block}{_theta_tag(a)}_{a.arm}_k{len(sel)}_s{a.seed}_split{split_epoch}", job_type="train",
                        config={**vars(a), **{k: v for k, v in info.items() if k != "sel"}})
        wb.log({"epoch": split_epoch, "test_acc": evaluate(model, test_loader, dev)[0]})

    def cb(ep, st):
        e = split_epoch + ep + 1
        if e % a.eval_every == 0 or e == a.epochs:
            te, tl = evaluate(model, test_loader, dev); st.update(test_acc=te, test_loss=tl)
        if e % a.train_eval_every == 0 or e == a.epochs:
            tr, trl = evaluate(model, train_loader, dev); st.update(train_eval_acc=tr, train_eval_loss=trl)
        snapshot()
        hist.append(dict(epoch=e, **st))
        if wb is not None:
            wb.log({"epoch": e, **st, **({"cos_sigma_mean": float(per["cos_sigma"][-1].mean()), "cos_euclid_mean": float(per["cos_euclid"][-1].mean())} if sel else {})})

    snapshot()
    train_epochs(model, opt, train_loader, dev, n_more, cb, tag=f"{a.arm} k{len(sel)} s{a.seed}", on_step=on_step)

    os.makedirs(a.out_dir, exist_ok=True)
    tag = f"{a.block}{_theta_tag(a)}_{a.arm}_k{len(sel)}_s{a.seed}_split{split_epoch}"
    fp = os.path.join(a.out_dir, f"{tag}.npz")
    out = {k: np.stack(v) for k, v in per.items() if len(v)}
    out.update({f"dense/{k}": np.stack(v) if isinstance(v[0], np.ndarray) else np.asarray(v) for k, v in dense.items() if len(v)})
    np.savez_compressed(fp, meta=json.dumps({"args": vars(a), "info": info, "history": hist}), **out)
    final = hist[-1]
    print("DONE", tag, json.dumps({k: final.get(k) for k in ("epoch", "run_acc", "train_eval_acc", "test_acc")}), flush=True)
    if wb is not None:
        wb.summary.update({f"final_{k}": v for k, v in final.items()})
        art = wandb.Artifact(f"chansplit_{tag}", type="analysis"); art.add_file(fp); wb.log_artifact(art); wb.finish()


def _theta_tag(a) -> str:
    th = getattr(a, "theta_deg", None)
    if th is None:
        return ""
    g = getattr(a, "gauge_norm_mult", None)
    return f"_th{int(round(th))}_r{a.norm_mult:g}" + ("" if g is None else f"_g{g:.3g}")


def cmd_dry_split(a):
    """Build the generalised split from the source checkpoint and print per-channel
    feasibility + neutrality diagnostics. No training, no W&B."""
    dev = torch.device("cpu")
    src = torch.load(a.src, map_location="cpu", weights_only=False)
    model = build_single(src["state_dict"]).to(dev).train()
    blk = model.stages[0][0]
    conv, bn = blk.conv3_branches[0][0], blk.conv3_branches[0][1]
    torch.manual_seed(a.seed + 1000); np.random.seed(a.seed + 1000)
    train_loader, _ = loaders(a)
    Sigma, _ = input_patch_cov(train_loader, a.cov_batches, dev)
    sel = list(range(conv.weight.shape[0]))
    W = conv.weight.detach().cpu()
    ev = torch.linalg.eigvalsh(Sigma.double())
    print(f"Sigma (27x27 patch cov): eigenvalues min {ev[0]:.4f} median {ev.median():.4f} max {ev[-1]:.4f}")
    rows = []
    for th in a.theta_list:
        gen = torch.Generator().manual_seed(a.seed * 100 + a.dir_seed)
        w1, w2, sdiag = make_split(W, sel, Sigma, gen, theta_deg=th, norm_mult=a.norm_mult, norm_auto=a.norm_auto)
        d = split_diagnostics(W, sel, w1, w2, Sigma)
        block = ChanSplitBlock0(conv, bn, sel, w1, w2, torch.ones(len(sel)), block="branch_sigma", theta_deg=th)
        _, cal = calibrate_rho(model, block, train_loader, a.calib_batches, dev)
        nf = sum(1 for f in sdiag["feasible"] if not f)
        target_cos = math.cos(math.radians(th))
        row = dict(theta=th, target_cos=target_cos, cos_sigma_med=float(np.median(d["cos_sigma"])), cos_sigma_maxdev=float(np.max(np.abs(d["cos_sigma"] - target_cos))),
                   cos_euclid_med=float(np.median(d["cos_euclid"])), branch_norm_over_parent_med=float(np.median(d["branch_norm_over_parent"])),
                   branch_norm_over_parent_maxdev=float(np.max(np.abs(d["branch_norm_over_parent"] - np.array(sdiag["r_used"])))),
                   r_used_med=float(np.median(sdiag["r_used"])), r_used_p90=float(np.percentile(sdiag["r_used"], 90)), r_used_max=float(np.max(sdiag["r_used"])),
                   n_above_floor=int(sum(1 for x in sdiag["r_used"] if x > a.norm_mult * 1.0001)),
                   branch_signorm_over_parent_med=float(np.median(d["branch_signorm_over_parent"])),
                   parent_var_per_norm2_med=float(np.median(d["parent_var_per_norm2"])), branch_var_per_norm2_med=float(np.median(d["branch_var_per_norm2"])),
                   n_infeasible=nf, rq_required_med=float(np.median(sdiag["rq_required"])), rq_required_max=float(np.max(sdiag["rq_required"])),
                   rq_max_med=float(np.median(sdiag["rq_max"])), rq_max_min=float(np.min(sdiag["rq_max"])), mix_frac_med=float(np.median(sdiag["mix_frac"])),
                   fn_match_err=cal["match_rel_rms_err_rho1"], fn_match_err_max=cal["match_rel_rms_err_rho1_max"])
        rows.append(row)
        print(json.dumps(row))
    return rows


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--seed", type=int, default=42)
    common.add_argument("--lr", type=float, default=0.1)
    common.add_argument("--momentum", type=float, default=0.9)
    common.add_argument("--wd", type=float, default=5e-4)
    common.add_argument("--batch-size", type=int, default=128)
    common.add_argument("--data-dir", default="data/cifar100")
    common.add_argument("--num-workers", type=int, default=6)
    common.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    common.add_argument("--out-dir", default="outputs/chansplit")
    common.add_argument("--wandb", action="store_true")
    p1 = sub.add_parser("make-ckpt", parents=[common])
    p1.add_argument("--ckpt", default=f"{ENTITY}/{WANDB_PROJECT}/ckpt_ckptflat_c100_wdconv_single_d3_wdconv_s42:v0")
    p1.add_argument("--split-epoch", type=int, default=30)
    p2 = sub.add_parser("run", parents=[common])
    p2.add_argument("--src", required=True, help="path to the make-ckpt .pt")
    p2.add_argument("--arm", required=True, choices=["none", "all", "down", "up", "rand", "downbeta", "downoracle", "uporacle", "downkappa", "upkappa"])
    p2.add_argument("--block", default="shared_sigma", choices=["shared_sigma", "branch_sigma"])
    p2.add_argument("--sel-from", default=None, help="npz with stages.0.0/gamma_end (single's endpoint) for the oracle arms")
    p2.add_argument("--k", type=int, default=10)
    p2.add_argument("--epochs", type=int, default=100)
    p2.add_argument("--dir-seed", type=int, default=0)
    p2.add_argument("--cov-batches", type=int, default=20)
    p2.add_argument("--calib-batches", type=int, default=20)
    p2.add_argument("--eval-every", type=int, default=5)
    p2.add_argument("--train-eval-every", type=int, default=10)
    p2.add_argument("--no-carry-momentum", dest="carry_momentum", action="store_false")
    p2.add_argument("--theta-deg", type=float, default=None, help="generalised split: branch angle in the Sigma metric (None = legacy 90° Sigma-only construction)")
    p2.add_argument("--norm-mult", type=float, default=1.0, help="generalised split: Euclidean branch norm = norm_mult × parent norm (must exceed 1/sqrt2)")
    p2.add_argument("--norm-auto", action="store_true", help="per channel, raise the branch norm above --norm-mult to the minimum that admits --theta-deg")
    p2.add_argument("--gauge-norm-mult", type=float, default=None, help="generalised split: set the branch Euclidean norm to this multiple of the parent norm by a per-channel gauge factor c (replaces the sqrt2 prefactor); e.g. 0.7071 = pair WD-equilibrium norm")
    p2.add_argument("--dense-steps", type=int, default=10, help="per-step snapshot interval during the first --dense-epochs after the split (0 = off)")
    p2.add_argument("--dense-epochs", type=int, default=3)
    p2.add_argument("--strict-feasible", action="store_true", help="abort if any channel cannot reach --theta-deg at --norm-mult")
    p3 = sub.add_parser("dry-split", parents=[common])
    p3.add_argument("--src", required=True)
    p3.add_argument("--theta-list", type=float, nargs="+", default=[30, 60, 90, 120, 150])
    p3.add_argument("--norm-mult", type=float, default=1.0)
    p3.add_argument("--norm-auto", action="store_true")
    p3.add_argument("--dir-seed", type=int, default=0)
    p3.add_argument("--cov-batches", type=int, default=20)
    p3.add_argument("--calib-batches", type=int, default=5)
    a = ap.parse_args()
    if a.cmd == "make-ckpt":
        cmd_make_ckpt(a)
    elif a.cmd == "dry-split":
        cmd_dry_split(a)
    else:
        cmd_run(a)


if __name__ == "__main__":
    main()
