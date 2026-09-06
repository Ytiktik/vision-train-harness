"""Pinned-kernel-norm SGD: the norm of every conv kernel (per output channel)
is held at its initial value by projection after each step, and the momentum
buffer's radial component is removed, so the only remaining scale parameters
are the BatchNorm gammas. No weight decay on kernels (a pinned norm has nothing
to decay). Motivation: for a conv followed by BatchNorm the kernel norm is a
gauge whose only effect is the angular learning rate lr/||w||^2; pinning it
removes the weight-decay-equilibrium confound in the pair-vs-single
conditioning comparison and makes the fixed-norm theory exact.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import torch
from torch import nn

LOGGER = logging.getLogger(__name__)


class PinnedNormSGD(torch.optim.SGD):
    """SGD (with momentum) that projects selected weights back onto their
    per-output-channel init-norm spheres after every step.

    For each pinned parameter ``p`` of shape ``(C, ...)`` with radii ``r`` of
    shape ``(C, 1)``:  ``p[c] <- r[c] * p[c] / ||p[c]||`` and the momentum
    buffer ``b[c] <- b[c] - (b[c] . p̂[c]) p̂[c]`` (radial part removed).
    """

    def __init__(self, param_groups, pinned: list[tuple[nn.Parameter, torch.Tensor]], **kw):
        super().__init__(param_groups, **kw)
        self._pinned = pinned

    @torch.no_grad()
    def project(self) -> None:
        for p, r in self._pinned:
            w = p.view(p.shape[0], -1)
            n = w.norm(dim=1, keepdim=True).clamp_min(1e-12)
            w.mul_(r.to(w.device, w.dtype) / n)
            buf = self.state.get(p, {}).get("momentum_buffer")
            if buf is not None:
                b = buf.view(buf.shape[0], -1)
                what = w / r.to(w.device, w.dtype)
                b.sub_((b * what).sum(dim=1, keepdim=True) * what)

    @torch.no_grad()
    def step(self, closure=None):
        loss = super().step(closure)
        self.project()
        return loss


def pinned_params(model: nn.Module) -> list[nn.Parameter]:
    """Every Conv2d weight (all convs in our RepVGG stacks are followed by BN)."""
    out: list[nn.Parameter] = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d) and m.weight.requires_grad:
            out.append(m.weight)
    return out


def build_pinned_norm_sgd(
    params: Iterable[nn.Parameter],
    model: nn.Module,
    lr: float,
    momentum: float = 0.0,
    weight_decay: float = 0.0,
    nesterov: bool = False,
    pin_radius: str | float = "init",
    pair_radius_scale: float = 1.0,
    **_: object,
) -> PinnedNormSGD:
    """Optimizer factory for ``train.optimizer.target``. ``pin_radius='init'``
    pins each output channel to its own init norm; a float pins every channel
    of every conv to that radius (rescaling the weights once at build time).
    ``pair_radius_scale`` multiplies the radius of every kernel that belongs to
    a block with two or more branches (a ``.convs`` ModuleList of length >= 2),
    rescaling those kernels at build time; 1/sqrt(2) gauges a collapsed pair to
    the single's per-kernel angular step and summed curvature.
    ``weight_decay`` applies to non-pinned ndim>=2 weights only (the fc)."""
    pin = pinned_params(model)
    pin_ids = {id(p) for p in pin}
    pair_ids = {id(c.weight) for m in model.modules() if hasattr(m, "convs") and len(m.convs) >= 2 for c in m.convs if isinstance(c, nn.Conv2d)}
    pinned: list[tuple[nn.Parameter, torch.Tensor]] = []
    with torch.no_grad():
        for p in pin:
            w = p.view(p.shape[0], -1)
            if pin_radius == "init":
                r = w.norm(dim=1, keepdim=True).clone()
            else:
                r = torch.full((p.shape[0], 1), float(pin_radius), device=p.device, dtype=p.dtype)
            if id(p) in pair_ids and pair_radius_scale != 1.0:
                r = r * float(pair_radius_scale)
            w.mul_(r / w.norm(dim=1, keepdim=True).clamp_min(1e-12))
            pinned.append((p, r))
    other_decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad or id(p) in pin_ids:
            continue
        (other_decay if p.ndim >= 2 else no_decay).append(p)
    if weight_decay:
        LOGGER.warning("pinned-norm SGD: weight_decay=%g applies to the %d non-pinned ndim>=2 tensors only", weight_decay, len(other_decay))
    groups = [
        {"params": pin, "weight_decay": 0.0},
        {"params": other_decay, "weight_decay": float(weight_decay)},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    opt = PinnedNormSGD(groups, pinned, lr=lr, momentum=momentum, nesterov=nesterov)
    LOGGER.info("pinned-norm SGD: %d conv kernels pinned (radius=%s, pair_radius_scale=%g on %d pair kernels), %d other decayed tensors, %d no-decay tensors",
                len(pin), pin_radius, pair_radius_scale, len(pair_ids & pin_ids), len(other_decay), len(no_decay))
    return opt


# ---------------------------------------------------------------------------
# Placement model: shared-scale RepVGG stack with a per-block branch count, so a
# pair can be placed in the last block only (block_branches [1,1,2]) while the
# other blocks stay one-branch. Same block class as the pinned cell's arms.
# ---------------------------------------------------------------------------
from collections.abc import Sequence  # noqa: E402

from structural_reparam.experiments.reparam_shared_scale.lab import SharedScaleRepVGGBlock  # noqa: E402


class PlacedSharedScaleRepVGGCifar(nn.Module):
    """RepVGG-CIFAR stack of ``SharedScaleRepVGGBlock`` with ``block_branches``
    giving the branch count of every block in network order (length must equal
    ``sum(stage_blocks)``). ``mode`` is the shared-scale mode (``shared_gamma``)."""

    def __init__(
        self,
        num_classes: int = 100,
        width_mult: float = 1.0,
        stage_channels: Sequence[int] = (64, 128, 256),
        stage_blocks: Sequence[int] = (1, 1, 1),
        stage_strides: Sequence[int] = (1, 2, 2),
        block_branches: Sequence[int] = (1, 1, 2),
        mode: str = "shared_gamma",
        eps: float = 1e-5,
        identical_init_blocks: Sequence[int] = (),
        identical_init_gamma: float | None = None,
        split_init: dict | None = None,
        single_beta_blocks: Sequence[int] = (),
        pair_kernel_init_scale: float = 1.0,
        init_sigma_select: dict | None = None,
        block_gamma_init: dict | None = None,
        init_whitened_angle: dict | None = None,
        init_whitened_isotropic: dict | None = None,
        input_filter_alpha: float = 0.0,
        input_filter_band: Sequence[float] | None = None,
        block0_damp_top: float = 1.0,
        block0_damp_renorm: bool = False,
        block0_damp_dir: str = "top",
        block0_damp_dir_seed: int = 0,
        block0_whiten_eps: float | None = None,
        input_zca_eps: float | None = None,
        input_zca_alpha: float = -0.5,
        input_zca_data_dir: str = "data/cifar100",
        **_: object,
    ) -> None:
        """``identical_init_blocks``: block indices (network order) whose branch
        kernels are made identical at init (w_i <- w_0), i.e. a collapsed pair;
        ``identical_init_gamma`` sets their shared gamma (1/num_branches makes
        the collapsed pair compute exactly the single's function at init).
        ``split_init``: {"blocks": [0], "zeta": 1e-2, "metric": "whitened"|"kernel"}.
        The infinitesimal split of the paper's first-order analysis (Sept 2026).
        Each listed two-branch block (block 0 only, needs the committed patch
        covariance Sigma) starts as an identical pair pushed apart by zeta: with w
        the branch-0 kernel and d a random direction per channel, the kernels become
        w +- zeta*d, where d is made Sigma-orthogonal to w and scaled to w's
        Sigma-norm, so the Sigma-whitened half-angle is exactly arctan(zeta), the
        bisector is w, and the two branch sigmas are equal at init. "whitened" draws
        d isotropically in the whitened metric (alignment with v_max near 1/sqrt(D));
        "kernel" draws it isotropically in kernel space (the whitening map tilts it
        toward v_max, as the standard init is tilted). zeta = 0 reproduces
        ``identical_init_blocks``. Gamma is left at its default.
        ``single_beta_blocks``: block indices whose pair keeps ONE trainable bias
        (beta_1); the other betas are zeroed and frozen, so the block's bias moves
        at the single's rate instead of num_branches times faster.
        ``pair_kernel_init_scale``: multiply the kernels of every block with two or
        more branches by this factor at init (function-preserving under BatchNorm);
        1/sqrt(2) gauges a collapsed pair's branches to half the single's norm^2, the
        init-radius gauge of the pinned cell, for free-norm (decayed) runs.
        ``init_sigma_select``: {"blocks": [0], "k": 2, "rule": "min"|"max"|"median"}.
        For each conv kernel of each listed block (network order), draw k candidate
        kernels with the standard init, score each candidate's init sigma^2 = w^T Cov w
        using the committed CIFAR-100 3x3-patch covariance (block 0 only; other blocks
        are rejected), keep one per output channel by the rule, and rescale it to the
        first candidate's norm. Only the direction's sigma draw changes.
        ``block_gamma_init``: {block index (network order): value} sets that
        block's gamma init to the value (all channels).
        ``init_whitened_angle``: {"blocks": [0], "theta_deg": 150.0}. For each
        listed block (block 0 only, needs the committed patch covariance), set
        every non-first branch kernel so that its Sigma-whitened cosine with
        branch 0 is exactly cos(theta_deg), per channel: w2 = a*w1 + b*r with a
        random draw r and (a, b) solved from the whitened inner products, then
        rescaled to w1's Euclidean norm. Controls the init Q = sin^2(theta/2)
        * exp(gamma^2/2) together with ``block_gamma_init``.
        ``init_whitened_isotropic``: {"blocks": [0], "eps": 0.01, "match": "sigma"}.
        For each listed block (block 0 only, needs the committed patch covariance)
        and every branch, replace the Kaiming kernel by one whose whitened
        direction is isotropic: draw u ~ N(0, I) and set w = (Sigma + eps I)^{-1/2} u.
        The isotropic Kaiming draw is tilted toward the top eigendirection in the
        whitened frame (its expected share of response variance along each
        eigendirection is lambda_k over the trace of Sigma, about 0.7 on v_max at
        block 0); this knob removes that tilt while leaving the input unchanged.
        ``eps`` (in Sigma's units) damps the quietest modes, which a bare
        Sigma^{-1/2} would amplify by 1/sqrt(lambda_min); modes louder than eps
        come out isotropic. ``match`` = "sigma" rescales each kernel so that its
        response variance w^T Sigma w equals the Kaiming kernel's for that channel
        (same gamma/sigma^2, so the same effective learning rate of the direction
        at step 0); "norm" keeps the Kaiming Euclidean norm instead."""
        super().__init__()
        n_blocks = int(sum(int(b) for b in stage_blocks))
        if len(block_branches) != n_blocks:
            raise ValueError(f"block_branches has {len(block_branches)} entries for {n_blocks} blocks")
        channels = [max(1, int(c * width_mult)) for c in stage_channels]
        in_channels = 3
        stages: list[nn.Module] = []
        k = 0
        for out_channels, blocks, stride in zip(channels, stage_blocks, stage_strides):
            layers: list[nn.Module] = []
            for block_idx in range(int(blocks)):
                layers.append(SharedScaleRepVGGBlock(
                    in_channels if block_idx == 0 else out_channels, out_channels,
                    stride=int(stride) if block_idx == 0 else 1,
                    num_branches=int(block_branches[k]), mode=mode, eps=eps))
                k += 1
            stages.append(nn.Sequential(*layers))
            in_channels = out_channels
        self.stages = nn.ModuleList(stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels[-1], num_classes)
        self.block_branches = [int(b) for b in block_branches]
        # Fixed, non-learned spectral shaping of the input: multiply the 2-D image
        # spectrum by (|f|/|f|max)^alpha and rescale to the original standard
        # deviation. alpha=0 is the untouched dataset; larger alpha flattens the
        # patch covariance spectrum by suppressing the dominant low-frequency mode.
        # It changes only what the first block sees, nothing else about the cell.
        self.input_filter_alpha = float(input_filter_alpha)
        # Alternative shaping: a band-pass (lo, hi, s) on the normalised radial
        # frequency. Frequencies below lo are removed, [lo, hi] pass at gain 1,
        # above hi are scaled by s. This can raise the two gradient modes to sit
        # level with the flat mode, which the power law cannot do.
        self.input_filter_band = None if input_filter_band is None else tuple(float(v) for v in input_filter_band)
        blocks_flat = [b for st in self.stages for b in st]
        with torch.no_grad():
            for bi in identical_init_blocks:
                blk = blocks_flat[int(bi)]
                for c in blk.convs[1:]:
                    c.weight.copy_(blk.convs[0].weight)
                if identical_init_gamma is not None:
                    blk.gamma.fill_(float(identical_init_gamma))
            if split_init:
                import os
                cfgz = dict(split_init)
                zeta = float(cfgz.get("zeta", 1e-2))
                metric = str(cfgz.get("metric", "whitened"))
                if metric not in ("whitened", "kernel"):
                    raise ValueError(f"split_init: metric must be 'whitened' or 'kernel', got {metric!r}")
                cov = torch.load(os.path.join(os.path.dirname(__file__), "cifar100_patch_cov.pt"), map_location="cpu")
                for bi in cfgz.get("blocks", [0]):
                    blk = blocks_flat[int(bi)]
                    if len(blk.convs) != 2:
                        raise ValueError(f"split_init: block {bi} must have exactly 2 branches, has {len(blk.convs)}")
                    w = blk.convs[0].weight
                    D = w.shape[1] * w.shape[2] * w.shape[3]
                    if D != cov.shape[0]:
                        raise ValueError(f"split_init: block {bi} kernel dim {D} != covariance dim {cov.shape[0]} (block 0 only)")
                    Sg = cov.to(torch.float64)
                    ev, V = torch.linalg.eigh(Sg)
                    ev = ev.clamp_min(1e-12)
                    S_inv_half = V @ torch.diag(ev.rsqrt()) @ V.T
                    f = w.flatten(1).to(torch.float64)                            # [C, D]
                    g = torch.randn(f.shape, dtype=torch.float64)
                    d = g @ S_inv_half if metric == "whitened" else g              # rows: Sigma^-1/2 g, or g
                    Sw = f @ Sg                                                    # [C, D] rows: Sigma w
                    wSw = (Sw * f).sum(1).clamp_min(1e-30)
                    d = d - ((d * Sw).sum(1) / wSw).unsqueeze(1) * f               # Sigma-orthogonal to w
                    dSd = ((d @ Sg) * d).sum(1).clamp_min(1e-30)
                    d = d * (wSw / dSd).sqrt().unsqueeze(1)                        # same Sigma-norm as w
                    w1 = (f + zeta * d).to(w.dtype); w2 = (f - zeta * d).to(w.dtype)
                    blk.convs[0].weight.copy_(w1.view_as(w))
                    blk.convs[1].weight.copy_(w2.view_as(w))
            for bi in single_beta_blocks:
                blk = blocks_flat[int(bi)]
                for b in list(blk.betas)[1:]:
                    b.zero_(); b.requires_grad_(False)
            for bi, val in (block_gamma_init or {}).items():
                blocks_flat[int(bi)].gamma.fill_(float(val))
            if init_sigma_select:
                import os
                cfgs = dict(init_sigma_select)
                cov = torch.load(os.path.join(os.path.dirname(__file__), "cifar100_patch_cov.pt"), map_location="cpu")
                k = int(cfgs.get("k", 2)); rule = cfgs.get("rule", "min")
                for bi in cfgs.get("blocks", [0]):
                    blk = blocks_flat[int(bi)]
                    for conv in blk.convs:
                        w = conv.weight
                        if w.shape[1] * w.shape[2] * w.shape[3] != cov.shape[0]:
                            raise ValueError("init_sigma_select: block %s kernel dim %d != covariance dim %d (block 0 only)" % (bi, w.shape[1]*w.shape[2]*w.shape[3], cov.shape[0]))
                        C = w.shape[0]
                        cands = torch.stack([w.clone()] + [torch.empty_like(w) for _ in range(k - 1)], 0)
                        for j in range(1, k):
                            nn.init.kaiming_normal_(cands[j], mode="fan_in", nonlinearity="relu")
                        flat = cands.flatten(2)                                   # [k, C, D]
                        sig2 = torch.einsum("kcd,de,kce->kc", flat, cov.to(flat.dtype), flat) / flat.pow(2).sum(-1)
                        if rule == "min": pick = sig2.argmin(0)
                        elif rule == "max": pick = sig2.argmax(0)
                        elif rule == "median": pick = (sig2 - sig2.median(0).values.unsqueeze(0)).abs().argmin(0)
                        elif rule == "mean": pick = (sig2 - sig2.mean(0, keepdim=True)).abs().argmin(0)
                        else: raise ValueError("bad rule %r" % rule)
                        r0 = flat[0].norm(dim=1)
                        chosen = flat[pick, torch.arange(C)]                       # [C, D]
                        chosen = chosen * (r0 / chosen.norm(dim=1).clamp_min(1e-12)).unsqueeze(1)
                        w.copy_(chosen.view_as(w))
            if init_whitened_angle:
                import math
                import os
                cfga = dict(init_whitened_angle)
                cov = torch.load(os.path.join(os.path.dirname(__file__), "cifar100_patch_cov.pt"), map_location="cpu")
                theta = math.radians(float(cfga.get("theta_deg", 90.0)))
                ct, st = math.cos(theta), math.sin(theta)
                for bi in cfga.get("blocks", [0]):
                    blk = blocks_flat[int(bi)]
                    if len(blk.convs) < 2:
                        raise ValueError(f"init_whitened_angle: block {bi} has fewer than 2 branches")
                    w1 = blk.convs[0].weight
                    D = w1.shape[1] * w1.shape[2] * w1.shape[3]
                    if D != cov.shape[0]:
                        raise ValueError(f"init_whitened_angle: block {bi} kernel dim {D} != covariance dim {cov.shape[0]} (block 0 only)")
                    Sg = cov.to(w1.dtype)
                    f1 = w1.flatten(1)                                            # [C, D]
                    s1 = torch.einsum("cd,de,ce->c", f1, Sg, f1).clamp_min(1e-12).sqrt()
                    for conv in blk.convs[1:]:
                        f2 = torch.empty_like(f1)
                        for c in range(f1.shape[0]):
                            for _ in range(20):
                                r = torch.randn(D, dtype=f1.dtype) * float(f1[c].std())
                                c_par = (f1[c] @ Sg @ r) / s1[c]                  # p1-component of S^1/2 r
                                sr2 = (r @ Sg @ r).clamp_min(1e-12)
                                d2 = sr2 - c_par ** 2
                                if d2 > 1e-6 * sr2:
                                    break
                            b = st / d2.clamp_min(1e-12).sqrt()
                            a = (ct - b * c_par) / s1[c]
                            v = a * f1[c] + b * r
                            f2[c] = v * (f1[c].norm() / v.norm().clamp_min(1e-12))
                        conv.weight.copy_(f2.view_as(conv.weight))
            if init_whitened_isotropic:
                import os
                cfgi = dict(init_whitened_isotropic)
                eps = float(cfgi.get("eps", 1e-2))
                match = str(cfgi.get("match", "sigma"))
                if match not in ("sigma", "norm"):
                    raise ValueError(f"init_whitened_isotropic: match must be 'sigma' or 'norm', got {match!r}")
                cov = torch.load(os.path.join(os.path.dirname(__file__), "cifar100_patch_cov.pt"), map_location="cpu")
                Sg = cov.to(torch.float64)
                ev, V = torch.linalg.eigh(Sg)
                S_reg_inv_half = V @ torch.diag((ev.clamp_min(0.0) + eps).rsqrt()) @ V.T
                for bi in cfgi.get("blocks", [0]):
                    blk = blocks_flat[int(bi)]
                    for conv in blk.convs:
                        w = conv.weight
                        D = w.shape[1] * w.shape[2] * w.shape[3]
                        if D != cov.shape[0]:
                            raise ValueError(f"init_whitened_isotropic: block {bi} kernel dim {D} != covariance dim {cov.shape[0]} (block 0 only)")
                        f = w.flatten(1).to(torch.float64)                            # the Kaiming kernel, [C, D]
                        u = torch.randn(f.shape, dtype=torch.float64)
                        g = u @ S_reg_inv_half                                        # rows: (Sigma + eps I)^{-1/2} u
                        if match == "sigma":
                            target = ((f @ Sg) * f).sum(1)
                            have = ((g @ Sg) * g).sum(1).clamp_min(1e-30)
                            g = g * (target / have).sqrt().unsqueeze(1)
                        else:
                            g = g * (f.norm(dim=1) / g.norm(dim=1).clamp_min(1e-30)).unsqueeze(1)
                        w.copy_(g.to(w.dtype).view_as(w))
            if float(pair_kernel_init_scale) != 1.0:
                for blk in blocks_flat:
                    if len(blk.convs) >= 2:
                        for c in blk.convs:
                            c.weight.mul_(float(pair_kernel_init_scale))

        # Direction-selective cooling of block 0's kernel gradient: scale its
        # component along the patch covariance's top eigenvector (the flat
        # brightness mode) by ``block0_damp_top``. That is the direction an open
        # pair removes from each branch's own tangent space by parking on it, so
        # this cools it in a single while changing nothing else about the cell.
        # 1.0 is a no-op, 0.0 freezes the direction. It hooks the loss gradient
        # only, so weight decay still acts on the damped component.
        # ``block0_damp_renorm`` restores each channel's gradient norm, making the
        # arm purely directional with no reduction in step size.
        # ``block0_damp_dir`` picks WHICH direction is scaled, so the same factor
        # can be applied somewhere else as a placebo: "top" is v_max (the default
        # and the only direction used before 2026-09-03), "next" is the runner-up
        # eigenvector of the same covariance, "random" is a fixed isotropic unit
        # vector, and "random_perp" is that vector projected off v_max and
        # renormalised, so it cools exactly none of v_max (a plain isotropic draw
        # in 27 dimensions still carries about 0.15 to 0.3 of v_max, which would
        # cool a few percent of the loud direction and blunt the contrast). The
        # draw uses a private generator seeded by ``block0_damp_dir_seed``, never
        # the global RNG, so every arm keeps the same kernel init and batch order.
        self.block0_damp_top = float(block0_damp_top)
        self.block0_damp_renorm = bool(block0_damp_renorm)
        self.block0_damp_dir = str(block0_damp_dir)
        if self.block0_damp_dir not in {"top", "next", "random", "random_perp"}:
            raise ValueError(
                "block0_damp_dir must be top, next, random or random_perp, "
                f"got {block0_damp_dir!r}"
            )
        if self.block0_damp_top != 1.0:
            import os  # os is function-local in this __init__ (see above)
            cov = torch.load(os.path.join(os.path.dirname(__file__), "cifar100_patch_cov.pt"),
                             map_location="cpu").float()
            w0 = blocks_flat[0].convs[0].weight
            D = w0.shape[1] * w0.shape[2] * w0.shape[3]
            if D != cov.shape[0]:
                raise ValueError(f"block0_damp_top: block 0 kernel dim {D} != covariance dim {cov.shape[0]}")
            ev, V = torch.linalg.eigh(cov.double())
            order = torch.argsort(ev, descending=True)
            if self.block0_damp_dir.startswith("random"):
                gen = torch.Generator().manual_seed(int(block0_damp_dir_seed))
                e = torch.randn(D, generator=gen, dtype=torch.float64)
                if self.block0_damp_dir == "random_perp":
                    vmax = V[:, int(order[0])]
                    e = e - (e @ vmax) * vmax
                e = (e / e.norm()).float()
            else:
                e = V[:, int(order[0 if self.block0_damp_dir == "top" else 1])].float()
            self.register_buffer("block0_damp_e", e if float(e.sum()) >= 0 else -e, persistent=False)
            c_keep, renorm = self.block0_damp_top, self.block0_damp_renorm

            def _damp_block0(grad: torch.Tensor) -> torch.Tensor:
                g = grad.flatten(1)
                ee = self.block0_damp_e.to(g.device, g.dtype)
                n0 = g.norm(dim=1, keepdim=True) if renorm else None
                g = g + torch.outer((g @ ee) * (c_keep - 1.0), ee)
                if renorm:
                    g = g * (n0 / g.norm(dim=1, keepdim=True).clamp_min(1e-12))
                return g.view_as(grad)

            for conv in blocks_flat[0].convs:
                conv.weight.register_hook(_damp_block0)

        # Whitening of block 0's patch space: train the kernel in the coordinate
        # u = (Sigma + eps I)^{1/2} w, which is what "divide the data by its own
        # covariance" means for this block, since u^T (Sigma+eps)^{-1/2} x =
        # ((Sigma+eps)^{-1/2} u)^T x.  Implemented as a gradient hook g -> g P with
        # P = (Sigma + eps I)^{-1}: for SGD with momentum and decay this is exactly
        # equivalent to training u (the decay term maps to itself, since
        # P^{1/2} (P^{1/2} g + lambda u) = P g + lambda w).
        # P is normalised to unit MEAN eigenvalue, so the arm reshapes the metric
        # without changing the average step size -- otherwise eps=0.01 would also be
        # a ~100x learning-rate change on the quiet directions.
        # eps sweeps from unwhitened (large) to fully whitened (small).
        self.block0_whiten_eps = None if block0_whiten_eps is None else float(block0_whiten_eps)
        if self.block0_whiten_eps is not None:
            import os  # os is function-local in this __init__ (see above)
            cov = torch.load(os.path.join(os.path.dirname(__file__), "cifar100_patch_cov.pt"),
                             map_location="cpu").float()
            w0 = blocks_flat[0].convs[0].weight
            D = w0.shape[1] * w0.shape[2] * w0.shape[3]
            if D != cov.shape[0]:
                raise ValueError(f"block0_whiten_eps: block 0 kernel dim {D} != covariance dim {cov.shape[0]}")
            ev, V = torch.linalg.eigh(cov.double())
            inv = 1.0 / (ev + float(self.block0_whiten_eps))
            inv = inv / inv.mean()                      # unit mean eigenvalue
            P = (V @ torch.diag(inv) @ V.T).float()
            self.register_buffer("block0_whiten_P", P, persistent=False)

            def _whiten_block0(grad: torch.Tensor) -> torch.Tensor:
                g = grad.flatten(1)
                return (g @ self.block0_whiten_P.to(g.device, g.dtype)).view_as(grad)

            for conv in blocks_flat[0].convs:
                conv.weight.register_hook(_whiten_block0)

        # ZCA whitening of the DATA, as opposed to the block-0 gradient above.
        # x <- (x - mu) (Sigma_img + eps I)^alpha on the flattened 32x32x3 image,
        # with input_zca_alpha = -1/2 (the default, and every config written before
        # 2026-09-06) the regularised ZCA whitening this block was written for.
        # Whitening the full image covariance drives every marginal to unit
        # covariance, and a 3x3 patch is a marginal, so this flattens the patch
        # spectrum block 0 sees -- but unlike block0_whiten_eps it changes what
        # EVERY layer sees, not just the first. eps regularises the inverse square
        # root; large eps is a no-op, small eps is full whitening.
        # A POSITIVE alpha runs the same family the other way and SHARPENS: each
        # image eigendirection's amplitude is multiplied by a positive power of its
        # own standard deviation, so the quiet band is suppressed relative to the
        # loud one and block 0's patch spectrum becomes more peaked. The block-seen
        # runner-up ratio then runs from CIFAR-100's 9.6 at alpha = 0 up to about 36,
        # where the family saturates; scripts/analysis/sharpen_calib.py is the
        # calibration and prints the ratio and mu_eff of any (alpha, eps).
        # The 3072x3072 eigendecomposition is cached next to the dataset, so it is
        # paid once rather than per run.
        self.input_zca_eps = None if input_zca_eps is None else float(input_zca_eps)
        if self.input_zca_eps is not None:
            import os  # os is function-local in this __init__ (see above)
            cache = os.path.join(input_zca_data_dir, "img_eig.pt")
            if os.path.exists(cache):
                d = torch.load(cache, map_location="cpu")
            else:
                import torchvision
                from structural_reparam.data.cifar100 import CIFAR100_MEAN, CIFAR100_STD
                ds = torchvision.datasets.CIFAR100(root=input_zca_data_dir, train=True, download=False)
                xx = torch.from_numpy(ds.data).float().permute(0, 3, 1, 2) / 255.0
                xx = (xx - torch.tensor(CIFAR100_MEAN).view(1, 3, 1, 1)) / torch.tensor(CIFAR100_STD).view(1, 3, 1, 1)
                fl = xx.reshape(xx.shape[0], -1)
                mu = fl.mean(0)
                fl = fl - mu
                Cimg = (fl.T @ fl) / fl.shape[0]
                evi, Vi = torch.linalg.eigh(Cimg.double())
                d = {"ev": evi, "V": Vi, "mu": mu}
                os.makedirs(input_zca_data_dir, exist_ok=True)
                torch.save(d, cache)
            evi, Vi, mu = d["ev"], d["V"], d["mu"]
            e = float(self.input_zca_eps)
            self.input_zca_alpha = float(input_zca_alpha)
            # unit-variance rescale, exact for the training set: the transformed
            # per-dimension variance is mean(lambda_i (lambda_i + eps)^{2 alpha}),
            # which at alpha = -1/2 is the whitened mean(lambda_i / (lambda_i + eps)).
            # The alpha = -1/2 branch keeps the original rsqrt arithmetic verbatim so
            # that every config written before 2026-09-06 reproduces bit for bit.
            if self.input_zca_alpha == -0.5:
                Wz = (Vi @ torch.diag((evi + e).rsqrt()) @ Vi.T).float()
                sc = float((evi / (evi + e)).mean().sqrt())
            else:
                ev0 = evi.clamp_min(0.0)
                pw = (ev0 + e).pow(self.input_zca_alpha)
                Wz = (Vi @ torch.diag(pw) @ Vi.T).float()
                sc = float((ev0 * pw.pow(2)).mean().sqrt())
            self.register_buffer("input_zca_W", Wz, persistent=False)
            self.register_buffer("input_zca_mu", mu.float(), persistent=False)
            self.input_zca_scale = max(sc, 1e-8)

    def _filter_input(self, x: torch.Tensor) -> torch.Tensor:
        if getattr(self, "input_zca_eps", None) is not None:
            n, shp = x.shape[0], x.shape
            W = self.input_zca_W.to(x.device, x.dtype)
            mu = self.input_zca_mu.to(x.device, x.dtype)
            x = ((x.reshape(n, -1) - mu) @ W / self.input_zca_scale).reshape(shp)
        a = self.input_filter_alpha
        band = self.input_filter_band
        if a == 0.0 and band is None:
            return x
        H, W = x.shape[-2:]
        fy = torch.fft.fftfreq(H, device=x.device, dtype=torch.float32).view(-1, 1)
        fx = torch.fft.fftfreq(W, device=x.device, dtype=torch.float32).view(1, -1)
        r = (fy ** 2 + fx ** 2).sqrt()
        r = r / r.max()
        if band is not None:
            lo, hi, sup = band
            g = torch.full_like(r, float(sup))
            g[(r >= lo) & (r <= hi)] = 1.0
            g[r < lo] = 0.0
        else:
            g = r.clamp_min(1e-8) ** a
            if a > 0:
                g = g.clone()
                g[0, 0] = 0.0                  # a > 0 removes the DC mode outright
        sd = x.std()
        y = torch.fft.ifft2(torch.fft.fft2(x.float()) * g).real
        return (y / y.std().clamp_min(1e-8) * sd).to(x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._filter_input(x)
        for stage in self.stages:
            x = stage(x)
        return self.fc(self.pool(x).flatten(1))


# ---------------------------------------------------------------------------
# Decayed (free-norm) cell: kernel-only weight decay with a per-block multiplier
# on the pair block's kernels, an optional gamma learning-rate multiplier, and
# per-module learning-rate overrides (for block-level continuation tests).
# ---------------------------------------------------------------------------


def _pair_kernel_ids(model: nn.Module) -> set[int]:
    return {id(c.weight) for m in model.modules()
            if hasattr(m, "convs") and len(m.convs) >= 2
            for c in m.convs if isinstance(c, nn.Conv2d)}


def build_kernel_wd_sgd(
    params: Iterable[nn.Parameter],
    model: nn.Module,
    lr: float,
    momentum: float = 0.0,
    weight_decay: float = 0.0,
    nesterov: bool = False,
    pair_wd_scale: float = 1.0,
    bn_gamma_lr_mult: float = 1.0,
    block_bias_lr_mult: float = 1.0,
    lr_overrides: dict[str, float] | None = None,
    wd_overrides: dict[str, float] | None = None,
    affine_overrides: dict[str, dict[str, float]] | None = None,
    **_: object,
) -> torch.optim.SGD:
    """Plain SGD (no projection). ``weight_decay`` applies to ndim>=2 weights
    only (conv kernels, fc.weight); the kernels of blocks with >= 2 branches get
    ``weight_decay * pair_wd_scale``; gammas (parameter names ending in
    ``.gamma``) get ``lr * bn_gamma_lr_mult``; block biases (``.betas.<i>``) get
    ``lr * block_bias_lr_mult`` (the fc bias is untouched); ``affine_overrides`` maps a
    module prefix to ``{"gamma": m_g, "beta": m_b}`` learning-rate multipliers for the
    gammas and block biases under that prefix only (composes with the global ones);
    ``lr_overrides`` maps a
    parameter-name prefix (module path, e.g. ``stages.2.0``) to a learning rate
    for every parameter under it; ``wd_overrides`` maps a prefix to a multiplier
    on the weight decay of the ndim>=2 weights under it (applied after
    ``pair_wd_scale``). Groups are keyed by (weight_decay, lr)."""
    pair_ids = _pair_kernel_ids(model)
    overrides = dict(lr_overrides or {})
    wd_over = dict(wd_overrides or {})
    groups: dict[tuple[float, float], list[nn.Parameter]] = {}
    n_pair = n_gamma = n_over = n_wdover = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        wd = float(weight_decay) if p.ndim >= 2 else 0.0
        if id(p) in pair_ids:
            wd *= float(pair_wd_scale)
            n_pair += 1
        for prefix, scale in wd_over.items():
            if p.ndim >= 2 and (name == prefix or name.startswith(prefix + ".")):
                wd *= float(scale)
                n_wdover += 1
                break
        plr = float(lr)
        if name.endswith(".gamma") and bn_gamma_lr_mult != 1.0:
            plr *= float(bn_gamma_lr_mult)
            n_gamma += 1
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
                n_over += 1
                break
        groups.setdefault((wd, plr), []).append(p)
    param_groups = [{"params": ps, "weight_decay": wd, "lr": plr} for (wd, plr), ps in groups.items()]
    LOGGER.info("kernel-WD SGD: %d groups; %d pair kernels at wd x%g; %d gammas at lr x%g; %d params with lr overrides %s; %d kernels with wd overrides %s",
                len(param_groups), n_pair, pair_wd_scale, n_gamma, bn_gamma_lr_mult, n_over, overrides or "", n_wdover, wd_over or "")
    return torch.optim.SGD(param_groups, lr=lr, momentum=momentum, nesterov=nesterov)


# ---------------------------------------------------------------------------
# Per-channel probe for shared-scale blocks: every epoch (and every ``step_every``
# steps during the first ``step_log_epochs`` epochs) it records, per block and
# channel, |gamma|, the bias sum, per-branch kernel norm, per-branch running
# sigma, per-branch batch sigma (epoch mean of the training batch std, and the
# last batch's), the branch cosine (Euclidean; Sigma-whitened for ``cov_blocks``),
# the output scale |gamma| sqrt(sum_ij rho_ij), and the derived magnifications
# gamma^2/||w||^2 and gamma^2/sigma^2 (per branch and summed) and ||w||^2/gamma.
# Block summaries go to W&B as scalars; the arrays are written to
# ``<output_dir>/pair_channel/<variant>_seed<s>.npz`` and uploaded at close().
# ---------------------------------------------------------------------------
import json as _json  # noqa: E402
import numpy as np  # noqa: E402

from structural_reparam.analysis.registry import ProbeContext, register_probe  # noqa: E402


def _q(t: torch.Tensor, q: float) -> float:
    return float(torch.quantile(t.float(), q))


@register_probe("pair_channel")
class PairChannelProbe:
    PROBE_NAME = "pair_channel"

    def __init__(self, model: nn.Module, out_path, variant: str, seed: int, group: str,
                 step_log_epochs: int = 3, step_every: int = 20, cov_blocks=(0,),
                 input_cov_batches: int = 20, dataset_cfg: dict | None = None) -> None:
        self.model = model
        self.out_path = out_path
        self.variant, self.seed, self.group = variant, int(seed), group
        self.step_log_epochs, self.step_every = int(step_log_epochs), int(step_every)
        self.blocks: list[tuple[int, str, nn.Module]] = []
        for name, m in model.named_modules():
            if isinstance(m, SharedScaleRepVGGBlock):
                self.blocks.append((len(self.blocks), name, m))
        self.cov: dict[int, torch.Tensor] = {}
        for bi in cov_blocks or ():
            try:
                self.cov[int(bi)] = self._input_cov(int(bi), int(input_cov_batches), dataset_cfg)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("pair_channel: input covariance for block %s skipped: %r", bi, exc)
        # per cov-block: Sigma^(1/2) and the top eigenvector, for the whitened
        # branch angle and the separation-axis alignment
        self._cov_aux: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for bi, S in self.cov.items():
            ev, V = torch.linalg.eigh(S.double())
            lam, V = ev.flip(0), V.flip(1)
            sh = (V @ torch.diag(lam.clamp_min(0).sqrt()) @ V.T).float()
            self._cov_aux[bi] = (sh, V[:, 0].float())
        # batch-sigma accumulators (per block, per branch) filled by forward hooks
        self._bvar_sum: dict[tuple[int, int], torch.Tensor] = {}
        self._bvar_n: dict[tuple[int, int], int] = {}
        self._bvar_last: dict[tuple[int, int], torch.Tensor] = {}
        self._step = 0
        self._epoch = 0
        self._handles = []
        for bi, _, blk in self.blocks:
            for j, st in enumerate(blk.stats):
                self._handles.append(st.register_forward_hook(self._make_hook(bi, j)))
        self.records: dict[str, list] = {"epoch": [], "step": []}   # lists of dicts of arrays
        self.step_records: list = []
        self.epoch_records: list = []
        self._record(epoch=0, step=0, kind="epoch")
        self._log_epoch0()

    # -- construction helpers -------------------------------------------------
    @classmethod
    def from_context(cls, ctx: ProbeContext) -> "PairChannelProbe":
        variant = ctx.variant.get("name", "variant")
        experiment = ctx.config.get("experiment", {}).get("name", "experiment")
        group = ctx.config.get("logging", {}).get("group", experiment)
        out_dir = ctx.output_dir / "pair_channel"
        out_dir.mkdir(parents=True, exist_ok=True)
        pc = ctx.probe_config or {}
        return cls(ctx.model, out_dir / f"{variant}_seed{ctx.seed}.npz", variant, ctx.seed, group,
                   step_log_epochs=pc.get("step_log_epochs", 3), step_every=pc.get("step_every", 20),
                   cov_blocks=pc.get("cov_blocks", [0]), input_cov_batches=pc.get("input_cov_batches", 20),
                   dataset_cfg=ctx.config.get("dataset"))

    def _input_cov(self, bi: int, n_batches: int, dataset_cfg: dict | None) -> torch.Tensor:
        """Covariance of the 3x3 input patches of block ``bi`` (centered), from
        ``n_batches`` training batches; only block 0 (image input) is supported
        without running the network, so other blocks use a forward pre-hook."""
        from structural_reparam.deploy.train import import_target  # local import, avoids cycles
        if dataset_cfg is None:
            raise RuntimeError("no dataset config")
        target = import_target(dataset_cfg["target"])
        args = dict(dataset_cfg.get("args", {}))
        args["num_workers"] = 0
        args["persistent_workers"] = False
        import inspect
        accepted = set(inspect.signature(target).parameters)
        args = {k: v for k, v in args.items() if k in accepted}
        loaders = target(**args)
        train_loader = loaders[0] if isinstance(loaders, (tuple, list)) else loaders
        blk = self.blocks[bi][2]
        dev = next(self.model.parameters()).device
        captured: list[torch.Tensor] = []
        h = blk.register_forward_pre_hook(lambda m, inp: captured.append(inp[0].detach()))
        was_training = self.model.training
        self.model.eval()
        cov = None
        n = 0
        try:
            with torch.no_grad():
                for k, batch in enumerate(train_loader):
                    if k >= n_batches:
                        break
                    x = batch[0].to(dev)
                    captured.clear()
                    self.model(x)
                    xin = captured[0]
                    patches = torch.nn.functional.unfold(xin, 3, padding=1, stride=blk.stride)  # [B, Cin*9, L]
                    patches = patches.transpose(1, 2).reshape(-1, patches.shape[1])
                    patches = patches - patches.mean(0, keepdim=True)
                    c = patches.T @ patches / patches.shape[0]
                    cov = c if cov is None else cov + c
                    n += 1
        finally:
            h.remove()
            self.model.train(was_training)
        return (cov / max(n, 1)).cpu()

    def _make_hook(self, bi: int, j: int):
        def hook(module, inp, out):
            if not module.training or module.last_var is None:
                return
            with torch.no_grad():
                v = module.last_var.detach()
                key = (bi, j)
                self._bvar_last[key] = v
                if key in self._bvar_sum:
                    self._bvar_sum[key] += v
                    self._bvar_n[key] += 1
                else:
                    self._bvar_sum[key] = v.clone()
                    self._bvar_n[key] = 1
                if bi == 0 and j == 0:
                    self._step += 1
                    if self._epoch < self.step_log_epochs and self._step % self.step_every == 0:
                        self._record(epoch=self._epoch + 1, step=self._step, kind="step")
        return hook

    # -- measurement -----------------------------------------------------------
    @torch.no_grad()
    def _snapshot(self) -> dict:
        out = {}
        for bi, name, blk in self.blocks:
            nb = len(blk.convs)
            g = blk.gamma.detach().abs().float().cpu()
            beta = sum(b.detach() for b in blk.betas).float().cpu()
            ws = [c.weight.detach().flatten(1).float() for c in blk.convs]
            norms = torch.stack([w.norm(dim=1) for w in ws], 0).cpu()                 # [nb, C]
            run_sig = torch.stack([(st.running_var + st.eps).sqrt() for st in blk.stats], 0).float().cpu()
            bsig_mean = torch.stack([((self._bvar_sum[(bi, j)] / self._bvar_n[(bi, j)]) + blk.stats[j].eps).sqrt()
                                     if (bi, j) in self._bvar_sum else torch.full_like(run_sig[0], float("nan"))
                                     for j in range(nb)], 0).float().cpu()
            bsig_last = torch.stack([(self._bvar_last[(bi, j)] + blk.stats[j].eps).sqrt()
                                     if (bi, j) in self._bvar_last else torch.full_like(run_sig[0], float("nan"))
                                     for j in range(nb)], 0).float().cpu()
            if nb >= 2:
                cos = ((ws[0] * ws[1]).sum(1) / (ws[0].norm(dim=1) * ws[1].norm(dim=1)).clamp_min(1e-12)).cpu()
                rho_sum = torch.full_like(cos, float(nb))
                for a in range(nb):
                    for b in range(nb):
                        if a != b:
                            rho_sum += ((ws[a] * ws[b]).sum(1) / (ws[a].norm(dim=1) * ws[b].norm(dim=1)).clamp_min(1e-12)).cpu()
                if bi in self.cov:
                    S = self.cov[bi].to(ws[0].device, ws[0].dtype)
                    a, b = ws[0], ws[1]
                    cos_sig = (((a @ S) * b).sum(1) / (((a @ S) * a).sum(1).clamp_min(1e-12).sqrt() * ((b @ S) * b).sum(1).clamp_min(1e-12).sqrt())).cpu()
                    sh, vt = self._cov_aux[bi]
                    sh = sh.to(ws[0].device, ws[0].dtype)
                    vt = vt.to(ws[0].device, ws[0].dtype)
                    y1, y2 = a @ sh, b @ sh
                    y1 = y1 / y1.norm(dim=1, keepdim=True).clamp_min(1e-12)
                    y2 = y2 / y2.norm(dim=1, keepdim=True).clamp_min(1e-12)
                    dax = y1 - y2
                    dax = dax / dax.norm(dim=1, keepdim=True).clamp_min(1e-12)
                    wangle = torch.rad2deg(torch.acos(cos_sig.clamp(-1, 1)))
                    align_vmax = (dax @ vt).abs().cpu()
                    # The unwhitened twin (paper Figure 14): the normalized difference
                    # of the two UNIT KERNELS in the plain inner product, against the
                    # same top eigenvector of Sigma (its whitened and unwhitened
                    # versions coincide). Added 2026-09-01.
                    r1 = a / a.norm(dim=1, keepdim=True).clamp_min(1e-12)
                    r2 = b / b.norm(dim=1, keepdim=True).clamp_min(1e-12)
                    dax_raw = r1 - r2
                    dax_raw = dax_raw / dax_raw.norm(dim=1, keepdim=True).clamp_min(1e-12)
                    align_vmax_raw = (dax_raw @ vt).abs().cpu()
                else:
                    cos_sig = torch.full_like(cos, float("nan"))
                    wangle = torch.full_like(cos, float("nan"))
                    align_vmax = torch.full_like(cos, float("nan"))
                    align_vmax_raw = torch.full_like(cos, float("nan"))
            else:
                cos = torch.ones_like(g); cos_sig = torch.full_like(g, float("nan")); rho_sum = torch.ones_like(g)
                wangle = torch.full_like(g, float("nan")); align_vmax = torch.full_like(g, float("nan"))
                align_vmax_raw = torch.full_like(g, float("nan"))
            output = g * rho_sum.clamp_min(0).sqrt()
            mag_k = (g[None, :] ** 2 / norms ** 2)                # per branch, kernel units
            mag_s = (g[None, :] ** 2 / run_sig ** 2)              # per branch, running-sigma units
            mag_b = (g[None, :] ** 2 / bsig_mean ** 2)            # per branch, batch-sigma units
            out[bi] = dict(name=name, nb=nb, gamma=g, beta=beta, norm=norms, run_sigma=run_sig,
                           batch_sigma_mean=bsig_mean, batch_sigma_last=bsig_last, cos=cos, cos_sigma=cos_sig,
                           wangle=wangle, align_vmax=align_vmax, align_vmax_raw=align_vmax_raw,
                           output=output, mag_k=mag_k, mag_s=mag_s, mag_b=mag_b,
                           mag_k_sum=mag_k.sum(0), mag_s_sum=mag_s.sum(0), mag_b_sum=mag_b.sum(0),
                           n2_over_gamma=norms[0] ** 2 / g.clamp_min(1e-12))
        return out

    def _record(self, epoch: int, step: int, kind: str) -> None:
        snap = self._snapshot()
        rec = {"epoch": epoch, "step": step, "blocks": snap}
        (self.epoch_records if kind == "epoch" else self.step_records).append(rec)

    def _scalars(self, snap: dict) -> dict[str, float]:
        s: dict[str, float] = {}
        for bi, d in snap.items():
            p = f"pair_channel/block{bi}/"
            s[p + "nb"] = float(d["nb"])
            for key in ("gamma", "output", "mag_k_sum", "mag_s_sum", "mag_b_sum", "n2_over_gamma"):
                t = d[key]
                s[p + key + "_p10"] = _q(t, 0.10); s[p + key + "_p50"] = _q(t, 0.50); s[p + key + "_p90"] = _q(t, 0.90)
                s[p + key + "_min"] = float(t.min()); s[p + key + "_max"] = float(t.max())
                s[p + key + "_p90_over_p10"] = _q(t, 0.90) / max(_q(t, 0.10), 1e-12)
            for j in range(d["nb"]):
                s[p + f"norm_b{j}_p50"] = _q(d["norm"][j], 0.5)
                s[p + f"norm_b{j}_p10"] = _q(d["norm"][j], 0.1)
                s[p + f"norm_b{j}_p90"] = _q(d["norm"][j], 0.9)
                s[p + f"run_sigma_b{j}_p50"] = _q(d["run_sigma"][j], 0.5)
                s[p + f"batch_sigma_b{j}_p50"] = _q(d["batch_sigma_mean"][j], 0.5)
                s[p + f"mag_k_b{j}_p50"] = _q(d["mag_k"][j], 0.5)
                s[p + f"mag_s_b{j}_p50"] = _q(d["mag_s"][j], 0.5)
            s[p + "beta_p50"] = _q(d["beta"], 0.5)
            if d["nb"] >= 2:
                c = d["cos"]
                s[p + "cos_p10"] = _q(c, 0.1); s[p + "cos_p50"] = _q(c, 0.5); s[p + "cos_p90"] = _q(c, 0.9)
                s[p + "cos_min"] = float(c.min()); s[p + "cos_frac_neg"] = float((c < 0).float().mean())
                cs = d["cos_sigma"]
                if not torch.isnan(cs).all():
                    s[p + "cos_sigma_p10"] = _q(cs, 0.1); s[p + "cos_sigma_p50"] = _q(cs, 0.5)
                    s[p + "cos_sigma_frac_neg"] = float((cs < 0).float().mean())
                wa = d["wangle"]
                if not torch.isnan(wa).all():
                    op = wa >= 10.0
                    s[p + "n_open"] = float(op.sum()); s[p + "n_closed"] = float((~op).sum())
                    if op.any():
                        s[p + "wangle_open_p10"] = _q(wa[op], 0.1)
                        s[p + "wangle_open_p50"] = _q(wa[op], 0.5)
                        s[p + "wangle_open_p90"] = _q(wa[op], 0.9)
                        av = d["align_vmax"][op]
                        s[p + "align_vmax_open_p10"] = _q(av, 0.1)
                        s[p + "align_vmax_open_p50"] = _q(av, 0.5)
                        s[p + "align_vmax_open_p90"] = _q(av, 0.9)
        return s

    def _log_epoch0(self) -> None:
        try:
            import wandb
            if wandb.run is not None:
                wandb.run.log(self._scalars(self.epoch_records[0]["blocks"]), step=0)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("pair_channel: epoch-0 W&B log skipped: %r", exc)

    def epoch_stats(self, epoch: int) -> dict[str, float]:
        self._epoch = int(epoch)
        self._record(epoch=int(epoch), step=self._step, kind="epoch")
        scal = self._scalars(self.epoch_records[-1]["blocks"])
        self._bvar_sum.clear(); self._bvar_n.clear()
        try:
            self._save()
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("pair_channel: save failed: %r", exc)
        return scal

    # -- persistence -------------------------------------------------------------
    def _save(self) -> None:
        arrays: dict[str, np.ndarray] = {}
        meta = {"variant": self.variant, "seed": self.seed, "group": self.group,
                "blocks": {bi: {"name": n, "nb": len(b.convs)} for bi, n, b in self.blocks}}
        for kind, recs in (("epoch", self.epoch_records), ("step", self.step_records)):
            if not recs:
                continue
            arrays[f"{kind}_epoch"] = np.array([r["epoch"] for r in recs])
            arrays[f"{kind}_step"] = np.array([r["step"] for r in recs])
            for bi, _, _ in self.blocks:
                for key in ("gamma", "beta", "norm", "run_sigma", "batch_sigma_mean", "batch_sigma_last", "cos",
                            "cos_sigma", "wangle", "align_vmax", "align_vmax_raw", "output", "mag_k", "mag_s", "mag_b",
                            "mag_k_sum", "mag_s_sum", "mag_b_sum", "n2_over_gamma"):
                    arrays[f"{kind}_block{bi}_{key}"] = np.stack([r["blocks"][bi][key].numpy() for r in recs], 0)
        np.savez_compressed(self.out_path, meta=_json.dumps(meta), **arrays)

    def close(self) -> None:
        for h in self._handles:
            h.remove()
        try:
            self._save()
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("pair_channel: final save failed: %r", exc)
            return
        try:
            import wandb
            if wandb.run is None:
                LOGGER.info("pair_channel: W&B disabled; arrays at %s", self.out_path)
                return
            import re
            name = re.sub(r"[^A-Za-z0-9_.-]", "_", f"pair_channel_{self.group}_{self.variant}_s{self.seed}")
            art = wandb.Artifact(name=name, type="probe", metadata={"group": self.group, "variant": self.variant, "seed": self.seed})
            art.add_file(str(self.out_path))
            wandb.run.log_artifact(art)
            art.wait()
            LOGGER.info("pair_channel: artifact %s uploaded", name)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("pair_channel: artifact upload failed: %r", exc)
