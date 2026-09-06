"""The offline pass: a recording (or checkpoints) in, derived arrays, a checks report
and summary tables out.

    python -m structural_reparam.analysis.blockmeasure.derive recording \\
        --recording outputs/blockmeasure_c100_d5_w1/blockmeasure/pair_all_seed42 \\
        --out outputs/blockmeasure_c100_d5_w1/derived/pair_all_seed42 [--frames fixed batch]

    python -m structural_reparam.analysis.blockmeasure.derive checkpoints \\
        --group stage2_place_c100_d3_w1 --variant pair_all --seeds 42 43 44 \\
        --epochs 0 5 25 50 75 100 --dataset-config configs/blockmeasure_c100_d5_w1.yaml \\
        --out outputs/blockmeasure_c100_d5_w1/derived/checkpoints_d3

    python -m structural_reparam.analysis.blockmeasure.derive reconcile \\
        --recording <dir> --probe-npz <pair_channel_open npz of the same run> --out <json>

Frames. "fixed" uses the recording's Sigma (the paper's convention, 50 batches of the
augmented stream at a fixed seed). "batch" uses, at each step, the covariance of the
batch that step actually saw, which is the frame in which BatchNorm's divisor and
the gradient identities are exact; the difference between the two frames is then a
measurement, not an assumption. Every array is written for each frame requested.

What is written per frame (``derived_<frame>.npz``, arrays over the T recorded steps
and C channels unless noted):
  geometry of the pre-step state: every scalar field of ``measure.Geometry`` with the
      prefix ``pre_`` (angle in degrees as ``pre_theta_deg``), plus the pre-step unit
      axis and bisector as ``pre_axis_hat``, ``pre_bisector_hat`` [T, C, D] float32;
  the same for the post-step state with the prefix ``post_``;
  forcing from the gradient (``grad_``) and from the momentum buffer (``buf_``): the
      scalar fields of ``measure.Forcing``;
  the decomposition of the applied step (``step_``), in per-step units. The applied
      step is -lr times torch's momentum buffer, and the buffer is the momentum filter
      of the past gradients, buffer_t = mu buffer_{t-1} + g_t + wd w_t. Each past
      gradient splits exactly into the kernel-space images of the components of its own
      r (see ``measure.kernel_vectors_from_r``), so the module keeps one running buffer
      per share, filtered with torch's own recursion from the first step of the
      recorded epoch, and pushes each share through the geometry of the current step.
      The shares are: from_bisector_component (the scale-gradient, or Oja, group),
      from_axis_component (the transfer group), from_out_of_plane (the gradient off the
      branch plane), from_vmax_component and from_quiet_component (the loud/quiet split
      of the same gradients), inconsistency (the part of each recorded gradient that no
      r reproduces, TF32 noise in the batch frame and the frame difference in the fixed
      frame), decay (the accumulated wd w terms, whose push is zero only for the current
      kernel), and prewindow (the buffer's memory from before the recorded epoch,
      recovered from the first recorded buffer and decaying as mu^j). For each share and
      for the total, ``step_<share>_d_overlap_axis``, ``_d_theta_deg``, ``_d_lambda_plus``
      and ``_d_logodds_alignment``. The primary shares plus inconsistency, decay and
      prewindow sum to the total exactly (check ``filtered_closure``), and the running
      buffers sum to the recorded buffer (check ``buffer_shares``). The sigma-mismatch
      split of the total, ``step_equal_sigma`` and ``step_sigma_mismatch``, is taken
      from the buffer directly (``measure.sigma_split_of_vectors``). The same from the
      instantaneous gradient alone times the learning rate (``gradstep_``), the old
      convention, for comparison;
  the realized change (``real_``): the fields of ``measure.Realized`` that are per
      channel, and ``real_d_gamma`` = gamma_post - gamma_pre with its first-order forms
      ``step_d_gamma`` (-lr_gamma * buffer) and ``gradstep_d_gamma`` (-lr_gamma * g);
  masks: ``open_pre``, ``parked_pre``, ``alignment_pre``;
  checks: ``check_<name>_error`` and ``check_<name>_flag`` for every check, and
      ``flags`` [T, C] uint32 with bit k set when check k flagged;
  bookkeeping: ``step``, ``epoch``, ``lr_w``, ``consecutive``.

``checks_<frame>.json`` is ``measure.checks_report`` plus the recording's provenance.
``summary_<frame>.json`` holds, per recorded epoch, summaries over open channels of
the main quantities and of the term groups, each with its population stated.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from structural_reparam.analysis.blockmeasure import BLOCKMEASURE_VERSION
from structural_reparam.analysis.blockmeasure import measure as M
from structural_reparam.analysis.blockmeasure.recorder import Recording, load_recording

GRAD_GROUPS = ("total", "from_bisector_component", "from_axis_component", "from_out_of_plane",
               "from_vmax_component", "from_quiet_component", "equal_sigma", "sigma_mismatch", "remainder")
"""Term groups of the instantaneous gradient (``measure.decompose``)."""

STEP_SHARES = ("from_bisector_component", "from_axis_component", "from_out_of_plane",
               "from_vmax_component", "from_quiet_component", "inconsistency", "decay", "prewindow")
"""Shares of the momentum buffer, each a running momentum filter of one share of the past gradients."""

STEP_GROUPS = ("total",) + STEP_SHARES + ("equal_sigma", "sigma_mismatch")
"""Everything written under the ``step_`` prefix."""

QUANTITIES = ("d_overlap_axis", "d_theta_deg", "d_lambda_plus", "d_logodds_alignment")

FORCING_SCALARS = ("r_axis", "r_bisector_kernel", "r_bisector_gamma", "r_bisector", "cond_axis",
                   "cond_bisector_kernel", "cond_bisector_gamma", "r_vmax", "r_axis_sigma", "r_out_norm", "K",
                   "gamma_dot", "gamma_dot_from_r", "drive", "res_reconstruct1", "res_reconstruct2", "res_out",
                   "res_bisector_routes", "radial1", "radial2")

REALIZED_SCALARS = ("flipped", "d_theta_deg", "d_overlap_axis", "d_alignment", "d_logodds_alignment",
                    "d_lambda_plus", "d_overlap_bisector")


def _quantities(geom: M.Geometry, dc: M.DirectionChange) -> dict[str, np.ndarray]:
    return {"d_overlap_axis": dc.d_overlap_axis, "d_theta_deg": dc.d_theta_deg,
            "d_lambda_plus": dc.d_lambda_plus, "d_logodds_alignment": M.first_order_logodds(geom, dc)}


def gradient_shares(geom: M.Geometry, cov: M.CovarianceEstimate, grad: M.Forcing, g1: np.ndarray, g2: np.ndarray,
                    gamma: np.ndarray) -> dict[str, np.ndarray]:
    """Split the recorded kernel gradients into the kernel-space images of the
    components of r, plus the part no r reproduces. Returns a dict share -> [2, C, D];
    the five component shares are two exact splits of the same reproduced gradient
    (bisector + axis + out_of_plane = vmax + quiet), and inconsistency = g - reproduced."""
    axis, bisector, v_max = geom.axis_hat, geom.bisector_hat, cov.v_max
    r = grad.r
    pieces = {
        "from_bisector_component": np.where(np.isfinite(grad.r_bisector), grad.r_bisector, 0.0)[:, None] * bisector,
        "from_axis_component": np.where(np.isfinite(grad.r_axis), grad.r_axis, 0.0)[:, None] * axis,
        "from_out_of_plane": grad.r_out,
    }
    r_v = (r @ v_max)[:, None] * v_max
    pieces["from_vmax_component"] = r_v
    pieces["from_quiet_component"] = r - r_v
    shares = {}
    for k, rr in pieces.items():
        k1, k2 = M.kernel_vectors_from_r(geom, cov, np.where(np.isfinite(rr), rr, 0.0), gamma)
        shares[k] = np.stack([k1, k2])
    full1, full2 = M.kernel_vectors_from_r(geom, cov, r, gamma)
    shares["inconsistency"] = np.stack([g1 - full1, g2 - full2])
    return shares


def euclid_quantities(w1: np.ndarray, w2: np.ndarray, v_max: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The Euclidean twins of the overlap and the angle, from kernels ``[C, D]``: the
    absolute overlap of the normalised difference of the unit kernels with v_max, and
    the plain angle between the kernels in degrees. Frame-independent yardsticks for
    the shares of the applied step, which are kernel-space vectors."""
    e1 = w1 / np.linalg.norm(w1, axis=-1, keepdims=True)
    e2 = w2 / np.linalg.norm(w2, axis=-1, keepdims=True)
    dd = e1 - e2
    n = np.linalg.norm(dd, axis=-1)
    overlap = np.where(n > 1e-12, np.abs(dd @ v_max) / np.maximum(n, 1e-12), np.nan)
    angle = np.degrees(np.arccos(np.clip(np.einsum("cd,cd->c", e1, e2), -1, 1)))
    return overlap, angle


def euclid_first_order(w1: np.ndarray, w2: np.ndarray, dw1: np.ndarray, dw2: np.ndarray, v_max: np.ndarray,
                       eps: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    """First-order change of the Euclidean overlap and angle for a kernel step
    (dw1, dw2), by a symmetric finite difference scaled by ``eps`` (the step is
    linear in the first-order limit, so this equals the Jacobian applied to the step
    up to a relative error of order eps^2)."""
    o_plus, a_plus = euclid_quantities(w1 + eps * dw1, w2 + eps * dw2, v_max)
    o_minus, a_minus = euclid_quantities(w1 - eps * dw1, w2 - eps * dw2, v_max)
    return (o_plus - o_minus) / (2 * eps), (a_plus - a_minus) / (2 * eps)


class MomentumFilter:
    """Running momentum buffers, one per share, following torch's recursion
    buffer <- mu * buffer + increment, started at the first step of a recorded epoch.
    ``prewindow`` holds the memory from before that step, recovered from the first
    recorded buffer as buffer_t0 - g_t0 - wd w_t0 and decayed by mu each step."""

    def __init__(self, momentum: float) -> None:
        self.mu = float(momentum)
        self.buffers: dict[str, np.ndarray] | None = None

    def step(self, shares: dict[str, np.ndarray], decay_term: np.ndarray, recorded_buffer: np.ndarray,
             gradient: np.ndarray, consecutive_with_previous: bool) -> dict[str, np.ndarray]:
        if self.buffers is None or not consecutive_with_previous:
            self.buffers = {k: s.copy() for k, s in shares.items()}
            self.buffers["decay"] = decay_term.copy()
            self.buffers["prewindow"] = recorded_buffer - gradient - decay_term
        else:
            for k, s in shares.items():
                self.buffers[k] = self.mu * self.buffers[k] + s
            self.buffers["decay"] = self.mu * self.buffers["decay"] + decay_term
            self.buffers["prewindow"] = self.mu * self.buffers["prewindow"]
        return self.buffers


# ---------------------------------------------------------------------------
# one step, one frame
# ---------------------------------------------------------------------------


def _step_covariance(rec: Recording, t: int, frame: str) -> M.CovarianceEstimate:
    if frame == "fixed":
        return rec.sigma
    if frame == "batch":
        sb = rec.arrays.get("sigma_batch")
        if sb is None or not np.isfinite(sb[t]).all():
            raise ValueError(f"step {t}: no batch covariance recorded; the batch frame needs record_batch_sigma")
        return M.CovarianceEstimate.from_matrix(sb[t], {"source": "batch", "step": int(rec.arrays["step"][t])})
    raise ValueError(f"unknown frame {frame!r}")


def _gradient_tolerance(rec: Recording) -> float:
    """TOL_GRADIENT_IDENTITY when the manifest says the convolutions ran without
    TF32 (a CPU run, or a GPU with TF32 disabled); TOL_GRADIENT_IDENTITY_TF32 when
    TF32 was allowed on a GPU, or when the manifest predates the flag and the device
    is unknown, which is reported in the checks provenance."""
    meta = rec.meta
    device = str(meta.get("device", "unknown"))
    tf32 = meta.get("tf32")
    if device.startswith("cpu"):
        return M.TOL_GRADIENT_IDENTITY
    if tf32 is not None and not tf32.get("cudnn_allow_tf32", True):
        return M.TOL_GRADIENT_IDENTITY
    return M.TOL_GRADIENT_IDENTITY_TF32


def derive_step(rec: Recording, t: int, frame: str) -> tuple[dict[str, np.ndarray], list[M.CheckResult], dict[str, Any]]:
    """Everything for one recorded step in one frame. Returns per-channel arrays and
    the checks evaluated at this step (each with a [C] error array)."""
    a = rec.arrays
    cov = _step_covariance(rec, t, frame)
    f64 = lambda x: np.asarray(x, dtype=np.float64)
    pre = M.BlockState(w1=f64(a["w_pre"][t, 0]), w2=f64(a["w_pre"][t, 1]), gamma=f64(a["gamma_pre"][t]))
    post = M.BlockState(w1=f64(a["w_post"][t, 0]), w2=f64(a["w_post"][t, 1]), gamma=f64(a["gamma_post"][t]))
    g_pre = M.geometry(pre, cov)
    g_post = M.geometry(post, cov)
    lr = float(a["lr_w"][t])
    out: dict[str, np.ndarray] = {}
    for prefix, g in (("pre_", g_pre), ("post_", g_post)):
        for k, v in g.scalars().items():
            out[prefix + k] = v
        out[prefix + "axis_hat"] = g.axis_hat.astype(np.float32)
        out[prefix + "bisector_hat"] = g.bisector_hat.astype(np.float32)
    out["pre_gamma"] = pre.gamma
    out["post_gamma"] = post.gamma
    gnorm = np.sqrt(np.linalg.norm(f64(a["g_w"][t, 0]), axis=-1) ** 2 + np.linalg.norm(f64(a["g_w"][t, 1]), axis=-1) ** 2)
    out["grad_norm"] = gnorm
    out["dead"] = gnorm < M.DEAD_GRADIENT_RELATIVE * np.median(gnorm)
    out["open_pre"] = M.open_mask(g_pre)
    out["parked_pre"] = M.parked_mask(g_pre)
    out["alignment_pre"] = M.alignment(g_pre)

    checks: list[M.CheckResult] = []
    # forcing from the gradient, and from the buffer
    grad = M.forcing(g_pre, cov, f64(a["g_w"][t, 0]), f64(a["g_w"][t, 1]), f64(a["g_gamma"][t]), pre.gamma, kind="gradient")
    buf_gamma = f64(a["buf_gamma"][t])
    buf = M.forcing(g_pre, cov, f64(a["buf_w"][t, 0]), f64(a["buf_w"][t, 1]),
                    buf_gamma if np.isfinite(buf_gamma).all() else None, pre.gamma, kind="buffer")
    for prefix, f in (("grad_", grad), ("buf_", buf)):
        for k in FORCING_SCALARS:
            v = getattr(f, k)
            out[prefix + k] = np.full(g_pre.theta_rad.shape, np.nan) if v is None else v
    # the instantaneous gradient's term groups times the learning rate (the old convention)
    dec = M.decompose(g_pre, cov, grad, pre.gamma, scale=lr)
    for name in GRAD_GROUPS:
        dc = getattr(dec, name)
        for q, val in _quantities(g_pre, dc).items():
            out[f"gradstep_{name}_{q}"] = val
    checks.extend(M.check_closure(dec))
    checks.append(M.check_mismatch_split(dec))
    real = M.realized_change(g_pre, g_post)
    for k in REALIZED_SCALARS:
        out["real_" + k] = getattr(real, k)
    # gamma: the realized change of the scale in this step, and its two first-order
    # forms (the applied step is -lr_gamma * buffer; the gradient alone is -lr_gamma * g)
    lr_gamma = float(a["lr_gamma"][t])
    out["real_d_gamma"] = post.gamma - pre.gamma
    out["step_d_gamma"] = -lr_gamma * buf_gamma
    out["gradstep_d_gamma"] = -lr_gamma * f64(a["g_gamma"][t])
    # the same form as the kernel check: gamma_post against gamma_pre - lr_gamma * buffer,
    # relative to gamma itself (comparing the two small differences would be dominated
    # by float32 cancellation, since gamma is about 0.7 and its step about 1e-3)
    checks.append(M._make("applied_step_gamma", "exact", M.TOL_FLOAT32,
                          np.abs(post.gamma - (pre.gamma - lr_gamma * buf_gamma)) / (np.abs(post.gamma) + 1e-300),
                          True, "gamma_post must equal gamma_pre minus the gamma learning rate times the gamma momentum buffer"))
    # identities
    checks.append(M.check_orthogonality(g_pre))
    # the gradient identities are exact only in the frame BatchNorm used; in the
    # fixed frame their error is the frame difference and is reported, not flagged
    exact = frame == "batch"
    tol = _gradient_tolerance(rec)
    checks.append(M.check_recovery(grad, exact=exact, tol=tol))
    checks.append(M.check_recovery(buf, exact=False, tol=tol))
    checks.append(M.check_gamma_gradient(grad, exact=exact, tol=tol))
    dw1 = f64(a["w_post"][t, 0]) - f64(a["w_pre"][t, 0])
    dw2 = f64(a["w_post"][t, 1]) - f64(a["w_pre"][t, 1])
    c7, plain = M.check_first_order_directions(g_pre, g_post, cov, dw1, dw2)
    checks.append(c7)
    out["first_order_plain_residual"] = plain
    ctx = {"geom": g_pre, "cov": cov, "grad": grad, "gamma": pre.gamma, "lr": lr,
           "g": np.stack([f64(a["g_w"][t, 0]), f64(a["g_w"][t, 1])]),
           "buf": np.stack([f64(a["buf_w"][t, 0]), f64(a["buf_w"][t, 1])]),
           "w_pre": np.stack([pre.w1, pre.w2]), "w_post": np.stack([post.w1, post.w2]), "wd": float(a["wd_w"][t])}
    return out, checks, ctx


# ---------------------------------------------------------------------------
# a whole recording
# ---------------------------------------------------------------------------


def derive_recording(rec: Recording, frame: str, progress: bool = True) -> tuple[dict[str, np.ndarray], list[M.CheckResult]]:
    """Run ``derive_step`` over every recorded step and stack. The applied-step and
    momentum-recurrence checks, which need whole arrays, are added here."""
    a = rec.arrays
    T = rec.n_steps
    rows: list[dict[str, np.ndarray]] = []
    check_rows: list[list[M.CheckResult]] = []
    mu = float(rec.meta.get("optimizer", {}).get("momentum", 0.9))
    filt = MomentumFilter(mu)
    for t in range(T):
        o, c, ctx = derive_step(rec, t, frame)
        # the applied step, share by share, through the momentum filter of the past gradients
        geom, cov, lr = ctx["geom"], ctx["cov"], ctx["lr"]
        shares = gradient_shares(geom, cov, ctx["grad"], ctx["g"][0], ctx["g"][1], ctx["gamma"])
        bufs = filt.step(shares, ctx["wd"] * ctx["w_pre"], ctx["buf"], ctx["g"], bool(rec.consecutive[t - 1]) if t > 0 else False)
        total = M.direction_change(geom, cov, *M.branch_pushes(geom, cov, ctx["buf"][0], ctx["buf"][1]))
        pieces: dict[str, M.DirectionChange] = {}
        for k in STEP_SHARES:
            pieces[k] = M.direction_change(geom, cov, *M.branch_pushes(geom, cov, bufs[k][0], bufs[k][1]))
        eq, mis = M.sigma_split_of_vectors(geom, cov, ctx["buf"][0], ctx["buf"][1])
        pieces["equal_sigma"], pieces["sigma_mismatch"] = eq, mis
        for k, dc in [("total", total)] + list(pieces.items()):
            for q, val in _quantities(geom, dc).items():
                o[f"step_{k}_{q}"] = lr * val
        # the same shares projected on the Euclidean overlap and angle (frame check)
        w1, w2 = ctx["w_pre"][0], ctx["w_pre"][1]
        for k, vec in [("total", ctx["buf"])] + [(kk, bufs[kk]) for kk in STEP_SHARES]:
            do, da = euclid_first_order(w1, w2, -lr * vec[0], -lr * vec[1], cov.v_max)
            o[f"step_{k}_d_overlap_axis_euclid"] = do
            o[f"step_{k}_d_theta_euclid_deg"] = da
        o_pre, a_pre = euclid_quantities(w1, w2, cov.v_max)
        w1p, w2p = ctx["w_post"][0], ctx["w_post"][1]
        o_post, a_post = euclid_quantities(w1p, w2p, cov.v_max)
        o["real_d_overlap_axis_euclid"] = o_post - o_pre
        o["real_d_theta_euclid_deg"] = a_post - a_pre
        o["overlap_axis_euclid_pre"] = o_pre
        # checks: the share buffers sum to the recorded buffer; the primary shares' pushes sum to the total
        summed = sum(bufs[k] for k in ("from_bisector_component", "from_axis_component", "from_out_of_plane",
                                        "inconsistency", "decay", "prewindow"))
        c.append(M._make("buffer_shares", "exact", M.TOL_FLOAT32,
                         np.linalg.norm(summed - ctx["buf"], axis=-1).max(0) / (np.linalg.norm(ctx["buf"], axis=-1).max(0) + 1e-300),
                         True, "the momentum-filtered shares of the past gradients must add up to the recorded buffer"))
        c.append(M._make("filtered_closure", "exact", M.TOL_FILTERED_CLOSURE,
                         M._closure_error([pieces[k] for k in ("from_bisector_component", "from_axis_component", "from_out_of_plane",
                                                               "inconsistency", "decay", "prewindow")], total),
                         total.defined, "the pushes of the shares must sum to the push of the recorded buffer"))
        c.append(M._make("filtered_closure_loud_quiet", "exact", M.TOL_FILTERED_CLOSURE,
                         M._closure_error([pieces[k] for k in ("from_vmax_component", "from_quiet_component",
                                                               "inconsistency", "decay", "prewindow")], total),
                         total.defined, "the loud and quiet shares with the non-r shares must sum to the total"))
        c.append(M._make("step_sigma_split", "exact", M.TOL_ALGEBRA, M._closure_error([eq, mis], total), total.defined,
                         "the equal-sigma and mismatch parts of the applied step must sum to its total"))
        rows.append(o)
        check_rows.append(c)
        if progress and (t % 500 == 0 or t == T - 1):
            print(f"  [{frame}] step {t + 1}/{T}", file=sys.stderr, flush=True)
    out = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
    out["step"] = a["step"]
    out["epoch"] = a["epoch"]
    out["lr_w"] = a["lr_w"]
    out["consecutive"] = rec.consecutive
    # stack the per-step checks into [T, C] checks
    checks: list[M.CheckResult] = []
    for i, first in enumerate(check_rows[0]):
        checks.append(M.CheckResult(
            name=first.name, level=first.level, tolerance=first.tolerance,
            error=np.stack([cr[i].error for cr in check_rows]),
            flag=np.stack([cr[i].flag for cr in check_rows]),
            conditioned=np.stack([cr[i].conditioned for cr in check_rows]), note=first.note))
    # whole-array checks on the primitives (frame independent)
    f64 = lambda x: np.asarray(x, dtype=np.float64)
    step_errs = []
    for j in range(2):
        step_errs.append(M.check_applied_step(f64(a["w_pre"][:, j]), f64(a["w_post"][:, j]), f64(a["buf_w"][:, j]), a["lr_w"]).error)
    checks.append(M._make("applied_step", "exact", M.TOL_FLOAT32, np.maximum(*step_errs), True,
                          "w_post must equal w_pre minus the learning rate times the momentum buffer that was applied"))
    cons = rec.consecutive[:-1]
    if cons.any():
        mu = float(rec.meta.get("optimizer", {}).get("momentum", 0.9))
        errs = np.full((T, rec.n_channels), np.nan)
        for j in range(2):
            r = M.check_momentum_recurrence(f64(a["buf_w"][1:, j]), f64(a["buf_w"][:-1, j]), f64(a["g_w"][1:, j]),
                                            f64(a["w_pre"][1:, j]), float(a["wd_w"][1]), mu).error
            e = np.where(cons[:, None], r, np.nan)
            errs[1:] = np.fmax(errs[1:], e) if j else e
        checks.append(M._make("momentum_recurrence", "exact", M.TOL_FLOAT32, errs, True,
                              "the momentum buffer must follow mu * previous + gradient + weight decay * weights on consecutive steps"))
    # dead channel-steps (no gradient) make every ratio of pushes meaningless: their
    # errors are set to NaN and never flagged; the dead mask itself is stored
    dead = out["dead"]
    for c in checks:
        c.error = np.where(dead, np.nan, c.error)
        c.flag = c.flag & ~dead
    flags = np.zeros((T, rec.n_channels), dtype=np.uint32)
    for k, c in enumerate(checks):
        out[f"check_{c.name}_error"] = c.error.astype(np.float32)
        out[f"check_{c.name}_flag"] = c.flag
        flags |= (c.flag.astype(np.uint32) << np.uint32(k))
    out["flags"] = flags
    return out, checks


def summarize_recording(out: dict[str, np.ndarray], frame: str, seed_label: str) -> dict[str, Any]:
    """Per recorded epoch, over open channels: the main quantities and the term groups.
    Every row states its population."""
    epochs = np.unique(out["epoch"])
    rows: list[dict[str, Any]] = []
    clean = out["flags"] == 0
    for e in epochs:
        m = out["epoch"] == e
        op = out["open_pre"] & m[:, None]
        pop = f"open channels (whitened angle >= {M.OPEN_ANGLE_DEG} deg) at every recorded step of epoch {int(e)}, {seed_label}, frame {frame}"
        for name in ("pre_theta_deg", "alignment_pre", "pre_lambda_plus", "pre_lambda_minus", "pre_B",
                     "pre_sigma_ratio", "pre_variance_share_axis", "pre_cos_theta_euclid"):
            rows.append(M.summarize(out[name], op, name, pop).as_row())
        last = np.flatnonzero(m)[-1]
        pop_end = f"open channels at the last recorded step of epoch {int(e)}, {seed_label}, frame {frame}"
        rows.append(M.summarize(out["post_theta_deg"][last], out["open_pre"][last], "end_theta_deg", pop_end).as_row())
        rows.append(M.summarize(np.abs(out["post_overlap_axis"][last]), out["open_pre"][last], "end_alignment", pop_end).as_row())
        rows.append({"name": "n_open_end", "population": pop_end, "n": int(out["open_pre"][last].sum())})
        rows.append({"name": "n_parked_end", "population": pop_end, "n": int(out["parked_pre"][last].sum())})
        # per-step motion over open, clean channel-steps: realized against first order, and the term groups
        mc = op & clean
        popm = f"open channel-steps with no flagged check, epoch {int(e)}, {seed_label}, frame {frame}"
        for q in ("d_theta_deg", "d_overlap_axis", "d_logodds_alignment", "d_lambda_plus"):
            rows.append(M.summarize(out["real_" + q], mc, "real_" + q, popm).as_row())
            for g in STEP_GROUPS:
                rows.append(M.summarize(out[f"step_{g}_{q}"], mc, f"step_{g}_{q}", popm).as_row())
            for g in GRAD_GROUPS:
                rows.append(M.summarize(out[f"gradstep_{g}_{q}"], mc, f"gradstep_{g}_{q}", popm).as_row())
        for q in ("real_d_gamma", "step_d_gamma", "gradstep_d_gamma"):
            rows.append(M.summarize(out[q], mc, q, popm).as_row())
        for q in ("grad_K", "grad_gamma_dot", "grad_drive", "grad_r_vmax", "grad_res_reconstruct1", "buf_res_reconstruct1",
                  "first_order_plain_residual"):
            rows.append(M.summarize(out[q], mc, q, popm).as_row())
    return {"frame": frame, "seed_label": seed_label, "blockmeasure_version": BLOCKMEASURE_VERSION,
            "conventions": M.CONVENTIONS, "rows": rows}


def run_recording(recording_dir: Path, out_dir: Path, frames: list[str]) -> None:
    rec = load_recording(recording_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    label = f"{rec.meta.get('variant')} seed {rec.meta.get('seed')}"
    print(f"recording {recording_dir}: {rec.n_steps} steps, {rec.n_channels} channels, epochs {rec.epochs}", file=sys.stderr)
    for frame in frames:
        arrays, checks = derive_recording(rec, frame)
        np.savez(out_dir / f"derived_{frame}.npz", **arrays)
        report = M.checks_report(checks)
        report["provenance"] = {"recording": str(recording_dir), "meta": rec.meta, "sigma_sha256": rec.sigma.sha256,
                                "frame": frame, "blockmeasure_version": BLOCKMEASURE_VERSION,
                                "gradient_identity_tolerance": _gradient_tolerance(rec),
                                "tf32_known": rec.meta.get("tf32") is not None}
        (out_dir / f"checks_{frame}.json").write_text(json.dumps(report, indent=2, default=str))
        (out_dir / f"summary_{frame}.json").write_text(json.dumps(summarize_recording(arrays, frame, label), indent=2, default=str))
        print(f"  wrote {out_dir}/derived_{frame}.npz, checks_{frame}.json, summary_{frame}.json", file=sys.stderr)
        for c in checks:
            s = c.summary()
            print(f"  check {c.name:28s} [{c.level:10s}] flagged {100 * s['fraction_flagged']:.3f} %  "
                  f"conditioned p50 {s['conditioned']['p50']:.2e} p99 {s['conditioned']['p99']:.2e} (n={s['conditioned']['n']})  "
                  f"ill-conditioned p50 {s['ill_conditioned']['p50']:.2e} (n={s['ill_conditioned']['n']})", file=sys.stderr)


# ---------------------------------------------------------------------------
# checkpoints
# ---------------------------------------------------------------------------


def run_checkpoints(group: str, variant: str, seeds: list[int], epochs: list[int], dataset_config: Path,
                    out_dir: Path, block_prefix: str = "stages.0.0") -> None:
    """Geometry only, on W&B checkpoints, with the paper's covariance convention;
    replaces the checkpoint paths of the old figure scripts."""
    import yaml
    import torch
    from structural_reparam.analysis.ckpt_loader import fetch_checkpoint

    cfg = yaml.safe_load(Path(dataset_config).read_text())["dataset"]
    cov = M.estimate_patch_covariance(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    cov.save(out_dir / "sigma")
    rows: list[dict[str, Any]] = []
    per_channel: dict[str, np.ndarray] = {}
    for seed in seeds:
        for ep in epochs:
            path = fetch_checkpoint(group, f"{variant}_s{seed}", ep, seed=seed)
            sd = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
            g = M.geometry(M.BlockState.from_state_dict(sd, block_prefix), cov)
            op = M.open_mask(g)
            pop = f"open channels of {group} {variant} seed {seed} epoch {ep}, block {block_prefix}"
            for name, v in (("theta_deg", g.theta_deg), ("alignment", M.alignment(g)), ("lambda_plus", g.lambda_plus),
                            ("lambda_minus", g.lambda_minus), ("B", g.B), ("cos_theta_euclid", g.cos_theta_euclid)):
                rows.append(M.summarize(v, op, name, pop).as_row())
                per_channel[f"s{seed}_ep{ep}_{name}"] = v
            rows.append({"name": "n_open", "population": pop, "n": int(op.sum())})
            per_channel[f"s{seed}_ep{ep}_open"] = op
    np.savez(out_dir / "checkpoint_geometry.npz", **per_channel)
    (out_dir / "summary.json").write_text(json.dumps({"group": group, "variant": variant, "seeds": seeds, "epochs": epochs,
                                                      "sigma_sha256": cov.sha256, "conventions": M.CONVENTIONS, "rows": rows},
                                                     indent=2, default=str))
    print(f"wrote {out_dir}/checkpoint_geometry.npz and summary.json", file=sys.stderr)


# ---------------------------------------------------------------------------
# reconciliation against the old per-epoch probe
# ---------------------------------------------------------------------------


def run_reconcile(recording_dir: Path, probe_npz: Path, out_path: Path) -> None:
    """Compare, at the last recorded step of each recorded epoch, the geometry from the
    recording's post-step weights (fixed frame) with what the pair_channel_open
    probe logged for the same epoch. The weights are the same, so every difference
    is the covariance estimator's (20 unseeded batches in the old probe against the
    recording's 50 seeded batches)."""
    rec = load_recording(recording_dir)
    d = np.load(probe_npz, allow_pickle=True)
    probe_epochs = d["epoch_epoch"]
    rows = []
    a = rec.arrays
    for e in rec.epochs:
        m = np.flatnonzero(a["epoch"] == e)
        t = m[-1]
        st = M.BlockState(w1=a["w_post"][t, 0].astype(np.float64), w2=a["w_post"][t, 1].astype(np.float64),
                          gamma=a["gamma_post"][t].astype(np.float64))
        g = M.geometry(st, rec.sigma)
        j = np.flatnonzero(probe_epochs == e)
        if j.size == 0:
            continue
        j = j[0]
        old_angle = d["epoch_block0_wangle"][j]
        old_align = d["epoch_block0_align_vmax"][j]
        op_new = M.open_mask(g)
        op_old = old_angle >= M.OPEN_ANGLE_DEG
        rows.append({
            "epoch": int(e),
            "n_open_new": int(op_new.sum()), "n_open_old": int(op_old.sum()),
            "angle_median_new": float(np.median(g.theta_deg[op_new])) if op_new.any() else None,
            "angle_median_old": float(np.median(old_angle[op_old])) if op_old.any() else None,
            "alignment_median_new": float(np.median(M.alignment(g)[op_new])) if op_new.any() else None,
            "alignment_median_old": float(np.median(old_align[op_old])) if op_old.any() else None,
            "per_channel_angle_abs_diff_max_deg": float(np.max(np.abs(g.theta_deg - old_angle))),
            "per_channel_angle_abs_diff_median_deg": float(np.median(np.abs(g.theta_deg - old_angle))),
            "per_channel_alignment_abs_diff_max": float(np.max(np.abs(M.alignment(g) - old_align))),
        })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"recording": str(recording_dir), "probe_npz": str(probe_npz),
                                    "note": "same weights; differences are the covariance estimator's", "rows": rows}, indent=2))
    for r in rows:
        print(r, file=sys.stderr)



# ---------------------------------------------------------------------------
# report: magnitude and coherence of every term, fast clock and slow clock
# ---------------------------------------------------------------------------
# A term may be neglected because it is small, or because it has no coherent sign,
# and the two are different claims. For every per-step quantity the report gives
# both, on two clocks: the fast clock is each channel's own climb (its first step
# above CLIMB_LOW to its first step at or above CLIMB_HIGH, in the consecutive
# recorded epochs 1 to 5), and the slow clock is each recorded epoch over open,
# unflagged channel-steps, plus the net over training.
#
#   magnitude   median |x| over the channel-steps of the window
#   coherence   mean(x) / std(x) over those channel-steps (per-step signal to noise)
#   net share   |sum x| / sum |x| over the window: 1 for a term that never changes
#               sign, 0 for a pure fluctuation
#   fraction +  fraction of channel-steps with x > 0
# On the fast clock these are computed per climbing channel and the median over
# channels is printed, with the number of climbing channels.


@dataclass
class Pooled:
    """Derived arrays of several seeds concatenated along the step axis, with the
    seed of every row and the epoch structure preserved."""
    arrays: dict[str, np.ndarray]
    seed_of_row: np.ndarray
    seeds: list[int]
    frame: str
    open_clean: np.ndarray        # [T, C] open and no flagged check
    climbs: list[tuple[int, int, int, int]]   # (seed, channel, start row, end row) in pooled rows
    preclimbs: list[tuple[int, int, int, int]] = field(default_factory=list)   # the same channels, from the first recorded step to the step before the climb


def load_pooled(derived_dirs: list[Path], frame: str) -> Pooled:
    parts, seeds, seed_rows = [], [], []
    for d in derived_dirs:
        a = dict(np.load(d / f"derived_{frame}.npz"))
        seed = int(json.loads((d / f"checks_{frame}.json").read_text())["provenance"]["meta"]["seed"])
        parts.append(a); seeds.append(seed); seed_rows.append(np.full(a["step"].shape[0], seed))
    keys = set.intersection(*(set(p) for p in parts))
    arrays = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    seed_of_row = np.concatenate(seed_rows)
    open_clean = arrays["open_pre"] & (arrays["flags"] == 0) & ~arrays["dead"]
    # climb windows on the consecutive early epochs of each seed
    climbs, preclimbs = [], []
    offset = 0
    for p, seed in zip(parts, seeds):
        early = np.flatnonzero(np.isin(p["epoch"], [1, 2, 3, 4, 5]))
        w = M.climb_windows(p["alignment_pre"][early])
        for ch, (s0, s1) in enumerate(w):
            if s0 >= 0:
                climbs.append((seed, ch, offset + early[s0], offset + early[s1]))
                if s0 >= 10:
                    preclimbs.append((seed, ch, offset + early[0], offset + early[s0] - 1))
        offset += p["step"].shape[0]
    return Pooled(arrays, seed_of_row, seeds, frame, open_clean, climbs, preclimbs)


def _per_channel(x: np.ndarray, mask: np.ndarray, min_steps: int = 20) -> list[dict[str, float]]:
    """Two-level statistics: for every channel, ``_stats`` over that channel's masked
    steps (at least ``min_steps``), returned as one dict per channel. Every quantity
    is formed per channel-step before this (a ratio at each step, never a ratio of
    averages), so the per-channel value is the channel's own statistic, and the
    caller summarises across channels."""
    out = []
    for ch in range(x.shape[1]):
        v = x[mask[:, ch], ch]
        v = v[np.isfinite(v)]
        if v.size >= min_steps:
            out.append(_stats(v))
    return out


def _across(rows: list[dict[str, float]], key: str) -> tuple[float, float, float]:
    """Median and quartiles across channels of a per-channel statistic."""
    if not rows:
        return (np.nan, np.nan, np.nan)
    vals = np.array([r[key] for r in rows], dtype=float)
    return (float(np.nanmedian(vals)), float(np.nanquantile(vals, .25)), float(np.nanquantile(vals, .75)))


def _stats(x: np.ndarray) -> dict[str, float]:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0, "magnitude": np.nan, "coherence": np.nan, "net_share": np.nan, "frac_pos": np.nan, "mean": np.nan}
    return {"n": int(x.size), "magnitude": float(np.median(np.abs(x))), "coherence": float(x.mean() / (x.std() + 1e-300)),
            "net_share": float(abs(x.sum()) / (np.abs(x).sum() + 1e-300)), "frac_pos": float((x > 0).mean()), "mean": float(x.mean())}


def _present(pool: "Pooled", epochs) -> tuple[int, ...]:
    have = set(np.unique(pool.arrays["epoch"]).tolist())
    return tuple(e for e in epochs if e in have)


def term_table(pool: Pooled, key: str, label: str, epochs: tuple[int, ...] = (1, 2, 5, 10, 20, 40, 60, 80, 100),
               sign_of: str | None = None) -> None:
    """Print magnitude and coherence of ``pool.arrays[key]`` on both clocks. If
    ``sign_of`` is given, x is multiplied by the sign of that array first (used for
    terms whose sign is only meaningful relative to another quantity)."""
    x = pool.arrays[key].astype(np.float64)
    if sign_of is not None:
        x = x * np.sign(pool.arrays[sign_of])
    print(f"\n--- {label}  [{key}]  frame {pool.frame}, seeds {pool.seeds}")
    print("fast clock: per-channel statistics over the window, medians over channels")
    for label, windows in (("before the climb (from the first recorded step to the climb's start, alignment below 0.5)", pool.preclimbs),
                           ("the climb (final ascent 0.5 -> 0.9)", pool.climbs)):
        rows = [_stats(x[s0:s1 + 1, ch]) for seed, ch, s0, s1 in windows]
        rows = [r for r in rows if r["n"] >= 10]
        if rows:
            med = lambda k: np.nanmedian([r[k] for r in rows])
            print("  %-42s channels %2d | steps median %4.0f | magnitude %.3e | coherence %+.3f | net share %.2f | positive net in %.2f of channels" % (
                label[:42], len(rows), np.median([r["n"] for r in rows]), med("magnitude"), med("coherence"), med("net_share"),
                np.mean([r["mean"] > 0 for r in rows])))
    print("slow clock (per epoch): each statistic per channel over its open unflagged steps, then the median across channels (quartiles in brackets)")
    print("  epoch | channels | magnitude | coherence [q25, q75] | net share [q25, q75] | channels with positive net | mean per step, median across channels")
    for e in _present(pool, epochs):
        m = pool.open_clean & (pool.arrays["epoch"] == e)[:, None]
        rows = []
        for seed in pool.seeds:
            rows += _per_channel(x, m & (pool.seed_of_row == seed)[:, None])
        if not rows:
            continue
        mag = _across(rows, "magnitude"); coh = _across(rows, "coherence"); ns = _across(rows, "net_share"); mn = _across(rows, "mean")
        print("  %5d | %3d | %.3e | %+.3f [%+.3f, %+.3f] | %.2f [%.2f, %.2f] | %.2f | %+.3e" % (
            e, len(rows), mag[0], coh[0], coh[1], coh[2], ns[0], ns[1], ns[2], np.mean([r["mean"] > 0 for r in rows]), mn[0]))




# ---------------------------------------------------------------------------
# per-seed report items 3 to 7 (each statistic per channel over its steps, then
# across channels; seeds are never pooled here)
# ---------------------------------------------------------------------------

_STAY_OPEN_ANGLE_DEG = 100.0    # a channel "stays open" if its angle at the end of epoch 5 is above this


def _pool_one(derived_dir: Path, frame: str) -> Pooled:
    return load_pooled([derived_dir], frame)


def _lambda_max_per_step(derived_dir: Path, frame: str) -> np.ndarray:
    """The top eigenvalue of the frame's covariance at every recorded step: the batch
    covariance's per step in the batch frame, the recording's fixed estimate otherwise."""
    prov = json.loads((derived_dir / f"checks_{frame}.json").read_text())["provenance"]
    rec_dir = Path(prov["recording"])
    if frame == "fixed":
        n = int(np.load(derived_dir / f"derived_{frame}.npz")["step"].shape[0])
        return np.full(n, float(M.CovarianceEstimate.load(rec_dir / "sigma.npz").eigenvalues[0]))
    rec = load_recording(rec_dir)
    return np.linalg.eigvalsh(rec.arrays["sigma_batch"])[:, -1]


def _stay_open_channels(pool: Pooled) -> list[int]:
    a = pool.arrays
    end5 = np.flatnonzero(a["epoch"] == 5)
    if end5.size == 0:
        return [ch for ch in range(a["open_pre"].shape[1]) if not a["dead"].any(0)[ch]]
    return [ch for ch in range(a["open_pre"].shape[1]) if not a["dead"].any(0)[ch] and a["post_theta_deg"][end5[-1], ch] > _STAY_OPEN_ANGLE_DEG]


def _fmt_q(v, fmt="%.3f") -> str:
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if v.size == 0:
        return "n/a"
    return (fmt + " [" + fmt + ", " + fmt + "]") % (np.median(v), *np.quantile(v, [.25, .75]))


def _epoch_medians(pool: Pooled, quantities: dict[str, np.ndarray], epochs) -> None:
    a = pool.arrays
    print(f"  seed {pool.seeds[0]}")
    print("  epoch | channels | " + " | ".join(quantities))
    for e in _present(pool, epochs):
        m = pool.open_clean & (a["epoch"] == e)[:, None]
        chans = [ch for ch in range(m.shape[1]) if m[:, ch].sum() >= 20]
        cols = []
        for name, x in quantities.items():
            per = [np.median(x[m[:, ch], ch]) for ch in chans]
            cols.append(_fmt_q(per, "%.0f" if "deg" in name else "%.3f"))
        print("  %5d | %2d | " % (e, len(chans)) + " | ".join(cols))


def _window_rows(x: np.ndarray, windows) -> list[dict[str, float]]:
    rows = [_stats(x[s0:s1 + 1, ch]) for _, ch, s0, s1 in windows]
    return [r for r in rows if r["n"] >= 10]


def _sign_line(label: str, rows: list[dict[str, float]]) -> None:
    if not rows:
        print("  %-14s no windows" % label); return
    print("  %-14s channels %2d | fraction of steps positive, per channel, median %.2f [%.2f, %.2f] | coherence %+.2f | net share %.2f | positive net in %.2f of channels | magnitude %.2e" % (
        label, len(rows), *np.quantile([r["frac_pos"] for r in rows], [.5, .25, .75]), np.median([r["coherence"] for r in rows]),
        np.median([r["net_share"] for r in rows]), np.mean([r["mean"] > 0 for r in rows]), np.median([r["magnitude"] for r in rows])))


def _report_signs(pool: Pooled, lam: np.ndarray) -> None:
    a = pool.arrays
    sign_product = a["pre_overlap_bisector"] * a["grad_r_vmax"]
    K = a["grad_K"]; K_loud = lam[:, None] * sign_product
    print(f"  seed {pool.seeds[0]}: {len(pool.climbs)} climbing channels")
    for name, x in (("(bisector . v_max)(v_max . r)", sign_product), ("K", K)):
        print(f"  -- {name}: fast clock (per climbing channel over its window), then per epoch over open unflagged steps")
        _sign_line("before climb", _window_rows(x, pool.preclimbs))
        _sign_line("climb", _window_rows(x, pool.climbs))
        for e in _present(pool, (1, 2, 5, 10, 20, 50, 100)):
            m = pool.open_clean & (a["epoch"] == e)[:, None]
            _sign_line("epoch %d" % e, _per_channel(x, m))
    print("  -- K against its loud approximation lambda_max (bisector . v_max)(v_max . r), per channel: sign agreement, regression slope, correlation")
    for label, windows in (("before climb", pool.preclimbs), ("climb", pool.climbs)):
        agree, slope, corr = [], [], []
        for _, ch, s0, s1 in windows:
            x = K_loud[s0:s1 + 1, ch]; y = K[s0:s1 + 1, ch]; ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() >= 10:
                agree.append((np.sign(x[ok]) == np.sign(y[ok])).mean()); slope.append(np.polyfit(x[ok], y[ok], 1)[0]); corr.append(np.corrcoef(x[ok], y[ok])[0, 1])
        if agree:
            print("  %-14s channels %2d | sign agreement %.2f | slope %s | correlation %+.2f" % (label, len(agree), np.median(agree), _fmt_q(slope, "%+.2f"), np.median(corr)))
    for e in _present(pool, (1, 2, 5, 10, 20, 50, 100)):
        m = pool.open_clean & (a["epoch"] == e)[:, None]
        agree, slope, corr, ov = [], [], [], []
        for ch in range(K.shape[1]):
            x = K_loud[m[:, ch], ch]; y = K[m[:, ch], ch]; ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() >= 20:
                agree.append((np.sign(x[ok]) == np.sign(y[ok])).mean()); slope.append(np.polyfit(x[ok], y[ok], 1)[0]); corr.append(np.corrcoef(x[ok], y[ok])[0, 1])
                ov.append(np.median(np.abs(a["pre_overlap_bisector"][m[:, ch], ch])))
        if agree:
            print("  epoch %-8d channels %2d | sign agreement %.2f | slope %s | correlation %+.2f | |bisector . v_max| median %.3f" % (e, len(agree), np.median(agree), _fmt_q(slope, "%+.2f"), np.median(corr), np.median(ov)))


def _report_K_constancy(pool: Pooled) -> None:
    a = pool.arrays; K = a["grad_K"]; stay = _stay_open_channels(pool)
    means: dict[int, dict[int, float]] = {}
    for e in _present(pool, (2, 3, 4, 5, 10, 20, 50)):
        m = pool.open_clean & (a["epoch"] == e)[:, None]
        means[e] = {ch: float(np.nanmean(K[m[:, ch], ch])) for ch in stay if m[:, ch].sum() >= 50}
    print(f"  seed {pool.seeds[0]}: {len(stay)} channels that stay open; per channel the mean of K over the epoch's open unflagged steps")
    print("  epoch pair | channels | ratio of the later mean to the earlier, median [q25, q75] | channels with the ratio within a factor 2")
    for e0, e1 in ((2, 3), (3, 4), (4, 5), (5, 10), (10, 20), (20, 50), (5, 50)):
        if e0 not in means or e1 not in means:
            continue
        chans = [ch for ch in stay if ch in means[e0] and ch in means[e1]]
        if not chans:
            continue
        r = np.array([means[e1][ch] / means[e0][ch] for ch in chans])
        print("  %2d -> %3d | %2d | %s | %.2f" % (e0, e1, len(chans), _fmt_q(r, "%.2f"), np.mean((r > 0.5) & (r < 2))))
    print("  epoch | per-step std / |mean| of K per channel, median across channels")
    for e in _present(pool, (1, 2, 5, 20, 50)):
        m = pool.open_clean & (a["epoch"] == e)[:, None]
        cv = []
        for ch in stay:
            k = K[m[:, ch], ch]; k = k[np.isfinite(k)]
            if k.size >= 50:
                cv.append(k.std() / max(abs(k.mean()), 1e-30))
        if cv:
            print("  %5d | %.1f" % (e, np.median(cv)))


def _report_angle_terms(pool: Pooled) -> None:
    a = pool.arrays; stay = _stay_open_channels(pool)
    print(f"  seed {pool.seeds[0]}: {len(stay)} channels that stay open; applied-step first-order angle shares summed per channel over the epoch's open unflagged steps (degrees), medians across channels")
    print("  the gamma-dot term is the share carried by the bisector component of the past gradients; the K term is the axis plus out-of-plane shares")
    print("  epoch | channels | gamma-dot term | K term | sigma mismatch | first-order total | realized | ratio |gamma-dot term| / K term, median [q25, q75] | channels where the K term wins")
    for e in _present(pool, (1, 2, 3, 5, 10, 20, 50, 100)):
        rows = []
        for ch in stay:
            m = pool.open_clean[:, ch] & (a["epoch"] == e)
            if m.sum() < 50:
                continue
            s = lambda k: float(np.nansum(a[f"step_{k}_d_theta_deg"][m, ch]))
            b = s("from_bisector_component"); K = s("from_axis_component") + s("from_out_of_plane")
            rows.append((b, K, s("sigma_mismatch"), s("total"), float(np.nansum(a["real_d_theta_deg"][m, ch])), -b / K if K > 0 else np.nan, K + b > 0))
        if not rows:
            continue
        r = np.array(rows, float)
        print("  %5d | %2d | %+6.1f | %+6.1f | %+5.1f | %+6.1f | %+6.1f | %s | %.2f" % (
            e, len(r), *np.median(r[:, :5], 0), _fmt_q(r[:, 5], "%.2f"), r[:, 6].mean()))
    print("  stable root of equation 43 from per-channel epoch means of K, gamma-dot, lambda_plus, lambda_minus (gradient), against the realized angle")
    print("  epoch | channels | gamma-dot mean > 0 | open root exists | stable root theta* median (deg) | realized angle median (deg)")
    K = a["grad_K"]; gd = a["grad_gamma_dot"]
    for e in _present(pool, (1, 2, 5, 10, 20, 50)):
        m = pool.open_clean & (a["epoch"] == e)[:, None]
        n = gpos = exists = 0; roots, thetas = [], []
        for ch in range(K.shape[1]):
            sel = m[:, ch]
            if sel.sum() < 50:
                continue
            n += 1
            Km = np.nanmean(K[sel, ch]); g = np.nanmean(gd[sel, ch]); lp = a["pre_lambda_plus"][sel, ch].mean(); lm = a["pre_lambda_minus"][sel, ch].mean()
            thetas.append(np.median(a["pre_theta_deg"][sel, ch])); gpos += g > 0
            disc = Km ** 2 - g ** 2 * lp * (lm - lp)
            if g > 0 and lm > lp and Km > 0 and disc >= 0:
                c = (Km - np.sqrt(disc)) / (g * (lm - lp))
                if 0 < c < 1:
                    exists += 1; roots.append(np.degrees(2 * np.arccos(c)))
        print("  %5d | %2d | %2d | %2d | %6.1f | %6.1f" % (e, n, gpos, exists, np.median(roots) if roots else np.nan, np.median(thetas)))


_LAMBDA_NEXT_CIFAR100 = 2.28    # second eigenvalue of the default cell's block-0 patch covariance, for the lambda_next edge of A.9


def _report_band(pool: Pooled, lam: np.ndarray) -> None:
    """A.9 and A.10 of the paper: the B band and the stability-condition units at epoch 100 (and 5)."""
    a = pool.arrays
    for e in _present(pool, (5, 100)):
        m = pool.open_clean & (a["epoch"] == e)[:, None]
        chans = [ch for ch in range(m.shape[1]) if m[:, ch].sum() >= 20]
        med = lambda x: np.array([np.median(x[m[:, ch], ch]) for ch in chans])
        lp, lm, c2, B, th = med(a["pre_lambda_plus"]), med(a["pre_lambda_minus"]), med(a["pre_cos_half"] ** 2), med(a["pre_B"]), med(a["pre_theta_deg"])
        lmax = med(np.broadcast_to(lam[:, None], a["pre_B"].shape))
        bound = lp / (lm - lp); edge_next = _LAMBDA_NEXT_CIFAR100 / (lm - _LAMBDA_NEXT_CIFAR100)
        print(f"  seed {pool.seeds[0]}, epoch {e}, {len(chans)} open channels; per channel the median over the epoch's open unflagged steps, then median [q25, q75] across channels")
        print("   lambda_plus %s | lambda_minus %s | cos^2(theta/2) %s | B %s | quiet term lambda_plus sin^2 %s | loud leakage lambda_minus cos^2 %s" % (
            _fmt_q(lp), _fmt_q(lm), _fmt_q(c2), _fmt_q(B), _fmt_q(lp * (1 - c2)), _fmt_q(lm * c2)))
        print("   B / lambda_plus %s | stability units cos^2 / [lambda_plus/(lambda_minus - lambda_plus)] %s | channels with cos^2 >= their own boundary %d of %d | channels clearing the lambda_next edge %d of %d" % (
            _fmt_q(B / lp), _fmt_q(c2 / bound), int((c2 >= bound).sum()), len(chans), int((c2 <= edge_next).sum()), len(chans)))
        print("   realized angle median %.1f deg | boundary angle at the median measured lambda_plus %.1f | at lambda_next %.1f | separation-axis coefficient B cos^2 / lambda_max %s" % (
            np.median(th), np.degrees(2 * np.arccos(np.sqrt(np.median(lp) / (np.median(lm) - np.median(lp))))),
            np.degrees(2 * np.arccos(np.sqrt(_LAMBDA_NEXT_CIFAR100 / (np.median(lm) - _LAMBDA_NEXT_CIFAR100)))), _fmt_q(B * c2 / lmax)))


_ALIGNMENT_BINS = ((0.0, 0.5), (0.5, 0.8), (0.8, 0.95), (0.95, 0.99), (0.99, 1.0))


def _report_diffusivity(pool: Pooled) -> None:
    """A.11 of the paper: the per-step rotation of the whitened axis binned by alignment,
    epochs 1 to 5, and the climb length."""
    a = pool.arrays
    early = np.isin(a["epoch"], [1, 2, 3, 4, 5])
    rot = np.degrees(np.arccos(np.clip(np.abs(np.einsum("tcd,tcd->tc", a["pre_axis_hat"], a["post_axis_hat"])), 0, 1)))
    along = np.abs(a["real_d_overlap_axis"]); al = a["alignment_pre"]
    m0 = pool.open_clean & early[:, None] & np.isfinite(rot)
    lengths = [s1 - s0 + 1 for _, _, s0, s1 in pool.climbs]
    print(f"  seed {pool.seeds[0]}: {len(pool.climbs)} climbing channels, climb length (final ascent 0.5 -> 0.9) median %.0f steps [q25 %.0f, q75 %.0f]" % tuple(np.quantile(lengths, [.5, .25, .75])) if lengths else f"  seed {pool.seeds[0]}: no climbing channels")
    print("  alignment bin | channels | per-step rotation of the axis (deg), per-channel median then median across channels | |step along v_max| | mean squared rotation (deg^2)")
    msq = {}
    for lo, hi in _ALIGNMENT_BINS:
        m = m0 & (al >= lo) & (al < hi)
        chans = [ch for ch in range(m.shape[1]) if m[:, ch].sum() >= 10]
        r = [np.median(rot[m[:, ch], ch]) for ch in chans]; v = [np.median(along[m[:, ch], ch]) for ch in chans]; q = [np.mean(rot[m[:, ch], ch] ** 2) for ch in chans]
        msq[(lo, hi)] = np.median(q) if q else np.nan
        print("  %.2f-%.2f | %2d | %.2f | %.4f | %.3f" % (lo, hi, len(chans), np.median(r) if r else np.nan, np.median(v) if v else np.nan, msq[(lo, hi)]))
    print("  parked (>= 0.99) against exploring (< 0.5): mean squared rotation ratio %.1f" % (msq[(0.0, 0.5)] / msq[(0.99, 1.0)]))


def _report_climb_shares(pool: Pooled) -> None:
    """A.16 of the paper: the applied step's alignment change over each channel's climb by
    the loud/quiet split of r and by the plane split, per channel then across channels."""
    a = pool.arrays
    print(f"  seed {pool.seeds[0]}: {len(pool.climbs)} climbing channels (final ascent 0.5 -> 0.9); net alignment change in log-odds per channel, median [q25, q75] across channels")
    print("  share | net | coherence median | positive net in fraction of channels | per-step magnitude median")
    for key, label in (("from_vmax_component", "loud: v_max component of r"), ("from_quiet_component", "quiet: the rest of r"),
                       ("from_axis_component", "axis component (p_- . r)"), ("from_bisector_component", "bisector component (p_+ . r)"),
                       ("from_out_of_plane", "out of plane (Q r)"), ("total", "total first order")):
        rows = [_stats(a[f"step_{key}_d_logodds_alignment"][s0:s1 + 1, ch]) for _, ch, s0, s1 in pool.climbs]
        if not rows:
            continue
        nets = np.array([r["mean"] * r["n"] for r in rows])
        print("  %-30s | %s | %+.2f | %.2f | %.4f" % (label, _fmt_q(nets, "%+.2f"), np.median([r["coherence"] for r in rows]), np.mean(nets > 0), np.median([r["magnitude"] for r in rows])))
    rows = [_stats(a["real_d_logodds_alignment"][s0:s1 + 1, ch]) for _, ch, s0, s1 in pool.climbs]
    if rows:
        nets = np.array([r["mean"] * r["n"] for r in rows])
        print("  %-30s | %s | %+.2f | %.2f | %.4f" % ("realized", _fmt_q(nets, "%+.2f"), np.median([r["coherence"] for r in rows]), np.mean(nets > 0), np.median([r["magnitude"] for r in rows])))
        dom = [np.mean(np.abs(a["step_from_vmax_component_d_logodds_alignment"][s0:s1 + 1, ch]) > np.abs(a["step_from_quiet_component_d_logodds_alignment"][s0:s1 + 1, ch])) for _, ch, s0, s1 in pool.climbs]
        print("  fraction of climb steps where the loud share exceeds the quiet share in magnitude, per channel: %s" % _fmt_q(dom, "%.2f"))


def _report_frames(derived_dir: Path) -> None:
    """Item 14: the batch frame against the fixed frame on one derived directory that holds both."""
    pb = load_pooled([derived_dir], "batch"); pf = load_pooled([derived_dir], "fixed"); ab, af = pb.arrays, pf.arrays
    m = pb.open_clean & pf.open_clean
    print(f"  seed {pb.seeds[0]}: per channel over the epoch's open steps in both frames, then across channels")
    print("  epoch | channels | |angle_batch - angle_fixed| (deg) | |alignment difference| | correlation of the per-step realized angle change | of the alignment log-odds change | net epoch angle change batch / fixed (deg) | net alignment log-odds batch / fixed")
    for e in _present(pb, (1, 2, 5, 10, 20, 50, 100)):
        me = m & (ab["epoch"] == e)[:, None]
        rows = []
        for ch in range(m.shape[1]):
            sel = me[:, ch]
            if sel.sum() < 20:
                continue
            x, y = ab["real_d_theta_deg"][sel, ch], af["real_d_theta_deg"][sel, ch]; ok = np.isfinite(x) & np.isfinite(y)
            u, v = ab["real_d_logodds_alignment"][sel, ch], af["real_d_logodds_alignment"][sel, ch]; ok2 = np.isfinite(u) & np.isfinite(v)
            rows.append((np.median(np.abs(ab["pre_theta_deg"][sel, ch] - af["pre_theta_deg"][sel, ch])), np.median(np.abs(ab["alignment_pre"][sel, ch] - af["alignment_pre"][sel, ch])),
                         np.corrcoef(x[ok], y[ok])[0, 1], np.corrcoef(u[ok2], v[ok2])[0, 1], np.nansum(x), np.nansum(y), np.nansum(u), np.nansum(v)))
        r = np.median(np.array(rows), 0)
        print("  %5d | %2d | %.2f | %.4f | %.3f | %.3f | %+.2f / %+.2f | %+.3f / %+.3f" % (e, len(rows), *r))


def _report_cross_term(derived_dir: Path, frame: str) -> None:
    """A.18 of the paper: the cross term p_+^T Sigma p_- against its approximation (37),
    and the bracket of (36) against the bracket of (38), per channel over its climb."""
    pool = load_pooled([derived_dir], frame); a = pool.arrays
    prov = json.loads((derived_dir / f"checks_{frame}.json").read_text())["provenance"]
    rec = load_recording(Path(prov["recording"]))
    if frame == "batch":
        S = rec.arrays["sigma_batch"]
    else:
        S = np.broadcast_to(rec.sigma.sigma, (a["step"].shape[0],) + rec.sigma.sigma.shape)
    w, V = np.linalg.eigh(S); lam = w[:, -1]; vmax = V[:, :, -1]
    bis, ax = a["pre_bisector_hat"], a["pre_axis_hat"]
    cross = np.einsum("tcd,tde,tce->tc", bis, S, ax)
    ob = np.einsum("tcd,td->tc", bis, vmax); oa = np.einsum("tcd,td->tc", ax, vmax)
    approx = lam[:, None] * ob * oa
    c2 = a["pre_cos_half"] ** 2; s2 = 1 - c2; lp, lm = a["pre_lambda_plus"], a["pre_lambda_minus"]
    first = ob * (2 * lam[:, None] - lp * c2 - lm * s2)
    br_exact = first - oa * cross
    br_38 = ob * (2 * lam[:, None] - lam[:, None] * oa ** 2 - lp * c2 - lm * s2)
    rows = []
    for _, ch, s0, s1 in pool.climbs:
        sl = slice(s0, s1 + 1)
        rows.append((np.median(cross[sl, ch] / approx[sl, ch]), (np.sign(cross[sl, ch]) == np.sign(approx[sl, ch])).mean(),
                     np.median(np.abs(cross[sl, ch] - approx[sl, ch]) / np.abs(first[sl, ch])), np.median(np.abs(oa[sl, ch] * cross[sl, ch]) / np.abs(first[sl, ch])),
                     np.median(br_exact[sl, ch] / br_38[sl, ch]), (np.sign(br_exact[sl, ch]) == np.sign(ob[sl, ch])).mean(), (np.sign(br_38[sl, ch]) == np.sign(ob[sl, ch])).mean()))
    if not rows:
        print(f"  seed {pool.seeds[0]}: no climbing channels"); return
    r = np.array(rows)
    print(f"  seed {pool.seeds[0]}: {len(r)} climbing channels; per channel over its climb, then median [q25, q75] across channels")
    for label, col, fmt in (("cross term / its approximation (37)", 0, "%.2f"), ("sign agreement with (37), fraction of steps", 1, "%.2f"),
                            ("|cross term - (37)| relative to the first term of (36)'s bracket", 2, "%.2f"), ("the whole last term relative to the first term", 3, "%.2f"),
                            ("exact bracket of (36) / bracket of (38)", 4, "%.2f"), ("steps where the exact bracket has the sign of p_+ . v_max", 5, "%.2f"),
                            ("steps where (38)'s bracket has that sign", 6, "%.2f")):
        print("   %-66s %s" % (label, _fmt_q(r[:, col], fmt)))


def run_report(derived_dirs: list[Path], frame: str, items: list[int]) -> None:
    pool = load_pooled(derived_dirs, frame)
    a = pool.arrays
    print(f"pooled {len(pool.seeds)} seeds, {a['step'].shape[0]} steps x {a['open_pre'].shape[1]} channels, {len(pool.climbs)} climbing channels")
    for seed in pool.seeds:
        rows = pool.seed_of_row == seed
        dead_ch = np.flatnonzero(a["dead"][rows].mean(0) > 0.5)
        print(f"  seed {seed}: dead channels (no gradient, moved by decay only) {dead_ch.tolist()}; excluded from every term statistic")
    if 1 in items:
        print("\n=== item 1: sigma_1 = sigma_2 ===")
        print("  branch-sigma ratio (larger over smaller, formed at each step): per channel the median over its open steps in the epoch, then across channels")
        for e in _present(pool, (1, 2, 5, 10, 20, 50, 100)):
            m = pool.open_clean & (a["epoch"] == e)[:, None]
            per = []
            for ch in range(a["open_pre"].shape[1]):
                r = a["pre_sigma_ratio"][m[:, ch], ch]
                if r.size >= 20:
                    per.append((np.median(r), np.quantile(r, .9)))
            per = np.array(per)
            print("  epoch %3d | %2d channels | median across channels of the per-channel median %.3f, quartiles %.3f %.3f, largest channel %.3f | per-channel 90th percentile, median across channels %.3f" % (
                e, len(per), np.median(per[:, 0]), *np.quantile(per[:, 0], [.25, .75]), per[:, 0].max(), np.median(per[:, 1])))
        print("  A.1 numbers per seed: the ratio at step 0 over open live channels; per channel the 99th percentile over epoch 1's open steps; per channel the median over the open steps of epochs 5 to 100")
        for d in derived_dirs:
            pool_s = _pool_one(d, frame); a_s = pool_s.arrays; r = a_s["pre_sigma_ratio"]
            live = ~a_s["dead"].any(0); r0 = r[0][live & a_s["open_pre"][0]]
            e1 = (a_s["epoch"] == 1)[:, None] & pool_s.open_clean
            p99 = [np.quantile(r[e1[:, ch], ch], .99) for ch in range(r.shape[1]) if e1[:, ch].sum() >= 20]
            late = np.isin(a_s["epoch"], [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])[:, None] & pool_s.open_clean
            med = [np.median(r[late[:, ch], ch]) for ch in range(r.shape[1]) if late[:, ch].sum() >= 100]
            print("  seed %d | step 0 median %.2f (%d channels) | epoch 1 per-channel 99th percentile: median %.1f, largest channel %.1f | epochs 5 to 100 per-channel median: %s" % (
                pool_s.seeds[0], np.median(r0), r0.size, np.median(p99), max(p99), _fmt_q(med, "%.2f")))
        term_table(pool, "step_sigma_mismatch_d_theta_deg", "sigma-mismatch share of the applied step's angle change (deg per step)")
        term_table(pool, "step_total_d_theta_deg", "for scale: the total first-order angle change of the applied step (deg per step)")
        term_table(pool, "step_sigma_mismatch_d_logodds_alignment", "sigma-mismatch share of the applied step's alignment change (log-odds per step)")
        term_table(pool, "step_total_d_logodds_alignment", "for scale: the total first-order alignment change (log-odds per step)")
    if 2 in items:
        print("\n=== item 2: the regime split, the three shares of the applied step ===")
        for q, unit in (("d_logodds_alignment", "log-odds per step"), ("d_theta_deg", "deg per step")):
            for share in ("from_bisector_component", "from_axis_component", "from_out_of_plane", "prewindow"):
                term_table(pool, f"step_{share}_{q}", f"{share} share, {unit}")
            term_table(pool, f"real_{q}", f"realized change, {unit}")
    if 8 in items:
        print("\n=== item 8: the drive gamma gamma-dot / (2 sigma^2) and its momentum-accumulated form ===")
        term_table(pool, "grad_drive", "drive from the gradient, per step")
        print("  momentum-accumulated: the exponent s over a window is sum of gamma * (realized d gamma) / (2 sigma^2); its excursions per channel over epochs 1-5")
        for seed in pool.seeds:
            rows = []
            sel_rows = np.flatnonzero((pool.seed_of_row == seed) & np.isin(a["epoch"], [1, 2, 3, 4, 5]))
            g = a["pre_gamma"][sel_rows]; dg = a["real_d_gamma"][sel_rows]
            s2 = 1.0 / (0.5 * (1 / a["pre_sigma1"][sel_rows] ** 2 + 1 / a["pre_sigma2"][sel_rows] ** 2))
            ds = g * dg / (2 * s2)
            s = np.cumsum(ds, axis=0)
            drawdown = (np.maximum.accumulate(s, axis=0) - s).max(0)
            print("  seed %d | net s over epochs 1-5: median %+.4f, 10th %+.4f, 90th %+.4f | largest negative excursion median %.4f, 90th %.4f | times the gap 19.5: median amplification exp %.2f, 90th %.2f" % (
                seed, np.median(s[-1]), np.quantile(s[-1], .1), np.quantile(s[-1], .9), np.median(drawdown), np.quantile(drawdown, .9),
                np.exp(np.median(drawdown) * 19.5), np.exp(np.quantile(drawdown, .9) * 19.5)))
    if 11 in items:
        print("\n=== item 11: initialization tilt (step 0 of each seed) ===")
        for seed in pool.seeds:
            r0 = np.flatnonzero(pool.seed_of_row == seed)[0]
            al = a["alignment_pre"][r0]; th = a["pre_theta_deg"][r0]; b = np.abs(a["pre_overlap_bisector"][r0])
            ale = np.abs(a["pre_overlap_axis_euclid"][r0]); the = np.degrees(np.arccos(np.clip(a["pre_cos_theta_euclid"][r0], -1, 1)))
            axis_holds = al > b
            print("  seed %d | whitened: alignment median %.2f (fraction > 0.5: %.2f), angle median %.1f, angle when the axis holds v_max %.1f (n=%d) / when the bisector does %.1f (n=%d), variance share of v_max in the axis median %.2f | Euclidean: alignment median %.2f, angle median %.1f" % (
                seed, np.median(al), (al > 0.5).mean(), np.median(th), np.median(th[axis_holds]), axis_holds.sum(), np.median(th[~axis_holds]), (~axis_holds).sum(),
                np.median(a["pre_variance_share_axis"][r0]), np.median(ale), np.median(the)))
    if 12 in items:
        print("\n=== item 12: the shares of the climb by the loud/quiet and plane splits of r (A.16), per seed ===")
        for d in derived_dirs:
            _report_climb_shares(_pool_one(d, frame))
    if 13 in items:
        print("\n=== item 13: the second-order residual of the angle (A.17), per seed; channels that stay open ===")
        print("  per channel over the epoch's open unflagged steps, then medians across channels; residual = realized change minus the first-order change of the applied step")
        for d in derived_dirs:
            pool_s = _pool_one(d, frame); a_s = pool_s.arrays; stay = _stay_open_channels(pool_s)
            fo = a_s["step_total_d_theta_deg"]; real = a_s["real_d_theta_deg"]; res = real - fo
            print(f"  seed {pool_s.seeds[0]}: {len(stay)} channels that stay open")
            print("  epoch | lr | per-step |first order| (deg) | per-step |residual| | residual over first order | corr(first order, realized) | epoch net: first order | residual | realized")
            for e in _present(pool_s, (1, 2, 5, 10, 20, 50)):
                rows = []
                for ch in stay:
                    m = pool_s.open_clean[:, ch] & (a_s["epoch"] == e) & np.isfinite(fo[:, ch]) & np.isfinite(real[:, ch])
                    if m.sum() < 50:
                        continue
                    f, r, q = fo[m, ch], real[m, ch], res[m, ch]
                    rows.append((np.median(np.abs(f)), np.median(np.abs(q)), np.median(np.abs(q) / np.abs(f)), np.corrcoef(f, r)[0, 1], f.sum(), q.sum(), r.sum()))
                if rows:
                    r = np.median(np.array(rows), 0)
                    print("  %5d | %.4f | %.3f | %.3f | %.2f | %.3f | %+6.1f | %+6.1f | %+6.1f" % (e, a_s["lr_w"][a_s["epoch"] == e][0], *r))
    if 14 in items:
        print("\n=== item 14: the batch frame against the fixed frame (needs derived_fixed.npz beside derived_batch.npz) ===")
        for d in derived_dirs:
            if (d / "derived_fixed.npz").exists() and (d / "derived_batch.npz").exists():
                _report_frames(d)
            else:
                print(f"  {d}: only one frame derived, skipped")
    if 15 in items:
        print("\n=== item 15: the per-channel stability relation at epoch 100, per seed (each channel's median over epoch 100's open steps) ===")
        for d in derived_dirs:
            pool_s = _pool_one(d, frame); a_s = pool_s.arrays
            m = pool_s.open_clean & (a_s["epoch"] == 100)[:, None]
            chans = [ch for ch in range(m.shape[1]) if m[:, ch].sum() >= 20]
            med = lambda x: np.array([np.median(x[m[:, ch], ch]) for ch in chans])
            th, lp, lm = med(a_s["pre_theta_deg"]), med(a_s["pre_lambda_plus"]), med(a_s["pre_lambda_minus"])
            ok = lm > 2 * lp
            thb = np.degrees(2 * np.arccos(np.sqrt(lp[ok] / (lm[ok] - lp[ok]))))
            slope = np.polyfit(thb, th[ok], 1)[0]
            print("  seed %d | open channels %d (with a boundary angle %d) | corr(realized angle, own boundary angle) %+.2f, slope %.2f, median |error| %.1f deg | corr(realized angle, lambda_plus) %+.2f" % (
                pool_s.seeds[0], len(chans), int(ok.sum()), np.corrcoef(th[ok], thb)[0, 1], slope, np.median(np.abs(th[ok] - thb)), np.corrcoef(th, lp)[0, 1]))
    if 3 in items:
        print("\n=== item 3: the sign assumption (bisector . v_max)(v_max . r) > 0 and K, per seed ===")
        for d in derived_dirs:
            _report_signs(_pool_one(d, frame), _lambda_max_per_step(d, frame))
    if 4 in items:
        print("\n=== item 4: lambda_minus is the top eigenvalue, per seed ===")
        print("  per channel the median over the epoch's open unflagged steps of lambda_minus / lambda_max (batch frame: each step's own top eigenvalue), then median [q25, q75] across channels")
        for d in derived_dirs:
            pool_s = _pool_one(d, frame); lam = _lambda_max_per_step(d, frame)
            _epoch_medians(pool_s, {"lambda_minus / lambda_max": pool_s.arrays["pre_lambda_minus"] / lam[:, None]}, (5, 20, 50, 100))
    if 5 in items:
        print("\n=== item 5: K after the angle has settled, per seed ===")
        for d in derived_dirs:
            _report_K_constancy(_pool_one(d, frame))
    if 6 in items:
        print("\n=== item 6: the two terms of the angle equation in the applied step, per seed ===")
        for d in derived_dirs:
            _report_angle_terms(_pool_one(d, frame))
    if 9 in items:
        print("\n=== item 9: the B band and the stability-condition units (A.9, A.10), per seed ===")
        for d in derived_dirs:
            _report_band(_pool_one(d, frame), _lambda_max_per_step(d, frame))
    if 10 in items:
        print("\n=== item 10: the diffusivity of the separation axis (A.11), per seed ===")
        for d in derived_dirs:
            _report_diffusivity(_pool_one(d, frame))
    if 16 in items:
        print("\n=== item 16: the cross term p_+^T Sigma p_- against its approximation (37), over the climb (A.18), per seed ===")
        for d in derived_dirs:
            _report_cross_term(d, frame)
    if 7 in items:
        print("\n=== item 7: the cooling factor on the separation axis, per seed ===")
        print("  per channel the median over the epoch's open unflagged steps, then median [q25, q75] across channels")
        for d in derived_dirs:
            pool_s = _pool_one(d, frame); a_s = pool_s.arrays
            c2 = a_s["pre_cos_half"] ** 2; B = a_s["pre_B"]; lm = a_s["pre_lambda_minus"]
            _epoch_medians(pool_s, {"cos^2(theta/2)": c2, "same-function factor B/(2 lambda_minus)": B / (2 * lm),
                                    "equal-scale factor 2 B cos^2 / lambda_minus": 2 * B * c2 / lm, "B / lambda_plus": B / a_s["pre_lambda_plus"], "angle (deg)": a_s["pre_theta_deg"]}, (5, 20, 50, 100))

# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("recording", help="derive everything from a recording directory")
    r.add_argument("--recording", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--frames", nargs="+", default=["fixed", "batch"], choices=["fixed", "batch"])
    c = sub.add_parser("checkpoints", help="geometry on W&B checkpoints")
    c.add_argument("--group", required=True)
    c.add_argument("--variant", required=True)
    c.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    c.add_argument("--epochs", type=int, nargs="+", default=[0, 5, 25, 50, 75, 100])
    c.add_argument("--dataset-config", type=Path, required=True)
    c.add_argument("--block-prefix", default="stages.0.0")
    c.add_argument("--out", type=Path, required=True)
    rp = sub.add_parser("report", help="magnitude and coherence of every term, fast and slow clock, pooled over seeds")
    rp.add_argument("--derived", type=Path, nargs="+", required=True)
    rp.add_argument("--frame", default="batch", choices=["fixed", "batch"])
    rp.add_argument("--items", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16])
    q = sub.add_parser("reconcile", help="compare a recording's epoch-end geometry with the old pair_channel_open probe")
    q.add_argument("--recording", type=Path, required=True)
    q.add_argument("--probe-npz", type=Path, required=True)
    q.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.cmd == "recording":
        run_recording(args.recording, args.out, args.frames)
    elif args.cmd == "report":
        run_report(args.derived, args.frame, args.items)
    elif args.cmd == "checkpoints":
        run_checkpoints(args.group, args.variant, args.seeds, args.epochs, args.dataset_config, args.out, args.block_prefix)
    else:
        run_reconcile(args.recording, args.probe_npz, args.out)


if __name__ == "__main__":
    main()
