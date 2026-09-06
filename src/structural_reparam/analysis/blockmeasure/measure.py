"""The mathematics of the two-branch block, written once.

This module is the single implementation of every quantity the paper's theory
(the vault note "New farmework paper", section 3) measures on a two-branch block

    output = gamma * ( bn_1(w_1 * x) + bn_2(w_2 * x) ) + beta_1 + beta_2 ,

with a shared per-channel scale gamma and a per-branch normalizer that divides by
the standard deviation of its own branch's conv output. Everything here is a pure
function of primitives: the two kernels w_1 and w_2, gamma, the block-seen patch
covariance Sigma, and, for the forcing sections, a kernel-space vector standing in
the gradient's place (the gradient itself or the optimizer's momentum buffer).

The module is organised in numbered sections, each reviewable on its own:

  1. Conventions: every threshold and convention as a named constant with its reason.
  2. Covariance: the one estimator of Sigma and the container that carries it,
     together with its settings and a hash, into every output.
  3. Geometry: what needs only kernels, gamma and Sigma (paper equations 3, 4, 6,
     15, 16 and the definition of B in section 3.5). Works on checkpoints and on
     recordings alike.
  4. Forcing: the recovery of the whitened output gradient r from the two branch
     vectors, and every named term and term group of equations 17, 33 and 39.
  5. Dynamics: realized and first-order changes between consecutive states.
  6. Checks: the identities, their tolerances, and the flag arrays.
  7. Aggregation: masks, climb windows, and summaries that name their population.

Frames. The theory lives in the whitened frame: for a kernel w (a vector of length
D = C_in * 3 * 3, in the order ``weight.flatten(1)`` produces, the same order as
``torch.nn.functional.unfold``), the whitened branch direction is

    p = Sigma^{1/2} w / sigma ,   sigma = sqrt( w^T Sigma w ) ,

a unit vector, because sigma is exactly the norm of Sigma^{1/2} w. Every quantity
is whitened unless its name carries the suffix ``_euclid``, which marks the plain
inner product on the kernels themselves.

Symbols in docstrings are the paper's: w, sigma, gamma, beta, Sigma, lambda_plus,
lambda_minus, B, K, theta, p_plus, p_minus, v_max, r, eta. Everything else is
written out in words. Array layouts are stated as shapes with the letters
T (recorded steps), C (output channels), D (patch dimension); a leading ``...``
means any number of batch dimensions.

Precision. All computation here is numpy float64. The recorder stores the model's
float32 tensors unchanged; conversion to float64 happens on the way in.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

# ---------------------------------------------------------------------------
# Section 1. Conventions
# ---------------------------------------------------------------------------
# Each constant carries the reason for its value. Nothing below this section may
# use a literal threshold; it imports one of these names.

OPEN_ANGLE_DEG: float = 10.0
"""A channel is "open" when its whitened branch angle theta is at least this many
degrees. Paper-wide convention (guidelines note, section 2)."""

PARKED_OVERLAP: float = 0.95
"""A channel is "parked" when the absolute overlap of its separation axis with v_max
is at least this. Paper-wide convention (guidelines note, section 2)."""

CLIMB_LOW: float = 0.5
CLIMB_HIGH: float = 0.9
"""A channel's climb is the window from its first step with absolute overlap above
CLIMB_LOW to its first later step above CLIMB_HIGH. This replaces three separate
copies of the same rule in the old analysis scripts."""

SIGMA_BATCHES: int = 50
SIGMA_SEED: int = 0
"""The covariance convention of the paper (guidelines note, section 4): 50 batches
of the augmented training stream, patch mean subtracted. The seed fixes which
batches, so two estimates from the same data are bit-identical."""

SIGMA_CROSSCHECK_SEED_OFFSET: int = 1
"""A second estimate with seed + this offset is stored beside every recording so the
estimator's own error is on record."""

EIGENVALUE_FLOOR_RELATIVE: float = 1e-12
"""Sigma^{-1/2} floors eigenvalues below this fraction of the largest one. The
block-0 spectrum spans about five decades (condition number about 1e5), so the
floor is never reached there; it exists so a rank-deficient Sigma cannot produce
infinities silently. The number of floored eigenvalues is stored with the estimate."""

WELL_CONDITIONED_MIN: float = 0.1
"""The recovery of r divides by cos^2(theta/2) for the axis component and by
sin^2(theta/2) for the kernel route to the bisector component. A channel-step is
"well conditioned" for a given division when that factor is at least this. Checks
on identities are reported separately for well-conditioned and other channel-steps;
nothing is dropped."""

DEGENERATE_NORM: float = 1e-12
"""A bisector or axis whose unnormalised length is below this is undefined (the
branches are exactly parallel or antiparallel). Its unit vector is set to NaN and
the ``bisector_defined`` / ``axis_defined`` flags record it."""

TOL_ALGEBRA: float = 1e-6
"""Tolerance for identities that are pure float64 algebra on the same inputs
(orthogonality, closure of term groups, the sigma-mismatch split). Their median error
is 1e-15; the tail reaches 1e-8 because the unit-axis, bisector and angle maps divide
by sin(theta/2), cos(theta/2) and sin(theta), which amplify rounding on the most
closed or most open channels. 1e-6 is still eight orders of magnitude below any
physical effect and flags only genuine algebra mistakes."""

TOL_FLOAT32: float = 1e-5
"""Tolerance for identities whose inputs were stored in float32 (the applied step
w_post = w_pre - eta * buffer, the momentum recurrence, the recovery of r from
float32 gradients)."""

TOL_GRADIENT_IDENTITY: float = 1e-4
"""Tolerance for the identities that relate the recorded gradients to r (the recovery
of r reproducing both kernel gradients, and the gamma gradient agreeing with the
kernel route). They hold exactly only in the batch frame, and there the normalizer
divides by sqrt(var + eps) with eps = 1e-5 while the theory's sigma has no eps, so
the identities carry an error of order eps over the batch variance; the recovery
also divides by sin^2(theta/2). A tenth of a percent is comfortably above both for
float32 models and far below the percent-level error of the fixed frame, which is
reported as a diagnostic, never flagged."""

TOL_GRADIENT_IDENTITY_TF32: float = 1e-2
"""The same identities when the convolutions ran in TF32 (torch's default for cuDNN
on Ampere and later GPUs, 10 mantissa bits): the recorded gradients are then
accurate to about 1e-4 relative after averaging over the batch, and the recovery's
divisions by cos^2(theta/2) or sin^2(theta/2), allowed down to WELL_CONDITIONED_MIN,
amplify that by up to ten. Measured on the first real recording (2026-09-02): median
7e-5, 99th percentile 5e-4 for the recovery and 7e-3 for the gamma identity. The
manifest records whether TF32 was allowed; derive picks this tolerance when it was
or when the manifest predates the flag."""

TOL_FILTERED_CLOSURE: float = 1e-2
"""Tolerance for the closure of the momentum-filtered shares' pushes against the
push of the recorded buffer. The shares are float64 running sums; the recorded
buffer is float32, so the two agree to about 1e-7 as vectors (check
``buffer_shares``), and the angle change is a difference of large opposing pushes
about a hundred times smaller than the pushes themselves, which turns 1e-7 into
about 1e-5 typically and 1e-3 in the tail. One percent flags only a real closure
failure."""

DEAD_GRADIENT_RELATIVE: float = 1e-6
"""A channel-step is "dead" when the norm of its two kernel gradients is below this
fraction of the median over channels at the same step: the channel's output is
never positive on the batch, the ReLU passes no gradient, and the kernels move only
under weight decay. Such channels keep a frozen whitened angle and would be counted
as open by the angle threshold; every term statistic excludes them and their count
is reported (two of 32 channels in seed 43 of the first recording, from epoch 3)."""

SECOND_ORDER_BOUND: float = 10.0
"""Check 7 compares the realized change of a whitened direction with its first-order
prediction at the recorded step. For a unit vector p = y/|y| the second-order
remainder of a displacement is at most about 1.5 |dy|^2 / |y|^2, so the residual is
declared consistent with first order when it is below this constant times the
squared relative whitened step (|Sigma^{1/2} dw| / sigma)^2. The bound is loose by
design; a genuine mismatch between the recorded step and the stored gradients or
buffers is orders of magnitude larger."""

AXIS_SIGN_RULE: str = "axis = p_1 - p_2 (branch 0 minus branch 1); never flipped inside geometry"
"""The separation axis has an arbitrary sign under branch relabelling. Geometry
never chooses one. Dynamics matches consecutive states by continuity and records
the flips it applied; aggregation reports absolute overlaps."""

CONVENTIONS: dict[str, Any] = {
    "open_angle_deg": OPEN_ANGLE_DEG,
    "parked_overlap": PARKED_OVERLAP,
    "climb_low": CLIMB_LOW,
    "climb_high": CLIMB_HIGH,
    "sigma_batches": SIGMA_BATCHES,
    "sigma_seed": SIGMA_SEED,
    "eigenvalue_floor_relative": EIGENVALUE_FLOOR_RELATIVE,
    "well_conditioned_min": WELL_CONDITIONED_MIN,
    "degenerate_norm": DEGENERATE_NORM,
    "tol_algebra": TOL_ALGEBRA,
    "tol_float32": TOL_FLOAT32,
    "tol_gradient_identity": TOL_GRADIENT_IDENTITY,
    "tol_gradient_identity_tf32": TOL_GRADIENT_IDENTITY_TF32,
    "tol_filtered_closure": TOL_FILTERED_CLOSURE,
    "dead_gradient_relative": DEAD_GRADIENT_RELATIVE,
    "second_order_bound": SECOND_ORDER_BOUND,
    "axis_sign_rule": AXIS_SIGN_RULE,
    "frame": "whitened unless the name ends in _euclid",
    "kernel_flattening": "weight.flatten(1): index = in_channel * 9 + row * 3 + col, the unfold order",
}
"""Everything above, as one dictionary, written into every output file."""


# ---------------------------------------------------------------------------
# Section 2. Covariance
# ---------------------------------------------------------------------------


@dataclass
class CovarianceEstimate:
    """The block-seen patch covariance Sigma and everything derived from it.

    Fields (all float64):
      sigma        [D, D]  the covariance itself, symmetric.
      eigenvalues  [D]     descending.
      eigenvectors [D, D]  columns, matching ``eigenvalues``; each column's sign is
                           fixed so that its largest-magnitude entry is positive.
      v_max        [D]     the first column: the top eigendirection.
      sqrt         [D, D]  Sigma^{1/2}.
      inv_sqrt     [D, D]  Sigma^{-1/2} with the eigenvalue floor applied.
      n_floored    int     how many eigenvalues the floor touched (0 for block 0).
      settings     dict    how Sigma was estimated (dataset target and arguments,
                           number of batches, seed, patch geometry, torch version).
      sha256       str     hash of the float64 bytes of ``sigma``; the identity of
                           this estimate wherever it is quoted.

    The sign rule on the eigenvectors matters for the signed overlaps of section 3:
    a different sign of v_max flips the sign of every overlap, so the rule is fixed
    here and nowhere else.
    """

    sigma: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    v_max: np.ndarray
    sqrt: np.ndarray
    inv_sqrt: np.ndarray
    n_floored: int
    settings: dict[str, Any] = field(default_factory=dict)
    sha256: str = ""

    @classmethod
    def from_matrix(cls, sigma: np.ndarray, settings: Mapping[str, Any] | None = None) -> "CovarianceEstimate":
        """Build the container from a covariance matrix.

        ``sigma`` is symmetrised (average with its transpose) before the
        eigendecomposition so that float noise in an estimate cannot make it
        asymmetric. The returned ``sigma`` is the symmetrised matrix, and the hash
        is of that matrix.
        """
        symmetric = np.asarray(sigma, dtype=np.float64)
        if symmetric.ndim != 2 or symmetric.shape[0] != symmetric.shape[1]:
            raise ValueError(f"sigma must be square, got shape {symmetric.shape}")
        symmetric = 0.5 * (symmetric + symmetric.T)
        ev, vecs = np.linalg.eigh(symmetric)          # ascending
        order = np.argsort(ev)[::-1]
        ev, vecs = ev[order], vecs[:, order]
        # sign rule: largest-magnitude entry of each column positive
        idx = np.argmax(np.abs(vecs), axis=0)
        signs = np.sign(vecs[idx, np.arange(vecs.shape[1])])
        signs[signs == 0] = 1.0
        vecs = vecs * signs[None, :]
        floor = EIGENVALUE_FLOOR_RELATIVE * ev[0]
        floored = ev < floor
        ev_for_inv = np.where(floored, floor, ev)
        ev_for_sqrt = np.clip(ev, 0.0, None)
        sqrt = (vecs * np.sqrt(ev_for_sqrt)[None, :]) @ vecs.T
        inv_sqrt = (vecs / np.sqrt(ev_for_inv)[None, :]) @ vecs.T
        return cls(
            sigma=symmetric,
            eigenvalues=ev,
            eigenvectors=vecs,
            v_max=vecs[:, 0].copy(),
            sqrt=sqrt,
            inv_sqrt=inv_sqrt,
            n_floored=int(floored.sum()),
            settings=dict(settings or {}),
            sha256=hashlib.sha256(np.ascontiguousarray(symmetric).tobytes()).hexdigest(),
        )

    @property
    def dim(self) -> int:
        return int(self.sigma.shape[0])

    @property
    def gap_ratio(self) -> float:
        """lambda_max / lambda_next, the paper's scale-free spectrum summary."""
        return float(self.eigenvalues[0] / self.eigenvalues[1])

    def save(self, path: str | Path) -> None:
        """Write ``<path>.npz`` with the matrices and ``<path>.json`` with settings and hash."""
        path = Path(path)
        np.savez(path.with_suffix(".npz"), sigma=self.sigma, eigenvalues=self.eigenvalues,
                 eigenvectors=self.eigenvectors, v_max=self.v_max, sqrt=self.sqrt,
                 inv_sqrt=self.inv_sqrt, n_floored=self.n_floored)
        path.with_suffix(".json").write_text(json.dumps(
            {"settings": self.settings, "sha256": self.sha256, "dim": self.dim,
             "eigenvalues": self.eigenvalues.tolist(), "gap_ratio": self.gap_ratio,
             "n_floored": self.n_floored}, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "CovarianceEstimate":
        """Read what ``save`` wrote. The container is rebuilt from ``sigma`` so the
        derived matrices always match this code's rules, and the stored hash is
        checked against the recomputed one."""
        path = Path(path)
        d = np.load(path.with_suffix(".npz"))
        meta = json.loads(path.with_suffix(".json").read_text())
        est = cls.from_matrix(d["sigma"], meta.get("settings", {}))
        if meta.get("sha256") and meta["sha256"] != est.sha256:
            raise ValueError(f"covariance hash mismatch for {path}: stored {meta['sha256'][:12]}, recomputed {est.sha256[:12]}")
        return est


def patch_covariance_of_batch(x, kernel_size: int = 3, padding: int = 1, stride: int = 1):
    """Covariance of the conv's input patches for one batch, as BatchNorm sees them.

    Input: ``x`` a torch tensor [B, C_in, H, W] on any device, the input of the conv
    (for block 0, the normalised image batch). Output: a torch float64 tensor
    [D, D] with D = C_in * kernel_size^2, on the same device.

    Patches are extracted with ``unfold`` using the conv's own kernel, padding and
    stride, so they are exactly the receptive fields the convolution multiplies.
    The patch mean over the batch is subtracted before the outer product because
    the normalizer divides by the *centred* standard deviation of the conv output:
    var(w * x) over the batch equals w^T Sigma w only for the centred Sigma. The
    covariance is the biased one (division by the number of patches), matching
    ``x.var(unbiased=False)`` in the normalizer.
    """
    import torch
    import torch.nn.functional as F

    patches = F.unfold(x, kernel_size, padding=padding, stride=stride)      # [B, D, L]
    patches = patches.transpose(1, 2).reshape(-1, patches.shape[1]).double()  # [B*L, D]
    patches = patches - patches.mean(0, keepdim=True)
    return patches.T @ patches / patches.shape[0]


def estimate_patch_covariance(dataset_config: Mapping[str, Any], *, n_batches: int = SIGMA_BATCHES,
                              seed: int = SIGMA_SEED, device=None, block_input=None,
                              kernel_size: int = 3, padding: int = 1, stride: int = 1) -> CovarianceEstimate:
    """The one estimator of the block-seen covariance, the paper's convention.

    Rebuilds the training loader from the run's dataset configuration (``target``
    and ``args``, as the trainer does), with ``num_workers = 0`` so iteration is
    deterministic, inside ``torch.random.fork_rng`` seeded with ``seed`` so that
    neither the loader's shuffling nor the augmentation disturbs the run's own
    random stream. Takes ``n_batches`` batches of the augmented, shuffled stream,
    accumulates ``patch_covariance_of_batch`` over them, and averages.

    ``block_input`` is an optional callable mapping an image batch (on ``device``)
    to the input of the block being measured; it is ``None`` for block 0, whose
    input is the image batch itself. Deeper blocks pass a function that runs the
    network up to the block, under ``no_grad`` and in evaluation mode.

    Returns a ``CovarianceEstimate`` whose ``settings`` record everything needed to
    reproduce the estimate bit for bit.
    """
    import inspect
    import torch
    from structural_reparam.deploy.train import import_target  # local import, avoids cycles

    if dataset_config is None:
        raise RuntimeError("estimate_patch_covariance needs the run's dataset config")
    target = import_target(dataset_config["target"])
    args = dict(dataset_config.get("args", {}))
    args["num_workers"] = 0
    args["persistent_workers"] = False
    accepted = set(inspect.signature(target).parameters)
    args = {k: v for k, v in args.items() if k in accepted}
    dev = torch.device(device) if device is not None else torch.device("cpu")
    devices = [dev] if dev.type == "cuda" else []
    cov = None
    n = 0
    n_patches = 0
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        loaders = target(**args)
        train_loader = loaders[0] if isinstance(loaders, (tuple, list)) else loaders
        with torch.no_grad():
            for k, batch in enumerate(train_loader):
                if k >= n_batches:
                    break
                x = batch[0].to(dev)
                if block_input is not None:
                    x = block_input(x)
                batch_cov = patch_covariance_of_batch(x, kernel_size, padding, stride)
                cov = batch_cov if cov is None else cov + batch_cov
                n += 1
                n_patches += x.shape[0] * ((x.shape[2] + 2 * padding - kernel_size) // stride + 1) ** 2
    if cov is None:
        raise RuntimeError("the loader yielded no batches")
    settings = {
        "dataset_target": dataset_config["target"],
        "dataset_args": {k: v for k, v in dict(dataset_config.get("args", {})).items()
                         if isinstance(v, (int, float, str, bool, type(None)))},
        "n_batches": int(n),
        "n_patches": int(n_patches),
        "seed": int(seed),
        "kernel_size": kernel_size, "padding": padding, "stride": stride,
        "patch_mean_subtracted": True, "biased": True,
        "torch_version": torch.__version__,
    }
    return CovarianceEstimate.from_matrix((cov / n).cpu().numpy(), settings)


# ---------------------------------------------------------------------------
# Section 3. Geometry
# ---------------------------------------------------------------------------


@dataclass
class BlockState:
    """The primitives of one two-branch block at one moment.

    w1, w2   [..., C, D] float64  the two kernels, flattened as ``weight.flatten(1)``.
    gamma    [..., C]              the shared scale.
    beta1, beta2 [..., C] or None  the per-branch biases (not used by geometry).

    ``...`` may be empty (a checkpoint) or a step axis (a recording).
    """

    w1: np.ndarray
    w2: np.ndarray
    gamma: np.ndarray
    beta1: np.ndarray | None = None
    beta2: np.ndarray | None = None

    @classmethod
    def from_state_dict(cls, state_dict: Mapping[str, Any], block_prefix: str = "stages.0.0") -> "BlockState":
        """Read one block from a checkpoint's ``state_dict``.

        Keys are ``<prefix>.convs.0.weight``, ``<prefix>.convs.1.weight``,
        ``<prefix>.gamma``, ``<prefix>.betas.0`` and ``<prefix>.betas.1`` (the
        ``SharedScaleRepVGGBlock`` layout). Tensors are converted to float64 numpy.
        This replaces the dozen private ``branches()`` readers of the old scripts.
        """
        def get(key: str, required: bool = True):
            k = f"{block_prefix}.{key}"
            if k not in state_dict:
                if required:
                    raise KeyError(f"{k} not in state dict (keys like {[x for x in state_dict if x.startswith(block_prefix)][:6]})")
                return None
            t = state_dict[k]
            return np.asarray(t.detach().cpu().numpy() if hasattr(t, "detach") else t, dtype=np.float64)

        w1 = get("convs.0.weight")
        w2 = get("convs.1.weight")
        w1 = w1.reshape(w1.shape[0], -1)   # weight.flatten(1): the unfold order
        w2 = w2.reshape(w2.shape[0], -1)
        return cls(w1=w1, w2=w2, gamma=get("gamma"), beta1=get("betas.0", False), beta2=get("betas.1", False))


@dataclass
class Geometry:
    """Every quantity that needs only the kernels, gamma and Sigma.

    Shapes: ``[..., C]`` unless noted; vectors are ``[..., C, D]``. Whitened frame
    unless the name ends in ``_euclid``.

    sigma1, sigma2       sqrt(w_i^T Sigma w_i), the standard deviation each branch's
                         normalizer divides by (paper equation 3).
    sigma_ratio          the larger of the two over the smaller (Appendix A.1).
    p1, p2               whitened unit branch directions Sigma^{1/2} w_i / sigma_i
                         (equation 4).
    cos_theta            p1 . p2, the whitened branch cosine (equation 6).
    theta_rad            arccos of it, in radians.
    cos_half, sin_half   cos(theta/2) and sin(theta/2), which are exactly the norms
                         of p_plus and p_minus (text after equation 6).
    p_plus, p_minus      (p1 + p2)/2 and (p1 - p2)/2, unnormalised (equation 6).
    bisector_hat         p_plus normalised: the output direction the block presents.
    axis_hat             p_minus normalised: the separation axis. Sign rule: branch 0
                         minus branch 1, never flipped here.
    bisector_defined,    False where the corresponding norm is below DEGENERATE_NORM;
    axis_defined         the unit vector is NaN there.
    overlap_bisector     signed bisector_hat . v_max.
    overlap_axis         signed axis_hat . v_max. The paper's "alignment" is the
                         absolute value; aggregation takes it, geometry keeps the sign.
    lambda_plus          bisector_hat^T Sigma bisector_hat (equation 15).
    lambda_minus         axis_hat^T Sigma axis_hat (equation 15).
    cross                bisector_hat^T Sigma axis_hat, zero when the axis is an
                         eigenvector of Sigma.
    B                    lambda_plus sin^2(theta/2) + lambda_minus cos^2(theta/2)
                         (section 3.5).
    variance_share_axis  overlap_axis squared: the fraction of the axis's response
                         variance that v_max supplies.
    cos_theta_euclid     the plain cosine of the two kernels.
    axis_hat_euclid      normalised difference of the two unit kernels, plain metric.
    overlap_axis_euclid  signed axis_hat_euclid . v_max.
    bisector_inv_sigma_quotient  bisector_hat^T Sigma^{-1} bisector_hat, which
                         relates the whitened and Euclidean angles.
    """

    sigma1: np.ndarray
    sigma2: np.ndarray
    sigma_ratio: np.ndarray
    p1: np.ndarray
    p2: np.ndarray
    cos_theta: np.ndarray
    theta_rad: np.ndarray
    cos_half: np.ndarray
    sin_half: np.ndarray
    p_plus: np.ndarray
    p_minus: np.ndarray
    bisector_hat: np.ndarray
    axis_hat: np.ndarray
    bisector_defined: np.ndarray
    axis_defined: np.ndarray
    overlap_bisector: np.ndarray
    overlap_axis: np.ndarray
    lambda_plus: np.ndarray
    lambda_minus: np.ndarray
    cross: np.ndarray
    B: np.ndarray
    variance_share_axis: np.ndarray
    cos_theta_euclid: np.ndarray
    axis_hat_euclid: np.ndarray
    overlap_axis_euclid: np.ndarray
    bisector_inv_sigma_quotient: np.ndarray

    @property
    def theta_deg(self) -> np.ndarray:
        return np.degrees(self.theta_rad)

    def scalars(self) -> dict[str, np.ndarray]:
        """The per-channel scalar fields, by name (no vectors), for storage."""
        out = {}
        for k, v in asdict(self).items():
            if isinstance(v, np.ndarray) and v.shape == self.theta_rad.shape:
                out[k] = v
        out["theta_deg"] = self.theta_deg
        return out


def _unit(v: np.ndarray, floor: float = DEGENERATE_NORM) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalise along the last axis. Returns (unit vector, norm, defined) where
    ``defined`` is False and the unit vector NaN wherever the norm is below ``floor``."""
    n = np.linalg.norm(v, axis=-1)
    defined = n >= floor
    safe = np.where(defined, n, 1.0)
    u = v / safe[..., None]
    u = np.where(defined[..., None], u, np.nan)
    return u, n, defined


def geometry(state: BlockState, cov: CovarianceEstimate) -> Geometry:
    """Compute the whitened (and Euclidean) geometry of every channel.

    Inputs: a ``BlockState`` with kernels ``[..., C, D]`` and a ``CovarianceEstimate``
    of the same D. Output: a ``Geometry`` with the fields documented on the class.

    Implements paper equations 3, 4, 6, 15 and 16 and the definition of B:
      sigma_i     = sqrt(w_i^T Sigma w_i)
      p_i         = Sigma^{1/2} w_i / sigma_i
      cos theta   = p_1 . p_2   ( = w_1^T Sigma w_2 / (sigma_1 sigma_2) )
      p_plus      = (p_1 + p_2)/2,  |p_plus|  = cos(theta/2)
      p_minus     = (p_1 - p_2)/2,  |p_minus| = sin(theta/2)
      lambda_plus = bisector_hat^T Sigma bisector_hat,  lambda_minus likewise for the axis
      B           = lambda_plus sin^2(theta/2) + lambda_minus cos^2(theta/2)

    Conventions relied on: the axis is p_1 - p_2 (branch 0 minus branch 1), and
    v_max has the sign rule of ``CovarianceEstimate``. The function applies no
    threshold, no mask and no aggregation, and converts no units: angles are in
    radians here, with ``theta_deg`` available as a property.
    """
    w1 = np.asarray(state.w1, dtype=np.float64)
    w2 = np.asarray(state.w2, dtype=np.float64)
    if w1.shape != w2.shape or w1.shape[-1] != cov.dim:
        raise ValueError(f"kernel shapes {w1.shape}, {w2.shape} do not match Sigma of dim {cov.dim}")
    sigma_matrix, sigma_sqrt = cov.sigma, cov.sqrt
    v_max = cov.v_max

    sigma1 = np.sqrt(np.einsum("...d,de,...e->...", w1, sigma_matrix, w1))
    sigma2 = np.sqrt(np.einsum("...d,de,...e->...", w2, sigma_matrix, w2))
    sigma_ratio = np.maximum(sigma1, sigma2) / np.minimum(sigma1, sigma2)
    # whitened unit directions; normalising by the actual norm of Sigma^{1/2} w keeps
    # |p_i| = 1 to machine precision even though sigma_i is computed from Sigma itself
    p1, _, _ = _unit(w1 @ sigma_sqrt)
    p2, _, _ = _unit(w2 @ sigma_sqrt)

    cos_theta = np.clip(np.einsum("...d,...d->...", p1, p2), -1.0, 1.0)
    theta = np.arccos(cos_theta)
    p_plus = 0.5 * (p1 + p2)
    p_minus = 0.5 * (p1 - p2)
    bisector_hat, cos_half, bisector_defined = _unit(p_plus)
    axis_hat, sin_half, axis_defined = _unit(p_minus)

    def rayleigh(u: np.ndarray, u2: np.ndarray | None = None) -> np.ndarray:
        return np.einsum("...d,de,...e->...", u, sigma_matrix, u if u2 is None else u2)

    lambda_plus = rayleigh(bisector_hat)
    lambda_minus = rayleigh(axis_hat)
    cross = rayleigh(bisector_hat, axis_hat)
    B = lambda_plus * sin_half ** 2 + lambda_minus * cos_half ** 2
    overlap_bisector = bisector_hat @ v_max
    overlap_axis = axis_hat @ v_max

    # Euclidean twins, plain inner product on the kernels
    e1, _, _ = _unit(w1)
    e2, _, _ = _unit(w2)
    cos_theta_euclid = np.clip(np.einsum("...d,...d->...", e1, e2), -1.0, 1.0)
    axis_hat_euclid, _, _ = _unit(e1 - e2)
    overlap_axis_euclid = axis_hat_euclid @ v_max
    inv_quot = np.einsum("...d,de,...e->...", bisector_hat, cov.inv_sqrt @ cov.inv_sqrt, bisector_hat)

    return Geometry(
        sigma1=sigma1, sigma2=sigma2, sigma_ratio=sigma_ratio, p1=p1, p2=p2,
        cos_theta=cos_theta, theta_rad=theta, cos_half=cos_half, sin_half=sin_half,
        p_plus=p_plus, p_minus=p_minus, bisector_hat=bisector_hat, axis_hat=axis_hat,
        bisector_defined=bisector_defined, axis_defined=axis_defined,
        overlap_bisector=overlap_bisector, overlap_axis=overlap_axis,
        lambda_plus=lambda_plus, lambda_minus=lambda_minus, cross=cross, B=B,
        variance_share_axis=overlap_axis ** 2,
        cos_theta_euclid=cos_theta_euclid, axis_hat_euclid=axis_hat_euclid,
        overlap_axis_euclid=overlap_axis_euclid, bisector_inv_sigma_quotient=inv_quot,
    )


# ---------------------------------------------------------------------------
# Section 4. Forcing
# ---------------------------------------------------------------------------
# The link between kernel space and the whitened frame. For a kernel w with
# sigma = sqrt(w^T Sigma w) and p = Sigma^{1/2} w / sigma, the Jacobian of p with
# respect to w is
#
#     d p / d w = (1/sigma) P Sigma^{1/2},     P = I - p p^T,
#
# so a kernel displacement dw moves the direction by (1/sigma) P Sigma^{1/2} dw
# to first order. Two consequences used throughout:
#   - the kernel gradient of a loss that depends on w only through v = gamma (p_1 + p_2)
#     is g_i = (gamma / sigma_i) Sigma^{1/2} P_i r with r = dL/dv   (paper, section 3.2);
#   - a radial displacement dw = c w gives Sigma^{1/2} dw = c sigma p, which P
#     annihilates: weight decay moves no direction, exactly.
# Nothing here assumes sigma_1 = sigma_2.


@dataclass
class DirectionChange:
    """The first-order response of the block's geometry to a pair of branch-direction
    displacements dp_1, dp_2 (each tangent to its own branch direction). Every field
    is linear in (dp_1, dp_2); the same object describes a rate (per unit flow time)
    or a step (per optimizer step), whichever the inputs were.

    dp1, dp2         [..., C, D]  the inputs, tangent to p_1 and p_2.
    d_p_minus        [..., C, D]  (dp_1 - dp_2)/2, the change of the unnormalised axis.
    d_axis_hat       [..., C, D]  change of the unit axis: (I - axis axis^T) d_p_minus / sin(theta/2)
                                  (paper equation 14).
    d_overlap_axis   [..., C]     v_max . d_axis_hat, the change of the signed overlap (equation 33's left side).
    d_theta_rad      [..., C]     -(dp_1 . p_2 + p_1 . dp_2) / sin(theta), the exact derivative of
                                  arccos(p_1 . p_2); equal to 2 axis . d_p_minus / cos(theta/2)
                                  (the form behind equation 39).
    d_bisector_hat   [..., C, D]  (I - bisector bisector^T) (dp_1 + dp_2)/2 / cos(theta/2).
    d_lambda_plus    [..., C]     2 (Sigma bisector) . d_bisector_hat  (equation 41's gradient).
    d_overlap_bisector [..., C]   v_max . d_bisector_hat.
    defined          [..., C]     False where sin(theta), cos(theta/2) or sin(theta/2) is
                                  below DEGENERATE_NORM; the affected fields are NaN there.
    """

    dp1: np.ndarray
    dp2: np.ndarray
    d_p_minus: np.ndarray
    d_axis_hat: np.ndarray
    d_overlap_axis: np.ndarray
    d_theta_rad: np.ndarray
    d_bisector_hat: np.ndarray
    d_lambda_plus: np.ndarray
    d_overlap_bisector: np.ndarray
    defined: np.ndarray

    @property
    def d_theta_deg(self) -> np.ndarray:
        return np.degrees(self.d_theta_rad)


def _safe_div(num: np.ndarray, den: np.ndarray, floor: float = DEGENERATE_NORM) -> tuple[np.ndarray, np.ndarray]:
    ok = np.abs(den) >= floor
    out = num / np.where(ok, den, 1.0)
    return np.where(ok, out, np.nan), ok


def direction_change(geom: Geometry, cov: CovarianceEstimate, dp1: np.ndarray, dp2: np.ndarray) -> DirectionChange:
    """The core linear map: branch-direction displacements to geometry changes.

    Inputs: the geometry at the current state and two displacements ``[..., C, D]``
    tangent to p_1 and p_2 (the function does not project them; callers pass
    tangent vectors by construction). Output: a ``DirectionChange``.

    This is the only place the derivatives of the axis, the angle, the bisector and
    lambda_plus are written down. The theory's flow (paper equation 5), the applied
    optimizer step, every term group and the sigma-mismatch split all go through
    this function with different inputs, so their sums close by linearity.
    """
    sigma_matrix = cov.sigma
    v_max = cov.v_max
    d_p_minus = 0.5 * (dp1 - dp2)
    d_p_plus = 0.5 * (dp1 + dp2)
    axis, bisector = geom.axis_hat, geom.bisector_hat
    # unit axis and unit bisector: remove the radial part, divide by the norm
    tang_minus = d_p_minus - np.einsum("...d,...d->...", axis, d_p_minus)[..., None] * axis
    d_axis_hat, ok_a = _safe_div(tang_minus, geom.sin_half[..., None])
    tang_plus = d_p_plus - np.einsum("...d,...d->...", bisector, d_p_plus)[..., None] * bisector
    d_bisector_hat, ok_b = _safe_div(tang_plus, geom.cos_half[..., None])
    sin_theta = np.sin(geom.theta_rad)
    num = -(np.einsum("...d,...d->...", dp1, geom.p2) + np.einsum("...d,...d->...", geom.p1, dp2))
    d_theta, ok_t = _safe_div(num, sin_theta)
    d_lambda_plus = 2.0 * np.einsum("...d,de,...e->...", bisector, sigma_matrix, d_bisector_hat)
    defined = ok_a[..., 0] & ok_b[..., 0] & ok_t & geom.axis_defined & geom.bisector_defined
    return DirectionChange(
        dp1=dp1, dp2=dp2, d_p_minus=d_p_minus, d_axis_hat=d_axis_hat,
        d_overlap_axis=d_axis_hat @ v_max, d_theta_rad=d_theta, d_bisector_hat=d_bisector_hat,
        d_lambda_plus=d_lambda_plus, d_overlap_bisector=d_bisector_hat @ v_max, defined=defined,
    )


def branch_pushes(geom: Geometry, cov: CovarianceEstimate, f1: np.ndarray, f2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The first-order direction change per unit step for kernel-space vectors f_i.

    For a kernel step dw_i = -eta f_i the direction moves by
        dp_i = -(eta / sigma_i) P_i Sigma^{1/2} f_i .
    This function returns the two vectors -(1/sigma_i) P_i Sigma^{1/2} f_i, shape
    ``[..., C, D]``; multiply by eta for the step, or read them as a rate. The
    formula uses Sigma^{1/2} only, never Sigma^{-1/2}, so it is exact for the
    momentum buffer including its weight-decay part, which is annihilated by P_i.
    """
    sigma_sqrt = cov.sqrt

    def push(f, p, sigma):
        y = f @ sigma_sqrt
        y = y - np.einsum("...d,...d->...", p, y)[..., None] * p
        return -y / sigma[..., None]

    return push(f1, geom.p1, geom.sigma1), push(f2, geom.p2, geom.sigma2)


@dataclass
class Forcing:
    """The whitened output gradient r recovered from two kernel-space vectors, and
    every named scalar of the paper's equations built from it.

    Inputs recorded: ``f1, f2`` ``[..., C, D]`` the kernel-space vectors (the
    gradients, or the momentum buffers), ``f_gamma`` ``[..., C]`` the matching entry
    for gamma (or None), ``kind`` a label ("gradient" or "buffer").

    Recovery (all ``[..., C]`` unless a vector). With q_i = (sigma_i / gamma) Sigma^{-1/2} f_i
    and a_i = P_i q_i its tangent part, an exact gradient has a_i = P_i r, and then
        axis . (a_1 + a_2)     = 2 cos^2(theta/2) (axis . r)
        bisector . (a_1 + a_2) = 2 sin^2(theta/2) (bisector . r)
        dL/dgamma              = 2 cos(theta/2)   (bisector . r)
        Q (a_1 + a_2) / 2      = Q r,   Q a_1 = Q a_2 = Q r
    with Q the projector off the branch plane. Each division is singular at one end
    of the angle range only, and the conditioning factor is reported.

    r_axis           axis . r, from the first line; conditioning cos^2(theta/2).
    r_bisector_kernel bisector . r from the second line; conditioning sin^2(theta/2).
    r_bisector_gamma bisector . r from the gamma entry; conditioning cos(theta/2). None if f_gamma is None.
    r_bisector       the one used to assemble r: the gamma route for a gradient (when the
                     gamma entry is given), the kernel route for a buffer, since only the
                     kernel buffers move the kernels (see the comment in ``forcing``).
    r_out            [..., C, D]  Q (a_1 + a_2)/2, the out-of-plane part.
    r                [..., C, D]  r_bisector bisector + r_axis axis + r_out.
    cond_axis, cond_bisector_kernel, cond_bisector_gamma   the three conditioning factors.
    r_vmax           v_max . r.
    r_axis_sigma     axis . Sigma r  (the third coupling of equation 17).
    r_out_norm       |r_out|.
    K                bisector . Sigma (I - bisector bisector^T) r  (equation 40).
    gamma_dot        -f_gamma, the scale's rate (equation 5) or its step; None if f_gamma is None.
    gamma_dot_from_r -2 cos(theta/2) r_bisector_kernel, the same from the kernels.
    drive            gamma gamma_dot / (2 sigma^2) with 1/sigma^2 = the mean of 1/sigma_1^2 and
                     1/sigma_2^2, the coefficient of the equal-sigma part of the exact split
                     (Appendix A.13); this is the one definition of sigma^2 in the drive.
    res_reconstruct1, res_reconstruct2   relative error of (gamma/sigma_i) Sigma^{1/2} P_i r against f_i:
                     zero for an exact gradient in the frame BatchNorm used, the momentum memory for a buffer.
    res_out          |Q (a_1 - a_2)| / |Q (a_1 + a_2)| : disagreement of the two branches on the out-of-plane part.
    res_bisector_routes  |r_bisector_kernel - r_bisector_gamma| / (|.| + |.|), where both are defined.
    radial1, radial2 p_i . q_i / |q_i|: zero for a gradient, the decay and memory content of a buffer.
    push1, push2     [..., C, D]  -(1/sigma_i) P_i Sigma^{1/2} f_i, the direction change per unit step.
    """

    kind: str
    f1: np.ndarray
    f2: np.ndarray
    f_gamma: np.ndarray | None
    r_axis: np.ndarray
    r_bisector_kernel: np.ndarray
    r_bisector_gamma: np.ndarray | None
    r_bisector: np.ndarray
    r_out: np.ndarray
    r: np.ndarray
    cond_axis: np.ndarray
    cond_bisector_kernel: np.ndarray
    cond_bisector_gamma: np.ndarray
    r_vmax: np.ndarray
    r_axis_sigma: np.ndarray
    r_out_norm: np.ndarray
    K: np.ndarray
    gamma_dot: np.ndarray | None
    gamma_dot_from_r: np.ndarray
    drive: np.ndarray | None
    res_reconstruct1: np.ndarray
    res_reconstruct2: np.ndarray
    res_out: np.ndarray
    res_bisector_routes: np.ndarray
    radial1: np.ndarray
    radial2: np.ndarray
    push1: np.ndarray
    push2: np.ndarray


def _rel(a: np.ndarray, b: np.ndarray, vector: bool) -> np.ndarray:
    """Relative difference |a - b| / (|a| + |b| + tiny): with ``vector`` True the
    norms are taken along the last axis (the D axis of a ``[..., C, D]`` array), with
    ``vector`` False the comparison is elementwise. The tiny term only prevents 0/0."""
    if vector:
        return np.linalg.norm(a - b, axis=-1) / (np.linalg.norm(a, axis=-1) + np.linalg.norm(b, axis=-1) + 1e-300)
    return np.abs(a - b) / (np.abs(a) + np.abs(b) + 1e-300)


def forcing(geom: Geometry, cov: CovarianceEstimate, f1: np.ndarray, f2: np.ndarray,
            f_gamma: np.ndarray | None, gamma: np.ndarray, kind: str = "gradient") -> Forcing:
    """Recover r from two kernel-space vectors and build the paper's scalars.

    Inputs: the geometry, the covariance, ``f1, f2`` ``[..., C, D]`` (float64), the
    matching gamma entry ``f_gamma`` ``[..., C]`` or None, ``gamma`` ``[..., C]``,
    and a label. Output: a ``Forcing`` (see its docstring for every field and the
    formulas).

    The function makes no assumption that ``f`` is a gradient: for the momentum
    buffer the same algebra gives the r whose flow best matches the applied step,
    and the reconstruction residuals say how well. It does not threshold on
    conditioning; it reports the conditioning factors, and the checks section
    decides what to flag.
    """
    sigma_matrix, sigma_sqrt, sigma_inv_sqrt = cov.sigma, cov.sqrt, cov.inv_sqrt
    v_max = cov.v_max
    p1, p2 = geom.p1, geom.p2
    axis, bisector = geom.axis_hat, geom.bisector_hat
    cos_half, sin_half = geom.cos_half, geom.sin_half
    gamma_column = gamma[..., None]

    q1 = (geom.sigma1[..., None] / gamma_column) * (f1 @ sigma_inv_sqrt)
    q2 = (geom.sigma2[..., None] / gamma_column) * (f2 @ sigma_inv_sqrt)
    dot = lambda a, b: np.einsum("...d,...d->...", a, b)
    a1 = q1 - dot(p1, q1)[..., None] * p1
    a2 = q2 - dot(p2, q2)[..., None] * p2
    asum = a1 + a2

    cond_axis = cos_half ** 2
    cond_bisector_kernel = sin_half ** 2
    cond_bisector_gamma = cos_half
    r_axis, _ = _safe_div(dot(axis, asum), 2.0 * cond_axis)
    r_bisector_kernel, _ = _safe_div(dot(bisector, asum), 2.0 * cond_bisector_kernel)
    if f_gamma is not None:
        r_bisector_gamma, _ = _safe_div(f_gamma, 2.0 * cond_bisector_gamma)
    else:
        r_bisector_gamma = None
    # Which route assembles r. For a gradient the two routes are one identity and the
    # gamma route is the better conditioned, so it is used when available. For the
    # momentum buffer they are not the same object: the gamma buffer moves gamma, the
    # kernel buffers move the kernels, and each carries its own momentum history along
    # a geometry that has rotated since. The direction change is driven by the kernel
    # buffers alone, so their own bisector component is the one that belongs in r.
    # (Using the gamma route for a buffer produced a large closing term cancelled by an
    # equally large remainder on the first real recording, 2026-09-02.)
    if kind == "gradient" and r_bisector_gamma is not None:
        r_bisector = r_bisector_gamma
    else:
        r_bisector = r_bisector_kernel
    # out-of-plane part: remove the two in-plane components of asum/2
    half = 0.5 * asum
    r_out = half - dot(axis, half)[..., None] * axis - dot(bisector, half)[..., None] * bisector
    r = r_bisector[..., None] * bisector + r_axis[..., None] * axis + r_out
    # replace NaN in-plane pieces (degenerate channels) by zero so r stays finite; the
    # conditioning fields tell the checks which channels these are
    r = np.where(np.isfinite(r), r, 0.0)

    sigma_r = r @ sigma_matrix
    sigma_bisector = bisector @ sigma_matrix
    K = dot(sigma_bisector, r) - geom.lambda_plus * dot(bisector, r)
    gamma_dot = None if f_gamma is None else -f_gamma
    gamma_dot_from_r = -2.0 * cos_half * r_bisector_kernel
    inv_sigma2_mean = 0.5 * (1.0 / geom.sigma1 ** 2 + 1.0 / geom.sigma2 ** 2)
    drive = None if gamma_dot is None else gamma * gamma_dot * inv_sigma2_mean / 2.0

    def reconstruct(p, sigma):
        Pr = r - dot(p, r)[..., None] * p
        return (gamma_column / sigma[..., None]) * (Pr @ sigma_sqrt)

    res1 = _rel(reconstruct(p1, geom.sigma1), f1, vector=True)
    res2 = _rel(reconstruct(p2, geom.sigma2), f2, vector=True)
    diff = a1 - a2
    out_diff = diff - dot(axis, diff)[..., None] * axis - dot(bisector, diff)[..., None] * bisector
    res_out = np.linalg.norm(out_diff, axis=-1) / (2.0 * np.linalg.norm(r_out, axis=-1) + 1e-300)
    res_routes = np.full(r_axis.shape, np.nan) if r_bisector_gamma is None else _rel(r_bisector_kernel, r_bisector_gamma, vector=False)
    radial1 = dot(p1, q1) / (np.linalg.norm(q1, axis=-1) + 1e-300)
    radial2 = dot(p2, q2) / (np.linalg.norm(q2, axis=-1) + 1e-300)
    push1, push2 = branch_pushes(geom, cov, f1, f2)

    return Forcing(
        kind=kind, f1=f1, f2=f2, f_gamma=f_gamma, r_axis=r_axis, r_bisector_kernel=r_bisector_kernel,
        r_bisector_gamma=r_bisector_gamma, r_bisector=r_bisector, r_out=r_out, r=r, cond_axis=cond_axis,
        cond_bisector_kernel=cond_bisector_kernel, cond_bisector_gamma=cond_bisector_gamma, r_vmax=r @ v_max,
        r_axis_sigma=dot(axis, sigma_r), r_out_norm=np.linalg.norm(r_out, axis=-1), K=K,
        gamma_dot=gamma_dot, gamma_dot_from_r=gamma_dot_from_r, drive=drive,
        res_reconstruct1=res1, res_reconstruct2=res2, res_out=res_out,
        res_bisector_routes=res_routes, radial1=radial1, radial2=radial2, push1=push1, push2=push2,
    )


def kernel_vectors_from_r(geom: Geometry, cov: CovarianceEstimate, r: np.ndarray, gamma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The kernel gradients an exact loss with output gradient r would give:
    g_i = (gamma / sigma_i) Sigma^{1/2} P_i r, shape ``[..., C, D]`` each. Linear in r,
    so the kernel-space image of a component of r is that component's share of the
    gradient; this is what the momentum-filtered term groups accumulate."""
    sigma_sqrt = cov.sqrt
    gamma_column = gamma[..., None]
    dot = lambda a, b: np.einsum("...d,...d->...", a, b)
    out = []
    for p, sigma in ((geom.p1, geom.sigma1), (geom.p2, geom.sigma2)):
        Pr = r - dot(p, r)[..., None] * p
        out.append((gamma_column / sigma[..., None]) * (Pr @ sigma_sqrt))
    return out[0], out[1]


def sigma_split_of_vectors(geom: Geometry, cov: CovarianceEstimate, f1: np.ndarray, f2: np.ndarray) -> tuple[DirectionChange, DirectionChange]:
    """The sigma-mismatch split of the direction change produced by any pair of
    kernel-space vectors, with no r involved (Appendix A.13's split applied to the
    applied step). The push of branch i is -(1/sigma_i) P_i Sigma^{1/2} f_i, and
    1/sigma_i = sigma_i (m +/- delta) with m the mean of 1/sigma_1^2 and 1/sigma_2^2
    and delta half their difference, so
        equal_sigma part:    dp_i = -sigma_i m P_i Sigma^{1/2} f_i
        sigma_mismatch part: dp_1 = -sigma_1 delta (...) f_1,  dp_2 = +sigma_2 delta (...) f_2.
    The two sum to the full push exactly."""
    push1, push2 = branch_pushes(geom, cov, f1, f2)     # -(1/sigma_i) P_i Sigma^{1/2} f_i
    inv1, inv2 = 1.0 / geom.sigma1 ** 2, 1.0 / geom.sigma2 ** 2
    mean_inverse_variance = 0.5 * (inv1 + inv2)
    half_difference_inverse_variance = 0.5 * (inv1 - inv2)
    # push_i = sigma_i^2 (m +/- delta) * push_i / (sigma_i^2 (1/sigma_i^2)) : rescale each push
    w1 = (geom.sigma1 ** 2)[..., None]
    w2 = (geom.sigma2 ** 2)[..., None]
    equal = direction_change(geom, cov, push1 * w1 * mean_inverse_variance[..., None], push2 * w2 * mean_inverse_variance[..., None])
    mismatch = direction_change(geom, cov, push1 * w1 * half_difference_inverse_variance[..., None],
                                -push2 * w2 * half_difference_inverse_variance[..., None])
    return equal, mismatch


def flow_from_r(geom: Geometry, cov: CovarianceEstimate, r: np.ndarray, gamma: np.ndarray) -> DirectionChange:
    """The theory's gradient flow of the two directions for a given r (paper equation 5):
        dp_i/dt = -(gamma / sigma_i^2) P_i Sigma P_i r ,
    returned as a ``DirectionChange`` per unit flow time. Linear in r, so evaluating
    it on a component of r gives that component's term group."""
    sigma_matrix = cov.sigma
    dot = lambda a, b: np.einsum("...d,...d->...", a, b)

    def one(p, sigma):
        Pr = r - dot(p, r)[..., None] * p
        SPr = Pr @ sigma_matrix
        PSPr = SPr - dot(p, SPr)[..., None] * p
        return -(gamma / sigma ** 2)[..., None] * PSPr

    return direction_change(geom, cov, one(geom.p1, geom.sigma1), one(geom.p2, geom.sigma2))


@dataclass
class Decomposition:
    """A ``DirectionChange`` split into the paper's term groups, all summing to ``total``.

    Every entry is a ``DirectionChange`` produced by the same linear map on a piece
    of the input, so ``total`` equals the sum of the pieces of either split to
    rounding (check 5), and equal_sigma plus sigma_mismatch equals ``total`` (check 6).

    Primary split, by the branch-plane components of r (the split behind the
    paper's two regimes):
      from_bisector_component   r_bisector * bisector : the terms carrying p_plus . r
                                (the scale-gradient, or Oja, group of section 3.3, regime one)
      from_axis_component       r_axis * axis : the terms carrying p_minus . r
                                (the transfer group, regime two)
      from_out_of_plane         r_out : the terms carrying the gradient off the branch plane
    Secondary split, loud against quiet:
      from_vmax_component       (v_max . r) v_max
      from_quiet_component      r minus that
    Sigma-mismatch split of the same total (Appendix A.13): with h_i = P_i Sigma P_i r and
    m, delta the mean and half-difference of 1/sigma_1^2 and 1/sigma_2^2,
      dp_1 = -gamma (m + delta) h_1,  dp_2 = -gamma (m - delta) h_2, so
      equal_sigma      uses dp_i = -gamma m h_i          (what sigma_1 = sigma_2 would give)
      sigma_mismatch   uses dp_1 = -gamma delta h_1, dp_2 = +gamma delta h_2
    ``remainder`` is the part of the applied displacement that no r explains (zero
    for an exact gradient in the frame BatchNorm used; the momentum memory and
    nothing else for a buffer, since weight decay's push is exactly zero).
    """

    total: DirectionChange
    from_bisector_component: DirectionChange
    from_axis_component: DirectionChange
    from_out_of_plane: DirectionChange
    from_vmax_component: DirectionChange
    from_quiet_component: DirectionChange
    equal_sigma: DirectionChange
    sigma_mismatch: DirectionChange
    remainder: DirectionChange
    scale: float

    def groups_primary(self) -> dict[str, DirectionChange]:
        return {"from_bisector_component": self.from_bisector_component,
                "from_axis_component": self.from_axis_component,
                "from_out_of_plane": self.from_out_of_plane, "remainder": self.remainder}

    def groups_loud(self) -> dict[str, DirectionChange]:
        return {"from_vmax_component": self.from_vmax_component,
                "from_quiet_component": self.from_quiet_component, "remainder": self.remainder}


def _scaled(dc: DirectionChange, k: float) -> DirectionChange:
    return DirectionChange(**{f: (getattr(dc, f) * k if f not in ("defined",) else getattr(dc, f))
                              for f in DirectionChange.__dataclass_fields__})


def decompose(geom: Geometry, cov: CovarianceEstimate, forc: Forcing, gamma: np.ndarray, scale: float = 1.0) -> Decomposition:
    """Split the direction change produced by ``forc`` into the paper's term groups.

    ``scale`` multiplies every piece: 1 for a rate per unit flow time, eta for the
    change over one optimizer step of learning rate eta. ``total`` is the exact
    first-order change from the pushes of ``forc`` (no r involved), and the pieces
    are the flow of equation 5 evaluated on each component of the recovered r,
    plus ``remainder`` = total minus the flow of the full r.
    """
    axis, bisector = geom.axis_hat, geom.bisector_hat
    v_max = cov.v_max
    r = forc.r
    total = direction_change(geom, cov, forc.push1, forc.push2)
    piece = lambda rr: flow_from_r(geom, cov, rr, gamma)
    full = piece(r)
    r_b = forc.r_bisector[..., None] * bisector
    r_a = forc.r_axis[..., None] * axis
    r_v = (r @ v_max)[..., None] * v_max
    parts = {
        "from_bisector_component": piece(np.where(np.isfinite(r_b), r_b, 0.0)),
        "from_axis_component": piece(np.where(np.isfinite(r_a), r_a, 0.0)),
        "from_out_of_plane": piece(forc.r_out),
        "from_vmax_component": piece(r_v),
        "from_quiet_component": piece(r - r_v),
    }
    # sigma-mismatch split of the flow of the full r
    sigma_matrix = cov.sigma
    dot = lambda a, b: np.einsum("...d,...d->...", a, b)

    def h(p):
        Pr = r - dot(p, r)[..., None] * p
        SPr = Pr @ sigma_matrix
        return SPr - dot(p, SPr)[..., None] * p

    h1, h2 = h(geom.p1), h(geom.p2)
    inv1, inv2 = 1.0 / geom.sigma1 ** 2, 1.0 / geom.sigma2 ** 2
    mean_inverse_variance = 0.5 * (inv1 + inv2)
    half_difference_inverse_variance = 0.5 * (inv1 - inv2)
    eq = direction_change(geom, cov, -(gamma * mean_inverse_variance)[..., None] * h1,
                          -(gamma * mean_inverse_variance)[..., None] * h2)
    mis = direction_change(geom, cov, -(gamma * half_difference_inverse_variance)[..., None] * h1,
                           (gamma * half_difference_inverse_variance)[..., None] * h2)
    remainder = direction_change(geom, cov, total.dp1 - full.dp1, total.dp2 - full.dp2)
    out = Decomposition(total=_scaled(total, scale), **{k: _scaled(p, scale) for k, p in parts.items()},
                        equal_sigma=_scaled(eq, scale), sigma_mismatch=_scaled(mis, scale),
                        remainder=_scaled(remainder, scale), scale=scale)
    return out


# ---------------------------------------------------------------------------
# Section 5. Dynamics
# ---------------------------------------------------------------------------


@dataclass
class Realized:
    """The change between two consecutive geometries of the same block, in the units
    used for every first-order prediction.

    flipped         [..., C] bool  True where the later axis was sign-flipped to keep
                                   continuity with the earlier one (dot product positive).
    d_theta_deg     [..., C]       later minus earlier whitened angle, degrees.
    d_overlap_axis  [..., C]       later minus earlier signed overlap, after the flip.
    d_alignment     [..., C]       later minus earlier absolute overlap.
    d_logodds_alignment [..., C]   change of log(a / sqrt(1 - a^2)) with a the absolute overlap.
    d_lambda_plus   [..., C]
    d_overlap_bisector [..., C]
    d_p1, d_p2      [..., C, D]    later minus earlier whitened branch directions.
    d_axis_hat      [..., C, D]    later (flipped) minus earlier unit axis.
    """

    flipped: np.ndarray
    d_theta_deg: np.ndarray
    d_overlap_axis: np.ndarray
    d_alignment: np.ndarray
    d_logodds_alignment: np.ndarray
    d_lambda_plus: np.ndarray
    d_overlap_bisector: np.ndarray
    d_p1: np.ndarray
    d_p2: np.ndarray
    d_axis_hat: np.ndarray


def logodds(a: np.ndarray) -> np.ndarray:
    """log(a / sqrt(1 - a^2)) for an absolute overlap a in [0, 1); the coordinate in
    which a multiplicative growth of the overlap is additive (Appendix A.11)."""
    a = np.clip(a, 0.0, 1.0 - 1e-15)
    return np.log(a / np.sqrt(1.0 - a ** 2))


def realized_change(before: Geometry, after: Geometry) -> Realized:
    """Difference two geometries (same channels, consecutive states).

    The axis sign is matched by continuity: where the later unit axis has negative
    dot product with the earlier one it is flipped, and the flip is recorded. Angles
    are differenced in degrees, overlaps as signed numbers after the flip, and the
    log-odds of the absolute overlap is differenced as well. Channels whose axis is
    undefined in either state get NaN in the axis-based fields.
    """
    dot = np.einsum("...d,...d->...", before.axis_hat, after.axis_hat)
    flipped = np.where(np.isfinite(dot), dot < 0, False)
    sign = np.where(flipped, -1.0, 1.0)
    axis_after = after.axis_hat * sign[..., None]
    overlap_after = after.overlap_axis * sign
    a_b, a_a = np.abs(before.overlap_axis), np.abs(after.overlap_axis)
    return Realized(
        flipped=flipped,
        d_theta_deg=np.degrees(after.theta_rad - before.theta_rad),
        d_overlap_axis=overlap_after - before.overlap_axis,
        d_alignment=a_a - a_b,
        d_logodds_alignment=logodds(a_a) - logodds(a_b),
        d_lambda_plus=after.lambda_plus - before.lambda_plus,
        d_overlap_bisector=after.overlap_bisector - before.overlap_bisector,
        d_p1=after.p1 - before.p1, d_p2=after.p2 - before.p2,
        d_axis_hat=axis_after - before.axis_hat,
    )


def first_order_logodds(before: Geometry, dc: DirectionChange) -> np.ndarray:
    """The first-order change of the log-odds of the absolute overlap implied by a
    ``DirectionChange``: d logodds = sign(overlap) d_overlap / (a (1 - a^2)) with a the
    absolute overlap. Undefined (NaN) at a = 0 or a = 1."""
    a = np.abs(before.overlap_axis)
    num = np.sign(before.overlap_axis) * dc.d_overlap_axis
    out, _ = _safe_div(num, a * (1.0 - a ** 2))
    return out


def step_from_vectors(before: Geometry, cov: CovarianceEstimate, dw1: np.ndarray, dw2: np.ndarray) -> DirectionChange:
    """First-order geometry change for the applied kernel steps dw_i = w_i(after) - w_i(before).

    Uses dp_i = (1/sigma_i) P_i Sigma^{1/2} dw_i, the exact Jacobian; equivalent to
    ``branch_pushes`` with f_i = -dw_i and eta = 1. This is what check 7 compares
    against the realized direction changes at the step sizes that occur.
    """
    push1, push2 = branch_pushes(before, cov, -dw1, -dw2)
    return direction_change(before, cov, push1, push2)


# ---------------------------------------------------------------------------
# Section 6. Checks
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    """One identity evaluated on every channel-step.

    name        which identity.
    level       "exact" (has a tolerance and produces flags) or "diagnostic"
                (distribution only; ``flag`` is all False).
    tolerance   the tolerance used, or NaN for diagnostics.
    error       [..., C] the relative error.
    flag        [..., C] bool, error above tolerance (exact) or False (diagnostic).
    conditioned [..., C] bool, the channel-step is well conditioned for this identity
                (True everywhere when no conditioning factor applies). The report
                splits the error distribution by this mask; nothing is dropped.
    note        one sentence saying what a failure means.
    """

    name: str
    level: str
    tolerance: float
    error: np.ndarray
    flag: np.ndarray
    conditioned: np.ndarray
    note: str

    def summary(self) -> dict[str, Any]:
        e = self.error
        finite = np.isfinite(e)
        out: dict[str, Any] = {"name": self.name, "level": self.level, "tolerance": self.tolerance,
                               "note": self.note, "n": int(e.size), "n_nonfinite": int((~finite).sum()),
                               "fraction_flagged": float(self.flag.mean()) if e.size else float("nan")}
        for label, m in (("conditioned", self.conditioned & finite), ("ill_conditioned", ~self.conditioned & finite)):
            x = e[m]
            out[label] = {"n": int(x.size), "p50": float(np.median(x)) if x.size else float("nan"),
                          "p99": float(np.quantile(x, 0.99)) if x.size else float("nan"),
                          "max": float(x.max()) if x.size else float("nan"),
                          "fraction_flagged": float(self.flag[m].mean()) if x.size else float("nan")}
        return out


def _make(name, level, tol, err, cond, note) -> CheckResult:
    err = np.asarray(err, dtype=np.float64)
    cond = np.broadcast_to(np.asarray(cond, dtype=bool), err.shape)
    flag = (err > tol) if level == "exact" else np.zeros(err.shape, dtype=bool)
    flag = flag & np.isfinite(err) if level == "exact" else flag
    return CheckResult(name, level, float(tol) if level == "exact" else float("nan"), err, flag, cond, note)


def check_applied_step(w_pre: np.ndarray, w_post: np.ndarray, buffer: np.ndarray, lr: np.ndarray) -> CheckResult:
    """Check 1: w_post = w_pre - eta * buffer, per channel (norm over the kernel).
    Inputs ``[T, C, D]`` and ``lr`` ``[T]`` (or ``[T, 1]``). Float32 tolerance. A
    failure means the hooks did not see the step they think they saw, or the
    learning rate was read from the wrong parameter group; the recording is then
    not usable and derive stops."""
    lr = np.asarray(lr, dtype=np.float64).reshape(-1, *([1] * (w_pre.ndim - 1)))
    pred = w_pre - lr * buffer
    err = _rel(pred, w_post, vector=True)
    return _make("applied_step", "exact", TOL_FLOAT32, err, True,
                 "w_post must equal w_pre minus the learning rate times the momentum buffer that was applied")


def check_momentum_recurrence(buffer_t: np.ndarray, buffer_prev: np.ndarray, grad_t: np.ndarray,
                              w_pre_t: np.ndarray, weight_decay: float, momentum: float) -> CheckResult:
    """Check 2: torch SGD's buffer recurrence buf_t = mu buf_{t-1} + g_t + wd w_t on
    consecutive recorded steps. Inputs ``[T-1, C, D]`` aligned so that row t of
    ``buffer_t`` follows row t of ``buffer_prev``. A failure means the recorded
    gradient is not the one the optimizer consumed (for instance, a scaled or
    clipped gradient), or the decay is not what the config says."""
    pred = momentum * buffer_prev + grad_t + weight_decay * w_pre_t
    return _make("momentum_recurrence", "exact", TOL_FLOAT32, _rel(pred, buffer_t, vector=True), True,
                 "the momentum buffer must follow mu * previous + gradient + weight decay * weights")


def check_orthogonality(geom: Geometry) -> CheckResult:
    """Check 4: bisector and axis unit vectors are orthogonal and of unit norm."""
    dot = np.abs(np.einsum("...d,...d->...", geom.bisector_hat, geom.axis_hat))
    n1 = np.abs(np.linalg.norm(geom.bisector_hat, axis=-1) - 1.0)
    n2 = np.abs(np.linalg.norm(geom.axis_hat, axis=-1) - 1.0)
    err = np.where(geom.axis_defined & geom.bisector_defined, np.maximum(dot, np.maximum(n1, n2)), np.nan)
    return _make("orthogonality", "exact", TOL_ALGEBRA, err, geom.axis_defined & geom.bisector_defined,
                 "p_plus and p_minus must be orthogonal unit vectors")


def check_recovery(forc: Forcing, exact: bool = True, tol: float = TOL_GRADIENT_IDENTITY) -> CheckResult:
    """Check 3: the recovered r reproduces both input vectors. Exact for a gradient in
    the frame BatchNorm used (``exact`` True, tolerance TOL_GRADIENT_IDENTITY);
    conditioned where both divisions of the recovery are at least
    WELL_CONDITIONED_MIN. For a buffer, or for a gradient in the fixed frame, the
    same number is the momentum memory or the frame difference and is reported as a
    diagnostic (``exact`` False or ``forc.kind`` not "gradient")."""
    err = np.maximum(forc.res_reconstruct1, forc.res_reconstruct2)
    cond = (forc.cond_axis >= WELL_CONDITIONED_MIN) & (
        (forc.cond_bisector_gamma >= WELL_CONDITIONED_MIN) if forc.r_bisector_gamma is not None
        else (forc.cond_bisector_kernel >= WELL_CONDITIONED_MIN))
    level = "exact" if (forc.kind == "gradient" and exact) else "diagnostic"
    return _make("recovery_" + forc.kind, level, tol, err, cond,
                 "the recovered r must reproduce the kernel-space vectors it was built from")


def check_gamma_gradient(forc: Forcing, exact: bool = True, tol: float = TOL_GRADIENT_IDENTITY) -> CheckResult:
    """Check 3b: the gamma entry equals 2 cos(theta/2) (bisector . r) with the bisector
    component taken from the kernels; conditioned on sin^2(theta/2) >= WELL_CONDITIONED_MIN
    (the kernel route's division) and on cos(theta/2) likewise. Exact only for a
    gradient in the batch frame (``exact`` True); a diagnostic otherwise."""
    if forc.f_gamma is None:
        raise ValueError("check_gamma_gradient needs f_gamma")
    err = _rel(forc.r_bisector_kernel, forc.r_bisector_gamma, vector=False)
    cond = (forc.cond_bisector_kernel >= WELL_CONDITIONED_MIN) & (forc.cond_bisector_gamma >= WELL_CONDITIONED_MIN)
    level = "exact" if (forc.kind == "gradient" and exact) else "diagnostic"
    return _make("gamma_gradient_" + forc.kind, level, tol, err, cond,
                 "the gamma gradient and the kernel gradients must agree on bisector . r")


def _sum_dc(parts: list[DirectionChange]) -> DirectionChange:
    fields = [f for f in DirectionChange.__dataclass_fields__ if f != "defined"]
    return DirectionChange(**{f: sum(getattr(p, f) for p in parts) for f in fields},
                           defined=np.logical_and.reduce([p.defined for p in parts]))


def _closure_error(parts: list[DirectionChange], total: DirectionChange) -> np.ndarray:
    """|sum of the parts - total| relative to the sum of the parts' magnitudes, the
    larger of the axis-change vector and the angle. Normalising by the parts rather
    than by the total keeps the error meaningful where the total crosses zero."""
    summed = _sum_dc(parts)
    scale_vec = sum(np.linalg.norm(p.d_p_minus, axis=-1) for p in parts)
    scale_ang = sum(np.abs(p.d_theta_rad) for p in parts)
    scale_lam = sum(np.abs(p.d_lambda_plus) for p in parts)
    # where the pushes themselves are numerically zero (a channel with no gradient,
    # moved only by decay), the ratio is meaningless and the identity holds trivially
    floor = 1e-12
    err_vec = np.linalg.norm(summed.d_p_minus - total.d_p_minus, axis=-1) / np.maximum(scale_vec, floor)
    err_ang = np.abs(summed.d_theta_rad - total.d_theta_rad) / np.maximum(scale_ang, floor)
    err_lam = np.abs(summed.d_lambda_plus - total.d_lambda_plus) / np.maximum(scale_lam, floor)
    return np.maximum.reduce([np.where(scale_vec < floor, 0.0, err_vec), np.where(scale_ang < floor, 0.0, err_ang),
                              np.where(scale_lam < floor, 0.0, err_lam)])


def check_closure(dec: Decomposition) -> list[CheckResult]:
    """Check 5: each split of the decomposition sums to the total, on the axis change,
    the angle and lambda_plus. Pure float64 algebra; the error is relative to the
    parts' magnitudes (see ``_closure_error``)."""
    out = []
    for label, groups in (("primary", dec.groups_primary()), ("loud_quiet", dec.groups_loud())):
        err = _closure_error(list(groups.values()), dec.total)
        out.append(_make("closure_" + label, "exact", TOL_ALGEBRA, err, dec.total.defined,
                         "the term groups must sum to the full first-order change"))
    return out


def check_mismatch_split(dec: Decomposition) -> CheckResult:
    """Check 6: equal_sigma + sigma_mismatch + remainder = total, same error measure."""
    err = _closure_error([dec.equal_sigma, dec.sigma_mismatch, dec.remainder], dec.total)
    return _make("mismatch_split", "exact", TOL_ALGEBRA, err, dec.total.defined,
                 "the equal-sigma part, the mismatch part and the remainder must sum to the total")


def check_first_order_directions(before: Geometry, after: Geometry, cov: CovarianceEstimate,
                                 dw1: np.ndarray, dw2: np.ndarray) -> tuple[CheckResult, np.ndarray]:
    """Check 7: at the recorded step sizes, the realized change of each whitened
    direction minus its first-order prediction is second order in the step.

    The criterion is second order itself, not a fixed tolerance: with the relative
    whitened step of branch i defined as |Sigma^{1/2} dw_i| / sigma_i, the residual
    |realized dp_i - first-order dp_i| must be at most SECOND_ORDER_BOUND times that
    step squared (the larger of the two branches is reported). Returns the check,
    whose ``error`` is the residual divided by the squared relative step (so the
    tolerance is SECOND_ORDER_BOUND), and the plain relative residual
    |residual| / |realized| ``[..., C]`` for information. This is the test that the
    gradients and buffers in the recording are the ones that moved the parameters,
    and that the module's linear maps are the right Jacobians at real step sizes.
    """
    fo = step_from_vectors(before, cov, dw1, dw2)
    real = realized_change(before, after)
    resid = np.maximum(np.linalg.norm(fo.dp1 - real.d_p1, axis=-1), np.linalg.norm(fo.dp2 - real.d_p2, axis=-1))
    rel_step = np.maximum(np.linalg.norm(dw1 @ cov.sqrt, axis=-1) / before.sigma1,
                          np.linalg.norm(dw2 @ cov.sqrt, axis=-1) / before.sigma2)
    error = resid / (rel_step ** 2 + 1e-300)
    plain = resid / (np.maximum(np.linalg.norm(real.d_p1, axis=-1), np.linalg.norm(real.d_p2, axis=-1)) + 1e-300)
    return _make("first_order_directions", "exact", SECOND_ORDER_BOUND, error, True,
                 "the realized direction change must match the Jacobian applied to the recorded step "
                 "up to a second-order remainder"), plain


def checks_report(results: list[CheckResult]) -> dict[str, Any]:
    """A JSON-ready report: one entry per check with fraction flagged and the error
    distribution split by conditioning, plus the conventions used."""
    return {"conventions": CONVENTIONS, "checks": [r.summary() for r in results]}


# ---------------------------------------------------------------------------
# Section 7. Aggregation
# ---------------------------------------------------------------------------
# Every mask and every summary convention lives here and nowhere else.


def open_mask(geom: Geometry) -> np.ndarray:
    """Boolean ``[..., C]``: whitened angle at least OPEN_ANGLE_DEG."""
    return geom.theta_deg >= OPEN_ANGLE_DEG


def parked_mask(geom: Geometry) -> np.ndarray:
    """Boolean ``[..., C]``: absolute axis overlap with v_max at least PARKED_OVERLAP.
    Undefined axes (parallel branches) are never parked."""
    return np.where(geom.axis_defined, np.abs(geom.overlap_axis) >= PARKED_OVERLAP, False)


def alignment(geom: Geometry) -> np.ndarray:
    """The paper's alignment: the absolute overlap of the separation axis with v_max."""
    return np.abs(geom.overlap_axis)


@dataclass
class Summary:
    """One aggregated number with its provenance.

    name        what was summarised.
    population  in words: which channels, which steps or epochs, which seeds.
    n           how many values went in.
    median, q25, q75, p10, p90, mean, std  the statistics over those values.
    """

    name: str
    population: str
    n: int
    median: float
    q25: float
    q75: float
    p10: float
    p90: float
    mean: float
    std: float

    def as_row(self) -> dict[str, Any]:
        return asdict(self)


def summarize(values: np.ndarray, mask: np.ndarray | None, name: str, population: str) -> Summary:
    """Summarise ``values`` where ``mask`` is True (all values if ``mask`` is None).

    NaN values are excluded and their count is appended to ``population`` so the
    exclusion is visible. Pooling over seeds or epochs is the caller's choice and
    must be stated in ``population``.
    """
    x = np.asarray(values, dtype=np.float64)
    if mask is not None:
        x = x[np.asarray(mask, dtype=bool)]
    x = x.ravel()
    n_nan = int(np.isnan(x).sum())
    x = x[~np.isnan(x)]
    if n_nan:
        population = f"{population}; {n_nan} NaN excluded"
    if x.size == 0:
        return Summary(name, population, 0, *([float("nan")] * 7))
    q = np.quantile(x, [0.5, 0.25, 0.75, 0.1, 0.9])
    return Summary(name, population, int(x.size), float(q[0]), float(q[1]), float(q[2]),
                   float(q[3]), float(q[4]), float(x.mean()), float(x.std()))


def climb_windows(alignment_by_step: np.ndarray, low: float = CLIMB_LOW, high: float = CLIMB_HIGH) -> np.ndarray:
    """For each channel, the window of its final ascent from ``low`` to ``high``.

    Input ``[T, C]`` absolute overlaps on consecutive recorded steps. Output an
    integer array ``[C, 2]`` with (start, end): end is the first step at or above
    ``high`` that comes after the channel has been below ``low``; start is the step
    after the last one below ``low`` that precedes end. So the window is the last
    uninterrupted ascent through ``low``, not the first crossing: a channel whose
    overlap crosses 0.5, wanders back through zero and climbs again (seed 42,
    channel 23 of the first recording) gets only its final climb, where the per-step
    log-odds changes are regular. Channels that never satisfy both get (-1, -1).
    """
    a = np.asarray(alignment_by_step, dtype=np.float64)
    T, C = a.shape
    out = np.full((C, 2), -1, dtype=np.int64)
    for c in range(C):
        below = np.flatnonzero(a[:, c] < low)
        if below.size == 0:
            continue
        reached = np.flatnonzero((a[:, c] >= high) & (np.arange(T) > below[0]))
        if reached.size == 0:
            continue
        end = reached[0]
        start = below[below < end][-1] + 1
        out[c] = (start, end)
    return out
