"""Synthetic model of the first-block pair with a deterministic normalizer.

The model is one normalized linear block (64 channels), a ReLU, a linear
readout, and a mean-squared error against a fixed two-layer teacher. Each
branch of a channel divides its projection wᵀx by σ = √(wᵀΣw), where Σ is the
empirical covariance of the fixed training sample. There are no batch
statistics, no running estimate and no mean subtraction (the sample is
centered, so wᵀx has zero mean by construction), so equations (3) to (7) of the
paper hold exactly, and full-batch gradient descent on the model is a
deterministic map.

Everything is written in the eigenbasis of Σ, which loses nothing because the
model is rotation-equivariant. Coordinate 0 is the top eigendirection v_max,
coordinate 1 is the runner-up, and so on down the spectrum. The whitened
input is the raw input divided coordinate-wise by the square root of the
eigenvalues, and the whitened branch direction p is the kernel times the
square root of the eigenvalues, divided by σ, which makes it a unit vector.

Names used throughout (chosen for readability on 2026-09-06; the keys of the
saved record and the keyword arguments of `train` keep their older names):
  eigenvalues      the spectrum of Σ, shape (input_dim,), descending, so
                   eigenvalues[0] is λ_max and eigenvalues[1] is λ_next
  inputs           the fixed training sample in raw coordinates,
                   shape (n_samples, input_dim)
  whitened_inputs  inputs divided coordinate-wise by √eigenvalues
  targets          the teacher's standardized outputs, (n_samples, n_outputs)
  kernels          the branch weights w, (n_branches, n_channels, input_dim);
                   n_branches is 1 for the single and 2 for the pair
  readout          the linear readout after the ReLU, (n_channels, n_outputs)
  gamma, beta      the per-channel scale and shift shared by both branches,
                   (n_channels,)
  sigmas           σ = √(wᵀΣw) per branch and channel, (n_branches, n_channels)
  whitened_dirs    the unit vectors p per branch and channel,
                   (n_branches, n_channels, input_dim)
  whitened_grad    the loss gradient with respect to the whitened effective
                   weight v = γ(p₁ + p₂), the r of the paper,
                   (n_channels, input_dim)
  sep_axis         the unit separation axis (p₁ − p₂)/|p₁ − p₂| of a pair channel
  bisector         the unit bisector (p₁ + p₂)/|p₁ + p₂| of a pair channel

History: agent synth_fc_teacher_c added the profile-matched teacher of section 2
(`make_teacher_profile`, hidden directions drawn in the whitened frame with
per-eigendirection energy taken from the CNN's measured block-0 kernel profile)
and the per-band energy readout of section 5 (`band_shares`, recorded as
`r_band`, `g_band`, `v_band`, `r_energy`, `g_energy` and `v_energy`), so that
the rig single's own whitened-gradient demand can be compared with the CNN's
measured demand profile. The profile teacher is retired; the results use the
whitened-isotropic teacher (`mode="whitened"` of `make_teacher`).

Sections
  1. spectrum and data
  2. teacher
  3. parameters and arms
  4. forward
  5. the measurement: per-channel geometry and the exact flow rates
  6. training
  7. end-of-run curvature
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

DTYPE = torch.float64
COV_PATH = (Path(__file__).resolve().parents[2] / "experiments" / "reparam_pinned_norm"
            / "cifar100_patch_cov.pt")   # the package root, so the path holds in any agent dir

OPEN_DEG = 10.0      # a channel is open when its whitened angle is at least this
PARK_ALIGN = 0.95    # and parked when |p_minus . v_max| is at least this


# ----------------------------------------------------------------------------
# 1. spectrum and data
# ----------------------------------------------------------------------------

def cifar_spectrum(path: Path = COV_PATH) -> torch.Tensor:
    """The 27 eigenvalues of the committed CIFAR-100 3x3 patch covariance,
    descending (19.41, 2.32, ...). This is the clean-image spectrum."""
    covariance = torch.load(path, map_location="cpu").to(DTYPE)
    return torch.linalg.eigvalsh(covariance).flip(0)


def ladder_spectrum(eigenvalues: torch.Tensor, ratio: float | None) -> torch.Tensor:
    """The CIFAR tail with λ_max set to `ratio` times λ_next. None keeps the
    spectrum as it is (the clean CIFAR-100 spectrum has a ratio of 8.4)."""
    eigenvalues = eigenvalues.clone()
    if ratio is not None:
        eigenvalues[0] = ratio * eigenvalues[1]
    return eigenvalues


def _spectrum_unsorted(spec_name: str, eigenvalues: torch.Tensor | None = None) -> torch.Tensor:
    """The spectrum families, by name. Each family moves one thing about the
    spectrum while holding the rest, and each is built here in its own order;
    `spectrum` sorts the result descending afterwards.

    "cifar": the clean CIFAR-100 patch spectrum, unchanged.
    "r<x>": the CIFAR tail with λ_max set to x times λ_next (the first
        campaign's ladder; the tail, and so λ_eff, does not move).
    "flat<x>": every quiet eigenvalue equal to the CIFAR λ_next and λ_max at
        x times it, so λ_max/λ_eff equals λ_max/λ_next; x = 1 is isotropic.
    "zca<eps>": the CIFAR spectrum through the regularised ZCA whitening of the
        paper's section 4.2, each eigenvalue mapped to λ/(λ + eps), which
        flattens both ratios together.
    "tail<x>", "deg<x>", "ftail<level>", "pin<mu>a<level>" and "scale<x>" are
        described at their branches below.
    """
    eigenvalues = cifar_spectrum() if eigenvalues is None else eigenvalues.clone()
    if spec_name == "cifar":
        return eigenvalues
    if spec_name.startswith("flat"):
        spectrum_out = torch.full_like(eigenvalues, float(eigenvalues[1]))
        spectrum_out[0] = float(spec_name[4:]) * eigenvalues[1]
        return spectrum_out
    if spec_name.startswith("zca"):
        eps = float(spec_name[3:])
        return eigenvalues / (eigenvalues + eps)
    if spec_name.startswith("tail"):
        # λ_max and λ_next fixed, the rest of the tail scaled by x: this moves
        # μ_eff through the trace while leaving λ_max/λ_next untouched.
        spectrum_out = eigenvalues.clone()
        spectrum_out[2:] = spectrum_out[2:] * float(spec_name[4:])
        return spectrum_out
    if spec_name.startswith("deg"):
        # a degenerate loud pair: the top two eigenvalues equal and as large
        # as asked, the CIFAR tail underneath. μ_eff is zero however loud the
        # pair is.
        spectrum_out = eigenvalues.clone()
        spectrum_out[0] = spectrum_out[1] = float(spec_name[3:])
        return spectrum_out
    if spec_name.startswith("ftail"):
        # λ_max and λ_next held at the CIFAR values while the trace is raised
        # by lifting every remaining eigenvalue to one common level, kept below
        # λ_next so the ordering never changes. This is the one family that
        # moves the trace with the margin and the ratio both fixed, so it
        # separates the numerator of μ_eff from its denominator.
        level = float(spec_name[5:])
        spectrum_out = eigenvalues.clone()
        spectrum_out[2:] = min(level, float(eigenvalues[1]))
        return spectrum_out
    if spec_name.startswith("pin"):
        # μ_eff pinned while both top eigenvalues slide up over the CIFAR tail.
        # "pin<mu>a<level>": λ_next = level, λ_max = level + margin, and the
        # margin is chosen so that (λ_max − λ_next)/trace equals mu exactly,
        # which gives margin = mu (2 level + tail total)/(1 − mu). Raising the
        # level therefore holds the selection rate fixed while the tail's share
        # of the input variance shrinks toward nothing.
        mu_text, level_text = spec_name[3:].split("a")
        mu, level = float(mu_text), float(level_text)
        tail = eigenvalues[2:].clone()
        margin = mu * (2 * level + float(tail.sum())) / (1 - mu)
        return torch.cat([torch.tensor([level + margin, level], dtype=DTYPE), tail])
    if spec_name.startswith("scale"):
        # the whole spectrum rescaled, which every quantity in the block is
        # invariant to, so this is a null check rather than a cell.
        return eigenvalues * float(spec_name[5:])
    if spec_name.startswith("r"):
        return ladder_spectrum(eigenvalues, float(spec_name[1:]))
    raise ValueError(f"unknown spectrum spec {spec_name!r}")


def spectrum(spec_name: str, eigenvalues: torch.Tensor | None = None) -> torch.Tensor:
    """`_spectrum_unsorted` with the eigenvalues put back in descending order.

    Coordinate 0 is v_max everywhere in this rig: `loud_dims`, the `vmax_axis`
    of the measurement, `align`, `jjt_vmax`, the band boundaries and the
    summaries in run.py all read index 0 as the top eigendirection and index 1
    as the runner-up. The "tail<x>" family scales the tail, and for x above
    λ_next divided by the third eigenvalue it lifted the tail past the top two,
    which left every one of those readings pointing at the wrong direction, so
    the sort is applied here once for every family. The families that were
    already descending ("cifar", "r", "flat", "zca", "deg", "scale") are
    unchanged by it.
    """
    spectrum_out = _spectrum_unsorted(spec_name, eigenvalues)
    return torch.sort(spectrum_out, descending=True).values


def loud_dims(eigenvalues: torch.Tensor, tol: float = 0.05) -> int:
    """How many leading eigenvalues lie within `tol` of λ_max: the dimension
    of the loud subspace the cooled arm cools (1 for a unique v_max).

    Returns 0 when every eigenvalue is within `tol` of the top, because then
    the loud subspace is the whole space and there is no direction to cool:
    scaling all coordinates of the kernel gradient is a change of the
    kernel's learning rate, not the paper's hand-cooling intervention. The
    caller skips the cooled arm on a 0.
    """
    n_loud = int((eigenvalues >= (1 - tol) * eigenvalues.max()).sum())
    return 0 if n_loud == eigenvalues.numel() else n_loud


def make_data(eigenvalues: torch.Tensor, n_samples: int, generator: torch.Generator) -> torch.Tensor:
    """n_samples centered samples whose empirical covariance is exactly diag(eigenvalues)."""
    input_dim = eigenvalues.numel()
    sample = torch.randn(n_samples, input_dim, generator=generator, dtype=DTYPE)
    sample = sample - sample.mean(0, keepdim=True)
    covariance = sample.T @ sample / n_samples
    cov_eigvals, cov_eigvecs = torch.linalg.eigh(covariance)
    sample = sample @ (cov_eigvecs * cov_eigvals.rsqrt()) @ cov_eigvecs.T   # empirical covariance = I
    return sample * eigenvalues.sqrt()


# ----------------------------------------------------------------------------
# 2. teacher
# ----------------------------------------------------------------------------

def _set_vmax_share(hidden_whitened: torch.Tensor, row_variance: torch.Tensor,
                    vmax_share: float | None) -> torch.Tensor:
    """Scale row 0 of a whitened teacher draw (the v_max coordinate of every
    hidden unit) so that the share of the teacher's whitened energy on v_max
    is `vmax_share`, in expectation over the draw. `row_variance` is the
    variance each row of the draw was given (ones for a whitened-isotropic
    draw, the eigenvalues for a raw-isotropic one). None leaves the draw
    untouched. 0 zeroes the row. The gain that gives share s is
    √(s · quiet energy / ((1 − s) · row-0 variance)), where the quiet energy
    is the total variance of the other rows; in the whitened frame with 27
    coordinates a share of 1/27 is the untouched draw and a share of 1/2 is a
    gain of √26."""
    if vmax_share is None:
        return hidden_whitened
    if not 0.0 <= vmax_share < 1.0:
        raise ValueError(f"vmax_share must lie in [0, 1), got {vmax_share}")
    if vmax_share == 0.0:
        hidden_whitened[0] = 0.0
        return hidden_whitened
    if float(row_variance[0]) == 0.0:
        raise ValueError("this teacher has no v_max coordinate to scale (quiet mode)")
    quiet_energy = float(row_variance[1:].sum())
    gain = math.sqrt(vmax_share * quiet_energy / ((1.0 - vmax_share) * float(row_variance[0])))
    hidden_whitened[0] = hidden_whitened[0] * gain
    return hidden_whitened


def _set_tail_share(hidden_whitened: torch.Tensor, row_variance: torch.Tensor,
                    tail_share: float | None) -> torch.Tensor:
    """Move the teacher's quiet energy between the fast quiet directions (ranks 1
    to 7) and the slow tail (ranks 8 to 26, the "slow" band of BANDS) at a fixed
    spectrum, leaving the v_max row and the total quiet energy untouched in
    expectation. `tail_share` is the share of the quiet energy that lies in the
    slow tail; None leaves the draw as it is (19/26 ≈ 0.73 for a whitened
    draw). The fast rows are scaled by √((1 − s) Q / F) and the slow rows by
    √(s Q / T), with Q, F and T the quiet, fast and slow row variances summed.
    Added 2026-09-06 for item 13 of the vault note: does the gain need signal
    in the slow directions, with the pair's cooling held fixed?"""
    if tail_share is None:
        return hidden_whitened
    if not 0.0 < tail_share < 1.0:
        raise ValueError(f"tail_share must lie in (0, 1), got {tail_share}")
    lo, hi = BANDS["slow"]
    quiet = float(row_variance[1:].sum())
    fast = float(row_variance[1:lo].sum())
    slow = float(row_variance[lo:hi].sum())
    if fast == 0.0 or slow == 0.0:
        raise ValueError("this teacher has no energy in one of the quiet bands to move")
    hidden_whitened[1:lo] = hidden_whitened[1:lo] * math.sqrt((1.0 - tail_share) * quiet / fast)
    hidden_whitened[lo:hi] = hidden_whitened[lo:hi] * math.sqrt(tail_share * quiet / slow)
    return hidden_whitened


def make_teacher(eigenvalues: torch.Tensor, n_hidden: int, n_outputs: int,
                 vmax_share: float | None, generator: torch.Generator, mode: str = "quiet",
                 tail_share: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """A fixed two-layer ReLU teacher whose dependence on v_max is a dial.

    `mode` sets the baseline. "quiet" draws each hidden unit isotropic in the
    whitened frame and then zeroes its v_max coordinate, so the target is
    blind to the loud direction. "whitened" is the same draw with the v_max
    coordinate kept, so v_max takes one share in input_dim of the teacher's
    energy, like every other direction; this is the teacher the results use.
    "raw" draws isotropic in the raw coordinates, so the input's own variance
    weights each direction and v_max takes the share λ_max over the trace.

    `vmax_share` then sets the share of the teacher's whitened energy that
    lies on v_max, by scaling that one coordinate of every unit up or down
    (see `_set_vmax_share`), and `tail_share` sets the share of the remaining
    quiet energy that lies in the slow tail (see `_set_tail_share`); the tail
    dial is applied first, so the v_max share is exact on top of it. None
    keeps the mode's baseline, which is what the spectrum cells use; a
    whitened teacher at 1/27 is the same as None. The v_max dial
    replaced, on 2026-09-06, a dial that overwrote the v_max coordinate of a
    fraction of the units with a random sign; no result in the vault note
    depends on the old dial except the loud-share-1 row of its Table 2.

    The draw is made in the whitened frame (`hidden_whitened`) and taken back
    to raw coordinates by dividing each row by the square root of its
    eigenvalue, so that a unit's raw weight applied to the raw input equals
    its whitened weight applied to the whitened input. Returns
    (teacher_hidden, teacher_readout) with teacher_hidden of shape
    (input_dim, n_hidden) and teacher_readout (n_hidden, n_outputs).
    """
    input_dim = eigenvalues.numel()
    hidden_whitened = torch.randn(input_dim, n_hidden, generator=generator, dtype=DTYPE)
    row_variance = torch.ones(input_dim, dtype=DTYPE)
    if mode == "quiet":
        hidden_whitened[0] = 0.0        # the target is blind to the loud direction
        row_variance[0] = 0.0
    elif mode == "raw":
        # Isotropic in the raw coordinates instead: the input's own variance
        # weights each direction, so the loud direction takes the share
        # λ_max over the trace of the teacher's energy.
        hidden_whitened = hidden_whitened * eigenvalues.sqrt().unsqueeze(1)
        row_variance = eigenvalues.clone()
    elif mode != "whitened":            # "whitened": isotropic, loud direction included
        raise ValueError(f"unknown teacher mode {mode!r}")
    hidden_whitened = _set_tail_share(hidden_whitened, row_variance, tail_share)
    hidden_whitened = _set_vmax_share(hidden_whitened, row_variance, vmax_share)
    teacher_hidden = hidden_whitened / eigenvalues.sqrt().unsqueeze(1)
    teacher_readout = torch.randn(n_hidden, n_outputs, generator=generator, dtype=DTYPE) / math.sqrt(n_hidden)
    return teacher_hidden, teacher_readout


def teacher_targets(inputs: torch.Tensor, teacher_hidden: torch.Tensor,
                    teacher_readout: torch.Tensor) -> torch.Tensor:
    """Teacher outputs, standardized per output to zero mean and unit variance."""
    outputs = torch.relu(inputs @ teacher_hidden) @ teacher_readout
    return (outputs - outputs.mean(0)) / outputs.std(0)


PROFILE_DIR = Path(__file__).resolve().parent
# Band boundaries by eigenvalue rank, the convention of agent synth_fc_teacher_a:
# rank 0 is v_max, ranks 1 to 5 are the next five, ranks 6 to 26 the rest, and the
# slow tail (eigenvalues below λ_max/100 in the CIFAR-100 spectrum) is ranks 8
# to 26. The ranks are fixed so that the bands read the same in every cell.
BANDS = {"vmax": (0, 1), "next5": (1, 6), "rest": (6, 27), "slow": (8, 27)}
BAND_NAMES = tuple(BANDS)
DISJOINT_BANDS = ((0, 1), (1, 6), (6, 8), (8, 27))   # the construction bands of the profile teacher


def load_profiles() -> dict[str, np.ndarray]:
    """The CNN's measured block-0 profiles over the clean-image eigenbasis, from
    agent synth_fc_teacher_a (copied into this directory): the whitened kernel
    energy share per eigendirection of the trained single's selector channels
    (the 22 of 96 that keep more than half their energy on v_max), of the other
    74 channels, and the share of the gradient with respect to the whitened
    kernel over training (the demand profile). Each is a length-27 vector that
    sums to one, index = eigenvalue rank, 0 = v_max."""
    return {name: np.load(PROFILE_DIR / f"profile_{name}.npy")
            for name in ("kernel_selectors", "kernel_others", "gradient_demand")}


def shift_profile(energy: np.ndarray, shift: float) -> np.ndarray:
    """Move a fraction `shift` of the energy of ranks 1 to 7 to ranks 8 to 17,
    spread equally over those ten ranks (the directive's remedy when the
    single's slow-tail demand falls short). Returns a new vector summing to one."""
    energy = energy.copy()
    moved = shift * energy[1:8].sum()
    energy[1:8] *= 1.0 - shift
    energy[8:18] += moved / 10.0
    return energy / energy.sum()


def make_teacher_profile(eigenvalues: torch.Tensor, n_hidden: int, n_outputs: int,
                         vmax_share: float | None, generator: torch.Generator,
                         selector_energy: np.ndarray, other_energy: np.ndarray,
                         selector_frac: float = 0.23,
                         ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The profile-matched teacher. Each hidden unit's direction is drawn in the
    whitened frame so that its whitened energy on each of the four disjoint
    rank bands (v_max; ranks 1 to 5; ranks 6 and 7; ranks 8 to 26) equals the
    profile's band total exactly, while the split inside a band is free: the
    band's coordinates are a Gaussian draw weighted by the square root of the
    profile's per-rank share and then rescaled to the band's energy, so a
    unit's direction is a unit vector in the whitened frame. The first
    `round(selector_frac * n_hidden)` units follow the selector profile (about
    73 percent on v_max) and the others the non-selector profile (about 3
    percent on v_max, 60 percent on the next five, 23 percent in the slow
    tail). `vmax_share` acts on top as in `make_teacher`, scaling the v_max
    row so that the mean unit's energy share on v_max becomes that value;
    None, which the campaign used, leaves the profile's own v_max signal. The
    draw order is the same as `make_teacher`'s (the direction matrix, then
    the readout). Returns
    (teacher_hidden, teacher_readout, energy) with `energy` the realized mean
    over units of each unit's whitened energy share per eigendirection, shape
    (input_dim,)."""
    input_dim = eigenvalues.numel()
    hidden_whitened = torch.randn(input_dim, n_hidden, generator=generator, dtype=DTYPE)
    n_selectors = int(round(selector_frac * n_hidden))
    scale = torch.zeros(input_dim, n_hidden, dtype=DTYPE)
    scale[:, :n_selectors] = torch.as_tensor(np.sqrt(selector_energy), dtype=DTYPE).unsqueeze(1)
    scale[:, n_selectors:] = torch.as_tensor(np.sqrt(other_energy), dtype=DTYPE).unsqueeze(1)
    hidden_whitened = hidden_whitened * scale
    for lo, hi in DISJOINT_BANDS:
        target_energy = torch.zeros(n_hidden, dtype=DTYPE)
        target_energy[:n_selectors] = float(np.sum(selector_energy[lo:hi]))
        target_energy[n_selectors:] = float(np.sum(other_energy[lo:hi]))
        have_energy = (hidden_whitened[lo:hi] * hidden_whitened[lo:hi]).sum(0).clamp_min(1e-300)
        hidden_whitened[lo:hi] *= (target_energy / have_energy).sqrt()
    mean_profile = torch.as_tensor(selector_frac * np.asarray(selector_energy)
                                   + (1 - selector_frac) * np.asarray(other_energy), dtype=DTYPE)
    hidden_whitened = _set_vmax_share(hidden_whitened, mean_profile, vmax_share)
    squared = hidden_whitened * hidden_whitened
    energy = (squared / squared.sum(0, keepdim=True)).mean(1)
    teacher_hidden = hidden_whitened / eigenvalues.sqrt().unsqueeze(1)
    teacher_readout = torch.randn(n_hidden, n_outputs, generator=generator, dtype=DTYPE) / math.sqrt(n_hidden)
    return teacher_hidden, teacher_readout, energy


# ----------------------------------------------------------------------------
# 3. parameters and arms
# ----------------------------------------------------------------------------

ARMS = ("single", "pair", "quad", "cooled", "boosted")
BRANCHES = {"pair": 2, "quad": 4}     # every other arm has one branch


def sigma_of(kernels: torch.Tensor, eigenvalues: torch.Tensor) -> torch.Tensor:
    """sqrt(w^T Sigma w) per branch and channel. `kernels` is
    (n_branches, n_channels, input_dim); the result is (n_branches, n_channels)."""
    return (kernels * kernels * eigenvalues).sum(-1).sqrt()


def whitened_directions(kernels: torch.Tensor, eigenvalues: torch.Tensor,
                        sigmas: torch.Tensor | None = None) -> torch.Tensor:
    """p = sqrt(eigenvalues) * w / sigma, unit vectors, shape
    (n_branches, n_channels, input_dim). Pass `sigmas` when the caller has
    already computed `sigma_of` for these kernels, so it is not recomputed."""
    if sigmas is None:
        sigmas = sigma_of(kernels, eigenvalues)
    return kernels * eigenvalues.sqrt() / sigmas.unsqueeze(-1)


def init_params(arm: str, eigenvalues: torch.Tensor, n_channels: int, n_outputs: int,
                generator: torch.Generator, init: str = "raw"):
    """Initialization-matched arms.

    Branch 0 and the readout are drawn first, so every arm at one seed shares
    them; the pair's second branch consumes the generator afterwards. `init`
    is "raw" for a Gaussian draw in the raw coordinates (which carries the
    whitened tilt toward v_max of Appendix A.14, as in the CNN) or "whitened"
    for a draw that is isotropic in the whitened frame (Appendix A.15).

    The pair's gamma is set per channel so the pair starts at the single's
    output scale: the summed whitened feature has standard deviation
    |p_1 + p_2|, so gamma is its reciprocal. The single and cooled arms start
    at gamma = 1. One beta per channel in every arm (the single-beta convention
    of the mechanism cell).

    Returns (kernels, gamma, beta, readout), each requiring grad.
    """
    input_dim = eigenvalues.numel()
    n_branches = BRANCHES.get(arm, 1)

    def draw_branch(shape):
        branch = torch.randn(*shape, generator=generator, dtype=DTYPE) / math.sqrt(input_dim)
        if init == "whitened":
            branch = branch / eigenvalues.sqrt()
        return branch

    first_branch = draw_branch((1, n_channels, input_dim))
    readout = torch.randn(n_channels, n_outputs, generator=generator, dtype=DTYPE) / math.sqrt(n_channels)
    if n_branches >= 2:
        # the extra branches are drawn after the readout, one after another, so
        # the pair's second branch is the quad's second branch at the same seed
        kernels = torch.cat([first_branch] + [draw_branch((1, n_channels, input_dim)) for _ in range(n_branches - 1)])
        whitened_dirs = whitened_directions(kernels, eigenvalues)
        gamma = 1.0 / whitened_dirs.sum(0).norm(dim=-1).clamp_min(1e-3)
    else:
        kernels = first_branch
        gamma = torch.ones(n_channels, dtype=DTYPE)
    beta = torch.zeros(n_channels, dtype=DTYPE)
    return (kernels.requires_grad_(), gamma.requires_grad_(), beta.requires_grad_(),
            readout.requires_grad_())


# ----------------------------------------------------------------------------
# 4. forward
# ----------------------------------------------------------------------------

def forward(inputs, eigenvalues, kernels, gamma, beta, readout):
    """One forward pass. Returns (output, preactivation).

    The preactivation of one channel, shape (n_samples, n_channels) over all
    channels, is γ times the sum over branches of wᵀx/σ, plus β. It keeps its
    gradient so that `train` can form the whitened gradient: the preactivation
    equals the whitened effective weight v = γ(p₁ + p₂) applied to the
    whitened input, plus β, so the loss gradient with respect to v is the
    preactivation gradient (transposed) times the whitened inputs."""
    n_branches, n_channels, input_dim = kernels.shape
    sigmas = sigma_of(kernels, eigenvalues)                                   # (n_branches, n_channels)
    branch_features = (inputs @ kernels.reshape(n_branches * n_channels, input_dim).T
                       ).reshape(-1, n_branches, n_channels) / sigmas
    preactivation = gamma * branch_features.sum(1) + beta
    preactivation.retain_grad()
    return torch.relu(preactivation) @ readout, preactivation


def loss_fn(output, targets):
    return ((output - targets) ** 2).mean()


# ----------------------------------------------------------------------------
# 5. the measurement: per-channel geometry and the exact flow rates
# ----------------------------------------------------------------------------

def _dot(left, right):
    return (left * right).sum(-1)


def band_shares(vectors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """For vectors of shape (n_channels, input_dim) in the whitened eigenbasis:
    the energy share of each channel's vector on each eigendirection,
    v_k^2 / |v|^2, shape (n_channels, input_dim), and its band totals in the
    order of BAND_NAMES, shape (n_channels, len(BANDS)). A zero vector gives
    zero shares."""
    shares = vectors * vectors
    shares = shares / shares.sum(-1, keepdim=True).clamp_min(1e-300)
    band_totals = torch.stack([shares[:, lo:hi].sum(-1) for lo, hi in BANDS.values()], dim=-1)
    return shares, band_totals


def _tangent_flow(whitened_dirs, eigenvalues, gamma, sigmas, whitened_grad):
    """The direction flow of equation (7) of the paper, for every branch: the
    rate of change of the whitened direction p of branch b is −(γ/σ_b²) times
    the whitened gradient projected onto the tangent space of p, multiplied by
    Σ, and projected onto that tangent space again. `whitened_dirs` is
    (n_branches, n_channels, input_dim), `whitened_grad` is
    (n_channels, input_dim), and the result has the shape of `whitened_dirs`."""
    tangent_grad = whitened_grad - whitened_dirs * _dot(whitened_dirs, whitened_grad).unsqueeze(-1)
    cov_tangent_grad = eigenvalues * tangent_grad                            # Sigma P_b r
    tangent_cov_tangent_grad = (cov_tangent_grad
                                - whitened_dirs * _dot(whitened_dirs, cov_tangent_grad).unsqueeze(-1))
    return -(gamma / sigmas ** 2).unsqueeze(-1) * tangent_cov_tangent_grad


def _quadratic_form_kernel(whitened_dirs, eigenvalues, gamma, sigmas, direction):
    """The kernel part of the preconditioner of equations (53) and (55) along
    a unit `direction`, shape (n_channels, input_dim) or (input_dim,): the sum
    over branches of (γ/σ_b)² times the Rayleigh quotient of Σ on the
    direction's projection onto the tangent space of branch b's whitened
    direction. Returns one value per channel."""
    tangent = direction - whitened_dirs * _dot(whitened_dirs, direction).unsqueeze(-1)
    return ((gamma / sigmas) ** 2 * _dot(tangent, eigenvalues * tangent)).sum(0)


def separation_modes(whitened_dirs: torch.Tensor):
    """The mode picture of a block with any number of branches. For each
    channel the branch directions p_b (unit vectors in the whitened frame) are
    split into their mean direction and the principal axes of their spread
    about it: the centred matrix (p_b − mean) of shape (n_branches, input_dim)
    is decomposed by a singular value decomposition, and its right singular
    vectors are the separation modes, ordered by the share of the spread they
    carry. For a pair there is one mode and it is the separation axis; for four
    branches there are three. Returns (modes, energy, mean_unit, mean_norm):
    modes (n_channels, n_branches − 1, input_dim) unit vectors with arbitrary
    sign, energy (n_channels, n_branches − 1) the normalised squared singular
    values, mean_unit (n_channels, input_dim) the unit mean direction (the
    bisector for a pair) and mean_norm (n_channels,) the mean's length
    (cos(θ/2) for a pair)."""
    mean = whitened_dirs.mean(0)                                  # (n_channels, input_dim)
    centred = (whitened_dirs - mean).permute(1, 0, 2)             # (n_channels, n_branches, input_dim)
    _, singular, modes = torch.linalg.svd(centred, full_matrices=False)
    k = whitened_dirs.shape[0] - 1
    singular, modes = singular[:, :k], modes[:, :k]
    energy = singular ** 2 / (singular ** 2).sum(-1, keepdim=True).clamp_min(1e-300)
    mean_norm = mean.norm(dim=-1).clamp_min(1e-300)
    return modes, energy, mean / mean_norm.unsqueeze(-1), mean_norm


def pairwise_angles(whitened_dirs: torch.Tensor) -> torch.Tensor:
    """The whitened angle between every pair of branches, shape
    (n_channels, n_branches (n_branches − 1) / 2), in the order (0,1), (0,2), ..."""
    n_branches = whitened_dirs.shape[0]
    cols = [torch.arccos(_dot(whitened_dirs[a], whitened_dirs[b]).clamp(-1.0, 1.0))
            for a in range(n_branches) for b in range(a + 1, n_branches)]
    return torch.stack(cols, dim=-1)


@torch.no_grad()
def measure(kernels, gamma, beta, readout, eigenvalues, whitened_grad):
    """Per-channel geometry and rates at one point of training. Every value
    is an (n_channels,) tensor unless noted. `whitened_grad` is the loss
    gradient with respect to the whitened effective weight, shape
    (n_channels, input_dim).

    Recorded for every arm: gamma, beta, the first branch's σ and kernel norm,
    the readout norm, the whitened gradient's v_max coordinate and norm, the
    rate of γ from equation (7) (`dgamma`), the preconditioner along v_max
    (`jjt_vmax`), and the energy shares by eigenvalue band of the whitened
    gradient (`r_band`), of the effective weight (`v_band`) and of the
    gradient with respect to each branch's whitened kernel (`g_band`).

    Recorded for the pair on top of that: the whitened angle θ between the two
    branch directions (`theta`, `cos`), the alignment of the separation axis
    with v_max (`align`) and of the bisector with v_max (`out_align`), the
    Rayleigh quotients of Σ on the bisector (`lam_plus`, the paper's λ₊) and
    on the separation axis (`lam_minus`, λ₋), the stability-condition value
    λ₊/(λ₋ − λ₊) (`stab`) next to cos²(θ/2) (`cos2_half`), the exact rates of
    the angle and of the alignment from the flow of equation (7) (`dtheta`,
    `dalign`), the two terms of the angle equation (`term_gamma`, `term_K`,
    and the loud product `K` itself), the projections of the whitened gradient
    onto the separation axis and the bisector that the sign assumption of
    section 3.3 is about (`pm_r`, `pp_r`, `sign_product`), and the
    preconditioner along the separation axis (`jjt_sep`). The γ-path term of
    equations (53) and (55), p pᵀ for a single and 4 p₊p₊ᵀ for the pair, is
    left out of `jjt_vmax` and `jjt_sep` (it annihilates the separation axis,
    and along v_max it is what the cooled arm cannot reach, since that arm
    acts on the kernel gradient alone) and included in `jjt_vmax_full`, which
    is the paper's section 4.2.4 quantity. For a single branch the record
    holds what applies to one direction.

    Conventions shared with the CNN measurement module
    (`structural_reparam.analysis.blockmeasure.measure`): the whitened
    direction, the angle, the separation axis as branch 0 minus branch 1, the
    alignment as the absolute overlap of that axis with v_max, λ₊ and λ₋ as
    Rayleigh quotients of the unit bisector and unit axis, K, the flow of
    equation (7), the open threshold of 10 degrees, the parked threshold of
    0.95, and 1/σ² as the mean of the inverse branch variances. The CNN module
    recovers r from the recorded kernel gradients; the rig reads it directly.
    """
    n_branches, n_channels, input_dim = kernels.shape
    sigmas = sigma_of(kernels, eigenvalues)                                  # (n_branches, n_channels)
    whitened_dirs = whitened_directions(kernels, eigenvalues, sigmas)        # (n_branches, n_channels, input_dim)
    vmax_axis = torch.zeros(input_dim, dtype=DTYPE)
    vmax_axis[0] = 1.0
    # whitened energy per eigendirection: of the whitened gradient (the
    # demand), and of the effective whitened weight itself, v = γ(p₁ + p₂)
    # (the kernel energy profile of agent synth_fc_teacher_a). Per channel by
    # band; the full per-direction vectors as the mean over channels (shares
    # sum to one per channel).
    grad_energy, grad_band = band_shares(whitened_grad)
    weight_energy, weight_band = band_shares(whitened_dirs.sum(0))
    # the gradient with respect to each branch's whitened kernel is the
    # whitened gradient projected onto the tangent space of that branch's
    # direction (times γ/σ_b, which drops out of a share): this is the
    # quantity agent synth_fc_teacher_a measured in the CNN. Per channel, the
    # mean over branches of the per-branch band shares.
    per_branch = [band_shares(whitened_grad
                              - whitened_dirs[b] * _dot(whitened_dirs[b], whitened_grad).unsqueeze(-1))
                  for b in range(n_branches)]
    kernel_grad_energy = torch.stack([energy for energy, _ in per_branch]).mean(0)
    kernel_grad_band = torch.stack([band for _, band in per_branch]).mean(0)
    record = {
        "gamma": gamma.clone(), "beta": beta.clone(),
        "sigma1": sigmas[0].clone(), "wnorm1": kernels[0].norm(dim=-1),
        "vnorm": readout.norm(dim=-1),
        "r_vmax": whitened_grad[:, 0].clone(), "rnorm": whitened_grad.norm(dim=-1),
        "dgamma": -_dot(whitened_dirs, whitened_grad).sum(0),                # gamma-dot of (7)
        "jjt_vmax": _quadratic_form_kernel(whitened_dirs, eigenvalues, gamma, sigmas, vmax_axis),
        # the same coefficient with the γ-path term of (53) and (55) added, p pᵀ
        # for a single and 4 p₊p₊ᵀ (p₊ the unnormalised half sum) for the pair:
        # this is the quantity the paper's section 4.2.4 cooling factor is
        # built from (scripts/analysis/jjt_top.py); the cooled arm's factor in
        # run.py uses the kernel part alone, which is what its intervention
        # can reach
        "jjt_vmax_full": (_quadratic_form_kernel(whitened_dirs, eigenvalues, gamma, sigmas, vmax_axis)
                          + (whitened_dirs.sum(0)[:, 0]) ** 2),
        "r_band": grad_band, "v_band": weight_band, "g_band": kernel_grad_band,
        "r_energy": grad_energy.mean(0), "v_energy": weight_energy.mean(0),
        "g_energy": kernel_grad_energy.mean(0),
    }
    if n_branches == 1:
        direction = whitened_dirs[0]
        record["out_align"] = direction[:, 0].abs()
        record["lam_plus"] = _dot(direction, eigenvalues * direction)
        record["sign_product"] = direction[:, 0] * whitened_grad[:, 0]
        return record

    # the mode picture, for any number of branches: the separation modes, their
    # energies, and how each lies on v_max, on the runner-up and in the loud
    # plane the two span; the mean direction's alignment; the preconditioner
    # along the runner-up; and every pairwise angle
    modes, mode_energy, mean_unit, mean_norm = separation_modes(whitened_dirs)
    next_axis = torch.zeros(input_dim, dtype=DTYPE)
    next_axis[1] = 1.0
    record.update({
        "mode_energy": mode_energy,                                    # (n_channels, n_branches − 1)
        "mode_align_vmax": modes[:, :, 0].abs(),
        "mode_align_next": modes[:, :, 1].abs(),
        "mode_loud_plane": modes[:, :, 0] ** 2 + modes[:, :, 1] ** 2,
        "mean_align_vmax": mean_unit[:, 0].abs(), "mean_norm": mean_norm,
        "jjt_next": _quadratic_form_kernel(whitened_dirs, eigenvalues, gamma, sigmas, next_axis),
        "pair_angles": pairwise_angles(whitened_dirs),
    })
    if n_branches > 2:
        # a block with more than two branches has no single angle or separation
        # axis; "theta" is the median pairwise angle and "align" the first
        # separation mode's alignment with v_max, so the open and parked masks
        # of the summaries keep their meaning
        theta = record["pair_angles"].median(-1).values
        record.update({"theta": theta, "cos": torch.cos(theta),
                       "align": modes[:, 0, 0].abs(), "out_align": mean_unit[:, 0].abs(),
                       "sigma2": sigmas[1].clone(), "wnorm2": kernels[1].norm(dim=-1)})
        return record

    dir1, dir2 = whitened_dirs[0], whitened_dirs[1]
    cos_theta = _dot(dir1, dir2).clamp(-1.0, 1.0)
    theta = torch.arccos(cos_theta)
    half_diff, half_sum = 0.5 * (dir1 - dir2), 0.5 * (dir1 + dir2)
    sin_half = half_diff.norm(dim=-1).clamp_min(1e-300)                      # sin(theta/2)
    cos_half = half_sum.norm(dim=-1).clamp_min(1e-300)                       # cos(theta/2)
    sep_axis = half_diff / sin_half.unsqueeze(-1)                            # the unit separation axis
    bisector = half_sum / cos_half.unsqueeze(-1)                             # the unit bisector
    lam_plus = _dot(bisector, eigenvalues * bisector)
    lam_minus = _dot(sep_axis, eigenvalues * sep_axis)
    cos2_half = 0.5 * (1.0 + cos_theta)
    gap = lam_minus - lam_plus
    stab = torch.where(gap > 0, lam_plus / gap.clamp_min(1e-300),
                       torch.full_like(gap, float("nan")))

    # exact flow rates of the angle and of the alignment
    dir_flow = _tangent_flow(whitened_dirs, eigenvalues, gamma, sigmas, whitened_grad)   # (2, n_channels, input_dim)
    dcos_theta = _dot(dir_flow[0], dir2) + _dot(dir1, dir_flow[1])
    sin_theta = torch.sin(theta).clamp_min(1e-12)
    dtheta = -dcos_theta / sin_theta
    half_diff_flow = 0.5 * (dir_flow[0] - dir_flow[1])
    sep_axis_flow = (half_diff_flow - sep_axis * _dot(sep_axis, half_diff_flow).unsqueeze(-1)
                     ) / sin_half.unsqueeze(-1)
    dalign = torch.sign(sep_axis[:, 0]) * sep_axis_flow[:, 0]

    # the two terms of the angle equation (43), the γ-rate term and the
    # loud-product term K. Equation (43) takes σ₁ = σ₂; away from that, 1/σ² is
    # the mean of 1/σ₁² and 1/σ₂², the equal-σ part of the exact split of
    # Appendix A.13 and the one definition the CNN measurement module
    # (blockmeasure, "drive") uses. Before 2026-09-06 the rig used the mean of
    # σ² instead. The two agree at σ₁ = σ₂, which the CNN nearly satisfies
    # (ratio 1.07 to 1.15 once aligned), but the rig's branch σ ratio runs at
    # 1.3 to 2.3 in the median over training, so here the choice moves the
    # terms by tens of percent. Runs saved before this date carry the old
    # convention in `rec_term_gamma` and `rec_term_K`.
    inv_sigma_sq = (1.0 / sigmas ** 2).mean(0)
    tan_half = torch.tan(0.5 * theta)
    dgamma = record["dgamma"]
    K = _dot(bisector, eigenvalues * (whitened_grad - bisector * _dot(bisector, whitened_grad).unsqueeze(-1)))
    term_gamma = (2 * gamma * inv_sigma_sq) * tan_half * (
        -0.5 * dgamma * (lam_plus * (1 - cos2_half) + lam_minus * cos2_half))
    term_K = (2 * gamma * inv_sigma_sq) * tan_half * torch.sqrt(cos2_half.clamp(0, 1)) * K

    record.update({
        "theta": theta, "cos": cos_theta,
        "align": sep_axis[:, 0].abs(), "out_align": bisector[:, 0].abs(),
        "lam_plus": lam_plus, "lam_minus": lam_minus,
        "stab": stab, "cos2_half": cos2_half,
        "sigma2": sigmas[1].clone(), "wnorm2": kernels[1].norm(dim=-1),
        "dtheta": dtheta, "dalign": dalign,
        "term_gamma": term_gamma, "term_K": term_K, "K": K,
        "pm_r": _dot(sep_axis, whitened_grad), "pp_r": _dot(bisector, whitened_grad),
        "sign_product": bisector[:, 0] * whitened_grad[:, 0],
        "jjt_sep": _quadratic_form_kernel(whitened_dirs, eigenvalues, gamma, sigmas, sep_axis),
        "dp_flow": dir_flow,     # (2, n_channels, input_dim), kept for the finite-difference test
    })
    return record


@torch.no_grad()
def angle_and_alignment(kernels, eigenvalues):
    """The whitened angle and the alignment of the separation axis with v_max
    alone, for the steps that do not carry the full measurement. For more than
    two branches: the median pairwise angle and the first separation mode's
    alignment, as in `measure`."""
    whitened_dirs = whitened_directions(kernels, eigenvalues)
    if whitened_dirs.shape[0] > 2:
        modes, _, _, _ = separation_modes(whitened_dirs)
        return pairwise_angles(whitened_dirs).median(-1).values, modes[:, 0, 0].abs()
    cos_theta = _dot(whitened_dirs[0], whitened_dirs[1]).clamp(-1.0, 1.0)
    diff = whitened_dirs[0] - whitened_dirs[1]
    align = diff[:, 0].abs() / diff.norm(dim=-1).clamp_min(1e-300)
    return torch.arccos(cos_theta), align


def open_mask(record: dict) -> torch.Tensor:
    return record["theta"] >= math.radians(OPEN_DEG)


# ----------------------------------------------------------------------------
# 6. training
# ----------------------------------------------------------------------------

def train(arm: str, eigenvalues: torch.Tensor, inputs: torch.Tensor, targets: torch.Tensor,
          n_channels: int, seed: int, steps: int = 10000, lr: float = 0.1,
          momentum: float = 0.9, wd: float = 5e-4, every: int = 10, record_first: int = 500,
          cool_factor: float | None = None, init: str = "raw",
          curvature: bool = True, band: int = 2000, cool_dims: int = 1) -> dict:
    """Train one arm at one seed: full-batch heavy-ball SGD with kernel-only
    weight decay for `steps` steps. Returns the run record as a dictionary.

    `arm` is "single", "pair", "quad" (four branches, one shared γ and β, the
    same construction as the pair), "cooled" or "boosted". The cooled arm is the
    single with the first `cool_dims` coordinates of its kernel gradient (the
    loud subspace, coordinate 0 alone for a unique v_max) multiplied by
    `cool_factor` before the optimizer step; the boosted arm is the single
    with gamma at four times and beta at twice the learning rate. `seed`
    draws the student's initialization only; the data and the teacher come in
    as `inputs` and `targets`. `init` is "raw" or "whitened" (see
    `init_params`). `every` and `record_first` set which steps carry the full
    measurement; `band` is the number of final steps the summaries average
    over and is only stored here.

    The record holds the loss at every step ("loss"; the plain mean-squared
    error, since weight decay acts only through the optimizer), for the pair
    the whitened angle and the alignment of every channel at every step
    ("theta_all", "align_all"), the full per-channel measurement at every one
    of the first `record_first` steps and then every `every` steps
    ("rec_steps" and one "rec_<key>" array per key of `measure`, each of shape
    (recorded steps, n_channels, ...)), the final parameters under the
    historical keys "W" (kernels), "gamma", "beta" and "V" (readout), and the
    end-of-run curvature ("hess_top", "edge", "hess_diag_kernel"). run.py adds
    "lam" for the eigenvalues and the cell's identity before saving.
    """
    if arm == "cooled" and cool_factor is None:
        raise ValueError("the cooled arm needs cool_factor")
    n_outputs = targets.shape[1]
    generator = torch.Generator().manual_seed(seed)
    kernels, gamma, beta, readout = init_params(arm, eigenvalues, n_channels, n_outputs,
                                                generator, init=init)
    # The boosted arm is the single with the learning rate of gamma times 4 and
    # of beta times 2, the closed-pair reproduction of the paper's section 4.1.
    boost_gamma, boost_beta = (4.0, 2.0) if arm == "boosted" else (1.0, 1.0)
    optimizer = torch.optim.SGD(
        [{"params": [kernels], "weight_decay": wd},
         {"params": [readout], "weight_decay": 0.0},
         {"params": [gamma], "weight_decay": 0.0, "lr": lr * boost_gamma},
         {"params": [beta], "weight_decay": 0.0, "lr": lr * boost_beta}],
        lr=lr, momentum=momentum)
    whitened_inputs = inputs / eigenvalues.sqrt()
    loss_hist = np.zeros(steps + 1)
    multi = kernels.shape[0] >= 2
    theta_all = np.zeros((steps + 1, n_channels), np.float32) if multi else None
    align_all = np.zeros((steps + 1, n_channels), np.float32) if multi else None
    rec_steps, record_lists = [], {}

    def is_record_step(step):
        return step % every == 0 or step < record_first or step == steps

    for step in range(steps + 1):
        optimizer.zero_grad(set_to_none=True)
        output, preactivation = forward(inputs, eigenvalues, kernels, gamma, beta, readout)
        loss = loss_fn(output, targets)
        loss.backward()
        loss_hist[step] = float(loss.detach())
        if arm == "cooled":
            kernels.grad[..., :cool_dims] *= cool_factor
        if is_record_step(step):
            with torch.no_grad():
                whitened_grad = preactivation.grad.T @ whitened_inputs   # (n_channels, input_dim)
                point = measure(kernels, gamma, beta, readout, eigenvalues, whitened_grad)
            rec_steps.append(step)
            for key, val in point.items():
                if key == "dp_flow":
                    continue
                record_lists.setdefault(key, []).append(val.to(torch.float32).numpy())
            if theta_all is not None:
                theta_all[step] = point["theta"].numpy()
                align_all[step] = point["align"].numpy()
        elif theta_all is not None:
            with torch.no_grad():
                theta, align = angle_and_alignment(kernels, eigenvalues)
            theta_all[step] = theta.numpy()
            align_all[step] = align.numpy()
        optimizer.step()

    result = {
        "arm": arm, "seed": seed, "steps": steps, "lr": lr, "momentum": momentum,
        "wd": wd, "band": band, "init": init, "every": every,
        "record_first": record_first,
        "cool_factor": float("nan") if cool_factor is None else cool_factor,
        "cool_dims": cool_dims,
        "loss": loss_hist, "rec_steps": np.array(rec_steps),
        "W": kernels.detach().numpy(), "gamma": gamma.detach().numpy(),
        "beta": beta.detach().numpy(), "V": readout.detach().numpy(),
    }
    for key, val in record_lists.items():
        result["rec_" + key] = np.stack(val)
    if theta_all is not None:
        result["theta_all"], result["align_all"] = theta_all, align_all
    if curvature:
        result.update(end_curvature(inputs, targets, eigenvalues, kernels.detach(),
                                    gamma.detach(), beta.detach(), readout.detach(),
                                    lr, momentum))
    return result


# ----------------------------------------------------------------------------
# 7. end-of-run curvature
# ----------------------------------------------------------------------------

def _flat_loss(inputs, targets, eigenvalues, shapes):
    """The loss as a function of one flat parameter vector (kernels, gamma,
    beta, readout concatenated), for Hessian-vector products."""
    sizes = [int(np.prod(shape)) for shape in shapes]

    def loss_of_flat(flat_params):
        parts = torch.split(flat_params, sizes)
        kernels, gamma, beta, readout = (part.reshape(shape) for part, shape in zip(parts, shapes))
        n_branches, n_channels, input_dim = kernels.shape
        sigmas = sigma_of(kernels, eigenvalues)
        branch_features = (inputs @ kernels.reshape(n_branches * n_channels, input_dim).T
                           ).reshape(-1, n_branches, n_channels) / sigmas
        preactivation = gamma * branch_features.sum(1) + beta
        return loss_fn(torch.relu(preactivation) @ readout, targets)
    return loss_of_flat


@torch.no_grad()
def end_curvature(inputs, targets, eigenvalues, kernels, gamma, beta, readout,
                  lr, momentum, iters=40):
    """The top eigenvalue of the full Hessian by power iteration (the edge of
    stability sits at 2(1 + momentum)/lr for heavy-ball SGD), and the
    per-channel, per-mode diagonal of the Hessian with respect to the kernel
    coordinates, one Hessian-vector product per (branch, channel, mode)."""
    shapes = [tuple(kernels.shape), tuple(gamma.shape), tuple(beta.shape), tuple(readout.shape)]
    flat_params = torch.cat([t.reshape(-1) for t in (kernels, gamma, beta, readout)]).clone()
    loss_of_flat = _flat_loss(inputs, targets, eigenvalues, shapes)

    def hessian_vector(vector):
        with torch.enable_grad():
            point = flat_params.clone().requires_grad_()
            grad = torch.autograd.grad(loss_of_flat(point), point, create_graph=True)[0]
            return torch.autograd.grad(grad, point, grad_outputs=vector)[0]

    generator = torch.Generator().manual_seed(0)
    vector = torch.randn(flat_params.numel(), generator=generator, dtype=DTYPE)
    vector = vector / vector.norm()
    top = float("nan")
    for _ in range(iters):
        product = hessian_vector(vector)
        top = float(vector @ product)
        vector = product / product.norm().clamp_min(1e-300)

    diag = torch.zeros(*kernels.shape, dtype=DTYPE)
    for idx in range(kernels.numel()):
        unit = torch.zeros(flat_params.numel(), dtype=DTYPE)
        unit[idx] = 1.0
        diag.view(-1)[idx] = hessian_vector(unit)[idx]
    return {"hess_top": top, "edge": 2 * (1 + momentum) / lr,
            "hess_diag_kernel": diag.numpy()}
