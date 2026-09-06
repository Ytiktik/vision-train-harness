"""Sweep, summary, report and figures for the deterministic-normalizer
synthetic model. The model itself is in rig.py; this file decides which cells
to run, runs them in a worker pool, and reads the saved runs back.

    python -m structural_reparam.agents.synth_fc_iso3.run sweep --out DIR --ratios cifar flat1 --seeds 42 43 44 --teachers w128
    python -m structural_reparam.agents.synth_fc_iso3.run summarize DIR
    python -m structural_reparam.agents.synth_fc_iso3.run report DIR
    python -m structural_reparam.agents.synth_fc_iso3.run figures DIR --fig-dir FIGDIR
    python -m structural_reparam.agents.synth_fc_iso3.run dose DIR
    python -m structural_reparam.agents.synth_fc_iso3.run doses
    python -m structural_reparam.agents.synth_fc_iso3.run teacher-stats

How a sweep runs. A cell is one spectrum name (see `rig.spectrum`; a bare
number x means the CIFAR tail with λ_max at x times λ_next), one teacher spec,
and optionally a v_max share for the teacher (`--vmax-shares`), named like
`cifar_w128` or `cifar_v0.5_w128`; directories written before 2026-09-06 use
the legacy name `cifar_l0_w128` for the first of these. `sweep` lists the
cells, checks that the arms are initialization-matched, and hands
(cell, arm, seed) jobs to a pool of workers. `run_job` builds the cell's data
and teacher with `make_cell` (one draw per spectrum from the fixed data seed,
so seeds vary only the student's initialization, as in the CNN), trains the
arm with `rig.train`, and saves one npz per run at `DIR/<cell>/<arm>_s<seed>.npz`.
The single and pair arms run first; the cooled arm runs in a second phase,
because `cooling_factor` reads its factor from the finished pair and single
of the same cell and seed. `summarize`, `report` and `figures` only read the
saved files.

A teacher spec is `u<width>` for the uniform teacher with the v_max
coordinate zeroed, `w<width>` for the same draw with v_max kept (isotropic in
the whitened frame; the teacher the results use), `n<width>` for a draw
isotropic in the raw coordinates, or `p<width>` for the retired
profile-matched teacher (hidden directions drawn with per-eigendirection
energy from the CNN's measured block-0 kernel profile), with the optional
suffixes `_s<shift>` (profile teachers only; moves that fraction of the
profile's energy on ranks 1 to 7 to ranks 8 to 17), `_f<fraction>` (the
selector fraction of the profile teacher) and `_t<seed>` (draws the teacher
from its own generator instead of continuing the data generator, for the
teacher-variance check). `u128` reproduces the campaign v2 cells exactly.

The `--samples` and `--precision` switches serve the pilot protocol (1,024
samples in single precision for a pilot, 4,096 in double precision for a
recorded sweep). `doses` solves for the ZCA epsilon of a wanted block-seen
eigenvalue ratio and `dose` puts the measured whitening ladder beside the
paper's Table 18; both belong to the unfinished dose-ladder study.

In the summary and figure functions a variable named `z` is one loaded run
file (a numpy npz), read by the keys `rig.train` documents.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import torch

from . import rig

D, C, K, HIDDEN, N = 27, 64, 10, 128, 4096
DATA_SEED = 7


def cell_name(spec, vmax_share, tag="", tail_share=None):
    """`<spec>_<teacher>` for a cell at the teacher's own shares,
    `<spec>_v<share>_<teacher>` with the v_max share set, and
    `<spec>_q<share>_<teacher>` with the slow-tail share set (both tokens
    when both are set, v first). Directories written before 2026-09-06 carry
    the legacy token `_l0` in that position (`cifar_l0_w128`); it means the
    same as no token."""
    v = "" if vmax_share is None else f"_v{vmax_share:g}"
    q = "" if tail_share is None else f"_q{tail_share:g}"
    return f"{spec}{v}{q}{tag}"


def parse_teacher(spec: str) -> dict:
    """'u128' is the uniform teacher of width 128 with the loud coordinate
    zeroed, 'w128' the same draw with the loud coordinate kept, 'n128' the draw
    that is isotropic in the raw coordinates; 'p128' the profile-matched
    teacher of width 128; '_s0.5' shifts half of the profile energy on ranks
    1 to 7 to ranks 8 to 17; '_f0.23' sets the selector fraction; '_t101' draws
    the teacher from its own generator seeded with 101 instead of continuing the
    data generator, so the teacher draw can be varied at a fixed data sample
    (the teacher-variance check of 2026-09-05)."""
    t = dict(kind="u", hidden=HIDDEN, shift=0.0, selector_frac=0.23, spec=spec,
             teacher_seed=None)
    for i, part in enumerate(spec.split("_")):
        if i == 0 and part[0] in "upwn":
            t["kind"], t["hidden"] = part[0], int(part[1:])
        elif part.startswith("s"):
            t["shift"] = float(part[1:])
        elif part.startswith("f"):
            t["selector_frac"] = float(part[1:])
        elif part.startswith("t"):
            t["teacher_seed"] = int(part[1:])
        else:
            raise ValueError(f"unknown teacher spec part {part!r} in {spec!r}")
    return t


def teacher_profiles(t: dict) -> tuple[np.ndarray, np.ndarray]:
    prof = rig.load_profiles()
    return (rig.shift_profile(prof["kernel_selectors"], t["shift"]),
            rig.shift_profile(prof["kernel_others"], t["shift"]))


def make_cell(spec, vmax_share, teacher=None, tail_share=None):
    """The data and the targets of one cell: the spectrum from `rig.spectrum`,
    the sample from `rig.make_data` with the fixed data seed, and the teacher
    drawn from the same generator (or from its own seed when the teacher spec
    carries `_t<seed>`). Returns (eigenvalues, inputs, targets, energy) with
    `energy` the teacher's realized mean whitened energy share per
    eigendirection (a length-27 vector), or None for the uniform teachers."""
    teacher_spec = parse_teacher("u%d" % HIDDEN) if teacher is None else teacher
    eigenvalues = rig.spectrum(spec)
    generator = torch.Generator().manual_seed(DATA_SEED)
    inputs = rig.make_data(eigenvalues, N, generator)
    if teacher_spec.get("teacher_seed") is not None:
        # a separately seeded teacher draw; without it the teacher continues the
        # data generator, which every cell before 2026-09-05 did
        generator = torch.Generator().manual_seed(int(teacher_spec["teacher_seed"]))
    if teacher_spec["kind"] in "uwn":
        mode = {"u": "quiet", "w": "whitened", "n": "raw"}[teacher_spec["kind"]]
        teacher_hidden, teacher_readout = rig.make_teacher(eigenvalues, teacher_spec["hidden"], K,
                                                           vmax_share, generator, mode=mode,
                                                           tail_share=tail_share)
        energy = None
    else:
        selector_energy, other_energy = teacher_profiles(teacher_spec)
        teacher_hidden, teacher_readout, energy = rig.make_teacher_profile(
            eigenvalues, teacher_spec["hidden"], K, vmax_share, generator, selector_energy,
            other_energy, teacher_spec["selector_frac"])
        energy = energy.numpy()
    targets = rig.teacher_targets(inputs, teacher_hidden, teacher_readout)
    return eigenvalues, inputs, targets, energy


def check_init_matching(spec, seeds):
    """The verification gate on initialization matching: at every seed the
    single and the pair share the first branch and the readout, and the pair'seed
    gamma puts it at the single'seed output scale (unit standard deviation of the
    summed whitened feature) at step 0."""
    eigenvalues = rig.spectrum(spec)
    for seed in seeds:
        single_kernels, single_gamma, single_beta, single_readout = rig.init_params("single", eigenvalues, C, K, torch.Generator().manual_seed(seed))
        pair_kernels, pair_gamma, pair_beta, pair_readout = rig.init_params("pair", eigenvalues, C, K, torch.Generator().manual_seed(seed))
        assert torch.equal(single_kernels[0], pair_kernels[0]) and torch.equal(single_readout, pair_readout), f"init not shared at seed {seed}"
        with torch.no_grad():
            scale = pair_gamma * rig.whitened_directions(pair_kernels, eigenvalues).sum(0).norm(dim=-1)
        assert torch.allclose(scale, torch.ones_like(scale), atol=1e-6), f"output scale at seed {seed}"
    print(f"init matching holds at seeds {list(seeds)}", flush=True)


def band_mean(rec_steps, arr, steps, band):
    """The mean over the final `band` steps of a per-recorded-step quantity
    (`arr` has one row per entry of `rec_steps`)."""
    sel = rec_steps > steps - band
    return arr[sel].mean(0)


def train_mean(z, arr):
    """The mean over the whole run of a per-recorded-step quantity with equal
    weight per unit of time: only the steps on the regular `every` grid count,
    so the densely recorded first steps do not dominate. `z` is a loaded run
    file."""
    sel = z["rec_steps"] % int(z["every"]) == 0 if "every" in z else np.ones(len(arr), bool)
    return arr[sel].mean(0)


def cooling_factor(out_dir, cell, seed):
    """The factor the cooled arm applies to the loud subspace of the kernel
    gradient, in the paper's section 4.2.4 convention (adopted 2026-09-06):
    the full coefficient of the preconditioner J Jᵀ along v_max, kernel part
    plus the γ-path term, at the end of training, medianed over all channels,
    in the trained pair over the same quantity in the trained single of the
    same cell and seed (`end_vmax_coefficient`). This is what
    `scripts/analysis/jjt_top.py` measures on the network's epoch-100
    checkpoints, where it gives 0.118.

    Before 2026-09-06 the rig used the kernel part alone, medianed over the
    second half of training and over the pair's open channels; on the same
    run the two conventions can differ tenfold (0.468 against 0.046 on the
    CIFAR cell with teacher draw 101, seed 42), because the single's
    coefficient swings a hundredfold at the stability edge. Runs saved under
    the old convention keep their `cool_factor` value and are not comparable
    with new ones. A ratio of medians is used rather than a median of
    per-channel ratios, because a per-channel ratio has a near-zero
    denominator wherever a single's channel sits on the loud direction.

    Returns (factor, per_channel), with factor None when the single's own
    coefficient along the loud direction is negligible (every channel on
    v_max, so the tangent projector removes almost all of it); the
    intervention is undefined there and the caller skips the run.
    """
    pair = np.load(out_dir / cell / f"pair_s{seed}.npz")
    single = np.load(out_dir / cell / f"single_s{seed}.npz")
    pair_per_channel = end_vmax_coefficient(pair)
    single_per_channel = end_vmax_coefficient(single)
    pair_coef = float(np.median(pair_per_channel))
    single_coef = float(np.median(single_per_channel))
    per_channel = pair_per_channel / np.maximum(single_per_channel, 1e-30)
    if single_coef <= 1e-6 or single_coef <= 1e-3 * float(np.max(single_per_channel)):
        return None, per_channel
    return pair_coef / single_coef, per_channel


def end_vmax_coefficient(z) -> np.ndarray:
    """The full coefficient of the preconditioner J Jᵀ along v_max at the end
    of a saved run, per channel: the kernel part plus the γ-path term (p pᵀ
    for a single, 4 p₊p₊ᵀ for the pair), computed from the saved final
    kernels and γ, which is what `scripts/analysis/jjt_top.py` computes on the
    network's epoch-100 checkpoint."""
    eigenvalues = torch.as_tensor(z["lam"], dtype=rig.DTYPE)
    kernels = torch.as_tensor(z["W"], dtype=rig.DTYPE)
    gamma = torch.as_tensor(z["gamma"], dtype=rig.DTYPE)
    sigmas = rig.sigma_of(kernels, eigenvalues)
    dirs = rig.whitened_directions(kernels, eigenvalues, sigmas)
    vmax_axis = torch.zeros(eigenvalues.numel(), dtype=rig.DTYPE)
    vmax_axis[0] = 1.0
    kernel_part = rig._quadratic_form_kernel(dirs, eigenvalues, gamma, sigmas, vmax_axis)
    gamma_part = dirs.sum(0)[:, 0] ** 2
    return (kernel_part + gamma_part).numpy()


def _worker_init(samples: int = N, precision: str = "float64"):
    """Each worker of the spawn pool re-imports this module, so the sample count
    and the precision of the run are set here rather than inherited."""
    global N
    torch.set_num_threads(1)
    N = samples
    rig.DTYPE = torch.float32 if precision == "float32" else torch.float64


# The recipe fields the cell name does not carry. Two sweeps that differ in any
# of them write the same file name, and `run_job` used to take the existing file
# as a finished run, so a pilot at 1,024 samples in single precision followed by
# a recorded sweep at 4,096 in double precision into the same directory returned
# the pilot's numbers under the recorded sweep's name. The skip now checks them.
RECIPE_KEYS = ("steps", "lr", "momentum", "wd", "init", "n_samples", "dtype")


def _recipe(kw, init, samples, dtype):
    return {"steps": int(kw["steps"]), "lr": float(kw["lr"]),
            "momentum": float(kw["momentum"]), "wd": float(kw["wd"]),
            "init": str(init), "n_samples": int(samples), "dtype": str(dtype)}


def run_job(job):
    out_dir, cell, ratio, vmax_share, arm, seed, kw, teacher, tail_share = job
    path = out_dir / cell / f"{arm}_s{seed}.npz"
    want = _recipe(kw, kw.get("init", "raw"), N, rig.DTYPE)
    if path.exists():
        saved = np.load(path)
        have = {k: (str(saved[k]) if k in ("init", "dtype") else
                    (int(saved[k]) if k in ("steps", "n_samples") else float(saved[k])))
                for k in RECIPE_KEYS if k in saved}
        bad = {k: (have[k], want[k]) for k in have if have[k] != want[k]}
        missing = [k for k in RECIPE_KEYS if k not in saved]
        if bad or missing:
            raise RuntimeError(
                f"{path} exists but was made with a different recipe "
                f"(differs: {bad}; not recorded in the file: {missing}). Write "
                "this sweep to its own directory rather than reusing that run.")
        return f"skip {cell} {arm} s{seed}"
    started = time.time()
    eigenvalues, inputs, targets, energy = make_cell(ratio, vmax_share, teacher, tail_share)
    cool = None
    extra = {}
    if arm == "cooled":
        cool, per_channel = cooling_factor(out_dir, cell, seed)
        if cool is None:
            return (f"skip {cell} cooled s{seed}: the single carries no loud "
                    "component, so the intervention is undefined")
        extra["cool_factor_per_channel"] = per_channel
        n_loud = rig.loud_dims(eigenvalues)
        if n_loud == 0:
            return (f"skip {cell} cooled s{seed}: every eigenvalue is within the "
                    "loud tolerance of the top, so there is no loud subspace to "
                    "cool and the intervention is undefined")
        kw = dict(kw, cool_dims=n_loud)
    result = rig.train(arm, eigenvalues, inputs, targets, C, seed, cool_factor=cool, **kw)
    result.update(extra)
    result.update(want)
    result.update({"cell": cell, "spec": ratio,
                   "vmax_share": float("nan") if vmax_share is None else float(vmax_share),
                   "tail_share": float("nan") if tail_share is None else float(tail_share),
                   "lam": eigenvalues.numpy(), "C": C,   # "lam" is the saved key
                "teacher_spec": teacher["spec"], "teacher_kind": teacher["kind"],
                "teacher_hidden": teacher["hidden"], "teacher_shift": teacher["shift"],
                "teacher_energy": np.full(D, np.nan) if energy is None else energy})
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **result)
    return (f"done {cell} {arm} s{seed} in {time.time() - started:.0f}s, "
            f"band loss {result['loss'][-kw['band']:].mean():.4e}")


def sweep(a):
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ratios = [("r" + r) if r.replace(".", "", 1).isdigit() else r for r in a.ratios]
    kw = dict(steps=a.steps, lr=a.lr, momentum=a.momentum, wd=a.wd,
              every=a.every, record_first=a.record_first, band=a.band,
              curvature=not a.no_curvature)
    teachers = [parse_teacher(t) for t in a.teachers]
    shares = a.vmax_shares if a.vmax_shares else [None]
    tails = a.tail_shares if a.tail_shares else [None]
    cells = [(cell_name(r, share, "_" + t["spec"], tail), r, share, kw, "raw", t, tail)
             for t in teachers for r in ratios for share in shares for tail in tails]
    if a.flow:
        kw_flow = dict(kw, steps=a.steps * a.flow_factor, lr=a.lr / a.flow_factor,
                       band=a.band * a.flow_factor, every=a.every * a.flow_factor,
                       record_first=a.record_first * a.flow_factor)
        cells.append((cell_name(ratios[0], shares[0], "_" + teachers[0]["spec"] + "_flow", tails[0]),
                      ratios[0], shares[0], kw_flow, "raw", teachers[0], tails[0]))
    if a.iso:
        cells.append((cell_name(ratios[0], shares[0], "_" + teachers[0]["spec"] + "_iso", tails[0]),
                      ratios[0], shares[0], dict(kw, init="whitened"), "whitened", teachers[0], tails[0]))
    (out_dir / "sweep_args.json").write_text(json.dumps(
        {k: v for k, v in vars(a).items() if k != "func"}, indent=2))
    _worker_init(a.samples, a.precision)          # the parent too, for the gate below
    check_init_matching(ratios[0], a.seeds)
    print(f"{a.samples} samples, {a.precision}", flush=True)

    def jobs(arms):
        return [(out_dir, cell, r, l, arm, s, dict(kw_c, init=init), t, tail)
                for cell, r, l, kw_c, init, t, tail in cells for s in a.seeds for arm in arms]

    phases = [[arm for arm in a.arms if arm != "cooled"]]
    if "cooled" in a.arms:
        phases.append(["cooled"])
    with mp.get_context("spawn").Pool(a.workers, initializer=_worker_init,
                                      initargs=(a.samples, a.precision)) as pool:
        for arms in phases:
            js = jobs(arms)
            if arms == ["cooled"] and a.cooled_ratios:
                js = [j for j in js if j[2] in a.cooled_ratios]
            for msg in pool.imap_unordered(run_job, js):
                print(msg, flush=True)


# ----------------------------------------------------------------------------
# summary
# ----------------------------------------------------------------------------

def summarize_run(path: Path) -> dict:
    """One row per saved run: the band-mean and final loss, the top Hessian
    eigenvalue over the stability edge, the cooling factor, the band-mean
    energy shares by eigenvalue band, and for the pair its band-averaged
    geometry (open and parked counts, median angle and alignment over open
    channels, and the angle in stability units)."""
    z = np.load(path)
    steps, band = int(z["steps"]), int(z["band"])
    row = {"cell": str(z["cell"]), "arm": str(z["arm"]), "seed": int(z["seed"]),
           "loss_band": float(z["loss"][-band:].mean()),
           "loss_end": float(z["loss"][-1]),
           "hess_top_over_edge": float(z["hess_top"] / z["edge"])
           if "hess_top" in z else float("nan"),
           "cool_factor": float(z["cool_factor"])}
    rs = z["rec_steps"]
    for i, name in enumerate(rig.BAND_NAMES):
        # the mean over channels of the per-channel band share, averaged over
        # the recorded steps of the final band, for the demand (r) and the
        # effective whitened weight (v)
        row[f"r_{name}_band"] = float(band_mean(rs, z["rec_r_band"][:, :, i].mean(1), steps, band))
        row[f"v_{name}_band"] = float(band_mean(rs, z["rec_v_band"][:, :, i].mean(1), steps, band))
        row[f"r_{name}_train"] = float(train_mean(z, z["rec_r_band"][:, :, i].mean(1)))
        if "rec_g_band" in z:
            row[f"g_{name}_train"] = float(train_mean(z, z["rec_g_band"][:, :, i].mean(1)))
            row[f"g_{name}_band"] = float(band_mean(rs, z["rec_g_band"][:, :, i].mean(1), steps, band))
    if str(z["arm"]) in ("pair", "quad"):
        theta = np.degrees(band_mean(rs, z["rec_theta"], steps, band))
        align = band_mean(rs, z["rec_align"], steps, band)
        is_open = theta >= rig.OPEN_DEG
        row.update({
            "open": int(is_open.sum()),
            "parked": int((is_open & (align >= rig.PARK_ALIGN)).sum()),
            "theta_open_med": float(np.median(theta[is_open])) if is_open.any() else float("nan"),
            "theta_all_med": float(np.median(theta)),
            "align_open_med": float(np.median(align[is_open])) if is_open.any() else float("nan"),
        })
        if "rec_stab" in z:
            stab_units = band_mean(rs, z["rec_cos2_half"] / z["rec_stab"], steps, band)
            row.update({
                "stab_units_open_med": float(np.nanmedian(stab_units[is_open])) if is_open.any() else float("nan"),
                "lam_plus_open_med": float(np.median(band_mean(rs, z["rec_lam_plus"], steps, band)[is_open])) if is_open.any() else float("nan"),
                "lam_minus_open_med": float(np.median(band_mean(rs, z["rec_lam_minus"], steps, band)[is_open])) if is_open.any() else float("nan"),
            })
        if "rec_mode_energy" in z:
            # the mode picture, band-averaged per channel then medianed over open channels:
            # each separation mode's share of the spread, its alignment with v_max, with
            # the runner-up, and its energy in the loud plane; the mean direction's
            # alignment with v_max; and the preconditioner along the runner-up
            def med_open(arr):
                a = band_mean(rs, arr, steps, band)
                return [float(np.median(a[is_open][:, k])) if is_open.any() else float("nan") for k in range(a.shape[1])]
            row.update({
                "mode_energy": med_open(z["rec_mode_energy"]),
                "mode_align_vmax": med_open(z["rec_mode_align_vmax"]),
                "mode_align_next": med_open(z["rec_mode_align_next"]),
                "mode_loud_plane": med_open(z["rec_mode_loud_plane"]),
                "mean_align_vmax": float(np.median(band_mean(rs, z["rec_mean_align_vmax"], steps, band)[is_open])) if is_open.any() else float("nan"),
                "jjt_next_med": float(np.median(band_mean(rs, z["rec_jjt_next"], steps, band))),
                "jjt_vmax_med": float(np.median(band_mean(rs, z["rec_jjt_vmax"], steps, band))),
            })
    return row


def summarize(a):
    out_dir = Path(a.out)
    rows = [summarize_run(p) for p in sorted(out_dir.glob("*/*.npz"))]
    with open(out_dir / "summary.csv", "w", newline="") as fh:
        keys = sorted({k for r in rows for k in r})
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    by = {}
    for r in rows:
        by.setdefault(r["cell"], {}).setdefault(r["seed"], {})[r["arm"]] = r

    def mean_se(vals):
        v = np.array([x for x in vals if not math.isnan(x)])
        if v.size == 0:
            return "   n/a      "
        se = v.std(ddof=1) / math.sqrt(v.size) if v.size > 1 else float("nan")
        return f"{v.mean():+8.4f} ± {se:6.4f}"

    print("Per cell, mean ± standard error over seeds. Loss is the band-mean "
          "training MSE; 'pair-single' and 'cooled-single' are paired per seed "
          "as the log10 ratio of the arm's band loss to the single's (negative "
          "means the arm is lower); 'boosted' is the single with gamma at 4x and "
          "beta at 2x the learning rate. Geometry is the pair's, band-averaged per "
          "channel: open count of 64 (whitened angle >= 10 deg), parked count "
          "(open and |p_minus . v_max| >= 0.95), median angle over open "
          "channels in degrees, median alignment over open channels, and the "
          "median over open channels of cos^2(theta/2) in units of the "
          "stability condition lambda_plus/(lambda_minus - lambda_plus).")
    hdr = (f"{'cell':16s} {'n':>2s} {'log10 pair/single':>18s} {'log10 cooled/single':>20s} {'log10 boosted/single':>20s} "
           f"{'open':>6s} {'parked':>7s} {'angle open':>11s} {'align open':>11s} "
           f"{'stab units':>11s} {'cool factor':>12s} {'single edge':>12s} {'pair edge':>12s}"
           f"{'log10 quad/single':>19s} {'quad parked':>12s} {'quad angle':>11s}")
    print(hdr)
    quad_rows = []
    for cell, seeds in sorted(by.items()):
        d_pair, d_cool, opens, parks, ths, als, sts, cfs, e_s, e_p = ([] for _ in range(10))
        d_boost, d_quad, q_parks, q_ths = [], [], [], []
        for seed, arms in sorted(seeds.items()):
            s = arms.get("single")
            if s is None:
                continue
            e_s.append(s["hess_top_over_edge"])
            if "pair" in arms:
                p = arms["pair"]
                d_pair.append(math.log10(p["loss_band"] / s["loss_band"]))
                opens.append(p["open"]); parks.append(p["parked"])
                ths.append(p["theta_open_med"]); als.append(p["align_open_med"])
                sts.append(p["stab_units_open_med"]); e_p.append(p["hess_top_over_edge"])
            if "cooled" in arms:
                c = arms["cooled"]
                d_cool.append(math.log10(c["loss_band"] / s["loss_band"]))
                cfs.append(c["cool_factor"])
            if "boosted" in arms:
                d_boost.append(math.log10(arms["boosted"]["loss_band"] / s["loss_band"]))
            if "quad" in arms:
                q = arms["quad"]
                d_quad.append(math.log10(q["loss_band"] / s["loss_band"]))
                q_parks.append(q["parked"]); q_ths.append(q["theta_open_med"])
                quad_rows.append((cell, seed, q))
        print(f"{cell:16s} {len(seeds):2d} {mean_se(d_pair)} {mean_se(d_cool):>20s} {mean_se(d_boost):>20s} "
              f"{np.mean(opens) if opens else float('nan'):6.1f} "
              f"{np.mean(parks) if parks else float('nan'):7.1f} "
              f"{np.nanmean(ths) if ths else float('nan'):11.1f} "
              f"{np.nanmean(als) if als else float('nan'):11.3f} "
              f"{np.nanmean(sts) if sts else float('nan'):11.2f} "
              f"{np.mean(cfs) if cfs else float('nan'):12.3f} "
              f"{np.nanmean(e_s) if e_s else float('nan'):12.2f} "
              f"{np.nanmean(e_p) if e_p else float('nan'):12.2f}"
              f"{mean_se(d_quad):>19s} {np.mean(q_parks) if q_parks else float('nan'):12.1f} "
              f"{np.nanmean(q_ths) if q_ths else float('nan'):11.1f}")
    if quad_rows:
        print()
        print("The mode picture of the four-branch block, per cell, mean over seeds of the per-run medians over open channels: "
              "for each of the three separation modes (the principal axes of the four branch directions' spread, ordered by "
              "the share of the spread they carry) its share, its alignment with v_max, with the runner-up, and its energy in "
              "the loud plane the two span; then the mean direction's alignment with v_max, and the block's preconditioner "
              "along v_max and along the runner-up (medians over all channels). The pair's values follow for comparison "
              "(one mode, the separation axis).")
        for arm_name in ("quad", "pair"):
            for cell in sorted({c for c, _, _ in quad_rows}):
                rows = [r for c, _, r in quad_rows if c == cell] if arm_name == "quad" else \
                       [arms["pair"] for c, seeds in by.items() if c == cell for arms in seeds.values() if "pair" in arms and "mode_energy" in arms["pair"]]
                if not rows:
                    continue
                k = len(rows[0]["mode_energy"])
                mean_k = lambda key: [float(np.nanmean([r[key][i] for r in rows])) for i in range(k)]
                fmt = lambda v: " ".join(f"{x:5.2f}" for x in v)
                print(f"  {arm_name:5s} {cell:16s} share {fmt(mean_k('mode_energy'))} | on v_max {fmt(mean_k('mode_align_vmax'))} | "
                      f"on runner-up {fmt(mean_k('mode_align_next'))} | in loud plane {fmt(mean_k('mode_loud_plane'))} | "
                      f"mean dir on v_max {np.nanmean([r['mean_align_vmax'] for r in rows]):5.2f} | "
                      f"jjt v_max {np.nanmean([r['jjt_vmax_med'] for r in rows]):7.2f} runner-up {np.nanmean([r['jjt_next_med'] for r in rows]):7.2f}")


# ----------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------

ARM_COLOR = {"single": "#009E73", "pair": "#0072B2", "cooled": "#D55E00",
             "boosted": "#CC79A7"}
CELL_ORDER = ["cifar", "r3.3", "r1.7", "r1"]
_RATIO_TITLE = {"cifar": "λ_max/λ_next = 8.4 (CIFAR-100 spectrum)", "r3.3": "λ_max/λ_next = 3.3",
                "r1.7": "λ_max/λ_next = 1.7", "r1": "λ_max/λ_next = 1.0 (tie)"}


def mu_eff(eigenvalues):
    """The paper's spectral summary (§3.6, equation 64): the loud direction's
    margin over its runner-up as a share of the input variance the block sees,
    (lam_max - lam_next) / trace. One for a rank-one input, zero both for
    isotropic data and for a degenerate top pair."""
    eigenvalues = torch.sort(torch.as_tensor(eigenvalues), descending=True).values
    return float((eigenvalues[0] - eigenvalues[1]) / eigenvalues.sum())


def _spec_title(base):
    try:
        eigenvalues = rig.spectrum(base).numpy()
    except ValueError:
        return _RATIO_TITLE.get(base, base)
    name = {"cifar": "CIFAR-100 spectrum", "flat1": "isotropic",
            "r1": "tie (top two equal)"}.get(base, base)
    return (f"{name}: mu_eff = {mu_eff(eigenvalues):.3f}, "
            f"λ_max/λ_next = {eigenvalues[0] / eigenvalues[1]:.2f}")
_TAG_TITLE = {"flow": ", learning rate 0.02 for 50k steps", "iso": ", whitened-isotropic init"}


def _teacher_title(tokens):
    """Readable teacher description from the cell's tag tokens ('p128', 's0.5')."""
    parts = []
    for tok in tokens:
        if tok in _TAG_TITLE:
            parts.append(_TAG_TITLE[tok])
        elif tok.startswith("u"):
            parts.append(f", uniform teacher of width {tok[1:]}")
        elif tok.startswith("w"):
            parts.append(f", whitened-isotropic teacher of width {tok[1:]}")
        elif tok.startswith("n"):
            parts.append(f", raw-isotropic teacher of width {tok[1:]}")
        elif tok.startswith("p"):
            parts.append(f", profile teacher of width {tok[1:]}")
        elif tok.startswith("s"):
            parts.append(f", tail shift {tok[1:]}")
        elif tok.startswith("f"):
            parts.append(f", selector fraction {tok[1:]}")
        elif tok.startswith("t"):
            parts.append(f", teacher draw {tok[1:]}")
        else:
            parts.append(", " + tok)
    return "".join(parts)


def split_cell(cell):
    """Split a cell name into (spectrum name, v_max share text or None,
    slow-tail share text or None, tag tokens). The v_max token is `v<x>` in
    names written from 2026-09-06 on and `l<x>` (the retired loud-share dial)
    in older directories, the tail token is `q<x>`; a name without them is the
    teacher's own shares. 'cifar_v0.5_w128' gives ('cifar', '0.5', None,
    ['w128']), 'cifar_q0.9_w128' gives ('cifar', None, '0.9', ['w128']) and
    'cifar_l0_w128' gives ('cifar', None, None, ['w128'])."""
    base, *tokens = cell.split("_")
    share = tail = None
    if tokens and _SHARE_TOKEN.match(tokens[0]):
        token = tokens.pop(0)
        share = None if token == "l0" else token[1:]
    if tokens and _TAIL_TOKEN.match(tokens[0]):
        tail = tokens.pop(0)[1:]
    return base, share, tail, tokens


_SHARE_TOKEN = re.compile(r"^[lv][0-9.]+$")
_TAIL_TOKEN = re.compile(r"^q[0-9.]+$")


class _CellTitle(dict):
    """Readable title for a cell name such as 'cifar_w128', 'cifar_v0.5_w128',
    'cifar_w128_flow', or the legacy 'flat1_l0_p128_s0.5'."""

    def get(self, cell, default=None):
        base, share, tail, tokens = split_cell(cell)
        share_text = "" if share is None else f", v_max share {share}"
        tail_text = "" if tail is None else f", slow-tail share {tail}"
        return f"{_spec_title(base)}{share_text}{tail_text}{_teacher_title(tokens)}"


CELL_TITLE = _CellTitle()


def _load_cell(out_dir, cell):
    runs = {}
    for p in sorted((out_dir / cell).glob("*.npz")):
        z = np.load(p)
        runs.setdefault(str(z["arm"]), {})[int(z["seed"])] = z
    return runs


def _rolling_median(x, w):
    return np.array([np.median(x[max(0, i - w):i + 1]) for i in range(len(x))])


def fig_loss(out_dir, fig_dir, cell, name):
    """Training loss against step for every arm of one cell, one figure."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    runs = _load_cell(out_dir, cell)
    for arm in ("single", "pair", "cooled", "boosted"):
        for j, (seed, z) in enumerate(sorted(runs.get(arm, {}).items())):
            L = z["loss"]
            ax.plot(L, color=ARM_COLOR[arm], lw=0.3, alpha=0.15)
            ax.plot(_rolling_median(L, 200), color=ARM_COLOR[arm], lw=1.8 if j == 0 else 0.9,
                    label=arm if j == 0 else None)
    ax.set_yscale("log"); ax.set_xlabel("full-batch step")
    ax.set_ylabel("training MSE (target variance 1)")
    ax.set_title(CELL_TITLE.get(cell, cell), fontsize=11)
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / name, dpi=130); plt.close(fig)


def fig_geometry(out_dir, fig_dir, cell, name):
    """Median whitened angle and median alignment over all channels against
    step for one cell, seeds pooled, with the 25th to 75th percentile band."""
    import matplotlib.pyplot as plt
    runs = _load_cell(out_dir, cell).get("pair", {})
    if not runs:
        return
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 7.5), sharex=True)
    TA = np.concatenate([np.degrees(z["theta_all"]) for z in runs.values()], axis=1)
    AA = np.concatenate([z["align_all"] for z in runs.values()], axis=1)
    steps = np.arange(TA.shape[0])
    for ax, arr, lab, thr in ((axes[0], TA, "whitened branch angle (deg)", 10.0),
                              (axes[1], AA, "|p_minus · v_max|", 0.95)):
        q25, q50, q75 = np.percentile(arr, [25, 50, 75], axis=1)
        ax.fill_between(steps, q25, q75, color="#0072B2", alpha=0.2)
        ax.plot(steps, q50, color="#0072B2", lw=1.6)
        ax.axhline(thr, color="k", lw=0.7, ls=":")
        ax.set_xscale("symlog", linthresh=100); ax.grid(alpha=0.3)
        ax.set_ylabel(lab)
    axes[0].set_title(CELL_TITLE.get(cell, cell) + f" — pair, {TA.shape[1]} channels pooled",
                      fontsize=11)
    axes[1].set_xlabel("full-batch step (symlog)")
    fig.tight_layout()
    fig.savefig(fig_dir / name, dpi=130); plt.close(fig)


def fig_stability(out_dir, fig_dir, cells, name):
    """Per open channel, the band-averaged angle against lambda_minus /
    lambda_plus with the stability boundary of the paper's section 3.4."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 6.5))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(cells)))
    for cell, colr in zip(cells, colors):
        runs = _load_cell(out_dir, cell).get("pair", {})
        xs, ys = [], []
        for z in runs.values():
            steps, band = int(z["steps"]), int(z["band"])
            th = np.degrees(band_mean(z["rec_steps"], z["rec_theta"], steps, band))
            lp = band_mean(z["rec_steps"], z["rec_lam_plus"], steps, band)
            lm = band_mean(z["rec_steps"], z["rec_lam_minus"], steps, band)
            op = th >= rig.OPEN_DEG
            xs.append((lm / lp)[op]); ys.append(th[op])
        if not xs:
            continue
        x, y = np.concatenate(xs), np.concatenate(ys)
        ax.scatter(x, y, s=9, alpha=0.5, color=colr, label=f"{cell} ({len(x)} open channels)")
    rr = np.linspace(2.02, 400, 500)
    ax.plot(rr, np.degrees(2 * np.arccos(np.sqrt(1 / (rr - 1)))), "k-", lw=1.2,
            label="stability boundary cos²(θ/2) = λ₊/(λ₋−λ₊)")
    rr2 = np.linspace(3.02, 400, 500)
    ax.plot(rr2, np.degrees(2 * np.arccos(np.sqrt(2 / (rr2 - 1)))), "k--", lw=1.0,
            label="band edge cos²(θ/2) = 2λ₊/(λ₋−λ₊)")
    ax.set_xscale("log"); ax.set_xlabel("λ₋ / λ₊ (band-averaged, per channel)")
    ax.set_ylabel("whitened branch angle (deg)"); ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(fig_dir / name, dpi=120); plt.close(fig)


BAND_COLORS = {"vmax": "#D55E00", "next5": "#0072B2", "rest": "#009E73", "slow": "#CC79A7"}
BAND_LABELS = {"vmax": "v_max (rank 0)", "next5": "next five (ranks 1 to 5)",
               "rest": "rest (ranks 6 to 26)", "slow": "slow tail (ranks 8 to 26)"}


def fig_bands(out_dir, fig_dir, cell, name):
    """The single's whitened energy by eigendirection band over training, seeds
    pooled (mean over channels, then over seeds): top, the share of the loss
    gradient with respect to the whitened effective weight (the demand), with
    the CNN's measured training-average demand of agent synth_fc_teacher_a as
    dotted lines of the same colour; bottom, the share of the effective
    whitened weight itself (the kernel energy), with the CNN's trained
    block-0 kernel profile as dotted lines. The teacher's own realized energy
    per band is printed in the title."""
    import matplotlib.pyplot as plt
    runs = _load_cell(out_dir, cell).get("single", {})
    if not runs:
        return
    prof = rig.load_profiles()
    demand = prof["gradient_demand"]
    kernel = 0.23 * prof["kernel_selectors"] + 0.77 * prof["kernel_others"]
    zs = list(runs.values())
    rs = zs[0]["rec_steps"]
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 7.5), sharex=True)
    gkey = "rec_g_band" if "rec_g_band" in zs[0] else "rec_r_band"
    for ax, key, ref, lab in ((axes[0], gkey, demand, "share of the whitened branch-gradient energy (demand)"),
                              (axes[1], "rec_v_band", kernel, "share of the effective whitened weight's energy")):
        for i, band in enumerate(rig.BAND_NAMES):
            y = np.mean([z[key][:, :, i].mean(1) for z in zs], axis=0)
            ax.plot(rs, y, color=BAND_COLORS[band], lw=1.5, label=BAND_LABELS[band])
            lo, hi = rig.BANDS[band]
            ax.axhline(ref[lo:hi].sum(), color=BAND_COLORS[band], lw=1.0, ls=":")
        ax.set_ylabel(lab); ax.set_ylim(0, 1); ax.grid(alpha=0.3)
        ax.set_xscale("symlog", linthresh=100)
    te = zs[0]["teacher_energy"]
    te_txt = "" if np.isnan(te).any() else (
        "; teacher energy " + ", ".join(f"{b} {te[lo:hi].sum():.2f}" for b, (lo, hi) in rig.BANDS.items()))
    axes[0].set_title(CELL_TITLE.get(cell, cell) + f"\nsingle, {len(zs)} seed(s), dotted = CNN block 0{te_txt}",
                      fontsize=9)
    axes[0].legend(fontsize=8, loc="upper right")
    axes[1].set_xlabel("full-batch step (symlog)")
    fig.tight_layout(); fig.savefig(fig_dir / name, dpi=130); plt.close(fig)


def _cell_sort_key(cell):
    """Sort the ladder by the block-seen lam_max/lam_next, steepest first, so
    that the named cells of the first campaigns and the zca doses of this one
    fall in one order."""
    base = split_cell(cell)[0]
    try:
        eigenvalues = rig.spectrum(base)
    except (ValueError, IndexError):
        return (1, 0.0, base)
    eigenvalues = torch.sort(eigenvalues, descending=True).values
    return (0, -float(eigenvalues[0] / eigenvalues[1]), base)


def _cells_by_teacher(present):
    """Group the present cells by teacher tag: {'p128': [cells...]}."""
    groups = {}
    for cell in present:
        tokens = split_cell(cell)[3]
        groups.setdefault("_".join(tokens) or "u128", []).append(cell)
    return groups


def fig_ladder(out_dir, fig_dir, cells, name, title):
    """For one teacher, the spectrum ladder side by side, seeds pooled, one
    colour per cell: the median whitened branch angle over the pair's channels
    (top), the median block gradient norm |r| per channel over recorded steps
    for the single (solid) and the pair (dashed) (middle), and the median
    kernel norm of branch 1 (bottom)."""
    import matplotlib.pyplot as plt
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#000000", "#999999"]
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 10), sharex=True)
    for cell, colr in zip(cells, colors):
        runs = _load_cell(out_dir, cell)
        base = split_cell(cell)[0]
        pairs = runs.get("pair", {})
        if pairs:
            TA = np.concatenate([np.degrees(z["theta_all"]) for z in pairs.values()], axis=1)
            axes[0].plot(np.arange(TA.shape[0]), np.median(TA, axis=1), color=colr, lw=1.5, label=base)
        for arm, ls in (("single", "-"), ("pair", "--")):
            zs = list(runs.get(arm, {}).values())
            if not zs:
                continue
            rs = zs[0]["rec_steps"]
            R = np.concatenate([z["rec_rnorm"] for z in zs], axis=1)
            Wn = np.concatenate([z["rec_wnorm1"] for z in zs], axis=1)
            axes[1].plot(rs, np.median(R, axis=1), color=colr, lw=1.3, ls=ls, label=f"{base} {arm}")
            axes[2].plot(rs, np.median(Wn, axis=1), color=colr, lw=1.3, ls=ls)
    axes[0].axhline(10.0, color="k", lw=0.7, ls=":")
    axes[0].set_ylabel("median whitened branch angle (deg), pair")
    axes[1].set_yscale("log"); axes[1].set_ylabel("median block gradient norm |r| per channel")
    axes[2].set_yscale("log"); axes[2].set_ylabel("median kernel norm |w_1| per channel")
    for ax in axes:
        ax.set_xscale("symlog", linthresh=100); ax.set_xlim(left=0); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, title="spectrum"); axes[1].legend(fontsize=7, ncol=2)
    axes[0].set_title(title, fontsize=11); axes[2].set_xlabel("full-batch step (symlog)")
    fig.tight_layout(); fig.savefig(fig_dir / name, dpi=130); plt.close(fig)


def figures(a):
    import matplotlib
    matplotlib.use("Agg")
    out_dir, fig_dir = Path(a.out), Path(a.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    pre = a.prefix
    present = sorted(p.name for p in out_dir.iterdir() if p.is_dir() and any(p.glob("*.npz")))
    for cell in present:
        fig_loss(out_dir, fig_dir, cell, f"{pre}loss_{cell}.png")
        fig_geometry(out_dir, fig_dir, cell, f"{pre}geometry_{cell}.png")
        fig_bands(out_dir, fig_dir, cell, f"{pre}bands_{cell}.png")
    fig_stability(out_dir, fig_dir, [c for c in present if not c.endswith(("_flow", "_iso"))],
                  f"{pre}stability.png")
    for teacher, cells in _cells_by_teacher(present).items():
        cells = sorted(cells, key=_cell_sort_key)
        fig_ladder(out_dir, fig_dir, cells, f"{pre}ladder_{teacher}.png",
                   f"teacher {teacher}: the spectrum ladder")
    print("figures written to", fig_dir)


# ----------------------------------------------------------------------------
# report: liveness, gamma, the angle trajectory, the energy bands, per cell and arm
# ----------------------------------------------------------------------------

def _band_sel(z):
    return z["rec_steps"] > int(z["steps"]) - int(z["band"])


def _chan_band_median(z, key):
    """Per channel the median over the recorded steps of the final band, then
    the median across channels: one number per run."""
    return float(np.median(np.median(z["rec_" + key][_band_sel(z)], axis=0)))


ANGLE_STEPS = (0, 1000, 2000, 5000, 10000)


def report_run(z) -> dict:
    row = {"cell": str(z["cell"]), "arm": str(z["arm"]), "seed": int(z["seed"]),
           "loss_band": float(z["loss"][-int(z["band"]):].mean()),
           "rnorm": _chan_band_median(z, "rnorm"),
           "wnorm": _chan_band_median(z, "wnorm1"),
           "sigma": _chan_band_median(z, "sigma1"),
           "gamma0": float(np.median(z["rec_gamma"][0])),
           "gamma_end": _chan_band_median(z, "gamma"),
           "hess_over_edge": float(z["hess_top"] / z["edge"]) if "hess_top" in z else float("nan")}
    for i, name in enumerate(rig.BAND_NAMES):
        row[f"r_{name}_train"] = float(train_mean(z, z["rec_r_band"][:, :, i].mean(1)))
        row[f"r_{name}_band"] = float(band_mean(z["rec_steps"], z["rec_r_band"][:, :, i].mean(1),
                                                int(z["steps"]), int(z["band"])))
        row[f"v_{name}_band"] = float(band_mean(z["rec_steps"], z["rec_v_band"][:, :, i].mean(1),
                                                int(z["steps"]), int(z["band"])))
        if "rec_g_band" in z:
            row[f"g_{name}_train"] = float(train_mean(z, z["rec_g_band"][:, :, i].mean(1)))
            row[f"g_{name}_band"] = float(band_mean(z["rec_steps"], z["rec_g_band"][:, :, i].mean(1),
                                                    int(z["steps"]), int(z["band"])))
    if str(z["arm"]) == "pair":
        TA = np.degrees(z["theta_all"])
        for s in ANGLE_STEPS:
            if s < TA.shape[0]:
                row[f"theta_{s}"] = float(np.median(TA[s]))
        lr = float(z["lr"])
        row["dtheta_band"] = float(np.degrees(np.median(np.median(
            z["rec_dtheta"][_band_sel(z)], axis=0))) * lr * 1000)
        theta_b = np.degrees(band_mean(z["rec_steps"], z["rec_theta"], int(z["steps"]), int(z["band"])))
        align_b = band_mean(z["rec_steps"], z["rec_align"], int(z["steps"]), int(z["band"]))
        is_open = theta_b >= rig.OPEN_DEG
        row["open"] = int(is_open.sum())
        row["parked"] = int((is_open & (align_b >= rig.PARK_ALIGN)).sum())
    return row


def report(a):
    out_dir = Path(a.out)
    rows = [report_run(np.load(p)) for p in sorted(out_dir.glob("*/*.npz"))]
    by = {}
    for r in rows:
        by.setdefault((r["cell"], r["arm"]), []).append(r)
    ref_key = (a.reference, "single")
    ref_r = np.mean([r["rnorm"] for r in by.get(ref_key, [])]) if ref_key in by else float("nan")
    ref_w = np.mean([r["wnorm"] for r in by.get(ref_key, [])]) if ref_key in by else float("nan")
    print("Per cell and arm, mean over seeds (n = seeds). 'loss' is the band-mean training MSE "
          "(target variance 1). '|r|' is the block gradient norm per channel (the loss gradient "
          "with respect to the whitened effective weight), '|w1|' the kernel norm of branch 1, "
          "'sigma' its normalizer, each as the per-channel median over the last-band recorded "
          "steps and then the median across the 64 channels, averaged over seeds; 'r/ref' and "
          f"'w/ref' divide by the reference single's ({a.reference}) values, and the liveness "
          "gate is both within a factor of three. 'gamma 0 -> end' is the median gamma across "
          "channels at step 0 and over the band. 'H/edge' is the top Hessian eigenvalue over "
          "2(1 + momentum)/lr. For the pair: the median whitened angle across channels at steps "
          f"{ANGLE_STEPS} in degrees, 'flow' the exact angle rate over the band in degrees per "
          "1000 steps (negative closes), and the open (angle >= 10 deg) and parked "
          "(open and alignment >= 0.95) counts of 64.")
    print(f"{'cell':22s} {'arm':7s} {'n':>2s} {'loss':>8s} {'|r|':>9s} {'r/ref':>6s} {'|w1|':>7s} "
          f"{'w/ref':>6s} {'sigma':>7s} {'gamma 0 -> end':>15s} {'H/edge':>7s} "
          f"{'angle at ' + ','.join(str(s) for s in ANGLE_STEPS):>28s} {'flow':>7s} {'open':>5s} {'park':>5s}")
    for (cell, arm), rs in sorted(by.items()):
        m = lambda k: float(np.mean([r[k] for r in rs if k in r])) if any(k in r for r in rs) else float("nan")
        line = (f"{cell:22s} {arm:7s} {len(rs):2d} {m('loss_band'):8.4f} {m('rnorm'):9.2e} "
                f"{m('rnorm') / ref_r:6.2f} {m('wnorm'):7.3f} {m('wnorm') / ref_w:6.2f} "
                f"{m('sigma'):7.3f} {m('gamma0'):6.2f} -> {m('gamma_end'):5.2f} {m('hess_over_edge'):7.2f}")
        if arm == "pair":
            angles = ",".join(f"{m(f'theta_{s}'):.0f}" for s in ANGLE_STEPS if f"theta_{s}" in rs[0])
            line += f" {angles:>28s} {m('dtheta_band'):7.2f} {m('open'):5.1f} {m('parked'):5.1f}"
        print(line)
    prof = rig.load_profiles()
    demand = prof["gradient_demand"]
    kernel = 0.23 * prof["kernel_selectors"] + 0.77 * prof["kernel_others"]
    print()
    print("Energy by eigendirection band, in percent, mean over the 64 channels and over "
          "seeds. The four bands are not a partition: the slow tail is contained in the rest, "
          "so v_max plus next five plus rest is 100 and the four columns sum to more. 'demand, train' is the share of the loss gradient "
          "with respect to the whitened effective weight, averaged over the run with equal "
          "weight per unit of time; 'demand, band' the same over the last-band steps; "
          "'branch grad, train' and 'branch grad, band' the same for the gradient with respect "
          "to a branch's whitened kernel (the tangent projection, mean over branches; the "
          "CNN's measured quantity); 'weight, "
          "band' the share of the effective whitened weight's energy over the last band. The "
          "CNN reference lines are agent synth_fc_teacher_a's clean-basis profiles: the "
          "training-average whitened-gradient demand and the trained single's block-0 kernel "
          "energy (23 percent selectors plus 77 percent others). Bands: v_max = rank 0, next "
          "five = ranks 1 to 5, rest = ranks 6 to 26, slow tail = ranks 8 to 26.")
    hdr = f"{'cell':22s} {'arm':7s} {'quantity':18s}" + "".join(f"{b:>9s}" for b in rig.BAND_NAMES)
    print(hdr)
    for label, ref in (("CNN branch grad", demand), ("CNN kernel", kernel)):
        print(f"{'reference':22s} {'':7s} {label:18s}" + "".join(
            f"{100 * ref[lo:hi].sum():9.1f}" for lo, hi in rig.BANDS.values()))
    for (cell, arm), rs in sorted(by.items()):
        m = lambda k: float(np.mean([r[k] for r in rs]))
        for label, key in (("demand, train", "r_{}_train"), ("demand, band", "r_{}_band"),
                           ("branch grad, train", "g_{}_train"), ("branch grad, band", "g_{}_band"),
                           ("weight, band", "v_{}_band")):
            if key.format("vmax") not in rs[0]:
                continue
            print(f"{cell:22s} {arm:7s} {label:18s}" + "".join(
                f"{100 * m(key.format(b)):9.1f}" for b in rig.BAND_NAMES))
    if a.csv:
        with open(out_dir / "report.csv", "w", newline="") as fh:
            keys = sorted({k for r in rows for k in r})
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader(); w.writerows(rows)


# ----------------------------------------------------------------------------
# dose: the whitening ladder beside the paper's measured one
# ----------------------------------------------------------------------------

# The paper's measured ZCA dose ladder ("New farmework paper.md", Table 18, with
# the same gains and angles in Table 13): the constant-rate depth-3 mechanism
# cell, block-0 pair, three seeds, epoch 100. Columns, in order: the block-seen
# lam_max/lam_next, the image-space ZCA epsilon that produced it, mu_eff of
# equation (64) ((lam_max - lam_next) / tr Sigma), the paired train-accuracy gain
# of the pair over the single in percentage points with its standard error, the
# median whitened branch angle over the pair's open channels pooled over seeds
# with its 25th and 75th percentiles in degrees, the mean open count out of 64,
# and the pooled median alignment |p_minus . v_max|.
PAPER_DOSE_LADDER = [
    (9.6, "unwhitened", 0.63, 2.22, 0.09, 145.0, 133.0, 154.0, 52.0, 0.999),
    (3.3, "eps=10", 0.32, 0.90, 0.01, 125.0, 107.0, 139.0, 44.0, 0.996),
    (1.7, "eps=1", 0.11, 0.10, 0.06, 107.0, 71.0, 122.0, 42.0, 0.986),
    (1.03, "eps=0.1", 0.00, -0.06, 0.14, 77.0, 58.0, 91.0, 29.0, 0.42),
    (1.05, "eps=0.01", 0.01, 0.20, 0.17, 49.0, 36.0, 64.0, 18.0, 0.39),
]


def spectrum_stats(eigenvalues) -> tuple[float, float, float]:
    """The three scale-free summaries of a spectrum: lam_max/lam_next (which sets
    the resting angle through the paper's equation (49)), mu_eff of equation (64)
    (which orders the selection clock), and lam_max/lam_eff, the retired predictor
    that survives in the paper's Appendix A.20 as the lambda-weighted mean of the
    quiet eigenvalues, (tr Sigma^2 - lam_max^2) / (tr Sigma - lam_max)."""
    eigenvalues = np.sort(np.asarray(eigenvalues, dtype=float))[::-1]   # index 0 is v_max, 1 the runner-up
    quiet = eigenvalues[1:]
    return (float(eigenvalues[0] / eigenvalues[1]),
            float((eigenvalues[0] - eigenvalues[1]) / eigenvalues.sum()),
            float(eigenvalues[0] / ((quiet * quiet).sum() / quiet.sum())))


def zca_eps_for_ratio(target: float, eigenvalues=None) -> float:
    """The ZCA epsilon of the rig's `zca<eps>` family, eigenvalues -> eigenvalues / (eigenvalues + eps)
    applied to the 27-dimensional patch spectrum, that gives a block-seen
    lam_max/lam_next of `target`. The family is monotone in eps, from 1.0 as eps
    goes to zero to the unwhitened ratio as eps grows, so a bisection suffices."""
    eigenvalues = rig.cifar_spectrum() if eigenvalues is None else eigenvalues
    def ratio(eps):
        z = eigenvalues / (eigenvalues + eps)
        return float(z[0] / z[1])
    lo, hi = 1e-6, 1e6
    if not ratio(lo) < target < ratio(hi):
        raise ValueError(f"ratio {target} outside the family's range "
                         f"({ratio(lo):.4f} to {ratio(hi):.4f})")
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if ratio(mid) < target:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


def doses(a):
    """Print, for each wanted block-seen eigenvalue ratio, the ZCA epsilon of the
    rig's spectrum family that realizes it and the three spectrum summaries, so
    that the correspondence with the paper's doses is auditable. The rig's
    epsilon is applied to the 27-dimensional patch spectrum and the paper's to
    the 3072-dimensional image, so the two epsilon values are unrelated; the
    doses are matched on the ratios, not on epsilon."""
    eigenvalues = rig.cifar_spectrum()
    r, m, e = spectrum_stats(eigenvalues)
    print("The rig's ZCA dose ladder, matched to the paper's on the block-seen "
          "eigenvalue ratio. 'spec' is the rig's spectrum name; 'lam_max/lam_next' "
          "is the ratio the paper's equation (49) turns into the resting angle; "
          "'mu_eff' is (lam_max - lam_next)/tr Sigma, equation (64), which orders "
          "the selection clock; 'lam_max/lam_eff' is the retired predictor of "
          "Appendix A.20. The last three columns are the paper's own values at the "
          "dose this row is matched to.")
    print(f"{'target':>7s} {'spec':>18s} {'lam_max/lam_next':>17s} {'mu_eff':>8s} "
          f"{'lam_max/lam_eff':>16s} | {'paper ratio':>11s} {'paper mu_eff':>12s} {'paper eps':>11s}")
    print(f"{'-':>7s} {'cifar':>18s} {r:17.3f} {m:8.4f} {e:16.3f} | "
          f"{PAPER_DOSE_LADDER[0][0]:11.2f} {PAPER_DOSE_LADDER[0][2]:12.2f} "
          f"{PAPER_DOSE_LADDER[0][1]:>11s}")
    for target in a.targets:
        row = min(PAPER_DOSE_LADDER, key=lambda t: abs(t[0] - target))
        if row[1] == "unwhitened":
            continue
        eps = zca_eps_for_ratio(target, eigenvalues)
        r, m, e = spectrum_stats(rig.spectrum(f"zca{eps:.6g}"))
        print(f"{target:7.2f} {'zca%.6g' % eps:>18s} {r:17.3f} {m:8.4f} {e:16.3f} | "
              f"{row[0]:11.2f} {row[2]:12.2f} {row[1]:>11s}")


def _pair_channels(z):
    """One pair run's per-channel band-averaged geometry: the whitened branch
    angle in degrees, the alignment |p_minus . v_max|, and the open mask."""
    steps, band = int(z["steps"]), int(z["band"])
    theta = np.degrees(band_mean(z["rec_steps"], z["rec_theta"], steps, band))
    align = band_mean(z["rec_steps"], z["rec_align"], steps, band)
    return theta, align, theta >= rig.OPEN_DEG


def dose_rows(out_dir: Path) -> list[dict]:
    """One row per cell of a dose sweep, pooled over seeds the way the paper's
    Table 18 pools: the angle and the alignment are medians over the open
    channels of all seeds at once, the open and parked counts are means over
    seeds, and the loss ratios are paired per seed."""
    out_dir = Path(out_dir)
    cells = sorted(p.name for p in out_dir.iterdir() if p.is_dir() and any(p.glob("*.npz")))
    rows = []
    for cell in cells:
        runs = _load_cell(out_dir, cell)
        if "single" not in runs:
            continue
        eigenvalues = list(runs["single"].values())[0]["lam"]
        ratio, mu_eff, lam_eff_ratio = spectrum_stats(eigenvalues)
        row = {"cell": cell, "ratio": ratio, "mu_eff": mu_eff, "lam_eff_ratio": lam_eff_ratio,
               "seeds": sorted(runs["single"])}
        band_loss = {arm: {s: float(z["loss"][-int(z["band"]):].mean()) for s, z in d.items()}
                     for arm, d in runs.items()}
        row["single_loss"] = float(np.mean(list(band_loss["single"].values())))
        for arm in ("pair", "cooled"):
            d = [math.log10(band_loss[arm][s] / band_loss["single"][s])
                 for s in sorted(band_loss.get(arm, {})) if s in band_loss["single"]]
            row[f"d_{arm}"] = (float(np.mean(d)), float(np.std(d, ddof=1) / math.sqrt(len(d)))
                               if len(d) > 1 else float("nan")) if d else None
        # liveness: the single's block gradient norm and kernel norm
        row["rnorm"] = float(np.mean([_chan_band_median(z, "rnorm") for z in runs["single"].values()]))
        row["wnorm"] = float(np.mean([_chan_band_median(z, "wnorm1") for z in runs["single"].values()]))
        row["gamma_pair"] = None
        if "pair" in runs:
            zs = list(runs["pair"].values())
            th, al, op, opens, parks = [], [], [], [], []
            for z in zs:
                t, g, o = _pair_channels(z)
                th.append(t[o]); al.append(g[o])
                opens.append(int(o.sum()))
                parks.append(int((o & (g >= rig.PARK_ALIGN)).sum()))
            th, al = np.concatenate(th), np.concatenate(al)
            row.update({
                "theta_med": float(np.median(th)) if th.size else float("nan"),
                "theta_q25": float(np.percentile(th, 25)) if th.size else float("nan"),
                "theta_q75": float(np.percentile(th, 75)) if th.size else float("nan"),
                "align_med": float(np.median(al)) if al.size else float("nan"),
                "open": float(np.mean(opens)), "parked": float(np.mean(parks)),
                "cool_factor": float(np.mean([float(z["cool_factor"]) for z in
                                              runs.get("cooled", {}).values()]))
                if runs.get("cooled") else float("nan"),
                "gamma_pair": (float(np.mean([float(np.median(z["rec_gamma"][0])) for z in zs])),
                               float(np.mean([_chan_band_median(z, "gamma") for z in zs]))),
            })
        rows.append(row)
    rows.sort(key=lambda r: -r["ratio"])
    return rows


def dose(a):
    rows = dose_rows(a.out)
    if not rows:
        print("no cells found in", a.out)
        return
    ref = rows[0] if a.reference is None else next(r for r in rows if r["cell"] == a.reference)
    null = 0.6745 / math.sqrt(D)   # median |coordinate| of a random unit vector in D dims
    print("The synthetic model's whitening dose ladder. One row per dose, ordered by the "
          "block-seen eigenvalue ratio. 'lam_max/lam_next' is the ratio of the rig's realized "
          "input spectrum; 'mu_eff' is (lam_max - lam_next)/tr Sigma, the paper's equation (64); "
          "'lam_max/lam_eff' the retired Appendix A.20 predictor. 'log10 pair/single' and "
          "'log10 cooled/single' are the paired per-seed log10 ratios of the band-mean training "
          "mean-squared error over the last 2000 steps, mean +- standard error over the seeds; "
          "negative means the arm is below the single. 'angle' is the median whitened branch "
          "angle over the pair's open channels (at least 10 degrees) pooled over seeds, with the "
          "25th to 75th percentile in parentheses; 'open' is the mean count out of 64; 'align' "
          "is the pooled median |p_minus . v_max| over the open channels, against an isotropic "
          f"null of {null:.2f} in 27 dimensions; 'parked' is the mean count that is both open and "
          "aligned above 0.95. 'r/ref' and 'w/ref' are the single's block gradient norm and "
          f"kernel norm divided by the reference cell's ({ref['cell']}), the liveness gate being "
          "both within a factor of three. 'gamma' is the pair's median gamma across channels at "
          "step 0 and over the last band.")
    hdr = (f"{'cell':20s} {'n':>2s} {'ratio':>6s} {'mu_eff':>7s} {'l/leff':>7s} "
           f"{'log10 pair/single':>19s} {'log10 cooled/single':>21s} {'cool':>5s} "
           f"{'angle (q25-q75)':>21s} {'open':>5s} {'park':>5s} {'align':>6s} "
           f"{'r/ref':>6s} {'w/ref':>6s} {'gamma':>13s}")
    print(hdr)
    for r in rows:
        def pm(key):
            v = r.get(key)
            if not v:
                return "n/a".rjust(19)
            return f"{v[0]:+.3f} +- {v[1]:.3f}".rjust(19)
        ang = (f"{r.get('theta_med', float('nan')):.0f} "
               f"({r.get('theta_q25', float('nan')):.0f}-{r.get('theta_q75', float('nan')):.0f})")
        g = r.get("gamma_pair")
        gtxt = f"{g[0]:.2f} -> {g[1]:.2f}" if g else "n/a"
        print(f"{r['cell']:20s} {len(r['seeds']):2d} {r['ratio']:6.2f} {r['mu_eff']:7.4f} "
              f"{r['lam_eff_ratio']:7.2f} {pm('d_pair')} {pm('d_cooled'):>21s} "
              f"{r.get('cool_factor', float('nan')):5.2f} {ang:>21s} "
              f"{r.get('open', float('nan')):5.1f} {r.get('parked', float('nan')):5.1f} "
              f"{r.get('align_med', float('nan')):6.3f} "
              f"{r['rnorm'] / ref['rnorm']:6.2f} {r['wnorm'] / ref['wnorm']:6.2f} {gtxt:>13s}")
    print()
    print("Beside the paper's measured ladder (Table 18 of 'New farmework paper.md', the "
          "constant-rate depth-3 cell, block-0 pair, three seeds, epoch 100). Each synthetic row "
          "is matched to the paper row whose block-seen lam_max/lam_next is nearest, as the "
          "campaign directive asks; the two epsilon values are unrelated, since the paper "
          "whitens the 3072-dimensional image and the rig whitens the 27-dimensional patch "
          "spectrum directly. The gain columns are not comparable in magnitude -- the paper's is "
          "a train-accuracy difference in percentage points and the model's a log10 training-loss "
          "ratio -- so only their ordering and the point at which they vanish are compared. "
          "Angles are medians over open channels pooled over three seeds with the 25th to 75th "
          "percentile; open is the mean count out of 64; alignment is the pooled median "
          "|p_minus . v_max| over the open channels.")
    print(f"{'':>10s} | {'paper (CNN, depth-3 cell)':^54s} | {'synthetic model':^54s}")
    print(f"{'matched':>10s} | {'ratio':>6s} {'mu_eff':>6s} {'gain (pts)':>13s} {'angle':>14s} "
          f"{'open':>5s} {'align':>6s} | {'ratio':>6s} {'mu_eff':>6s} {'log10 loss':>13s} "
          f"{'angle':>14s} {'open':>5s} {'align':>6s}")
    for r in rows:
        pr = min(PAPER_DOSE_LADDER, key=lambda t: abs(math.log(t[0]) - math.log(r["ratio"])))
        pang = f"{pr[5]:.0f} ({pr[6]:.0f}-{pr[7]:.0f})"
        sang = (f"{r.get('theta_med', float('nan')):.0f} "
                f"({r.get('theta_q25', float('nan')):.0f}-{r.get('theta_q75', float('nan')):.0f})")
        dp = r.get("d_pair")
        sgain = f"{dp[0]:+.3f} +- {dp[1]:.3f}" if dp else "n/a"
        print(f"{pr[1]:>10s} | {pr[0]:6.2f} {pr[2]:6.2f} {pr[3]:+8.2f} +- {pr[4]:.2f} {pang:>14s} "
              f"{pr[8]:5.0f} {pr[9]:6.3f} | {r['ratio']:6.2f} {r['mu_eff']:6.3f} {sgain:>13s} "
              f"{sang:>14s} {r.get('open', float('nan')):5.1f} {r.get('align_med', float('nan')):6.3f}")


def teacher_stats(a):
    """Training-free readouts of each teacher at each spectrum: the teacher's
    realized whitened energy per band (mean over hidden units), the share of
    the target variance that a linear readout of the input explains (least
    squares on the 27 coordinates plus a constant, averaged over the 10
    outputs), and the share explained by the best readout of the 64 random
    ReLU features of the student's raw initialization at seed 42 (a
    random-feature ceiling for what the student reaches without moving its
    kernels)."""
    torch.set_num_threads(1)
    print("teacher spec, spectrum: the teacher's realized whitened energy per band in percent "
          "(v_max, next five, rest, slow tail; mean over hidden units), the linear R^2 of the "
          "target, and the R^2 of a 64-unit random ReLU feature regression (student init at "
          "seed 42, kernels frozen, readout by least squares)")
    for spec in a.ratios:
        for teacher_name in a.teachers:
            eigenvalues, inputs, targets, energy = make_cell(spec, None, parse_teacher(teacher_name))
            inputs_with_bias = torch.cat([inputs, torch.ones(inputs.shape[0], 1, dtype=rig.DTYPE)], 1)
            coefficients = torch.linalg.lstsq(inputs_with_bias, targets).solution
            linear_r2 = 1 - ((targets - inputs_with_bias @ coefficients) ** 2).mean() / targets.var(0).mean()
            generator = torch.Generator().manual_seed(42)
            kernels, gamma, beta, readout = rig.init_params("single", eigenvalues, C, K, generator)
            with torch.no_grad():
                activations = gamma * (inputs @ kernels[0].T) / rig.sigma_of(kernels, eigenvalues)[0] + beta
                features_with_bias = torch.cat([torch.relu(activations), torch.ones(inputs.shape[0], 1, dtype=rig.DTYPE)], 1)
                coefficients = torch.linalg.lstsq(features_with_bias, targets).solution
                random_feature_r2 = 1 - ((targets - features_with_bias @ coefficients) ** 2).mean() / targets.var(0).mean()
            bands = "n/a" if energy is None else " ".join(
                f"{100 * energy[lo:hi].sum():5.1f}" for lo, hi in rig.BANDS.values())
            print(f"{teacher_name:12s} {spec:8s} bands {bands:>24s}   linear R^2 {float(linear_r2):.3f}   "
                  f"random-feature R^2 {float(random_feature_r2):.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sweep")
    s.add_argument("--out", required=True)
    s.add_argument("--ratios", nargs="+", default=["cifar", "3.3", "1.7", "1.0"],
                   help="spectrum specs (see rig.spectrum): 'cifar', a bare ratio x "
                        "(the CIFAR tail with lam_max = x lam_next), 'flat<x>', 'zca<eps>'")
    s.add_argument("--tail-shares", type=float, nargs="*", default=[],
                   help="the share of the teacher's quiet energy in the slow tail (ranks 8 to 26), "
                        "in (0, 1); omit for the teacher's own share (19/27 of 26 for the whitened "
                        "teacher). Cells are the product with --vmax-shares.")
    s.add_argument("--vmax-shares", type=float, nargs="*", default=[],
                   help="the share of the teacher's whitened energy on v_max, in [0, 1); "
                        "omit for the teacher's own share (1/27 for the whitened teacher), "
                        "which is what the spectrum cells use")
    s.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    s.add_argument("--arms", nargs="+", default=list(rig.ARMS))
    s.add_argument("--steps", type=int, default=10000)
    s.add_argument("--lr", type=float, default=0.1)
    s.add_argument("--momentum", type=float, default=0.9)
    s.add_argument("--wd", type=float, default=5e-4)
    s.add_argument("--every", type=int, default=10)
    s.add_argument("--record-first", type=int, default=500)
    s.add_argument("--band", type=int, default=2000)
    s.add_argument("--flow", action="store_true",
                   help="add the reference cell at lr / flow_factor for steps * flow_factor")
    s.add_argument("--flow-factor", type=int, default=5)
    s.add_argument("--iso", action="store_true",
                   help="add the reference cell with the whitened-isotropic initialization")
    s.add_argument("--no-curvature", action="store_true")
    s.add_argument("--samples", type=int, default=N,
                   help="fixed training samples (4096 for a recorded sweep, 1024 for a pilot)")
    s.add_argument("--precision", choices=["float32", "float64"], default="float64",
                   help="float64 for a recorded sweep, float32 for a pilot")
    s.add_argument("--workers", type=int, default=8)
    s.add_argument("--teachers", nargs="+", default=["p128"],
                   help="teacher specs: u<width> (uniform) or p<width> (profile-matched), "
                        "with optional _s<tail shift>; each becomes a cell tag")
    s.add_argument("--cooled-ratios", nargs="+", default=None,
                   help="run the cooled arm only at these spectrum specs (default: all)")
    s.set_defaults(func=sweep)
    m = sub.add_parser("summarize")
    m.add_argument("out")
    m.set_defaults(func=summarize)
    f = sub.add_parser("figures")
    f.add_argument("out")
    f.add_argument("--fig-dir", required=True)
    f.add_argument("--prefix", default="synthmodel_synth_fc_teacher_e_")
    f.set_defaults(func=figures)
    r = sub.add_parser("report")
    r.add_argument("out")
    r.add_argument("--reference", default="cifar_w128",
                   help="the cell whose single sets the liveness reference")
    r.add_argument("--csv", action="store_true")
    r.set_defaults(func=report)
    dd = sub.add_parser("dose")
    dd.add_argument("out")
    dd.add_argument("--reference", default=None,
                    help="the cell whose single sets the liveness reference "
                         "(default: the steepest spectrum present)")
    dd.set_defaults(func=dose)
    ds = sub.add_parser("doses")
    ds.add_argument("--targets", type=float, nargs="+",
                    default=[r for r, *_ in PAPER_DOSE_LADDER])
    ds.set_defaults(func=doses)
    ts = sub.add_parser("teacher-stats")
    ts.add_argument("--ratios", nargs="+", default=["cifar", "flat1"])
    ts.add_argument("--teachers", nargs="+", default=["u128", "p128", "p128_s0.5"])
    ts.set_defaults(func=teacher_stats)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
