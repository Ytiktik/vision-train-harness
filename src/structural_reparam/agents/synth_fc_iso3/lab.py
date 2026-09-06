"""Thunder entry point for the synthetic model.

The Thunder submitter invokes an entry module as
``python -m <module> --device <d> --output-dir <dir> --config <path>``, so this
file adapts that interface to the ``sweep`` command of ``run.py``: it reads the
``sweep`` block of the config (spectra, v_max shares, seeds, arms, teacher specs
and the training recipe), turns it into command-line flags, and calls
``run.main``. The model runs on the CPU in double precision; ``--device`` is
accepted for the submitter's sake and ignored.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from structural_reparam.agents.synth_fc_iso3 import run as sweep_run


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    cfg = yaml.safe_load(Path(a.config).read_text()).get("sweep", {})
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    argv = ["run", "sweep", "--out", str(out)]
    for key in ("ratios", "vmax_shares", "seeds", "arms", "teachers"):
        if key in cfg:
            argv += [f"--{key}"] + [str(v) for v in cfg[key]]
    for key in ("steps", "lr", "momentum", "wd", "every", "record-first",
                "band", "samples", "precision", "workers"):
        val = cfg.get(key.replace("-", "_"))
        if val is not None:
            argv += [f"--{key}", str(val)]
    if a.smoke:
        argv += ["--steps", "20", "--seeds", "42", "--workers", "2"]

    print("sweep argv:", " ".join(argv[1:]), flush=True)
    sys.argv = argv
    sweep_run.main()
    print("sweep finished; results in", out, flush=True)


if __name__ == "__main__":
    main()
