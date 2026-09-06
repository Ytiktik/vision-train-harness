"""Evaluation entrypoint placeholder."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained experiment.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Experiment output directory.")
    parser.add_argument("--fuse", action="store_true", help="Evaluate deploy-time fused model.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError(f"Evaluation pipeline is not ported yet for {args.run_dir}.")


if __name__ == "__main__":
    main()
