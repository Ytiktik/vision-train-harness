#!/usr/bin/env python3
"""Run a list of configs back to back on a local GPU, unattended.

Written 2026-09-06 for a workstation that cannot be reached from outside: the
code arrives by `git pull`, the runs are started once by hand, and everything
after that -- ordering, restarts, telemetry -- happens here. It is the local
counterpart of `submit_thunder_experiment.py`, minus the provisioning.

--------------------------------------------------------------------------------
What it does, and why in this shape
--------------------------------------------------------------------------------
ONE GPU MEANS A QUEUE, NOT A FAN-OUT. Thunder runs a wave concurrently on eight
instances; a workstation runs it one config at a time. So this takes an ordered
list and works through it, and the order in the file is the order of the science:
put the arms whose results gate later decisions first.

IT RESUMES, TWICE OVER. Each config's `job.entry_module` is honoured, which for
this campaign is `agents.stage2.train_resumable` -- it checkpoints every epoch and
re-attaches to its own W&B run, so a killed process, a reboot or a crash costs one
epoch. On top of that this script keeps its own state file, so restarting it skips
what has already finished and re-enters what had not.

IT REFUSES TO START A CAMPAIGN IT CANNOT FINISH. Before the first run it checks the
GPU, the dataset, the W&B credentials and the working tree, and prints the commit
it is running. A queue that dies eight hours in because W&B was never logged in is
the failure this is written against.

--------------------------------------------------------------------------------
Usage
--------------------------------------------------------------------------------
    python scripts/run_queue.py --queue queue.txt            # run it
    python scripts/run_queue.py --queue queue.txt --check    # gates only, run nothing
    python scripts/run_queue.py --queue queue.txt --dry-run  # print the commands

`queue.txt` is one config path per line; blank lines and `#` comments are ignored.
Leave it running under tmux or with nohup:

    tmux new -s runs 'python scripts/run_queue.py --queue queue.txt'
    nohup python scripts/run_queue.py --queue queue.txt > outputs/queue.log 2>&1 &
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
STATE = REPO / "outputs" / "run_queue_state.json"
MAX_ATTEMPTS = 3


def say(*a: object) -> None:
    print(f"[{dt.datetime.now():%H:%M:%S}]", *a, flush=True)


# --------------------------------------------------------------------------------
# 1. The gates
# --------------------------------------------------------------------------------

def check_environment(configs: list[Path], require_wandb: bool = True) -> list[str]:
    """Everything that would make a run die hours in, checked in seconds."""
    problems: list[str] = []

    try:
        import torch
        if not torch.cuda.is_available():
            problems.append("torch reports no CUDA device")
        else:
            name = torch.cuda.get_device_name(0)
            free, total = torch.cuda.mem_get_info()
            say(f"gpu: {name}, {total / 2**30:.0f} GiB, {free / 2**30:.0f} GiB free")
            say(f"torch {torch.__version__}, cuda {torch.version.cuda}, "
                f"{torch.cuda.device_count()} device(s)")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"torch is not importable: {exc!r}")

    say(f"cpu: {os.cpu_count()} cores visible")

    for cfg_path in configs:
        if not cfg_path.exists():
            problems.append(f"config missing: {cfg_path}")
            continue
        cfg = yaml.safe_load(cfg_path.read_text())
        args = (cfg.get("dataset") or {}).get("args") or {}
        data_dir = args.get("data_dir")
        if data_dir:
            resolved = Path(data_dir)
            if not resolved.is_absolute():
                resolved = REPO / resolved
            if not resolved.exists():
                problems.append(
                    f"{cfg_path.name}: dataset directory {resolved} does not exist "
                    "(symlink it to wherever the data really is)")
            else:
                shards = list(resolved.glob("*.bin")) or list(resolved.glob("*/*.bin"))
                say(f"data: {resolved} present, {len(shards)} shard file(s)")

    if require_wandb:
        has_env = bool(os.environ.get("WANDB_API_KEY"))
        netrc = Path.home() / ".netrc"
        has_netrc = netrc.exists() and "api.wandb.ai" in netrc.read_text(errors="ignore")
        if not (has_env or has_netrc):
            problems.append(
                "no W&B credentials: set WANDB_API_KEY or run `wandb login` "
                "(the runs log there and the resume checkpoints live there)")
        else:
            say(f"wandb: credentials found ({'env' if has_env else '~/.netrc'})")

    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                              capture_output=True, text=True, timeout=30).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO,
                               capture_output=True, text=True, timeout=30).stdout.strip()
        say(f"repo: commit {head}" + (" (working tree has local changes)" if dirty else ""))
    except Exception as exc:  # noqa: BLE001
        say(f"repo: could not read the commit ({exc!r})")

    return problems


# --------------------------------------------------------------------------------
# 2. The queue
# --------------------------------------------------------------------------------

def read_queue(path: Path) -> list[Path]:
    out: list[Path] = []
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        p = Path(line)
        out.append(p if p.is_absolute() else REPO / p)
    return out


def load_state() -> dict:
    try:
        return json.loads(STATE.read_text())
    except Exception:  # noqa: BLE001
        return {}


def save_state(state: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=2))


def entry_module(cfg_path: Path) -> str:
    cfg = yaml.safe_load(cfg_path.read_text())
    return (cfg.get("job") or {}).get("entry_module", "structural_reparam.deploy.train")


def run_one(cfg_path: Path, device: str, dry_run: bool) -> int:
    name = cfg_path.stem.replace("config_", "")
    out_dir = REPO / "outputs" / name
    cmd = [sys.executable, "-m", entry_module(cfg_path), "--device", device,
           "--config", str(cfg_path), "--output-dir", str(out_dir)]
    say("run:", " ".join(cmd))
    if dry_run:
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / "run.log"
    started = time.time()
    with log.open("ab") as fh:
        fh.write(f"\n=== {dt.datetime.now():%Y-%m-%d %H:%M:%S} {' '.join(cmd)}\n".encode())
        fh.flush()
        proc = subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT)
    say(f"exit {proc.returncode} after {(time.time() - started) / 3600:.2f} h; log {log}")
    return proc.returncode


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queue", type=Path, required=True, help="file of config paths")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--check", action="store_true", help="run the gates and stop")
    ap.add_argument("--dry-run", action="store_true", help="print commands, run nothing")
    ap.add_argument("--no-wandb-check", action="store_true")
    ap.add_argument("--retries", type=int, default=MAX_ATTEMPTS,
                    help="attempts per config; a resumable run continues where it stopped")
    args = ap.parse_args()

    configs = read_queue(args.queue)
    say(f"queue: {len(configs)} config(s) from {args.queue}")
    problems = check_environment(configs, require_wandb=not args.no_wandb_check)
    if problems:
        say("REFUSING TO START:")
        for p in problems:
            say("   " + p)
        sys.exit(1)
    say("all gates passed")
    if args.check:
        return

    state = load_state()
    for i, cfg in enumerate(configs, 1):
        key = str(cfg.relative_to(REPO)) if cfg.is_relative_to(REPO) else str(cfg)
        done = state.get(key, {})
        if done.get("status") == "finished":
            say(f"[{i}/{len(configs)}] {cfg.name}: already finished, skipping")
            continue
        for attempt in range(done.get("attempts", 0) + 1, args.retries + 1):
            say(f"[{i}/{len(configs)}] {cfg.name}: attempt {attempt} of {args.retries}")
            rc = run_one(cfg, args.device, args.dry_run)
            state[key] = {"attempts": attempt,
                          "status": "finished" if rc == 0 else "failed",
                          "returncode": rc, "when": dt.datetime.now().isoformat()}
            save_state(state)
            if rc == 0:
                break
            say(f"   failed with {rc}; the entry module resumes from its last epoch")
        else:
            say(f"[{i}/{len(configs)}] {cfg.name}: giving up after {args.retries} attempts")
    say("queue finished")


if __name__ == "__main__":
    main()
