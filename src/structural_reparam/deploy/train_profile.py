"""Profiling entry point: runs a warmup epoch then profiles one full epoch.

Reuses all infrastructure from train.py. Wraps the profiled epoch with
torch.profiler, prints the CUDA-time table to stdout, and saves a chrome
trace to the output directory for offline inspection.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.profiler as tprof

from structural_reparam.deploy.train import (
    LOGGER,
    build_loaders,
    build_model,
    build_optimizer,
    build_scheduler,
    get_output_dir,
    parse_args,
    run_epoch,
    seed_everything,
    select_metrics,
    start_wandb,
)
import torch.nn as nn
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]


def load_config(args) -> dict[str, Any]:
    if args.config:
        config_path = Path(args.config)
    else:
        config_path = REPO_ROOT / "configs" / f"{args.experiment}.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def profile_variant(
    config: dict[str, Any],
    variant: dict[str, Any],
    seed: int,
    train_loader,
    test_loader,
    device: torch.device,
) -> None:
    variant_name = variant["name"]
    prof_config = config.get("profiling", {})
    warmup_epochs = int(prof_config.get("warmup_epochs", 1))
    profile_epochs = int(prof_config.get("profile_epochs", 1))
    total_epochs = warmup_epochs + profile_epochs

    LOGGER.info("Profiling variant=%s seed=%s warmup=%s profile=%s",
                variant_name, seed, warmup_epochs, profile_epochs)
    seed_everything(seed)

    model = build_model(config, variant).to(device)
    optimizer = build_optimizer(config, model, variant)
    scheduler = build_scheduler(config, optimizer)
    criterion = nn.CrossEntropyLoss()
    configured_metrics = config.get("metrics", ["train_loss", "test_accuracy"])
    run = start_wandb(config, variant, seed)

    output_dir = get_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)

    probe_config = config.get("gradient_probe", {})
    if probe_config.get("enabled"):
        from structural_reparam.analysis import GradientProbe
        probe = GradientProbe(model)
    else:
        probe = None

    for epoch in range(1, total_epochs + 1):
        is_profile_epoch = epoch > warmup_epochs

        if is_profile_epoch:
            trace_path = output_dir / f"trace_{variant_name}_seed{seed}_ep{epoch}.json"
            with tprof.profile(
                activities=[tprof.ProfilerActivity.CPU, tprof.ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
            ) as prof:
                with tprof.record_function("train_epoch"):
                    train_stats = run_epoch(model, train_loader, criterion, device, optimizer)
                with tprof.record_function("test_epoch"):
                    test_stats = run_epoch(model, test_loader, criterion, device)
                if probe is not None:
                    with tprof.record_function("epoch_stats"):
                        grad_stats = probe.epoch_stats()

            print(f"\n=== PROFILER epoch={epoch} variant={variant_name} (CUDA time) ===",
                  flush=True)
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30),
                  flush=True)
            print(f"\n=== PROFILER epoch={epoch} variant={variant_name} (CPU time) ===",
                  flush=True)
            print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20),
                  flush=True)
            prof.export_chrome_trace(str(trace_path))
            print(f"Chrome trace -> {trace_path}", flush=True)
        else:
            LOGGER.info("Warmup epoch %s/%s", epoch, warmup_epochs)
            train_stats = run_epoch(model, train_loader, criterion, device, optimizer)
            test_stats = run_epoch(model, test_loader, criterion, device)
            grad_stats = probe.epoch_stats() if probe is not None else {}

        if scheduler is not None:
            scheduler.step()

        metrics = select_metrics(configured_metrics, train_stats, test_stats, model)
        record = {"epoch": epoch, "seed": seed, "variant": variant_name, **metrics, **grad_stats}
        if run is not None:
            run.log(record, step=epoch)
        print(json.dumps(record), flush=True)

    if probe is not None:
        probe.detach()
    if run is not None:
        run.finish()


def main() -> None:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    args = parse_args()
    config = load_config(args)
    device = torch.device(args.device)

    train_loader, test_loader = build_loaders(config)

    base_seed = int(config.get("base_seed", 42))
    variants = config["model"].get("variants", [{"name": "default", "args": {}}])

    for variant in variants:
        profile_variant(config, variant, base_seed, train_loader, test_loader, device)


if __name__ == "__main__":
    main()
