"""Dual-checkpoint training scheme.

Two independently-initialized copies of the same model architecture (θ_A, θ_B)
are trained together. At each training step:

  1. Forward + backward both models on the same class-concentrated batch
     to compute g_A and g_B.
  2. Snapshot params/momentum of each, take a candidate ``optimizer.step()`` on
     each in isolation, evaluate loss on a shuffled (oracle) batch, then restore.
  3. The "winner" is whichever candidate step gave the lower oracle loss.
  4. Apply the winner's gradient to *both* models, then take a real
     ``optimizer.step()`` on each. Both checkpoints move every step — the loser
     is pulled along by the winner's direction, preventing it from getting
     stuck.

Per-step records (winner, candidate losses, etc.) go to JSONL. Per-epoch
aggregates (frac-A-wins, both models' train/test accuracy) are logged at
epoch end alongside standard metrics.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

LOGGER = logging.getLogger(__name__)


def _snapshot_params(model: nn.Module) -> list[torch.Tensor]:
    return [p.detach().clone() for p in model.parameters()]


def _restore_params(model: nn.Module, snap: list[torch.Tensor]) -> None:
    for p, p_snap in zip(model.parameters(), snap):
        p.data.copy_(p_snap)


def _snapshot_momentum(
    optimizer: torch.optim.Optimizer, model: nn.Module
) -> list[torch.Tensor | None]:
    snap: list[torch.Tensor | None] = []
    for p in model.parameters():
        buf = optimizer.state.get(p, {}).get("momentum_buffer")
        snap.append(buf.detach().clone() if buf is not None else None)
    return snap


def _restore_momentum(
    optimizer: torch.optim.Optimizer,
    model: nn.Module,
    snap: list[torch.Tensor | None],
) -> None:
    for p, buf in zip(model.parameters(), snap):
        state = optimizer.state.get(p)
        if buf is None:
            if state is not None and "momentum_buffer" in state:
                del state["momentum_buffer"]
        else:
            if state is None:
                optimizer.state[p] = {"momentum_buffer": buf.clone()}
            elif "momentum_buffer" in state:
                state["momentum_buffer"].copy_(buf)
            else:
                state["momentum_buffer"] = buf.clone()


def _snapshot_grads(model: nn.Module) -> list[torch.Tensor | None]:
    return [
        p.grad.detach().clone() if p.grad is not None else None
        for p in model.parameters()
    ]


def _apply_grads(model: nn.Module, grads: list[torch.Tensor | None]) -> None:
    for p, g in zip(model.parameters(), grads):
        if g is None:
            p.grad = None
        else:
            if p.grad is None:
                p.grad = g.clone()
            else:
                p.grad.copy_(g)


@torch.no_grad()
def _eval_loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor, criterion: nn.Module) -> float:
    was_training = model.training
    model.eval()
    logits = model(x)
    loss = criterion(logits, y).item()
    if was_training:
        model.train()
    return loss


@torch.no_grad()
def _eval_loader(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)
        total_loss += loss.item() * targets.size(0)
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_examples += targets.size(0)
    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


def run_dual_epoch(
    model_A: nn.Module,
    model_B: nn.Module,
    optimizer_A: torch.optim.Optimizer,
    optimizer_B: torch.optim.Optimizer,
    train_loader: DataLoader,
    oracle_loader_state: dict[str, Any],
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    start_global_step: int,
    log_file=None,
) -> tuple[dict[str, float], int]:
    """One dual-checkpoint training epoch.

    ``oracle_loader_state`` is a dict carrying the oracle loader and a persistent
    iterator across epochs (so we don't restart the shuffle each epoch).
    """
    model_A.train()
    model_B.train()

    total_loss_A = 0.0
    total_loss_B = 0.0
    total_correct_A = 0
    total_correct_B = 0
    total_examples = 0
    n_A_wins = 0
    n_B_wins = 0
    n_ties = 0
    sum_oracle_loss_A = 0.0
    sum_oracle_loss_B = 0.0
    sum_oracle_loss_winner = 0.0
    global_step = start_global_step

    oracle_loader = oracle_loader_state["loader"]

    def next_oracle_batch():
        try:
            return next(oracle_loader_state["iter"])
        except StopIteration:
            oracle_loader_state["iter"] = iter(oracle_loader)
            return next(oracle_loader_state["iter"])

    for images, targets in train_loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Forward + backward on both models with their own grads.
        optimizer_A.zero_grad(set_to_none=True)
        logits_A = model_A(images)
        loss_A = criterion(logits_A, targets)
        loss_A.backward()

        optimizer_B.zero_grad(set_to_none=True)
        logits_B = model_B(images)
        loss_B = criterion(logits_B, targets)
        loss_B.backward()

        # Snapshot for candidate evaluation.
        snap_p_A = _snapshot_params(model_A)
        snap_p_B = _snapshot_params(model_B)
        snap_m_A = _snapshot_momentum(optimizer_A, model_A)
        snap_m_B = _snapshot_momentum(optimizer_B, model_B)
        grads_A = _snapshot_grads(model_A)
        grads_B = _snapshot_grads(model_B)

        x_orac, y_orac = next_oracle_batch()
        x_orac = x_orac.to(device, non_blocking=True)
        y_orac = y_orac.to(device, non_blocking=True)

        # Candidate A: apply A's update, eval, restore.
        optimizer_A.step()
        oracle_loss_A = _eval_loss(model_A, x_orac, y_orac, criterion)
        _restore_params(model_A, snap_p_A)
        _restore_momentum(optimizer_A, model_A, snap_m_A)

        # Candidate B: apply B's update, eval, restore.
        optimizer_B.step()
        oracle_loss_B = _eval_loss(model_B, x_orac, y_orac, criterion)
        _restore_params(model_B, snap_p_B)
        _restore_momentum(optimizer_B, model_B, snap_m_B)

        # Pick winner.
        if oracle_loss_A < oracle_loss_B:
            winner = "A"
            winner_grads = grads_A
            winner_loss = oracle_loss_A
            n_A_wins += 1
        elif oracle_loss_B < oracle_loss_A:
            winner = "B"
            winner_grads = grads_B
            winner_loss = oracle_loss_B
            n_B_wins += 1
        else:
            winner = "tie"
            winner_grads = grads_A
            winner_loss = oracle_loss_A
            n_ties += 1

        # Apply winner's gradient to BOTH models, then real step on each.
        _apply_grads(model_A, winner_grads)
        _apply_grads(model_B, winner_grads)
        optimizer_A.step()
        optimizer_B.step()

        sum_oracle_loss_A += oracle_loss_A
        sum_oracle_loss_B += oracle_loss_B
        sum_oracle_loss_winner += winner_loss

        # Accumulate train metrics (using the original loss/logits from each model).
        bs = targets.size(0)
        total_loss_A += loss_A.detach().item() * bs
        total_loss_B += loss_B.detach().item() * bs
        total_correct_A += (logits_A.detach().argmax(dim=1) == targets).sum().item()
        total_correct_B += (logits_B.detach().argmax(dim=1) == targets).sum().item()
        total_examples += bs

        if log_file is not None:
            log_file.write(json.dumps({
                "global_step": global_step,
                "epoch": epoch,
                "oracle_loss_A": oracle_loss_A,
                "oracle_loss_B": oracle_loss_B,
                "winner": winner,
            }) + "\n")

        global_step += 1

    n_steps = max(1, n_A_wins + n_B_wins + n_ties)
    stats = {
        "train_loss_A": total_loss_A / total_examples,
        "train_loss_B": total_loss_B / total_examples,
        "train_accuracy_A": total_correct_A / total_examples,
        "train_accuracy_B": total_correct_B / total_examples,
        "frac_A_wins": n_A_wins / n_steps,
        "frac_B_wins": n_B_wins / n_steps,
        "frac_ties": n_ties / n_steps,
        "oracle_loss_A_mean": sum_oracle_loss_A / n_steps,
        "oracle_loss_B_mean": sum_oracle_loss_B / n_steps,
        "oracle_loss_winner_mean": sum_oracle_loss_winner / n_steps,
        "n_steps": float(n_steps),
    }
    if log_file is not None:
        log_file.flush()
    return stats, global_step


def train_variant_dual_checkpoint(
    config: dict[str, Any],
    variant: dict[str, Any],
    seed: int,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    start_notification: str | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    from structural_reparam.deploy.train import (
        build_model,
        build_optimizer,
        build_scheduler,
        get_output_dir,
        get_wandb_url,
        import_target,
        resolve_repo_path,
        seed_everything,
        send_telegram,
        start_wandb,
    )

    variant_name = variant["name"]
    dual_cfg = config.get("dual_checkpoint", {})
    seed_offset = int(dual_cfg.get("seed_offset", 10000))
    seed_A = seed
    seed_B = seed + seed_offset

    LOGGER.info(
        "Starting dual-checkpoint seed=%s (A=%s, B=%s) variant=%s",
        seed,
        seed_A,
        seed_B,
        variant_name,
    )
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    seed_everything(seed_A)
    model_A = build_model(config, variant).to(device)
    seed_everything(seed_B)
    model_B = build_model(config, variant).to(device)

    optimizer_A = build_optimizer(config, model_A, variant)
    optimizer_B = build_optimizer(config, model_B, variant)
    scheduler_A = build_scheduler(config, optimizer_A)
    scheduler_B = build_scheduler(config, optimizer_B)
    criterion = nn.CrossEntropyLoss()

    # Oracle loader: shuffled batches from the same training set.
    oracle_dataset_args = dict(config["dataset"].get("args", {}))
    oracle_dataset_args["batch_mode"] = "shuffled"
    oracle_dataset_args.pop("classes_per_batch", None)
    if "data_dir" in oracle_dataset_args:
        oracle_dataset_args["data_dir"] = resolve_repo_path(oracle_dataset_args["data_dir"])
    oracle_train_loader, _ = import_target(config["dataset"]["target"])(**oracle_dataset_args)
    oracle_state = {"loader": oracle_train_loader, "iter": iter(oracle_train_loader)}

    output_dir = get_output_dir(config) / "dual_checkpoint"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"{variant_name}_seed{seed}.jsonl"
    log_file = log_path.open("w", encoding="utf-8")
    LOGGER.info("Dual-checkpoint per-step log -> %s", log_path)

    run = start_wandb(config, variant, seed)
    wandb_url = get_wandb_url(run)
    if start_notification is not None:
        send_telegram(start_notification + f"\nW&B: {wandb_url or 'not available'}", config)

    total_epochs = int(config["train"]["epochs"])
    history: list[dict[str, Any]] = []
    started_at = time.perf_counter()
    global_step = 0

    try:
        for epoch in range(1, total_epochs + 1):
            train_stats, global_step = run_dual_epoch(
                model_A,
                model_B,
                optimizer_A,
                optimizer_B,
                train_loader,
                oracle_state,
                criterion,
                device,
                epoch=epoch,
                start_global_step=global_step,
                log_file=log_file,
            )
            test_A = _eval_loader(model_A, test_loader, criterion, device)
            test_B = _eval_loader(model_B, test_loader, criterion, device)
            if scheduler_A is not None:
                scheduler_A.step()
            if scheduler_B is not None:
                scheduler_B.step()

            record = {
                "epoch": epoch,
                "seed": seed,
                "variant": variant_name,
                "train_loss_A": train_stats["train_loss_A"],
                "train_loss_B": train_stats["train_loss_B"],
                "train_accuracy_A": train_stats["train_accuracy_A"],
                "train_accuracy_B": train_stats["train_accuracy_B"],
                "test_loss_A": test_A["loss"],
                "test_loss_B": test_B["loss"],
                "test_accuracy_A": test_A["accuracy"],
                "test_accuracy_B": test_B["accuracy"],
                "frac_A_wins": train_stats["frac_A_wins"],
                "frac_B_wins": train_stats["frac_B_wins"],
                "frac_ties": train_stats["frac_ties"],
                "oracle_loss_A_mean": train_stats["oracle_loss_A_mean"],
                "oracle_loss_B_mean": train_stats["oracle_loss_B_mean"],
                "oracle_loss_winner_mean": train_stats["oracle_loss_winner_mean"],
            }
            history.append(record)
            if run is not None:
                run.log(record, step=epoch)
            LOGGER.info(
                "Dual epoch %s/%s variant=%s acc_A=%.4f acc_B=%.4f frac_A_wins=%.3f",
                epoch,
                total_epochs,
                variant_name,
                test_A["accuracy"],
                test_B["accuracy"],
                train_stats["frac_A_wins"],
            )
            print(json.dumps(record), flush=True)
    finally:
        log_file.close()
        if run is not None:
            run.finish()

    LOGGER.info(
        "Finished dual variant=%s epochs=%s elapsed_sec=%.2f",
        variant_name,
        total_epochs,
        time.perf_counter() - started_at,
    )
    return history, wandb_url


def main() -> None:
    """Standalone entry point: ``python -m structural_reparam.deploy.dual_checkpoint``.

    Reuses CLI parsing, config loading, dataset construction, and final
    metrics writing from ``train.py``; replaces only the per-variant inner
    loop with the dual-checkpoint scheme.
    """
    from structural_reparam.deploy.train import (
        build_loaders,
        experiment_seeds,
        format_duration,
        format_experiment_summary,
        format_final_metrics,
        get_output_dir,
        load_env_file,
        parse_args,
        resolve_config_path,
        seed_everything,
        send_telegram,
        setup_logging,
        write_metrics,
    )
    import sys
    import yaml

    started_at = time.perf_counter()
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available, but --device is set to 'cuda'.")
    load_env_file()
    config_path = resolve_config_path(args)
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if args.output_dir is not None:
        config.setdefault("logging", {})["output_dir"] = str(args.output_dir)
    if args.smoke:
        config["num_seeds"] = 1
        config["base_seed"] = config.get("base_seed", 42)
        config["train"]["epochs"] = 2
        variants = config["model"].get("variants") or [{"name": config["experiment"]["name"]}]
        config["model"]["variants"] = [variants[0]]
        config.setdefault("logging", {})["backend"] = "none"

    output_dir = get_output_dir(config)
    log_path = setup_logging(output_dir)
    LOGGER.info("Logging to %s", log_path)
    LOGGER.info("Command: %s", " ".join(sys.argv))
    LOGGER.info("Loaded config from %s", config_path)
    LOGGER.info("Dispatching to dual-checkpoint training scheme.")

    device = torch.device(args.device)
    seeds = experiment_seeds(config)
    variants = config["model"].get("variants") or [{"name": config["experiment"]["name"]}]
    experiment_summary = format_experiment_summary(config, variants)
    LOGGER.info(
        "Starting dual-checkpoint sweep experiment=%s variants=%s epochs=%s",
        config["experiment"]["name"],
        [variant["name"] for variant in variants],
        config["train"]["epochs"],
    )
    start_notification = (
        "Experiment started (dual_checkpoint)\n"
        "------------------\n"
        f"{experiment_summary}\n"
        f"Device: {device}\n"
        f"Command: {' '.join(sys.argv)}"
    )

    history: list[dict[str, Any]] = []
    wandb_urls: list[str] = []
    try:
        total_runs = len(seeds) * len(variants)
        run_index = 0
        for seed in seeds:
            for variant in variants:
                run_index += 1
                seed_everything(seed)
                train_loader, test_loader = build_loaders(config, variant)
                LOGGER.info(
                    "Sweep progress run=%s/%s seed=%s variant=%s",
                    run_index,
                    total_runs,
                    seed,
                    variant["name"],
                )
                variant_history, wandb_url = train_variant_dual_checkpoint(
                    config,
                    variant,
                    seed,
                    train_loader,
                    test_loader,
                    device,
                    start_notification=start_notification if run_index == 1 else None,
                )
                history.extend(variant_history)
                if wandb_url is not None:
                    wandb_urls.append(wandb_url)
        metrics_path = write_metrics(config, history)
    except BaseException as exc:
        elapsed = time.perf_counter() - started_at
        LOGGER.exception("Dual-checkpoint sweep failed after %s", format_duration(elapsed))
        send_telegram(
            "Experiment failed (dual_checkpoint)\n"
            "-----------------\n"
            f"{experiment_summary}\n"
            f"Elapsed: {format_duration(elapsed)}\n"
            f"Error: {type(exc).__name__}: {exc}",
            config,
        )
        raise

    elapsed = time.perf_counter() - started_at
    LOGGER.info(
        "Finished dual-checkpoint sweep records=%s metrics_path=%s elapsed_sec=%.2f",
        len(history),
        metrics_path,
        elapsed,
    )
    send_telegram(
        "Experiment finished (dual_checkpoint)\n"
        "-------------------\n"
        f"{experiment_summary}\n"
        f"{format_final_metrics(history)}\n"
        f"Metrics file: {metrics_path}\n"
        f"W&B: {', '.join(wandb_urls) if wandb_urls else 'not available'}\n"
        f"Elapsed: {format_duration(elapsed)}",
        config,
    )


if __name__ == "__main__":
    main()
