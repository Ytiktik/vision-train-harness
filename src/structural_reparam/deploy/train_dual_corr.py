"""Dual-model training with cross-model correlation-based gradient reweighting.

Two independent BN models (γ=1 frozen, β=0 frozen, FC frozen) are trained on
the same batches — only the linear weight w is learned per model.
Per-example weights w_i = exp(β·ρ_i) / mean_j(exp(β·ρ_j)) reweight each
model's per-example loss before backward; ρ is computed stop-grad.

corr_mode = 'activation':
    ρ_i = Pearson(w_A·x_i, w_B·x_i) over the feature dim (pre-BN values).

corr_mode = 'gradient':
    ρ_i = Pearson(∂L/∂(w_A·x_i), ∂L/∂(w_B·x_i)) over the feature dim.
    Captured via retain_grad on pre-BN tensors + a probe backward with uniform
    weights, then a real backward with the corr-weighted loss.

β=0 recovers uniform weights (independent training baseline).
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class _DualCorrModel(nn.Module):
    """Linear(bias=False) → BN(γ=1,β=0 frozen) → ReLU → FC(frozen).

    Only linear.weight is trained. Returns (logits, pre_bn).
    """

    def __init__(self, in_features: int, width: int, num_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, width, bias=False)
        self.bn = nn.BatchNorm1d(width)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(width, num_classes)

        nn.init.kaiming_normal_(self.linear.weight, mode="fan_in", nonlinearity="relu")
        nn.init.ones_(self.bn.weight)
        nn.init.zeros_(self.bn.bias)
        self.bn.weight.requires_grad_(False)
        self.bn.bias.requires_grad_(False)
        self.fc.weight.requires_grad_(False)
        self.fc.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.flatten(1)
        pre_bn = self.linear(x)
        h = self.relu(self.bn(pre_bn))
        return self.fc(h), pre_bn


# ---------------------------------------------------------------------------
# Correlation helpers
# ---------------------------------------------------------------------------

def _pearson(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-example Pearson correlation. a, b: (N, F) → (N,)."""
    a_c = a - a.mean(dim=1, keepdim=True)
    b_c = b - b.mean(dim=1, keepdim=True)
    num = (a_c * b_c).sum(dim=1)
    denom = a_c.norm(dim=1) * b_c.norm(dim=1)
    return torch.where(denom > eps, num / denom, torch.zeros_like(num))


# ---------------------------------------------------------------------------
# Mechanistic probe
# ---------------------------------------------------------------------------

class _DualMechanisticProbe:
    """Per-epoch diagnostics for both models in a dual-corr run.

    Runs one balanced-batch forward+backward per model per epoch.
    Reports per-model: w_grad_norm, w_norm, sigma (mean pre-BN batch std),
    fc_input_batch_var_mean, dead_relu_frac, class_feature_separation.
    Metrics are prefixed mech_A/ and mech_B/.
    Grad state is snapshotted and restored so the probe doesn't pollute
    the next training step.
    """

    def __init__(
        self,
        model_A: nn.Module,
        model_B: nn.Module,
        balanced_loader: DataLoader,
        device: torch.device,
    ) -> None:
        self.model_A = model_A
        self.model_B = model_B
        self.balanced_loader = balanced_loader
        self._iter = iter(balanced_loader)
        self.device = device
        self._criterion = nn.CrossEntropyLoss()

    def _next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            return next(self._iter)
        except StopIteration:
            self._iter = iter(self.balanced_loader)
            return next(self._iter)

    def _probe_one(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
        device = self.device
        grad_snap = [(p, p.grad.detach().clone() if p.grad is not None else None)
                     for p in model.parameters()]

        model.train()
        for p in model.parameters():
            p.grad = None

        # Capture pre-BN sigma via hook on linear output.
        captured_pre: list[torch.Tensor] = []
        handle = model.linear.register_forward_hook(
            lambda m, inp, out: captured_pre.append(out.detach())
        )
        logits, _ = model(x)
        loss = self._criterion(logits, y)
        loss.backward()
        handle.remove()

        w_grad_norm = (
            model.linear.weight.grad.detach().norm().item()
            if model.linear.weight.grad is not None else 0.0
        )
        w_norm = model.linear.weight.detach().norm().item()
        sigma = captured_pre[0].std(dim=0, unbiased=False).mean().item() if captured_pre else float("nan")

        # Restore grads.
        for p, g in grad_snap:
            p.grad = g.clone() if g is not None else None

        # Eval-mode pass for activation diagnostics.
        model.eval()
        fc_inputs: list[torch.Tensor] = []
        h_handle = model.fc.register_forward_hook(
            lambda m, inp, out: fc_inputs.append(inp[0].detach())
        )
        with torch.no_grad():
            model(x)
        h_handle.remove()

        h = fc_inputs[0]  # (N, width)
        fc_var = h.var(dim=0, unbiased=False).mean().item()
        dead_relu = (h <= 0).all(dim=0).float().mean().item()

        classes = torch.unique(y)
        class_means: list[torch.Tensor] = []
        within_acc = torch.zeros((), device=device)
        count = 0
        for c in classes:
            mask = y == c
            n_c = int(mask.sum().item())
            if n_c < 2:
                continue
            h_c = h[mask]
            class_means.append(h_c.mean(dim=0))
            within_acc += h_c.var(dim=0, unbiased=False).mean() * n_c
            count += n_c
        if class_means and count > 0:
            within_var = (within_acc / count).item()
            between_var = torch.stack(class_means).var(dim=0, unbiased=False).mean().item()
            separation = between_var / (within_var + 1e-12)
        else:
            separation = float("nan")

        model.train()
        return {
            "w_grad_norm": w_grad_norm,
            "w_norm": w_norm,
            "sigma": sigma,
            "fc_input_batch_var_mean": fc_var,
            "dead_relu_frac": dead_relu,
            "class_feature_separation": separation,
        }

    def epoch_stats(self) -> dict[str, float]:
        x, y = self._next_batch()
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)
        stats_A = self._probe_one(self.model_A, x, y)
        stats_B = self._probe_one(self.model_B, x, y)
        out: dict[str, float] = {}
        for k, v in stats_A.items():
            out[f"mech_A/{k}"] = v
        for k, v in stats_B.items():
            out[f"mech_B/{k}"] = v
        return out


# ---------------------------------------------------------------------------
# Training / eval loops
# ---------------------------------------------------------------------------

@torch.no_grad()
def _eval_loader(
    model_A: nn.Module,
    model_B: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model_A.eval()
    model_B.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss_A = total_loss_B = 0.0
    total_correct_A = total_correct_B = 0
    total_examples = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits_A, _ = model_A(images)
        logits_B, _ = model_B(images)
        total_loss_A += criterion(logits_A, targets).item() * targets.size(0)
        total_loss_B += criterion(logits_B, targets).item() * targets.size(0)
        total_correct_A += (logits_A.argmax(1) == targets).sum().item()
        total_correct_B += (logits_B.argmax(1) == targets).sum().item()
        total_examples += targets.size(0)
    return {
        "test_loss_A": total_loss_A / total_examples,
        "test_loss_B": total_loss_B / total_examples,
        "test_accuracy_A": total_correct_A / total_examples,
        "test_accuracy_B": total_correct_B / total_examples,
    }


def run_dual_corr_epoch(
    model_A: nn.Module,
    model_B: nn.Module,
    opt_A: torch.optim.Optimizer,
    opt_B: torch.optim.Optimizer,
    loader: DataLoader,
    device: torch.device,
    beta: float,
    corr_mode: str,
) -> dict[str, float]:
    model_A.train()
    model_B.train()
    criterion = nn.CrossEntropyLoss(reduction="none")

    total_loss_A = total_loss_B = 0.0
    total_correct_A = total_correct_B = 0
    total_examples = 0
    rho_sum = weight_std_sum = 0.0
    n_steps = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        opt_A.zero_grad(set_to_none=True)
        opt_B.zero_grad(set_to_none=True)

        if corr_mode == "activation":
            logits_A, pre_A = model_A(images)
            logits_B, pre_B = model_B(images)
            losses_A = criterion(logits_A, targets)
            losses_B = criterion(logits_B, targets)

            with torch.no_grad():
                rho = _pearson(pre_A, pre_B)
                weights = torch.exp(beta * rho)
                weights = weights / weights.mean().clamp(min=1e-12)

            loss = (weights * losses_A).mean() + (weights * losses_B).mean()
            loss.backward()

        elif corr_mode == "gradient":
            # Forward with retain_grad so we can read ∂L/∂pre_bn after backward.
            logits_A, pre_A = model_A(images)
            logits_B, pre_B = model_B(images)
            pre_A.retain_grad()
            pre_B.retain_grad()
            losses_A = criterion(logits_A, targets)
            losses_B = criterion(logits_B, targets)

            # Probe backward: uniform weights, retain graph for the real backward.
            probe_loss = losses_A.mean() + losses_B.mean()
            probe_loss.backward(retain_graph=True)

            with torch.no_grad():
                rho = _pearson(pre_A.grad, pre_B.grad)
                weights = torch.exp(beta * rho)
                weights = weights / weights.mean().clamp(min=1e-12)

            # Clear probe gradients from parameters, then real weighted backward.
            opt_A.zero_grad(set_to_none=True)
            opt_B.zero_grad(set_to_none=True)
            loss = (weights * losses_A).mean() + (weights * losses_B).mean()
            loss.backward()

        else:
            raise ValueError(f"Unknown corr_mode={corr_mode!r}. Expected 'activation' or 'gradient'.")

        opt_A.step()
        opt_B.step()

        with torch.no_grad():
            N = targets.size(0)
            total_loss_A += losses_A.detach().mean().item() * N
            total_loss_B += losses_B.detach().mean().item() * N
            total_correct_A += (logits_A.detach().argmax(1) == targets).sum().item()
            total_correct_B += (logits_B.detach().argmax(1) == targets).sum().item()
            total_examples += N
            rho_sum += rho.mean().item()
            weight_std_sum += weights.std().item()
            n_steps += 1

    return {
        "train_loss_A": total_loss_A / total_examples,
        "train_loss_B": total_loss_B / total_examples,
        "train_accuracy_A": total_correct_A / total_examples,
        "train_accuracy_B": total_correct_B / total_examples,
        "mean_rho": rho_sum / max(n_steps, 1),
        "weight_std": weight_std_sum / max(n_steps, 1),
    }


# ---------------------------------------------------------------------------
# Per-variant training
# ---------------------------------------------------------------------------

def train_variant_dual_corr(
    config: dict[str, Any],
    variant: dict[str, Any],
    seed: int,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    start_notification: str | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    from structural_reparam.deploy.train import (
        get_wandb_url,
        seed_everything,
        send_telegram,
        start_wandb,
    )

    variant_name = variant["name"]
    dual_cfg = config.get("dual_corr", {})
    corr_mode = dual_cfg["corr_mode"]
    in_features = int(dual_cfg.get("in_features", 784))
    width = int(dual_cfg.get("width", 32))
    num_classes = int(dual_cfg.get("num_classes", 10))
    seed_offset = int(dual_cfg.get("seed_offset", 10000))
    beta = float(variant.get("args", {}).get("beta", 0.0))

    LOGGER.info(
        "Starting dual-corr mode=%s beta=%.2f seed=%s variant=%s",
        corr_mode, beta, seed, variant_name,
    )
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    seed_everything(seed)
    model_A = _DualCorrModel(in_features, width, num_classes).to(device)
    seed_everything(seed + seed_offset)
    model_B = _DualCorrModel(in_features, width, num_classes).to(device)

    train_cfg = config["train"]
    lr = float(train_cfg["lr"])
    momentum = float(train_cfg.get("momentum", 0.9))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))

    opt_A = torch.optim.SGD(
        [p for p in model_A.parameters() if p.requires_grad],
        lr=lr, momentum=momentum, weight_decay=weight_decay,
    )
    opt_B = torch.optim.SGD(
        [p for p in model_B.parameters() if p.requires_grad],
        lr=lr, momentum=momentum, weight_decay=weight_decay,
    )

    total_epochs = int(train_cfg["epochs"])
    scheduler_A = torch.optim.lr_scheduler.CosineAnnealingLR(opt_A, T_max=total_epochs)
    scheduler_B = torch.optim.lr_scheduler.CosineAnnealingLR(opt_B, T_max=total_epochs)

    # Mechanistic probe: balanced shuffled loader for diagnostics.
    from structural_reparam.deploy.train import import_target, resolve_repo_path
    balanced_args = dict(config["dataset"].get("args", {}))
    balanced_args["batch_mode"] = "shuffled"
    balanced_args.pop("classes_per_batch", None)
    if "data_dir" in balanced_args:
        balanced_args["data_dir"] = resolve_repo_path(balanced_args["data_dir"])
    balanced_loader, _ = import_target(config["dataset"]["target"])(**balanced_args)
    mech_probe = _DualMechanisticProbe(model_A, model_B, balanced_loader, device)

    run = start_wandb(config, variant, seed)
    wandb_url = get_wandb_url(run)
    if start_notification is not None:
        send_telegram(start_notification + f"\nW&B: {wandb_url or 'not available'}", config)

    history: list[dict[str, Any]] = []
    started_at = time.perf_counter()

    for epoch in range(1, total_epochs + 1):
        train_stats = run_dual_corr_epoch(
            model_A, model_B, opt_A, opt_B,
            train_loader, device, beta, corr_mode,
        )
        test_stats = _eval_loader(model_A, model_B, test_loader, device)
        mech_stats = mech_probe.epoch_stats()
        scheduler_A.step()
        scheduler_B.step()

        record: dict[str, Any] = {
            "epoch": epoch,
            "seed": seed,
            "variant": variant_name,
            **train_stats,
            **test_stats,
            **mech_stats,
        }
        history.append(record)
        if run is not None:
            run.log(record, step=epoch)
        LOGGER.info(
            "epoch %s/%s variant=%s acc_A=%.4f acc_B=%.4f rho=%.4f sep_A=%.4f sep_B=%.4f",
            epoch, total_epochs, variant_name,
            test_stats["test_accuracy_A"], test_stats["test_accuracy_B"],
            train_stats["mean_rho"],
            mech_stats["mech_A/class_feature_separation"],
            mech_stats["mech_B/class_feature_separation"],
        )
        print(json.dumps(record), flush=True)

    if run is not None:
        run.finish()
    LOGGER.info(
        "Finished dual-corr variant=%s epochs=%s elapsed_sec=%.2f",
        variant_name, total_epochs, time.perf_counter() - started_at,
    )
    return history, wandb_url


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """python -m structural_reparam.deploy.train_dual_corr --config <path>"""
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
    LOGGER.info(
        "Dispatching to dual-corr training mode=%s",
        config.get("dual_corr", {}).get("corr_mode"),
    )

    device = torch.device(args.device)
    seeds = experiment_seeds(config)
    variants = config["model"].get("variants") or [{"name": config["experiment"]["name"]}]
    experiment_summary = format_experiment_summary(config, variants)
    LOGGER.info(
        "Starting dual-corr sweep experiment=%s variants=%s epochs=%s",
        config["experiment"]["name"],
        [v["name"] for v in variants],
        config["train"]["epochs"],
    )
    start_notification = (
        "Experiment started (dual_corr)\n"
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
                    run_index, total_runs, seed, variant["name"],
                )
                variant_history, wandb_url = train_variant_dual_corr(
                    config, variant, seed, train_loader, test_loader, device,
                    start_notification=start_notification if run_index == 1 else None,
                )
                history.extend(variant_history)
                if wandb_url is not None:
                    wandb_urls.append(wandb_url)
        metrics_path = write_metrics(config, history)
    except BaseException as exc:
        elapsed = time.perf_counter() - started_at
        LOGGER.exception("Dual-corr sweep failed after %s", format_duration(elapsed))
        send_telegram(
            "Experiment failed (dual_corr)\n"
            "-----------------\n"
            f"{experiment_summary}\n"
            f"Elapsed: {format_duration(elapsed)}\n"
            f"Error: {type(exc).__name__}: {exc}",
            config,
        )
        raise

    elapsed = time.perf_counter() - started_at
    LOGGER.info(
        "Finished dual-corr sweep records=%s metrics_path=%s elapsed_sec=%.2f",
        len(history), metrics_path, elapsed,
    )
    send_telegram(
        "Experiment finished (dual_corr)\n"
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
