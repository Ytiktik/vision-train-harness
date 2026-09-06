"""Generic config-driven training loop for controlled experiments."""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import math
import os
import random
import socket
import sys
import time
from pathlib import Path
from typing import Any
from urllib import parse, request

import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import yaml

from torch.nn import functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
LOGGER = logging.getLogger(__name__)
TELEGRAM_TIMEOUT_SEC = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a structural reparameterization experiment."
    )
    parser.add_argument("--experiment", help="Experiment name. Resolves to configs/<name>.yaml.")
    parser.add_argument("--config", type=Path, help="Path to a YAML experiment config.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override logging.output_dir from the config.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke-test mode: 1 seed, 2 epochs, first variant only, no W&B.",
    )
    return parser.parse_args()


def resolve_config_path(args: argparse.Namespace) -> Path:
    if args.config is not None:
        return args.config
    if args.experiment is None:
        raise SystemExit("Pass --experiment <name> or --config <path>.")
    return REPO_ROOT / "configs" / f"{args.experiment}.yaml"


def get_output_dir(config: dict[str, Any]) -> Path:
    return REPO_ROOT / config.get("logging", {}).get("output_dir", "outputs")


def setup_logging(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    return log_path


def format_duration(seconds: float) -> str:
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def telegram_enabled(config: dict[str, Any]) -> bool:
    logging_config = config.get("logging", {})
    return bool(logging_config.get("telegram", True))


def send_telegram(message: str, config: dict[str, Any]) -> None:
    if not telegram_enabled(config):
        LOGGER.info("Telegram notification skipped because logging.telegram is disabled")
        return

    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        LOGGER.info(
            "Telegram notification skipped because TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is unset"
        )
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = parse.urlencode({"chat_id": chat_id, "text": message}).encode("utf-8")
    try:
        with request.urlopen(url, data=payload, timeout=TELEGRAM_TIMEOUT_SEC) as response:
            if response.status >= 400:
                body = response.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"Telegram API error {response.status}: {body}")
    except Exception as exc:
        LOGGER.warning("Failed to send Telegram notification: %s", exc)
        return

    LOGGER.info("Sent Telegram notification")


def format_experiment_summary(config: dict[str, Any], variants: list[dict[str, Any]]) -> str:
    variant_names = ", ".join(variant["name"] for variant in variants)
    return (
        f"Experiment: {config['experiment']['name']}\n"
        f"Variants: {variant_names}\n"
        f"Epochs: {config['train']['epochs']}\n"
        f"Seeds: {', '.join(str(seed) for seed in experiment_seeds(config))}"
    )


def get_wandb_url(run: Any) -> str | None:
    if run is None:
        return None
    get_url = getattr(run, "get_url", None)
    if callable(get_url):
        url = get_url()
        if url:
            return str(url)
    url = getattr(run, "url", None)
    if url:
        return str(url)
    return None


def format_metric_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def format_final_metrics(history: list[dict[str, Any]]) -> str:
    if not history:
        return "Final metrics: none"

    latest_by_variant: dict[tuple[int, str], dict[str, Any]] = {}
    for record in history:
        latest_by_variant[(int(record["seed"]), str(record["variant"]))] = record

    lines = ["Final metrics:"]
    for (seed, variant), record in latest_by_variant.items():
        metrics = {
            key: value
            for key, value in record.items()
            if key not in {"epoch", "seed", "variant"}
        }
        formatted_metrics = ", ".join(
            f"{key}={format_metric_value(value)}" for key, value in metrics.items()
        )
        lines.append(f"seed={seed} {variant}: epoch={record['epoch']}, {formatted_metrics}")
    return "\n".join(lines)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def experiment_seeds(config: dict[str, Any]) -> list[int]:
    base_seed = int(config.get("base_seed", config.get("seed", 0)))
    num_seeds = int(config.get("num_seeds", 1))
    if num_seeds < 1:
        raise ValueError(f"num_seeds must be >= 1, got {num_seeds}")
    return [base_seed + offset for offset in range(num_seeds)]


def load_env_file(path: Path = REPO_ROOT / ".env") -> None:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def import_target(target: str) -> Any:
    module_name, attr_name = target.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def resolve_repo_path(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    path = Path(value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def build_loaders(
    config: dict[str, Any], variant: dict[str, Any] | None = None
) -> tuple[DataLoader, DataLoader]:
    dataset_config = config["dataset"]
    dataset_args = dict(dataset_config.get("args", {}))
    variant_dataset_args = (variant or {}).get("dataset", {}).get("args", {})
    dataset_args.update(variant_dataset_args)
    if "data_dir" in dataset_args:
        dataset_args["data_dir"] = resolve_repo_path(dataset_args["data_dir"])
    return import_target(dataset_config["target"])(**dataset_args)


def build_model(config: dict[str, Any], variant: dict[str, Any]) -> nn.Module:
    model_config = config["model"]
    model_args = dict(model_config.get("args", {}))
    model_args.update(variant.get("args", {}))
    depth_mult = model_args.pop("depth_mult", None)
    if depth_mult is not None:
        base_blocks = model_config.get("args", {}).get("stage_blocks", [1, 2, 2, 2, 1])
        model_args["stage_blocks"] = [max(1, round(b * depth_mult)) for b in base_blocks]
    return import_target(model_config["target"])(**model_args)


OPTIMIZER_ARG_KEYS = {
    "lr",
    "momentum",
    "dampening",
    "weight_decay",
    "nesterov",
    "betas",
    "eps",
    "amsgrad",
    "foreach",
    "maximize",
    "capturable",
    "differentiable",
    "fused",
}


def build_optimizer(
    config: dict[str, Any], model: nn.Module, variant: dict[str, Any] | None = None
) -> torch.optim.Optimizer:
    train_config = config["train"]
    variant_train = (variant or {}).get("train", {})

    optimizer_config = variant_train.get("optimizer", train_config.get("optimizer", "sgd"))
    if isinstance(optimizer_config, str):
        optimizer_target = "structural_reparam.optim.build_optimizer"
        optimizer_args = {"name": optimizer_config}
    else:
        optimizer_target = optimizer_config["target"]
        optimizer_args = dict(optimizer_config.get("args", {}))

    for key in OPTIMIZER_ARG_KEYS:
        if key in train_config:
            optimizer_args.setdefault(key, train_config[key])
        if key in variant_train:
            optimizer_args[key] = variant_train[key]

    import inspect
    fn = import_target(optimizer_target)
    if "model" in inspect.signature(fn).parameters:
        return fn(model.parameters(), model=model, **optimizer_args)
    return fn(model.parameters(), **optimizer_args)


def build_scheduler(
    config: dict[str, Any], optimizer: torch.optim.Optimizer
) -> torch.optim.lr_scheduler.LRScheduler | None:
    train_config = config["train"]
    scheduler = train_config.get("scheduler")
    if scheduler is None:
        return None
    epochs = int(train_config["epochs"])
    # Linear LR warmup is opt-in: a run warms up only when ``train.warmup_epochs``
    # is set (>0). Without the key behaviour is unchanged, so existing configs are
    # unaffected. Needed for large-batch / high-LR runs where the un-warmed first
    # epochs would otherwise diverge (linear-scaling rule, Goyal et al. 2017).
    warmup_epochs = int(train_config.get("warmup_epochs", 0) or 0)
    if scheduler == "cosine":
        if warmup_epochs > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1e-3,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, epochs - warmup_epochs)
            )
            return torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
            )
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    raise ValueError(f"Unsupported scheduler: {scheduler}")


def weight_decay_bounds(
    config: dict[str, Any], variant: dict[str, Any] | None = None
) -> tuple[float, float] | None:
    """(wd_init, wd_final) for cosine weight-decay annealing, or None if disabled.

    Annealing is opt-in: a run anneals WD only when ``train.weight_decay_final``
    is set (variant override wins). MobileOne's recipe anneals 1e-4 -> 1e-5 over
    training; without this key WD stays constant, so existing experiments are
    unaffected.
    """
    train_config = config["train"]
    variant_train = (variant or {}).get("train", {})
    wd_init = variant_train.get("weight_decay", train_config.get("weight_decay"))
    wd_final = variant_train.get("weight_decay_final", train_config.get("weight_decay_final"))
    if wd_final is None or wd_init is None:
        return None
    return float(wd_init), float(wd_final)


def cosine_weight_decay(wd_init: float, wd_final: float, epoch: int, total_epochs: int) -> float:
    """Cosine-annealed WD for a 1-based ``epoch``, mirroring CosineAnnealingLR.

    Epoch 1 (t=0) returns ``wd_init``; the value decays toward ``wd_final`` with
    the same half-cosine curve the LR scheduler uses (T_max = total_epochs).
    """
    if total_epochs <= 1:
        return wd_init
    cos = (1 + math.cos(math.pi * (epoch - 1) / total_epochs)) / 2
    return wd_final + (wd_init - wd_final) * cos


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    custom_l2: float = 0.0,
    distance_lambda: float = 0.0,
    num_classes: int | None = None,
    coarse_map: torch.Tensor | None = None,
    max_grad_norm: float | None = None,
    step_callback=None,
    distributed: bool = False,
    amp_dtype: torch.dtype | None = None,
    heartbeat_every: int = 0,
    channels_last: bool = False,
) -> dict[str, float]:
    is_train = optimizer is not None
    phase = "train" if is_train else "eval"
    hb_start = time.perf_counter()
    hb_prev = hb_start
    model.train(is_train)
    # DataParallel/DDP hide the model's custom methods behind .module; unwrap so
    # feature-detection (hasattr) and the calls below hit the real model.
    core = model.module if isinstance(model, (nn.DataParallel, DDP)) else model
    apply_custom_l2 = is_train and custom_l2 > 0.0 and hasattr(core, "custom_l2")
    apply_distance = is_train and distance_lambda > 0.0 and hasattr(core, "distance_loss")

    total_loss = torch.tensor(0.0, device=device)
    total_correct = torch.tensor(0, device=device)
    total_examples = 0
    if num_classes is not None:
        loss_sum = torch.zeros(num_classes, device=device)
        class_correct = torch.zeros(num_classes, device=device)
        class_counts = torch.zeros(num_classes, device=device)
        _ones_buf = torch.ones(loader.batch_size or 1, dtype=torch.float, device=device)
    else:
        loss_sum = None
        class_correct = None
        class_counts = None
        _ones_buf = None

    step = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        if channels_last:
            # NHWC layout: faster cuDNN conv kernels + better-coalesced BN memory
            # access (the bottleneck for MobileOne's multi-branch training graph).
            images = images.contiguous(memory_format=torch.channels_last)
        targets = targets.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train), torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None
        ):
            logits = model(images)
            loss = criterion(logits, targets)
            if apply_custom_l2:
                loss = loss + custom_l2 * core.custom_l2()
            if apply_distance:
                loss = loss + distance_lambda * core.distance_loss()

        if is_train:
            loss.backward()
            if max_grad_norm is not None:
                for p in model.parameters():
                    if p.grad is not None:
                        torch.nn.utils.clip_grad_norm_([p], max_grad_norm)
            optimizer.step()

        total_loss += loss.detach() * targets.size(0)
        detached_logits = logits.detach()
        preds = detached_logits.argmax(dim=1)
        correct = preds == targets
        total_correct += correct.sum()
        total_examples += targets.size(0)

        if loss_sum is not None and class_correct is not None and class_counts is not None:
            losses = F.cross_entropy(detached_logits, targets, reduction="none")
            loss_sum.scatter_add_(0, targets, losses)
            class_correct.scatter_add_(0, targets, correct.float())
            class_counts.scatter_add_(0, targets, _ones_buf[:targets.size(0)])

        step += 1
        if heartbeat_every and step % heartbeat_every == 0:
            # Sync so timing reflects real GPU progress; if the GPU is hung on a
            # CUDA-graph replay this sync blocks here, and the previous heartbeat
            # pinpoints the last good step. Cheap at this cadence.
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            now = time.perf_counter()
            alloc = torch.cuda.memory_allocated(device) / 1e6 if torch.cuda.is_available() else 0.0
            reserved = torch.cuda.memory_reserved(device) / 1e6 if torch.cuda.is_available() else 0.0
            LOGGER.info(
                "heartbeat phase=%s step=%d window=%.1fs total=%.1fs mem_alloc=%.0fMB mem_reserved=%.0fMB",
                phase, step, now - hb_prev, now - hb_start, alloc, reserved,
            )
            hb_prev = now

    if distributed and dist.is_available() and dist.is_initialized():
        # Each rank only saw its 1/world_size shard; sum the running totals so the
        # reported loss/accuracy are over the full (global) set of examples.
        total_examples_t = torch.tensor(float(total_examples), device=device)
        dist.all_reduce(total_loss)
        dist.all_reduce(total_correct)
        dist.all_reduce(total_examples_t)
        total_examples = int(total_examples_t.item())
        if loss_sum is not None and class_correct is not None and class_counts is not None:
            dist.all_reduce(loss_sum)
            dist.all_reduce(class_correct)
            dist.all_reduce(class_counts)

    stats = {
        "loss": (total_loss / total_examples).item(),
        "accuracy": (total_correct / total_examples).item(),
    }
    if loss_sum is not None and class_correct is not None and class_counts is not None:
        stats.update(_summarize_classwise(loss_sum, class_correct, class_counts))
        if coarse_map is not None:
            stats.update(_summarize_superclasswise(loss_sum, class_correct, class_counts, coarse_map))
    return stats


def run_train_epoch_with_branch_probe(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    probe,
    epoch: int,
    start_global_step: int,
    max_grad_norm: float | None = None,
) -> tuple[dict[str, float], int]:
    """Training epoch instrumented with per-step BranchProbe.

    The probe arms BN forward_pre_hooks once per step (auto-removing after
    firing), runs an observational measurement after backward, then the real
    optimizer.step() proceeds as normal.
    """
    model.train()
    total_loss = torch.tensor(0.0, device=device)
    total_correct = torch.tensor(0, device=device)
    total_examples = 0
    global_step = start_global_step

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        probe.arm_hooks()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()

        if max_grad_norm is not None:
            for p in model.parameters():
                if p.grad is not None:
                    torch.nn.utils.clip_grad_norm_([p], max_grad_norm)

        probe.step(global_step=global_step, epoch=epoch)
        optimizer.step()

        total_loss += loss.detach() * targets.size(0)
        preds = logits.detach().argmax(dim=1)
        total_correct += (preds == targets).sum()
        total_examples += targets.size(0)
        global_step += 1

    stats = {
        "loss": (total_loss / total_examples).item(),
        "accuracy": (total_correct / total_examples).item(),
    }
    return stats, global_step


def _build_train_eval_loader(
    config: dict[str, Any], variant: dict[str, Any] | None
) -> DataLoader:
    dataset_config = config["dataset"]
    dataset_name = dataset_config.get("name")
    dataset_args = dict(dataset_config.get("args", {}))
    dataset_args.update((variant or {}).get("dataset", {}).get("args", {}))
    if "data_dir" in dataset_args:
        dataset_args["data_dir"] = resolve_repo_path(dataset_args["data_dir"])
    batch_size = int(dataset_args.pop("batch_size")) if "batch_size" in dataset_args else 256
    if dataset_name == "cifar100":
        from structural_reparam.data.cifar100 import build_train_eval_loader
        kwargs = {k: dataset_args[k] for k in ("data_dir", "num_workers", "persistent_workers", "class_subset", "superclass_pair", "subset_per_class", "label_noise", "noise_seed", "label_noise_mode") if k in dataset_args}
        return build_train_eval_loader(batch_size=batch_size, **kwargs)
    if dataset_name == "cifar10":
        from structural_reparam.data.cifar10 import build_train_eval_loader
        kwargs = {k: dataset_args[k] for k in ("data_dir", "num_workers", "persistent_workers", "keep_classes", "subset_per_class", "noise_seed") if k in dataset_args}
        return build_train_eval_loader(batch_size=batch_size, **kwargs)
    if dataset_name == "tiny_imagenet":
        from structural_reparam.data.tiny_imagenet import build_train_eval_loader
        kwargs = {k: dataset_args[k] for k in ("data_dir", "num_workers", "num_classes", "keep_classes") if k in dataset_args}
        return build_train_eval_loader(batch_size=batch_size, **kwargs)
    raise ValueError(
        f"train-eval loader not implemented for dataset {dataset_name!r}"
    )


def _resolve_coarse_map(
    config: dict[str, Any], num_classes: int, device: torch.device
) -> torch.Tensor:
    dataset_name = config.get("dataset", {}).get("name")
    if dataset_name == "cifar100":
        from structural_reparam.data.cifar100 import CIFAR100_FINE_TO_COARSE
        if num_classes != len(CIFAR100_FINE_TO_COARSE):
            raise ValueError(
                f"cifar100 coarse map expects num_classes={len(CIFAR100_FINE_TO_COARSE)}, "
                f"got {num_classes}"
            )
        return torch.tensor(CIFAR100_FINE_TO_COARSE, dtype=torch.long, device=device)
    raise ValueError(
        f"super-class metrics requested but no coarse map known for dataset {dataset_name!r}"
    )


def _summarize_classwise(
    loss_sum: torch.Tensor,
    correct: torch.Tensor,
    counts: torch.Tensor,
) -> dict[str, float]:
    present = counts > 0
    if not present.any():
        raise ValueError("Cannot compute classwise metrics on an empty loader.")

    class_loss = loss_sum[present] / counts[present]
    class_acc = correct[present] / counts[present]
    return {
        "class_loss_min": class_loss.min().item(),
        "class_loss_max": class_loss.max().item(),
        "class_loss_std": class_loss.std(unbiased=False).item(),
        "class_acc_min": class_acc.min().item(),
        "class_acc_max": class_acc.max().item(),
        "class_acc_std": class_acc.std(unbiased=False).item(),
    }


def _summarize_superclasswise(
    loss_sum: torch.Tensor,
    correct: torch.Tensor,
    counts: torch.Tensor,
    coarse_map: torch.Tensor,
) -> dict[str, float]:
    if coarse_map.numel() != loss_sum.numel():
        raise ValueError(
            f"coarse_map length {coarse_map.numel()} must match num_classes {loss_sum.numel()}"
        )
    num_super = int(coarse_map.max().item()) + 1
    coarse_map = coarse_map.to(loss_sum.device, dtype=torch.long)
    super_loss_sum = torch.zeros(num_super, device=loss_sum.device).scatter_add_(0, coarse_map, loss_sum)
    super_correct = torch.zeros(num_super, device=loss_sum.device).scatter_add_(0, coarse_map, correct)
    super_counts = torch.zeros(num_super, device=loss_sum.device).scatter_add_(0, coarse_map, counts)
    base = _summarize_classwise(super_loss_sum, super_correct, super_counts)
    return {
        "super_loss_min": base["class_loss_min"],
        "super_loss_max": base["class_loss_max"],
        "super_loss_std": base["class_loss_std"],
        "super_acc_min": base["class_acc_min"],
        "super_acc_max": base["class_acc_max"],
        "super_acc_std": base["class_acc_std"],
    }


def classwise_stats(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    coarse_map: torch.Tensor | None = None,
    include_overall: bool = False,
) -> dict[str, float]:
    model.eval()
    loss_sum = torch.zeros(num_classes, device=device)
    correct = torch.zeros(num_classes, device=device)
    counts = torch.zeros(num_classes, device=device)

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            losses = criterion(logits, targets)
            preds = logits.argmax(dim=1)

            loss_sum.scatter_add_(0, targets, losses)
            correct.scatter_add_(0, targets, (preds == targets).float())
            counts.scatter_add_(0, targets, torch.ones_like(targets, dtype=torch.float))

    stats = _summarize_classwise(loss_sum, correct, counts)
    if coarse_map is not None:
        stats.update(_summarize_superclasswise(loss_sum, correct, counts, coarse_map))
    if include_overall:
        # Opt-in so the "accuracy"/"loss" keys cannot shadow the training-loop
        # test_accuracy/test_loss metrics when this feeds test_class_stats.
        total = counts.sum()
        stats["accuracy"] = (correct.sum() / total).item()
        stats["loss"] = (loss_sum.sum() / total).item()
    return stats


def select_metrics(
    configured_metrics: list[str],
    train_stats: dict[str, float],
    test_stats: dict[str, float],
    model: nn.Module,
    train_class_stats: dict[str, float] | None = None,
    test_class_stats: dict[str, float] | None = None,
    train_eval_class_stats: dict[str, float] | None = None,
) -> dict[str, float | int]:
    # test_stats is None on epochs where validation is skipped (eval_interval);
    # report NaN for the test metrics so the per-epoch schema stays consistent.
    available: dict[str, float | int] = {
        "train_loss": train_stats["loss"],
        "train_accuracy": train_stats["accuracy"],
        "test_loss": test_stats["loss"] if test_stats is not None else float("nan"),
        "test_accuracy": test_stats["accuracy"] if test_stats is not None else float("nan"),
        "param_count": sum(param.numel() for param in model.parameters()),
    }
    if train_class_stats is not None:
        available.update(
            {f"train_{name}": value for name, value in train_class_stats.items()}
        )
    if test_class_stats is not None:
        available.update(
            {f"test_{name}": value for name, value in test_class_stats.items()}
        )
    if train_eval_class_stats is not None:
        available.update(
            {f"train_eval_{name}": value for name, value in train_eval_class_stats.items()}
        )
    # NaN-fill train_eval_* metrics on epochs where the eval-train pass is skipped,
    # so the configured metric set stays consistent across epochs.
    for metric in configured_metrics:
        if metric.startswith("train_eval_") and metric not in available:
            available[metric] = float("nan")
    unknown = sorted(set(configured_metrics) - set(available))
    if unknown:
        raise ValueError(f"Unknown configured metrics: {unknown}")
    return {name: available[name] for name in configured_metrics}


def start_wandb(config: dict[str, Any], variant: dict[str, Any], seed: int):
    logging_config = config.get("logging", {})
    if logging_config.get("backend") != "wandb":
        LOGGER.info("W&B logging disabled for variant=%s", variant["name"])
        return None
    import wandb

    experiment = config["experiment"]["name"]
    run_name = f"{experiment}/{variant['name']}"
    if int(config.get("num_seeds", 1)) > 1:
        run_name = f"{run_name}/seed_{seed}"
    api_key = os.environ.get("WANDB_API_KEY")
    if api_key:
        wandb.login(key=api_key)

    init_kwargs = {
        "project": logging_config.get("project", "structural-reparam"),
        "name": run_name,
        "group": logging_config.get("group", experiment),
        "job_type": logging_config.get("job_type", "train"),
        "tags": config["experiment"].get("tags", []),
        "config": {**config, "seed": seed, "variant": variant},
    }
    optional_keys = ("entity", "mode")
    for key in optional_keys:
        if key in logging_config:
            init_kwargs[key] = logging_config[key]

    LOGGER.info(
        "Starting W&B run project=%s group=%s name=%s",
        init_kwargs["project"],
        init_kwargs["group"],
        init_kwargs["name"],
    )
    return wandb.init(
        **init_kwargs,
    )


def _distributed_train_loader(
    loader: DataLoader, rank: int, world_size: int
) -> tuple[DataLoader, DistributedSampler]:
    """Re-wrap a train loader with a DistributedSampler for DDP.

    Keeps the GLOBAL batch size fixed by giving each rank ``batch // world_size``
    samples per step (paper batch 256 -> 128/GPU on 2 GPUs). Reuses the source
    loader's worker/pin/prefetch settings.
    """
    global_batch = loader.batch_size or 256
    if global_batch % world_size != 0:
        raise ValueError(
            f"batch_size {global_batch} not divisible by world_size {world_size}; "
            "cannot keep the global batch fixed across ranks."
        )
    per_rank_batch = global_batch // world_size
    sampler = DistributedSampler(
        loader.dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
    )
    num_workers = loader.num_workers
    use_workers = num_workers > 0
    new_loader = DataLoader(
        loader.dataset,
        batch_size=per_rank_batch,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=loader.pin_memory,
        prefetch_factor=loader.prefetch_factor if use_workers else None,
        persistent_workers=loader.persistent_workers if use_workers else False,
        drop_last=True,
    )
    return new_loader, sampler


def _distributed_eval_loader(loader: DataLoader, rank: int, world_size: int) -> DataLoader:
    """Shard an eval loader across ranks with a strided, non-padding split.

    Rank r takes dataset indices ``r, r+world_size, r+2*world_size, ...`` so every
    sample is evaluated by exactly one rank (no duplication, unlike
    DistributedSampler's padding). Summing the per-rank totals via all-reduce then
    yields metrics identical to a single-process eval over the whole set.
    """
    from torch.utils.data import Subset

    dataset = loader.dataset
    indices = list(range(rank, len(dataset), world_size))
    num_workers = loader.num_workers
    use_workers = num_workers > 0
    return DataLoader(
        Subset(dataset, indices),
        batch_size=loader.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=loader.pin_memory,
        prefetch_factor=loader.prefetch_factor if use_workers else None,
        persistent_workers=loader.persistent_workers if use_workers else False,
        drop_last=False,
    )


def train_variant(
    config: dict[str, Any],
    variant: dict[str, Any],
    seed: int,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    start_notification: str | None = None,
    rank: int = 0,
    world_size: int = 1,
    is_main: bool = True,
) -> tuple[list[dict[str, Any]], str | None]:
    variant_name = variant["name"]
    distributed = world_size > 1
    LOGGER.info("Starting seed=%s variant=%s args=%s", seed, variant_name, variant.get("args", {}))
    # cudnn.benchmark autotunes (faster, non-deterministic) conv kernels; opt-in via
    # train.cudnn_benchmark for the fixed-resolution ImageNet runs. Default stays
    # deterministic so the reproducibility-sensitive CIFAR experiments are unchanged.
    cudnn_benchmark = bool(config["train"].get("cudnn_benchmark", False))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.deterministic = not cudnn_benchmark
    # Mixed precision (bf16 autocast): runs convs/matmuls on the A100 Tensor Cores.
    # bf16 keeps FP32's exponent range, so no GradScaler is needed. Opt-in via
    # train.amp; master weights/grads stay FP32 (only the forward math is bf16).
    amp_dtype = torch.bfloat16 if (config["train"].get("amp") and device.type == "cuda") else None
    if amp_dtype is not None:
        LOGGER.info("AMP enabled (bfloat16 autocast)")
    # channels_last (NHWC): a memory-layout change (full FP32) that speeds up the
    # memory-bound conv+BN traffic of MobileOne's multi-branch training graph.
    channels_last = bool(config["train"].get("channels_last", False)) and device.type == "cuda"
    seed_everything(seed)
    started_at = time.perf_counter()
    base_model = build_model(config, variant).to(device)
    if channels_last:
        base_model = base_model.to(memory_format=torch.channels_last)
        LOGGER.info("channels_last (NHWC) memory format enabled")
    # base_model stays the canonical handle for the optimizer, probes, and custom
    # methods; `model` is the (possibly DDP-wrapped) handle used for the forward
    # pass. Under DDP each rank runs one GPU in its own process (no GIL-bound
    # scatter/gather), and gradients are all-reduced across ranks.
    model = base_model
    if distributed:
        # gradient_as_bucket_view avoids a grad copy into DDP's all-reduce buckets.
        model = DDP(
            base_model,
            device_ids=[device.index],
            output_device=device.index,
            gradient_as_bucket_view=True,
        )
        LOGGER.info("Multi-GPU: DistributedDataParallel rank %d/%d", rank, world_size)
        # Rebuild the train loader to shard across ranks and hold the GLOBAL batch
        # at the configured size (per-rank batch = batch // world_size), so the
        # effective batch and LR stay paper-faithful.
        train_loader, train_sampler = _distributed_train_loader(train_loader, rank, world_size)
        # Shard validation too, so both GPUs evaluate instead of rank 0 alone while
        # rank 1 idles. The strided split covers every image exactly once (no
        # DistributedSampler padding), so the all-reduced metrics are identical to
        # a single-GPU eval over the full set.
        eval_loader = _distributed_eval_loader(test_loader, rank, world_size)
    else:
        train_sampler = None
        eval_loader = test_loader
    # torch.compile fuses the many tiny conv/BN/act kernels (and with
    # mode="reduce-overhead" captures CUDA graphs), attacking the launch-overhead
    # bottleneck that dominates this small model. Compile the training handle only;
    # eval runs on the eager base_model. base_model keeps the real parameters, so
    # the optimizer (built from it) and gradients are unaffected. Full FP32.
    if device.type == "cuda" and bool(config["train"].get("compile", False)):
        compile_mode = config["train"].get("compile_mode") or None
        model = torch.compile(model, mode=compile_mode)
        LOGGER.info("torch.compile enabled (mode=%s)", compile_mode or "default")
    optimizer = build_optimizer(config, base_model, variant)
    scheduler = build_scheduler(config, optimizer)
    label_smoothing = float(
        variant.get("train", {}).get(
            "label_smoothing", config["train"].get("label_smoothing", 0.0)
        )
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    configured_metrics = config.get("metrics", ["train_loss", "test_accuracy"])
    needs_super_metrics = any("_super_" in metric for metric in configured_metrics)
    needs_class_metrics = needs_super_metrics or any("_class_" in metric for metric in configured_metrics)
    needs_train_eval = any(metric.startswith("train_eval_") for metric in configured_metrics)
    num_classes = int(config["model"].get("args", {}).get("num_classes", 0))
    if needs_class_metrics and num_classes < 1:
        raise ValueError("model.args.num_classes is required for classwise metrics.")
    coarse_map = _resolve_coarse_map(config, num_classes, device) if needs_super_metrics else None
    train_eval_loader = _build_train_eval_loader(config, variant) if needs_train_eval else None
    train_eval_interval = int(config.get("train", {}).get("train_eval_interval", 10))
    classwise_criterion = nn.CrossEntropyLoss(reduction="none")
    # Only rank 0 owns the external side effects (W&B, Telegram, metric files);
    # the other ranks just compute their data shard.
    run = start_wandb(config, variant, seed) if is_main else None
    wandb_url = get_wandb_url(run)
    if start_notification is not None and is_main:
        send_telegram(
            start_notification + f"\nW&B: {wandb_url or 'not available'}",
            config,
        )
    history = []

    train_overrides = variant.get("train", {})
    custom_l2 = float(train_overrides.get("custom_l2", config["train"].get("custom_l2", 0.0)))
    distance_lambda = float(
        train_overrides.get("distance_lambda", config["train"].get("distance_lambda", 0.0))
    )
    _max_grad_norm = train_overrides.get("max_grad_norm", config["train"].get("max_grad_norm"))
    max_grad_norm = float(_max_grad_norm) if _max_grad_norm is not None else None
    wd_bounds = weight_decay_bounds(config, variant)
    if wd_bounds is not None:
        LOGGER.info("Cosine WD annealing enabled: %g -> %g", wd_bounds[0], wd_bounds[1])
    probe_config = config.get("gradient_probe", {})
    if probe_config.get("enabled"):
        from structural_reparam.analysis import GradientProbe
        probe = GradientProbe(base_model)
        probe.attach_optimizer(optimizer)
    else:
        probe = None
    log_every = int(probe_config.get("log_every", 1))

    branch_probe_config = config.get("branch_probe", {})
    branch_probe = None
    branch_probe_global_step = 0
    if branch_probe_config.get("enabled"):
        from structural_reparam.analysis import BranchProbe

        oracle_dataset_args = dict(config["dataset"].get("args", {}))
        # Force shuffled (class-balanced) batches for the oracle loader, ignoring
        # any variant override that switched to class_concentrated.
        oracle_dataset_args["batch_mode"] = "shuffled"
        oracle_dataset_args.pop("classes_per_batch", None)
        if "data_dir" in oracle_dataset_args:
            oracle_dataset_args["data_dir"] = resolve_repo_path(oracle_dataset_args["data_dir"])
        oracle_train_loader, _ = import_target(config["dataset"]["target"])(**oracle_dataset_args)

        probe_output = get_output_dir(config) / "branch_probe" / f"{variant_name}_seed{seed}.jsonl"
        branch_probe = BranchProbe(
            model=base_model,
            optimizer=optimizer,
            oracle_loader=oracle_train_loader,
            device=device,
            criterion=criterion,
            log_path=probe_output,
        )
        LOGGER.info("BranchProbe enabled; per-step records -> %s", probe_output)

    # Generic registry-driven snapshot probes. See analysis/registry.py — add
    # new probes via @register_probe and a `probes:` entry in the YAML.
    from structural_reparam.analysis import build_probes, ProbeContext
    # Importing analysis triggers registration of all known probes.
    import structural_reparam.analysis  # noqa: F401

    def _probe_ctx(name: str, probe_cfg: dict[str, Any]) -> ProbeContext:
        return ProbeContext(
            model=base_model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            config=config,
            variant=variant,
            output_dir=get_output_dir(config),
            seed=seed,
            probe_config=probe_cfg,
        )

    generic_probes = build_probes(_probe_ctx, config.get("probes"))
    for p in generic_probes:
        LOGGER.info("Registered snapshot probe '%s'", getattr(p, "PROBE_NAME", type(p).__name__))

    conflict_probe_config = config.get("conflict_probe", {})
    if conflict_probe_config.get("enabled") and num_classes > 0:
        from structural_reparam.analysis.class_conflict import ClassConflictProbe, log_conflict_to_wandb
        conflict_probe = ClassConflictProbe(base_model, num_classes)
        conflict_interval = int(conflict_probe_config.get("interval", 5))
        conflict_loader = _build_train_eval_loader(config, variant)
    else:
        conflict_probe = None
        conflict_interval = 0
        conflict_loader = None

    # DDP shards/evaluates differently per rank; the per-step probes and the
    # train-eval pass assume a single process over the full dataset. Guard on the
    # actually-constructed probes (an `enabled: false` config block is harmless).
    if distributed and (
        needs_train_eval
        or probe is not None
        or branch_probe is not None
        or conflict_probe is not None
        or generic_probes
    ):
        raise ValueError(
            "The DDP (multi-GPU) path does not support per-step probes or the "
            "train-eval pass; disable them or run on a single GPU."
        )

    total_epochs = config["train"]["epochs"]
    # Validate every eval_interval epochs (always epoch 1 and the last) — fewer
    # eager eval passes means less CUDA-allocator churn between train steps, which
    # is the suspected trigger for the reduce-overhead (CUDA-graph) hang.
    eval_interval = int(config["train"].get("eval_interval", 1))
    # >0 emits per-step heartbeats + per-epoch CUDA memory stats to diagnose hangs.
    heartbeat_every = int(config["train"].get("heartbeat_every", 0))
    for epoch in range(1, total_epochs + 1):
        if heartbeat_every and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        if train_sampler is not None:
            # Reshuffle each rank's shard deterministically per epoch.
            train_sampler.set_epoch(epoch)
        current_wd = None
        if wd_bounds is not None:
            current_wd = cosine_weight_decay(wd_bounds[0], wd_bounds[1], epoch, total_epochs)
            for group in optimizer.param_groups:
                group["weight_decay"] = current_wd
        if branch_probe is not None:
            train_stats, branch_probe_global_step = run_train_epoch_with_branch_probe(
                model,
                train_loader,
                criterion,
                device,
                optimizer,
                probe=branch_probe,
                epoch=epoch,
                start_global_step=branch_probe_global_step,
                max_grad_norm=max_grad_norm,
            )
        else:
            train_stats = run_epoch(
                model,
                train_loader,
                criterion,
                device,
                optimizer,
                custom_l2=custom_l2,
                distance_lambda=distance_lambda,
                max_grad_norm=max_grad_norm,
                distributed=distributed,
                amp_dtype=amp_dtype,
                heartbeat_every=heartbeat_every,
                channels_last=channels_last,
            )
        if scheduler is not None:
            scheduler.step()

        if heartbeat_every and torch.cuda.is_available():
            LOGGER.info(
                "epoch=%d post-train mem peak_alloc=%.0fMB reserved=%.0fMB",
                epoch,
                torch.cuda.max_memory_allocated(device) / 1e6,
                torch.cuda.memory_reserved(device) / 1e6,
            )

        # All ranks evaluate their shard of the val set on the unwrapped model
        # (base_model, so no DDP buffer-broadcast); run_epoch all-reduces the
        # totals to global metrics identical to a single-process eval. Validation
        # only runs every eval_interval epochs (+ first/last). All ranks make the
        # same decision (depends only on epoch), so the collective stays in sync.
        do_eval = (epoch % eval_interval == 0) or epoch == 1 or epoch == total_epochs
        test_stats = (
            run_epoch(
                base_model,
                eval_loader,
                criterion,
                device,
                num_classes=num_classes if needs_class_metrics else None,
                coarse_map=coarse_map,
                distributed=distributed,
                amp_dtype=amp_dtype,
                heartbeat_every=heartbeat_every,
                channels_last=channels_last,
            )
            if do_eval
            else None
        )
        if not is_main:
            continue

        run_train_eval = (
            train_eval_loader is not None
            and (epoch % train_eval_interval == 0 or epoch == total_epochs)
        )
        train_eval_class_stats = (
            classwise_stats(
                model,
                train_eval_loader,
                classwise_criterion,
                device,
                num_classes=num_classes,
                coarse_map=coarse_map,
                include_overall=True,
            )
            if run_train_eval
            else None
        )

        metrics = select_metrics(
            configured_metrics,
            train_stats,
            test_stats,
            model,
            test_class_stats=test_stats if needs_class_metrics else None,
            train_eval_class_stats=train_eval_class_stats,
        )
        record = {"epoch": epoch, "seed": seed, "variant": variant_name, **metrics}
        if current_wd is not None:
            record["weight_decay"] = current_wd
        if probe is not None and epoch % log_every == 0:
            record.update(probe.epoch_stats())
        if branch_probe is not None:
            record.update(branch_probe.flush_epoch(epoch))
        for gp in generic_probes:
            record.update(gp.epoch_stats(epoch))
        history.append(record)
        if run is not None:
            run.log(record, step=epoch)
        if conflict_probe is not None and run is not None:
            is_last = epoch == total_epochs
            if epoch % conflict_interval == 0 or is_last:
                conflict_result = conflict_probe.compute(conflict_loader, device)
                log_conflict_to_wandb(
                    conflict_result, run, epoch, log_matrix=is_last
                )
        LOGGER.info(
            "Finished epoch variant=%s epoch=%s/%s metrics=%s",
            variant_name,
            epoch,
            config["train"]["epochs"],
            json.dumps(metrics, sort_keys=True),
        )
        print(json.dumps(record), flush=True)

    if probe is not None:
        probe.detach()
    if branch_probe is not None:
        branch_probe.close()
    for gp in generic_probes:
        gp.close()
    if run is not None:
        run.finish()
    LOGGER.info(
        "Finished variant=%s epochs=%s elapsed_sec=%.2f",
        variant_name,
        config["train"]["epochs"],
        time.perf_counter() - started_at,
    )
    return history, wandb_url


def write_metrics(config: dict[str, Any], history: list[dict[str, Any]]) -> Path:
    output_dir = get_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    with metrics_path.open("w", encoding="utf-8") as f:
        for record in history:
            f.write(json.dumps(record) + "\n")
    LOGGER.info("Wrote %s metric records to %s", len(history), metrics_path)
    return metrics_path


def _load_config(args: argparse.Namespace) -> dict[str, Any]:
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
    return config


def _run_sweep(rank: int, world_size: int, args: argparse.Namespace) -> None:
    """Run the full seed x variant sweep for one process (rank)."""
    started_at = time.perf_counter()
    is_main = rank == 0
    load_env_file()
    config = _load_config(args)
    config_path = resolve_config_path(args)

    output_dir = get_output_dir(config)
    if is_main:
        log_path = setup_logging(output_dir)
        LOGGER.info("Logging to %s", log_path)
        LOGGER.info("Command: %s", " ".join(sys.argv))
        LOGGER.info("Loaded config from %s", config_path)
    else:
        # Non-rank-0 processes must not clobber rank 0's train.log; keep them quiet.
        logging.getLogger().setLevel(logging.WARNING)

    device = torch.device(f"cuda:{rank}") if world_size > 1 else torch.device(args.device)
    if config.get("train", {}).get("compile_debug"):
        # Surface torch.compile recompiles / graph breaks / cudagraph events to
        # diagnose the reduce-overhead hang.
        try:
            torch._logging.set_logs(recompiles=True, graph_breaks=True)
            LOGGER.info("compile_debug: dynamo recompile/graph-break logging enabled")
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("compile_debug: could not enable dynamo logging: %s", exc)
    seeds = experiment_seeds(config)
    LOGGER.info("rank=%s/%s using device=%s seeds=%s", rank, world_size, device, seeds)
    variants = config["model"].get("variants") or [{"name": config["experiment"]["name"]}]
    experiment_summary = format_experiment_summary(config, variants)
    if is_main:
        LOGGER.info(
            "Starting training sweep experiment=%s variants=%s epochs=%s world_size=%s",
            config["experiment"]["name"],
            [variant["name"] for variant in variants],
            config["train"]["epochs"],
            world_size,
        )
    start_notification = (
        "Experiment started\n"
        "------------------\n"
        f"{experiment_summary}\n"
        f"Device: {device} (world_size={world_size})\n"
        f"Command: {' '.join(sys.argv)}"
    )

    history = []
    wandb_urls = []
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
                variant_history, wandb_url = train_variant(
                    config,
                    variant,
                    seed,
                    train_loader,
                    test_loader,
                    device,
                    start_notification=start_notification if run_index == 1 else None,
                    rank=rank,
                    world_size=world_size,
                    is_main=is_main,
                )
                history.extend(variant_history)
                if wandb_url is not None:
                    wandb_urls.append(wandb_url)
        if is_main:
            metrics_path = write_metrics(config, history)
    except BaseException as exc:
        elapsed = time.perf_counter() - started_at
        LOGGER.exception("Training sweep failed after %s", format_duration(elapsed))
        if is_main:
            send_telegram(
                "Experiment failed\n"
                "-----------------\n"
                f"{experiment_summary}\n"
                f"Elapsed: {format_duration(elapsed)}\n"
                f"Error: {type(exc).__name__}: {exc}",
                config,
            )
        raise

    if not is_main:
        return

    elapsed = time.perf_counter() - started_at
    LOGGER.info(
        "Finished training sweep records=%s metrics_path=%s elapsed_sec=%.2f",
        len(history),
        metrics_path,
        elapsed,
    )
    send_telegram(
        "Experiment finished\n"
        "-------------------\n"
        f"{experiment_summary}\n"
        f"{format_final_metrics(history)}\n"
        f"Metrics file: {metrics_path}\n"
        f"W&B: {', '.join(wandb_urls) if wandb_urls else 'not available'}\n"
        f"Elapsed: {format_duration(elapsed)}",
        config,
    )


def _ddp_worker(rank: int, world_size: int, args: argparse.Namespace) -> None:
    # mp.spawn starts each rank under the 'spawn' context, which a child inherits
    # as its default. DataLoader would then *spawn* its workers, pickling the whole
    # ImageFolder (1.28M paths) to each over a pipe — minutes of stall + huge RAM.
    # Force 'fork' so workers copy memory instantly, exactly like a torchrun rank
    # or the original single-process run. Workers do CPU-only decode (never touch
    # CUDA), so forking after CUDA/NCCL init is safe here.
    import multiprocessing

    multiprocessing.set_start_method("fork", force=True)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    try:
        _run_sweep(rank, world_size, args)
    finally:
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available, but --device is set to 'cuda'.")

    # Decide on DDP: one process per visible GPU. Single-GPU/CPU runs (and any
    # config with train.distributed=false) take the plain in-process path so the
    # existing CIFAR / tiny-imagenet experiments are byte-for-byte unchanged.
    world_size = 1
    if args.device == "cuda" and not args.smoke and torch.cuda.device_count() > 1:
        if bool(_load_config(args)["train"].get("distributed", True)):
            world_size = torch.cuda.device_count()

    if world_size > 1:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        if "MASTER_PORT" not in os.environ:
            # Bind port 0 to grab a free port, avoiding EADDRINUSE when a just-killed
            # previous run still holds the old rendezvous port in TIME_WAIT.
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.bind(("", 0))
                os.environ["MASTER_PORT"] = str(probe.getsockname()[1])
        mp.spawn(_ddp_worker, args=(world_size, args), nprocs=world_size, join=True)
    else:
        _run_sweep(0, 1, args)


if __name__ == "__main__":
    main()
