"""A resumable entry module for the full-resolution ImageNet campaign.

Written 2026-09-06, after four runs were lost to Thunder Compute runtime panics in
31 A100-hours -- one of them at epoch 32 of 40, which cost 7.8 hours because
nothing here could continue from where it stopped.

--------------------------------------------------------------------------------
1. What this is, and what it deliberately is not
--------------------------------------------------------------------------------
It is the paper's own training loop with two things added: a checkpoint written
every epoch that holds everything needed to continue, and a resume that finds the
newest such checkpoint and carries on from it. It is NOT a second trainer. Every
piece of arithmetic -- the model, the optimizer, the scheduler, the epoch step,
the metric selection, the probes -- is imported from `structural_reparam.deploy.train`
and called in the order that module calls it, so a run through here and a run
through the generic trainer take the same path through the same code.

The price of that promise is that this module supports ONLY what this campaign
uses, and refuses everything else rather than quietly ignoring it: one seed, one
variant, one GPU, no automatic mixed precision, no torch.compile, no channels-last,
no gradient clipping, no weight-decay schedule, no label smoothing, no class or
train-eval metrics, no branch or conflict probe. `_refuse_unsupported` raises on
any of them. A silent difference between this loop and the generic one would
corrupt the comparison between the runs that used each, which is the whole reason
the arithmetic is borrowed rather than rewritten.

--------------------------------------------------------------------------------
2. Why the checkpoint goes to Weights and Biases
--------------------------------------------------------------------------------
The instance dies with the run. Thunder's job wrapper deletes the box on any exit,
and even when it does not, a new attempt gets a new box restored from the
snapshot, so a checkpoint on local disk is a checkpoint nobody can reach. Each
epoch's state is therefore uploaded as a W&B artifact named for the group, variant
and seed -- not for the run -- so the next attempt can find its predecessor's work
by name. Only the two newest versions are kept; older ones are deleted as they are
superseded, which holds the cost at about twice one checkpoint per arm rather than
forty times.

--------------------------------------------------------------------------------
3. What "resumed" means here, precisely
--------------------------------------------------------------------------------
The checkpoint carries the model's parameters and buffers (BatchNorm running
statistics included), the optimizer's state (SGD momentum buffers), the
scheduler's state (so the cosine continues at the right temperature rather than
restarting hot), the epoch just finished, and the random-number state of Python,
NumPy, torch and CUDA. Restoring the torch generator at an epoch boundary is what
makes the next epoch's shuffle and augmentation follow the sequence they would
have followed, so a resumed run is not merely similar to an uninterrupted one --
`tests/test_train_resumable.py` requires it to be bit-identical.

It also carries the W&B run id, and the resumed process re-attaches to that run
rather than starting a new one, so a run interrupted three times still reads as
one continuous history with one row per epoch.

--------------------------------------------------------------------------------
4. What a panic now costs
--------------------------------------------------------------------------------
The epoch in progress, plus the snapshot restore and warm pass: about 25 minutes
against the up-to-11-hours it cost before.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

# The arithmetic is the generic trainer's, imported rather than reimplemented.
from structural_reparam.deploy.train import (
    build_loaders,
    build_model,
    build_optimizer,
    build_scheduler,
    get_output_dir,
    load_env_file,
    resolve_config_path,
    run_epoch,
    seed_everything,
    select_metrics,
    setup_logging,
    write_metrics,
    _load_config,
)
from structural_reparam.analysis.registry import ProbeContext, build_probes
from structural_reparam.agents.stage2 import probe as _probe  # noqa: F401  (registers pair_channel_open)

LOGGER = logging.getLogger(__name__)

CHECKPOINT_FILE = "resume_state.pt"
KEEP_VERSIONS = 2
# How often the resume state is written and uploaded, in epochs. One means a panic
# costs the epoch in progress; N means it costs up to N epochs, and the upload's
# cost is paid N times less often. Set from `train.checkpoint_every_epochs`, and
# the last epoch is always written whatever the interval, so a run that finishes
# leaves a complete state behind.
DEFAULT_CHECKPOINT_EVERY = 1


# --------------------------------------------------------------------------------
# 1. The refusal gate
# --------------------------------------------------------------------------------

UNSUPPORTED = {
    "train": ["amp", "compile", "channels_last", "max_grad_norm", "weight_decay_final",
              "label_smoothing", "train_eval_interval", "custom_l2", "distance_lambda"],
    "top": ["conflict_probe"],
}


def _refuse_unsupported(config: dict[str, Any]) -> None:
    """Raise rather than silently run a different experiment from the generic loop."""
    bad = [f"train.{k}" for k in UNSUPPORTED["train"] if config.get("train", {}).get(k)]
    bad += [k for k in UNSUPPORTED["top"] if (config.get(k) or {}).get("enabled")]
    if any(m.startswith(("train_eval_", "test_class_", "test_super_"))
           for m in config.get("metrics", [])):
        bad.append("metrics: class, super-class or train-eval metrics")
    if int(config.get("num_seeds", 1)) != 1:
        bad.append(f"num_seeds={config.get('num_seeds')} (this module runs one seed)")
    if len(config["model"].get("variants") or [{}]) != 1:
        bad.append("more than one variant (this module runs one arm per process)")
    if torch.cuda.device_count() > 1:
        bad.append(f"{torch.cuda.device_count()} visible GPUs (no DDP path here)")
    if bad:
        raise SystemExit(
            "train_resumable supports only the full-resolution campaign's cell; "
            "the generic trainer handles the rest. Refusing: " + ", ".join(bad)
        )


# --------------------------------------------------------------------------------
# 2. The checkpoint
# --------------------------------------------------------------------------------

def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _load_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu() if torch.is_tensor(state["torch"]) else state["torch"])
    if torch.cuda.is_available() and state.get("cuda"):
        try:
            torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
        except Exception as exc:  # noqa: BLE001  (a different GPU count on the new box)
            LOGGER.warning("could not restore the CUDA generator state: %r", exc)


def save_state(path: Path, *, epoch: int, model: nn.Module, optimizer, scheduler,
               history: list[dict[str, Any]], run_id: str | None, seed: int,
               variant_name: str) -> Path:
    """Everything needed to continue, in one file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "seed": int(seed),
            "variant": variant_name,
            "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "rng": _rng_state(),
            "history": history,
            "wandb_run_id": run_id,
        },
        path,
    )
    return path


def restore_state(path: Path, *, model: nn.Module, optimizer, scheduler, device) -> dict[str, Any]:
    """Load a checkpoint into live objects and return its bookkeeping."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(blob["model"])
    model.to(device)
    optimizer.load_state_dict(blob["optimizer"])
    # SGD's momentum buffers come back on the CPU; move them where the parameters are.
    for state in optimizer.state.values():
        for k, v in list(state.items()):
            if torch.is_tensor(v):
                state[k] = v.to(device)
    if scheduler is not None and blob.get("scheduler") is not None:
        scheduler.load_state_dict(blob["scheduler"])
    _load_rng_state(blob["rng"])
    return blob


# --------------------------------------------------------------------------------
# 3. Carrying the checkpoint off the box
# --------------------------------------------------------------------------------

def artifact_name(config: dict[str, Any], variant_name: str, seed: int) -> str:
    group = config.get("logging", {}).get("group", config["experiment"]["name"])
    # This campaign's variants are already named `<arm>_s<seed>`; only add the seed
    # when the variant does not carry it, so the name stays readable and stable.
    tail = variant_name if variant_name.endswith(f"_s{seed}") else f"{variant_name}_s{seed}"
    return f"resume_{group}_{tail}"


def upload_state(run, path: Path, name: str, epoch: int) -> None:
    """Publish this epoch's state, then drop versions older than the newest two."""
    if run is None:
        return
    try:
        import wandb
        art = wandb.Artifact(name, type="resume_state", metadata={"epoch": epoch})
        art.add_file(str(path), name=CHECKPOINT_FILE)
        run.log_artifact(art)
        art.wait()
    except Exception as exc:  # noqa: BLE001  (never let bookkeeping kill a run)
        LOGGER.warning("could not upload the resume state for epoch %d: %r", epoch, exc)
        return
    try:
        import wandb
        api = wandb.Api()
        versions = list(api.artifacts(type_name="resume_state",
                                      name=f"{run.entity}/{run.project}/{name}"))
        versions.sort(key=lambda a: int(str(a.version).lstrip("v")))
        for old in versions[:-KEEP_VERSIONS]:
            try:
                old.delete(delete_aliases=True)
            except Exception:  # noqa: BLE001
                pass
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("could not prune old resume states: %r", exc)


def fetch_state(config: dict[str, Any], variant_name: str, seed: int,
                into: Path) -> Path | None:
    """The newest resume state for this arm, from any earlier attempt, or None.

    Local disk first, then W&B. The local copy only exists when the process is
    restarting on the same box -- which the Thunder workflow does not usually do --
    but consulting it costs nothing and covers the case where W&B is unreachable at
    startup, where the artifact path would otherwise silently start the run over
    from epoch zero.
    """
    local = into / CHECKPOINT_FILE
    if local.exists():
        LOGGER.info("Found a resume state on local disk: %s", local)
        return local
    name = artifact_name(config, variant_name, seed)
    logging_cfg = config.get("logging", {})
    project = logging_cfg.get("project", "structural-reparam")
    entity = logging_cfg.get("entity")
    try:
        import wandb
        api = wandb.Api()
        ref = f"{entity + '/' if entity else ''}{project}/{name}:latest"
        art = api.artifact(ref, type="resume_state")
        into.mkdir(parents=True, exist_ok=True)
        art.download(root=str(into))
        got = into / CHECKPOINT_FILE
        if got.exists():
            LOGGER.info("Found a resume state: %s (epoch %s)", ref,
                        (art.metadata or {}).get("epoch"))
            return got
    except Exception as exc:  # noqa: BLE001  (no artifact yet is the normal first run)
        LOGGER.info("No resume state to continue from (%s)", type(exc).__name__)
    return None


# --------------------------------------------------------------------------------
# 4. The loop
# --------------------------------------------------------------------------------

def start_wandb_resumable(config: dict[str, Any], variant: dict[str, Any], seed: int,
                          run_id: str | None):
    """`deploy.train.start_wandb`, with the one change this module needs: it
    re-attaches to a previous attempt's run so a restarted arm keeps one history."""
    logging_config = config.get("logging", {})
    if logging_config.get("backend") != "wandb":
        return None
    import wandb
    experiment = config["experiment"]["name"]
    name = f"{experiment}/{variant['name']}"
    if int(config.get("num_seeds", 1)) > 1:
        name = f"{name}/seed_{seed}"
    if os.environ.get("WANDB_API_KEY"):
        wandb.login(key=os.environ["WANDB_API_KEY"])
    kwargs = {
        "project": logging_config.get("project", "structural-reparam"),
        "name": name,
        "group": logging_config.get("group", experiment),
        "job_type": logging_config.get("job_type", "train"),
        "tags": config["experiment"].get("tags", []),
        "config": {**config, "seed": seed, "variant": variant},
    }
    for key in ("entity", "mode"):
        if key in logging_config:
            kwargs[key] = logging_config[key]
    if run_id:
        kwargs["id"] = run_id
        kwargs["resume"] = "allow"
        LOGGER.info("Re-attaching to W&B run %s", run_id)
    return wandb.init(**kwargs)


def main() -> None:
    # The types matter and are the generic trainer's, not a stylistic choice:
    # `deploy.train.resolve_config_path` returns `args.config` unchanged and then
    # calls `.open()` on it, so a plain string raises AttributeError before the run
    # starts. That cost one instance on 2026-09-06 -- the job died in its first
    # minutes and the wrapper's trap deleted the box, leaving no log to read.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--experiment")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    load_env_file()
    config = _load_config(args)
    _refuse_unsupported(config)
    output_dir = get_output_dir(config)
    setup_logging(output_dir)
    LOGGER.info("Loaded config from %s", resolve_config_path(args))

    seed = int(config.get("base_seed", 42))
    variant = (config["model"].get("variants") or [{"name": config["experiment"]["name"]}])[0]
    variant_name = variant["name"]
    device = torch.device(args.device)
    total_epochs = int(config["train"]["epochs"])
    eval_interval = int(config["train"].get("eval_interval", 1))
    heartbeat_every = int(config["train"].get("heartbeat_every", 0))
    checkpoint_every = max(1, int(config["train"].get("checkpoint_every_epochs",
                                                      DEFAULT_CHECKPOINT_EVERY)))
    torch.backends.cudnn.benchmark = bool(config["train"].get("cudnn_benchmark", False))
    torch.backends.cudnn.deterministic = not torch.backends.cudnn.benchmark

    # The generic trainer seeds TWICE, and both matter: `_run_sweep` seeds before
    # building the loaders, and `train_variant` seeds again before building the
    # model, so the model's initialization starts from a generator that is at the
    # same place whatever the loaders did. Dropping the second call gives a
    # differently initialized network -- a different experiment at the same seed --
    # and every run already finished used the two-call order.
    seed_everything(seed)
    train_loader, test_loader = build_loaders(config, variant)
    seed_everything(seed)
    model = build_model(config, variant).to(device)
    optimizer = build_optimizer(config, model, variant)
    scheduler = build_scheduler(config, optimizer)
    criterion = nn.CrossEntropyLoss()

    # Resume before the probes are built, so they see the restored network.
    ckpt_dir = output_dir / "resume"
    start_epoch, history, run_id = 0, [], None
    found = fetch_state(config, variant_name, seed, ckpt_dir)
    if found is not None:
        blob = restore_state(found, model=model, optimizer=optimizer,
                             scheduler=scheduler, device=device)
        start_epoch = int(blob["epoch"])
        history = list(blob.get("history") or [])
        run_id = blob.get("wandb_run_id")
        if blob.get("seed") != seed or blob.get("variant") != variant_name:
            raise SystemExit(
                f"the resume state is for {blob.get('variant')} seed {blob.get('seed')}, "
                f"not {variant_name} seed {seed}")
        LOGGER.info("Resuming %s at epoch %d of %d", variant_name, start_epoch + 1, total_epochs)
    if start_epoch >= total_epochs:
        LOGGER.info("Nothing to do: the resume state is already at epoch %d", start_epoch)
        return

    run = start_wandb_resumable(config, variant, seed, run_id)
    run_id = run.id if run is not None else None

    def _probe_ctx(name: str, probe_cfg: dict[str, Any]) -> ProbeContext:
        return ProbeContext(model=model, optimizer=optimizer, criterion=criterion,
                            device=device, config=config, variant=variant,
                            output_dir=output_dir, seed=seed, probe_config=probe_cfg)

    probes = build_probes(_probe_ctx, config.get("probes"))
    for p in probes:
        LOGGER.info("Registered snapshot probe '%s'", getattr(p, "PROBE_NAME", type(p).__name__))
    LOGGER.info("Resume checkpoints every %d epoch(s); a panic costs at most that much work",
                checkpoint_every)

    started_at = time.perf_counter()
    for epoch in range(start_epoch + 1, total_epochs + 1):
        train_stats = run_epoch(model, train_loader, criterion, device, optimizer,
                                heartbeat_every=heartbeat_every)
        if scheduler is not None:
            scheduler.step()
        do_eval = (epoch % eval_interval == 0) or epoch == 1 or epoch == total_epochs
        test_stats = run_epoch(model, test_loader, criterion, device, None) if do_eval else None
        metrics = select_metrics(config.get("metrics", ["train_loss", "test_accuracy"]),
                                 train_stats, test_stats, model)
        record = {"epoch": epoch, "seed": seed, "variant": variant_name, **metrics}
        for p in probes:
            record.update(p.epoch_stats(epoch))
        history.append(record)
        if run is not None:
            run.log(record, step=epoch)
        LOGGER.info("Finished epoch variant=%s epoch=%s/%s metrics=%s",
                    variant_name, epoch, total_epochs, json.dumps(metrics, sort_keys=True))
        print(json.dumps(record), flush=True)

        if epoch % checkpoint_every == 0 or epoch == total_epochs:
            state_path = save_state(ckpt_dir / CHECKPOINT_FILE, epoch=epoch, model=model,
                                    optimizer=optimizer, scheduler=scheduler, history=history,
                                    run_id=run_id, seed=seed, variant_name=variant_name)
            upload_state(run, state_path, artifact_name(config, variant_name, seed), epoch)

    for p in probes:
        p.close()
    write_metrics(config, history)
    if run is not None:
        run.finish()
    LOGGER.info("Finished variant=%s epochs=%s elapsed_sec=%.2f",
                variant_name, total_epochs, time.perf_counter() - started_at)


if __name__ == "__main__":
    main()
