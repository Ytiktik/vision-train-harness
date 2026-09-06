"""The resumable entry module: does a run that was interrupted end where it would have?

This is the whole question. A resume that lands anywhere else is not a resumed
run, it is a new experiment wearing the old one's name, and it would corrupt the
comparison against the arms that were never interrupted. So the central test runs
the same short training twice -- once straight through, once stopped after two
epochs, saved, rebuilt from nothing and continued -- and requires the two to agree
bit for bit on every parameter and buffer.

Everything is CPU-sized and synthetic: this is a fast local gate, not training.
"""
from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from structural_reparam.deploy.train import run_epoch, seed_everything
from structural_reparam.agents.stage2.train_resumable import (
    _refuse_unsupported,
    restore_state,
    save_state,
)

EPOCHS = 4
BREAK_AT = 2


def _loader(seed: int = 0) -> DataLoader:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(48, 3, 8, 8, generator=g)
    y = torch.randint(0, 4, (48,), generator=g)
    # shuffle=True on purpose: the epoch order is drawn from the global generator,
    # so it is part of what a resume has to reproduce.
    return DataLoader(TensorDataset(x, y), batch_size=8, shuffle=True)


def _build():
    """A small conv-BatchNorm network, so BatchNorm's running statistics -- which
    live in buffers rather than parameters, and which no optimizer restores -- are
    part of what the checkpoint has to carry."""
    seed_everything(1234)
    model = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1, bias=False), nn.BatchNorm2d(8), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 4),
    )
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    return model, opt, sched


def _train(model, opt, sched, loader, criterion, epochs):
    for _ in range(epochs):
        run_epoch(model, loader, criterion, torch.device("cpu"), opt)
        sched.step()


def _fingerprint(model) -> dict[str, torch.Tensor]:
    out = {k: v.detach().clone() for k, v in model.state_dict().items()}
    return out


def test_a_resumed_run_lands_exactly_where_an_uninterrupted_one_does(tmp_path):
    criterion = nn.CrossEntropyLoss()

    # (a) straight through
    seed_everything(7)
    model_a, opt_a, sched_a = _build()
    loader_a = _loader()
    seed_everything(7)                       # the loop's own starting RNG state
    _train(model_a, opt_a, sched_a, loader_a, criterion, EPOCHS)
    want = _fingerprint(model_a)
    want_lr = opt_a.param_groups[0]["lr"]

    # (b) stopped after BREAK_AT epochs, saved, rebuilt from nothing, continued
    seed_everything(7)
    model_b, opt_b, sched_b = _build()
    loader_b = _loader()
    seed_everything(7)
    _train(model_b, opt_b, sched_b, loader_b, criterion, BREAK_AT)
    path = save_state(tmp_path / "state.pt", epoch=BREAK_AT, model=model_b, optimizer=opt_b,
                      scheduler=sched_b, history=[], run_id=None, seed=7,
                      variant_name="unit")
    del model_b, opt_b, sched_b

    model_c, opt_c, sched_c = _build()       # a fresh process would look like this
    blob = restore_state(path, model=model_c, optimizer=opt_c, scheduler=sched_c,
                         device=torch.device("cpu"))
    assert blob["epoch"] == BREAK_AT
    loader_c = _loader()
    _train(model_c, opt_c, sched_c, loader_c, criterion, EPOCHS - BREAK_AT)
    got = _fingerprint(model_c)

    assert set(got) == set(want)
    for key in want:
        a, b = want[key], got[key]
        if a.is_floating_point():
            assert torch.equal(a, b), (key, float((a - b).abs().max()))
        else:
            assert torch.equal(a, b), key          # BatchNorm's num_batches_tracked
    assert opt_c.param_groups[0]["lr"] == pytest.approx(want_lr, rel=0, abs=0)


def test_the_checkpoint_carries_the_optimizer_and_the_schedule_not_just_weights(tmp_path):
    """The three things the old checkpoints lacked, which is why they could not be
    continued: momentum buffers, the scheduler's position, and the epoch."""
    model, opt, sched = _build()
    criterion = nn.CrossEntropyLoss()
    _train(model, opt, sched, _loader(), criterion, 2)
    blob = torch.load(save_state(tmp_path / "s.pt", epoch=2, model=model, optimizer=opt,
                                 scheduler=sched, history=[{"epoch": 1}], run_id="abc",
                                 seed=7, variant_name="unit"),
                      map_location="cpu", weights_only=False)
    assert blob["epoch"] == 2 and blob["wandb_run_id"] == "abc"
    assert blob["scheduler"]["last_epoch"] == 2
    momentum = [s for s in blob["optimizer"]["state"].values() if "momentum_buffer" in s]
    assert momentum, "SGD momentum buffers are not in the checkpoint"
    assert blob["rng"]["torch"] is not None
    assert any(k.endswith("running_mean") for k in blob["model"]), "BatchNorm buffers missing"


def test_a_checkpoint_from_another_arm_is_refused(tmp_path):
    """Resuming arm A from arm B's state would silently produce a mislabelled run."""
    model, opt, sched = _build()
    path = save_state(tmp_path / "s.pt", epoch=1, model=model, optimizer=opt, scheduler=sched,
                      history=[], run_id=None, seed=43, variant_name="pair_both_s43")
    blob = torch.load(path, map_location="cpu", weights_only=False)
    assert blob["seed"] == 43 and blob["variant"] == "pair_both_s43"


@pytest.mark.parametrize("config,needle", [
    ({"train": {"epochs": 1, "amp": True}, "model": {"variants": [{}]}}, "train.amp"),
    ({"train": {"epochs": 1, "compile": True}, "model": {"variants": [{}]}}, "train.compile"),
    ({"train": {"epochs": 1}, "model": {"variants": [{}, {}]}}, "more than one variant"),
    ({"train": {"epochs": 1}, "num_seeds": 3, "model": {"variants": [{}]}}, "num_seeds"),
    ({"train": {"epochs": 1}, "model": {"variants": [{}]},
      "metrics": ["train_eval_accuracy"]}, "train-eval"),
])
def test_it_refuses_anything_it_would_run_differently_from_the_generic_trainer(config, needle):
    with pytest.raises(SystemExit, match=needle.replace(".", r"\.")):
        _refuse_unsupported(config)


def test_it_accepts_the_campaign_cell():
    _refuse_unsupported({
        "train": {"epochs": 40, "eval_interval": 1, "heartbeat_every": 500,
                  "lr": 0.1, "momentum": 0.9, "weight_decay": 1e-4, "scheduler": "cosine"},
        "num_seeds": 1,
        "model": {"variants": [{"name": "pair_both_s42"}]},
        "metrics": ["train_loss", "train_accuracy", "test_loss", "test_accuracy"],
    })


def test_the_module_seeds_the_way_the_generic_trainer_does():
    """`_run_sweep` seeds, builds loaders, and `train_variant` seeds again before
    building the model. A single seeding gives a different network at the same
    seed, so the order is asserted here rather than left to the reader."""
    import inspect
    from structural_reparam.agents.stage2 import train_resumable

    body = inspect.getsource(train_resumable.main)
    seed_calls = [i for i, line in enumerate(body.splitlines())
                  if "seed_everything(seed)" in line and not line.strip().startswith("#")]
    loaders = next(i for i, line in enumerate(body.splitlines()) if "build_loaders(" in line)
    model = next(i for i, line in enumerate(body.splitlines()) if "build_model(" in line)
    assert len(seed_calls) == 2, "the trainer seeds twice; so must this"
    assert seed_calls[0] < loaders < seed_calls[1] < model


# --------------------------------------------------------------------------------
# End to end, as a subprocess, the way the instance runs it
# --------------------------------------------------------------------------------
#
# The unit tests above check the arithmetic of a resume. They cannot catch a
# startup fault, because they import the module rather than run it -- and a startup
# fault is expensive: the job dies in its first minutes and the Thunder wrapper's
# exit trap deletes the box, so there is no log to read and the campaign only sees
# an instance that vanished. That is exactly what happened on 2026-09-06, when
# `--config` was parsed as a string and `resolve_config_path` called `.open()` on
# it. This test runs the module as a process, on synthetic data, and would have
# caught it in a second.

def _fake_modules(tmp_path):
    (tmp_path / "fake_data.py").write_text(
        "import torch\n"
        "from torch.utils.data import DataLoader, TensorDataset\n"
        "def build_loaders(n=32, batch_size=8, **kw):\n"
        "    g = torch.Generator().manual_seed(0)\n"
        "    ds = TensorDataset(torch.randn(n, 3, 8, 8, generator=g),\n"
        "                       torch.randint(0, 4, (n,), generator=g))\n"
        "    return DataLoader(ds, batch_size=batch_size, shuffle=True), DataLoader(ds, batch_size=batch_size)\n"
    )
    (tmp_path / "fake_model.py").write_text(
        "import torch.nn as nn\n"
        "def build(num_classes=4, **kw):\n"
        "    return nn.Sequential(nn.Conv2d(3, 8, 3, padding=1, bias=False), nn.BatchNorm2d(8),\n"
        "                         nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(),\n"
        "                         nn.Linear(8, num_classes))\n"
    )


def _config(epochs: int) -> dict:
    return {
        "experiment": {"name": "e2e", "tags": []},
        "base_seed": 42, "num_seeds": 1,
        "job": {"entry_module": "structural_reparam.agents.stage2.train_resumable"},
        "dataset": {"name": "fake", "target": "fake_data.build_loaders",
                    "args": {"n": 32, "batch_size": 8}},
        "model": {"name": "fake", "target": "fake_model.build", "args": {"num_classes": 4},
                  "variants": [{"name": "single_s42", "args": {}}]},
        "train": {"epochs": epochs, "lr": 0.1, "momentum": 0.9, "weight_decay": 1e-4,
                  "scheduler": "cosine", "eval_interval": 1},
        "metrics": ["train_loss", "train_accuracy", "test_loss", "test_accuracy"],
        "probes": {}, "logging": {"backend": "none", "output_dir": "outputs/e2e"},
    }


def _run(tmp_path, epochs):
    import os
    import subprocess
    import sys as _sys
    import yaml
    repo = Path(__file__).resolve().parents[1]
    cfg = tmp_path / f"config_{epochs}.yaml"
    cfg.write_text(yaml.safe_dump(_config(epochs)))
    env = dict(os.environ)
    env.update({"PYTHONPATH": f"{repo}/src:{tmp_path}", "WANDB_MODE": "disabled",
                "HOME": str(tmp_path)})
    return subprocess.run(
        [_sys.executable, "-m", "structural_reparam.agents.stage2.train_resumable",
         "--config", str(cfg), "--device", "cpu", "--output-dir", str(tmp_path / "out")],
        cwd=repo, env=env, capture_output=True, text=True, timeout=600)


from pathlib import Path  # noqa: E402  (kept beside the helpers that use it)


def test_the_module_runs_as_a_process_and_writes_a_resumable_checkpoint(tmp_path):
    _fake_modules(tmp_path)
    done = _run(tmp_path, epochs=2)
    assert done.returncode == 0, done.stderr[-2000:]
    state = tmp_path / "out" / "resume" / "resume_state.pt"
    assert state.exists(), "no checkpoint was written"
    blob = torch.load(state, map_location="cpu", weights_only=False)
    assert blob["epoch"] == 2 and len(blob["history"]) == 2
    assert blob["scheduler"]["last_epoch"] == 2


def test_a_second_invocation_continues_rather_than_starting_over(tmp_path):
    """The behaviour the campaign depends on: a restarted attempt picks up the
    epoch after the last one it finished, instead of paying for them again."""
    _fake_modules(tmp_path)
    assert _run(tmp_path, epochs=2).returncode == 0
    again = _run(tmp_path, epochs=4)
    assert again.returncode == 0, again.stderr[-2000:]
    assert "Resuming single_s42 at epoch 3 of 4" in again.stderr, again.stderr[-800:]
    blob = torch.load(tmp_path / "out" / "resume" / "resume_state.pt",
                      map_location="cpu", weights_only=False)
    assert blob["epoch"] == 4
    # epochs 1 and 2 were not repeated: the history has one row per epoch
    assert [r["epoch"] for r in blob["history"]] == [1, 2, 3, 4]


def test_the_checkpoint_interval_is_honoured_and_the_last_epoch_always_lands(tmp_path):
    """With `train.checkpoint_every_epochs: 3` over 4 epochs, the state is written
    at epoch 3 and again at 4 -- the interval, and the end whatever the interval --
    so a finished run always leaves a complete state and a panic costs at most the
    interval."""
    import os
    import subprocess
    import sys as _sys
    import yaml
    _fake_modules(tmp_path)
    cfg = _config(4)
    cfg["train"]["checkpoint_every_epochs"] = 3
    path = tmp_path / "interval.yaml"
    path.write_text(yaml.safe_dump(cfg))
    repo = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env.update({"PYTHONPATH": f"{repo}/src:{tmp_path}", "WANDB_MODE": "disabled",
                "HOME": str(tmp_path)})
    done = subprocess.run(
        [_sys.executable, "-m", "structural_reparam.agents.stage2.train_resumable",
         "--config", str(path), "--device", "cpu", "--output-dir", str(tmp_path / "out")],
        cwd=repo, env=env, capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stderr[-2000:]
    assert "Resume checkpoints every 3 epoch(s)" in done.stderr
    blob = torch.load(tmp_path / "out" / "resume" / "resume_state.pt",
                      map_location="cpu", weights_only=False)
    assert blob["epoch"] == 4
