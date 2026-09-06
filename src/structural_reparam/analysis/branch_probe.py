"""Per-step branch probe for depth-1 BranchedLinear MLPs.

At each training step (after `backward()` on the class-concentrated batch and
before `optimizer.step()`), this probe records:

  1. Per-branch parameter-gradient cosine vs an *oracle* gradient computed on a
     shuffled (class-balanced) batch, with respect to the effective folded
     weight ``u = sum_i alpha_i * W_i`` (BN fold using the current
     class-concentrated batch stats).
  2. Loss on the *same* oracle batch after applying:
       (a) the SGD update restricted to branch 1's parameters,
       (b) the SGD update restricted to branch 2's parameters,
       (c) the full combined update.
     All three candidates snapshot state and restore — the probe is purely
     observational; real training takes its own ``optimizer.step()`` afterward.
  3. Per-branch raw gradient norms, the oracle-gradient norm, and the
     pre-step oracle loss.

The BN forward hook is *armed* just before each training forward via
``arm_hooks()``; it fires once (capturing the per-branch BN input from the
training batch), then auto-removes. Candidate-eval forwards do not fire it.

Per-step records go to a JSONL file. Per-epoch aggregates are returned by
``flush_epoch()`` for wandb logging.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from structural_reparam.models.mlp import BranchedLinear


class BranchProbe:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        oracle_loader: DataLoader,
        device: torch.device,
        criterion: nn.Module,
        log_path: Path | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.oracle_loader = oracle_loader
        self.oracle_iter = iter(oracle_loader)
        self.device = device
        self.criterion = criterion
        self.log_path = log_path
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_file = log_path.open("w", encoding="utf-8")
        else:
            self._log_file = None

        branched_layers = [m for m in model.modules() if isinstance(m, BranchedLinear)]
        if len(branched_layers) != 1:
            raise ValueError(
                "BranchProbe requires exactly one BranchedLinear layer; found "
                f"{len(branched_layers)}."
            )
        self.branched = branched_layers[0]
        if not isinstance(getattr(model, "fc", None), nn.Linear):
            raise ValueError("BranchProbe expects `model.fc` to be a nn.Linear head.")
        self.fc: nn.Linear = model.fc

        self.num_branches = len(self.branched.branches)
        self._branch_bn_inputs: list[torch.Tensor | None] = [None] * self.num_branches
        self._hook_handles: list[Any] = []

        self._branch_params: list[list[nn.Parameter]] = [
            [p for p in branch.parameters() if p.requires_grad]
            for branch in self.branched.branches
        ]
        self._all_params: list[nn.Parameter] = [p for p in model.parameters() if p.requires_grad]
        self._epoch_records: list[dict[str, float]] = []

    # ------------------------------------------------------------------ hooks
    def arm_hooks(self) -> None:
        """Register BN forward_pre_hooks that fire once then auto-remove.

        Must be called immediately before the training forward pass, so that
        the captured BN inputs correspond to the current class-concentrated
        training batch (not the probe's own internal forwards).
        """
        self._branch_bn_inputs = [None] * self.num_branches
        self._hook_handles = []

        def make_hook(idx: int):
            def hook(module, args):
                self._branch_bn_inputs[idx] = args[0].detach()
                # Self-remove so internal eval forwards do not overwrite.
                handle = self._hook_handles[idx]
                if handle is not None:
                    handle.remove()
                    self._hook_handles[idx] = None
            return hook

        for i, branch in enumerate(self.branched.branches):
            self._hook_handles.append(None)
        for i, branch in enumerate(self.branched.branches):
            h = branch[1].register_forward_pre_hook(make_hook(i))
            self._hook_handles[i] = h

    def _release_hooks(self) -> None:
        for i, h in enumerate(self._hook_handles):
            if h is not None:
                h.remove()
                self._hook_handles[i] = None

    def close(self) -> None:
        self._release_hooks()
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

    # ----------------------------------------------------------------- batch
    def _next_oracle_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            batch = next(self.oracle_iter)
        except StopIteration:
            self.oracle_iter = iter(self.oracle_loader)
            batch = next(self.oracle_iter)
        x, y = batch
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

    # ------------------------------------------------------------- snapshot
    def _snapshot(self) -> tuple[list[torch.Tensor], list[torch.Tensor | None]]:
        params = [p.detach().clone() for p in self._all_params]
        momentum: list[torch.Tensor | None] = []
        for p in self._all_params:
            buf = self.optimizer.state.get(p, {}).get("momentum_buffer")
            momentum.append(buf.detach().clone() if buf is not None else None)
        return params, momentum

    def _restore(self, snapshot: tuple[list[torch.Tensor], list[torch.Tensor | None]]) -> None:
        params, momentum = snapshot
        for p, p_snap, m_snap in zip(self._all_params, params, momentum):
            p.data.copy_(p_snap)
            state = self.optimizer.state.get(p)
            if m_snap is None:
                if state is not None and "momentum_buffer" in state:
                    del state["momentum_buffer"]
            else:
                if state is None:
                    self.optimizer.state[p] = {"momentum_buffer": m_snap.clone()}
                elif "momentum_buffer" in state:
                    state["momentum_buffer"].copy_(m_snap)
                else:
                    state["momentum_buffer"] = m_snap.clone()

    def _zero_grads_except(self, keep_params: list[nn.Parameter]) -> None:
        keep_ids = {id(p) for p in keep_params}
        for p in self._all_params:
            if id(p) not in keep_ids:
                p.grad = None

    @torch.no_grad()
    def _eval_loss(self, x: torch.Tensor, y: torch.Tensor) -> float:
        was_training = self.model.training
        self.model.eval()
        logits = self.model(x)
        loss = self.criterion(logits, y).item()
        if was_training:
            self.model.train()
        return loss

    # --------------------------------------------------------------- step
    def step(self, global_step: int, epoch: int) -> dict[str, float]:
        """Run probe; assumes training forward+backward already happened and
        ``arm_hooks()`` was called immediately before that forward."""

        # Per-branch param gradients on the class-concentrated training batch.
        branch_W_grads: list[torch.Tensor | None] = []
        for branch in self.branched.branches:
            g = branch[0].weight.grad
            branch_W_grads.append(g.detach().clone() if g is not None else None)

        # Verify BN hooks fired.
        if any(x is None for x in self._branch_bn_inputs):
            raise RuntimeError(
                "BranchProbe.step called without armed BN hooks firing; "
                "call arm_hooks() before the training forward."
            )

        # Build effective folded weight u from each branch using current batch stats.
        x_oracle, y_oracle = self._next_oracle_batch()
        x_oracle_flat = x_oracle.flatten(1)

        u_total: torch.Tensor | None = None
        bias_BN_eff = torch.zeros_like(self.branched.bias.detach())
        for i, branch in enumerate(self.branched.branches):
            x_in = self._branch_bn_inputs[i]
            mu = x_in.mean(dim=0)
            var = x_in.var(dim=0, unbiased=False) + branch[1].eps
            sigma = var.sqrt()
            gamma = branch[1].weight.detach()
            alpha = gamma / sigma
            W_i = branch[0].weight.detach()
            u_branch = alpha.unsqueeze(1) * W_i
            u_total = u_branch if u_total is None else u_total + u_branch
            bias_BN_eff = bias_BN_eff - alpha * mu

        u_param = u_total.clone().requires_grad_(True)
        bias_eff = (bias_BN_eff + self.branched.bias.detach()).detach()

        hidden = x_oracle_flat @ u_param.T + bias_eff
        hidden = F.relu(hidden)
        logits = F.linear(hidden, self.fc.weight.detach(), self.fc.bias.detach())
        loss_oracle_pre_t = self.criterion(logits, y_oracle)
        grad_u = torch.autograd.grad(loss_oracle_pre_t, u_param)[0].detach()
        oracle_loss_pre = loss_oracle_pre_t.item()

        # Cosines + norms.
        u_flat = grad_u.reshape(-1)
        u_norm = u_flat.norm()
        cosines: list[float] = []
        norms: list[float] = []
        for g in branch_W_grads:
            if g is None:
                cosines.append(float("nan"))
                norms.append(0.0)
                continue
            g_flat = g.reshape(-1)
            g_norm = g_flat.norm()
            denom = g_norm * u_norm
            cos = ((g_flat * u_flat).sum() / denom).item() if denom > 0 else float("nan")
            cosines.append(cos)
            norms.append(g_norm.item())
        oracle_grad_norm = u_norm.item()

        # Snapshot state + grads for purely-observational candidate evals.
        snapshot = self._snapshot()
        grad_snapshot = {
            id(p): (p.grad.detach().clone() if p.grad is not None else None)
            for p in self._all_params
        }

        def reapply_grads() -> None:
            for p in self._all_params:
                snap = grad_snapshot[id(p)]
                if snap is None:
                    p.grad = None
                else:
                    if p.grad is None:
                        p.grad = snap.clone()
                    else:
                        p.grad.copy_(snap)

        candidates: list[tuple[str, list[nn.Parameter]]] = []
        if self.num_branches >= 1:
            candidates.append(("branch_1_only", self._branch_params[0]))
        if self.num_branches >= 2:
            candidates.append(("branch_2_only", self._branch_params[1]))
        candidates.append(("combined", self._all_params))

        cand_losses: dict[str, float] = {}
        for label, keep in candidates:
            reapply_grads()
            if label != "combined":
                self._zero_grads_except(keep)
            self.optimizer.step()
            cand_losses[label] = self._eval_loss(x_oracle, y_oracle)
            self._restore(snapshot)

        reapply_grads()  # leave caller's grads as they found them

        record: dict[str, float] = {
            "global_step": global_step,
            "epoch": epoch,
            "oracle_loss_pre_step": oracle_loss_pre,
            "oracle_loss_branch1_only": cand_losses.get("branch_1_only", float("nan")),
            "oracle_loss_branch2_only": cand_losses.get("branch_2_only", float("nan")),
            "oracle_loss_combined": cand_losses["combined"],
            "delta_loss_branch1_only": oracle_loss_pre - cand_losses.get("branch_1_only", float("nan")),
            "delta_loss_branch2_only": oracle_loss_pre - cand_losses.get("branch_2_only", float("nan")),
            "delta_loss_combined": oracle_loss_pre - cand_losses["combined"],
            "cos_branch1_vs_oracle": cosines[0] if len(cosines) > 0 else float("nan"),
            "cos_branch2_vs_oracle": cosines[1] if len(cosines) > 1 else float("nan"),
            "grad_norm_branch1": norms[0] if len(norms) > 0 else 0.0,
            "grad_norm_branch2": norms[1] if len(norms) > 1 else 0.0,
            "oracle_grad_norm": oracle_grad_norm,
        }
        if self._log_file is not None:
            self._log_file.write(json.dumps(record) + "\n")
            self._log_file.flush()
        self._epoch_records.append(record)
        return record

    # ---------------------------------------------------------- aggregation
    def flush_epoch(self, epoch: int) -> dict[str, float]:
        """Return per-epoch aggregates of all per-step records since last flush;
        clears the buffer."""
        records = self._epoch_records
        self._epoch_records = []
        if not records:
            return {}

        def _mean(key: str) -> float:
            vals = [r[key] for r in records if not math.isnan(r[key])]
            return float(sum(vals) / len(vals)) if vals else float("nan")

        def _frac(pred) -> float:
            return float(sum(1 for r in records if pred(r)) / len(records))

        agg = {
            "probe/oracle_loss_pre_step_mean": _mean("oracle_loss_pre_step"),
            "probe/oracle_loss_combined_mean": _mean("oracle_loss_combined"),
            "probe/oracle_loss_branch1_only_mean": _mean("oracle_loss_branch1_only"),
            "probe/oracle_loss_branch2_only_mean": _mean("oracle_loss_branch2_only"),
            "probe/delta_loss_combined_mean": _mean("delta_loss_combined"),
            "probe/delta_loss_branch1_only_mean": _mean("delta_loss_branch1_only"),
            "probe/delta_loss_branch2_only_mean": _mean("delta_loss_branch2_only"),
            "probe/cos_branch1_vs_oracle_mean": _mean("cos_branch1_vs_oracle"),
            "probe/cos_branch2_vs_oracle_mean": _mean("cos_branch2_vs_oracle"),
            "probe/grad_norm_branch1_mean": _mean("grad_norm_branch1"),
            "probe/grad_norm_branch2_mean": _mean("grad_norm_branch2"),
            "probe/oracle_grad_norm_mean": _mean("oracle_grad_norm"),
            "probe/frac_branch1_helped": _frac(lambda r: r["delta_loss_branch1_only"] > 0),
            "probe/frac_branch2_helped": _frac(lambda r: r["delta_loss_branch2_only"] > 0),
            "probe/frac_combined_helped": _frac(lambda r: r["delta_loss_combined"] > 0),
            "probe/frac_branch1_better_than_branch2": _frac(
                lambda r: r["delta_loss_branch1_only"] > r["delta_loss_branch2_only"]
            ),
            "probe/num_steps": float(len(records)),
        }
        return agg
