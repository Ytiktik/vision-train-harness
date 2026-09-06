"""K_eff trajectory and update statistics during training.

Measures dimensionality of the path that K_eff traces through weight space, plus
per-epoch diagnostics (delta-K_eff norm, branch-weight-update alignment). All
metrics live on the fused equivalent kernel so 1-branch and N-branch models are
directly comparable.

All statistics are collected in a single snapshot at the end of each epoch —
no per-step hooks — to avoid repeated CPU interruptions during training.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)).item()


def _rank_pair_from_gram(gram: torch.Tensor) -> tuple[float, float] | None:
    """Effective/stable rank from an E×E Gram matrix M @ M.T.

    The squared singular values of M are the eigenvalues of M @ M.T. For our
    trajectory matrices E ≪ N (epochs vs K_eff params), so working in Gram
    space is dramatically cheaper than full SVD on M.
    """
    gram = 0.5 * (gram + gram.T)
    eigvals = torch.linalg.eigvalsh(gram)
    s2 = eigvals.clamp(min=0.0)
    s2 = s2[s2 > 1e-24]
    if s2.numel() == 0:
        return None
    p = s2 / s2.sum()
    eff_rank = (-(p * p.log()).sum()).exp().item()
    stable_rank = (s2.sum() / s2.max()).item()
    return eff_rank, stable_rank


def _rank_pair(M: torch.Tensor) -> tuple[float, float] | None:
    """SVD-entropy effective rank and stable rank of matrix M (rows are samples).

    Routes through the Gram trick when rows ≤ cols.
    """
    if M.shape[0] <= M.shape[1]:
        return _rank_pair_from_gram(M @ M.T)
    s = torch.linalg.svdvals(M)
    s2 = (s ** 2)
    s2 = s2[s2 > 1e-24]
    if s2.numel() == 0:
        return None
    p = s2 / s2.sum()
    eff_rank = (-(p * p.log()).sum()).exp().item()
    stable_rank = (s2.sum() / s2.max()).item()
    return eff_rank, stable_rank


def _trajectory_metrics(deltas: list[torch.Tensor]) -> dict[str, float]:
    """Effective/stable rank of the K_eff trajectory under several framings.

    `deltas[k]` = vec(K_eff(t_k) - K_eff(t_0)) on CPU as fp32. Rows of T.

    Emits:
      - `trajectory/*`: cumulative path from init (biased toward early-large moves).
      - `trajectory_centered/*`: rows mean-centered (PCA on the path), removing
        the "distance from init" bias.
      - `update/*`: per-epoch steps `K_eff(t_k) - K_eff(t_{k-1})`, capturing the
        diversity of actual update directions.
      - `update_normalized/*`: same step rows, each rescaled to unit norm so
        large updates do not dominate small ones.
    """
    if len(deltas) < 2:
        return {}
    out: dict[str, float] = {}

    T = torch.stack(deltas).double()
    E = T.shape[0]

    # Single E×N matmul; all four rank computations derive from this Gram.
    G = T @ T.T

    if (r := _rank_pair_from_gram(G)) is not None:
        out["trajectory/effective_rank"], out["trajectory/stable_rank"] = r

    # Centered Gram: H G H where H = I - 1/E * 11.T is the centering matrix.
    row_means = G.mean(dim=1, keepdim=True)
    grand_mean = row_means.mean()
    Gc = G - row_means - row_means.T + grand_mean
    if (r := _rank_pair_from_gram(Gc)) is not None:
        out["trajectory_centered/effective_rank"], out["trajectory_centered/stable_rank"] = r

    # Step deltas S = T[1:] - T[:-1]. S @ S.T = D G D.T where D applies the
    # row-difference, which on a Gram is equivalent to differencing rows then
    # columns of G.
    if E >= 3:
        Gd = G[1:] - G[:-1]
        Gs = Gd[:, 1:] - Gd[:, :-1]
        if (r := _rank_pair_from_gram(Gs)) is not None:
            out["update/effective_rank"], out["update/stable_rank"] = r

        # Row-normalized step Gram: diag(1/||S_i||) Gs diag(1/||S_i||).
        # ||S_i||^2 = Gs[i,i].
        diag = Gs.diagonal().clamp(min=0.0)
        mask = diag > 1e-24
        if mask.any():
            idx = mask.nonzero(as_tuple=True)[0]
            Gs_kept = Gs.index_select(0, idx).index_select(1, idx)
            inv_norms = diag.index_select(0, idx).rsqrt()
            Gn = Gs_kept * inv_norms.unsqueeze(0) * inv_norms.unsqueeze(1)
            if (r := _rank_pair_from_gram(Gn)) is not None:
                out["update_normalized/effective_rank"], out["update_normalized/stable_rank"] = r

    return out


class GradientProbe:
    """Per-epoch K_eff trajectory + update statistics.

    All statistics are collected in a single snapshot at the end of each epoch
    (via `epoch_stats()`). There are no per-step hooks, so training throughput
    is not interrupted between batches.

    Metrics emitted by `epoch_stats()` (means across blocks unless noted):
      - `trajectory/effective_rank`, `trajectory/stable_rank`: rank of the
        cumulative trajectory T (rows = K_eff(t_k) - K_eff(t_0)). Biased toward
        early-large directions because every later row still contains them.
      - `trajectory/effective_rank_max` / `_min`: per-block extremes.
      - `trajectory_centered/effective_rank`, `trajectory_centered/stable_rank`:
        same trajectory after subtracting the mean row (PCA-style), removing the
        "distance from init" bias.
      - `update/effective_rank`, `update/stable_rank`: rank of per-epoch step
        deltas K_eff(t_k) - K_eff(t_{k-1}) — diversity of actual update
        directions, not cumulative position.
      - `update_normalized/effective_rank`, `update_normalized/stable_rank`:
        step-delta rows rescaled to unit norm, so large updates do not dominate
        small ones.
      - `trajectory/distance_from_init`: ||K_eff(t) - K_eff(0)||, paired with
        rank metrics to disentangle "moves further" from "moves in more
        directions."
      - `update/cos_step_vs_total`: cos(K_eff(t) - K_eff(t-1), K_eff(t) - K_eff(0)).
        +1 = drifting straight away from init; near 0 = exploring sideways;
        negative = doubling back.
      - `branch/dominant_share`, `branch/share_entropy` (>=2 3x3 branches):
        per-block ||ΔK_branch_i|| share of the total per-epoch update across
        branches. High dominant_share / low entropy = one branch dominates.
      - `grad/branch_alignment` (multi-branch only): mean pairwise cosine of
        per-branch weight update vectors (fused kernel at epoch end minus epoch
        start), measuring whether branches move in the same direction.
      - `grad/delta_keff_norm`: ||K_eff(epoch_end) - K_eff(epoch_start)||,
        the magnitude of the per-epoch K_eff update.

    Memory: stores one CPU fp32 vector of K_eff per block per epoch. For a
    width=1, standard-depth CIFAR RepVGG (~2.4M K_eff params total) this is
    ~10 MB/epoch — bounded and fine through 200 epochs.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        from structural_reparam.base_models.repvgg import RepVGGBlock

        self._handles: list = []
        self._block_buffers: list[dict] = []
        self._blocks: list = []

        for module in model.modules():
            if not isinstance(module, RepVGGBlock) or module.deployed:
                continue

            buf: dict = {
                "num_3x3": module.num_3x3,
                "keff_init": None,           # set lazily on first snapshot
                "deltas": [],                # list[Tensor], CPU fp32, flat
                # Previous epoch's fused per-3x3-branch kernels (CPU fp32 flat).
                # Only the previous epoch is kept — constant memory.
                "prev_branch_kernels": None,   # list[Tensor] | None
                # Set by _take_snapshot each epoch.
                "this_epoch_branch_shares": None,
                "this_epoch_branch_updates": None,  # list[Tensor] | None
            }
            self._block_buffers.append(buf)
            self._blocks.append(module)

    def attach_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        """No-op: statistics are now collected once per epoch at snapshot time."""

    def _take_snapshot(self) -> None:
        """Append current K_eff (CPU fp32 delta from init) and compute per-branch
        update shares and update vectors against the previous epoch's fused kernels."""
        for block, buf in zip(self._blocks, self._block_buffers):
            with torch.no_grad():
                keff = block.equivalent_kernel().detach().to("cpu", dtype=torch.float32).flatten()
                curr_branch = [
                    block._fuse_conv_bn(branch)[0].detach().to("cpu", dtype=torch.float32).flatten()
                    for branch in block.conv3_branches
                ]

            if buf["keff_init"] is None:
                buf["keff_init"] = keff.clone()
                buf["deltas"].append(torch.zeros_like(keff))
                buf["this_epoch_branch_shares"] = None
                buf["this_epoch_branch_updates"] = None
            else:
                buf["deltas"].append(keff - buf["keff_init"])
                prev = buf["prev_branch_kernels"]
                if prev is not None and len(prev) == len(curr_branch):
                    updates = [c - p for c, p in zip(curr_branch, prev)]
                    norms = [u.norm().item() for u in updates]
                    total = sum(norms)
                    buf["this_epoch_branch_shares"] = (
                        [n / total for n in norms] if total > 1e-12 else None
                    )
                    buf["this_epoch_branch_updates"] = updates
                else:
                    buf["this_epoch_branch_shares"] = None
                    buf["this_epoch_branch_updates"] = None
            buf["prev_branch_kernels"] = curr_branch

    def epoch_stats(self) -> dict[str, float]:
        """Snapshot K_eff, then aggregate buffers into epoch-level metrics."""
        self._take_snapshot()

        rank_keys = (
            "trajectory/effective_rank",
            "trajectory/stable_rank",
            "trajectory_centered/effective_rank",
            "trajectory_centered/stable_rank",
            "update/effective_rank",
            "update/stable_rank",
            "update_normalized/effective_rank",
            "update_normalized/stable_rank",
        )
        per_block: dict[str, list[float]] = {k: [] for k in rank_keys}
        all_alignments: list[float] = []
        all_delta_norms: list[float] = []
        all_distances: list[float] = []
        all_cos_step_total: list[float] = []
        all_dominant_share: list[float] = []
        all_share_entropy: list[float] = []

        for buf in self._block_buffers:
            tm = _trajectory_metrics(buf["deltas"])
            for k in rank_keys:
                if k in tm:
                    per_block[k].append(tm[k])

            deltas = buf["deltas"]
            if len(deltas) >= 1:
                all_distances.append(deltas[-1].norm().item())
            if len(deltas) >= 2:
                total = deltas[-1]
                step = deltas[-1] - deltas[-2]
                tn, sn = total.norm().item(), step.norm().item()
                if tn > 1e-12 and sn > 1e-12:
                    all_cos_step_total.append(_cosine(step, total))

            shares = buf.get("this_epoch_branch_shares")
            if shares is not None and len(shares) >= 2:
                all_dominant_share.append(max(shares))
                ent = -sum(s * math.log(s) for s in shares if s > 1e-12)
                all_share_entropy.append(ent)

            if buf["num_3x3"] >= 2:
                updates = buf.get("this_epoch_branch_updates")
                if updates is not None and len(updates) >= 2:
                    pair_cosines = [
                        _cosine(updates[i], updates[j])
                        for i in range(len(updates))
                        for j in range(i + 1, len(updates))
                    ]
                    if pair_cosines:
                        all_alignments.append(sum(pair_cosines) / len(pair_cosines))

            if len(deltas) >= 2:
                all_delta_norms.append((deltas[-1] - deltas[-2]).norm().item())

        stats: dict[str, float] = {}
        for k, vals in per_block.items():
            if not vals:
                continue
            stats[k] = sum(vals) / len(vals)
            if k.endswith("/effective_rank"):
                stats[f"{k}_max"] = max(vals)
                stats[f"{k}_min"] = min(vals)
        if all_alignments:
            stats["grad/branch_alignment"] = sum(all_alignments) / len(all_alignments)
        if all_delta_norms:
            stats["grad/delta_keff_norm"] = sum(all_delta_norms) / len(all_delta_norms)
        if all_distances:
            stats["trajectory/distance_from_init"] = sum(all_distances) / len(all_distances)
        if all_cos_step_total:
            stats["update/cos_step_vs_total"] = sum(all_cos_step_total) / len(all_cos_step_total)
        if all_dominant_share:
            stats["branch/dominant_share"] = sum(all_dominant_share) / len(all_dominant_share)
            stats["branch/share_entropy"] = sum(all_share_entropy) / len(all_share_entropy)

        # For each layer: compute one scalar ratio across branches (min/max, avg/max of branch norms).
        # Then report avg/min/max of that scalar across all layers.
        # min/max across layers reveals whether imbalance is universal or layer-specific.
        w_min_over_max: list[float] = []  # one value per layer
        w_avg_over_max: list[float] = []
        g_min_over_max: list[float] = []
        g_avg_over_max: list[float] = []
        for block in self._blocks:
            if block.deployed:
                continue
            w_norms = [branch[0].weight.detach().norm().item() for branch in block.conv3_branches]
            g_norms = [branch[1].weight.detach().norm().item() for branch in block.conv3_branches]
            w_mx = max(w_norms)
            if w_mx > 0:
                w_min_over_max.append(min(w_norms) / w_mx)
                w_avg_over_max.append(sum(w_norms) / len(w_norms) / w_mx)
            g_mx = max(g_norms)
            if g_mx > 0:
                g_min_over_max.append(min(g_norms) / g_mx)
                g_avg_over_max.append(sum(g_norms) / len(g_norms) / g_mx)

        if w_min_over_max:
            stats["branch/w_norm_min_over_max/avg"] = sum(w_min_over_max) / len(w_min_over_max)
            stats["branch/w_norm_min_over_max/min"] = min(w_min_over_max)
            stats["branch/w_norm_min_over_max/max"] = max(w_min_over_max)
        if w_avg_over_max:
            stats["branch/w_norm_avg_over_max/avg"] = sum(w_avg_over_max) / len(w_avg_over_max)
            stats["branch/w_norm_avg_over_max/min"] = min(w_avg_over_max)
            stats["branch/w_norm_avg_over_max/max"] = max(w_avg_over_max)
        if g_min_over_max:
            stats["branch/gamma_norm_min_over_max/avg"] = sum(g_min_over_max) / len(g_min_over_max)
            stats["branch/gamma_norm_min_over_max/min"] = min(g_min_over_max)
            stats["branch/gamma_norm_min_over_max/max"] = max(g_min_over_max)
        if g_avg_over_max:
            stats["branch/gamma_norm_avg_over_max/avg"] = sum(g_avg_over_max) / len(g_avg_over_max)
            stats["branch/gamma_norm_avg_over_max/min"] = min(g_avg_over_max)
            stats["branch/gamma_norm_avg_over_max/max"] = max(g_avg_over_max)

        return stats

    def detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
