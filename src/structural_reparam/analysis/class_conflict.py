"""Per-class gradient conflict probe at the feature boundary.

For each class c, computes the mean gradient signal that flows back from the
loss to the feature extractor output (the pooled vector before the linear head):

    grad_c = W.T @ (mean_softmax_c - one_hot_c)        [shape: D]

where mean_softmax_c is the average predicted probability vector over all
class-c samples in the evaluation set.

Pairwise cosine similarity of these D-dimensional vectors gives the conflict
matrix: negative cosine means the two classes want the feature extractor to
move in opposite directions (high conflict); near-zero means orthogonal
(soft conflict); positive means aligned (no conflict).

This requires no backward pass — a single forward pass suffices.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader


class ClassConflictProbe:
    """Measure per-class gradient conflict at the feature extractor boundary.

    Assumes the model exposes `model.fc` (a Linear layer) as the classifier
    head, preceded by a global average pool. Computes a single forward pass
    over the provided loader, then derives the full 100×100 pairwise cosine
    conflict matrix with no backward pass.

    Call `compute()` periodically during training and log the returned stats
    dict; save the matrix artifact at the final epoch.
    """

    def __init__(self, model: nn.Module, num_classes: int) -> None:
        self._model = model
        self._num_classes = num_classes

    def compute(
        self,
        loader: DataLoader,
        device: torch.device,
    ) -> dict[str, object]:
        """Run the conflict probe.

        Returns a dict with:
          "stats": flat metric dict ready to log to W&B
          "cos_matrix": (C, C) CPU float tensor of pairwise cosines
          "present_classes": 1-D tensor of class indices that appeared in loader
          "top_conflicting_pairs": list of (class_a, class_b, cosine) sorted
              ascending by cosine (most conflicting first)
        """
        model = self._model
        C = self._num_classes

        model.eval()
        soft_sum = torch.zeros(C, C, device=device)
        class_counts = torch.zeros(C, device=device)

        with torch.no_grad():
            for images, targets in loader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                logits = model(images)
                probs = F.softmax(logits, dim=1)
                soft_sum.scatter_add_(
                    0, targets.unsqueeze(1).expand_as(probs), probs
                )
                class_counts.scatter_add_(
                    0, targets, torch.ones_like(targets, dtype=torch.float)
                )

        present = class_counts > 0
        mean_softmax = soft_sum.clone()
        mean_softmax[present] /= class_counts[present].unsqueeze(1)

        one_hot = torch.eye(C, device=device)
        residual = mean_softmax - one_hot  # [C, C]: how far predictions are from ideal

        # grad_c = W.T @ residual[c]  →  all at once: residual @ W  [C, D]
        fc_weight = model.fc.weight.detach()  # [C, D]
        grad_features = residual @ fc_weight  # [C, D]

        present_idx = present.nonzero(as_tuple=True)[0]
        grad_norm = F.normalize(grad_features[present_idx], dim=1)  # [C', D]
        cos_matrix = grad_norm @ grad_norm.T  # [C', C']

        C_prime = present_idx.shape[0]
        off_diag_mask = ~torch.eye(C_prime, dtype=torch.bool, device=device)
        off_diag = cos_matrix[off_diag_mask]

        stats: dict[str, float] = {
            "conflict/mean_cos": off_diag.mean().item(),
            "conflict/std_cos": off_diag.std().item(),
            "conflict/min_cos": off_diag.min().item(),
            "conflict/frac_negative": (off_diag < 0).float().mean().item(),
        }

        rows, cols = torch.triu_indices(C_prime, C_prime, offset=1, device=device)
        pair_cosines = cos_matrix[rows, cols]
        sorted_idx = pair_cosines.argsort()
        top_k = min(20, sorted_idx.shape[0])
        top_pairs: list[tuple[int, int, float]] = []
        for rank in range(top_k):
            i = sorted_idx[rank].item()
            class_a = present_idx[rows[i]].item()
            class_b = present_idx[cols[i]].item()
            top_pairs.append((int(class_a), int(class_b), float(pair_cosines[i].item())))

        return {
            "stats": stats,
            "cos_matrix": cos_matrix.cpu(),
            "present_classes": present_idx.cpu(),
            "top_conflicting_pairs": top_pairs,
        }


def log_conflict_to_wandb(
    result: dict[str, object],
    run: object,
    epoch: int,
    class_names: list[str] | None = None,
    log_matrix: bool = False,
) -> None:
    """Log conflict probe results to an active W&B run.

    Logs scalar stats every call; logs the top-20 conflict table and optionally
    the full matrix image only when log_matrix=True (intended for final epoch).
    """
    import wandb

    payload: dict[str, object] = dict(result["stats"])  # type: ignore[arg-type]

    top_pairs = result["top_conflicting_pairs"]
    if top_pairs:
        table = wandb.Table(columns=["epoch", "class_a", "class_b", "cosine"])
        for class_a, class_b, cos in top_pairs:
            name_a = class_names[class_a] if class_names else str(class_a)
            name_b = class_names[class_b] if class_names else str(class_b)
            table.add_data(epoch, name_a, name_b, round(cos, 4))
        payload["conflict/top_pairs"] = table

    if log_matrix:
        mat = result["cos_matrix"]  # type: ignore[assignment]
        if isinstance(mat, torch.Tensor):
            payload["conflict/matrix"] = wandb.Image(
                _matrix_to_pil(mat),
                caption=f"Pairwise gradient cosine conflict — epoch {epoch}",
            )

    run.log(payload, step=epoch)


def _matrix_to_pil(mat: torch.Tensor):
    """Convert a [-1, 1] cosine matrix to a PIL image (red=negative, blue=positive)."""
    from PIL import Image
    import numpy as np

    m = mat.float().numpy()
    rgb = np.zeros((*m.shape, 3), dtype=np.uint8)
    neg = (-m).clip(0, 1)
    pos = m.clip(0, 1)
    rgb[..., 0] = (neg * 255).astype(np.uint8)   # red channel = conflict
    rgb[..., 2] = (pos * 255).astype(np.uint8)   # blue channel = alignment
    return Image.fromarray(rgb).resize((400, 400), Image.NEAREST)
