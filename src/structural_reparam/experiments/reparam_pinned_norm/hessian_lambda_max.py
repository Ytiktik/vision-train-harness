"""Top Hessian eigenvalue (power iteration, Hessian-vector products by double
backprop) of the training cross-entropy on a fixed batch, BatchNorm in train
mode, for checkpoints of the pinned-norm cell.  For each checkpoint reports
lambda_max over three parameter groups:
  all      : every trainable parameter,
  last     : the last block's parameters (kernels, gamma, betas),
  last_tan : the last block's kernels only, with the power vector projected
             onto the tangent space of the per-channel norm spheres (the
             directions the pinned optimizer actually moves in).
Threshold for SGD with momentum mu at lr eta: 2(1+mu)/eta.
Usage (on a GPU box):
  python -m structural_reparam.experiments.reparam_pinned_norm.hessian_lambda_max \
     --ckpt-root outputs/pinned_c100_d3_wd0/checkpoints --variants single_d3_pinned shared_gamma_d3_pinned \
     --seeds 42 43 44 --epochs 50 100 --data-dir data/cifar100 --batch 1024 --iters 40 --out outputs/pinned_c100_d3_wd0/lambda_max.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

from structural_reparam.experiments.reparam_pinned_norm.lab import PlacedSharedScaleRepVGGCifar


def build_from_sd(sd: dict, num_classes: int = 100, single_beta: bool = False) -> PlacedSharedScaleRepVGGCifar:
    blocks = sorted({m.group(1) for k in sd for m in [re.match(r"(stages\.\d+\.\d+)\.", k)] if m},
                    key=lambda s: tuple(int(x) for x in s.split(".")[1:]))
    branches = [len({k for k in sd if k.startswith(b + ".convs.") and k.endswith(".weight")}) for b in blocks]
    stage_ids = sorted({int(b.split(".")[1]) for b in blocks})
    stage_blocks = [sum(1 for b in blocks if int(b.split(".")[1]) == s) for s in stage_ids]
    sbb = [i for i, b in enumerate(branches) if b >= 2] if single_beta else []
    m = PlacedSharedScaleRepVGGCifar(num_classes=num_classes, stage_blocks=stage_blocks, block_branches=branches, mode="shared_gamma", single_beta_blocks=sbb)
    m.load_state_dict(sd, strict=True)
    return m, blocks[-1]


# (mean, std, torchvision class, extra kwargs) per dataset, matching the training loaders
_DATASETS = {
    "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762), "CIFAR100", {"train": True}, None),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616), "CIFAR10", {"train": True}, None),
    "stl10": ((0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713), "STL10", {"split": "train"}, 64),
}


def fixed_batch(data_dir: str, n: int, device, dataset: str = "cifar100"):
    """A fixed, un-augmented training batch — the sharpness probe must not vary
    with the augmentation draw."""
    if dataset not in _DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}; known: {sorted(_DATASETS)}")
    mean, std, cls, kw, size = _DATASETS[dataset]
    ops = ([T.Resize(size)] if size else []) + [T.ToTensor(), T.Normalize(mean, std)]
    ds = getattr(torchvision.datasets, cls)(data_dir, download=False, transform=T.Compose(ops), **kw)
    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(len(ds), generator=g)[:n]
    xs, ys = zip(*[ds[int(i)] for i in idx])
    return torch.stack(xs).to(device), torch.tensor(ys).to(device)


def _power(model, params, x, y, iters, project, shift, seed):
    torch.manual_seed(seed)
    v = [torch.randn_like(p) for p in params]
    if project: project(v)
    nrm = torch.sqrt(sum((a * a).sum() for a in v)); v = [a / nrm for a in v]
    lam = 0.0; conv = 1.0
    for it in range(iters):
        model.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), y)
        grads = torch.autograd.grad(loss, params, create_graph=True)
        hv = torch.autograd.grad(sum((g * a).sum() for g, a in zip(grads, v)), params, retain_graph=False)
        hv = [h.detach() + shift * a for h, a in zip(hv, v)]
        if project: project(hv)
        lam_new = float(sum((h * a).sum() for h, a in zip(hv, v)))
        nrm = torch.sqrt(sum((h * h).sum() for h in hv)).clamp_min(1e-20)
        v = [h / nrm for h in hv]
        conv = abs(lam_new - lam) / max(abs(lam_new), 1e-12); lam = lam_new
    return lam, conv


def lambda_max(model, params, x, y, iters: int, project=None, seed: int = 0) -> tuple[float, float]:
    """Top (most positive) eigenvalue: pass 1 finds the largest-magnitude
    eigenvalue lam1; pass 2 runs on H + |lam1| I, whose spectrum is >= 0, so its
    largest-magnitude eigenvalue is lam_max + |lam1|."""
    lam1, _ = _power(model, params, x, y, iters, project, 0.0, seed)
    if lam1 >= 0:
        # lam1 is already the top eigenvalue unless a more negative one exists;
        # a shifted pass settles both cases.
        pass
    c = abs(lam1)
    lam2, conv = _power(model, params, x, y, iters, project, c, seed + 1)
    return lam2 - c, conv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-root", type=Path, required=True)
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, nargs="+", default=[50, 100])
    ap.add_argument("--data-dir", default="data/cifar100")
    ap.add_argument("--dataset", default="cifar100", choices=("cifar100", "cifar10", "stl10"))
    ap.add_argument("--num-classes", type=int, default=100)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = torch.device(a.device)
    x, y = fixed_batch(a.data_dir, a.batch, dev, a.dataset)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("a") as fo:
        for var in a.variants:
            for s in a.seeds:
                for ep in a.epochs:
                    f = a.ckpt_root / f"{var}_seed{s}" / f"ckpt_ep{ep}.pt"
                    if not f.exists():
                        print("missing", f); continue
                    ck = torch.load(f, map_location="cpu")
                    model, last = build_from_sd(ck["state_dict"], num_classes=a.num_classes); model.to(dev).train()
                    for m in model.modules():  # BN train mode but do not update running stats
                        if hasattr(m, "momentum"): m.momentum = 0.0
                    allp = [p for p in model.parameters() if p.requires_grad]
                    lastp = [p for n, p in model.named_parameters() if n.startswith(last + ".")]
                    lastk = [p for n, p in model.named_parameters() if n.startswith(last + ".convs.")]
                    def proj(vs, ks=lastk):
                        for vv, w in zip(vs, ks):
                            W = w.detach().view(w.shape[0], -1); what = W / W.norm(dim=1, keepdim=True)
                            V = vv.view(vv.shape[0], -1); V.sub_((V * what).sum(1, keepdim=True) * what)
                    with torch.no_grad():
                        loss0 = F.cross_entropy(model(x), y).item()
                    res = {"variant": var, "seed": s, "epoch": ep, "loss_batch": loss0, "batch": a.batch}
                    blocks = sorted({m_.group(1) for n, _ in model.named_parameters() for m_ in [re.match(r"(stages\.\d+\.\d+)\.", n)] if m_},
                                    key=lambda b: tuple(int(t) for t in b.split(".")[1:]))
                    groups = [("all", allp, None), ("last", lastp, None), ("last_tan", lastk, proj)]
                    def make_proj(ks):
                        def pr(vs, ks=ks):
                            for vv, w in zip(vs, ks):
                                W = w.detach().view(w.shape[0], -1); what = W / W.norm(dim=1, keepdim=True)
                                V = vv.view(vv.shape[0], -1); V.sub_((V * what).sum(1, keepdim=True) * what)
                        return pr
                    for b in blocks:
                        if b == last:
                            continue
                        groups.append((b, [p for n, p in model.named_parameters() if n.startswith(b + ".")], None))
                        bk = [p for n, p in model.named_parameters() if n.startswith(b + ".convs.")]
                        groups.append((b + "_tan", bk, make_proj(bk)))
                    groups.append(("fc", [p for n, p in model.named_parameters() if n.startswith("fc.")], None))
                    groups.append(("gammas", [p for n, p in model.named_parameters() if n.endswith(".gamma")], None))
                    groups.append(("kernels", [p for n, p in model.named_parameters() if ".convs." in n], None))
                    for name, params, pr in groups:
                        lam, conv = lambda_max(model, params, x, y, a.iters, project=pr)
                        res[name] = lam; res[name + "_conv"] = conv
                    print(json.dumps(res), flush=True); fo.write(json.dumps(res) + "\n"); fo.flush()


if __name__ == "__main__":
    main()
