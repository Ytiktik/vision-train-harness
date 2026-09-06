"""Is the last block's gamma at a stationary point? For each checkpoint, run
N mini-batches (train mode, running stats frozen) and record the per-batch
gradient dL/dgamma of the last block; report the batch-mean gradient (signed;
negative means the loss wants gamma larger), its std across batches, and the
implied 'pressure' ratio |mean| / std, plus the same for the fc weight norm
direction and for the block-0 gamma."""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import torch, torch.nn.functional as F, torchvision, torchvision.transforms as T
from structural_reparam.experiments.reparam_pinned_norm.hessian_lambda_max import build_from_sd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-root", type=Path, required=True); ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44]); ap.add_argument("--epoch", type=int, default=100)
    ap.add_argument("--data-dir", default="data/cifar100"); ap.add_argument("--batches", type=int, default=100); ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--device", default="cuda")
    a = ap.parse_args(); dev = torch.device(a.device)
    tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])
    ds = torchvision.datasets.CIFAR100(a.data_dir, train=True, download=False, transform=tf)
    g = torch.Generator().manual_seed(0)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.batch, shuffle=True, generator=g, num_workers=2, drop_last=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("a") as fo:
        for var in a.variants:
            for s in a.seeds:
                f = a.ckpt_root / f"{var}_seed{s}" / f"ckpt_ep{a.epoch}.pt"
                if not f.exists(): print("missing", f); continue
                model, last = build_from_sd(torch.load(f, map_location="cpu")["state_dict"]); model.to(dev).train()
                for m in model.modules():
                    if hasattr(m, "momentum"): m.momentum = 0.0
                gam_last = dict(model.named_parameters())[last + ".gamma"]; gam0 = dict(model.named_parameters())["stages.0.0.gamma"]; fcw = model.fc.weight
                G_last, G0, G_fc = [], [], []
                it = iter(dl)
                for _ in range(a.batches):
                    x, y = next(it); x, y = x.to(dev), y.to(dev)
                    model.zero_grad(set_to_none=True); loss = F.cross_entropy(model(x), y); loss.backward()
                    G_last.append(gam_last.grad.detach().clone()); G0.append(gam0.grad.detach().clone())
                    G_fc.append((fcw.grad.detach() * fcw.detach()).sum().item() / fcw.detach().norm().item())  # radial gradient of fc
                GL = torch.stack(G_last); G0 = torch.stack(G0)
                # signed radial: gradient projected on sign(gamma) so negative = wants |gamma| larger
                sg = torch.sign(gam_last.detach()); mean_l = (GL.mean(0) * sg); std_l = GL.std(0)
                sg0 = torch.sign(gam0.detach()); mean_0 = (G0.mean(0) * sg0); std_0 = G0.std(0)
                res = dict(variant=var, seed=s, epoch=a.epoch, batches=a.batches,
                    last_mean_grad_med=float(mean_l.median()), last_mean_grad_mean=float(mean_l.mean()), last_frac_wants_larger=float((mean_l < 0).float().mean()),
                    last_std_med=float(std_l.median()), last_pressure_med=float((mean_l.abs() / std_l).median()), last_pressure_sum=float(mean_l.sum() / (std_l.pow(2).sum().sqrt())),
                    block0_mean_grad_med=float(mean_0.median()), block0_frac_wants_larger=float((mean_0 < 0).float().mean()), block0_pressure_med=float((mean_0.abs() / std_0).median()),
                    fc_radial_mean=float(torch.tensor(G_fc).mean()), fc_radial_std=float(torch.tensor(G_fc).std()))
                print(json.dumps(res), flush=True); fo.write(json.dumps(res) + "\n"); fo.flush()

if __name__ == "__main__":
    main()
