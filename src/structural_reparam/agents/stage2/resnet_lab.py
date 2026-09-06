"""CIFAR ResNet-18 and a module-targeted affine learning-rate multiplier.

Written 2026-09-04 to test whether the last-block radial boost of the paper's
section 4.1 survives on a standard architecture under a standard recipe. Nothing
here is used by any other campaign.

Three sections: the model, the set of BatchNorms that own the logit scale, and
the optimizer that boosts their learning rate with a gate that proves it did.

--------------------------------------------------------------------------------
Why three BatchNorms, and which
--------------------------------------------------------------------------------
In the paper's plain conv-BatchNorm stack the last block's scale is a single
knob: nothing downstream renormalizes it, so doubling that gamma doubles the
logits. A ResNet has no such single knob, because its blocks add rather than
compose.

Reading the torchvision BasicBlock (`torchvision/models/resnet.py`, `forward` at
line 89), `bn2` is applied to the conv path BEFORE the residual addition, so the
skip path never passes through it. And `_make_layer` (line 239) gives the FIRST
block of a stage a downsample shortcut -- a 1x1 conv plus its own BatchNorm --
whenever the stride or the channel count changes, while later blocks in the stage
get a bare identity.

For stage 4 of ResNet-18 that means the whole carried stream is renormalized once,
at `layer4.0.downsample.1`, which erases the scales of stages 1 to 3. After that
point only additions happen, so exactly three BatchNorms set what reaches the
classifier:

    layer4.0.downsample.1   the skip stream entering stage 4
    layer4.0.bn2            the first block's conv-path contribution
    layer4.1.bn2            the second block's conv-path contribution

The two `bn1`s are not in the set: each is followed by another BatchNorm inside
the same conv path, and BatchNorm is invariant to the scale of its input, so a
`bn1` gamma has no effect on the output at all.

Verified numerically on 2026-09-04 by scaling gamma and beta of candidate sets by
3 in train mode (batch statistics) with the classifier bias zeroed, and comparing
against exactly tripled logits: all three together give a relative error of 1e-05,
any single one of them 1.5 to 2.0, and adding the two `bn1`s leaves the 1e-05
unchanged. `tests/test_resnet_scale_setting.py` keeps that check.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn as nn
from torchvision.models import resnet18


# --------------------------------------------------------------------------------
# 1. The model
# --------------------------------------------------------------------------------

def build_cifar_resnet18(num_classes: int = 100, **_: object) -> nn.Module:
    """torchvision ResNet-18 with the standard CIFAR stem.

    The ImageNet stem (7x7 stride-2 conv, 3x3 max-pool) destroys most of a 32x32
    image before the network starts. The CIFAR adaptation everyone uses replaces
    it with a single 3x3 stride-1 conv and drops the max-pool; everything after is
    untouched, so the module names -- and therefore the paths this file targets --
    are torchvision's own.
    """
    model = resnet18(num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


class _CifarResNet(nn.Module):
    """A width-scalable ResNet with torchvision's exact block structure.

    Named for the stem it was written with; ``stem="imagenet"`` gives it
    torchvision's 224-pixel input stage instead, and nothing else differs.

    torchvision's ResNet hardcodes 64/128/256/512, so a narrower variant cannot be
    built from it. This mirrors `torchvision.models.resnet.BasicBlock` exactly --
    two 3x3 convolutions each followed by a BatchNorm, ReLU between them, `bn2`
    applied BEFORE the residual addition, and a 1x1-conv-plus-BatchNorm downsample
    on the first block of a stage whenever the stride or channel count changes --
    so the scale-setting argument carries over unchanged. Module names match
    torchvision's (`layer4.0.downsample.1`, `layer4.1.bn2`, ...).
    """

    def __init__(self, num_classes: int = 100, widths: Sequence[int] = (64, 128, 256, 512),
                 blocks_per_stage: Sequence[int] = (2, 2, 2, 2),
                 head_norm: bool = False, stem: str = "cifar") -> None:
        super().__init__()
        from torchvision.models.resnet import BasicBlock
        if stem not in ("cifar", "imagenet"):
            raise ValueError(f"stem must be 'cifar' or 'imagenet', got {stem!r}")
        widths = [int(w) for w in widths]
        self.stem = stem
        self.inplanes = widths[0]
        # The stem (2026-09-05, the ImageNet option). `cifar` is the adaptation
        # everyone uses at 32 pixels: a 3x3 stride-1 convolution and no pooling,
        # because torchvision's own stem throws away most of a 32x32 image before
        # the network starts. `imagenet` restores the downsampling torchvision does
        # at 224 pixels -- a stride-2 convolution followed by a 3x3 stride-2
        # max-pool, so stage 1 sees 56x56 -- but keeps the kernel at 3x3 rather
        # than torchvision's 7x7. That last choice is deliberate and is the whole
        # reason this option exists: position 0 is the pair the paper's spectral
        # cooling mechanism is measured at, and the mechanism is a statement about
        # the covariance of the patches a 3x3 kernel sees on raw pixels. A 7x7 stem
        # would pair a different operator against a 49-pixel patch covariance, and
        # nothing measured in the paper would transfer to it.
        stem_stride = 2 if stem == "imagenet" else 1
        self.conv1 = nn.Conv2d(3, widths[0], kernel_size=3, stride=stem_stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(widths[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = (nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                        if stem == "imagenet" else nn.Identity())
        strides = [1] + [2] * (len(widths) - 1)
        for i, (w, n, s) in enumerate(zip(widths, blocks_per_stage, strides), start=1):
            setattr(self, f"layer{i}", self._make_layer(BasicBlock, w, n, s))
        self.n_stages = len(widths)
        # The head normalization (2026-09-05). A ResNet is the outlier among modern
        # architectures in going straight from stage four's ADDITION into global
        # average pooling: ViT and ConvNeXt end with a LayerNorm before the head,
        # MobileNetV2 and V3 with a conv-BatchNorm head, and a fused MobileOne with
        # its own BatchNorm. Because a ResNet has none, there is no single parameter
        # whose scaling multiplies the logits -- stage four's three normalizations
        # are three addends of a sum, so boosting any of them changes the MIXTURE
        # rather than the scale. `head_norm=True` inserts the missing layer, after
        # layer4 and before avgpool, so nothing but linear operations (pooling, the
        # classifier) sit downstream and its gamma alone owns the logit scale --
        # the property the paper's plain stack has in its last block's gamma.
        self.head_norm = nn.BatchNorm2d(widths[-1]) if head_norm else None
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(widths[-1], num_classes)
        for m in self.modules():                      # torchvision's initialization
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes: int, blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        layers += [block(self.inplanes, planes) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        for i in range(1, self.n_stages + 1):
            x = getattr(self, f"layer{i}")(x)
        if self.head_norm is not None:
            x = self.head_norm(x)
        return self.fc(torch.flatten(self.avgpool(x), 1))


def build_cifar_resnet_from_ckpt(
    num_classes: int = 100,
    width_mult: float = 1.0,
    blocks_per_stage: Sequence[int] = (2, 2, 2, 2),
    *,
    ckpt_group: str,
    ckpt_variant: str,
    ckpt_epoch: int,
    ckpt_seed: "int | str" = "auto",
    ckpt_cache: str = "outputs/ckpt_cache",
    **_: object,
) -> nn.Module:
    """A width-scaled CIFAR ResNet loaded from a finished run's checkpoint.

    For the reheat protocol: rather than training from scratch with a hot scale,
    continue an already-converged network. Reuses the campaign-agnostic artifact
    fetch of ``reparam_split_rescue_cifar`` (artifact ``ckpt_<group>_<variant>_s<seed>``,
    file ``ckpt_ep<epoch>.pt``), which also validates that the checkpoint's own
    recorded seed and epoch match what was asked for.

    Loading is strict, so a width or block-count mismatch against the source run
    raises rather than silently training a partly-random network.
    """
    from structural_reparam.experiments.reparam_split_rescue_cifar.lab import (
        _load_checkpoint_state,
    )

    model = build_cifar_resnet(num_classes=num_classes, width_mult=width_mult,
                               blocks_per_stage=blocks_per_stage)
    state, seed = _load_checkpoint_state(ckpt_group, ckpt_variant, ckpt_epoch,
                                         ckpt_seed, ckpt_cache)
    model.load_state_dict(state, strict=True)
    print(f"[reheat] loaded {ckpt_group}/{ckpt_variant} epoch {ckpt_epoch} seed {seed} "
          f"into a width-{width_mult} CIFAR ResNet", flush=True)
    return model


HEAD_NORM_MODULE = "head_norm"


def build_cifar_resnet(num_classes: int = 100, width_mult: float = 1.0,
                       blocks_per_stage: Sequence[int] = (2, 2, 2, 2),
                       head_norm: bool = False,
                       widths: Sequence[int] | None = None,
                       stem: str = "cifar", **_: object) -> nn.Module:
    """CIFAR ResNet-18 with its channel counts scaled by ``width_mult``.

    ``width_mult=1.0`` reproduces ResNet-18's 64/128/256/512; 0.25 gives
    16/32/64/128. Capacity is the only thing that changes -- depth, block
    structure, stem and recipe are identical -- so a cell built this way isolates
    train headroom from every other difference.

    ``widths`` names the stage channel counts outright and overrides
    ``width_mult``, so a configuration can ask for a stage ladder that is not a
    multiple of ResNet-18's. Its length sets the number of stages, and the stride
    pattern stays 1 then 2 for every stage after the first, as it is at every
    width. It exists for the shallow ResNet of ``resnet_pair_lab.py``, whose
    32/64/128 ladder over three stages matches the paper's plain stack rather
    than ResNet-18.

    ``stem`` picks the input stage: ``"cifar"`` is the 3x3 stride-1 convolution
    with no pooling that every 32-pixel adaptation uses, and ``"imagenet"`` is
    torchvision's downsampling layout for 224-pixel images -- a stride-2
    convolution then a 3x3 stride-2 max-pool -- with the kernel held at 3x3 so
    that position 0 stays the paper's own block on raw pixels. Nothing after the
    stem changes, so an ImageNet-stem network and a CIFAR-stem one of the same
    widths and block counts differ in exactly two modules.
    """
    if widths is None:
        widths = [max(1, round(w * width_mult)) for w in (64, 128, 256, 512)]
    else:
        widths = [int(w) for w in widths]
        if len(widths) != len(blocks_per_stage):
            raise ValueError(
                f"widths names {len(widths)} stages but blocks_per_stage names "
                f"{len(blocks_per_stage)}"
            )
    return _CifarResNet(num_classes=num_classes, widths=widths,
                        blocks_per_stage=blocks_per_stage, head_norm=head_norm,
                        stem=stem)


# --------------------------------------------------------------------------------
# 2. The scale-setting BatchNorms
# --------------------------------------------------------------------------------

SCALE_SETTING_MODULES: tuple[str, ...] = (
    "layer4.0.downsample.1",
    "layer4.0.bn2",
    "layer4.1.bn2",
)


def scale_setting_modules(model: nn.Module, stage: str = "layer4") -> tuple[str, ...]:
    """The BatchNorms in the last stage whose joint scaling multiplies the logits.

    The rule, from the block structure above: after the last stage's downsample
    BatchNorm renormalizes the whole carried stream, only additions happen, so the
    set is that downsample BatchNorm plus the `bn2` of every block in the stage.
    Each `bn1` is excluded because a later BatchNorm in the same conv path divides
    its scale straight out. Written as a rule rather than a list so it stays right
    when the width or the block count changes.
    """
    names: list[str] = []
    blocks = model.get_submodule(stage)
    for i, block in enumerate(blocks):
        if getattr(block, "downsample", None) is not None:
            names.append(f"{stage}.{i}.downsample.1")
        names.append(f"{stage}.{i}.bn2")
    return tuple(names)

ALL_BATCHNORM = "__all_batchnorm__"


def _resolve_modules(model: nn.Module, spec: Sequence[str] | str) -> list[str]:
    if spec == ALL_BATCHNORM:
        return [n for n, m in model.named_modules() if isinstance(m, nn.BatchNorm2d)]
    names = list(spec)
    known = dict(model.named_modules())
    missing = [n for n in names if n not in known]
    if missing:
        raise RuntimeError(f"affine_lr_modules names no such module: {missing}")
    wrong = [n for n in names if not isinstance(known[n], nn.BatchNorm2d)]
    if wrong:
        raise RuntimeError(f"affine_lr_modules names a non-BatchNorm module: {wrong}")
    return names


# --------------------------------------------------------------------------------
# 3. The optimizer, and the gate
# --------------------------------------------------------------------------------

def build_resnet_sgd(
    params: Iterable[nn.Parameter],
    model: nn.Module,
    lr: float,
    momentum: float = 0.0,
    weight_decay: float = 0.0,
    nesterov: bool = False,
    kernel_only_wd: bool = True,
    affine_lr_modules: Sequence[str] | str | None = None,
    gamma_mult: float = 1.0,
    beta_mult: float = 1.0,
    extra_boosts: dict[str, float] | None = None,
    **_: object,
) -> torch.optim.SGD:
    """SGD with kernel-only weight decay and a learning-rate multiplier on the
    affine parameters of named BatchNorm modules.

    ``kernel_only_wd`` puts ``weight_decay`` on parameters with ``ndim >= 2``
    (conv kernels and ``fc.weight``) and zero on everything else, which is the
    "no bias decay" recipe; set it False for the classic all-parameter variant.
    ``affine_lr_modules`` is a list of module paths (or ``ALL_BATCHNORM``); the
    ``.weight`` of each -- BatchNorm's gamma -- gets ``lr * gamma_mult`` and the
    ``.bias`` -- its beta -- gets ``lr * beta_mult``. Groups are keyed by
    (weight decay, learning rate).

    ``extra_boosts`` maps a *parameter* name to its multiplier, for models whose
    scale does not live on a BatchNorm ``.weight``. The pair ladder's substituted
    blocks are the case that needs it: their scale is ``<block>.gamma`` and their
    shifts are ``<block>.betas.*``, so ``affine_lr_modules`` cannot see them.
    ``build_pair_resnet_sgd`` builds this map; the gate below then checks these
    boosts landed exactly as it checks the BatchNorm ones.

    The gate below raises rather than training a run whose parameter groups are
    not what the arm claims, and prints every group so the log carries the proof.
    """
    targets = _resolve_modules(model, affine_lr_modules) if affine_lr_modules else []
    boosted: dict[str, float] = {}
    for name in targets:
        boosted[f"{name}.weight"] = float(lr) * float(gamma_mult)
        boosted[f"{name}.bias"] = float(lr) * float(beta_mult)
    if extra_boosts:
        known = {n for n, _ in model.named_parameters()}
        unknown = [n for n in extra_boosts if n not in known]
        if unknown:
            raise RuntimeError(f"extra_boosts names no such parameter: {unknown}")
        for name, mult in extra_boosts.items():
            boosted[name] = float(lr) * float(mult)

    groups: dict[tuple[float, float], list[nn.Parameter]] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        wd = float(weight_decay) if (p.ndim >= 2 or not kernel_only_wd) else 0.0
        plr = boosted.get(name, float(lr))
        groups.setdefault((wd, plr), []).append(p)

    optimizer = torch.optim.SGD(
        [{"params": ps, "weight_decay": wd, "lr": plr} for (wd, plr), ps in groups.items()],
        lr=lr, momentum=momentum, nesterov=nesterov,
    )

    # --- the gate -------------------------------------------------------------
    name_of = {id(p): n for n, p in model.named_parameters()}
    seen: set[int] = set()
    offenders: list[str] = []
    summary: dict[str, float] = {}
    landed: dict[str, float] = {}

    for gi, group in enumerate(optimizer.param_groups):
        wd, glr = float(group["weight_decay"]), float(group["lr"])
        tensors = elements = 0
        for p in group["params"]:
            n = name_of.get(id(p), "<unnamed>")
            if id(p) in seen:
                offenders.append(f"{n} is in more than one group")
            seen.add(id(p))
            tensors += 1
            elements += p.numel()
            if kernel_only_wd and p.ndim < 2 and wd != 0.0:
                offenders.append(f"{n} has ndim={p.ndim} but weight_decay={wd}")
            if n in boosted:
                landed[n] = glr
        summary[f"group/{gi}_weight_decay"] = wd
        summary[f"group/{gi}_lr"] = glr
        summary[f"group/{gi}_tensors"] = tensors
        summary[f"group/{gi}_elements"] = elements
        print(f"[resnet gate] group {gi}: weight_decay={wd:g} lr={glr:g} "
              f"tensors={tensors} elements={elements}", flush=True)

    missing = [n for n, p in model.named_parameters() if p.requires_grad and id(p) not in seen]
    if missing:
        offenders.append("parameters in no optimizer group: " + ", ".join(missing[:8]))
    for n, want in boosted.items():
        got = landed.get(n)
        if got is None:
            offenders.append(f"{n} was targeted but is in no group")
        elif abs(got - want) > 1e-12:
            offenders.append(f"{n} should sit at lr {want:g} but sits at {got:g}")
    if offenders:
        raise RuntimeError("resnet optimizer gate FAILED:\n  " + "\n  ".join(offenders))

    for n in sorted(boosted):
        print(f"[resnet gate] boosted {n}: lr={landed[n]:g}", flush=True)
    print(f"[resnet gate] PASSED: {len(seen)} tensors in {len(optimizer.param_groups)} groups; "
          f"{len(boosted)} boosted ({len(targets)} BatchNorm modules, "
          f"{len(extra_boosts or {})} named parameters); "
          f"kernel_only_wd={kernel_only_wd}", flush=True)
    summary["gate_passed"] = 1
    summary["n_boosted_tensors"] = len(boosted)

    state = {"handle": None, "done": False}

    def _record_once(opt, *_a, **_k):  # pragma: no cover - needs a live run
        if state["done"]:
            return
        try:
            import wandb
            if wandb.run is None:
                return
            wandb.run.summary.update(summary)
        except Exception:
            pass
        state["done"] = True
        if state["handle"] is not None:
            state["handle"].remove()

    try:
        state["handle"] = optimizer.register_step_post_hook(_record_once)
    except AttributeError:
        pass
    return optimizer
