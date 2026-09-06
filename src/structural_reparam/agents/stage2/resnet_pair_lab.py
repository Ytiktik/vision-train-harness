"""The paper's placement ladder on the smallest ResNet analogue of its own rig.

Written 2026-09-05. The two ResNet arcs so far both put ONE intervention on a
ResNet-18 that is much larger and deeper than the paper's rig: the last-block
radial boost (`resnet_lab.py`) and the first-block two-branch pair on the stem
(`resnet_stem_lab.py`). Both found the mechanism present and the benefit absent.
That leaves two candidate explanations tangled together -- the architecture, and
the size -- because no ResNet has yet been run at the paper's own scale.

This module unties them. It builds a ResNet whose stage ladder, strides and
convolution count are as close to the paper's depth-5 plain stack as a residual
network can be, and then runs the paper's own placement ladder on it:
`single`, `pair_first`, `pair_last`, `pair_both`. If the gains come back at the
paper's size, the earlier nulls were size. If they stay absent, they are the
residual architecture.

--------------------------------------------------------------------------------
1. The cell: what "the smallest ResNet analogue" means
--------------------------------------------------------------------------------
The paper's default cell is `PlacedSharedScaleRepVGGCifar` with stage channels
32/64/128, stage strides 1/2/2 and stage blocks 1/2/2 -- five convolution blocks,
291k parameters with the 100-class head.

The analogue here is `_CifarResNet` with widths 32/64/128 and ONE BasicBlock per
stage. That keeps the same channel ladder, the same two stride-2 steps and the
same spatial resolutions; it costs seven 3x3 convolutions rather than five (a
stem plus two per block) and two extra 1x1 downsample convolutions, and about
320k parameters. Nothing closer exists: a BasicBlock cannot have one convolution
without ceasing to be a residual block, so seven is the minimum a three-stage
residual network with a stem can have.

--------------------------------------------------------------------------------
2. The positions, and what the arms mean
--------------------------------------------------------------------------------
In the paper's stack every block is a `SharedScaleRepVGGBlock` and the arm is a
branch count per block, so `single` is the all-one-branch network rather than
some other architecture. The same rule is used here: EVERY 3x3 convolution and
its normalization becomes a `SharedScaleRepVGGBlock`, and an arm is a branch
count for each. In forward order the seven positions are

    0   conv1  / bn1              the stem, outside every residual branch
    1   layer1.0.conv1 / bn1      stage 1, first convolution of the block
    2   layer1.0.conv2 / bn2      stage 1, the addend entering the addition
    3   layer2.0.conv1 / bn1
    4   layer2.0.conv2 / bn2
    5   layer3.0.conv1 / bn1
    6   layer3.0.conv2 / bn2      the last convolution in the network

so `pair_first` is [2,1,1,1,1,1,1], `pair_last` is [1,1,1,1,1,1,2] and
`pair_both` is [2,1,1,1,1,1,2]. The 1x1 downsample convolutions of stages 2 and
3 are left alone: they carry the skip stream, they are not 3x3, and the paper has
no counterpart to them.

Two structural facts are recorded here rather than assumed away, because they are
the first things to suspect if a result comes out strange.

**Position 0 is the paper's block 0 exactly.** A 3x3 stride-1 convolution on the
raw images outside any residual branch. This is the position `resnet_stem_lab.py`
already measured, where the separation axis parks on the input covariance's top
eigendirection and the preconditioner is cooled by the paper's own factor.

**Position 6 is NOT the paper's last block.** In the plain stack the last block's
gamma multiplies everything reaching the classifier, which is what makes the
radial boost a scale. Here `bn2` is applied before the residual addition, so this
gamma scales ONE ADDEND of a sum against a skip path it does not touch. That is
the structural obstruction the last-block arc ran into, and it is why a pair
here is a weaker claim than a pair in the plain stack's last block. The
experiment measures what a pair does in that position; it does not assert the two
positions are the same object.

--------------------------------------------------------------------------------
3. Where the activation goes
--------------------------------------------------------------------------------
`SharedScaleRepVGGBlock` ends in a ReLU, because in the paper's stack every block
is followed by one. A ResNet's ReLUs are not all placed that way -- the stem and
each `conv1` position are followed by one, each `conv2` position is not, because
the residual addition comes first and the ReLU follows that. So the rule here is
that the substitution replaces the convolution and the normalization and nothing
else: every block's own activation is set to an identity, and every ReLU in the
network stays the ResNet's own, exactly where the ResNet puts it.

The alternative -- letting the block keep its ReLU at the positions that are
followed by one, on the ground that ReLU is idempotent -- is not merely redundant
but wrong in this codebase: both activations are `inplace=True`, so the second
one overwrites the first's saved output and the backward pass raises. That was
caught by a dry run on 2026-09-05, and `tests/test_resnet_pair_ladder.py` keeps
both the rule and a gradient check that would have caught it.

--------------------------------------------------------------------------------
4. The ImageNet stem, and the networks it builds
--------------------------------------------------------------------------------
Added 2026-09-05 for the full-resolution campaign. Everything above runs at 32
pixels, on CIFAR-100 and on ImageNet32, and both found the two mechanisms firing
in the positions the paper predicts while the benefit stayed small. Every ImageNet
result the project has is at 32 pixels, so the next question is whether real
resolution changes either half of that, and answering it needs the 224-pixel input
stage: `build_paired_resnet(..., stem="imagenet")`.

The stem is a 3x3 stride-2 convolution followed by a 3x3 stride-2 max-pool, which
is torchvision's layout with one deliberate departure -- the kernel stays 3x3
rather than becoming 7x7. Position 0 is the position the spectral cooling
mechanism is measured at, and that mechanism is a statement about the covariance
of the 3x3 patches a kernel sees on raw pixels: the separation axis parks on that
covariance's top eigendirection, and the paper's selection predictor is built from
its spectrum. A 7x7 stem would pair a different operator against a 49-pixel patch
covariance, so nothing measured in the paper would carry over, and the mu_eff the
probe reports would not be comparable with any number the paper quotes. The
max-pool is kept because without it stage 1 would see 112x112 and every activation
map in the network would be four times the area, which changes the cell for a
reason that has nothing to do with the mechanism.

The two networks of the full-resolution campaign, at the canonical widths
64/128/256/512:

    ResNet-10   one BasicBlock per stage    nine 3x3 convolutions, positions 0-8
    ResNet-18   two BasicBlocks per stage   seventeen, positions 0-16

`pair_both` pairs position 0 and the last position and nothing else, which is the
paper's two-placement arm: the stem for spectral cooling, the last 3x3 convolution
for the radial boost.

**The junction the last pair sits at.** As section 2 says of the shallow ResNet,
the last position is `layer4.N.conv2`, whose normalization is applied before the
residual addition. Its gamma therefore scales ONE ADDEND of a sum, against a skip
path it does not touch, and not the logits. That is why a last-position pair here
is a weaker claim than the paper's last-block pair in the plain stack, where the
same gamma multiplies everything the classifier sees, and it is the first thing to
recall if the last-position term comes out small. Nothing about the stem changes
that, and the ImageNet stem adds no position anywhere else: the network keeps one
pairable position per 3x3 convolution, and the max-pool carries no parameters.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn as nn

from structural_reparam.experiments.reparam_shared_scale.lab import SharedScaleRepVGGBlock
from structural_reparam.agents.stage2.resnet_lab import build_cifar_resnet, build_resnet_sgd
from structural_reparam.agents.stage2 import probe as _probe  # noqa: F401  (registers pair_channel_open)


# --------------------------------------------------------------------------------
# 1. The pairable positions
# --------------------------------------------------------------------------------

def pairable_positions(model: nn.Module) -> list[tuple[nn.Module, str, str, bool, str]]:
    """Every 3x3-convolution-plus-normalization position, in forward order.

    Each entry is ``(owner, conv_attr, norm_attr, followed_by_relu, path)``: the
    module that holds the pair of attributes, the two attribute names, whether a
    ReLU is applied to that position's output before anything else happens to it,
    and the dotted path for messages. The order is the order the forward pass
    visits them, which is also the order ``named_modules`` reports the blocks in
    once they are installed, so a probe's block index matches this list's index.
    """
    out: list[tuple[nn.Module, str, str, bool, str]] = [(model, "conv1", "bn1", True, "conv1")]
    for i in range(1, int(model.n_stages) + 1):
        stage = getattr(model, f"layer{i}")
        for j, block in enumerate(stage):
            out.append((block, "conv1", "bn1", True, f"layer{i}.{j}.conv1"))
            out.append((block, "conv2", "bn2", False, f"layer{i}.{j}.conv2"))
    return out


# --------------------------------------------------------------------------------
# 2. The model
# --------------------------------------------------------------------------------

def build_paired_resnet(num_classes: int = 100,
                        widths: Sequence[int] = (32, 64, 128),
                        blocks_per_stage: Sequence[int] = (1, 1, 1),
                        block_branches: Sequence[int] | None = None,
                        mode: str = "shared_gamma",
                        head_norm: bool = False,
                        stem: str = "cifar",
                        **_: object) -> nn.Module:
    """A shallow CIFAR ResNet whose 3x3 positions are N-branch shared-scale blocks.

    ``block_branches`` gives one branch count per position of
    ``pairable_positions``, in forward order. ``None`` leaves the network stock:
    torchvision's own convolutions and BatchNorms, untouched, which is the control
    that says what the block substitution alone costs. Everything the substitution
    does not reach -- the downsample paths, the pooling, the classifier and the
    initialization of all of them -- is `build_cifar_resnet`'s.

    The stock arm is expressed as a branch count rather than as a different model
    target because the trainer takes the model target from the model block and
    ignores a `target` key on a variant, so a stock arm written as another target
    would silently be built from this one.

    ``stem="imagenet"`` builds the same ladder on torchvision's 224-pixel input
    stage; see section 4 of the module docstring for what that does and does not
    change about the positions.
    """
    model = build_cifar_resnet(num_classes=num_classes, widths=widths,
                               blocks_per_stage=blocks_per_stage, head_norm=head_norm,
                               stem=stem)
    positions = pairable_positions(model)
    if block_branches is None:
        print(f"[pair ladder] stock ResNet, untouched; {len(positions)} pairable positions; "
              f"parameters {sum(p.numel() for p in model.parameters())}", flush=True)
        return model

    branches = [int(n) for n in block_branches]
    if len(branches) != len(positions):
        raise ValueError(
            f"block_branches has {len(branches)} entries but the network has "
            f"{len(positions)} pairable positions: {[p[4] for p in positions]}"
        )

    before = sum(p.numel() for p in model.parameters())
    for (owner, conv_attr, norm_attr, _relu_after, _path), n in zip(positions, branches):
        conv = getattr(owner, conv_attr)
        block = SharedScaleRepVGGBlock(
            conv.in_channels, conv.out_channels, stride=int(conv.stride[0]),
            num_branches=n, mode=str(mode),
        )
        # The block replaces the convolution and the normalization only; see
        # section 3 of the module docstring. Every ReLU stays the ResNet's own.
        block.activation = nn.Identity()
        setattr(owner, conv_attr, block)
        setattr(owner, norm_attr, nn.Identity())
    after = sum(p.numel() for p in model.parameters())
    print(f"[pair ladder] branches {branches} over positions "
          f"{[p[4] for p in positions]}; mode={mode}; "
          f"parameters {before} -> {after}", flush=True)
    return model


def pair_blocks(model: nn.Module) -> list[SharedScaleRepVGGBlock]:
    """The shared-scale blocks in the order the probe indexes them."""
    return [m for m in model.modules() if isinstance(m, SharedScaleRepVGGBlock)]


# --------------------------------------------------------------------------------
# 3. The optimizer
# --------------------------------------------------------------------------------

def build_pair_resnet_sgd(params: Iterable[nn.Parameter], model: nn.Module, lr: float,
                          block_lr_modules: Sequence[str] | None = None,
                          gamma_mult: float = 1.0, beta_mult: float = 1.0,
                          **kwargs) -> torch.optim.SGD:
    """`resnet_lab.build_resnet_sgd`, named from this module.

    ``block_lr_modules`` names substituted blocks -- the ones this module builds,
    whose scale is ``<block>.gamma`` and whose shifts are ``<block>.betas.*``
    rather than a BatchNorm ``.weight`` and ``.bias``. Each named block's scale
    gets ``lr * gamma_mult`` and each of its shifts ``lr * beta_mult``, which is
    the single-branch stand-in for a closed pair that section 4.1 uses. Leave
    ``beta_mult`` at 1 to move the scale alone: the paper's Table 23 finds the
    scale carries almost all of the gain while the shift buys a fifth of it and
    pays 40 percent of the test penalty.

    A block named here may carry one branch or two. The arms that use it carry
    one, so the multiplier is measured against the same `single` the pairs are.

    Delegating rather than reimplementing keeps every ResNet campaign on one
    recipe and one gate: kernel-only weight decay is decided by ``ndim >= 2``, so
    each block's shared gamma and its per-branch betas are decay-free for the same
    reason a BatchNorm's affine parameters are, and the gate does not need to know
    that some convolutions are pairs.

    Naming this module as the optimizer target also guarantees that importing it
    -- and with it the `pair_channel_open` registration above -- happens before
    the trainer reads the probe configuration.
    """
    extra: dict[str, float] = dict(kwargs.pop("extra_boosts", None) or {})
    if block_lr_modules:
        known = dict(model.named_modules())
        missing = [n for n in block_lr_modules if n not in known]
        if missing:
            raise RuntimeError(f"block_lr_modules names no such module: {missing}")
        for name in block_lr_modules:
            block = known[name]
            have = {pn for pn, _ in block.named_parameters()}
            if "gamma" not in have:
                raise RuntimeError(
                    f"block_lr_modules names {name!r}, which has no `gamma`; it is a "
                    f"{type(block).__name__}, not a substituted shared-scale block")
            extra[f"{name}.gamma"] = float(gamma_mult)
            for pn in sorted(p for p in have if p.startswith("betas.")):
                extra[f"{name}.{pn}"] = float(beta_mult)
    return build_resnet_sgd(params, model=model, lr=lr,
                            gamma_mult=gamma_mult, beta_mult=beta_mult,
                            extra_boosts=extra or None, **kwargs)
