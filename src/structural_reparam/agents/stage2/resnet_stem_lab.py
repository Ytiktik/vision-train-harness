"""A two-branch pair on the ResNet stem: the paper's block-0 mechanism on a
standard architecture.

Written 2026-09-05, as the first-block counterpart to `resnet_lab.py`'s last-block
radial boost. Nothing here is used by any other campaign, and nothing here edits
`resnet_lab.py`: the model is built by that file's `build_cifar_resnet` and then
has its stem replaced, and the optimizer is that file's `build_resnet_sgd`
unchanged, so both campaigns share one recipe and one gate.

Three sections: why the stem is the right place, the model, and the optimizer.

--------------------------------------------------------------------------------
Why the stem, and why it is structurally sound where the last block was not
--------------------------------------------------------------------------------
The last-block arc found that a ResNet has no single parameter whose scaling
multiplies the logits, because stage four's three normalizations are three addends
of a sum rather than one composed scale. That is a structural obstruction, and it
is what `resnet_lab.py`'s `head_norm` exists to remove.

The first block has no such problem. A ResNet -- and every ResNet-like model --
begins with a plain convolution outside any residual branch: `conv1`, then `bn1`,
then a ReLU, and only then does `layer1` start adding. The CIFAR adaptation used
here replaces the 7x7 stride-2 convolution with a 3x3 stride-1 one and drops the
max-pool, so the stem sees 3x3 patches of the raw images at stride 1 -- exactly the
geometry of block 0 in the paper's plain conv-BatchNorm stack. The selection
predictor mu_eff, the margin of the input patch covariance's top eigenvalue over
its runner-up as a share of the trace, is therefore the same quantity in both:
0.633 for CIFAR-100 and 0.626 for ImageNet32.

So a pair on the stem is the same object the paper measures, dropped into a
standard architecture under a standard recipe. `SharedScaleRepVGGBlock` is used
unchanged rather than reimplemented, which also means the `pair_channel_open`
probe finds the stem by `isinstance` and measures the separation axis, its
alignment with the covariance's top eigendirection, and the per-channel opening
angles without any change to the probe.

One difference from the plain stack, recorded here because it is the first thing
to suspect if the result comes out strange. In the plain stack the next block's
normalization divides block 0's scale straight out, so the stem's gamma is
invisible downstream. In a ResNet, `layer1.0` has equal input and output channel
counts at stride 1, so its shortcut is a bare identity and carries the stem's
output into the first addition unrenormalized: the stem's gamma is partly visible
at the output. `tests/test_resnet_stem_pair.py` measures how much. The spectral
cooling mechanism is a statement about the gradient geometry along the separation
axis rather than about an output scale, so this should not matter, but it is a
real difference between the two architectures and it is not assumed away.

--------------------------------------------------------------------------------
The arms
--------------------------------------------------------------------------------
`stem_branches=1` is a one-branch `SharedScaleRepVGGBlock`, which has exactly the
stock stem's parameter count (one 3x3 kernel, one per-channel gamma, one
per-channel beta) and exactly its function form (normalize, scale, shift, ReLU).
It differs from the stock stem only in how the kernel is drawn: the block uses
Kaiming fan-in, torchvision's ResNet uses fan-out. A BatchNorm makes the block's
output invariant to that scale, but not the effective learning rate of the
direction, so the two are not identical runs and the campaign carries the stock
stem as a third arm to measure what the substitution alone costs.

The paired arm is `stem_branches=2` in `shared_gamma` mode: two independently
drawn 3x3 kernels, each normalized by its own batch statistics, summed, then one
shared per-channel gamma (initialized to 2^-1/2 so the output variance is about 1
at init) plus the sum of the two per-branch betas. That is the paper's `sharedg`
arm, the default cell's pair.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn as nn

from structural_reparam.experiments.reparam_shared_scale.lab import SharedScaleRepVGGBlock
from structural_reparam.agents.stage2.resnet_lab import build_cifar_resnet, build_resnet_sgd
from structural_reparam.agents.stage2 import probe as _probe  # noqa: F401  (registers pair_channel_open)


# --------------------------------------------------------------------------------
# 1. The model
# --------------------------------------------------------------------------------

def build_stem_pair_resnet(num_classes: int = 100, width_mult: float = 1.0,
                           blocks_per_stage: Sequence[int] = (2, 2, 2, 2),
                           stem_branches: int = 1,
                           stem_mode: str = "shared_gamma",
                           head_norm: bool = False,
                           **_: object) -> nn.Module:
    """A width-scaled CIFAR ResNet whose stem is an N-branch shared-scale block.

    The stem's convolution, normalization and activation are all owned by the
    block, so `bn1` and `relu` become identities and the forward pass
    `relu(bn1(conv1(x)))` reduces to `block(x)`. Everything downstream -- the four
    stages, the pooling, the classifier, the initialization of all of them -- is
    `build_cifar_resnet`'s, untouched, so a run built this way differs from that
    campaign's baseline in the stem and nothing else.

    `stem_branches=0` leaves the stock stem in place, so the stock arm can sit in
    the same configuration file as the paired arms. The trainer's `build_model`
    takes the model target from the model block and ignores a `target` key on a
    variant, so a stock arm expressed as a different target would silently be built
    from this one; expressing it as a branch count cannot go wrong that way.
    """
    model = build_cifar_resnet(num_classes=num_classes, width_mult=width_mult,
                               blocks_per_stage=blocks_per_stage, head_norm=head_norm)
    if int(stem_branches) == 0:
        print(f"[stem pair] stock stem, untouched; "
              f"parameters {sum(p.numel() for p in model.parameters())}", flush=True)
        return model
    conv1 = model.conv1
    block = SharedScaleRepVGGBlock(
        conv1.in_channels, conv1.out_channels, stride=int(conv1.stride[0]),
        num_branches=int(stem_branches), mode=str(stem_mode),
    )
    before = sum(p.numel() for p in model.parameters())
    model.conv1 = block
    model.bn1 = nn.Identity()
    model.relu = nn.Identity()
    after = sum(p.numel() for p in model.parameters())
    print(f"[stem pair] {stem_branches} branch(es), mode={stem_mode}, "
          f"{conv1.in_channels}->{conv1.out_channels} channels at stride "
          f"{int(conv1.stride[0])}; parameters {before} -> {after}", flush=True)
    return model


def stem_block(model: nn.Module) -> SharedScaleRepVGGBlock:
    """The stem block of a model built above, by type rather than by name.

    The same rule the `pair_channel_open` probe uses, so a test that passes here
    is a test of what the probe will actually measure.
    """
    found = [m for m in model.modules() if isinstance(m, SharedScaleRepVGGBlock)]
    if len(found) != 1:
        raise RuntimeError(f"expected exactly one shared-scale block, found {len(found)}")
    return found[0]


# --------------------------------------------------------------------------------
# 2. The optimizer
# --------------------------------------------------------------------------------

def build_stem_resnet_sgd(params: Iterable[nn.Parameter], model: nn.Module, lr: float,
                          **kwargs) -> torch.optim.SGD:
    """`resnet_lab.build_resnet_sgd`, named from this module.

    Delegating rather than reimplementing keeps the two ResNet campaigns on one
    recipe and one gate: kernel-only weight decay is decided by `ndim >= 2`, so the
    pair's shared gamma and its two per-branch betas are decay-free for the same
    reason a BatchNorm's affine parameters are, without the gate needing to know
    that the stem is a pair at all.

    Naming this module as the optimizer target also guarantees that importing it
    -- and with it the `pair_channel_open` registration above -- happens before the
    trainer reads the probe configuration.
    """
    return build_resnet_sgd(params, model=model, lr=lr, **kwargs)
