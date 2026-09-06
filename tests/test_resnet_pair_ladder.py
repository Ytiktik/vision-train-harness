"""The paper's placement ladder on a shallow ResNet: what it replaces, and what it leaves alone.

The campaign asks whether the paper's first-block and last-block pairs do
anything on a ResNet built at the paper's own size, so the arms have to differ in
branch count and in nothing else. These tests keep that honest: the positions are
the ones documented and in forward order, the one-branch arm computes exactly the
stock network's function, the residual structure survives the substitution, the
activation lands where the ResNet puts it rather than where the block would, a
paired position still folds exactly, and the probe indexes the blocks the way the
module's position list does.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from structural_reparam.experiments.reparam_shared_scale.lab import SharedScaleRepVGGBlock
from structural_reparam.agents.stage2.resnet_lab import build_cifar_resnet
from structural_reparam.agents.stage2.resnet_pair_lab import (
    build_paired_resnet,
    pair_blocks,
    pairable_positions,
)

WIDTHS = (32, 64, 128)          # the cell: the paper's channel ladder, one block per stage
BLOCKS = (1, 1, 1)
N_POS = 7                       # stem plus two convolutions in each of three blocks

SINGLE = [1] * N_POS
PAIR_FIRST = [2] + [1] * 6
PAIR_LAST = [1] * 6 + [2]
PAIR_BOTH = [2] + [1] * 5 + [2]


def _build(branches):
    torch.manual_seed(0)
    return build_paired_resnet(num_classes=100, widths=WIDTHS, blocks_per_stage=BLOCKS,
                               block_branches=branches)


def _images(n: int = 8, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(n, 3, 32, 32)


def _n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# --------------------------------------------------------------------------------
# The positions
# --------------------------------------------------------------------------------

def test_positions_are_the_documented_seven_in_forward_order():
    model = _build(None)
    paths = [p[4] for p in pairable_positions(model)]
    assert paths == [
        "conv1",
        "layer1.0.conv1", "layer1.0.conv2",
        "layer2.0.conv1", "layer2.0.conv2",
        "layer3.0.conv1", "layer3.0.conv2",
    ]


def test_a_wrong_length_branch_list_raises_rather_than_training_the_wrong_arm():
    with pytest.raises(ValueError, match="pairable positions"):
        _build([2, 1, 1])


def test_probe_block_order_matches_the_position_order():
    """The probe enumerates by ``named_modules``; the module documents forward order.

    If these two disagree, ``cov_blocks: [0]`` whitens the wrong block and every
    per-block number in the campaign is mislabelled.
    """
    model = _build(SINGLE)
    installed = [getattr(owner, attr) for owner, attr, _, _, _ in pairable_positions(model)]
    assert pair_blocks(model) == installed


def test_the_stock_arm_has_no_shared_scale_block_so_the_probe_is_inert():
    assert pair_blocks(_build(None)) == []


# --------------------------------------------------------------------------------
# The one-branch arm is the stock network's function
# --------------------------------------------------------------------------------

def test_single_and_stock_have_the_same_parameter_count():
    assert _n_params(_build(SINGLE)) == _n_params(_build(None))


@pytest.mark.parametrize("branches,extra_positions", [
    (PAIR_FIRST, ["conv1"]),
    (PAIR_LAST, ["layer3.0.conv2"]),
    (PAIR_BOTH, ["conv1", "layer3.0.conv2"]),
])
def test_a_pair_costs_exactly_one_more_kernel_and_one_more_beta(branches, extra_positions):
    """A second branch adds its kernel and its own beta, and nothing else.

    The shared gamma is shared, so it is not duplicated; the per-branch
    normalization carries no affine parameters of its own.
    """
    stock = _build(None)
    by_path = {p[4]: getattr(p[0], p[1]) for p in pairable_positions(stock)}
    expected = sum(by_path[path].weight.numel() + by_path[path].out_channels
                   for path in extra_positions)
    assert _n_params(_build(branches)) - _n_params(_build(SINGLE)) == expected


def test_single_computes_the_stock_network_function_after_transplanting_weights():
    """Substituting the block changes the parameterization, not the function.

    Every weight of a stock network is copied into the one-branch network -- the
    kernel into the block's only branch, the BatchNorm's gamma into the shared
    gamma, its beta into the only branch's beta, its running statistics into the
    branch's -- and the two must then agree on the same input. It is also the test
    that removing the block's own activation left the network's nonlinearities
    where they were: a missing or a doubled ReLU shows up here as a mismatch.
    """
    stock, single = _build(None), _build(SINGLE)
    torch.manual_seed(7)
    with torch.no_grad():                       # a trained-looking stock network
        for m in stock.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.uniform_(0.5, 2.0)
                m.bias.uniform_(-0.5, 0.5)
                m.running_mean.uniform_(-0.3, 0.3)
                m.running_var.uniform_(0.5, 2.0)
            elif isinstance(m, nn.Conv2d):
                m.weight.normal_(0.0, 0.1)
        single.load_state_dict(
            {k: v.clone() for k, v in stock.state_dict().items()
             if k in single.state_dict() and single.state_dict()[k].shape == v.shape},
            strict=False,
        )
        for (owner, conv_attr, norm_attr, _, path) in pairable_positions(single):
            src_owner = stock
            for part in path.split(".")[:-1]:
                src_owner = src_owner[int(part)] if part.isdigit() else getattr(src_owner, part)
            src_conv = getattr(src_owner, conv_attr)
            src_norm = getattr(src_owner, norm_attr)
            blk = getattr(owner, conv_attr)
            blk.convs[0].weight.copy_(src_conv.weight)
            blk.gamma.copy_(src_norm.weight)
            blk.betas[0].copy_(src_norm.bias)
            blk.stats[0].running_mean.copy_(src_norm.running_mean)
            blk.stats[0].running_var.copy_(src_norm.running_var)

    stock.eval(), single.eval()
    x = _images(seed=3)
    with torch.no_grad():
        a, b = stock(x), single(x)
    assert torch.allclose(a, b, atol=1e-4), float((a - b).abs().max())


# --------------------------------------------------------------------------------
# What the substitution must not disturb
# --------------------------------------------------------------------------------

@pytest.mark.parametrize("branches", [SINGLE, PAIR_FIRST, PAIR_LAST, PAIR_BOTH])
def test_the_residual_structure_survives(branches):
    """Stage 1's block keeps a bare identity shortcut; stages 2 and 3 keep their
    1x1 downsample, which the substitution never touches."""
    model = _build(branches)
    assert model.layer1[0].downsample is None
    for stage in ("layer2", "layer3"):
        ds = getattr(model, stage)[0].downsample
        assert isinstance(ds[0], nn.Conv2d) and ds[0].kernel_size == (1, 1)
        assert isinstance(ds[1], nn.BatchNorm2d)


@pytest.mark.parametrize("branches", [SINGLE, PAIR_BOTH])
def test_the_block_carries_no_activation_and_every_relu_stays_the_resnets_own(branches):
    """The substitution replaces the convolution and the normalization, nothing else.

    The block's own ReLU is removed at every position, and the ResNet's ReLUs are
    left exactly where the ResNet puts them: after the stem, after each `bn1`, and
    after each residual addition. Doubling them instead would apply the same
    function -- ReLU is idempotent -- but both are in-place, so the second would
    overwrite the first's saved output and the backward pass would raise. The
    gradient test below is what catches that.
    """
    model = _build(branches)
    for (owner, conv_attr, _, _, path) in pairable_positions(model):
        assert isinstance(getattr(owner, conv_attr).activation, nn.Identity), path
    assert isinstance(model.relu, nn.ReLU)              # the stem's
    for stage in ("layer1", "layer2", "layer3"):
        assert isinstance(getattr(model, stage)[0].relu, nn.ReLU)


@pytest.mark.parametrize("branches", [SINGLE, PAIR_FIRST, PAIR_LAST, PAIR_BOTH])
def test_every_arm_produces_finite_logits_of_the_right_shape(branches):
    model = _build(branches)
    model.train()
    out = model(_images(16, seed=5))
    assert out.shape == (16, 100)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("branches", [SINGLE, PAIR_FIRST, PAIR_LAST, PAIR_BOTH])
def test_every_arm_backpropagates_and_reaches_every_parameter(branches):
    """The check the dry run of 2026-09-05 failed.

    A forward pass alone does not notice an in-place operation that has clobbered
    a tensor the backward pass needs; only calling backward does. Every parameter
    is also required to receive a finite gradient, so a position that has been
    disconnected by the substitution cannot pass silently.
    """
    model = _build(branches)
    model.train()
    model(_images(8, seed=9)).square().mean().backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, missing
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())


# --------------------------------------------------------------------------------
# The fold
# --------------------------------------------------------------------------------

@pytest.mark.parametrize("n_branches", [1, 2, 3])
def test_a_paired_position_still_folds_to_one_convolution_and_one_bias(n_branches):
    """The whole framework rests on a closed pair being a single branch, so the
    fold is checked at the position this campaign pairs rather than in general."""
    branches = [1] * 6 + [n_branches]
    model = _build(branches)
    model.train()
    model(_images(32, seed=11))                 # populate the running statistics
    model.eval()
    blk = getattr(model.layer3[0], "conv2")
    torch.manual_seed(13)
    with torch.no_grad():
        blk.gamma.uniform_(0.5, 2.0)
        for beta in blk.betas:
            beta.uniform_(-0.5, 0.5)
        z = torch.randn(4, blk.in_channels, 8, 8)
        direct = blk(z)
        w_eff, b_eff = blk.fused_conv_bias()
        folded = torch.nn.functional.conv2d(z, w_eff, b_eff, stride=blk.stride, padding=1)
    assert torch.allclose(direct, folded, atol=1e-5), float((direct - folded).abs().max())


# --------------------------------------------------------------------------------
# The ImageNet stem: the same rules at 224 pixels
# --------------------------------------------------------------------------------
#
# The full-resolution campaign of 2026-09-05 runs the same ladder on ResNet-10 and
# ResNet-18 with torchvision's 224-pixel input stage, so every rule above has to
# hold there too. What changes is the stem alone -- a 3x3 stride-2 convolution and
# a 3x3 stride-2 max-pool in place of the 3x3 stride-1 convolution and no pooling
# -- and the kernel stays 3x3 rather than becoming torchvision's 7x7, so that
# position 0 remains the paper's own block on raw pixels and its patch covariance
# remains the statistic the paper measures. These tests keep both halves of that:
# the stem is the shape claimed, and the substitution behaves exactly as it does
# at 32 pixels.

IN1K_WIDTHS = (64, 128, 256, 512)
R10_BLOCKS = (1, 1, 1, 1)       # ResNet-10: nine 3x3 convolutions
R18_BLOCKS = (2, 2, 2, 2)       # ResNet-18: seventeen
R10_POS, R18_POS = 9, 17


def _build_in1k(blocks, branches, num_classes: int = 1000):
    torch.manual_seed(0)
    return build_paired_resnet(num_classes=num_classes, widths=IN1K_WIDTHS,
                               blocks_per_stage=blocks, block_branches=branches,
                               stem="imagenet")


def _in1k_images(n: int = 2, size: int = 224, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(n, 3, size, size)


def _pair_both(n_positions: int) -> list[int]:
    return [2] + [1] * (n_positions - 2) + [2]


@pytest.mark.parametrize("blocks,n_positions", [(R10_BLOCKS, R10_POS), (R18_BLOCKS, R18_POS)])
def test_the_imagenet_networks_have_the_documented_positions_in_forward_order(blocks, n_positions):
    model = _build_in1k(blocks, None)
    paths = [p[4] for p in pairable_positions(model)]
    assert len(paths) == n_positions
    assert paths[0] == "conv1"
    assert paths[-1] == f"layer4.{blocks[-1] - 1}.conv2"
    expected = ["conv1"] + [f"layer{i}.{j}.conv{k}"
                            for i, n in enumerate(blocks, start=1)
                            for j in range(n) for k in (1, 2)]
    assert paths == expected


def test_the_imagenet_stem_is_a_3x3_stride_2_convolution_and_a_3x3_stride_2_maxpool():
    """The one departure from torchvision is the kernel size, and it is the point.

    A 7x7 stem would pair a different operator against a 49-pixel patch covariance,
    and the paper's block-0 measurements -- the separation axis on v_max, the
    selection predictor built from that spectrum -- would not carry over to it.
    """
    model = _build_in1k(R10_BLOCKS, None)
    assert isinstance(model.conv1, nn.Conv2d)
    assert model.conv1.kernel_size == (3, 3) and model.conv1.stride == (2, 2)
    assert isinstance(model.maxpool, nn.MaxPool2d)
    assert model.maxpool.kernel_size == 3 and model.maxpool.stride == 2


def test_the_cifar_stem_is_untouched_by_the_imagenet_option():
    """A regression guard: every result above this line was measured with the
    stride-1 stem and no pooling, and adding the option must not have moved it."""
    model = _build(None)
    assert model.conv1.kernel_size == (3, 3) and model.conv1.stride == (1, 1)
    assert isinstance(model.maxpool, nn.Identity)


def test_the_imagenet_stem_downsamples_the_way_torchvision_does():
    """224 -> 112 at the stem, 56 into stage 1, 7 out of stage 4.

    Without the max-pool every activation map would be four times the area, which
    would change the cell for a reason unrelated to the mechanism.
    """
    model = _build_in1k(R10_BLOCKS, [1] * R10_POS)
    model.eval()
    sizes = {}
    x = _in1k_images(2)
    with torch.no_grad():
        h = model.relu(model.bn1(model.conv1(x)))
        sizes["stem"] = h.shape[-1]
        h = model.maxpool(h)
        sizes["into_stage1"] = h.shape[-1]
        for i in range(1, 5):
            h = getattr(model, f"layer{i}")(h)
        sizes["out_of_stage4"] = h.shape[-1]
    assert sizes == {"stem": 112, "into_stage1": 56, "out_of_stage4": 7}


@pytest.mark.parametrize("blocks,n_positions", [(R10_BLOCKS, R10_POS), (R18_BLOCKS, R18_POS)])
def test_imagenet_single_and_stock_have_the_same_parameter_count(blocks, n_positions):
    assert _n_params(_build_in1k(blocks, [1] * n_positions)) == _n_params(_build_in1k(blocks, None))


@pytest.mark.parametrize("blocks,n_positions", [(R10_BLOCKS, R10_POS), (R18_BLOCKS, R18_POS)])
def test_an_imagenet_pair_costs_exactly_one_more_kernel_and_one_more_beta(blocks, n_positions):
    stock = _build_in1k(blocks, None)
    positions = pairable_positions(stock)
    extra = [positions[0], positions[-1]]        # the two `pair_both` pairs
    expected = sum(getattr(owner, attr).weight.numel() + getattr(owner, attr).out_channels
                   for owner, attr, _, _, _ in extra)
    grew = (_n_params(_build_in1k(blocks, _pair_both(n_positions)))
            - _n_params(_build_in1k(blocks, [1] * n_positions)))
    assert grew == expected


def test_imagenet_single_computes_the_stock_networks_function_after_transplanting_weights():
    """As at 32 pixels: substituting the block changes the parameterization only.

    This is also the test that the stem's stride and the max-pool sit where the
    substitution left them -- a pooling layer dropped or applied twice shows up
    here as a mismatch.
    """
    stock, single = _build_in1k(R10_BLOCKS, None), _build_in1k(R10_BLOCKS, [1] * R10_POS)
    torch.manual_seed(7)
    with torch.no_grad():
        for m in stock.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.uniform_(0.5, 2.0)
                m.bias.uniform_(-0.5, 0.5)
                m.running_mean.uniform_(-0.3, 0.3)
                m.running_var.uniform_(0.5, 2.0)
            elif isinstance(m, nn.Conv2d):
                m.weight.normal_(0.0, 0.1)
        single.load_state_dict(
            {k: v.clone() for k, v in stock.state_dict().items()
             if k in single.state_dict() and single.state_dict()[k].shape == v.shape},
            strict=False,
        )
        for (owner, conv_attr, norm_attr, _, path) in pairable_positions(single):
            src_owner = stock
            for part in path.split(".")[:-1]:
                src_owner = src_owner[int(part)] if part.isdigit() else getattr(src_owner, part)
            src_conv = getattr(src_owner, conv_attr)
            src_norm = getattr(src_owner, norm_attr)
            blk = getattr(owner, conv_attr)
            blk.convs[0].weight.copy_(src_conv.weight)
            blk.gamma.copy_(src_norm.weight)
            blk.betas[0].copy_(src_norm.bias)
            blk.stats[0].running_mean.copy_(src_norm.running_mean)
            blk.stats[0].running_var.copy_(src_norm.running_var)

    stock.eval(), single.eval()
    x = _in1k_images(2, seed=3)
    with torch.no_grad():
        a, b = stock(x), single(x)
    # Compared relative to the logit scale rather than at a fixed tolerance. The
    # two networks compute the same function but not in the same order -- the block
    # normalizes and scales where the stock path has a BatchNorm -- and with random
    # weights on 224-pixel inputs the logits reach order 10^4, so float32
    # accumulation alone moves the last few digits. It comes out at about 2e-7 of
    # the logit scale, which is that and nothing else; a real mismatch, such as a
    # dropped or doubled max-pool, moves it by a factor of the input.
    scale = float(b.abs().max())
    err = float((a - b).abs().max())
    assert err <= 1e-5 * scale, (err, scale)


@pytest.mark.parametrize("blocks,n_positions", [(R10_BLOCKS, R10_POS), (R18_BLOCKS, R18_POS)])
def test_the_imagenet_residual_structure_survives(blocks, n_positions):
    """Stage 1 keeps its bare identity shortcut and stages 2 to 4 their 1x1
    downsample, which the substitution never touches."""
    model = _build_in1k(blocks, _pair_both(n_positions))
    assert model.layer1[0].downsample is None
    for stage in ("layer2", "layer3", "layer4"):
        ds = getattr(model, stage)[0].downsample
        assert isinstance(ds[0], nn.Conv2d) and ds[0].kernel_size == (1, 1)
        assert isinstance(ds[1], nn.BatchNorm2d)


@pytest.mark.parametrize("blocks,n_positions", [(R10_BLOCKS, R10_POS), (R18_BLOCKS, R18_POS)])
def test_every_imagenet_arm_backpropagates_and_reaches_every_parameter(blocks, n_positions):
    """The check the dry run of 2026-09-05 failed at 32 pixels, at 224.

    Both arms of the campaign are covered: a forward pass alone does not notice an
    in-place operation that has clobbered a tensor the backward pass needs, and a
    position disconnected by the substitution cannot pass silently.
    """
    for branches in ([1] * n_positions, _pair_both(n_positions)):
        model = _build_in1k(blocks, branches)
        model.train()
        out = model(_in1k_images(2, seed=9))
        assert out.shape == (2, 1000)
        assert torch.isfinite(out).all()
        out.square().mean().backward()
        missing = [n for n, p in model.named_parameters() if p.grad is None]
        assert not missing, missing
        assert all(torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("blocks,n_positions", [(R10_BLOCKS, R10_POS), (R18_BLOCKS, R18_POS)])
def test_the_imagenet_paired_positions_still_fold(blocks, n_positions):
    """Both of `pair_both`'s positions, since the whole framework rests on a closed
    pair being a single branch: the stem, at stride 2 on raw pixels, and the last
    3x3 convolution."""
    model = _build_in1k(blocks, _pair_both(n_positions))
    model.train()
    model(_in1k_images(4, size=64, seed=11))     # populate the running statistics
    model.eval()
    positions = pairable_positions(model)
    for owner, conv_attr, _, _, path in (positions[0], positions[-1]):
        blk = getattr(owner, conv_attr)
        torch.manual_seed(13)
        with torch.no_grad():
            blk.gamma.uniform_(0.5, 2.0)
            for beta in blk.betas:
                beta.uniform_(-0.5, 0.5)
            z = torch.randn(2, blk.in_channels, 16, 16)
            direct = blk(z)
            w_eff, b_eff = blk.fused_conv_bias()
            folded = torch.nn.functional.conv2d(z, w_eff, b_eff, stride=blk.stride, padding=1)
        assert torch.allclose(direct, folded, atol=1e-5), (path, float((direct - folded).abs().max()))
