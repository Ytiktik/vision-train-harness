"""RepVGG-style foldable convolution blocks."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from structural_reparam.base_models.custom_batchnorm import (
    SqrtkBias,
    LearnableScaleBias,
    NoScaleBias,
    RunningVarNorm,
    WeightNorm,
    ZeroMeanBatchNorm,
    fold_bn,
)
from structural_reparam.base_models.repvgg_common import RepVGGBlockBase, max_fusion_error
from structural_reparam.base_models.init_kernels import (
    fixed_3x3,
    init_3x3,
    is_learnable,
)


class FixedFilterBranch(nn.Module):
    """Frozen depthwise 3x3 filter followed by a learned 1x1 + BN.

    Foldable: at deploy time the depthwise + pointwise composition becomes a
    single 3x3 kernel of shape (out_c, in_c, 3, 3) by ``K[o,i] = P[o,i] * D[i]``.
    """

    def __init__(self, name: str, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        depthwise_kernel = fixed_3x3(name).repeat(in_channels, 1, 1, 1)
        self.register_buffer("depthwise_kernel", depthwise_kernel, persistent=True)

        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

        nn.init.kaiming_normal_(self.pointwise.weight, mode="fan_in", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.conv2d(
            x,
            self.depthwise_kernel,
            bias=None,
            stride=self.stride,
            padding=1,
            groups=self.in_channels,
        )
        y = self.pointwise(y)
        return self.bn(y)

    def equivalent_kernel_bias(self) -> tuple[torch.Tensor, torch.Tensor]:
        # composed (out_c, in_c, 3, 3) before BN folding
        pw = self.pointwise.weight.view(self.out_channels, self.in_channels, 1, 1)
        dw = self.depthwise_kernel.view(self.in_channels, 1, 3, 3).squeeze(1)  # (in_c, 3, 3)
        kernel = pw * dw.unsqueeze(0)
        return fold_bn(self.bn, kernel, torch.zeros(self.out_channels, device=kernel.device))


class WeightNormConvBranch(nn.Module):
    """Conv (no bias) -> WeightNorm: a foldable weight-space normalized branch.

    The normalizer's divisor is the per-filter kernel L2 norm ``||w||`` (a
    function of the weights only), so unlike BatchNorm it never depends on the
    activations: at inference the scale ``gamma/||w||`` is a fixed tensor and the
    branch folds into the equivalent conv at zero extra cost. Algebraically the
    pre-activation is ``gamma (w·x)/sqrt(w^T I w)`` -- the BatchNorm expression
    ``gamma (w·x)/sqrt(w^T C_x w)`` with the activation covariance ``C_x``
    replaced by the identity, which is what breaks the ``w1<->w2`` branch
    symmetry when paired against a BatchNorm branch.

    Exposes ``[0]`` -> conv and ``[1]`` -> norm so it drops straight into the
    RepVGGBlock branch plumbing (init, gamma-set, freeze, fold) alongside the
    plain ``nn.Sequential(Conv2d, BatchNorm2d)`` branches.
    """

    def __init__(self, conv: nn.Conv2d, norm: WeightNorm) -> None:
        super().__init__()
        self.conv = conv
        self.norm = norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.conv(x), self.conv.weight)

    def __getitem__(self, idx: int) -> nn.Module:
        return (self.conv, self.norm)[idx]


class RepVGGBlock(RepVGGBlockBase):
    """Foldable RepVGG-style block with configurable training-time branches."""

    _fuse_delete_attrs = ("conv3_branches", "conv1", "identity_bn", "fixed_branches", "global_bn")
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        use_1x1: bool = True,
        use_identity: bool = True,
        num_3x3: int = 1,
        kernel_size: int = 3,
        init_strategies: Sequence[str] | None = None,
        conv3_branch_norms: Sequence[str] | None = None,
        fixed_filters: Sequence[str] | None = None,
        track_distance: bool = False,
        bn_use_bias: bool = False,
        bn_sum_norm: str = "biasonly",
        freeze_branches: list[int] | None = None,
        frozen_branch_scale: float = 1.0,
        freeze_identity: bool = False,
        frozen_identity_scale: float = 1.0,
        identity_use_bn: bool = True,
        freeze_bn_scale: bool = False,
        branch_init: str = "inv_sqrt_n",
        kaiming_branch_sum_k: int = 1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.use_1x1 = use_1x1
        self.kernel_size = kernel_size
        self.deployed = False
        self.bn_use_bias = bn_use_bias
        if bn_sum_norm not in ("global_bn", "sqrtk_bias", "running_var", "zero_mean_bn", "affine", "biasonly"):
            raise ValueError(
                f"bn_sum_norm must be 'global_bn', 'sqrtk_bias', 'running_var', "
                f"'zero_mean_bn', 'affine', or 'biasonly', got {bn_sum_norm!r}"
            )
        if branch_init not in ("inv_sqrt_n", "overlap_aware", "ones"):
            raise ValueError(
                f"branch_init must be 'inv_sqrt_n', 'overlap_aware', or 'ones', got {branch_init!r}"
            )
        self.branch_init = branch_init
        if num_3x3 < 1:
            raise ValueError(f"num_3x3 must be >= 1, got {num_3x3}")
        if init_strategies is None:
            init_strategies = ["kaiming"] * num_3x3
        if len(init_strategies) != num_3x3:
            raise ValueError(
                f"init_strategies length {len(init_strategies)} does not match num_3x3 {num_3x3}"
            )
        for s in init_strategies:
            if not is_learnable(s):
                raise ValueError(f"Unknown learnable 3x3 init strategy: {s!r}")
        self.init_strategies = list(init_strategies)
        self.num_3x3 = num_3x3
        if conv3_branch_norms is None:
            conv3_branch_norms = ["batch"] * num_3x3
        if len(conv3_branch_norms) != num_3x3:
            raise ValueError(
                f"conv3_branch_norms length {len(conv3_branch_norms)} does not match "
                f"num_3x3 {num_3x3}"
            )
        for norm in conv3_branch_norms:
            if norm not in ("batch", "weight"):
                raise ValueError(
                    f"conv3_branch_norms entries must be 'batch' or 'weight', got {norm!r}"
                )
        self.conv3_branch_norms = list(conv3_branch_norms)
        if kaiming_branch_sum_k < 1:
            raise ValueError(
                f"kaiming_branch_sum_k must be >= 1, got {kaiming_branch_sum_k}"
            )
        self.kaiming_branch_sum_k = kaiming_branch_sum_k

        padding = kernel_size // 2

        def _make_conv3_branch(norm: str) -> nn.Module:
            conv = nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            )
            # WeightNorm folds via its weight-only divisor ||w||; BatchNorm folds
            # via its frozen running stats. Both leave a fixed affine at deploy.
            if norm == "weight":
                return WeightNormConvBranch(conv, WeightNorm(out_channels))
            return nn.Sequential(conv, nn.BatchNorm2d(out_channels))

        self.conv3_branches = nn.ModuleList(
            [_make_conv3_branch(norm) for norm in self.conv3_branch_norms]
        )
        # A k-branch block reparameterizes into a single conv whose kernel is the
        # SUM of the k branch kernels (variance k x a single branch). To mimic that
        # fused-kernel init with one branch, draw the kernel as the sum of k i.i.d.
        # init_3x3 draws. k=1 leaves the vanilla single-draw init untouched.
        for branch, strategy in zip(self.conv3_branches, self.init_strategies):
            kernel = init_3x3(strategy, out_channels, in_channels)
            for _ in range(kaiming_branch_sum_k - 1):
                kernel = kernel + init_3x3(strategy, out_channels, in_channels)
            with torch.no_grad():
                branch[0].weight.copy_(kernel)

        # The 1x1 branch is tied to the identity slot and never duplicates it:
        #   - where identity is possible (stride 1, in==out), `use_1x1` adds a
        #     1x1 in parallel with the identity branch;
        #   - where identity is impossible (downsample / channel change),
        #     `use_identity` falls back to a 1x1 standing in for the missing
        #     identity, and `use_1x1` adds nothing extra (so we never build two
        #     parallel 1x1 convs).
        identity_possible = stride == 1 and in_channels == out_channels
        self.has_identity = use_identity and identity_possible
        has_1x1 = (use_1x1 and identity_possible) or (use_identity and not identity_possible)
        self.conv1 = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            if has_1x1
            else None
        )
        if self.conv1 is not None:
            nn.init.kaiming_normal_(self.conv1[0].weight, mode="fan_in", nonlinearity="relu")

        self.identity_use_bn = identity_use_bn
        self.identity_bn = nn.BatchNorm2d(out_channels) if self.has_identity and identity_use_bn else None

        fixed_filters = list(fixed_filters or [])
        self.fixed_filter_names = fixed_filters
        self.fixed_branches = nn.ModuleDict(
            {
                name: FixedFilterBranch(name, in_channels, out_channels, stride)
                for name in fixed_filters
            }
        )

        self.track_distance = track_distance and num_3x3 >= 2
        self._cos_sim_sq: torch.Tensor | None = None

        if branch_init == "overlap_aware":
            self._init_overlap_aware_gammas()
        elif branch_init == "ones":
            self._init_bn_weights_ones()
        else:
            self._init_bn_weights_inv_sqrt_n()

        if freeze_bn_scale:
            for branch in self.conv3_branches:
                bn = branch[1]
                with torch.no_grad():
                    bn.weight.fill_(1.0)
                bn.weight.requires_grad_(False)

        if freeze_branches:
            for idx in freeze_branches:
                if idx < 0 or idx >= num_3x3:
                    raise ValueError(
                        f"freeze_branches index {idx} out of range for num_3x3={num_3x3}"
                    )
                bn = self.conv3_branches[idx][1]
                with torch.no_grad():
                    bn.weight.mul_(frozen_branch_scale)
                for p in self.conv3_branches[idx].parameters():
                    p.requires_grad_(False)

        if freeze_identity and self.identity_bn is not None:
            with torch.no_grad():
                self.identity_bn.weight.mul_(frozen_identity_scale)
            for p in self.identity_bn.parameters():
                p.requires_grad_(False)

        if bn_use_bias:
            self.global_bn: nn.Module = nn.Identity()
        else:
            self._disable_bn_biases()
            if bn_sum_norm == "global_bn":
                self.global_bn = nn.BatchNorm2d(out_channels)
            elif bn_sum_norm == "sqrtk_bias":
                self.global_bn = SqrtkBias(out_channels, self._count_branches() ** -0.5)
            elif bn_sum_norm == "running_var":
                self.global_bn = RunningVarNorm(out_channels)
            elif bn_sum_norm == "zero_mean_bn":
                self.global_bn = ZeroMeanBatchNorm(out_channels)
            elif bn_sum_norm == "biasonly":
                self.global_bn = NoScaleBias(out_channels)
            else:
                self.global_bn = LearnableScaleBias(out_channels, 1.0)

        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deployed:
            return self.act(self.fused_conv(x))

        out = self.conv3_branches[0](x)
        first_branch_out = out if self.track_distance else None
        second_branch_out: torch.Tensor | None = None
        for idx, branch in enumerate(self.conv3_branches[1:], start=1):
            extra = branch(x)
            if idx == 1 and self.track_distance:
                second_branch_out = extra
            out = out + extra
        if self.conv1 is not None:
            out = out + self.conv1(x)
        if self.has_identity:
            out = out + (self.identity_bn(x) if self.identity_bn is not None else x)
        for branch in self.fixed_branches.values():
            out = out + branch(x)

        if self.track_distance and first_branch_out is not None and second_branch_out is not None:
            a = first_branch_out.flatten(1)
            b = second_branch_out.flatten(1)
            cos = F.cosine_similarity(a, b, dim=1, eps=1e-8)
            self._cos_sim_sq = (cos ** 2).mean()
        else:
            self._cos_sim_sq = None

        return self.act(self.global_bn(out))

    def equivalent_kernel(self) -> torch.Tensor:
        if self.deployed:
            return self.fused_conv.weight

        kernel, _ = self.equivalent_kernel_bias()
        return kernel

    def equivalent_kernel_bias(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.deployed:
            return self.fused_conv.weight, self.fused_conv.bias

        kernel, bias = self._fuse_conv_bn(self.conv3_branches[0])
        for branch in self.conv3_branches[1:]:
            k, b = self._fuse_conv_bn(branch)
            kernel = kernel + k
            bias = bias + b
        if self.conv1 is not None:
            k1, b1 = self._fuse_conv_bn(self.conv1)
            p = self.kernel_size // 2
            kernel = kernel + F.pad(k1, [p, p, p, p])
            bias = bias + b1
        if self.has_identity:
            if self.identity_bn is not None:
                ki, bi = self._fuse_identity_bn()
            else:
                ki = self._identity_kernel()
                bi = torch.zeros(self.out_channels, device=ki.device)
            kernel = kernel + ki
            bias = bias + bi
        for branch in self.fixed_branches.values():
            kf, bf = branch.equivalent_kernel_bias()
            kernel = kernel + kf
            bias = bias + bf

        gbn = self.global_bn
        if isinstance(gbn, nn.BatchNorm2d):
            kernel, bias = fold_bn(gbn, kernel, bias)
        elif not isinstance(gbn, nn.Identity):
            kernel, bias = gbn.fold(kernel, bias)

        return kernel, bias

    def _fuse_conv_bn(self, branch: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
        conv = branch[0]
        norm = branch[1]
        zeros = torch.zeros(conv.weight.shape[0], device=conv.weight.device)
        # WeightNorm's divisor is the weight norm, so it folds via its own .fold
        # (scale = gamma/||w||); BatchNorm folds via its frozen running stats.
        if isinstance(norm, WeightNorm):
            return norm.fold(conv.weight, zeros)
        return fold_bn(norm, conv.weight, zeros)

    def _fuse_identity_bn(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.identity_bn is None:
            raise RuntimeError("Cannot fuse identity branch because it is disabled.")
        return fold_bn(self.identity_bn, self._identity_kernel(), torch.zeros(self.out_channels, device=self.identity_bn.weight.device))

    def _identity_kernel(self) -> torch.Tensor:
        ref = self.conv3_branches[0][0].weight
        eye = torch.eye(self.in_channels, device=ref.device, dtype=ref.dtype)
        p = self.kernel_size // 2
        return F.pad(eye.reshape(self.in_channels, self.in_channels, 1, 1), [p, p, p, p])

    def _count_branches(self) -> int:
        n = self.num_3x3
        if self.conv1 is not None:
            n += 1
        n += len(self.fixed_branches)
        return n

    def _disable_bn_biases(self) -> None:
        """Zeros out and freezes the beta (bias) term in all branch batchnorms."""
        modules_with_bn = [
            *[branch[1] for branch in self.conv3_branches],
        ]
        if self.conv1 is not None:
            modules_with_bn.append(self.conv1[1])
        if self.identity_bn is not None:
            modules_with_bn.append(self.identity_bn)
        modules_with_bn.extend([branch.bn for branch in self.fixed_branches.values()])

        for bn in modules_with_bn:
            nn.init.zeros_(bn.bias)
            bn.bias.requires_grad_(False)

    def _init_bn_weights_ones(self) -> None:
        """Leave γ of every branch BN at 1 — the vanilla RepVGG init (Ding et al.
        2021). No branch rescaling, so the block reduces to the paper's exact
        train-time structure (conv->BN(γ=1)->sum->ReLU)."""
        bns = [*[branch[1] for branch in self.conv3_branches]]
        if self.conv1 is not None:
            bns.append(self.conv1[1])
        if self.identity_bn is not None:
            bns.append(self.identity_bn)
        bns.extend([branch.bn for branch in self.fixed_branches.values()])
        for bn in bns:
            nn.init.constant_(bn.weight, 1.0)

    def _init_bn_weights_inv_sqrt_n(self) -> None:
        """Initialise γ of every branch BN to 1/sqrt(N), where N is the branch count."""
        inv_sqrt_n = self._count_branches() ** -0.5
        bns = [*[branch[1] for branch in self.conv3_branches]]
        if self.conv1 is not None:
            bns.append(self.conv1[1])
        if self.identity_bn is not None:
            bns.append(self.identity_bn)
        bns.extend([branch.bn for branch in self.fixed_branches.values()])
        for bn in bns:
            nn.init.constant_(bn.weight, inv_sqrt_n)

    def _init_overlap_aware_gammas(self) -> None:
        """Overlap-aware gamma init for the standard RepVGG branch set.

        Instead of the uniform 1/sqrt(N) gamma, scale each branch by 1/sqrt(k)
        where k is how many branches' spatial supports cover that branch's region:
        the 3x3 outer ring sees only the 3x3 branches (k=num_3x3), the 1x1 overlaps
        the 3x3 centres (k=num_3x3+1), and the identity overlaps both
        (k=num_3x3 + has_1x1 + 1). For the canonical 3-branch block this gives
        3x3 -> 1, 1x1 -> 1/sqrt(2), identity -> 1/sqrt(3). The scale lives in gamma
        (not the conv kernel) because each branch's BatchNorm normalises any
        conv-weight scale away; conv kernels keep their standard init.
        """
        for branch in self.conv3_branches:
            nn.init.constant_(branch[1].weight, self.num_3x3 ** -0.5)
        k = self.num_3x3
        if self.conv1 is not None:
            k += 1
            nn.init.constant_(self.conv1[1].weight, k ** -0.5)
        if self.identity_bn is not None:
            nn.init.constant_(self.identity_bn.weight, (k + 1) ** -0.5)
