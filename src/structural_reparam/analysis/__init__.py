"""Analysis helpers for figures, tables, and run comparisons."""

from structural_reparam.analysis.gradient_stats import GradientProbe
from structural_reparam.analysis.class_conflict import ClassConflictProbe
from structural_reparam.analysis.branch_probe import BranchProbe
from structural_reparam.analysis.branch_symmetry_probe import BranchSymmetryProbe
from structural_reparam.analysis.checkpoint_probe import CheckpointProbe
from structural_reparam.analysis.cue_attribution_probe import CueAttributionProbe
from structural_reparam.analysis.kernel_ratio_probe import KernelRatioProbe
from structural_reparam.analysis.mechanistic_probe import MechanisticProbe
from structural_reparam.analysis.repvgg_gamma_probe import RepVGGGammaProbe
from structural_reparam.analysis.repvgg_mech_probe import RepVGGMechProbe
from structural_reparam.analysis.sigma_ratio_probe import SigmaRatioProbe
from structural_reparam.analysis.registry import (
    PROBE_REGISTRY,
    ProbeContext,
    build_probes,
    register_probe,
)

__all__ = [
    "GradientProbe",
    "ClassConflictProbe",
    "BranchProbe",
    "CheckpointProbe",
    "CueAttributionProbe",
    "KernelRatioProbe",
    "MechanisticProbe",
    "RepVGGGammaProbe",
    "RepVGGMechProbe",
    "SigmaRatioProbe",
    "PROBE_REGISTRY",
    "ProbeContext",
    "build_probes",
    "register_probe",
]
