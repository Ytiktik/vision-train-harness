"""Experiment packages that belong to the preconditioning-hypothesis line.

Only the packages the current thesis direction still uses live here: the
``reparam_sweeps100`` backbone (``LayerwiseRepVGGCifar``), the
``reparam_split_rescue_cifar`` checkpoint-continuation trainer, the pinned-norm,
weight-norm-pair, channel-split, shared-scale, sigma-gate and sigma-kappa blocks,
and the ``reparam_ckpt_*`` checkpoint cells the replays start from. Closed arcs
were moved with ``git mv`` to ``archive/src/experiments/`` (2026-08-19); they
are not importable from there. See ``archive/README.md``.
"""
