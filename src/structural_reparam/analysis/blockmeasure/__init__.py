"""blockmeasure: one measurement module for the two-branch conv-BatchNorm block.

The package has three modules:

- ``measure``: every definition of the theory as a pure function of primitives
  (kernels, gamma, the block-seen covariance Sigma, and a kernel-space vector such
  as the gradient or the optimizer's momentum buffer), the identity checks, and the
  aggregation conventions. Numpy, float64, no training code.
- ``recorder``: the registry probe that records those primitives from inside the
  project's trainer at every selected optimizer step, and the file format.
- ``derive``: the command line that turns a recording (or checkpoints) into derived
  arrays, a checks report and summary tables.

Nothing in this package edits the trainer, the model, or the optimizer.
"""

BLOCKMEASURE_VERSION = "0.1"
