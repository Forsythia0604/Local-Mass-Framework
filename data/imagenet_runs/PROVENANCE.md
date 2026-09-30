# ImageNet phase-1 run records

These 12 JSON summaries, 12 stdout logs and 10 PBS records were copied unchanged
from `Experiments4and5/data/imagenet_runs/`.

The original provenance document is preserved in
[`docs/originals/imagenet_runs_PROVENANCE.md`](../../docs/originals/imagenet_runs_PROVENANCE.md).
Its old paths and job instructions describe the source environment, not this
consolidated folder. Use the current neural-network guide for local commands.

The supplied records describe ResNet-50 on ImageNet, phase 1, four variational
families, seeds 1–3, a frozen pretrained base, 8 epochs, prior slab weight 0.2,
KL weight 1e-4 and 7,552 gate channels. Phase 2 was not part of these results.

Current locations:

- Saved marginals: `data/marginals/imagenet_marginals_s<seed>.npz`.
- Optional gate weights: `data/imagenet_checkpoints/` (excluded from this copy).
- Plotting: `experiments/neural_networks/experiment_scripts/imagenet_multiseed.py`.
- Source job script: `experiments/neural_networks/pbs/run_e1_imagenet_phase1.pbs`.

Historical run statistics and values derived from pooled saved marginals are
retained in their respective source files; this merge does not reinterpret them.
