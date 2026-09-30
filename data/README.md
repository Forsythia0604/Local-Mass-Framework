# Data in this lightweight copy

- `marginals/`: the three original ImageNet `.npz` inputs, copied byte-for-byte.
- `imagenet_runs/`: original JSON summaries, logs and PBS records.
- `imagenet_checkpoints/README.md`: instructions for optional excluded weights.
- `raw/`: suggested local download location, ignored by Git.
- `generated/`: suggested location for newly trained or re-extracted marginals,
  ignored by Git. Move selected validated runs into a documented tracked location
  if they are later chosen as reference results.

New MNIST/CIFAR training or ImageNet extraction also saves a companion JSON
with the command, versions and `.npz` checksum. See
[the reproduction guide](../docs/REPRODUCIBILITY.md) for data preparation.

The original MNIST and CIFAR-10 marginal files are absent. The CIFAR base model
weight is under `experiments/neural_networks/experiment_scripts/flopp/` because
the original training code expects it there.
