# Runtime inputs

- `marginals/`: three saved ImageNet marginal input files, one for each seed.
  The multi-seed analysis script reads these directly.
- `imagenet_checkpoints/README.md`: instructions for optional external checkpoints.
- `raw/`: default location for downloaded MNIST/CIFAR datasets; created when needed
  and ignored by Git.
- `generated/`: suggested location for new training or extraction outputs;
  created when needed and ignored by Git.

The CIFAR ResNet-56 base weight is retained at
`experiments/neural_networks/experiment_scripts/flopp/cifar10_resnet56.pth`
relative to the repository root because the training code expects it there.

Original MNIST and CIFAR-10 marginal inputs are absent. ImageNet training data
and large ImageNet gate checkpoints are not included. Historical training logs,
scheduler records and numerical result summaries have been removed.

See the [neural-network usage guide](../experiments/neural_networks/README.md)
for data preparation, training, and extraction commands.
