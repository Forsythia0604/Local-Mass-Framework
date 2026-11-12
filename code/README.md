# Local-mass experiments

Experiment code and inputs for [*Beyond Global Divergences: A Local-Mass
Perspective on Bayesian Inference*](../main.pdf) by Hanli Xu, Fengxiang He,
and Sarat Moka. The manuscript figures and result exports are in
[\`../results/\`](../results/).

## Contents

```text
experiments/local_mass_experiments.py  Synthetic and UCI experiments (1-3)
experiments/neural_networks/          Neural-network experiments (4-5)
data/marginals/                      Three ImageNet marginal inputs
data/imagenet_checkpoints/README.md   Optional checkpoint instructions
requirements*.txt                    Runtime dependency lists
THIRD_PARTY.md                       Source and third-party notes
```

The neural-network directory includes Python modules, two generic PBS submission
templates, and the CIFAR ResNet-56 base weight expected by the training code.

## Quick start

Run commands from this directory. The supplied pinned dependency files record a
previous Python 3.11 CPU environment; dependencies have not been freshly installed
or run as part of this packaging step.

```sh
python -m venv .venv
# macOS / Linux:
source .venv/bin/activate
# Windows PowerShell instead: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-verified.txt
```

Run experiments 1-3:

```sh
python experiments/local_mass_experiments.py
```

This creates figures and numerical exports in `results/generated/exp01_03/`.
The default uses 2^15 Sobol points per run. The script accepts `--qmc-power`
and `--output-dir` overrides.

Run ImageNet analysis using the included marginal inputs:

```sh
python experiments/neural_networks/experiment_scripts/imagenet_multiseed.py --seeds 1 2 3 --marg_dir data/marginals
```

This creates figures, a LaTeX table and a per-seed CSV under
`results/generated/exp04_05/figures/`. No ImageNet dataset, GPU or ImageNet
checkpoint is needed for this saved-input analysis.

For neural-network training and extraction, see the
[neural-network usage guide](experiments/neural_networks/README.md).
See [data notes](data/README.md) for included and missing inputs, and
[source notes](THIRD_PARTY.md) for third-party material.

## Reproduction limits

The original MNIST/CIFAR marginal inputs and large ImageNet checkpoints are not
included. New MNIST/CIFAR runs require training; ImageNet training requires a
separately prepared dataset. Full training and GPU execution were not verified
by this packaging step.

The [third-party notes](THIRD_PARTY.md) describe the provenance of the
CIFAR-10 base weight.
