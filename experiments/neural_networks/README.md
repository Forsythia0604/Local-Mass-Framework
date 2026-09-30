# Experiments 4 and 5

The original `Experiments4and5/code/` tree is preserved here, including helper
modules and older drivers. `experiment_scripts/` remains one level below the
module root so its existing imports continue to resolve.

All commands below run from the repository root. Reference figures and tables
are in `results/figures/` and `results/tables/`. New plots default to
`results/generated/exp04_05/figures/`; plotting commands accept `--output-dir`.

## Environment

For saved-marginal plotting, install the root `requirements-verified.txt`. Training or
checkpoint extraction also requires Torch and Torchvision:

```sh
python -m pip install -r requirements-training-verified.txt
```

The source package reports Python 3.12, Torch 2.11 and Torchvision 0.26 for its
local runs. Those are historical notes. The current CPU checks used Python 3.11.16,
Torch 2.13.0 and Torchvision 0.28.0, recorded in the verified requirement files.
Select a matched Torch/Torchvision build for the intended device. The supplied
device selector uses MPS when available for `auto`; pass `--device cuda` explicitly
for an NVIDIA GPU.

## Saved ImageNet marginals

```sh
python experiments/neural_networks/experiment_scripts/imagenet_multiseed.py --seeds 1 2 3 --marg_dir data/marginals
```

This writes the small-ball and directional RE-KL plots, a four-family LaTeX
table and a per-seed CSV, and prints the multi-seed summary. Included inputs contain all four
families, with 7,552 channel entries per family and seed.

## Experiment 4: MNIST

The original marginal input was not supplied. The following creates a new run.
The script creates the output directory and saves a companion JSON with its
arguments, effective KL weights, software versions and the marginal-file checksum.

```sh
python experiments/neural_networks/experiment_scripts/train_marginals.py --arch lenet --dataset mnist --pi 0.5 --epochs 15 --anneal_epochs 5 --seed 1 --device auto --data_dir data/raw --out data/generated/mnist_marginals_s1.npz
python experiments/neural_networks/experiment_scripts/local_mass_nn.py --mode trained --npz data/generated/mnist_marginals_s1.npz --pi 0.5 --slab_loc 0.0 --grid_max 0.3 --tag _mnist
```

Use `--device cuda` explicitly on a CUDA system. The portable data default is now `./data/raw`; the command records it explicitly.

## Experiment 5: CIFAR-10

The supplied frozen ResNet-56 base remains under `experiment_scripts/flopp/`.
The original gate marginal input was not supplied, so this also creates a new run.

```sh
python experiments/neural_networks/experiment_scripts/structured_marginals.py --arch resnet56 --seed 1 --device auto --data_dir data/raw --out data/generated/cifar_marginals_s1.npz
python experiments/neural_networks/experiment_scripts/local_mass_nn.py --mode trained --npz data/generated/cifar_marginals_s1.npz --pi 0.2 --slab_loc 1.0 --grid_max 1.2 --tag _cifar
```

## Optional ImageNet checkpoint extraction

The 12 ImageNet checkpoints are excluded from this lightweight folder. Obtain
them separately and place them as described in
[`data/imagenet_checkpoints/README.md`](../../data/imagenet_checkpoints/README.md).
The example below re-extracts seed 1 into a new file without replacing the shipped
marginals; repeat for seeds 2 and 3 if needed.

```sh
python experiments/neural_networks/experiment_scripts/structured_marginals.py --arch resnet50 --families gaussian horseshoe spikeslab hardconcrete --pi 0.2 --seed 1 --device cpu --ckpt_dir data/imagenet_checkpoints --out data/generated/imagenet_marginals_s1.npz
```

## Synthetic demonstration

```sh
python experiments/neural_networks/experiment_scripts/local_mass_nn.py --mode synthetic --tag _synthetic
```

Synthetic output is a demonstration, not a replacement for trained paper results.

## ImageNet from scratch and exact settings

See the [minimum reproduction guide](../../docs/REPRODUCIBILITY.md) for the
ImageNet directory structure, a generic training command, all family-specific
hyperparameters, and current verification evidence. It does not require a Gadi
account. Keep new training JSON, checkpoints and extracted marginal JSON together.

## Cluster and older drivers

The two PBS files are preserved unmodified as original Gadi job records/templates.
They contain site-specific project IDs, email settings and storage paths; adapt
them to the target account before use. The ImageNet job expects submission from
`experiments/neural_networks/`; the MNIST job expects submission from
`experiments/neural_networks/experiment_scripts/`.

Drivers such as `run_e1.py`, `run_matrix.py`, `aggregate.py` and `preflight.py`
remain available for provenance. They are not the default paper reproduction
entry points. Phase-2 knowledge-distillation support is preserved in the shared
code but is outside the supplied Exp5 ImageNet phase-1 results.
