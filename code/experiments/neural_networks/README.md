# Neural-network usage guide

Python modules and drivers retain their original directory structure.
`experiment_scripts/` remains one level below the module root so its existing
imports continue to resolve.

All commands below run from the repository root. New plots default to
`results/generated/exp04_05/figures/`; plotting commands accept `--output-dir`.
Pre-generated figures, tables and historical run records are not included.

## Environment

For saved-marginal plotting, install the root `requirements-verified.txt`. Training or
checkpoint extraction also requires Torch and Torchvision:

```sh
python -m pip install -r requirements-training-verified.txt
```

The source package reports Python 3.12, Torch 2.11 and Torchvision 0.26 for its
local runs. Those are historical notes. The earlier CPU checks used Python 3.11.16,
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

ImageNet gate checkpoints are not included. Place compatible checkpoints as
described in
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

Synthetic output demonstrates the analysis pipeline and does not use trained inputs.

## Cluster and older drivers

The two PBS files are generic templates. Set the target scheduler's queue,
project and storage flags through `qsub`; no personal account or email is stored.
Activate a compatible Python environment first. The ImageNet template requires
`DATA_SRC` to point to the prepared ImageNet directory and `PBS_JOBFS` for staging.
It expects submission from `experiments/neural_networks/`; the MNIST template
expects submission from `experiments/neural_networks/experiment_scripts/`.
Both use package-relative defaults for outputs; the MNIST data directory defaults
to the repository's `data/raw/`. Override `OUT` and `DATA_DIR` as needed.

Auxiliary drivers include `run_e1.py`, `run_matrix.py`, `aggregate.py` and
`preflight.py`. Phase-2 knowledge-distillation support is retained in shared code;
the documented ImageNet workflow below uses phase 1.


## Data preparation


- Exp2 loads the three small datasets bundled with scikit-learn.
- MNIST and CIFAR-10 are downloaded by Torchvision on first training use. The
  documented `--data_dir data/raw` keeps their caches outside tracked inputs.
- ImageNet training requires the user's own prepared ILSVRC2012 data. It is not
  downloaded or included by this repository. The loader uses `ImageFolder` and
  expects **both** `train/` and `validation/`, each containing matching class-name
  subdirectories. A flat validation image directory, or a directory named only
  `val/`, does not match the supplied loader. Organize validation images using the
  official class labels. Use the ImageNet ILSVRC2012 source at
  https://www.image-net.org/challenges/LSVRC/2012/ for access and dataset information.

```text
<imagenet-root>/
  train/
    n01440764/*.JPEG
    ...
  validation/
    n01440764/*.JPEG
    ...
```

ImageNet plotting from the saved `.npz` files does not use this dataset.

## Parameters used by the documented commands


| Setting | Exp2 | Exp4 / MNIST | Exp5 / CIFAR | Exp5 / ImageNet |
|---|---|---|---|---|
| Seeds | 0–4 for each dataset | 1 | 1 | 1, 2, 3 |
| Epochs | Not applicable | 15 | 8 | 8 |
| Batch size | Not applicable | 128 | 128 | 192 |
| Learning rate | Laplace optimization | Adam 1e-3 | Gate SGD 0.05 | Gate SGD 0.05 |
| Anneal epochs | Not applicable | 5 | 4 | 4 |
| Warmup epochs | Not applicable | Not applicable | 0 | 0 |
| Prior | Gaussian scale 2 | Spike-slab, slab weight 0.5 | Spike-slab, slab weight 0.2, slab centre 1 | Same as CIFAR |
| KL weight: Gaussian / Student-t / spike-slab / hard-concrete | Not applicable | 1e-3 / 1e-3 / 1e-3 / 0.1 | 1e-3 / 1e-3 / 1e-2 / 1e-2 | 1e-4 for all four |
| Held-out split | Test fraction 0.30, seed per run | Validation fraction 0.10, split seed 42 | Validation fraction 0.05, split seed 42 | Validation fraction 0.05, split seed 42 |
| Saved coordinates | Dimension 5 | Up to 50,000 per family | 1,008 channels per family | 7,552 channels per family |
| Plot radius grid | 24 points, 0.03–0.45 | 24 points, 1e-3–0.3 | 24 points, 1e-3–1.2 | 24 points, 1e-3–1.2 |

Exp2 uses four PCA covariates plus an intercept, a Laplace Gaussian posterior and
2^15 Sobol points per run. NN plots use RE-KL order 0.5 and the original finite-radius
Power Mass Index estimator.

The channel-gate workflows freeze base parameters while retaining the supplied
BatchNorm training-mode behavior. Hard-concrete analysis uses the supplied
effective Gaussian approximation for the active component.

## ImageNet training on a GPU workstation or cluster


The generic PBS template is optional. Prepare the dataset above, install a
matched Torch/Torchvision build for the intended GPU, then run one family/seed:

```sh
python experiments/neural_networks/run_e1_structured.py --arch resnet50 --dataset imagenet --phase 1 --pretrained --qfamily spikeslab --slab gaussian --slab_loc 1.0 --pi 0.2 --kl_weight 1e-4 --epochs 8 --warmup_epochs 0 --anneal_epochs 4 --gate_lr 0.05 --batch_size 192 --val_frac 0.05 --seed 1 --device cuda --data_dir /path/to/imagenet --results_file results/generated/imagenet_training/r50_spikeslab_p1_s1.json
```

Replace `/path/to/imagenet`. Repeat for `gaussian`, `horseshoe`, `spikeslab`,
`hardconcrete` and seeds `1`, `2`, `3`, changing both the arguments and output
filename. The pretrained base is Torchvision's `IMAGENET1K_V1`; its weights are
downloaded on first use unless cached. The driver creates the output directory
and saves a `_gnet.pt` checkpoint plus JSON, now including the full CLI settings.
The ImageNet loader is configured for a workstation/cluster (12 training workers,
8 evaluation workers); full training was not rerun during packaging.

Extract the new checkpoints into new marginals, for example:

```sh
python experiments/neural_networks/experiment_scripts/structured_marginals.py --arch resnet50 --seed 1 --device cpu --ckpt_dir results/generated/imagenet_training --out data/generated/imagenet_marginals_s1.npz
```

Repeat for seeds 2 and 3, then use `imagenet_multiseed.py --marg_dir data/generated`.
New marginals get a companion JSON containing arguments, versions and the `.npz`
checksum. Extraction metadata describes extraction settings, not the checkpoint's
original training; keep the training JSON alongside its checkpoint.

## Execution limits

Original MNIST/CIFAR marginal inputs and ImageNet gate checkpoints are absent.
Full training, dataset downloads, fresh dependency installation and GPU execution
were not verified during packaging. The pinned requirements list direct
dependencies, rather than a complete transitive lockfile.
