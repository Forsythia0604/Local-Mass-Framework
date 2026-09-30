# Minimum reproduction guide

All commands run from the repository root. Reference figures and tables remain
unchanged in `results/`; keep new runs in `results/generated/` and `data/generated/`.

## Tested environment

On 2026-09-30, runtime checks used an existing macOS arm64 environment with Python
3.11.16, NumPy 2.4.6, SciPy 1.17.1, Matplotlib 3.11.1, scikit-learn 1.9.0,
Torch 2.13.0 and Torchvision 0.28.0. Computation used CPU.

`requirements-verified.txt` records the plotting versions;
`requirements-training-verified.txt` adds the tested Torch pair. The original
range-based requirements are also retained. These files pin direct dependencies;
they are not complete transitive lockfiles. A fresh installation and CUDA/MPS
execution have not been validated. An attempted separate dependency download timed
out; all completed runtime checks used the already installed environment.

Use Python 3.11 for this tested configuration. Source syntax is mostly compatible
with Python 3.10+, but that does not establish that these pinned package versions
support every such Python version or operating system.

## Experiment-to-output map

| Experiment | Entry | Input | Supplied output stem / manuscript label |
|---|---|---|---|
| 1 | `experiments/local_mass_experiments.py` | Analytic/synthetic examples | `figure1_synthetic_small_ball` |
| 2 | Same script | sklearn Breast Cancer, binary Iris and Wine | `figure2_uci_bayes_aggregate`; `figure2_runs_long.csv`, `figure2_slope_summary.csv`, `figure2_metadata.json` |
| 3 | Same script | Analytic/synthetic example | `figure3_local_rekl_directionality` |
| 4 | `train_marginals.py` then `local_mass_nn.py` | MNIST; original marginals absent | `fig_nn_small_ball`, `fig_nn_directional_rekl`, `table_nn_mi`; labels `fig:nn_smallball`, `fig:nn_directional`, `tab:nn_mi` |
| 5 / CIFAR | `structured_marginals.py --arch resnet56` then `local_mass_nn.py` | CIFAR-10 and included base weights; original marginals absent | `fig_nn_small_ball_cifar`, `fig_nn_directional_rekl_cifar`, `table_nn_mi_cifar`; labels `fig:nn_sb_cifar`, `fig:nn_dir_cifar` |
| 5 / ImageNet | `imagenet_multiseed.py` | Three included ImageNet marginal files | `fig_nn_small_ball_imagenet`, `fig_nn_directional_rekl_imagenet`, new four-family table and per-seed CSV; labels `fig:nn_sb_imagenet`, `fig:nn_dir_imagenet` |

The NN entry files are under `experiments/neural_networks/experiment_scripts/`.
The current MNIST command adds `_mnist` to new output names; the unsuffixed files
in `results/figures/` remain the original paper artifacts. The manuscript's
combined ResNet table is `tab:nn_resnet`; generated per-dataset tables are numerical
exports, not an automatic reconstruction of that combined LaTeX layout.

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
Power Mass Index estimator. The CLI family key `horseshoe` and plot label
`Horseshoe` denote the supplied mean-field Student-t representative, not a claim
that the implemented density is an exact horseshoe distribution.

The channel-gate training code keeps base parameters fixed for the paper paths;
other behavior, including its BatchNorm training mode, is unchanged from the
provided implementation. The plotted hard-concrete active component is the
original effective Gaussian approximation. This review does not change either
scientific definition.

## ImageNet training without a Gadi account

The historical PBS job is not required. Prepare the dataset above, install a
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
8 evaluation workers); this full training route has not been rerun in this review.

Extract the new checkpoints into new marginals, for example:

```sh
python experiments/neural_networks/experiment_scripts/structured_marginals.py --arch resnet50 --seed 1 --device cpu --ckpt_dir results/generated/imagenet_training --out data/generated/imagenet_marginals_s1.npz
```

Repeat for seeds 2 and 3, then use `imagenet_multiseed.py --marg_dir data/generated`.
New marginals get a companion JSON containing arguments, versions and the `.npz`
checksum. Extraction metadata describes extraction settings, not the checkpoint's
original training; keep the training JSON alongside its checkpoint.

## What has actually been checked

- Full default Exp1–3 plotting (QMC power 15) completed. All three PNGs were
  pixel-identical to the supplied files.
- Exp2 CSVs matched within `rtol=1e-6, atol=1e-10`; the largest absolute difference
  was about 1.66e-10 over 360 run-level rows, and 1.26e-12 over 23 slope rows.
- ImageNet plotting completed for all three seeds. Both PNGs were pixel-identical
  to the supplied files; the new export writes all four families and 12 run rows.
- All 12 original ImageNet checkpoints were re-extracted locally. Every array in
  each of the three `.npz` files matched the included arrays exactly.
- MNIST and CIFAR entry points each completed one synthetic batch for all four
  families, including actual forward/backward steps, extraction, automatic parent
  creation and metadata saving. This is a plumbing test, not full training or a
  confirmation of the paper's trained MNIST/CIFAR numbers.
- The structured driver completed a synthetic-batch run and saved a readable
  checkpoint plus full CLI metadata into a previously absent directory.

Full MNIST/CIFAR/ImageNet training, actual dataset downloads and CUDA/MPS execution
remain unverified here. Original MNIST/CIFAR marginal inputs are still absent.
Machine-readable evidence is in `docs/VALIDATION.json`.
