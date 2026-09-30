# Experiments 4 and 5 — local mass of Bayesian neural-network posteriors

Everything needed to reproduce Experiments 4 and 5 of *Beyond Global Divergences: A
Local-Mass Perspective on Bayesian Inference* (Xu, He, Moka).

* **Experiment 4** — per-weight variational posteriors, LeNet-5 on MNIST.
* **Experiment 5** — structured channel gates, ResNet-56 on CIFAR-10 and ResNet-50 on
  ImageNet.

Both measure the same two objects as Experiments 1–3: the small-ball mass `q(B_r(0))`,
which gives the Power Mass Index, and the directional local RE-KL
`Dbar_alpha(q||p ; B_r)` against `Dbar_alpha(p||q ; B_r)` as `r -> 0`.

---

## Layout

```
Experiments4and5/
├── code/
│   ├── experiment_scripts/          the four scripts that produce the results
│   │   ├── train_marginals.py         Exp 4: train LeNet-5/MNIST, emit marginals
│   │   ├── structured_marginals.py    Exp 5: gate marginals (CIFAR trains, ImageNet loads)
│   │   ├── local_mass_nn.py           figures + Mass Index table (single run)
│   │   ├── imagenet_multiseed.py      Exp 5 ImageNet figures + table (3 seeds)
│   │   ├── run_marginals.pbs          Gadi job for Exp 4  (submit from THIS directory)
│   │   └── flopp/                     ResNet-56 definition + frozen CIFAR-10 base
│   ├── pbs/run_e1_imagenet_phase1.pbs Gadi job for Exp 5 ImageNet (submit from code/)
│   ├── run_e1_structured.py           driver the ImageNet PBS job calls
│   └── distributions/ layers/ qfamilies/ structured/ *.py
│                                      shared modules the scripts import
├── data/
│   ├── marginals/                   per-channel ImageNet gate marginals (.npz), 3 seeds
│   ├── imagenet_checkpoints/        12 trained gate checkpoints (1.1 GB) — OPTIONAL
│   └── imagenet_runs/               per-run JSON + logs + PROVENANCE.md + PBS records
├── figures/                         the figures and tables used in the paper
└── paper/                           the two LaTeX sections, as submitted
```

`code/experiment_scripts/` must stay one level below `code/`: the scripts add their parent
directory to `sys.path` to import `distributions`, `structured`, `qfamilies`, `layers` and
`data.py`. Moving them breaks the imports.

**`data/imagenet_checkpoints/` is 1.1 GB of the ~1.2 GB total and is only needed to
re-extract the marginals.** Delete that one directory to get a 5.4 MB folder that still
reproduces every figure and table from the `.npz` files.

---

## Environment

Figures and tables only (`local_mass_nn.py`, `imagenet_multiseed.py`): numpy, scipy,
matplotlib.

Anything that touches a network (`train_marginals.py`, `structured_marginals.py`) also
needs torch and torchvision; it was run with Python 3.12, torch 2.11 and torchvision 0.26.
On Gadi use `module load python3/3.10.4`; the code is 3.10-compatible.

The CIFAR-10 commands below take `--data_dir`, which defaults to `./data` and downloads the
dataset there on first use. Pass an explicit path to reuse an existing copy.

---

## Reproducing the results

### Experiment 5, ImageNet — no GPU and no ImageNet data required

Everything below runs on CPU in seconds. Run from `code/experiment_scripts/`.

Rebuild the figures and the table straight from the saved marginals:

    python imagenet_multiseed.py --seeds 1 2 3 --marg_dir ../../data/marginals

Re-extract the marginals from the gate checkpoints first, if you want to check that step:

    python structured_marginals.py --arch resnet50 \
        --families gaussian horseshoe spikeslab hardconcrete \
        --pi 0.2 --seed 1 --device cpu \
        --ckpt_dir ../../data/imagenet_checkpoints \
        --out ../../data/marginals/imagenet_marginals_s1.npz

Both paths were verified on 2026-08-24: re-extraction reproduces the shipped `.npz`
exactly, and the rebuilt figures are pixel-identical to those in `figures/`.

### Experiment 5, CIFAR-10 — ~15 min on a laptop GPU/MPS

Trains the gates on the frozen FLOPP ResNet-56 base included in `flopp/`:

    python structured_marginals.py --arch resnet56 --seed 1 \
        --out ../../data/marginals/cifar_marginals_s1.npz
    python local_mass_nn.py --mode trained --npz ../../data/marginals/cifar_marginals_s1.npz \
        --pi 0.2 --slab_loc 1.0 --grid_max 1.2 --tag _cifar

### Experiment 4, MNIST — ~9 min on a laptop GPU/MPS

    python train_marginals.py --arch lenet --dataset mnist --pi 0.5 \
        --epochs 15 --anneal_epochs 5 --device auto \
        --out ../../data/marginals/mnist_marginals_s1.npz
    python local_mass_nn.py --mode trained --npz ../../data/marginals/mnist_marginals_s1.npz \
        --pi 0.5 --slab_loc 0.0 --grid_max 0.3

`local_mass_nn.py --mode synthetic` runs instantly on generated marginals and is the
quickest way to see the machinery work.

### Producing the ImageNet checkpoints from scratch (Gadi)

Only needed if the checkpoints are regenerated. Submit from `code/`:

    qsub -v QFAMILY=spikeslab,SEED=1 pbs/run_e1_imagenet_phase1.pbs

One job per family and seed; ~4h35m and ~160 SU each. See
`data/imagenet_runs/PROVENANCE.md` for the exact settings and job IDs.
**Phase 2 of the original Tier-C pipeline is knowledge-distillation fine-tuning and is
deliberately not part of these experiments** — the gate marginals come from the phase-1
checkpoint alone.

---

## Results

ImageNet / ResNet-50 channel gates, 4 families x 3 seeds, identical settings throughout
(`pi=0.2` so the prior atom mass is 0.8, `kl_weight=1e-4`, 8 epochs, frozen base, 7552 gate
channels). Only the family and the seed differ.

| family | atom | MI_pow | atom mass (mean over 3 seeds) |
|---|---|---|---|
| Gaussian       | no  | 1        | 0 |
| Student-t      | no  | 1        | 0 |
| Spike-and-slab | yes | infinity | 0.06468 |
| Hard-concrete  | yes | infinity | 0.02303 |

The local-mass quantities are essentially deterministic across seeds — the atom mass varies
by about 5e-6, because gate initialisation is a fixed constant and the base network is
frozen, so only the data ordering changes. That is why the figures show no error bands; the
spread is reported numerically in the table caption instead.

Directional local RE-KL at the smallest radius of the grid:

| | `Dbar(pi_0||q)` | `Dbar(q||pi_0)` |
|---|---|---|
| Gaussian       | 4.4e6 | 0.99999 |
| Student-t      | diverges | 0.99990 |
| Spike-and-slab | 6.3   | 0.51 |
| Hard-concrete  | 24.0  | 0.69 |

For the continuous families `Dbar(q||pi_0)` increases monotonically to `f_alpha(0) = 1`, so
the `limsup < 1` hypothesis of Theorem (i) fails by exactly zero margin, and Theorem (ii)
fails because the other direction diverges. For the atom-capable families `Dbar(q||pi_0)` is
strictly below 1, so Theorem (i) applies and returns
`MI_pow(q, theta) = MI_pow(pi_0, theta) = infinity`, matching the table.

---

## Known gaps

* **The MNIST and CIFAR-10 marginals were never saved.** `data/marginals/` holds ImageNet
  only. The MNIST and CIFAR figures in `figures/` therefore cannot be regenerated without
  retraining (~9 and ~15 min respectively), and a retrained run will not reproduce the
  published numbers exactly. Save the `.npz` on any future run.
* **The legend entry "Horseshoe"** in `fig_nn_small_ball.pdf` and
  `fig_nn_small_ball_cifar.pdf` is the mean-field Student-t family. The class is named
  `StudentTQ` in `local_mass_nn.py` but carries `name = "Horseshoe"`; the paper captions
  gloss this. Fixing the label properly needs a one-line code change plus regenerating
  those two figures.
* **The Logarithmic Mass Index is not implemented** anywhere in this code.
