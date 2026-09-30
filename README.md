# Local-Mass Framework — Experiments 1–5

Code and supplied results for **Beyond Global Divergences: A Local-Mass Perspective
on Bayesian Inference**, Hanli Xu, Fengxiang He and Sarat Moka (2026).
[Paper link from the original repository](http://fengxianghe.github.io/paper/LocalMass.pdf).

This is a lightweight consolidation of the existing Experiments 1–3 repository
and the supervisor-provided Experiments 4–5 package. Original figures, tables,
small data files and run records are preserved. Newly generated outputs go into
`results/generated/`, separately from the supplied results.

## Experiment index

| Experiment | Subject | Entry point under `experiments/` | Supplied material |
|---|---|---|---|
| 1 | Synthetic small-ball mass | `local_mass_experiments.py` | PDF and PNG |
| 2 | UCI Bayesian logistic regression | `local_mass_experiments.py` | PDF, PNG, run-level CSV, summaries and metadata |
| 3 | Synthetic directional local RE-KL | `local_mass_experiments.py` | PDF and PNG |
| 4 | LeNet-5 / MNIST weight marginals | `neural_networks/experiment_scripts/train_marginals.py` | Code and reference figures; original marginal input is missing |
| 5 / CIFAR-10 | ResNet-56 channel gates | `neural_networks/experiment_scripts/structured_marginals.py` | Code, reference figures and frozen CIFAR base weights; original marginal input is missing |
| 5 / ImageNet | ResNet-50 channel gates, three seeds | `neural_networks/experiment_scripts/imagenet_multiseed.py` | Three marginal files, reference figures and training records |

Experiments 1–3 intentionally share one script. Experiments 4–5 retain their
original internal module hierarchy. See the [neural-network guide](experiments/neural_networks/README.md)
for training and extraction commands.

## Quick start

Run commands from this repository's root directory. Use Python 3.11 for the
verified version set below. Exp1–3 and saved-marginal ImageNet plotting were
validated on macOS arm64 / CPU; see the [reproduction guide](docs/REPRODUCIBILITY.md)
for exact versions, data preparation, parameters and verification limits.

```sh
python -m venv .venv
# macOS / Linux:
source .venv/bin/activate
# Windows PowerShell instead: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-verified.txt
```

Generate Experiments 1–3:

```sh
python experiments/local_mass_experiments.py
```

This creates `results/generated/exp01_03/figures/` and
`results/generated/exp01_03/data/`. The default uses 2^15 Sobol points per run.
Use `--qmc-power` and `--output-dir` for explicit overrides.

Regenerate the two ImageNet figures from the included marginals (no training,
GPU, ImageNet dataset or ImageNet checkpoint is needed):

```sh
python experiments/neural_networks/experiment_scripts/imagenet_multiseed.py --seeds 1 2 3 --marg_dir data/marginals
```

The figures go to `results/generated/exp04_05/figures/`, together with a
four-family LaTeX table and a 12-row per-seed CSV. The script also prints the
summary. Supplied reference results are not overwritten.

## Layout

```text
experiments/                  Exp1–3 script and intact Exp4–5 module hierarchy
data/marginals/               Included ImageNet marginal inputs, three seeds
data/imagenet_runs/           Original per-run JSON, logs and PBS records
data/imagenet_checkpoints/    Instructions for optional external checkpoints
results/figures/              Supplied reference PDF and PNG figures
results/tables/               Supplied reference LaTeX tables
results/data/                 Supplied Exp2 CSV and JSON data
results/generated/            New outputs; ignored by Git
paper/sections/               Two original LaTeX manuscript fragments
docs/originals/               Original READMEs and provenance, for historical context
docs/                        Merge notes, limitations and file checksums
```

## Reproduction status

- The original MNIST and CIFAR marginal inputs are missing. Retraining is needed
  to produce new curves; exact agreement with the supplied figures is not promised.
- ImageNet has all four families across three saved seeds. The supplied ImageNet
  `.tex` table contains only two families and is retained as a historical artifact;
  the current script generates a separate complete table.
- Several plots and internal keys use `Horseshoe` for the mean-field Student-t
  representative. This initial merge preserves those names and numbers.
- Hard-concrete plots use the original effective Gaussian approximation to the
  active component. Exp1–3 finite-radius illustrations and Exp4–5 pooled marginal
  diagnostics retain their original computational definitions.
- Exp1–3 and ImageNet figures were regenerated successfully and matched the
  supplied PNGs pixel-for-pixel. All ImageNet checkpoint-to-marginal arrays matched
  exactly. Small synthetic-batch NN training checks passed; full training remains
  unverified. See [verification details](docs/REPRODUCIBILITY.md).

See [known issues](docs/KNOWN_ISSUES.md) and [merge notes](docs/MERGE_NOTES.md).

## Sources and attribution

Experiments 1–3 come from `Local-Mass-Framework` (Hanli Xu's supplied repository).
Experiments 4–5 come from the supplied `Experiments4and5` package. This merge
preserves original comments and does not reassign scientific or code authorship.
The original repository remote was
`https://github.com/Forsythia0604/Local-Mass-Framework.git`.

The CIFAR ResNet-56 implementation describes compatibility with the
Torch-Pruning / DepGraph v1.1.4 checkpoint. Its supplied 3.4 MiB base weight is
retained at the location expected by the training script. See
[source and third-party notes](docs/SOURCES.md).

No license file was present in either supplied tree, and no new license is
assigned by this consolidation. Large ImageNet weights and the delivery ZIP are
kept outside this lightweight folder. Git metadata is not copied or initialized.
