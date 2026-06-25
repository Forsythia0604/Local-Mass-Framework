# Local-Mass Experiment Figures

This repository contains a compact, reproducible implementation of the
experiment plan in `experiment_section_requirements_figure2_uci_aggregate.md`.
The code generates publication-oriented, single-column figures for finite-radius
local-mass illustrations. More details are in the following paper:

Hanli Xu, Fengxiang He, Sarat Moka, Beyond Global Divergences: A Local-Mass Perspective on Bayesian Inference. 2026. [paper](http://fengxianghe.github.io/paper/LocalMass.pdf)

## What the script generates

- `figure1_synthetic_small_ball.pdf`: synthetic finite-radius small-ball curves.
- `figure2_uci_bayes_aggregate.pdf`: aggregated real-data Bayesian reweighting
  example over Breast Cancer, Iris, and Wine, with 5 seeds each.
- `figure3_local_rekl_directionality.pdf`: synthetic local RE-KL directionality.
- CSV and JSON files under `results/data/` containing the Figure 2 run-level
  curves, slope summaries, and metadata.

PNG previews are also written at 600 dpi unless `--no-png` is passed.

## Model choices

Figure 2 uses three small UCI datasets from `scikit-learn`:

- Breast Cancer, binary classification.
- Iris, restricted to classes 0 and 1.
- Wine, restricted to classes 0 and 1.

For each dataset and seed, covariates are standardized within the training split
and reduced to four PCA covariates. An intercept is then added, so the Bayesian
logistic-regression parameter has dimension 5. The Gaussian-prior posterior is
approximated by a Laplace Gaussian. The center `theta0` is the mean of this
Laplace posterior, which is the posterior mode under the Gaussian approximation.
Both prior and posterior small-ball masses are evaluated around this same
`theta0`.

Small-ball masses are not estimated by rare-event sampling. They are computed by
Sobol quadrature over the Euclidean ball:

```text
mass(B_r(theta0)) = volume(B_r) * average density on B_r(theta0).
```

This keeps the experiment in the finite-radius, low-dimensional regime described
by the MD requirements. It is a finite-radius illustration, not a scalable
diagnostic procedure and not an asymptotic estimator.

## Run

If the packages are installed globally:

```powershell
python experiments/local_mass_experiments.py
```

In this workspace, dependencies were installed into `.codex_deps`. Run with the
bundled Python from Codex:

```powershell
C:\Users\JoKannritsu\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe experiments\local_mass_experiments.py
```

Optional arguments:

```powershell
python experiments/local_mass_experiments.py --qmc-power 15 --output-dir results
```

`--qmc-power 15` uses `2^15` Sobol points per run for the Figure 2 ball
integrals.
