# Mass Index Diagnostic Experiments

This repository is a clean experiment package for the three diagnostic experiments used in the local mass / Mass Index / variational approximation analysis.

## Repository Layout

```text
configs/
  exp1_bayesian_update.yaml
  exp2_global_vs_local.yaml
  exp3_directional_normalisation.yaml
  exp1_small_bayes_regression.yaml
  exp2_actual_vi_training.yaml

scripts/
  run_exp1_bayesian_update.py
  run_exp2_global_vs_local.py
  run_exp3_directional_normalisation.py
  make_two_column_plots.py

results/
  exp1_small_bayes_regression/20260527_111537/
  exp2_actual_vi_training/20260527_111628/
  exp3_directional_normalisation/20260528_094900/

docs/
  experiment_implementation_instructions.md
  exp1_small_bayes_regression.md
  exp2_actual_vi_training.md
  exp3_directional_normalisation.md
  experiment_package_manifest.md

```

## Experiments

### EXP1: Small Bayesian Regression

Question: does Bayesian updating preserve local mass order near zero in inactive sparse coordinates?

Packaged output:

```text
results/exp1_small_bayes_regression/20260527_111537/
```

Main finding: inactive-coordinate posterior local mass near zero changes by a constant-scale factor, while the local power order remains close to one.

### EXP2: Actual VI Training

Question: can global KL / ELBO training miss a local spike region?

Packaged output:

```text
results/exp2_actual_vi_training/20260527_111628/
```

Main finding: diagonal Gaussian VI can have moderate global KL while assigning almost no mass to the spike region; a two-component mixture recovers local mass coverage.

### EXP3: Directional Normalisation

Question: do directional normalised local RE-KL diagnostics identify local under-coverage?

Packaged output:

```text
results/exp3_directional_normalisation/20260528_094900/
```

Main finding: for \(\rho_a(x)\propto |x|^{a-1}\exp(-x^2/2)\), MI preservation is exactly characterised by \(a_q\le a_p\), and all 45 tested cases match the theoretical classification.

## Setup

Create an environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

The experiment runners use NumPy, SciPy, pandas, PyYAML, tqdm, and PyTorch. The standalone plotting script also uses matplotlib. EXP2 can use CUDA if available, but CPU execution is supported.

## Running Experiments

Run from the repository root:

```bash
python scripts/run_exp1_bayesian_update.py --config configs/exp1_bayesian_update.yaml
python scripts/run_exp2_global_vs_local.py --config configs/exp2_global_vs_local.yaml
python scripts/run_exp3_directional_normalisation.py --config configs/exp3_directional_normalisation.yaml
```

Each experiment run creates a timestamped directory under `results/<experiment_name>/` and saves data, logs, config snapshots, and metadata only.

## Regenerating Figures

All plotting is centralised in one script:

```bash
python scripts/make_two_column_plots.py
```

The plotting script reads saved CSV files from the latest packaged result directories and writes compact two-column PNG figures under each run's `figures/` directory. It does not generate PDF files.

## Notes on Configurations

The packaged EXP1 and EXP2 result runs use these matching configurations:

```text
configs/exp1_small_bayes_regression.yaml
configs/exp2_actual_vi_training.yaml
```

The generic diagnostic configurations are also kept for additional reruns:

```text
configs/exp1_bayesian_update.yaml
configs/exp2_global_vs_local.yaml
configs/exp3_directional_normalisation.yaml
```
