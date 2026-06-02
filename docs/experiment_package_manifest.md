# Experiment Package Manifest

This repository is a self-contained experiment package for the Mass Index diagnostic experiments.

## Included Components

| Component | Contents |
|---|---|
| `scripts/` | Three data-only experiment runners plus the standalone two-column plotting script. |
| `configs/` | Run configurations for the package experiments. |
| `results/` | Saved metrics, PNG figures, logs, and run configuration snapshots. |
| `docs/` | Experiment design notes and per-experiment documentation. |

## Packaged Result Runs

| Experiment | Run directory |
|---|---|
| EXP1 | `results/exp1_small_bayes_regression/20260527_111537` |
| EXP2 | `results/exp2_actual_vi_training/20260527_111628` |
| EXP3 | `results/exp3_directional_normalisation/20260528_094900` |

## Configuration Files

| File | Purpose |
|---|---|
| `configs/exp1_small_bayes_regression.yaml` | Configuration matching the packaged EXP1 run. |
| `configs/exp2_actual_vi_training.yaml` | Configuration matching the packaged EXP2 run. |
| `configs/exp3_directional_normalisation.yaml` | Configuration for EXP3 directional normalisation. |
| `configs/exp1_bayesian_update.yaml` | Additional generic Bayesian updating diagnostic configuration. |
| `configs/exp2_global_vs_local.yaml` | Additional generic global-vs-local VI diagnostic configuration. |

Each result run also contains its own run-local configuration file, either `config_used.yaml` or `config.yaml`, so the saved outputs remain reproducible from the exact settings used for that run.

Experiment runners save data only. Figures are regenerated from saved CSV files with `scripts/make_two_column_plots.py`, which writes PNG outputs and does not create PDFs.
