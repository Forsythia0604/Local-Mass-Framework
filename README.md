# MI and Normalised Local Divergence Experiments, V4

This directory contains a clean implementation of the three experiments around:

- local mass exponent `kappa`
- Mass Index `MI = dimension / kappa`
- log-domain normalised local divergence

The fitted log-log local mass slope is never reported as MI. Local balls are denoted `B_r`. Zero is only one finite centre and is not described as sparsity detection.

## Structure

```text
experiments/
  common.py
  exp1_compute.py
  exp1_plot.py
  exp2_compute.py
  exp2_plot.py
  exp3_compute.py
  exp3_plot.py
  validate_results.py
results/
  exp1/{raw,processed,figures}
  exp2/{raw,processed,figures}
  exp3/{raw,processed,figures}
```

## Run Commands

```bash
python experiments/exp1_compute.py --dim 1 --datasets breast_cancer wine diabetes --power-indices 0.7 1.0 1.6 --seed 0 --adaptive-radii --eps-likelihood 0.05
python experiments/exp1_plot.py

python experiments/exp2_compute.py --dataset breast_cancer --dim 2 --power-index 1.0 --mismatched-power-index 0.35 --seed 0 --adaptive-radii --eps-likelihood 0.05 --align-case-centres
python experiments/exp2_plot.py

python experiments/exp3_compute.py --digits 0 1 --dim 2 --max-n 2500 --seed 0 --vi-steps 10000 --vi-mc-samples 32 --adaptive-radii --eps-likelihood 0.05 --num-mc-samples-eval 1024 --use-best-elbo-checkpoint
python experiments/exp3_plot.py

python experiments/exp3_compute.py --digits 3 8 --dim 2 --max-n 2500 --seed 0 --vi-steps 10000 --vi-mc-samples 32 --adaptive-radii --eps-likelihood 0.05 --num-mc-samples-eval 1024 --use-best-elbo-checkpoint
python experiments/exp3_plot.py

python experiments/validate_results.py
```

## Main Outputs

```text
results/exp1/processed/exp1_local_exponent_arrays.npz
results/exp1/processed/exp1_kappa_mi_table.csv
results/exp1/processed/exp1_metadata.json

results/exp2/processed/exp2_divergence_arrays.npz
results/exp2/processed/exp2_kappa_mi_divergence_table.csv
results/exp2/processed/exp2_metadata.json

results/exp3/raw/exp3_reference_posterior.npz
results/exp3/raw/exp3_vi_training_traces.npz
results/exp3/processed/exp3_mnist_arrays.npz
results/exp3/processed/exp3_vi_family_comparison.csv
results/exp3/processed/exp3_metadata.json

results/validation_summary.md
```

Plot scripts load saved results only and save `.png` figures only. `EXP3` expects MNIST through `data/mnist/mnist.npz` or `torchvision`; `--allow-synthetic-fallback` is only for smoke tests and is recorded in metadata.
