# EXP2: Actual VI Training

## Purpose

EXP2 tests whether global KL / ELBO optimisation is enough to guarantee local mass coverage in a spike-and-main target distribution.

The local diagnostic is:

```text
R_r = q(B_r) / p(B_r)
```

where `B_r` is the small ball around the spike component.

## Packaged Run

```text
results/exp2_actual_vi_training/20260527_111628/
```

Key files:

- `config_used.yaml`: exact run configuration snapshot.
- `metrics/exp2_final_metrics.csv`: global KL and local mass metrics.
- `metrics/exp2_learned_parameters.csv`: learned means, scales, and mixture weights.
- `metrics/exp2_local_mass_by_radius.csv`: radius-wise local mass coverage.
- `metrics/exp2_q_mass_reliability.csv`: QMC reliability diagnostics.
- `metrics/exp2_training_curves.csv`: optimisation traces.
- `figures/*.png`: plots used in the report.

## Setup

| Item | Value |
|---|---|
| Dimensions | `d=2,10` |
| Spike weights | `epsilon=0.01,0.05,0.1` |
| Spike scales | `tau=0.05,0.1` |
| Families | diagonal Gaussian; two-component diagonal Gaussian mixture |
| Objectives | exclusive KL; inclusive KL |
| Evaluation samples | `262144` |
| Local mass estimation | Sobol / QMC |

## Main Result

The diagonal Gaussian severely under-covers the spike region. In `d=10`, `q(B_r)` has zero-count behaviour under QMC, so the local mass ratio is reported as zero even when global KL is moderate.

The two-component mixture family recovers local mass ratios close to one and learns the spike weight accurately.

## Interpretation

The main comparison is not only exclusive versus inclusive KL. The stronger effect is variational family expressiveness: a diagonal Gaussian cannot represent the spike-and-main local structure, while a two-component mixture can.
