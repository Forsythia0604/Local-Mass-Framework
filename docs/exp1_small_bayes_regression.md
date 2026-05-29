# EXP1: Small Bayesian Regression

## Purpose

EXP1 tests whether Bayesian updating preserves the local mass order near zero in inactive sparse coordinates.

The local diagnostic is:

```text
R_r = q((-r, r)) / p((-r, r))
```

where `p` is the prior marginal and `q` is the MAP-centred Laplace approximation to the posterior marginal.

## Packaged Run

```text
results/exp1_small_bayes_regression/20260527_111537/
```

Key files:

- `config_used.yaml`: exact run configuration snapshot.
- `metrics/exp1_inactive_summary.csv`: inactive-coordinate summary.
- `metrics/exp1_active_diagnostic_summary.csv`: active-coordinate contrast.
- `metrics/exp1_local_slope_metrics.csv`: local slope estimates.
- `metrics/exp1_hessian_diagnostics.csv`: Laplace approximation stability diagnostics.
- `figures/*.png`: plots used in the report.

## Setup

| Item | Value |
|---|---|
| Sample size | `n=100` |
| Dimension | `d=20` |
| Active coordinates | `4` |
| Seeds | `0,1,2,3,4` |
| Priors | Gaussian, Laplace, Student-t |
| Posterior approximation | MAP-centred Laplace approximation |
| Radius range | `10^-6` to `10^-0.5` |

## Main Result

Inactive-coordinate diagnostics support local power-order preservation:

```text
p((-r,r)) asymp r  =>  q((-r,r)) asymp r
```

The posterior approximation has larger local mass near zero than the prior, but this is mainly a constant-factor change. The median local slopes remain approximately one.

## Interpretation

Active coordinates are a contrast case: the likelihood moves posterior mass away from zero, so local mass near zero becomes extremely small. The main preservation evidence is therefore the inactive-coordinate result.
