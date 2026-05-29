# EXP3: Directional Normalisation

## Purpose

EXP3 validates directional normalised local RE-KL diagnostics in a controlled one-dimensional family:

```text
rho_a(x) proportional to |x|^(a-1) exp(-x^2/2)
```

Near zero:

```text
P_a((-r,r)) asymp r^a
MI_pow(P_a,0) = 1/a
```

With `p=P_{a_p}` and `q=P_{a_q}`, preservation is theoretically equivalent to:

```text
a_q <= a_p
```

## Packaged Run

```text
results/exp3_directional_normalisation/20260528_094900/
```

Key files:

- `config.yaml`: exact run configuration.
- `metadata.json`: runtime and package metadata.
- `raw_metrics.csv`: radius-wise deterministic quadrature results.
- `summary_metrics.csv`: case-level summary and classification.
- `monte_carlo_metrics.csv`: Monte Carlo validation rows.
- `logs/failures.csv`: failure log; empty except header when no failures occur.
- `figures/*.png` and `figures/*.pdf`: generated plots.

## Setup

| Item | Value |
|---|---|
| `a_p` | `0.5,1.0,2.0` |
| `a_q` | `0.25,0.5,1.0,2.0,4.0` |
| `alpha` | `0.3,0.5,0.8` |
| Radius range | `10^-4` to `10^-0.5` |
| Summary rows | `45` |
| Failures | `0` |

## Main Result

All 45 cases match the theoretical preservation classification. Estimated MI values also match the theoretical values `1/a`:

| `a` | Theoretical MI | Estimated MI |
|---:|---:|---:|
| 0.25 | 4.000 | 4.004 |
| 0.50 | 2.000 | 2.002 |
| 1.00 | 1.000 | 1.001 |
| 2.00 | 0.500 | 0.500 |
| 4.00 | 0.250 | 0.250 |

## Interpretation

The two normalised directions behave differently:

```text
D_alpha^{B_r}(q||p) / p(B_r)
D_alpha^{B_r}(p||q) / q(B_r)
```

The `p||q` direction normalised by `q(B_r)` is the sensitive diagnostic for local under-coverage when `q` assigns too little local mass near zero.
