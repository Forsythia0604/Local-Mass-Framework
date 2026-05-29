# Experiment Implementation Instructions for Codex

## 0. Project Goal

Implement three experiments for the paper on **Mass Indices, local RE-KL divergence, and local sparsity preservation in variational inference**.

The experiments should not be treated as large-scale benchmark experiments. They are **diagnostic experiments** designed to test whether local mass structure is preserved, distorted, or lost under Bayesian updating and variational approximation.

The central quantities are local probabilities, local mass ratios, local RE-KL values, and normalised local RE-KL quantities.

Do **not** optimise for visually pleasing results. Do **not** tune parameters after seeing results in order to make the theory look better. Record all raw data, including failed or unexpected runs.

---

## 1. Non-negotiable Reproducibility Rules

### 1.1 Raw data logging

Every experiment must save raw numerical data.

Do not only save plots. Every plot must be reproducible from saved CSV or JSON files.

For each experiment, save:

- raw metric values;
- random seeds;
- hyperparameters;
- distribution parameters;
- optimisation settings;
- package versions;
- runtime information;
- command-line/config arguments;
- failure messages, if any.

### 1.2 No reward hacking

The implementation must not:

- drop random seeds because they give inconvenient results;
- change the \(r\)-grid after inspecting the curves;
- change distribution parameters after inspecting results;
- selectively report successful runs only;
- hide unstable optimisation runs;
- smooth curves unless the raw unsmoothed values are also saved;
- clip values silently;
- overwrite previous results without saving a timestamped run directory;
- modify the theoretical metric definitions to obtain nicer figures.

If a result does not support the expected pattern, save it and report it.

### 1.3 Determinism

Use explicit random seeds.

Recommended:

```python
SEEDS = [0, 1, 2, 3, 4]
```

For PyTorch:

```python
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
```

If CUDA is used, log CUDA availability and device name.

### 1.4 Runtime target

Target environment:

- Google Colab;
- CPU acceptable for Exp1 and Exp3;
- GPU optional for Exp2;
- total runtime target: under 1 hour;
- memory target: under 12 GB.

---

## 2. Suggested Repository Structure

Create the following structure:

```text
experiments/
  configs/
    exp1_bayesian_update.yaml
    exp2_global_vs_local.yaml
    exp3_directional_normalisation.yaml

  src/
    distributions.py
    local_metrics.py
    divergence.py
    vi_training.py
    logging_utils.py
    plotting.py
    utils.py

  scripts/
    run_exp1_bayesian_update.py
    run_exp2_global_vs_local.py
    run_exp3_directional_normalisation.py
    make_all_plots.py
    summarise_results.py

  results/
    exp1_bayesian_update/
    exp2_global_vs_local/
    exp3_directional_normalisation/

  README.md
  requirements.txt
```

Each run should create a timestamped directory:

```text
results/<experiment_name>/<YYYYMMDD_HHMMSS>/
  config.yaml
  metadata.json
  raw_metrics.csv
  summary_metrics.csv
  figures/
  logs/
```

---

## 3. Core Mathematical Quantities

Implement these quantities carefully.

Let \(B_r(0)\) denote the local region around zero. In one dimension:

```python
B_r(0) = {theta: abs(theta) < r}
```

In \(d\) dimensions:

```python
B_r(0) = {theta: ||theta||_2 < r}
```

For probability measures \(p\) and \(q\), compute:

```text
p_mass_r = p(B_r(0))
q_mass_r = q(B_r(0))
```

Local mass ratio:

```text
R_r = q(B_r(0)) / p(B_r(0))
```

Local additive discrepancy:

```text
Delta_r = abs(q(B_r(0)) - p(B_r(0)))
```

RE-KL integrand:

```text
f_alpha(x) = (x**alpha - alpha*x + (alpha - 1)) / (alpha - 1)
```

where:

```text
alpha in (0, 1)
```

Default:

```python
ALPHA = 0.5
```

Local RE-KL:

```text
D_alpha^{B_r(0)}(q || p)
```

Normalised local quantities:

```text
A_r^{q||p} = D_alpha^{B_r(0)}(q || p) / p(B_r(0))
A_r^{p||q} = D_alpha^{B_r(0)}(p || q) / q(B_r(0))
```

The key diagnostic is not merely global KL. The key diagnostic is how these local quantities behave as \(r \to 0\).

Use a fixed logarithmic grid:

```python
R_GRID = np.logspace(-3, -0.5, 30)
```

Do not change this grid after seeing results. If numerical underflow occurs, log it and optionally run an additional grid as a separate robustness run.

---

## 4. Exp1: Bayesian Updating Preserves Local Mass Order

### 4.1 Goal

Test whether Bayesian updating preserves local mass order near zero under a regular likelihood.

This experiment corresponds to the theorem that if the likelihood is continuous and strictly positive near zero, then the posterior local mass is comparable to prior local mass, so the Mass Index near zero should be preserved.

### 4.2 Model

Use a synthetic Bayesian regression model.

Data:

```text
y_i = x_i^T theta_true + epsilon_i
epsilon_i ~ N(0, sigma_noise^2)
```

Default settings:

```python
DIMS = [10, 50]
N_SAMPLES = [200, 1000]
SIGMA_NOISE = 0.5
SPARSITY = 0.1
SEEDS = [0, 1, 2, 3, 4]
```

Generate sparse ground truth:

```python
theta_true[j] = 0 for most coordinates
theta_true[j] ~ N(0, 1) for active coordinates
```

Use standard Gaussian covariates:

```python
X_ij ~ N(0, 1)
```

### 4.3 Priors

Implement at least:

1. Laplace prior;
2. Student-\(t_\nu\) prior;
3. Horseshoe-like prior.

Suggested prior definitions:

```text
Laplace: density proportional to exp(-lambda * |theta|)
Student-t: scipy.stats.t(df=nu)
Horseshoe-like marginal: density proportional to log(1 + 1/theta^2)
```

For numerical stability, define a small epsilon inside logs:

```python
EPS = 1e-12
```

### 4.4 Posterior approximation

Use one of the following methods:

Preferred:

```text
PyTorch SGLD or HMC-style approximate sampling
```

Acceptable fallback:

```text
Laplace approximation around MAP
```

Do not silently switch methods. If a fallback is used, log it explicitly.

Recommended sampling settings:

```python
NUM_SAMPLES = 20000
BURN_IN = 5000
THIN = 5
STEP_SIZE = 1e-4 or tuned by config before running
```

If SGLD is unstable, record instability instead of hiding it.

### 4.5 Local mass diagnostics

For selected coordinates:

```python
selected_coordinates = active coordinates + inactive coordinates + random coordinates
```

For each selected coordinate \(j\), estimate:

```text
m_prior_j(r) = prior(|theta_j| < r)
m_post_j(r) = posterior(|theta_j| < r | data)
ratio_j(r) = m_post_j(r) / m_prior_j(r)
```

Estimate the local power slope by regressing:

```text
log m(r) against log r
```

Use only the pre-fixed \(r\)-grid.

Record:

```text
seed
dimension
n_samples
prior_name
coordinate_index
coordinate_type: active / inactive / random
r
m_prior
m_posterior
ratio_posterior_to_prior
estimated_prior_slope
estimated_posterior_slope
estimated_prior_MI
estimated_posterior_MI
```

### 4.6 Expected pattern

The posterior-to-prior local mass ratio should remain bounded away from zero and infinity over sufficiently small \(r\), up to Monte Carlo error.

The estimated prior and posterior local power slopes should be close.

If this fails for some configuration, record it and report it.

### 4.7 Figures

Generate:

1. local mass curves:
   ```text
   log m_prior(r) and log m_post(r) against log r
   ```

2. ratio curves:
   ```text
   m_post(r) / m_prior(r) against r
   ```

3. slope summary table across seeds.

---

## 5. Exp2: Global Divergence Can Miss Local Under-coverage

### 5.1 Goal

Show that a variational approximation can have acceptable global discrepancy while still assigning wrong mass to a sparsity-relevant local region.

This experiment should demonstrate why local mass diagnostics are necessary.

### 5.2 Target distribution

Use a sparse-mixture target:

```text
p(theta) = (1 - eps) N(mu, Sigma) + eps N(0, tau^2 I_d)
```

The small component near zero is sparsity-relevant.

Default parameters:

```python
DIMS = [2, 10, 50]
EPS_VALUES = [0.01, 0.05, 0.1]
TAU_VALUES = [0.02, 0.05, 0.1]
SEEDS = [0, 1, 2, 3, 4]
```

For the main component:

```python
mu = c * ones(d)
c = 1.0
Sigma = I_d
```

### 5.3 Variational families

Implement at least:

1. mean-field Gaussian;
2. diagonal Gaussian mixture with 2 components.

Optional:

3. small normalising flow.

Do not add a normalising flow unless the first two are complete.

### 5.4 Training objectives

Train variational approximations using:

```text
minimise D_KL(q || p)
```

Approximate this by Monte Carlo samples from \(q\):

```text
E_q[log q(theta) - log p(theta)]
```

Training settings:

```python
NUM_STEPS = 5000
BATCH_SIZE = 2048
LEARNING_RATE = 1e-3
SEEDS = [0, 1, 2, 3, 4]
```

Log training loss every 100 steps.

Do not use early stopping based on local mass ratio. Early stopping, if used, may only depend on global objective and must be specified in config before the run.

### 5.5 Metrics

For each trained \(q\), estimate:

Global metrics:

```text
D_KL(q || p)
E_q[log q - log p]
```

Local metrics for every \(r\) in the fixed grid:

```text
p(B_r(0))
q(B_r(0))
R_r = q(B_r(0)) / p(B_r(0))
Delta_r = abs(q(B_r(0)) - p(B_r(0)))
D_alpha^{B_r(0)}(q || p)
A_r^{q||p} = D_alpha^{B_r(0)}(q || p) / p(B_r(0))
```

Use Monte Carlo estimates with enough samples:

```python
EVAL_SAMPLES = 200000
```

If dimension \(d=50\) makes small-ball estimates too noisy, log the effective count inside each \(B_r(0)\). Do not hide the noise.

### 5.6 Required raw-data columns

Save at least:

```text
seed
dim
eps
tau
variational_family
training_step
final_global_kl_estimate
r
p_mass
q_mass
local_mass_ratio
local_additive_error
local_rekl_q_p
normalised_local_rekl_q_p
num_eval_samples
num_p_samples_inside_ball
num_q_samples_inside_ball
```

### 5.7 Expected pattern

A mean-field Gaussian may achieve moderate global KL while severely under-covering \(B_r(0)\), especially when \(\varepsilon\) is small and \(\tau\) is small.

The mixture variational family should usually preserve more local mass near zero, but this is not guaranteed. Record all cases.

### 5.8 Figures

Generate:

1. global KL vs local mass ratio plot;
2. local mass ratio curves over \(r\);
3. normalised local RE-KL curves over \(r\);
4. bar plot comparing variational families.

---

## 6. Exp3: Directional Local Normalisation and One-sided MI Preservation

### 6.1 Goal

Directly test the two directional normalised local quantities appearing in the one-sided MI preservation result.

Do not reduce this experiment to a generic exclusive-vs-inclusive KL demonstration.

The main quantities are:

```text
A_r^{q||p} = D_alpha^{B_r(0)}(q || p) / p(B_r(0))
A_r^{p||q} = D_alpha^{B_r(0)}(p || q) / q(B_r(0))
R_r = q(B_r(0)) / p(B_r(0))
```

### 6.2 Distribution family

Use a controlled local-order family on \(\mathbb R\):

```text
p_a(theta) proportional to |theta|^{a_p - 1} exp(-theta^2 / 2)
q_b(theta) proportional to |theta|^{a_q - 1} exp(-theta^2 / 2)
```

Near zero:

```text
p_a(B_r(0)) ~ r^{a_p}
q_b(B_r(0)) ~ r^{a_q}
```

In one dimension:

```text
MI_pow(p, 0) = 1 / a_p
MI_pow(q, 0) = 1 / a_q
```

### 6.3 Parameter grid

Use:

```python
A_P_VALUES = [0.5, 1.0, 2.0]
A_Q_VALUES = [0.25, 0.5, 1.0, 2.0, 4.0]
ALPHA_VALUES = [0.3, 0.5, 0.8]
R_GRID = np.logspace(-4, -0.5, 40)
```

Do not change this grid after seeing results.

### 6.4 Computation

Use numerical quadrature as the primary method.

Also implement Monte Carlo estimates as a robustness check.

For each pair \((a_p, a_q)\), compute:

```text
p_mass(r)
q_mass(r)
R_r
D_alpha^{B_r(0)}(q || p)
D_alpha^{B_r(0)}(p || q)
A_r^{q||p}
A_r^{p||q}
estimated_MI_p
estimated_MI_q
condition_MI_preserved = estimated_MI_q >= estimated_MI_p
```

The local RE-KL should be computed by integrating the density-level expression on \((-r,r)\):

```text
D_alpha^{B_r(0)}(q || p)
=
int_{-r}^{r} f_alpha(q(theta) / p(theta)) p(theta) dtheta
```

and similarly:

```text
D_alpha^{B_r(0)}(p || q)
=
int_{-r}^{r} f_alpha(p(theta) / q(theta)) q(theta) dtheta
```

Use scipy.integrate.quad with strict tolerances:

```python
epsabs = 1e-10
epsrel = 1e-8
```

If integration warnings occur, log them.

### 6.5 Classification logic

For each \((a_p,a_q)\):

```text
If a_q <= a_p:
    MI(q,0) >= MI(p,0)
    local mass is not under-covered in the MI sense.
If a_q > a_p:
    MI(q,0) < MI(p,0)
    q under-covers p near zero.
```

Record whether the observed local metrics match this classification.

### 6.6 Required raw-data columns

Save at least:

```text
a_p
a_q
alpha
r
p_mass
q_mass
local_mass_ratio
local_rekl_q_p
local_rekl_p_q
normalised_local_rekl_q_p
normalised_local_rekl_p_q
estimated_MI_p
estimated_MI_q
theoretical_MI_p
theoretical_MI_q
mi_preserved_theoretical
mi_preserved_estimated
integration_status
```

### 6.7 Expected pattern

When \(a_q \le a_p\):

```text
MI_pow(q,0) >= MI_pow(p,0)
```

The local mass ratio should not vanish as \(r \to 0\).

When \(a_q > a_p\):

```text
MI_pow(q,0) < MI_pow(p,0)
```

The local mass ratio should decay toward zero.

The normalised local quantities should reflect the different strength of the two directions:

```text
A_r^{q||p}
A_r^{p||q}
```

The purpose is to inspect the local normalisation behaviour, not to make a generic claim that inclusive KL is always better.

### 6.8 Figures

Generate:

1. heatmap over \((a_p,a_q)\) for estimated MI preservation;
2. curves of \(R_r\) against \(r\);
3. curves of \(A_r^{q||p}\) and \(A_r^{p||q}\) against \(r\);
4. table comparing theoretical and estimated MI values.

---

## 7. Implementation Details

### 7.1 Numerically stable log-density functions

Implement log densities wherever possible.

For mixture targets, use logsumexp:

```python
log_p = logsumexp([
    log(1 - eps) + log_normal_main,
    log(eps) + log_normal_spike
])
```

### 7.2 Small-ball probability estimation

For analytic or quadrature-based experiments, use exact numerical integration when available.

For Monte Carlo-based experiments, estimate:

```python
mass = mean(norm(theta_samples, axis=-1) < r)
```

Also save the raw count:

```python
count_inside = sum(norm(theta_samples, axis=-1) < r)
```

If the count is too small, the estimate is noisy. Do not hide this.

### 7.3 Slope and MI estimation

For local mass values:

```text
log m(r) = slope * log r + intercept
```

Then in one dimension:

```text
MI_hat = 1 / slope
```

In \(d\) dimensions near zero:

```text
MI_hat = d / slope
```

Use only valid points where:

```text
m(r) > 0
```

Record how many points were used in the regression.

### 7.4 Plotting standards

All figures should:

- include axis labels;
- include legends;
- include experiment parameters in title or caption;
- save as both PNG and PDF;
- be reproducible from raw CSV files.

Do not manually edit figures outside the script.

---

## 8. Summary Report

After running all experiments, generate:

```text
results/summary_report.md
```

The report should include:

1. environment information;
2. experiment configurations;
3. tables of main metrics;
4. links to raw data files;
5. links to figures;
6. unexpected results;
7. failed runs;
8. whether each experiment supports, partially supports, or does not support the intended diagnostic claim.

Do not write exaggerated conclusions.

Use cautious language:

```text
The results are consistent with...
The diagnostic suggests...
In this configuration, local under-coverage is observed...
This run does not support the expected pattern...
```

Avoid language like:

```text
This proves the theorem empirically.
This confirms the theory completely.
This shows our method is superior.
```

---

## 9. Requirements File

Create a `requirements.txt` containing at least:

```text
numpy
scipy
pandas
matplotlib
pyyaml
tqdm
torch
```

Optional:

```text
seaborn
arviz
numpyro
jax
jaxlib
```

Do not require optional packages unless they are actually used.

---

## 10. Final Deliverables

Codex should produce:

```text
experiments/
  configs/
  src/
  scripts/
  results/
  README.md
  requirements.txt
```

The final result should include runnable commands:

```bash
python scripts/run_exp1_bayesian_update.py --config configs/exp1_bayesian_update.yaml
python scripts/run_exp2_global_vs_local.py --config configs/exp2_global_vs_local.yaml
python scripts/run_exp3_directional_normalisation.py --config configs/exp3_directional_normalisation.yaml
python scripts/make_all_plots.py
python scripts/summarise_results.py
```

All scripts should run without manual editing after dependencies are installed.

---

## 11. Minimal Acceptance Criteria

The implementation is acceptable only if:

1. all three experiments run end-to-end;
2. every experiment saves raw CSV or JSON data;
3. every figure can be regenerated from saved data;
4. all seeds and hyperparameters are logged;
5. failed runs are recorded rather than deleted;
6. Exp3 computes both \(A_r^{q||p}\) and \(A_r^{p||q}\);
7. the summary report explicitly states whether results support or do not support the intended diagnostic claim.

