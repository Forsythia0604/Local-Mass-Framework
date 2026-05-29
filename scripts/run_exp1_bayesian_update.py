import argparse
import json
import platform
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import scipy.optimize
import scipy.stats
import yaml
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EPS = 1.0e-12


def load_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def make_run_dir(config):
    root = Path(config["output_root"])
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    run_dir = root / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "figures").mkdir()
    (run_dir / "logs").mkdir()
    return run_dir


def save_yaml(path, data):
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def package_versions():
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "matplotlib": plt.matplotlib.__version__,
        "pyyaml": yaml.__version__,
    }


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def r_grid_from_config(config):
    spec = config["r_grid"]
    return np.logspace(spec["start_exp"], spec["stop_exp"], int(spec["num"]))


def generate_regression(seed, dim, n_samples, sigma_noise, sparsity):
    rng = np.random.default_rng(seed)
    active_count = max(1, int(round(sparsity * dim)))
    active = np.sort(rng.choice(dim, size=active_count, replace=False))
    theta_true = np.zeros(dim)
    theta_true[active] = rng.normal(0.0, 1.0, size=active_count)
    x = rng.normal(0.0, 1.0, size=(n_samples, dim))
    y = x @ theta_true + rng.normal(0.0, sigma_noise, size=n_samples)
    return x, y, theta_true, active


def prior_log_density_1d(x, prior_name, params):
    ax = np.abs(x)
    if prior_name == "laplace":
        lam = float(params.get("lambda", 1.0))
        return np.log(lam / 2.0) - lam * ax
    if prior_name == "student_t":
        nu = float(params.get("nu", 3.0))
        return scipy.stats.t.logpdf(x, df=nu)
    if prior_name == "horseshoe_like":
        eps = float(params.get("epsilon", EPS))
        return np.log(np.log1p(1.0 / (x * x + eps))) - np.log(2.0 * np.pi)
    raise ValueError(f"Unknown prior: {prior_name}")


def negative_log_prior(theta, prior_name, params):
    if prior_name == "laplace":
        lam = float(params.get("lambda", 1.0))
        return lam * np.sum(np.sqrt(theta * theta + EPS))
    if prior_name == "student_t":
        nu = float(params.get("nu", 3.0))
        return 0.5 * (nu + 1.0) * np.sum(np.log1p(theta * theta / nu))
    if prior_name == "horseshoe_like":
        eps = float(params.get("epsilon", EPS))
        values = np.log1p(1.0 / (theta * theta + eps))
        return -np.sum(np.log(values + EPS))
    raise ValueError(f"Unknown prior: {prior_name}")


def grad_negative_log_prior(theta, prior_name, params):
    if prior_name == "laplace":
        lam = float(params.get("lambda", 1.0))
        return lam * theta / np.sqrt(theta * theta + EPS)
    if prior_name == "student_t":
        nu = float(params.get("nu", 3.0))
        return (nu + 1.0) * theta / (nu + theta * theta)
    if prior_name == "horseshoe_like":
        eps = float(params.get("epsilon", EPS))
        s = theta * theta + eps
        l_value = np.log1p(1.0 / s)
        return 2.0 * theta / (s * (s + 1.0) * (l_value + EPS))
    raise ValueError(f"Unknown prior: {prior_name}")


def second_negative_log_prior(theta, prior_name, params):
    if prior_name == "laplace":
        lam = float(params.get("lambda", 1.0))
        return lam * EPS / np.power(theta * theta + EPS, 1.5)
    if prior_name == "student_t":
        nu = float(params.get("nu", 3.0))
        return (nu + 1.0) * (nu - theta * theta) / np.square(nu + theta * theta)

    step = 1.0e-4
    second = np.zeros_like(theta)
    for idx in range(theta.size):
        basis = np.zeros_like(theta)
        basis[idx] = step
        g_plus = grad_negative_log_prior(theta + basis, prior_name, params)[idx]
        g_minus = grad_negative_log_prior(theta - basis, prior_name, params)[idx]
        second[idx] = (g_plus - g_minus) / (2.0 * step)
    return second


def fit_map_laplace(x, y, sigma_noise, prior_name, prior_params, opt_config):
    dim = x.shape[1]
    xtx = x.T @ x
    xty = x.T @ y
    inv_noise_var = 1.0 / (sigma_noise * sigma_noise)

    def objective(theta):
        resid = y - x @ theta
        nll = 0.5 * inv_noise_var * np.dot(resid, resid)
        return nll + negative_log_prior(theta, prior_name, prior_params)

    def gradient(theta):
        return inv_noise_var * (xtx @ theta - xty) + grad_negative_log_prior(theta, prior_name, prior_params)

    result = scipy.optimize.minimize(
        objective,
        np.zeros(dim),
        method="L-BFGS-B",
        jac=gradient,
        options={
            "maxiter": int(opt_config.get("max_iter", 1000)),
            "ftol": float(opt_config.get("tolerance", 1.0e-7)),
        },
    )

    theta_map = result.x
    prior_second = second_negative_log_prior(theta_map, prior_name, prior_params)
    hessian = inv_noise_var * xtx + np.diag(prior_second)
    jitter = 1.0e-8
    hessian_status = "ok"
    covariance = None
    for _ in range(8):
        try:
            covariance = np.linalg.inv(hessian + jitter * np.eye(dim))
            np.linalg.cholesky(covariance + 1.0e-12 * np.eye(dim))
            break
        except np.linalg.LinAlgError:
            jitter *= 10.0
            covariance = None
    if covariance is None:
        covariance = np.linalg.pinv(hessian + jitter * np.eye(dim))
        hessian_status = "pinv_used"
    if not result.success:
        hessian_status = f"optimisation_warning:{result.message}"

    variances = np.maximum(np.diag(covariance), 1.0e-12)
    return theta_map, variances, result, hessian_status


def prior_mass_abs_less_than(r, prior_name, params):
    if prior_name == "laplace":
        lam = float(params.get("lambda", 1.0))
        return 1.0 - np.exp(-lam * r)
    if prior_name == "student_t":
        nu = float(params.get("nu", 3.0))
        return scipy.stats.t.cdf(r, df=nu) - scipy.stats.t.cdf(-r, df=nu)
    if prior_name == "horseshoe_like":
        return (r * np.log1p(1.0 / (r * r)) + 2.0 * np.arctan(r)) / np.pi
    raise ValueError(f"Unknown prior: {prior_name}")


def normal_mass_abs_less_than(r, mean, variance):
    sd = np.sqrt(max(variance, 1.0e-12))
    return scipy.stats.norm.cdf((r - mean) / sd) - scipy.stats.norm.cdf((-r - mean) / sd)


def estimate_slope(r_values, masses):
    r_values = np.asarray(r_values)
    masses = np.asarray(masses)
    mask = np.isfinite(masses) & (masses > 0.0)
    if mask.sum() < 3:
        return np.nan, np.nan, int(mask.sum())
    slope, intercept = np.polyfit(np.log(r_values[mask]), np.log(masses[mask]), 1)
    mi_hat = 1.0 / slope if slope > 0 else np.nan
    return float(slope), float(mi_hat), int(mask.sum())


def selected_coordinates(active, dim, rng, config):
    active = list(map(int, active))
    inactive = [idx for idx in range(dim) if idx not in set(active)]
    max_active = min(int(config["selection"]["max_active"]), len(active))
    max_inactive = min(int(config["selection"]["max_inactive"]), len(inactive))
    max_random = min(int(config["selection"]["max_random"]), dim)

    selected = []
    for idx in active[:max_active]:
        selected.append((idx, "active"))
    for idx in inactive[:max_inactive]:
        selected.append((idx, "inactive"))
    for idx in rng.choice(dim, size=max_random, replace=False):
        selected.append((int(idx), "random"))
    return selected


def run_experiment(config, run_dir):
    rows = []
    summaries = []
    failures = []
    r_values = r_grid_from_config(config)

    total = (
        len(config["seeds"])
        * len(config["dims"])
        * len(config["n_samples"])
        * len(config["priors"])
    )
    with tqdm(total=total, desc="Exp1 configurations") as progress:
        for seed in config["seeds"]:
            set_seed(int(seed))
            for dim in config["dims"]:
                for n_samples in config["n_samples"]:
                    x, y, theta_true, active = generate_regression(
                        int(seed),
                        int(dim),
                        int(n_samples),
                        float(config["sigma_noise"]),
                        float(config["sparsity"]),
                    )
                    rng = np.random.default_rng(int(seed) + 1009 * int(dim) + int(n_samples))
                    coords = selected_coordinates(active, int(dim), rng, config)
                    active_set = set(map(int, active))

                    for prior_name, prior_params in config["priors"].items():
                        start = time.time()
                        try:
                            theta_map, variances, opt_result, hessian_status = fit_map_laplace(
                                x,
                                y,
                                float(config["sigma_noise"]),
                                prior_name,
                                prior_params or {},
                                config["map_optimisation"],
                            )

                            for coord, coord_type in coords:
                                prior_masses = np.array(
                                    [prior_mass_abs_less_than(r, prior_name, prior_params or {}) for r in r_values]
                                )
                                post_masses = np.array(
                                    [normal_mass_abs_less_than(r, theta_map[coord], variances[coord]) for r in r_values]
                                )
                                prior_slope, prior_mi, prior_points = estimate_slope(r_values, prior_masses)
                                post_slope, post_mi, post_points = estimate_slope(r_values, post_masses)

                                for r, m_prior, m_post in zip(r_values, prior_masses, post_masses):
                                    ratio = m_post / m_prior if m_prior > 0 else np.nan
                                    rows.append(
                                        {
                                            "seed": seed,
                                            "dimension": dim,
                                            "n_samples": n_samples,
                                            "sigma_noise": config["sigma_noise"],
                                            "sparsity": config["sparsity"],
                                            "prior_name": prior_name,
                                            "coordinate_index": coord,
                                            "coordinate_type": coord_type,
                                            "is_active_true": coord in active_set,
                                            "theta_true": theta_true[coord],
                                            "posterior_mean_laplace": theta_map[coord],
                                            "posterior_variance_laplace": variances[coord],
                                            "r": r,
                                            "m_prior": m_prior,
                                            "m_posterior": m_post,
                                            "ratio_posterior_to_prior": ratio,
                                            "estimated_prior_slope": prior_slope,
                                            "estimated_posterior_slope": post_slope,
                                            "estimated_prior_MI": prior_mi,
                                            "estimated_posterior_MI": post_mi,
                                            "prior_slope_points": prior_points,
                                            "posterior_slope_points": post_points,
                                            "posterior_approximation": config["posterior_approximation"]["method"],
                                            "fallback_used": config["posterior_approximation"]["fallback_used"],
                                            "optimisation_success": bool(opt_result.success),
                                            "optimisation_message": str(opt_result.message),
                                            "hessian_status": hessian_status,
                                            "runtime_seconds_config": time.time() - start,
                                        }
                                    )

                                summaries.append(
                                    {
                                        "seed": seed,
                                        "dimension": dim,
                                        "n_samples": n_samples,
                                        "prior_name": prior_name,
                                        "coordinate_index": coord,
                                        "coordinate_type": coord_type,
                                        "estimated_prior_slope": prior_slope,
                                        "estimated_posterior_slope": post_slope,
                                        "estimated_prior_MI": prior_mi,
                                        "estimated_posterior_MI": post_mi,
                                        "min_ratio": float(np.nanmin(post_masses / prior_masses)),
                                        "max_ratio": float(np.nanmax(post_masses / prior_masses)),
                                        "optimisation_success": bool(opt_result.success),
                                        "hessian_status": hessian_status,
                                    }
                                )
                        except Exception as exc:
                            failures.append(
                                {
                                    "seed": seed,
                                    "dimension": dim,
                                    "n_samples": n_samples,
                                    "prior_name": prior_name,
                                    "failure_message": repr(exc),
                                }
                            )
                        progress.update(1)

    raw = pd.DataFrame(rows)
    summary = pd.DataFrame(summaries)
    failure_df = pd.DataFrame(failures)
    raw.to_csv(run_dir / "raw_metrics.csv", index=False)
    summary.to_csv(run_dir / "summary_metrics.csv", index=False)
    failure_df.to_csv(run_dir / "logs" / "failures.csv", index=False)
    make_figures(raw, summary, run_dir)
    return raw, summary, failures


def make_figures(raw, summary, run_dir):
    if raw.empty:
        return
    figures = run_dir / "figures"
    example_keys = raw[["seed", "dimension", "n_samples", "prior_name", "coordinate_index", "coordinate_type"]].drop_duplicates()
    for _, key in example_keys.head(12).iterrows():
        mask = np.ones(len(raw), dtype=bool)
        for col, value in key.items():
            mask &= raw[col].to_numpy() == value
        part = raw.loc[mask].sort_values("r")
        label = f"s{key.seed}_d{key.dimension}_n{key.n_samples}_{key.prior_name}_j{key.coordinate_index}_{key.coordinate_type}"

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(np.log(part["r"]), np.log(part["m_prior"]), label="log prior mass")
        ax.plot(np.log(part["r"]), np.log(part["m_posterior"]), label="log posterior mass")
        ax.set_xlabel("log r")
        ax.set_ylabel("log local mass")
        ax.set_title(label)
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures / f"local_mass_{label}.png", dpi=160)
        fig.savefig(figures / f"local_mass_{label}.pdf")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(part["r"], part["ratio_posterior_to_prior"], marker="o", markersize=2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("r")
        ax.set_ylabel("posterior mass / prior mass")
        ax.set_title(label)
        fig.tight_layout()
        fig.savefig(figures / f"ratio_{label}.png", dpi=160)
        fig.savefig(figures / f"ratio_{label}.pdf")
        plt.close(fig)

    if not summary.empty:
        grouped = (
            summary.groupby(["dimension", "n_samples", "prior_name", "coordinate_type"], as_index=False)[
                ["estimated_prior_slope", "estimated_posterior_slope"]
            ]
            .mean()
            .sort_values(["dimension", "n_samples", "prior_name", "coordinate_type"])
        )
        fig, ax = plt.subplots(figsize=(10, 5))
        positions = np.arange(len(grouped))
        ax.bar(positions - 0.2, grouped["estimated_prior_slope"], width=0.4, label="prior slope")
        ax.bar(positions + 0.2, grouped["estimated_posterior_slope"], width=0.4, label="posterior slope")
        ax.set_xticks(positions)
        ax.set_xticklabels(
            [
                f"d={r.dimension},n={r.n_samples},{r.prior_name},{r.coordinate_type}"
                for r in grouped.itertuples()
            ],
            rotation=75,
            ha="right",
            fontsize=7,
        )
        ax.set_ylabel("mean estimated slope")
        ax.set_title("Exp1 slope summary across seeds")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures / "slope_summary.png", dpi=160)
        fig.savefig(figures / "slope_summary.pdf")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "exp1_bayesian_update.yaml"))
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()
    config = load_config(config_path)
    run_dir = make_run_dir(config)
    save_yaml(run_dir / "config.yaml", config)

    metadata = {
        "experiment_name": config["experiment_name"],
        "start_time": datetime.now().isoformat(),
        "command_line_args": vars(args),
        "config_path": str(config_path),
        "package_versions": package_versions(),
        "posterior_approximation": config["posterior_approximation"],
    }

    start = time.time()
    raw, summary, failures = run_experiment(config, run_dir)
    metadata["end_time"] = datetime.now().isoformat()
    metadata["runtime_seconds"] = time.time() - start
    metadata["num_raw_rows"] = int(len(raw))
    metadata["num_summary_rows"] = int(len(summary))
    metadata["num_failures"] = int(len(failures))
    with open(run_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"Saved Exp1 run to {run_dir}")


if __name__ == "__main__":
    main()
