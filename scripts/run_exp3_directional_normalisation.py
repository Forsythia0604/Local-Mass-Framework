import argparse
import json
import math
import platform
import random
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import scipy.integrate
import scipy.special
import yaml
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TINY = 1.0e-300


def load_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_yaml(path, data):
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def make_run_dir(config):
    root = Path(config["output_root"])
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    run_dir = root / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "logs").mkdir()
    return run_dir


def package_versions():
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "pyyaml": yaml.__version__,
    }


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def r_grid_from_config(config):
    spec = config["r_grid"]
    return np.logspace(spec["start_exp"], spec["stop_exp"], int(spec["num"]))


def log_norm_const(a):
    return 0.5 * a * math.log(2.0) + scipy.special.gammaln(0.5 * a)


def log_density(x, a):
    ax = max(abs(float(x)), TINY)
    return (a - 1.0) * math.log(ax) - 0.5 * ax * ax - log_norm_const(a)


def density(x, a):
    return math.exp(log_density(x, a))


def mass_ball(r, a):
    return float(scipy.special.gammainc(0.5 * a, 0.5 * r * r))


def f_alpha_from_ratio(x, alpha):
    return (x**alpha - alpha * x + (alpha - 1.0)) / (alpha - 1.0)


def local_rekl_integrand(x, a_num, a_den, alpha):
    log_num = log_density(x, a_num)
    log_den = log_density(x, a_den)
    log_cross = alpha * log_num + (1.0 - alpha) * log_den
    num_density = math.exp(log_num)
    den_density = math.exp(log_den)
    cross_density = math.exp(log_cross)
    return (cross_density - alpha * num_density + (alpha - 1.0) * den_density) / (alpha - 1.0)


def integrate_local_rekl(r, a_num, a_den, alpha, epsabs, epsrel):
    messages = []
    status = "ok"

    def integrand(x):
        return local_rekl_integrand(x, a_num, a_den, alpha)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", scipy.integrate.IntegrationWarning)
        try:
            value, error = scipy.integrate.quad(
                integrand,
                0.0,
                float(r),
                epsabs=float(epsabs),
                epsrel=float(epsrel),
                limit=200,
                points=[0.0],
            )
            value *= 2.0
            error *= 2.0
        except Exception as exc:
            value = np.nan
            error = np.nan
            status = "failure"
            messages.append(repr(exc))
        for item in caught:
            status = "warning" if status == "ok" else status
            messages.append(str(item.message))
    return float(value), float(error), status, " | ".join(messages)


def estimate_slope_and_mi(r_values, masses):
    r_values = np.asarray(r_values)
    masses = np.asarray(masses)
    mask = np.isfinite(masses) & (masses > 0.0)
    if mask.sum() < 3:
        return np.nan, np.nan, int(mask.sum())
    slope, _ = np.polyfit(np.log(r_values[mask]), np.log(masses[mask]), 1)
    mi = 1.0 / slope if slope > 0 else np.nan
    return float(slope), float(mi), int(mask.sum())


def sample_local_order(a, n, rng):
    magnitude = np.sqrt(rng.gamma(shape=0.5 * a, scale=2.0, size=n))
    signs = rng.choice([-1.0, 1.0], size=n)
    return signs * magnitude


def monte_carlo_metrics(config, r_values):
    if not bool(config["monte_carlo"].get("enabled", True)):
        return pd.DataFrame()
    rng = np.random.default_rng(int(config["monte_carlo"].get("seed", 0)))
    samples_n = int(config["monte_carlo"].get("samples", 200000))
    rows = []
    for a_p in config["a_p_values"]:
        p_samples = sample_local_order(float(a_p), samples_n, rng)
        p_abs = np.abs(p_samples)
        for a_q in config["a_q_values"]:
            q_samples = sample_local_order(float(a_q), samples_n, rng)
            q_abs = np.abs(q_samples)
            for r in r_values:
                p_count = int(np.sum(p_abs < r))
                q_count = int(np.sum(q_abs < r))
                p_mass = p_count / samples_n
                q_mass = q_count / samples_n
                rows.append(
                    {
                        "a_p": a_p,
                        "a_q": a_q,
                        "r": r,
                        "mc_p_mass": p_mass,
                        "mc_q_mass": q_mass,
                        "mc_local_mass_ratio": q_mass / p_mass if p_mass > 0 else np.nan,
                        "mc_samples": samples_n,
                        "mc_p_count_inside": p_count,
                        "mc_q_count_inside": q_count,
                    }
                )
    return pd.DataFrame(rows)


def run_experiment(config, run_dir):
    r_values = r_grid_from_config(config)
    rows = []
    failures = []
    quad_cfg = config["quadrature"]
    total = len(config["a_p_values"]) * len(config["a_q_values"]) * len(config["alpha_values"])

    with tqdm(total=total, desc="Exp3 parameter pairs") as progress:
        for a_p in config["a_p_values"]:
            p_masses = np.array([mass_ball(r, float(a_p)) for r in r_values])
            p_slope, estimated_mi_p, p_points = estimate_slope_and_mi(r_values, p_masses)
            for a_q in config["a_q_values"]:
                q_masses = np.array([mass_ball(r, float(a_q)) for r in r_values])
                q_slope, estimated_mi_q, q_points = estimate_slope_and_mi(r_values, q_masses)
                theoretical_mi_p = 1.0 / float(a_p)
                theoretical_mi_q = 1.0 / float(a_q)
                mi_preserved_theoretical = bool(float(a_q) <= float(a_p))
                mi_preserved_estimated = bool(estimated_mi_q >= estimated_mi_p) if np.isfinite(estimated_mi_q + estimated_mi_p) else False

                for alpha in config["alpha_values"]:
                    for r, p_mass, q_mass in zip(r_values, p_masses, q_masses):
                        try:
                            rekl_q_p, err_q_p, status_q_p, msg_q_p = integrate_local_rekl(
                                r,
                                float(a_q),
                                float(a_p),
                                float(alpha),
                                quad_cfg["epsabs"],
                                quad_cfg["epsrel"],
                            )
                            rekl_p_q, err_p_q, status_p_q, msg_p_q = integrate_local_rekl(
                                r,
                                float(a_p),
                                float(a_q),
                                float(alpha),
                                quad_cfg["epsabs"],
                                quad_cfg["epsrel"],
                            )
                            status = status_q_p if status_q_p == status_p_q else f"{status_q_p};{status_p_q}"
                            message = " | ".join([text for text in [msg_q_p, msg_p_q] if text])
                            rows.append(
                                {
                                    "a_p": a_p,
                                    "a_q": a_q,
                                    "alpha": alpha,
                                    "r": r,
                                    "p_mass": p_mass,
                                    "q_mass": q_mass,
                                    "local_mass_ratio": q_mass / p_mass if p_mass > 0 else np.nan,
                                    "local_rekl_q_p": rekl_q_p,
                                    "local_rekl_p_q": rekl_p_q,
                                    "normalised_local_rekl_q_p": rekl_q_p / p_mass if p_mass > 0 else np.nan,
                                    "normalised_local_rekl_p_q": rekl_p_q / q_mass if q_mass > 0 else np.nan,
                                    "estimated_MI_p": estimated_mi_p,
                                    "estimated_MI_q": estimated_mi_q,
                                    "theoretical_MI_p": theoretical_mi_p,
                                    "theoretical_MI_q": theoretical_mi_q,
                                    "estimated_slope_p": p_slope,
                                    "estimated_slope_q": q_slope,
                                    "slope_points_p": p_points,
                                    "slope_points_q": q_points,
                                    "mi_preserved_theoretical": mi_preserved_theoretical,
                                    "mi_preserved_estimated": mi_preserved_estimated,
                                    "metrics_match_theoretical_classification": bool(
                                        mi_preserved_theoretical == mi_preserved_estimated
                                    ),
                                    "integration_status": status,
                                    "integration_message": message,
                                    "integration_error_q_p": err_q_p,
                                    "integration_error_p_q": err_p_q,
                                }
                            )
                        except Exception as exc:
                            failures.append(
                                {
                                    "a_p": a_p,
                                    "a_q": a_q,
                                    "alpha": alpha,
                                    "r": r,
                                    "failure_message": repr(exc),
                                }
                            )
                    progress.update(1)

    raw = pd.DataFrame(rows)
    summary = summarise(raw)
    failures_df = pd.DataFrame(failures)
    mc = monte_carlo_metrics(config, r_values)
    raw.to_csv(run_dir / "raw_metrics.csv", index=False)
    summary.to_csv(run_dir / "summary_metrics.csv", index=False)
    failures_df.to_csv(run_dir / "logs" / "failures.csv", index=False)
    mc.to_csv(run_dir / "monte_carlo_metrics.csv", index=False)
    return raw, summary, failures, mc


def summarise(raw):
    if raw.empty:
        return pd.DataFrame()
    return (
        raw.groupby(["a_p", "a_q", "alpha"], as_index=False)
        .agg(
            estimated_MI_p=("estimated_MI_p", "first"),
            estimated_MI_q=("estimated_MI_q", "first"),
            theoretical_MI_p=("theoretical_MI_p", "first"),
            theoretical_MI_q=("theoretical_MI_q", "first"),
            mi_preserved_theoretical=("mi_preserved_theoretical", "first"),
            mi_preserved_estimated=("mi_preserved_estimated", "first"),
            metrics_match_theoretical_classification=("metrics_match_theoretical_classification", "first"),
            min_local_mass_ratio=("local_mass_ratio", "min"),
            max_normalised_local_rekl_q_p=("normalised_local_rekl_q_p", "max"),
            max_normalised_local_rekl_p_q=("normalised_local_rekl_p_q", "max"),
            warning_count=("integration_status", lambda x: int(np.sum(x != "ok"))),
        )
        .sort_values(["alpha", "a_p", "a_q"])
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "exp3_directional_normalisation.yaml"))
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
    }
    start = time.time()
    raw, summary, failures, mc = run_experiment(config, run_dir)
    metadata["end_time"] = datetime.now().isoformat()
    metadata["runtime_seconds"] = time.time() - start
    metadata["num_raw_rows"] = int(len(raw))
    metadata["num_summary_rows"] = int(len(summary))
    metadata["num_failures"] = int(len(failures))
    metadata["num_monte_carlo_rows"] = int(len(mc))
    with open(run_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"Saved Exp3 run to {run_dir}")


if __name__ == "__main__":
    main()
