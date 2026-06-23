"""Generate finite-radius local-mass experiment figures.

The script follows the experiment constraints in
experiment_section_requirements_figure2_uci_aggregate.md:

- low-dimensional controlled synthetic examples;
- one small real-data Bayesian logistic-regression example;
- three UCI datasets, five seeds each, aggregated in Figure 2;
- posterior-mean centers theta0 for prior and posterior small-ball masses;
- finite-radius curves rather than asymptotic Mass Index estimation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
LOCAL_DEPS = ROOT / ".codex_deps"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))

MPL_CONFIG = ROOT / "results" / ".mplconfig"
MPL_CONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np
from scipy import optimize
from scipy.special import erf, expit
from scipy.stats import norm, qmc
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


FIG_WIDTH_IN = 3.35
PRIOR_SCALE = 2.0
DATASET_SEEDS = tuple(range(5))


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    x: np.ndarray
    y: np.ndarray


@dataclass(frozen=True)
class LaplacePosterior:
    theta_mean: np.ndarray
    covariance: np.ndarray
    optimizer_success: bool
    optimizer_message: str
    gradient_inf_norm: float
    objective_value: float


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.minor.width": 0.5,
            "ytick.minor.width": 0.5,
            "lines.linewidth": 1.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.015,
        }
    )


def ensure_output_dirs(output_dir: Path) -> tuple[Path, Path]:
    figure_dir = output_dir / "figures"
    data_dir = output_dir / "data"
    figure_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    return figure_dir, data_dir


def save_figure(fig: plt.Figure, figure_dir: Path, stem: str, write_png: bool) -> None:
    fig.savefig(figure_dir / f"{stem}.pdf")
    if write_png:
        fig.savefig(figure_dir / f"{stem}.png", dpi=600)
    plt.close(fig)


def set_fixed_log_x_ticks(ax: plt.Axes, ticks: Iterable[float], labels: Iterable[str]) -> None:
    ax.set_xticks(list(ticks))
    ax.set_xticklabels(list(labels))
    ax.xaxis.set_minor_formatter(NullFormatter())


def unit_ball_log_volume(dim: int) -> float:
    return 0.5 * dim * math.log(math.pi) - math.lgamma(0.5 * dim + 1.0)


def logmeanexp(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    max_value = np.max(values, axis=axis, keepdims=True)
    out = np.log(np.mean(np.exp(values - max_value), axis=axis, keepdims=True))
    out = out + max_value
    if axis is None:
        return np.asarray(out).reshape(())
    return np.squeeze(out, axis=axis)


def sobol_points_in_unit_ball(dim: int, power: int, seed: int) -> np.ndarray:
    """Deterministic low-discrepancy points uniformly distributed in B_1(0)."""
    sampler = qmc.Sobol(d=dim + 1, scramble=True, seed=seed)
    uniforms = sampler.random_base2(power)
    eps = np.finfo(float).eps
    normals = norm.ppf(np.clip(uniforms[:, :dim], eps, 1.0 - eps))
    norms = np.linalg.norm(normals, axis=1)
    directions = normals / norms[:, None]
    radii = uniforms[:, dim] ** (1.0 / dim)
    return directions * radii[:, None]


def gaussian_logpdf(points: np.ndarray, mean: np.ndarray, chol: np.ndarray) -> np.ndarray:
    diff = points - mean[None, :]
    solved = np.linalg.solve(chol, diff.T)
    quadratic = np.sum(solved * solved, axis=0)
    dim = mean.size
    logdet = 2.0 * np.sum(np.log(np.diag(chol)))
    return -0.5 * (dim * math.log(2.0 * math.pi) + logdet + quadratic)


def gaussian_ball_logmasses_by_quadrature(
    center: np.ndarray,
    radii: np.ndarray,
    gaussian_mean: np.ndarray,
    gaussian_covariance: np.ndarray,
    unit_ball_points: np.ndarray,
) -> np.ndarray:
    """Approximate Gaussian mass of B_r(center) by Sobol quadrature over the ball."""
    dim = center.size
    chol = np.linalg.cholesky(gaussian_covariance)
    log_vol_unit = unit_ball_log_volume(dim)
    log_masses = []
    for radius in radii:
        points = center[None, :] + radius * unit_ball_points
        local_log_density = gaussian_logpdf(points, gaussian_mean, chol)
        log_masses.append(log_vol_unit + dim * math.log(radius) + logmeanexp(local_log_density))
    return np.asarray(log_masses, dtype=float)


def load_binary_uci_datasets() -> list[DatasetSpec]:
    breast = load_breast_cancer()
    iris = load_iris()
    wine = load_wine()

    iris_mask = iris.target < 2
    wine_mask = wine.target < 2

    return [
        DatasetSpec(
            name="Breast Cancer",
            x=np.asarray(breast.data, dtype=float),
            y=np.asarray(breast.target, dtype=float),
        ),
        DatasetSpec(
            name="Iris 0 vs 1",
            x=np.asarray(iris.data[iris_mask], dtype=float),
            y=np.asarray(iris.target[iris_mask], dtype=float),
        ),
        DatasetSpec(
            name="Wine 0 vs 1",
            x=np.asarray(wine.data[wine_mask], dtype=float),
            y=np.asarray(wine.target[wine_mask], dtype=float),
        ),
    ]


def prepare_design_matrix(spec: DatasetSpec, seed: int) -> tuple[np.ndarray, np.ndarray]:
    x_train, _x_test, y_train, _y_test = train_test_split(
        spec.x,
        spec.y,
        test_size=0.30,
        random_state=seed,
        stratify=spec.y,
    )

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_train)

    # Four PCA covariates plus an intercept give dim(theta) = 5.
    pca = PCA(n_components=4, svd_solver="full", random_state=seed)
    x_low = pca.fit_transform(x_scaled)
    design = np.column_stack([np.ones(x_low.shape[0]), x_low])
    return design, y_train.astype(float)


def logistic_laplace_posterior(
    design: np.ndarray,
    labels: np.ndarray,
    prior_scale: float,
) -> LaplacePosterior:
    dim = design.shape[1]
    prior_precision = 1.0 / (prior_scale * prior_scale)

    def objective(theta: np.ndarray) -> float:
        logits = design @ theta
        nll = np.sum(np.logaddexp(0.0, logits) - labels * logits)
        prior = 0.5 * prior_precision * float(theta @ theta)
        return float(nll + prior)

    def gradient(theta: np.ndarray) -> np.ndarray:
        probs = expit(design @ theta)
        return design.T @ (probs - labels) + prior_precision * theta

    def hessian(theta: np.ndarray) -> np.ndarray:
        probs = expit(design @ theta)
        weights = probs * (1.0 - probs)
        hess = design.T @ (design * weights[:, None])
        hess = hess + prior_precision * np.eye(dim)
        return 0.5 * (hess + hess.T)

    initial = np.zeros(dim, dtype=float)
    result = optimize.minimize(
        objective,
        initial,
        method="trust-exact",
        jac=gradient,
        hess=hessian,
        options={"gtol": 1e-9, "maxiter": 200},
    )
    if not result.success:
        fallback = optimize.minimize(
            objective,
            initial,
            method="L-BFGS-B",
            jac=gradient,
            options={"gtol": 1e-10, "ftol": 1e-12, "maxiter": 1000},
        )
        if fallback.fun <= result.fun:
            result = fallback

    theta = np.asarray(result.x, dtype=float)
    grad_inf_norm = float(np.linalg.norm(gradient(theta), ord=np.inf))
    hess = hessian(theta)
    eigvals, eigvecs = np.linalg.eigh(hess)
    eigvals = np.clip(eigvals, 1e-10, None)
    covariance = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T
    covariance = 0.5 * (covariance + covariance.T)

    return LaplacePosterior(
        theta_mean=theta,
        covariance=covariance,
        optimizer_success=bool(result.success or grad_inf_norm < 1e-6),
        optimizer_message=str(result.message),
        gradient_inf_norm=grad_inf_norm,
        objective_value=float(result.fun),
    )


def write_rows_csv(path: Path, fieldnames: Iterable[str], rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def finite_radius_slopes(log_radii: np.ndarray, log_masses: np.ndarray) -> np.ndarray:
    return np.diff(log_masses, axis=1) / np.diff(log_radii)[None, :]


def aggregate_mean_sem(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(values, axis=0)
    if values.shape[0] <= 1:
        sem = np.zeros_like(mean)
    else:
        sem = np.std(values, axis=0, ddof=1) / math.sqrt(values.shape[0])
    return mean, sem


def plot_figure1(figure_dir: Path, write_png: bool) -> None:
    radii = np.geomspace(1e-3, 0.6, 300)

    gaussian_mass = erf(radii / math.sqrt(2.0))
    power_beta_one_mass = radii**2
    cusp_beta_half_mass = np.sqrt(radii)
    log_singular_mass = radii * (1.0 - np.log(radii))
    spike_weight = 0.25
    slab_scale = 0.35
    spike_slab_mass = spike_weight + (1.0 - spike_weight) * erf(
        radii / (slab_scale * math.sqrt(2.0))
    )

    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH_IN, 1.55), constrained_layout=True)

    ax = axes[0]
    ax.loglog(radii, gaussian_mass, label="Gaussian")
    ax.loglog(radii, power_beta_one_mass, label="power beta=1", linestyle="--")
    ax.loglog(radii, cusp_beta_half_mass, label="cusp beta=-1/2", linestyle=":")
    ax.set_xlabel("radius r")
    ax.set_ylabel("small-ball mass")
    ax.text(0.03, 0.92, "(a)", transform=ax.transAxes, fontweight="bold")
    ax.legend(frameon=False, loc="lower right", handlelength=1.6)

    ax = axes[1]
    ax.loglog(radii, gaussian_mass, label="Gaussian")
    ax.loglog(radii, log_singular_mass, label="log-singular", linestyle="--")
    ax.loglog(radii, spike_slab_mass, label="spike-slab", linestyle=":")
    ax.set_xlabel("radius r")
    ax.text(0.03, 0.92, "(b)", transform=ax.transAxes, fontweight="bold")
    ax.legend(frameon=False, loc="lower right", handlelength=1.6)

    for ax in axes:
        ax.set_xlim(radii[0], radii[-1])
        ax.tick_params(which="both", direction="out", length=2.5)
        ax.grid(False)

    save_figure(fig, figure_dir, "figure1_synthetic_small_ball", write_png)


def run_figure2_experiment(qmc_power: int, data_dir: Path) -> dict[str, object]:
    datasets = load_binary_uci_datasets()
    radii = np.geomspace(0.03, 0.45, 24)
    log_radii = np.log(radii)
    prior_log_curves: list[np.ndarray] = []
    posterior_log_curves: list[np.ndarray] = []
    run_rows: list[dict[str, object]] = []
    run_metadata: list[dict[str, object]] = []

    run_index = 0
    for spec in datasets:
        for seed in DATASET_SEEDS:
            design, labels = prepare_design_matrix(spec, seed)
            posterior = logistic_laplace_posterior(design, labels, PRIOR_SCALE)
            theta0 = posterior.theta_mean
            dim = theta0.size

            unit_ball_points = sobol_points_in_unit_ball(
                dim=dim,
                power=qmc_power,
                seed=1009 + 37 * run_index,
            )

            prior_covariance = (PRIOR_SCALE * PRIOR_SCALE) * np.eye(dim)
            prior_log_mass = gaussian_ball_logmasses_by_quadrature(
                center=theta0,
                radii=radii,
                gaussian_mean=np.zeros(dim),
                gaussian_covariance=prior_covariance,
                unit_ball_points=unit_ball_points,
            )
            posterior_log_mass = gaussian_ball_logmasses_by_quadrature(
                center=theta0,
                radii=radii,
                gaussian_mean=theta0,
                gaussian_covariance=posterior.covariance,
                unit_ball_points=unit_ball_points,
            )

            prior_log_curves.append(prior_log_mass)
            posterior_log_curves.append(posterior_log_mass)

            for radius, prior_log, post_log in zip(radii, prior_log_mass, posterior_log_mass):
                run_rows.append(
                    {
                        "dataset": spec.name,
                        "seed": seed,
                        "radius": float(radius),
                        "prior_log_mass": float(prior_log),
                        "posterior_log_mass": float(post_log),
                        "prior_mass": float(math.exp(prior_log)),
                        "posterior_mass": float(math.exp(post_log)),
                    }
                )

            eigvals = np.linalg.eigvalsh(posterior.covariance)
            run_metadata.append(
                {
                    "dataset": spec.name,
                    "seed": seed,
                    "n_train": int(design.shape[0]),
                    "theta_dimension": int(dim),
                    "optimizer_success": posterior.optimizer_success,
                    "optimizer_message": posterior.optimizer_message,
                    "gradient_inf_norm": posterior.gradient_inf_norm,
                    "objective_value": posterior.objective_value,
                    "theta0_norm": float(np.linalg.norm(theta0)),
                    "posterior_covariance_min_eig": float(np.min(eigvals)),
                    "posterior_covariance_max_eig": float(np.max(eigvals)),
                }
            )
            run_index += 1

    prior_array = np.vstack(prior_log_curves)
    posterior_array = np.vstack(posterior_log_curves)

    prior_mean_log, prior_sem_log = aggregate_mean_sem(prior_array)
    posterior_mean_log, posterior_sem_log = aggregate_mean_sem(posterior_array)

    prior_slopes = finite_radius_slopes(log_radii, prior_array)
    posterior_slopes = finite_radius_slopes(log_radii, posterior_array)
    prior_slope_mean, prior_slope_sem = aggregate_mean_sem(prior_slopes)
    posterior_slope_mean, posterior_slope_sem = aggregate_mean_sem(posterior_slopes)
    slope_radii = np.sqrt(radii[:-1] * radii[1:])

    write_rows_csv(
        data_dir / "figure2_runs_long.csv",
        [
            "dataset",
            "seed",
            "radius",
            "prior_log_mass",
            "posterior_log_mass",
            "prior_mass",
            "posterior_mass",
        ],
        run_rows,
    )

    slope_rows = []
    for idx, radius in enumerate(slope_radii):
        slope_rows.append(
            {
                "radius_midpoint": float(radius),
                "prior_slope_mean": float(prior_slope_mean[idx]),
                "prior_slope_sem": float(prior_slope_sem[idx]),
                "posterior_slope_mean": float(posterior_slope_mean[idx]),
                "posterior_slope_sem": float(posterior_slope_sem[idx]),
            }
        )
    write_rows_csv(
        data_dir / "figure2_slope_summary.csv",
        [
            "radius_midpoint",
            "prior_slope_mean",
            "prior_slope_sem",
            "posterior_slope_mean",
            "posterior_slope_sem",
        ],
        slope_rows,
    )

    metadata = {
        "datasets": [spec.name for spec in datasets],
        "seeds": list(DATASET_SEEDS),
        "n_runs": len(run_metadata),
        "prior_scale": PRIOR_SCALE,
        "theta_dimension": 5,
        "covariates": "four PCA covariates plus intercept",
        "posterior_approximation": "Laplace Gaussian; theta0 is the Laplace posterior mean",
        "radius_grid": [float(x) for x in radii],
        "qmc_points_per_run": int(2**qmc_power),
        "ball_mass_method": "Sobol quadrature over Euclidean balls",
        "runs": run_metadata,
    }
    (data_dir / "figure2_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    return {
        "radii": radii,
        "slope_radii": slope_radii,
        "prior_mean_log": prior_mean_log,
        "prior_sem_log": prior_sem_log,
        "posterior_mean_log": posterior_mean_log,
        "posterior_sem_log": posterior_sem_log,
        "prior_slope_mean": prior_slope_mean,
        "prior_slope_sem": prior_slope_sem,
        "posterior_slope_mean": posterior_slope_mean,
        "posterior_slope_sem": posterior_slope_sem,
        "metadata": metadata,
    }


def plot_figure2(summary: dict[str, object], figure_dir: Path, write_png: bool) -> None:
    radii = np.asarray(summary["radii"])
    slope_radii = np.asarray(summary["slope_radii"])

    prior_mean_log = np.asarray(summary["prior_mean_log"])
    prior_sem_log = np.asarray(summary["prior_sem_log"])
    posterior_mean_log = np.asarray(summary["posterior_mean_log"])
    posterior_sem_log = np.asarray(summary["posterior_sem_log"])
    prior_slope_mean = np.asarray(summary["prior_slope_mean"])
    prior_slope_sem = np.asarray(summary["prior_slope_sem"])
    posterior_slope_mean = np.asarray(summary["posterior_slope_mean"])
    posterior_slope_sem = np.asarray(summary["posterior_slope_sem"])

    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH_IN, 1.55), constrained_layout=True)

    prior_color = "#1f77b4"
    posterior_color = "#d95f02"

    ax = axes[0]
    prior_mean = np.exp(prior_mean_log)
    posterior_mean = np.exp(posterior_mean_log)
    ax.loglog(radii, prior_mean, color=prior_color, label="prior")
    ax.fill_between(
        radii,
        np.exp(prior_mean_log - prior_sem_log),
        np.exp(prior_mean_log + prior_sem_log),
        color=prior_color,
        alpha=0.16,
        linewidth=0,
    )
    ax.loglog(radii, posterior_mean, color=posterior_color, label="posterior")
    ax.fill_between(
        radii,
        np.exp(posterior_mean_log - posterior_sem_log),
        np.exp(posterior_mean_log + posterior_sem_log),
        color=posterior_color,
        alpha=0.16,
        linewidth=0,
    )
    ax.set_xlabel("radius r")
    ax.set_ylabel("mean small-ball mass")
    ax.text(0.03, 0.92, "(a)", transform=ax.transAxes, fontweight="bold")
    ax.legend(frameon=False, loc="lower right", handlelength=1.6)
    set_fixed_log_x_ticks(ax, [0.03, 0.1, 0.3], ["0.03", "0.1", "0.3"])

    ax = axes[1]
    ax.set_xscale("log")
    ax.plot(slope_radii, prior_slope_mean, color=prior_color, label="prior")
    ax.fill_between(
        slope_radii,
        prior_slope_mean - prior_slope_sem,
        prior_slope_mean + prior_slope_sem,
        color=prior_color,
        alpha=0.16,
        linewidth=0,
    )
    ax.plot(slope_radii, posterior_slope_mean, color=posterior_color, label="posterior")
    ax.fill_between(
        slope_radii,
        posterior_slope_mean - posterior_slope_sem,
        posterior_slope_mean + posterior_slope_sem,
        color=posterior_color,
        alpha=0.16,
        linewidth=0,
    )
    ax.axhline(5.0, color="0.45", linestyle=":", linewidth=0.8)
    ax.set_xlabel("radius r")
    ax.set_ylabel("finite-radius slope")
    ax.text(0.03, 0.92, "(b)", transform=ax.transAxes, fontweight="bold")
    ax.legend(frameon=False, loc="lower left", handlelength=1.6)
    set_fixed_log_x_ticks(ax, [0.03, 0.1, 0.3], ["0.03", "0.1", "0.3"])

    for ax in axes:
        ax.tick_params(which="both", direction="out", length=2.5)
        ax.grid(False)

    save_figure(fig, figure_dir, "figure2_uci_bayes_aggregate", write_png)


def plot_figure3(figure_dir: Path, write_png: bool) -> None:
    radii = np.geomspace(1e-4, 1.0, 300)

    # For p(x)=1/2 and q(x)=1/4 |x|^{-1/2} on (-1,1),
    # p(B_r)=r and q(B_r)=sqrt(r). The plotted normalised local
    # RE-KL terms are mass-ratio weighted conditional KL terms.
    conditional_pq = math.log(2.0) - 0.5
    conditional_qp = 1.0 - math.log(2.0)
    local_pq = np.sqrt(radii) * conditional_pq
    local_qp = conditional_qp / np.sqrt(radii)

    fig, ax = plt.subplots(1, 1, figsize=(FIG_WIDTH_IN, 1.45), constrained_layout=True)
    ax.loglog(radii, local_pq, label=r"$p\|q$", color="#1f77b4")
    ax.loglog(radii, local_qp, label=r"$q\|p$", color="#d95f02", linestyle="--")
    ax.set_xlabel("radius r")
    ax.set_ylabel("local RE-KL")
    ax.legend(frameon=False, loc="center right", handlelength=1.8)
    ax.text(0.05, 0.18, "bounded", transform=ax.transAxes, color="#1f77b4")
    ax.text(0.09, 0.82, "diverges", transform=ax.transAxes, color="#d95f02")
    ax.tick_params(which="both", direction="out", length=2.5)
    ax.grid(False)

    save_figure(fig, figure_dir, "figure3_local_rekl_directionality", write_png)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results",
        help="Directory for figures and data.",
    )
    parser.add_argument(
        "--qmc-power",
        type=int,
        default=15,
        help="Use 2^qmc_power Sobol points per Figure 2 run.",
    )
    parser.add_argument(
        "--no-png",
        action="store_true",
        help="Only write PDF figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    figure_dir, data_dir = ensure_output_dirs(args.output_dir)
    write_png = not args.no_png

    plot_figure1(figure_dir, write_png)
    figure2_summary = run_figure2_experiment(args.qmc_power, data_dir)
    plot_figure2(figure2_summary, figure_dir, write_png)
    plot_figure3(figure_dir, write_png)

    generated = sorted(path.name for path in figure_dir.glob("figure*.*"))
    print("Generated figures:")
    for name in generated:
        print(f"  {figure_dir / name}")
    print(f"Figure 2 data: {data_dir}")


if __name__ == "__main__":
    main()
