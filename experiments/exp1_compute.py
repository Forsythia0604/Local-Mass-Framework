from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from experiments.common import (
    DensitySpec,
    add_common_arguments,
    center_to_string,
    centre_type,
    choose_adaptive_radii,
    fit_local_exponent_stable,
    fit_map_and_cov,
    load_sklearn_uci_dataset,
    log_likelihood_derivatives_analytic,
    local_log_integrals_radial,
    result_dirs,
    save_json,
    seed_all,
    select_device,
    write_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="EXP1: robust real-data MI scaling with adaptive local quadrature.")
    add_common_arguments(parser)
    parser.add_argument("--dim", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--datasets", nargs="+", default=["breast_cancer", "wine", "diabetes"])
    parser.add_argument("--power-indices", nargs="+", type=float, default=[0.7, 1.0, 1.6])
    parser.add_argument("--radii", type=int, default=32)
    parser.add_argument("--radial-points", type=int, default=220)
    parser.add_argument("--directions", type=int, default=144)
    args = parser.parse_args()

    seed_all(args.seed)
    device_name, device = select_device(prefer_gpu=not args.no_gpu)
    dirs = result_dirs("exp1")
    arrays = {}
    raw_arrays = {}
    rows = []
    metadata = {
        "experiment_name": "EXP1",
        "claim": "Bayesian updating changes local constants but preserves finite-point MI.",
        "local_ball_notation": "B_r",
        "dataset_names": args.datasets,
        "dimension_requested": args.dim,
        "alpha_value": args.alpha,
        "random_seed": args.seed,
        "device_info": device_name,
        "posterior_approximation_method": "deterministic local radial-direction quadrature",
        "adaptive_radii": True,
        "eps_likelihood": args.eps_likelihood,
        "expected_kappa_tolerance": args.expected_kappa_tolerance,
    }

    for dataset_name in args.datasets:
        X, y, model_type, task, data_source = load_sklearn_uci_dataset(
            dataset_name,
            args.dim,
            args.seed,
            allow_synthetic_fallback=args.allow_synthetic_fallback,
        )
        dim = X.shape[1]
        map_est, map_cov = fit_map_and_cov(X, y, task, dim, args.seed)
        local_scale = float(np.sqrt(np.max(np.diag(map_cov))))
        centers = [np.zeros(dim), map_est.copy()]
        raw_arrays[f"{dataset_name}_X"] = X
        raw_arrays[f"{dataset_name}_y"] = y
        raw_arrays[f"{dataset_name}_map_estimate"] = map_est
        raw_arrays[f"{dataset_name}_map_covariance"] = map_cov

        for a in args.power_indices:
            for center in centers:
                ctype = centre_type(center)
                deriv = log_likelihood_derivatives_analytic(center, X, y, task)
                radius_diag = choose_adaptive_radii(
                    center,
                    local_scale,
                    deriv["grad_norm"],
                    deriv["hessian_spectral_norm"],
                    num_radii=args.radii,
                    eps_likelihood=args.eps_likelihood,
                )
                radii = radius_diag["radii"]
                prior = DensitySpec("prior", task, X, y, center, float(a), likelihood_scale=0.0)
                posterior = DensitySpec("posterior", task, X, y, center, float(a), likelihood_scale=1.0)
                prior_log_mass, prior_quad = local_log_integrals_radial(
                    prior,
                    center,
                    radii,
                    n_radial=args.radial_points,
                    n_dirs=args.directions,
                    device=device,
                )
                post_log_mass, post_quad = local_log_integrals_radial(
                    posterior,
                    center,
                    radii,
                    n_radial=args.radial_points,
                    n_dirs=args.directions,
                    device=device,
                )
                prior_fit = fit_local_exponent_stable(
                    radii,
                    prior_log_mass,
                    dimension=dim,
                    expected_kappa=float(a),
                    expected_kappa_tolerance=args.expected_kappa_tolerance,
                    min_points=args.min_fit_points,
                    prefer_small_radii=args.prefer_small_radii,
                )
                post_fit = fit_local_exponent_stable(
                    radii,
                    post_log_mass,
                    dimension=dim,
                    expected_kappa=float(a),
                    expected_kappa_tolerance=args.expected_kappa_tolerance,
                    min_points=args.min_fit_points,
                    prefer_small_radii=args.prefer_small_radii,
                )
                if not radius_diag["radius_selection_reliable"]:
                    prior_fit["reliable"] = False
                    post_fit["reliable"] = False
                    prior_fit["reason_if_unreliable"] = "radius selection unreliable: " + radius_diag["radius_selection_reason"]
                    post_fit["reason_if_unreliable"] = "radius selection unreliable: " + radius_diag["radius_selection_reason"]
                key = f"{dataset_name}_a{a:g}_{ctype}"
                arrays[f"{key}_radii"] = radii
                arrays[f"{key}_prior_log_mass"] = prior_log_mass
                arrays[f"{key}_posterior_log_mass"] = post_log_mass
                arrays[f"{key}_prior_mass_for_display_only"] = np.exp(np.clip(prior_log_mass, -50.0, 50.0))
                arrays[f"{key}_posterior_mass_for_display_only"] = np.exp(np.clip(post_log_mass, -50.0, 50.0))
                arrays[f"{key}_prior_fit_indices"] = np.asarray(prior_fit["selected_indices"], dtype=int)
                arrays[f"{key}_posterior_fit_indices"] = np.asarray(post_fit["selected_indices"], dtype=int)
                raw_arrays[f"{key}_prior_quadrature_theta"] = prior_quad["theta"]
                raw_arrays[f"{key}_prior_quadrature_log_weights"] = prior_quad["log_weights"]
                raw_arrays[f"{key}_posterior_quadrature_log_values"] = post_quad["log_values"]
                rows.append(
                    {
                        "experiment_name": "EXP1",
                        "dataset": dataset_name,
                        "dataset_source": data_source,
                        "model_type": model_type,
                        "dimension": dim,
                        "a": float(a),
                        "centre_type": ctype,
                        "centre_value": center_to_string(center),
                        "alpha": args.alpha,
                        "random_seed": args.seed,
                        "device": device_name,
                        "posterior_approximation_method": "deterministic local radial-direction quadrature",
                        "prior_kappa_theoretical": float(a),
                        "prior_kappa_estimated": prior_fit["estimated_kappa"],
                        "posterior_kappa_estimated": post_fit["estimated_kappa"],
                        "expected_kappa": float(a),
                        "prior_MI_theoretical": float(dim) / float(a),
                        "prior_MI_estimated": prior_fit["estimated_MI"],
                        "posterior_MI_estimated": post_fit["estimated_MI"],
                        "expected_MI": float(dim) / float(a),
                        "posterior_kappa_error": post_fit["kappa_error_against_expected"],
                        "prior_fit_reliable": prior_fit["reliable"],
                        "posterior_fit_reliable": post_fit["reliable"],
                        "prior_fit_r2": prior_fit["r2"],
                        "posterior_fit_r2": post_fit["r2"],
                        "fit_radius_min": post_fit["fit_radius_min"],
                        "fit_radius_max": post_fit["fit_radius_max"],
                        "num_fit_points": post_fit["num_fit_points"],
                        "local_integration_method": "local radial-direction quadrature over B_r",
                        "local_scale": radius_diag["local_scale"],
                        "grad_norm_log_likelihood": radius_diag["grad_norm"],
                        "hessian_scale_log_likelihood": radius_diag["hessian_scale"],
                        "eps_likelihood": radius_diag["eps_likelihood"],
                        "radius_selection_reliable": radius_diag["radius_selection_reliable"],
                        "radius_selection_reason": radius_diag["radius_selection_reason"],
                        "selected_radius_indices": post_fit["selected_indices"].tolist(),
                        "local_slope_std": post_fit["local_slope_std"],
                        "curvature_score": post_fit["curvature_score"],
                        "reason_if_unreliable": post_fit["reason_if_unreliable"],
                        "prior_reason_if_unreliable": prior_fit["reason_if_unreliable"],
                    }
                )

    np.savez_compressed(dirs["raw"] / "exp1_local_quadrature_raw.npz", **raw_arrays)
    np.savez_compressed(dirs["processed"] / "exp1_local_exponent_arrays.npz", **arrays)
    write_csv(dirs["processed"] / "exp1_kappa_mi_table.csv", rows)
    save_json(dirs["processed"] / "exp1_metadata.json", metadata)


if __name__ == "__main__":
    main()
