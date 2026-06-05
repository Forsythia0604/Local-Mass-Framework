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
    local_alpha_divergence_log_domain,
    log_likelihood_derivatives_analytic,
    local_log_integrals_radial,
    log_model_density_on_device,
    make_global_integration_grid,
    radial_quadrature_grid,
    result_dirs,
    safe_log_integral_from_grid,
    save_json,
    seed_all,
    select_device,
    trend_from_log_values,
    write_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="EXP2: log-domain raw and normalised local divergence.")
    add_common_arguments(parser)
    parser.add_argument("--dataset", default="breast_cancer")
    parser.add_argument("--dim", type=int, default=2, choices=[1, 2, 3])
    parser.add_argument("--power-index", type=float, default=1.0)
    parser.add_argument("--mismatched-power-index", type=float, default=0.35)
    parser.add_argument("--radii", type=int, default=32)
    parser.add_argument("--radial-points", type=int, default=220)
    parser.add_argument("--directions", type=int, default=160)
    parser.add_argument("--align-case-centres", action="store_true", default=True)
    args = parser.parse_args()

    seed_all(args.seed)
    device_name, device = select_device(prefer_gpu=not args.no_gpu)
    dirs = result_dirs("exp2")
    X, y, model_type, task, data_source = load_sklearn_uci_dataset(
        args.dataset,
        args.dim,
        args.seed,
        allow_synthetic_fallback=args.allow_synthetic_fallback,
    )
    dim = X.shape[1]
    map_est, map_cov = fit_map_and_cov(X, y, task, dim, args.seed)
    local_scale = float(np.sqrt(np.max(np.diag(map_cov))))
    centers = [np.zeros(dim), map_est.copy()]
    global_theta, global_log_w = make_global_integration_grid(map_est, map_cov, dim)
    arrays = {"global_grid_center": map_est}
    raw_arrays = {
        "X": X,
        "y": y,
        "map_estimate": map_est,
        "map_covariance": map_cov,
        "global_theta": global_theta,
        "global_log_weights": global_log_w,
    }
    rows = []

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
        arrays[f"{ctype}_radii"] = radii
        p = DensitySpec("p", task, X, y, center, args.power_index, likelihood_scale=1.0)
        q_specs = [
            (
                "q1",
                "MI-matched",
                "close to reference posterior",
                DensitySpec("q1", task, X, y, center, args.power_index, likelihood_scale=1.0, local_bump=0.02, local_bump_width=max(local_scale, 0.2), bump_center=center),
                args.power_index,
            ),
            (
                "q2",
                "MI-matched",
                "MI-matched but locally distorted",
                DensitySpec("q2", task, X, y, center, args.power_index, likelihood_scale=1.0, local_bump=3.0, local_bump_width=max(0.08, 0.7 * local_scale), bump_center=center),
                args.power_index,
            ),
            (
                "q3",
                "MI-mismatched",
                "MI-mismatched at the diagnostic centre",
                DensitySpec("q3", task, X, y, center, args.mismatched_power_index, likelihood_scale=1.0),
                args.mismatched_power_index,
            ),
        ]
        p_global_log = log_model_density_on_device(global_theta, p, device=device)
        log_zp = safe_log_integral_from_grid(p_global_log, global_log_w)
        local_theta, local_log_w, _ = radial_quadrature_grid(center, float(np.max(radii)), dim, args.radial_points, args.directions)
        p_local_log = log_model_density_on_device(local_theta, p, device=device)
        p_log_mass_unnorm, _ = local_log_integrals_radial(p, center, radii, args.radial_points, args.directions, device=device)
        p_fit = fit_local_exponent_stable(
            radii,
            p_log_mass_unnorm,
            dimension=dim,
            expected_kappa=args.power_index,
            expected_kappa_tolerance=args.expected_kappa_tolerance,
            min_points=args.min_fit_points,
        )
        arrays[f"p_{ctype}_log_ball_mass"] = p_log_mass_unnorm - log_zp
        arrays[f"p_{ctype}_ball_mass_for_display_only"] = np.exp(np.clip(p_log_mass_unnorm - log_zp, -50, 50))
        for case, mi_relation, interpretation, q, expected_q_kappa in q_specs:
            q_global_log = log_model_density_on_device(global_theta, q, device=device)
            log_zq = safe_log_integral_from_grid(q_global_log, global_log_w)
            q_local_log = log_model_density_on_device(local_theta, q, device=device)
            q_log_mass_unnorm, _ = local_log_integrals_radial(q, center, radii, args.radial_points, args.directions, device=device)
            q_fit = fit_local_exponent_stable(
                radii,
                q_log_mass_unnorm,
                dimension=dim,
                expected_kappa=expected_q_kappa,
                expected_kappa_tolerance=args.expected_kappa_tolerance,
                min_points=args.min_fit_points,
            )
            if not radius_diag["radius_selection_reliable"]:
                q_fit["reliable"] = False
                q_fit["reason_if_unreliable"] = "radius selection unreliable: " + radius_diag["radius_selection_reason"]
            div = local_alpha_divergence_log_domain(
                p_local_log,
                q_local_log,
                local_log_w,
                local_theta,
                center,
                radii,
                args.alpha,
                normalise_by="p",
                log_z_p=log_zp,
                log_z_q=log_zq,
            )
            prefix = f"{case}_{ctype}"
            arrays[f"{prefix}_q_log_ball_mass"] = q_log_mass_unnorm - log_zq
            arrays[f"{prefix}_log_raw_divergence"] = div["log_raw_divergence"]
            arrays[f"{prefix}_log_p_ball_mass"] = div["log_denominator_mass"]
            arrays[f"{prefix}_log_normalised_divergence"] = div["log_normalised_divergence"]
            arrays[f"{prefix}_raw_divergence_for_plot"] = div["raw_divergence_for_plot"]
            arrays[f"{prefix}_p_ball_mass_for_plot"] = div["denominator_mass_for_plot"]
            arrays[f"{prefix}_normalised_divergence_for_plot"] = div["normalised_divergence_for_plot"]
            arrays[f"{prefix}_log_normalised_divergence_from_ratio"] = div["log_normalised_divergence_from_ratio"]
            arrays[f"{prefix}_normalised_divergence_from_conditional_expectation"] = div["normalised_divergence_from_conditional_expectation"]
            arrays[f"{prefix}_reliable_mask"] = div["reliable_mask"]
            rows.append(
                {
                    "experiment_name": "EXP2",
                    "dataset": args.dataset,
                    "dataset_source": data_source,
                    "model_type": model_type,
                    "dimension": dim,
                    "centre_type": ctype,
                    "centre_value": center_to_string(center),
                    "alpha": args.alpha,
                    "random_seed": args.seed,
                    "device": device_name,
                    "posterior_approximation_method": "low-dimensional deterministic grid plus local radial quadrature",
                    "case": case,
                    "kappa_relation": mi_relation.replace("MI", "kappa"),
                    "MI_relation": mi_relation,
                    "estimated_kappa": q_fit["estimated_kappa"],
                    "estimated_MI": q_fit["estimated_MI"],
                    "expected_kappa_if_known": expected_q_kappa,
                    "expected_MI_if_known": float(dim) / float(expected_q_kappa),
                    "kappa_fit_reliable": q_fit["reliable"],
                    "reference_kappa_estimated": p_fit["estimated_kappa"],
                    "reference_MI_estimated": p_fit["estimated_MI"],
                    "raw_divergence_trend": trend_from_log_values(div["log_raw_divergence"]),
                    "normalised_divergence_trend": trend_from_log_values(div["log_normalised_divergence"]),
                    "log_raw_divergence_smallest_r": float(div["log_raw_divergence"][0]),
                    "log_raw_divergence_largest_r": float(div["log_raw_divergence"][-1]),
                    "log_p_ball_mass_smallest_r": float(div["log_denominator_mass"][0]),
                    "log_p_ball_mass_largest_r": float(div["log_denominator_mass"][-1]),
                    "log_normalised_divergence_smallest_r": float(div["log_normalised_divergence"][0]),
                    "log_normalised_divergence_largest_r": float(div["log_normalised_divergence"][-1]),
                    "reliable_radius_count": int(np.sum(div["reliable_mask"])),
                    "interpretation": interpretation,
                    "local_scale": radius_diag["local_scale"],
                    "grad_norm_log_likelihood": radius_diag["grad_norm"],
                    "hessian_scale_log_likelihood": radius_diag["hessian_scale"],
                    "eps_likelihood": radius_diag["eps_likelihood"],
                    "radius_selection_reliable": radius_diag["radius_selection_reliable"],
                    "radius_selection_reason": radius_diag["radius_selection_reason"],
                    "reason_if_unreliable": q_fit["reason_if_unreliable"],
                }
            )

    np.savez_compressed(dirs["raw"] / "exp2_reference_and_local_grids.npz", **raw_arrays)
    np.savez_compressed(dirs["processed"] / "exp2_divergence_arrays.npz", **arrays)
    write_csv(dirs["processed"] / "exp2_kappa_mi_divergence_table.csv", rows)
    save_json(
        dirs["processed"] / "exp2_metadata.json",
        {
            "experiment_name": "EXP2",
            "claim": "Raw local divergence vanishes on shrinking balls; normalised local divergence is scale-free.",
            "dataset_name": args.dataset,
            "dataset_source": data_source,
            "model_type": model_type,
            "dimension": dim,
            "centre_values": [center_to_string(c) for c in centers],
            "centre_types": [centre_type(c) for c in centers],
            "alpha_value": args.alpha,
            "random_seed": args.seed,
            "device_info": device_name,
            "normalised_divergence_definition": "log_normalised = log_raw - log_p(B_r); no epsilon denominator is used",
            "plot_values": "for display only: exp(clip(log_value, -50, 50))",
            "adaptive_radii": True,
            "eps_likelihood": args.eps_likelihood,
            "align_case_centres": args.align_case_centres,
        },
    )


if __name__ == "__main__":
    main()
