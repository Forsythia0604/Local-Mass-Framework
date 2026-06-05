from __future__ import annotations

import argparse
import math
import pathlib
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from experiments.common import (
    DensitySpec,
    add_common_arguments,
    ball_mask,
    center_to_string,
    centre_type,
    choose_adaptive_radii,
    fit_local_exponent_stable,
    fit_map_and_cov,
    import_torch,
    load_mnist_binary,
    local_alpha_divergence_log_domain,
    log_likelihood_derivatives_analytic,
    local_log_integrals_radial,
    log_likelihood,
    log_model_density_on_device,
    log_shifted_power_prior,
    make_global_integration_grid,
    radial_directions,
    radial_quadrature_grid,
    result_dirs,
    safe_log_integral_from_grid,
    save_json,
    seed_all,
    select_device,
    sphere_area,
    weighted_mean_cov,
    write_csv,
)


def standard_log_prior(w: np.ndarray, dim: int) -> np.ndarray:
    return log_shifted_power_prior(w, np.zeros(dim), float(dim), 3.0)


def lower_from_params(raw: np.ndarray, d: int) -> np.ndarray:
    L = np.zeros((d, d), dtype=np.float64)
    k = 0
    for i in range(d):
        for j in range(i + 1):
            L[i, j] = math.exp(raw[k]) if i == j else raw[k]
            k += 1
    return L


def mf_sample_logq(params: np.ndarray, base: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    d = base.shape[1]
    mu = params[:d]
    sigma = np.exp(params[d:])
    w = mu.reshape(1, d) + base * sigma.reshape(1, d)
    logq = -0.5 * np.sum(base * base, axis=1) - np.sum(np.log(sigma)) - 0.5 * d * math.log(2.0 * math.pi)
    return w, logq


def fr_sample_logq(params: np.ndarray, base: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    d = base.shape[1]
    mu = params[:d]
    L = lower_from_params(params[d:], d)
    w = mu.reshape(1, d) + base @ L.T
    logdet = float(np.sum(np.log(np.clip(np.diag(L), 1.0e-10, None))))
    logq = -0.5 * np.sum(base * base, axis=1) - logdet - 0.5 * d * math.log(2.0 * math.pi)
    return w, logq


def power_base_samples(rng: np.random.Generator, n: int, d: int, b: float) -> np.ndarray:
    z = rng.normal(size=(n, d))
    dirs = z / np.clip(np.linalg.norm(z, axis=1, keepdims=True), 1.0e-12, None)
    radius = np.sqrt(rng.gamma(shape=b / 2.0, scale=2.0, size=n))
    return dirs * radius.reshape(n, 1)


def power_sample_logq(params: np.ndarray, base: np.ndarray, b: float) -> Tuple[np.ndarray, np.ndarray]:
    d = base.shape[1]
    mu = params[:d]
    scale = np.exp(params[d:])
    w = mu.reshape(1, d) + base * scale.reshape(1, d)
    zr = np.linalg.norm(base, axis=1)
    log_norm = math.log(sphere_area(d)) + (b / 2.0 - 1.0) * math.log(2.0) + math.lgamma(b / 2.0)
    logq = (b - d) * np.log(zr + 1.0e-12) - 0.5 * np.sum(base * base, axis=1) - log_norm - np.sum(np.log(scale))
    return w, logq


def objective(params: np.ndarray, family: str, base: np.ndarray, X: np.ndarray, y: np.ndarray, b: Optional[float] = None) -> Tuple[float, Dict[str, float]]:
    d = X.shape[1]
    if family == "q_MF":
        w, logq = mf_sample_logq(params, base)
    elif family == "q_FR":
        w, logq = fr_sample_logq(params, base)
    else:
        assert b is not None
        w, logq = power_sample_logq(params, base, b)
    ll = log_likelihood(w, X, y, "classification")
    lp = standard_log_prior(w, d)
    terms = ll + lp - logq
    return float(np.mean(terms)), {
        "log_likelihood": float(np.mean(ll)),
        "log_prior": float(np.mean(lp)),
        "entropy_or_negative_log_q": float(np.mean(-logq)),
    }


def finite_difference_grad(params: np.ndarray, family: str, base: np.ndarray, X: np.ndarray, y: np.ndarray, b: Optional[float]) -> np.ndarray:
    grad = np.zeros_like(params)
    step = 1.0e-4
    for k in range(params.size):
        pp = params.copy()
        pm = params.copy()
        pp[k] += step
        pm[k] -= step
        fp, _ = objective(pp, family, base, X, y, b)
        fm, _ = objective(pm, family, base, X, y, b)
        grad[k] = (fp - fm) / (2.0 * step)
    return grad


def init_params(family: str, init_mu: np.ndarray, init_cov: np.ndarray) -> np.ndarray:
    d = init_mu.size
    if family == "q_MF" or family == "q_b":
        return np.concatenate([init_mu.copy(), np.log(np.sqrt(np.clip(np.diag(init_cov), 1.0e-4, None)))])
    L = np.linalg.cholesky(init_cov + 1.0e-5 * np.eye(d))
    raw = []
    for i in range(d):
        for j in range(i + 1):
            raw.append(math.log(max(L[i, j], 1.0e-5)) if i == j else L[i, j])
    return np.concatenate([init_mu.copy(), np.asarray(raw)])


def train_vi_family_numpy(
    family: str,
    X: np.ndarray,
    y: np.ndarray,
    num_steps: int,
    num_mc_samples: int,
    learning_rate: float,
    seed: int,
    init_mu: np.ndarray,
    init_cov: np.ndarray,
    b: Optional[float] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    params = init_params(family, init_mu, init_cov)
    m = np.zeros_like(params)
    v = np.zeros_like(params)
    trace: Dict[str, List[float]] = {k: [] for k in ["step", "elbo", "negative_elbo", "log_likelihood", "log_prior", "entropy_or_negative_log_q", "grad_norm", "learning_rate"]}
    best_params = params.copy()
    best_elbo = -np.inf
    for step_idx in range(1, num_steps + 1):
        base = power_base_samples(rng, num_mc_samples, d, float(b)) if family == "q_b" else rng.normal(size=(num_mc_samples, d))
        grad = finite_difference_grad(params, family, base, X, y, b)
        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * (grad * grad)
        mhat = m / (1.0 - 0.9**step_idx)
        vhat = v / (1.0 - 0.999**step_idx)
        params = params + learning_rate * mhat / (np.sqrt(vhat) + 1.0e-8)
        elbo, parts = objective(params, family, base, X, y, b)
        if elbo > best_elbo:
            best_elbo = elbo
            best_params = params.copy()
        trace["step"].append(float(step_idx))
        trace["elbo"].append(elbo)
        trace["negative_elbo"].append(-elbo)
        trace["log_likelihood"].append(parts["log_likelihood"])
        trace["log_prior"].append(parts["log_prior"])
        trace["entropy_or_negative_log_q"].append(parts["entropy_or_negative_log_q"])
        trace["grad_norm"].append(float(np.linalg.norm(grad)))
        trace["learning_rate"].append(learning_rate)
    return best_params, {k: np.asarray(vv, dtype=np.float64) for k, vv in trace.items()}


def _torch_lower(raw: Any, d: int, torch: Any) -> Any:
    L = torch.zeros((d, d), dtype=raw.dtype, device=raw.device)
    k = 0
    for i in range(d):
        for j in range(i + 1):
            L[i, j] = torch.exp(raw[k]) if i == j else raw[k]
            k += 1
    return L


def _torch_power_base(torch: Any, n: int, d: int, b: float, device: Any, dtype: Any) -> Any:
    z = torch.randn((n, d), dtype=dtype, device=device)
    dirs = z / torch.clamp(torch.linalg.norm(z, dim=1, keepdim=True), min=1.0e-12)
    gamma = torch.distributions.Gamma(torch.tensor(b / 2.0, dtype=dtype, device=device), torch.tensor(0.5, dtype=dtype, device=device))
    radius = torch.sqrt(gamma.sample((n,)))
    return dirs * radius.reshape(n, 1)


def _torch_elbo(params: Any, family: str, base: Any, X: Any, y: Any, b: Optional[float], torch: Any) -> Tuple[Any, Dict[str, Any]]:
    d = X.shape[1]
    if family == "q_MF":
        mu = params[:d]
        sigma = torch.exp(params[d:])
        w = mu.reshape(1, d) + base * sigma.reshape(1, d)
        logq = -0.5 * torch.sum(base * base, dim=1) - torch.sum(torch.log(sigma)) - 0.5 * d * math.log(2.0 * math.pi)
    elif family == "q_FR":
        mu = params[:d]
        L = _torch_lower(params[d:], d, torch)
        w = mu.reshape(1, d) + base @ L.T
        logdet = torch.sum(torch.log(torch.clamp(torch.diag(L), min=1.0e-10)))
        logq = -0.5 * torch.sum(base * base, dim=1) - logdet - 0.5 * d * math.log(2.0 * math.pi)
    else:
        assert b is not None
        mu = params[:d]
        scale = torch.exp(params[d:])
        w = mu.reshape(1, d) + base * scale.reshape(1, d)
        zr = torch.linalg.norm(base, dim=1)
        log_norm = math.log(sphere_area(d)) + (b / 2.0 - 1.0) * math.log(2.0) + math.lgamma(b / 2.0)
        logq = (b - d) * torch.log(zr + 1.0e-12) - 0.5 * torch.sum(base * base, dim=1) - log_norm - torch.sum(torch.log(scale))
    eta = X @ w.T
    yy = y.reshape(-1, 1)
    ll = torch.sum(yy * eta - torch.logaddexp(torch.zeros_like(eta), eta), dim=0)
    lp = -0.5 * torch.sum((w / 3.0) ** 2, dim=1)
    terms = ll + lp - logq
    return torch.mean(terms), {
        "log_likelihood": torch.mean(ll),
        "log_prior": torch.mean(lp),
        "entropy_or_negative_log_q": torch.mean(-logq),
    }


def train_vi_family_torch(
    family: str,
    X: np.ndarray,
    y: np.ndarray,
    num_steps: int,
    num_mc_samples: int,
    learning_rate: float,
    seed: int,
    init_mu: np.ndarray,
    init_cov: np.ndarray,
    b: Optional[float],
    device: Any,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    torch = import_torch()
    assert torch is not None
    dtype = torch.float64
    torch.manual_seed(seed)
    d = X.shape[1]
    params = torch.nn.Parameter(torch.as_tensor(init_params(family, init_mu, init_cov), dtype=dtype, device=device))
    X_t = torch.as_tensor(X, dtype=dtype, device=device)
    y_t = torch.as_tensor(y, dtype=dtype, device=device)
    opt = torch.optim.Adam([params], lr=learning_rate)
    trace: Dict[str, List[float]] = {k: [] for k in ["step", "elbo", "negative_elbo", "log_likelihood", "log_prior", "entropy_or_negative_log_q", "grad_norm", "learning_rate"]}
    best_params = params.detach().cpu().numpy().copy()
    best_elbo = -np.inf
    for step_idx in range(1, num_steps + 1):
        opt.zero_grad(set_to_none=True)
        base = _torch_power_base(torch, num_mc_samples, d, float(b), device, dtype) if family == "q_b" else torch.randn((num_mc_samples, d), dtype=dtype, device=device)
        elbo, parts = _torch_elbo(params, family, base, X_t, y_t, b, torch)
        (-elbo).backward()
        grad_norm = float(torch.linalg.norm(params.grad).detach().cpu()) if params.grad is not None else 0.0
        opt.step()
        value = float(elbo.detach().cpu())
        if value > best_elbo:
            best_elbo = value
            best_params = params.detach().cpu().numpy().copy()
        trace["step"].append(float(step_idx))
        trace["elbo"].append(value)
        trace["negative_elbo"].append(-value)
        trace["log_likelihood"].append(float(parts["log_likelihood"].detach().cpu()))
        trace["log_prior"].append(float(parts["log_prior"].detach().cpu()))
        trace["entropy_or_negative_log_q"].append(float(parts["entropy_or_negative_log_q"].detach().cpu()))
        trace["grad_norm"].append(grad_norm)
        trace["learning_rate"].append(learning_rate)
    return best_params, {k: np.asarray(vv, dtype=np.float64) for k, vv in trace.items()}


def train_vi_family(
    family: str,
    X: np.ndarray,
    y: np.ndarray,
    prior_spec: Dict[str, Any],
    num_steps: int,
    num_mc_samples: int,
    learning_rate: float,
    seed: int,
    device: Any,
    init_mu: np.ndarray,
    init_cov: np.ndarray,
    b: Optional[float] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    torch = import_torch()
    if torch is not None and device is not None:
        return train_vi_family_torch(family, X, y, num_steps, num_mc_samples, learning_rate, seed, init_mu, init_cov, b, device)
    return train_vi_family_numpy(family, X, y, num_steps, num_mc_samples, learning_rate, seed, init_mu, init_cov, b)


def vi_log_density(theta: np.ndarray, family: str, params: np.ndarray, d: int, b: Optional[float] = None) -> np.ndarray:
    theta = np.atleast_2d(theta)
    if family == "q_MF":
        mu = params[:d]
        sigma = np.exp(params[d:])
        z = (theta - mu.reshape(1, d)) / sigma.reshape(1, d)
        return -0.5 * np.sum(z * z, axis=1) - np.sum(np.log(sigma)) - 0.5 * d * math.log(2.0 * math.pi)
    if family == "q_FR":
        mu = params[:d]
        L = lower_from_params(params[d:], d)
        z = np.linalg.solve(L, (theta - mu.reshape(1, d)).T).T
        logdet = float(np.sum(np.log(np.clip(np.diag(L), 1.0e-10, None))))
        return -0.5 * np.sum(z * z, axis=1) - logdet - 0.5 * d * math.log(2.0 * math.pi)
    assert b is not None
    mu = params[:d]
    scale = np.exp(params[d:])
    z = (theta - mu.reshape(1, d)) / scale.reshape(1, d)
    zr = np.linalg.norm(z, axis=1)
    log_norm = math.log(sphere_area(d)) + (b / 2.0 - 1.0) * math.log(2.0) + math.lgamma(b / 2.0)
    return (b - d) * np.log(zr + 1.0e-12) - 0.5 * np.sum(z * z, axis=1) - log_norm - np.sum(np.log(scale))


def vi_log_density_on_device(theta: np.ndarray, family: str, params: np.ndarray, d: int, b: Optional[float], device: Any = None, batch_size: int = 200_000) -> np.ndarray:
    torch = import_torch()
    if torch is None or device is None:
        return vi_log_density(theta, family, params, d, b)
    dtype = torch.float64
    params_t = torch.as_tensor(params, dtype=dtype, device=device)
    outs: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, theta.shape[0], batch_size):
            th = torch.as_tensor(theta[start : start + batch_size], dtype=dtype, device=device)
            if family == "q_MF":
                mu = params_t[:d]
                sigma = torch.exp(params_t[d:])
                z = (th - mu.reshape(1, d)) / sigma.reshape(1, d)
                logq = -0.5 * torch.sum(z * z, dim=1) - torch.sum(torch.log(sigma)) - 0.5 * d * math.log(2.0 * math.pi)
            elif family == "q_FR":
                mu = params_t[:d]
                L = _torch_lower(params_t[d:], d, torch)
                z = torch.linalg.solve_triangular(L, (th - mu.reshape(1, d)).T, upper=False).T
                logdet = torch.sum(torch.log(torch.clamp(torch.diag(L), min=1.0e-10)))
                logq = -0.5 * torch.sum(z * z, dim=1) - logdet - 0.5 * d * math.log(2.0 * math.pi)
            else:
                assert b is not None
                mu = params_t[:d]
                scale = torch.exp(params_t[d:])
                z = (th - mu.reshape(1, d)) / scale.reshape(1, d)
                zr = torch.linalg.norm(z, dim=1)
                log_norm = math.log(sphere_area(d)) + (b / 2.0 - 1.0) * math.log(2.0) + math.lgamma(b / 2.0)
                logq = (b - d) * torch.log(zr + 1.0e-12) - 0.5 * torch.sum(z * z, dim=1) - log_norm - torch.sum(torch.log(scale))
            outs.append(logq.detach().cpu().numpy())
    return np.concatenate(outs)


def expected_q_kappa(family: str, b: Optional[float], center: np.ndarray, dim: int, params: np.ndarray) -> float:
    singular_center = params[:dim]
    if family == "q_b" and b is not None and np.linalg.norm(np.asarray(center) - singular_center) < 1.0e-6:
        return float(b)
    return float(dim)


def elbo_summary(trace: Dict[str, np.ndarray]) -> Dict[str, float]:
    elbo = np.asarray(trace["elbo"], dtype=np.float64)
    last = elbo[-100:] if elbo.size >= 100 else elbo
    best_idx = int(np.nanargmax(elbo)) if elbo.size else 0
    return {
        "ELBO_initial": float(elbo[0]) if elbo.size else np.nan,
        "ELBO_final": float(elbo[-1]) if elbo.size else np.nan,
        "ELBO_best": float(elbo[best_idx]) if elbo.size else np.nan,
        "ELBO_best_step": float(trace["step"][best_idx]) if elbo.size else np.nan,
        "ELBO_mean_last_100": float(np.nanmean(last)) if last.size else np.nan,
        "ELBO_std_last_100": float(np.nanstd(last)) if last.size else np.nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="EXP3: MNIST binary VI training and bidirectional normalised divergence.")
    add_common_arguments(parser)
    parser.add_argument("--digits", nargs=2, type=int, default=[0, 1])
    parser.add_argument("--dim", type=int, choices=[2, 3, 5], default=2)
    parser.add_argument("--max-n", type=int, default=2500)
    parser.add_argument("--radii", type=int, default=32)
    parser.add_argument("--mnist-dir", default=None)
    parser.add_argument("--vi-steps", type=int, default=10000)
    parser.add_argument("--vi-mc-samples", type=int, default=32)
    parser.add_argument("--num-mc-samples-eval", type=int, default=1024)
    parser.add_argument("--use-best-elbo-checkpoint", action="store_true", default=True)
    parser.add_argument("--learning-rate", type=float, default=0.025)
    parser.add_argument("--grid-points-per-axis", type=int, default=None)
    args = parser.parse_args()

    seed_all(args.seed)
    device_name, device = select_device(prefer_gpu=not args.no_gpu)
    dirs = result_dirs("exp3")
    X, y, task_name, data_source = load_mnist_binary(
        args.digits[0],
        args.digits[1],
        args.dim,
        args.max_n,
        args.seed,
        data_dir=args.mnist_dir,
        allow_synthetic_fallback=args.allow_synthetic_fallback,
    )
    dim = X.shape[1]
    map_est, map_cov = fit_map_and_cov(X, y, "classification", dim, args.seed)
    theta_global, log_w_global = make_global_integration_grid(map_est, map_cov, dim, max_points_per_axis=args.grid_points_per_axis)
    p_spec = DensitySpec("reference_posterior_p", "classification", X, y, np.zeros(dim), float(dim), likelihood_scale=1.0)
    log_p_global = log_model_density_on_device(theta_global, p_spec, device=device)
    log_zp = safe_log_integral_from_grid(log_p_global, log_w_global)
    p_mean, p_cov = weighted_mean_cov(theta_global, log_p_global + log_w_global)
    centers = [np.zeros(dim), p_mean.copy()]
    local_scale = float(np.sqrt(np.max(np.diag(p_cov))))

    vi_specs: List[Tuple[str, str, np.ndarray, Dict[str, np.ndarray], Optional[float]]] = []
    for family in ["q_MF", "q_FR"]:
        params, trace = train_vi_family(
            family,
            X,
            y,
            {"prior_scale": 3.0},
            args.vi_steps,
            args.vi_mc_samples,
            args.learning_rate,
            args.seed + len(vi_specs),
            device,
            p_mean,
            p_cov,
        )
        vi_specs.append((family, family, params, trace, None))
    b_records = []
    for b in [0.5 * dim, float(dim), 1.5 * dim, 2.0 * dim]:
        params, trace = train_vi_family(
            "q_b",
            X,
            y,
            {"prior_scale": 3.0},
            args.vi_steps,
            args.vi_mc_samples,
            args.learning_rate,
            args.seed + int(100 * b),
            device,
            p_mean,
            p_cov,
            b=b,
        )
        candidate_name = f"q_b_b{b:g}"
        vi_specs.append(("q_b", candidate_name, params, trace, b))
        b_records.append({"candidate_name": candidate_name, "b": b, "ELBO_final": float(trace["elbo"][-1]), "ELBO_best": float(np.max(trace["elbo"]))})
    best_b = max([item for item in b_records], key=lambda row: row["ELBO_best"])["candidate_name"]

    arrays = {"global_log_zp": np.asarray(log_zp)}
    traces = {}
    rows = []
    for method, candidate_name, params, trace, b in vi_specs:
        for key, values in trace.items():
            traces[f"{candidate_name}_{key}"] = values
        q_family = "q_b" if method == "q_b" else method
        q_global_log = vi_log_density_on_device(theta_global, q_family, params, dim, b, device=device)
        arrays[f"{candidate_name}_global_log_q"] = q_global_log
        for center in centers:
            ctype = centre_type(center)
            deriv = log_likelihood_derivatives_analytic(center, X, y, "classification")
            radius_diag = choose_adaptive_radii(
                center,
                local_scale,
                deriv["grad_norm"],
                deriv["hessian_spectral_norm"],
                num_radii=args.radii,
                eps_likelihood=args.eps_likelihood,
            )
            radii = radius_diag["radii"]
            arrays[f"{candidate_name}_{ctype}_radii"] = radii
            local_theta, local_log_w, _ = radial_quadrature_grid(center, float(np.max(radii)), dim, 220, 160)
            p_local_log = log_model_density_on_device(local_theta, p_spec, device=device)
            q_local_log = vi_log_density_on_device(local_theta, q_family, params, dim, b, device=device)
            p_log_mass_unnorm, _ = local_log_integrals_radial(p_spec, center, radii, 220, 160, device=device)
            q_log_mass = []
            for r in radii:
                mask = ball_mask(local_theta, center, float(r))
                q_log_mass.append(safe_log_integral_from_grid(q_local_log[mask], local_log_w[mask]) if np.any(mask) else -np.inf)
            q_log_mass_arr = np.asarray(q_log_mass)
            p_expected = float(dim)
            q_expected = expected_q_kappa(q_family, b, center, dim, params)
            p_fit = fit_local_exponent_stable(
                radii,
                p_log_mass_unnorm - log_zp,
                dimension=dim,
                expected_kappa=p_expected,
                expected_kappa_tolerance=args.expected_kappa_tolerance,
                min_points=args.min_fit_points,
            )
            q_fit = fit_local_exponent_stable(
                radii,
                q_log_mass_arr,
                dimension=dim,
                expected_kappa=q_expected,
                expected_kappa_tolerance=args.expected_kappa_tolerance,
                min_points=args.min_fit_points,
            )
            if not radius_diag["radius_selection_reliable"]:
                p_fit["reliable"] = False
                q_fit["reliable"] = False
                p_fit["reason_if_unreliable"] = "radius selection unreliable: " + radius_diag["radius_selection_reason"]
                q_fit["reason_if_unreliable"] = "radius selection unreliable: " + radius_diag["radius_selection_reason"]
            n_p = local_alpha_divergence_log_domain(
                p_local_log,
                q_local_log,
                local_log_w,
                local_theta,
                center,
                radii,
                args.alpha,
                "p",
                log_z_p=log_zp,
                log_z_q=0.0,
            )
            n_q = local_alpha_divergence_log_domain(
                p_local_log,
                q_local_log,
                local_log_w,
                local_theta,
                center,
                radii,
                args.alpha,
                "q",
                log_z_p=log_zp,
                log_z_q=0.0,
            )
            prefix = f"{candidate_name}_{ctype}"
            arrays[f"p_{ctype}_log_mass"] = p_log_mass_unnorm - log_zp
            arrays[f"{prefix}_q_log_mass"] = q_log_mass_arr
            arrays[f"{prefix}_p_mass_for_display_only"] = np.exp(np.clip(p_log_mass_unnorm - log_zp, -50, 50))
            arrays[f"{prefix}_q_mass_for_display_only"] = np.exp(np.clip(q_log_mass_arr, -50, 50))
            arrays[f"{prefix}_log_N_p_q_over_p"] = n_p["log_normalised_divergence"]
            arrays[f"{prefix}_log_N_q_p_over_q"] = n_q["log_normalised_divergence"]
            arrays[f"{prefix}_log_raw_p_q"] = n_p["log_raw_divergence"]
            arrays[f"{prefix}_log_raw_q_p"] = n_q["log_raw_divergence"]
            arrays[f"{prefix}_log_p_ball_mass"] = n_p["log_denominator_mass"]
            arrays[f"{prefix}_log_q_ball_mass"] = n_q["log_denominator_mass"]
            arrays[f"{prefix}_N_p_q_over_p_for_plot"] = n_p["normalised_divergence_for_plot"]
            arrays[f"{prefix}_N_q_p_over_q_for_plot"] = n_q["normalised_divergence_for_plot"]
            arrays[f"{prefix}_reliable_mask_p_scale"] = n_p["reliable_mask"]
            arrays[f"{prefix}_reliable_mask_q_scale"] = n_q["reliable_mask"]
            display_method = "q_b" if candidate_name == best_b else method
            summary = elbo_summary(trace)
            rows.append(
                {
                    "experiment_name": "EXP3",
                    "task": task_name,
                    "data_source": data_source,
                    "feature_dimension": dim,
                    "model_type": "Bayesian logistic regression",
                    "VI_method": display_method,
                    "VI_candidate_name": candidate_name,
                    "VI_training_method": "ELBO optimisation with reparameterised Monte Carlo; torch autograd Adam when available, NumPy finite-difference Adam fallback otherwise",
                    "ELBO_initial": summary["ELBO_initial"],
                    "ELBO_final": summary["ELBO_final"],
                    "ELBO_best": summary["ELBO_best"],
                    "ELBO_best_step": summary["ELBO_best_step"],
                    "ELBO_mean_last_100": summary["ELBO_mean_last_100"],
                    "ELBO_std_last_100": summary["ELBO_std_last_100"],
                    "num_training_steps": args.vi_steps,
                    "num_mc_samples_train": args.vi_mc_samples,
                    "num_mc_samples_eval": args.num_mc_samples_eval,
                    "learning_rate": args.learning_rate,
                    "selected_checkpoint": "best" if args.use_best_elbo_checkpoint else "best",
                    "centre_type": ctype,
                    "centre_value": center_to_string(center),
                    "alpha": args.alpha,
                    "random_seed": args.seed,
                    "device": device_name,
                    "kappa_p": p_fit["estimated_kappa"],
                    "kappa_q": q_fit["estimated_kappa"],
                    "MI_p": p_fit["estimated_MI"],
                    "MI_q": q_fit["estimated_MI"],
                    "kappa_p_reliable": p_fit["reliable"],
                    "kappa_q_reliable": q_fit["reliable"],
                    "kappa_p_expected": p_expected,
                    "kappa_q_expected_if_known": q_expected,
                    "MI_p_expected": float(dim) / float(p_expected),
                    "MI_q_expected_if_known": float(dim) / float(q_expected),
                    "kappa_p_error_if_known": p_fit["kappa_error_against_expected"],
                    "kappa_q_error_if_known": q_fit["kappa_error_against_expected"],
                    "log_N_alpha_r_p_q_over_p_smallest_r": float(n_p["log_normalised_divergence"][0]),
                    "log_N_alpha_r_q_p_over_q_smallest_r": float(n_q["log_normalised_divergence"][0]),
                    "log_N_alpha_r_p_q_over_p_largest_r": float(n_p["log_normalised_divergence"][-1]),
                    "log_N_alpha_r_q_p_over_q_largest_r": float(n_q["log_normalised_divergence"][-1]),
                    "reliable_radius_count_p_scale": int(np.sum(n_p["reliable_mask"])),
                    "reliable_radius_count_q_scale": int(np.sum(n_q["reliable_mask"])),
                    "adaptive_r_min": radius_diag["r_min"],
                    "adaptive_r_max": radius_diag["r_max"],
                    "grad_norm_log_likelihood": radius_diag["grad_norm"],
                    "hessian_scale_log_likelihood": radius_diag["hessian_scale"],
                    "radius_selection_reliable": radius_diag["radius_selection_reliable"],
                    "reason_if_unreliable": q_fit["reason_if_unreliable"],
                    "interpretation": "trained VI posterior local diagnostics; zero is only a finite centre",
                }
            )

    np.savez_compressed(
        dirs["raw"] / "exp3_reference_posterior.npz",
        theta_global=theta_global,
        log_base_weights=log_w_global,
        log_p_unnormalised=log_p_global,
        log_p_normalised=log_p_global - log_zp,
        posterior_mean=p_mean,
        posterior_covariance=p_cov,
        map_estimate=map_est,
        map_covariance=map_cov,
        X=X,
        y=y,
    )
    np.savez_compressed(dirs["raw"] / "exp3_vi_training_traces.npz", **traces)
    np.savez_compressed(dirs["processed"] / "exp3_mnist_arrays.npz", **arrays)
    write_csv(dirs["processed"] / "exp3_vi_family_comparison.csv", rows)
    save_json(
        dirs["processed"] / "exp3_metadata.json",
        {
            "experiment_name": "EXP3",
            "task": task_name,
            "data_source": data_source,
            "feature_dimension": dim,
            "model_type": "Bayesian logistic regression",
            "reference_posterior_method": "dense low-dimensional grid approximation",
            "VI_training_method": "actual ELBO optimisation; no moment-fitted VI labels",
            "q_b_candidates": b_records,
            "q_b_best_candidate": best_b,
            "centre_values": [center_to_string(c) for c in centers],
            "centre_types": [centre_type(c) for c in centers],
            "alpha_value": args.alpha,
            "random_seed": args.seed,
            "device_info": device_name,
            "normalised_divergence_definition": "log_N_p = log_raw - log_p(B_r); log_N_q = log_raw - log_q(B_r)",
            "plot_values": "for display only: exp(clip(log_value, -50, 50))",
            "adaptive_radii": True,
            "eps_likelihood": args.eps_likelihood,
        },
    )


if __name__ == "__main__":
    main()
