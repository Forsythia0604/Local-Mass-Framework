from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

EPS = 1.0e-12


def import_torch() -> Any:
    try:
        import torch  # type: ignore

        return torch
    except Exception:
        return None


def select_device(prefer_gpu: bool = True) -> Tuple[str, Any]:
    torch = import_torch()
    if torch is None:
        return "numpy-cpu", None
    if prefer_gpu and torch.cuda.is_available():
        return "cuda", torch.device("cuda")
    if prefer_gpu and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", torch.device("mps")
    return "cpu", torch.device("cpu")


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def result_dirs(exp_name: str) -> Dict[str, pathlib.Path]:
    base = repo_root() / "results" / exp_name
    dirs = {
        "base": base,
        "raw": base / "raw",
        "processed": base / "processed",
        "figures": base / "figures",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def seed_all(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch = import_torch()
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    return np.random.default_rng(seed)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def save_json(path: str | pathlib.Path, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, indent=2, sort_keys=True)


def write_csv(path: str | pathlib.Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: to_jsonable(row.get(k, "")) for k in keys})


def logsumexp(x: np.ndarray, axis: Optional[int] = None, keepdims: bool = False) -> np.ndarray:
    try:
        from scipy.special import logsumexp as scipy_logsumexp  # type: ignore

        return scipy_logsumexp(x, axis=axis, keepdims=keepdims)
    except Exception:
        arr = np.asarray(x, dtype=np.float64)
        if axis is None:
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                return np.asarray(-np.inf)
            m = np.max(finite)
            return np.asarray(m + np.log(np.sum(np.exp(finite - m))))
        masked = np.where(np.isfinite(arr), arr, -np.inf)
        m = np.max(masked, axis=axis, keepdims=True)
        bad = ~np.isfinite(m)
        total = m + np.log(np.sum(np.exp(masked - m), axis=axis, keepdims=True))
        total = np.where(bad, -np.inf, total)
        return total if keepdims else np.squeeze(total, axis=axis)


def logdiffexp(a: float | np.ndarray, b: float | np.ndarray) -> np.ndarray:
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    if np.any(b_arr > a_arr + 1.0e-12):
        raise ValueError("logdiffexp requires a >= b.")
    return a_arr + np.log1p(-np.exp(np.minimum(b_arr - a_arr, 0.0)))


def safe_log_integral_from_grid(log_values: np.ndarray, log_weights: np.ndarray) -> float:
    return float(logsumexp(np.asarray(log_values, dtype=np.float64) + np.asarray(log_weights, dtype=np.float64)))


def add_common_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--adaptive-radii", action="store_true", default=True)
    parser.add_argument("--eps-likelihood", type=float, default=0.05)
    parser.add_argument("--expected-kappa-tolerance", type=float, default=0.15)
    parser.add_argument("--min-fit-points", type=int, default=5)
    parser.add_argument("--prefer-small-radii", action="store_true", default=True)
    parser.add_argument("--allow-synthetic-fallback", action="store_true")
    return parser


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))


def standardise(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    return (X - mu) / np.where(sd < EPS, 1.0, sd)


def low_dimensionalise(X: np.ndarray, dim: int) -> np.ndarray:
    Xs = standardise(X)
    if Xs.shape[1] <= dim:
        return Xs
    _, _, vt = np.linalg.svd(Xs, full_matrices=False)
    return Xs @ vt[:dim].T


def synthetic_dataset(name: str, dim: int, seed: int) -> Tuple[np.ndarray, np.ndarray, str, str, str]:
    rng = np.random.default_rng(abs(hash((name, dim, seed))) % (2**32))
    n = {"breast_cancer": 420, "wine": 140, "diabetes": 360}.get(name, 260)
    X = rng.normal(size=(n, max(dim, 6)))
    beta = np.linspace(1.25, -0.7, max(dim, 6))
    eta = X @ beta
    if name == "diabetes":
        y = eta + 0.7 * rng.normal(size=n)
        y = (y - y.mean()) / (y.std() + EPS)
        return low_dimensionalise(X, dim), y, "Bayesian linear regression", "regression", "synthetic_fallback"
    p = sigmoid(eta)
    y = rng.binomial(1, p).astype(np.float64)
    return low_dimensionalise(X, dim), y, "Bayesian logistic regression", "classification", "synthetic_fallback"


def load_sklearn_uci_dataset(name: str, dim: int, seed: int, allow_synthetic_fallback: bool = False) -> Tuple[np.ndarray, np.ndarray, str, str, str]:
    try:
        from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine  # type: ignore
    except Exception as exc:
        if allow_synthetic_fallback:
            return synthetic_dataset(name, dim, seed)
        raise RuntimeError("scikit-learn is required for EXP1/EXP2 real UCI datasets.") from exc
    if name == "breast_cancer":
        b = load_breast_cancer()
        return low_dimensionalise(b.data, dim), b.target.astype(np.float64), "Bayesian logistic regression", "classification", "sklearn_uci"
    if name == "wine":
        b = load_wine()
        mask = b.target != 2
        return low_dimensionalise(b.data[mask], dim), b.target[mask].astype(np.float64), "Bayesian logistic regression", "classification", "sklearn_uci"
    if name == "diabetes":
        b = load_diabetes()
        y = b.target.astype(np.float64)
        y = (y - y.mean()) / (y.std() + EPS)
        return low_dimensionalise(b.data, dim), y, "Bayesian linear regression", "regression", "sklearn_uci"
    raise ValueError(f"Unknown dataset: {name}")


def train_subset(X: np.ndarray, y: np.ndarray, max_n: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if X.shape[0] <= max_n:
        return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], size=max_n, replace=False)
    return X[idx], y[idx]


def load_mnist_binary(digit_a: int, digit_b: int, dim: int, max_n: int, seed: int, data_dir: Optional[str] = None, allow_synthetic_fallback: bool = False) -> Tuple[np.ndarray, np.ndarray, str, str]:
    root = pathlib.Path(data_dir) if data_dir else repo_root() / "data" / "mnist"
    npz = root / "mnist.npz"
    X: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None
    source = "unknown"
    if npz.exists():
        z = np.load(npz)
        if "x_train" in z and "y_train" in z:
            X = z["x_train"].reshape(z["x_train"].shape[0], -1).astype(np.float64) / 255.0
            y = z["y_train"].astype(int)
            source = "mnist_npz"
        elif "X" in z and "y" in z:
            X = z["X"].reshape(z["X"].shape[0], -1).astype(np.float64)
            y = z["y"].astype(int)
            source = "mnist_npz"
    if X is None or y is None:
        try:
            from torchvision import datasets  # type: ignore

            ds = datasets.MNIST(root=str(root), train=True, download=True)
            X = ds.data.numpy().reshape(len(ds), -1).astype(np.float64) / 255.0
            y = ds.targets.numpy().astype(int)
            source = "mnist_torchvision"
        except Exception:
            if not allow_synthetic_fallback:
                raise RuntimeError("Provide MNIST at data/mnist/mnist.npz or install torchvision. Use --allow-synthetic-fallback only for smoke tests.")
            rng = np.random.default_rng(seed)
            n = max(max_n, 600)
            X = rng.normal(size=(n, 64))
            labels = rng.binomial(1, 0.5, size=n)
            X[:, :16] += labels[:, None] * 1.4
            y = np.where(labels == 0, digit_a, digit_b)
            source = "synthetic_fallback"
    mask = (y == digit_a) | (y == digit_b)
    X2, y2 = train_subset(X[mask], (y[mask] == digit_b).astype(np.float64), max_n=max_n, seed=seed)
    return low_dimensionalise(X2, dim), y2, f"MNIST {digit_a} vs {digit_b}", source


def center_to_string(center: Sequence[float]) -> str:
    return json.dumps([round(float(v), 8) for v in center])


def centre_type(center: Sequence[float]) -> str:
    return "zero" if np.linalg.norm(np.asarray(center, dtype=np.float64)) < 1.0e-10 else "nonzero"


def ball_mask(theta: np.ndarray, center: np.ndarray, radius: float | np.ndarray) -> np.ndarray:
    dist = np.linalg.norm(np.atleast_2d(theta) - np.asarray(center).reshape(1, -1), axis=-1)
    r = np.asarray(radius)
    if r.ndim == 0:
        return dist <= float(r)
    return dist[:, None] <= r.reshape(1, -1)


def sphere_area(d: int) -> float:
    return 2.0 * math.pi ** (d / 2.0) / math.gamma(d / 2.0)


def radial_directions(d: int, n_dirs: int) -> np.ndarray:
    if d == 1:
        return np.array([[-1.0], [1.0]])
    if d == 2:
        t = np.linspace(0.0, 2.0 * math.pi, n_dirs, endpoint=False)
        return np.column_stack([np.cos(t), np.sin(t)])
    if d == 3:
        k = np.arange(n_dirs)
        z = 1.0 - 2.0 * (k + 0.5) / n_dirs
        phi = math.pi * (3.0 - math.sqrt(5.0)) * k
        rr = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        return np.column_stack([rr * np.cos(phi), rr * np.sin(phi), z])
    rng = np.random.default_rng(1234)
    z = rng.normal(size=(n_dirs, d))
    return z / np.clip(np.linalg.norm(z, axis=1, keepdims=True), EPS, None)


def radial_quadrature_grid(center: np.ndarray, r_max: float, d: int, n_radial: int = 180, n_dirs: int = 128) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    dirs = radial_directions(d, n_dirs)
    n_dirs_actual = dirs.shape[0]
    r_min = max(float(r_max) * 1.0e-6, 1.0e-10)
    rho = np.geomspace(r_min, float(r_max), n_radial)
    dr = np.empty_like(rho)
    dr[0] = rho[1] - rho[0]
    dr[-1] = rho[-1] - rho[-2]
    dr[1:-1] = 0.5 * (rho[2:] - rho[:-2])
    points = center.reshape(1, 1, d) + rho.reshape(-1, 1, 1) * dirs.reshape(1, n_dirs_actual, d)
    log_w = math.log(sphere_area(d) / n_dirs_actual) + np.log(dr).reshape(-1, 1) + (d - 1) * np.log(rho).reshape(-1, 1)
    return points.reshape(-1, d), np.broadcast_to(log_w, (n_radial, n_dirs_actual)).reshape(-1), np.repeat(rho, n_dirs_actual)


def tensor_grid(center: np.ndarray, half_width: float, d: int, points_per_axis: int) -> Tuple[np.ndarray, np.ndarray]:
    axes = [np.linspace(center[i] - half_width, center[i] + half_width, points_per_axis) for i in range(d)]
    mesh = np.meshgrid(*axes, indexing="ij")
    theta = np.column_stack([m.ravel() for m in mesh])
    dx = (2.0 * half_width) / max(points_per_axis - 1, 1)
    return theta, np.full(theta.shape[0], d * math.log(dx))


def make_global_integration_grid(center: np.ndarray, cov: np.ndarray, d: int, multiplier: float = 6.0, max_points_per_axis: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    scale = float(np.sqrt(np.max(np.diag(cov))))
    half_width = max(multiplier * scale, 1.0)
    default = {1: 2000, 2: 260, 3: 72, 5: 18}.get(d, 28)
    ppa = max_points_per_axis or default
    return tensor_grid(center, half_width, d, ppa)


def log_likelihood(theta: np.ndarray, X: np.ndarray, y: np.ndarray, task: str, noise_scale: float = 1.0) -> np.ndarray:
    theta = np.atleast_2d(theta).astype(np.float64)
    eta = X @ theta.T
    if task == "classification":
        yy = y.reshape(-1, 1)
        return np.sum(yy * eta - np.logaddexp(0.0, eta), axis=0)
    resid = y.reshape(-1, 1) - eta
    return -0.5 * np.sum(resid * resid, axis=0) / (noise_scale * noise_scale)


def log_likelihood_derivatives_analytic(center: np.ndarray, X: np.ndarray, y: np.ndarray, task: str, noise_scale: float = 1.0) -> Dict[str, Any]:
    c = np.asarray(center, dtype=np.float64)
    eta = X @ c
    if task == "classification":
        p = sigmoid(eta)
        grad = X.T @ (y - p)
        W = p * (1.0 - p)
        hess = -(X.T @ (X * W[:, None]))
    else:
        resid = y - X @ c
        grad = X.T @ resid / (noise_scale * noise_scale)
        hess = -(X.T @ X) / (noise_scale * noise_scale)
    spectral = float(np.linalg.norm(hess, ord=2))
    return {
        "log_likelihood_at_center": float(log_likelihood(c.reshape(1, -1), X, y, task, noise_scale)[0]),
        "grad_norm": float(np.linalg.norm(grad)),
        "hessian_operator_norm": spectral,
        "hessian_spectral_norm": spectral,
        "hessian_frobenius_norm": float(np.linalg.norm(hess, ord="fro")),
        "derivative_reliable": True,
        "derivative_failure_reason": "",
    }


def local_likelihood_derivatives(log_likelihood_fn: Callable[[np.ndarray], np.ndarray], center: np.ndarray, *, device: Any = None, method: str = "autograd") -> Dict[str, Any]:
    torch = import_torch()
    if method == "autograd" and torch is not None:
        try:
            c = torch.tensor(np.asarray(center, dtype=np.float64), dtype=torch.float64, device=device, requires_grad=True)
            val = log_likelihood_fn(c.reshape(1, -1))
            if not torch.is_tensor(val):
                raise TypeError("log_likelihood_fn did not return a torch tensor")
            scalar = val.reshape(() if val.numel() == 1 else (-1,))[0]
            grad = torch.autograd.grad(scalar, c, create_graph=True)[0]
            rows = []
            for g in grad:
                rows.append(torch.autograd.grad(g, c, retain_graph=True)[0])
            h = torch.stack(rows)
            return {
                "log_likelihood_at_center": float(scalar.detach().cpu()),
                "grad_norm": float(torch.linalg.norm(grad).detach().cpu()),
                "hessian_operator_norm": float(torch.linalg.matrix_norm(h, ord=2).detach().cpu()),
                "hessian_spectral_norm": float(torch.linalg.matrix_norm(h, ord=2).detach().cpu()),
                "hessian_frobenius_norm": float(torch.linalg.matrix_norm(h, ord="fro").detach().cpu()),
                "derivative_reliable": True,
                "derivative_failure_reason": "",
            }
        except Exception as exc:
            return {
                "log_likelihood_at_center": np.nan,
                "grad_norm": np.nan,
                "hessian_operator_norm": np.nan,
                "hessian_spectral_norm": np.nan,
                "hessian_frobenius_norm": np.nan,
                "derivative_reliable": False,
                "derivative_failure_reason": str(exc),
            }
    raise RuntimeError("Use log_likelihood_derivatives_analytic for NumPy model likelihoods.")


def choose_adaptive_radii(
    center: np.ndarray,
    local_scale: float,
    grad_norm: float,
    hessian_scale: float,
    *,
    num_radii: int = 32,
    eps_likelihood: float = 0.05,
    min_radius_factor: float = 1.0e-5,
    max_radius_factor: float = 0.5,
    absolute_r_min: Optional[float] = None,
    absolute_r_max: Optional[float] = None,
) -> Dict[str, Any]:
    scale = max(float(local_scale), 1.0e-8)
    cap = max_radius_factor * scale
    if absolute_r_max is not None:
        cap = min(cap, float(absolute_r_max))
    g = float(grad_norm) if np.isfinite(grad_norm) else np.inf
    h = float(hessian_scale) if np.isfinite(hessian_scale) else 0.0
    reliable = True
    reason = "local-likelihood derivative criterion satisfied"
    if np.isfinite(g) and np.isfinite(h) and h > 0:
        r_deriv = (-g + math.sqrt(max(g * g + 2.0 * h * eps_likelihood, 0.0))) / h
    elif np.isfinite(g) and g > 0:
        r_deriv = eps_likelihood / g
        reason = "first-order derivative criterion used"
    elif np.isfinite(g):
        r_deriv = cap
        reason = "zero-gradient derivative criterion limited by local scale"
    else:
        r_deriv = min(cap, scale * 0.05)
        reliable = False
        reason = "derivative diagnostics unavailable; conservative radius cap used"
    r_max = min(cap, max(float(r_deriv), 0.0))
    if absolute_r_min is not None:
        r_min = float(absolute_r_min)
    else:
        r_min = max(min_radius_factor * scale, 1.0e-10)
    if r_max <= r_min * 1.05:
        reliable = False
        reason = "adaptive r_max is below numerical radius resolution"
        r_max = max(r_min * 1.05, cap * 0.1)
    radii = np.geomspace(r_min, r_max, num_radii)
    return {
        "radii": radii,
        "r_min": float(r_min),
        "r_max": float(r_max),
        "local_scale": float(scale),
        "eps_likelihood": float(eps_likelihood),
        "grad_norm": float(g) if np.isfinite(g) else np.nan,
        "hessian_scale": float(h) if np.isfinite(h) else np.nan,
        "radius_selection_reliable": bool(reliable),
        "radius_selection_reason": reason,
    }


@dataclass
class DensitySpec:
    name: str
    model_task: str
    X: np.ndarray
    y: np.ndarray
    center: np.ndarray
    power_index: float
    likelihood_scale: float = 1.0
    prior_scale: float = 3.0
    noise_scale: float = 1.0
    local_bump: float = 0.0
    local_bump_width: float = 0.25
    bump_center: Optional[np.ndarray] = None
    gaussian_mean: Optional[np.ndarray] = None
    gaussian_cov: Optional[np.ndarray] = None
    gaussian_diag_only: bool = False
    power_family_center: Optional[np.ndarray] = None
    power_family_scale: Optional[np.ndarray] = None


def log_shifted_power_prior(theta: np.ndarray, center: np.ndarray, a: float, scale: float = 3.0) -> np.ndarray:
    theta = np.atleast_2d(theta)
    d = theta.shape[1]
    radius = np.linalg.norm(theta - center.reshape(1, -1), axis=1)
    return (a - d) * np.log(radius + EPS) - 0.5 * np.sum((theta / scale) ** 2, axis=1)


def log_gaussian(theta: np.ndarray, mean: np.ndarray, cov: np.ndarray, diag_only: bool = False) -> np.ndarray:
    theta = np.atleast_2d(theta)
    d = theta.shape[1]
    diff = theta - mean.reshape(1, -1)
    if diag_only:
        var = np.clip(np.diag(cov), 1.0e-8, None)
        return -0.5 * (d * math.log(2.0 * math.pi) + np.sum(np.log(var)) + np.sum(diff * diff / var, axis=1))
    cov2 = cov + 1.0e-7 * np.eye(d)
    sign, logdet = np.linalg.slogdet(cov2)
    if sign <= 0:
        cov2 = cov2 + 1.0e-5 * np.eye(d)
        sign, logdet = np.linalg.slogdet(cov2)
    inv = np.linalg.inv(cov2)
    q = np.einsum("ni,ij,nj->n", diff, inv, diff)
    return -0.5 * (d * math.log(2.0 * math.pi) + logdet + q)


def log_power_gaussian_family(theta: np.ndarray, center: np.ndarray, b: float, mean: np.ndarray, scale_diag: np.ndarray) -> np.ndarray:
    theta = np.atleast_2d(theta)
    d = theta.shape[1]
    radius = np.linalg.norm(theta - center.reshape(1, -1), axis=1)
    diff = (theta - mean.reshape(1, -1)) / scale_diag.reshape(1, -1)
    log_norm_radial = math.log(sphere_area(d)) + (b / 2.0 - 1.0) * math.log(2.0) + math.lgamma(b / 2.0)
    return (b - d) * np.log(radius + EPS) - 0.5 * np.sum(diff * diff, axis=1) - np.sum(np.log(scale_diag)) - log_norm_radial


def log_model_density(theta: np.ndarray, spec: DensitySpec) -> np.ndarray:
    if spec.gaussian_mean is not None and spec.gaussian_cov is not None:
        out = log_gaussian(theta, spec.gaussian_mean, spec.gaussian_cov, spec.gaussian_diag_only)
    elif spec.power_family_center is not None and spec.power_family_scale is not None:
        out = log_power_gaussian_family(theta, spec.power_family_center, spec.power_index, spec.center, spec.power_family_scale)
    else:
        out = log_shifted_power_prior(theta, spec.center, spec.power_index, spec.prior_scale)
        if spec.likelihood_scale != 0.0:
            out = out + spec.likelihood_scale * log_likelihood(theta, spec.X, spec.y, spec.model_task, spec.noise_scale)
    if spec.local_bump != 0.0:
        bcenter = spec.center if spec.bump_center is None else spec.bump_center
        dist2 = np.sum((np.atleast_2d(theta) - bcenter.reshape(1, -1)) ** 2, axis=1)
        out = out + np.log1p(spec.local_bump * np.exp(-0.5 * dist2 / (spec.local_bump_width**2)))
    return out


def _torch_log_likelihood(theta: Any, X: Any, y: Any, task: str, noise_scale: float = 1.0) -> Any:
    torch = import_torch()
    eta = X @ theta.T
    if task == "classification":
        yy = y.reshape(-1, 1)
        return torch.sum(yy * eta - torch.logaddexp(torch.zeros_like(eta), eta), dim=0)
    resid = y.reshape(-1, 1) - eta
    return -0.5 * torch.sum(resid * resid, dim=0) / (noise_scale * noise_scale)


def log_model_density_on_device(theta: np.ndarray, spec: DensitySpec, device: Any = None, batch_size: int = 200_000) -> np.ndarray:
    torch = import_torch()
    if torch is None or device is None:
        return log_model_density(theta, spec)
    dtype = torch.float64
    outs: List[np.ndarray] = []
    with torch.no_grad():
        X_t = torch.as_tensor(spec.X, dtype=dtype, device=device)
        y_t = torch.as_tensor(spec.y, dtype=dtype, device=device)
        center_t = torch.as_tensor(spec.center, dtype=dtype, device=device).reshape(1, -1)
        for start in range(0, theta.shape[0], batch_size):
            th = torch.as_tensor(theta[start : start + batch_size], dtype=dtype, device=device)
            if spec.gaussian_mean is not None and spec.gaussian_cov is not None:
                mean = torch.as_tensor(spec.gaussian_mean, dtype=dtype, device=device).reshape(1, -1)
                cov = torch.as_tensor(spec.gaussian_cov + 1.0e-7 * np.eye(theta.shape[1]), dtype=dtype, device=device)
                L = torch.linalg.cholesky(cov)
                z = torch.linalg.solve_triangular(L, (th - mean).T, upper=False).T
                logdet = 2.0 * torch.sum(torch.log(torch.diag(L)))
                out = -0.5 * (theta.shape[1] * math.log(2.0 * math.pi) + logdet + torch.sum(z * z, dim=1))
            else:
                radius = torch.linalg.norm(th - center_t, dim=1)
                out = (spec.power_index - theta.shape[1]) * torch.log(radius + EPS) - 0.5 * torch.sum((th / spec.prior_scale) ** 2, dim=1)
                if spec.likelihood_scale != 0.0:
                    out = out + spec.likelihood_scale * _torch_log_likelihood(th, X_t, y_t, spec.model_task, spec.noise_scale)
            if spec.local_bump != 0.0:
                bcenter = spec.center if spec.bump_center is None else spec.bump_center
                bc = torch.as_tensor(bcenter, dtype=dtype, device=device).reshape(1, -1)
                dist2 = torch.sum((th - bc) ** 2, dim=1)
                out = out + torch.log1p(spec.local_bump * torch.exp(-0.5 * dist2 / (spec.local_bump_width**2)))
            outs.append(out.detach().cpu().numpy())
    return np.concatenate(outs)


def fit_map_and_cov(X: np.ndarray, y: np.ndarray, task: str, dim: int, seed: int, prior_scale: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    w = rng.normal(scale=0.02, size=dim)
    if task == "classification":
        for _ in range(900):
            p = sigmoid(X @ w)
            grad_neg_log_post = X.T @ (p - y) + w / (prior_scale * prior_scale)
            w -= 0.2 * grad_neg_log_post / max(1, X.shape[0])
        p = sigmoid(X @ w)
        W = p * (1.0 - p)
        hess = X.T @ (X * W[:, None]) + np.eye(dim) / (prior_scale * prior_scale)
    else:
        hess = X.T @ X + np.eye(dim) / (prior_scale * prior_scale)
        w = np.linalg.solve(hess, X.T @ y)
    cov = np.linalg.inv(hess + 1.0e-6 * np.eye(dim))
    return w, cov


def local_log_integrals_radial(spec: DensitySpec, center: np.ndarray, radii: np.ndarray, n_radial: int = 180, n_dirs: int = 128, device: Any = None) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    d = int(center.size)
    theta, log_w, rho = radial_quadrature_grid(center, float(np.max(radii)), d, n_radial, n_dirs)
    lv = log_model_density_on_device(theta, spec, device=device)
    vals = []
    for r in radii:
        mask = rho <= float(r)
        vals.append(safe_log_integral_from_grid(lv[mask], log_w[mask]) if np.any(mask) else -np.inf)
    return np.asarray(vals), {"theta": theta, "log_weights": log_w, "rho": rho, "log_values": lv}


def _linear_fit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    coef = np.polyfit(x, y, 1)
    pred = coef[0] * x + coef[1]
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, EPS)
    return float(coef[0]), float(coef[1]), float(r2)


def _fit_kappa_with_adaptive_window(
    radii: Sequence[float],
    log_masses: Sequence[float],
    *,
    expected_kappa: Optional[float] = None,
    expected_kappa_tolerance: float = 0.15,
    min_points: int = 5,
    prefer_small_radii: bool = True,
    max_local_slope_std: float = 0.20,
    max_curvature: float = 0.25,
) -> Dict[str, Any]:
    r = np.asarray(radii, dtype=np.float64)
    lm = np.asarray(log_masses, dtype=np.float64)
    finite = (r > 0) & np.isfinite(lm)
    idx = np.where(finite)[0]
    base = {
        "kappa": np.nan,
        "intercept": np.nan,
        "r2": np.nan,
        "selected_indices": idx,
        "local_slopes": np.asarray([], dtype=float),
        "local_slope_std": np.nan,
        "curvature_score": np.nan,
        "num_fit_points": int(idx.size),
        "fit_radius_min": np.nan,
        "fit_radius_max": np.nan,
        "expected_kappa": expected_kappa,
        "expected_kappa_tolerance": expected_kappa_tolerance,
        "kappa_error_against_expected": np.nan,
        "reliable": False,
        "reason_if_unreliable": "",
    }
    if idx.size < min_points:
        base["reason_if_unreliable"] = "fewer than min_points finite non-saturated masses"
        return base
    x_all = np.log(r)
    candidates: List[Dict[str, Any]] = []
    for start_pos in range(0, idx.size - min_points + 1):
        max_end = idx.size + 1
        for end_pos in range(start_pos + min_points, max_end):
            sel = idx[start_pos:end_pos]
            if sel[-1] - sel[0] + 1 != sel.size:
                continue
            x = x_all[sel]
            y = lm[sel]
            slope, intercept, r2 = _linear_fit(x, y)
            local_slopes = np.diff(y) / np.diff(x)
            slope_std = float(np.std(local_slopes)) if local_slopes.size else np.inf
            if local_slopes.size >= 2:
                curvature = float(np.max(np.abs(np.diff(local_slopes))))
            else:
                curvature = np.inf
            kappa_error = abs(slope - expected_kappa) if expected_kappa is not None and np.isfinite(slope) else np.nan
            stable = slope_std <= max_local_slope_std and curvature <= max_curvature
            expected_ok = True if expected_kappa is None else kappa_error <= expected_kappa_tolerance
            reliable = bool(stable and expected_ok and r2 >= 0.95)
            score = (0 if reliable else -1000) - 0.001 * start_pos if prefer_small_radii else 0.001 * sel.size
            score += r2 - 0.05 * slope_std - 0.03 * curvature
            candidates.append(
                {
                    "score": score,
                    "kappa": slope,
                    "intercept": intercept,
                    "r2": r2,
                    "selected_indices": sel,
                    "local_slopes": local_slopes,
                    "local_slope_std": slope_std,
                    "curvature_score": curvature,
                    "kappa_error_against_expected": kappa_error,
                    "reliable": reliable,
                }
            )
    if not candidates:
        base["reason_if_unreliable"] = "no contiguous finite radius window"
        return base
    best = sorted(candidates, key=lambda v: v["score"], reverse=True)[0]
    sel = best["selected_indices"]
    reason = ""
    if not best["reliable"]:
        parts = []
        if best["local_slope_std"] > max_local_slope_std:
            parts.append(f"local_slope_std={best['local_slope_std']:.4g} exceeds {max_local_slope_std}")
        if best["curvature_score"] > max_curvature:
            parts.append(f"curvature_score={best['curvature_score']:.4g} exceeds {max_curvature}")
        if expected_kappa is not None and best["kappa_error_against_expected"] > expected_kappa_tolerance:
            parts.append(f"kappa error {best['kappa_error_against_expected']:.4g} exceeds tolerance {expected_kappa_tolerance}")
        if best["r2"] < 0.95:
            parts.append(f"r2={best['r2']:.4g} below 0.95")
        reason = "; ".join(parts) or "best window failed reliability checks"
    best.update(
        {
            "num_fit_points": int(sel.size),
            "fit_radius_min": float(r[sel[0]]),
            "fit_radius_max": float(r[sel[-1]]),
            "expected_kappa": expected_kappa,
            "expected_kappa_tolerance": expected_kappa_tolerance,
            "reason_if_unreliable": reason,
        }
    )
    return best


def fit_local_exponent_stable(
    radii: Sequence[float],
    log_masses: Sequence[float],
    *,
    dimension: int,
    expected_kappa: Optional[float] = None,
    expected_kappa_tolerance: float = 0.15,
    min_points: int = 5,
    prefer_small_radii: bool = True,
    max_local_slope_std: float = 0.20,
    max_curvature: float = 0.25,
) -> Dict[str, Any]:
    """
    Fit the local mass exponent kappa from log mass against log radius.

    The fitted slope is kappa. Mass Index is only computed afterwards as
    dimension / kappa. Reliability checks apply to kappa, not to MI.
    """
    fit = _fit_kappa_with_adaptive_window(
        radii,
        log_masses,
        expected_kappa=expected_kappa,
        expected_kappa_tolerance=expected_kappa_tolerance,
        min_points=min_points,
        prefer_small_radii=prefer_small_radii,
        max_local_slope_std=max_local_slope_std,
        max_curvature=max_curvature,
    )
    kappa = float(fit.get("kappa", np.nan))
    reason = str(fit.get("reason_if_unreliable", ""))
    reliable = bool(fit.get("reliable", False))
    if not np.isfinite(kappa) or kappa <= 0.0:
        reliable = False
        reason = (reason + "; " if reason else "") + "estimated_kappa is non-positive or non-finite"
        estimated_mi = np.nan
    else:
        estimated_mi = float(dimension) / kappa
    expected_mi = np.nan if expected_kappa is None or expected_kappa <= 0 else float(dimension) / float(expected_kappa)
    return {
        "estimated_kappa": kappa,
        "estimated_MI": estimated_mi,
        "intercept": fit.get("intercept", np.nan),
        "r2": fit.get("r2", np.nan),
        "selected_indices": np.asarray(fit.get("selected_indices", []), dtype=int),
        "local_slopes": np.asarray(fit.get("local_slopes", []), dtype=np.float64),
        "local_slope_std": fit.get("local_slope_std", np.nan),
        "curvature_score": fit.get("curvature_score", np.nan),
        "reliable": reliable,
        "reason_if_unreliable": reason,
        "expected_kappa": expected_kappa,
        "expected_MI": expected_mi,
        "expected_kappa_tolerance": expected_kappa_tolerance,
        "kappa_error_against_expected": fit.get("kappa_error_against_expected", np.nan),
        "num_fit_points": fit.get("num_fit_points", 0),
        "fit_radius_min": fit.get("fit_radius_min", np.nan),
        "fit_radius_max": fit.get("fit_radius_max", np.nan),
    }


def alpha_f_from_log_ratio(log_ratio: np.ndarray, alpha: float) -> np.ndarray:
    lr = np.asarray(log_ratio, dtype=np.float64)
    t = np.exp(np.clip(lr, -80.0, 80.0))
    if abs(alpha - 1.0) < 1.0e-8:
        return np.maximum(t * lr - t + 1.0, EPS)
    if abs(alpha) < 1.0e-8:
        return np.maximum(-lr + t - 1.0, EPS)
    return np.maximum((np.exp(np.clip(alpha * lr, -80.0, 80.0)) - alpha * (t - 1.0) - 1.0) / (alpha * (alpha - 1.0)), EPS)


def local_alpha_divergence_log_domain(
    log_p_unnorm: np.ndarray,
    log_q_unnorm: np.ndarray,
    log_base_weights: np.ndarray,
    theta: np.ndarray,
    center: np.ndarray,
    radii: Sequence[float],
    alpha: float,
    normalise_by: str = "p",
    log_z_p: Optional[float] = None,
    log_z_q: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    log_zp = safe_log_integral_from_grid(log_p_unnorm, log_base_weights) if log_z_p is None else float(log_z_p)
    log_zq = safe_log_integral_from_grid(log_q_unnorm, log_base_weights) if log_z_q is None else float(log_z_q)
    lp = np.asarray(log_p_unnorm) - log_zp
    lq = np.asarray(log_q_unnorm) - log_zq
    if normalise_by == "q":
        base = lq
        log_ratio = lp - lq
    elif normalise_by == "p":
        base = lp
        log_ratio = lq - lp
    else:
        raise ValueError("normalise_by must be 'p' or 'q'.")
    log_raw: List[float] = []
    log_den: List[float] = []
    log_norm: List[float] = []
    reliable: List[bool] = []
    conditional: List[float] = []
    for r in radii:
        mask = ball_mask(theta, center, float(r))
        if not np.any(mask):
            log_raw.append(-np.inf)
            log_den.append(-np.inf)
            log_norm.append(-np.inf)
            conditional.append(np.nan)
            reliable.append(False)
            continue
        ld = safe_log_integral_from_grid(base[mask], log_base_weights[mask])
        f_val = alpha_f_from_log_ratio(log_ratio[mask], alpha)
        lr = safe_log_integral_from_grid(base[mask] + np.log(f_val), log_base_weights[mask])
        ln = lr - ld if np.isfinite(lr) and np.isfinite(ld) else -np.inf
        weights = np.exp(base[mask] + log_base_weights[mask] - ld) if np.isfinite(ld) else np.full(np.sum(mask), np.nan)
        conditional.append(float(np.sum(weights * f_val)) if np.all(np.isfinite(weights)) else np.nan)
        log_raw.append(lr)
        log_den.append(ld)
        log_norm.append(ln)
        reliable.append(bool(np.isfinite(ln) and ld > -745.0))
    log_raw_arr = np.asarray(log_raw)
    log_den_arr = np.asarray(log_den)
    log_norm_arr = np.asarray(log_norm)
    return {
        "log_raw_divergence": log_raw_arr,
        "log_denominator_mass": log_den_arr,
        "log_normalised_divergence": log_norm_arr,
        "raw_divergence_for_plot": np.exp(np.clip(log_raw_arr, -50.0, 50.0)),
        "denominator_mass_for_plot": np.exp(np.clip(log_den_arr, -50.0, 50.0)),
        "normalised_divergence_for_plot": np.exp(np.clip(log_norm_arr, -50.0, 50.0)),
        "log_normalised_divergence_from_ratio": log_norm_arr,
        "normalised_divergence_from_conditional_expectation": np.asarray(conditional),
        "reliable_mask": np.asarray(reliable, dtype=bool),
    }


def trend_from_log_values(log_values: np.ndarray) -> str:
    vals = np.asarray(log_values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size < 3:
        return "insufficient reliable radii"
    first = float(np.mean(vals[: max(1, vals.size // 4)]))
    last = float(np.mean(vals[-max(1, vals.size // 4) :]))
    if first < last - 0.35:
        return "decays as r shrinks"
    if first > last + 0.35:
        return "increases as r shrinks"
    return "approximately stable"


def save_simple_png_figure(
    png_path: str | pathlib.Path,
    panels: Sequence[Dict[str, Any]],
    title: str,
    size: Tuple[int, int] = (1200, 820),
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 24)
        small = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
        small = ImageFont.load_default()
    draw.text((24, 16), title, fill=(20, 20, 20), font=font)
    cols = max(1, len(panels))
    margin = 58
    top = 70
    panel_w = (size[0] - 2 * margin) // cols
    panel_h = size[1] - top - 78
    colours = [(47, 111, 159), (196, 154, 44), (157, 63, 84), (40, 40, 40), (189, 93, 56)]
    for i, panel in enumerate(panels):
        x0 = margin + i * panel_w + 14
        y0 = top
        x1 = x0 + panel_w - 30
        y1 = y0 + panel_h
        draw.rectangle((x0, y0, x1, y1), outline=(70, 70, 70), width=1)
        draw.text((x0, y0 - 22), str(panel.get("title", "")), fill=(20, 20, 20), font=small)
        series = panel.get("series", [])
        xs_all: List[float] = []
        ys_all: List[float] = []
        for s in series:
            xs = np.asarray(s.get("x", []), dtype=float)
            ys = np.asarray(s.get("y", []), dtype=float)
            mask = np.isfinite(xs) & np.isfinite(ys)
            if panel.get("xlog", True):
                mask &= xs > 0
            if panel.get("ylog", False):
                mask &= ys > 0
            xs_all.extend(xs[mask].tolist())
            ys_all.extend(ys[mask].tolist())
        if len(xs_all) < 2 or len(ys_all) < 2:
            continue
        tx = np.log10(np.asarray(xs_all)) if panel.get("xlog", True) else np.asarray(xs_all)
        ty = np.log10(np.asarray(ys_all)) if panel.get("ylog", False) else np.asarray(ys_all)
        xmin, xmax = float(np.min(tx)), float(np.max(tx))
        ymin, ymax = float(np.min(ty)), float(np.max(ty))
        if abs(xmax - xmin) < EPS:
            xmax += 1.0
        if abs(ymax - ymin) < EPS:
            ymax += 1.0
        for k, s in enumerate(series):
            xs = np.asarray(s.get("x", []), dtype=float)
            ys = np.asarray(s.get("y", []), dtype=float)
            mask = np.isfinite(xs) & np.isfinite(ys)
            if panel.get("xlog", True):
                mask &= xs > 0
            if panel.get("ylog", False):
                mask &= ys > 0
            if np.sum(mask) < 1:
                continue
            xx = np.log10(xs[mask]) if panel.get("xlog", True) else xs[mask]
            yy = np.log10(ys[mask]) if panel.get("ylog", False) else ys[mask]
            pts = [
                (
                    int(x0 + 10 + (vx - xmin) / (xmax - xmin) * (x1 - x0 - 20)),
                    int(y1 - 10 - (vy - ymin) / (ymax - ymin) * (y1 - y0 - 20)),
                )
                for vx, vy in zip(xx, yy)
            ]
            if len(pts) == 1:
                px, py = pts[0]
                draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=colours[k % len(colours)])
            else:
                draw.line(pts, fill=colours[k % len(colours)], width=3)
            draw.text((x0 + 12, y0 + 12 + 20 * k), str(s.get("label", "")), fill=colours[k % len(colours)], font=small)
        draw.text((x0, y1 + 16), str(panel.get("xlabel", "radius r")), fill=(20, 20, 20), font=small)
        draw.text((x0, y0 + 10), str(panel.get("ylabel", "")), fill=(20, 20, 20), font=small)
    Image.Image.save(img, png_path)


def weighted_mean_cov(theta: np.ndarray, log_weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    lw = log_weights - float(logsumexp(log_weights))
    w = np.exp(lw)
    mean = np.sum(theta * w.reshape(-1, 1), axis=0)
    diff = theta - mean.reshape(1, -1)
    cov = diff.T @ (diff * w.reshape(-1, 1)) + 1.0e-6 * np.eye(theta.shape[1])
    return mean, cov


def parse_json_center(text: str) -> List[float]:
    try:
        return [float(v) for v in json.loads(text)]
    except Exception:
        return []
