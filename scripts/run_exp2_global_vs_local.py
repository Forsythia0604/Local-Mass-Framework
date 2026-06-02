import argparse
import json
import math
import platform
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import torch
import yaml
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
    cuda_available = torch.cuda.is_available()
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "pyyaml": yaml.__version__,
        "torch": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_device_name": torch.cuda.get_device_name(0) if cuda_available else None,
    }


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(config):
    requested = str(config.get("device", "auto"))
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def r_grid_from_config(config):
    spec = config["r_grid"]
    return np.logspace(spec["start_exp"], spec["stop_exp"], int(spec["num"]))


def torch_log_normal_diag(x, mean, log_scale):
    dim = x.shape[-1]
    z = (x - mean) / torch.exp(log_scale)
    return -0.5 * dim * math.log(2.0 * math.pi) - torch.sum(log_scale, dim=-1) - 0.5 * torch.sum(z * z, dim=-1)


def torch_log_target(theta, eps, tau, c):
    dim = theta.shape[-1]
    mean_main = torch.full((dim,), float(c), device=theta.device, dtype=theta.dtype)
    log_main = math.log1p(-float(eps)) + torch_log_normal_diag(
        theta,
        mean_main,
        torch.zeros(dim, device=theta.device, dtype=theta.dtype),
    )
    log_spike = math.log(float(eps)) + torch_log_normal_diag(
        theta,
        torch.zeros(dim, device=theta.device, dtype=theta.dtype),
        torch.full((dim,), math.log(float(tau)), device=theta.device, dtype=theta.dtype),
    )
    return torch.logsumexp(torch.stack([log_main, log_spike], dim=0), dim=0)


class MeanFieldGaussian(torch.nn.Module):
    def __init__(self, dim, init_mean=0.8):
        super().__init__()
        self.loc = torch.nn.Parameter(torch.full((dim,), float(init_mean)))
        self.log_scale = torch.nn.Parameter(torch.zeros(dim))

    def sample_and_log_prob(self, n):
        eps = torch.randn(n, self.loc.numel(), device=self.loc.device)
        sample = self.loc + torch.exp(self.log_scale) * eps
        return sample, self.log_prob(sample)

    def log_prob(self, x):
        return torch_log_normal_diag(x, self.loc, self.log_scale)

    def parameter_snapshot(self):
        return {
            "loc": self.loc.detach().cpu().numpy().tolist(),
            "scale": torch.exp(self.log_scale).detach().cpu().numpy().tolist(),
        }


class DiagonalGaussianMixture2(torch.nn.Module):
    def __init__(self, dim, c):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(2))
        self.locs = torch.nn.Parameter(torch.stack([torch.full((dim,), float(c)), torch.zeros(dim)], dim=0))
        self.log_scales = torch.nn.Parameter(torch.stack([torch.zeros(dim), torch.full((dim,), -2.5)], dim=0))

    def sample_and_log_prob(self, n):
        weights = torch.softmax(self.logits, dim=0)
        component = torch.distributions.Categorical(probs=weights).sample((n,))
        eps = torch.randn(n, self.locs.shape[1], device=self.locs.device)
        loc = self.locs[component]
        scale = torch.exp(self.log_scales[component])
        sample = loc + scale * eps
        return sample, self.log_prob(sample)

    def log_prob(self, x):
        per_component = []
        log_weights = torch.log_softmax(self.logits, dim=0)
        for k in range(2):
            per_component.append(log_weights[k] + torch_log_normal_diag(x, self.locs[k], self.log_scales[k]))
        return torch.logsumexp(torch.stack(per_component, dim=0), dim=0)

    def parameter_snapshot(self):
        return {
            "weights": torch.softmax(self.logits, dim=0).detach().cpu().numpy().tolist(),
            "locs": self.locs.detach().cpu().numpy().tolist(),
            "scales": torch.exp(self.log_scales).detach().cpu().numpy().tolist(),
        }


def build_family(name, dim, c, device):
    if name == "mean_field_gaussian":
        return MeanFieldGaussian(dim, init_mean=c).to(device)
    if name == "diagonal_gaussian_mixture_2":
        return DiagonalGaussianMixture2(dim, c).to(device)
    raise ValueError(f"Unknown variational family: {name}")


def train_variational(config, seed, dim, eps, tau, family_name, device):
    set_seed(int(seed))
    model = build_family(family_name, int(dim), float(config["c"]), device)
    opt = torch.optim.Adam(model.parameters(), lr=float(config["training"]["learning_rate"]))
    logs = []
    num_steps = int(config["training"]["num_steps"])
    batch_size = int(config["training"]["batch_size"])
    log_every = int(config["training"]["log_every"])
    start = time.time()
    final_loss = np.nan
    status = "ok"

    try:
        for step in range(1, num_steps + 1):
            opt.zero_grad(set_to_none=True)
            sample, log_q = model.sample_and_log_prob(batch_size)
            log_p = torch_log_target(sample, eps, tau, config["c"])
            loss = torch.mean(log_q - log_p)
            loss.backward()
            opt.step()
            final_loss = float(loss.detach().cpu())

            if step % log_every == 0 or step == 1 or step == num_steps:
                logs.append(
                    {
                        "seed": seed,
                        "dim": dim,
                        "eps": eps,
                        "tau": tau,
                        "variational_family": family_name,
                        "training_step": step,
                        "training_loss": final_loss,
                        "runtime_seconds": time.time() - start,
                    }
                )
    except Exception as exc:
        status = f"training_failure:{repr(exc)}"

    return model, logs, final_loss, status


def sample_target_numpy(n, dim, eps, tau, c, rng):
    component = rng.random(n) < float(eps)
    samples = rng.normal(float(c), 1.0, size=(n, dim)).astype(np.float32)
    spike_count = int(component.sum())
    if spike_count > 0:
        samples[component] = rng.normal(0.0, float(tau), size=(spike_count, dim)).astype(np.float32)
    return samples


def log_target_numpy(samples, eps, tau, c):
    dim = samples.shape[1]
    log_main = (
        math.log1p(-float(eps))
        - 0.5 * dim * math.log(2.0 * math.pi)
        - 0.5 * np.sum((samples - float(c)) ** 2, axis=1)
    )
    log_spike = (
        math.log(float(eps))
        - 0.5 * dim * math.log(2.0 * math.pi)
        - dim * math.log(float(tau))
        - 0.5 * np.sum((samples / float(tau)) ** 2, axis=1)
    )
    stacked = np.stack([log_main, log_spike], axis=0)
    max_value = np.max(stacked, axis=0)
    return max_value + np.log(np.sum(np.exp(stacked - max_value), axis=0))


def model_log_prob_numpy(model, samples, device, chunk_size):
    values = []
    with torch.no_grad():
        for start in range(0, len(samples), chunk_size):
            chunk = torch.as_tensor(samples[start : start + chunk_size], dtype=torch.float32, device=device)
            values.append(model.log_prob(chunk).detach().cpu().numpy())
    return np.concatenate(values)


def model_sample_numpy(model, n, device, chunk_size):
    samples = []
    log_q = []
    with torch.no_grad():
        for start in range(0, n, chunk_size):
            size = min(chunk_size, n - start)
            chunk, chunk_log_q = model.sample_and_log_prob(size)
            samples.append(chunk.detach().cpu().numpy().astype(np.float32))
            log_q.append(chunk_log_q.detach().cpu().numpy())
    return np.concatenate(samples, axis=0), np.concatenate(log_q)


def f_alpha(x, alpha):
    return (np.power(x, alpha) - alpha * x + (alpha - 1.0)) / (alpha - 1.0)


def counts_by_radius(norms, r_values):
    return np.array([int(np.sum(norms < r)) for r in r_values], dtype=np.int64)


def evaluate_model(config, model, seed, dim, eps, tau, family_name, device):
    eval_samples = int(config["evaluation"]["eval_samples"])
    chunk_size = int(config["evaluation"].get("chunk_size", 50000))
    r_values = r_grid_from_config(config)
    alpha = float(config["alpha"])
    rng = np.random.default_rng(int(seed) + 9173)

    q_samples, q_log_q = model_sample_numpy(model, eval_samples, device, chunk_size)
    q_log_p = log_target_numpy(q_samples, eps, tau, config["c"])
    global_kl = float(np.mean(q_log_q - q_log_p))

    p_samples = sample_target_numpy(eval_samples, int(dim), eps, tau, config["c"], rng)
    p_log_p = log_target_numpy(p_samples, eps, tau, config["c"])
    p_log_q = model_log_prob_numpy(model, p_samples, device, chunk_size)

    p_norms = np.linalg.norm(p_samples, axis=1)
    q_norms = np.linalg.norm(q_samples, axis=1)
    p_counts = counts_by_radius(p_norms, r_values)
    q_counts = counts_by_radius(q_norms, r_values)

    ratio_q_over_p = np.exp(p_log_q - p_log_p)
    local_integrand = f_alpha(ratio_q_over_p, alpha)

    rows = []
    for r, p_count, q_count in zip(r_values, p_counts, q_counts):
        inside = p_norms < r
        p_mass = p_count / eval_samples
        q_mass = q_count / eval_samples
        local_rekl = float(np.mean(local_integrand[inside])) * p_mass if p_count > 0 else 0.0
        rows.append(
            {
                "seed": seed,
                "dim": dim,
                "eps": eps,
                "tau": tau,
                "variational_family": family_name,
                "training_step": int(config["training"]["num_steps"]),
                "final_global_kl_estimate": global_kl,
                "r": r,
                "p_mass": p_mass,
                "q_mass": q_mass,
                "local_mass_ratio": q_mass / p_mass if p_mass > 0 else np.nan,
                "local_additive_error": abs(q_mass - p_mass),
                "local_rekl_q_p": local_rekl,
                "normalised_local_rekl_q_p": local_rekl / p_mass if p_mass > 0 else np.nan,
                "num_eval_samples": eval_samples,
                "num_p_samples_inside_ball": int(p_count),
                "num_q_samples_inside_ball": int(q_count),
            }
        )
    return rows, global_kl


def run_experiment(config, run_dir):
    device = get_device(config)
    raw_rows = []
    training_rows = []
    summary_rows = []
    failures = []

    total = (
        len(config["seeds"])
        * len(config["dims"])
        * len(config["eps_values"])
        * len(config["tau_values"])
        * len(config["variational_families"])
    )
    with tqdm(total=total, desc="Exp2 configurations") as progress:
        for seed in config["seeds"]:
            for dim in config["dims"]:
                for eps in config["eps_values"]:
                    for tau in config["tau_values"]:
                        for family_name in config["variational_families"]:
                            start = time.time()
                            model = None
                            status = "ok"
                            final_loss = np.nan
                            try:
                                model, logs, final_loss, status = train_variational(
                                    config, seed, dim, eps, tau, family_name, device
                                )
                                training_rows.extend(logs)
                                if not status.startswith("ok"):
                                    raise RuntimeError(status)
                                rows, global_kl = evaluate_model(
                                    config, model, seed, dim, eps, tau, family_name, device
                                )
                                raw_rows.extend(rows)
                                summary_rows.append(
                                    {
                                        "seed": seed,
                                        "dim": dim,
                                        "eps": eps,
                                        "tau": tau,
                                        "variational_family": family_name,
                                        "final_training_loss": final_loss,
                                        "final_global_kl_estimate": global_kl,
                                        "min_local_mass_ratio": float(np.nanmin([row["local_mass_ratio"] for row in rows])),
                                        "median_local_mass_ratio": float(np.nanmedian([row["local_mass_ratio"] for row in rows])),
                                        "max_normalised_local_rekl_q_p": float(
                                            np.nanmax([row["normalised_local_rekl_q_p"] for row in rows])
                                        ),
                                        "runtime_seconds_config": time.time() - start,
                                        "training_status": status,
                                        "model_parameters": json.dumps(model.parameter_snapshot()) if model is not None else "{}",
                                    }
                                )
                            except Exception as exc:
                                failures.append(
                                    {
                                        "seed": seed,
                                        "dim": dim,
                                        "eps": eps,
                                        "tau": tau,
                                        "variational_family": family_name,
                                        "failure_message": repr(exc),
                                        "training_status": status,
                                    }
                                )
                            progress.update(1)

    raw = pd.DataFrame(raw_rows)
    training_log = pd.DataFrame(training_rows)
    summary = pd.DataFrame(summary_rows)
    failure_df = pd.DataFrame(failures)
    raw.to_csv(run_dir / "raw_metrics.csv", index=False)
    training_log.to_csv(run_dir / "training_log.csv", index=False)
    summary.to_csv(run_dir / "summary_metrics.csv", index=False)
    failure_df.to_csv(run_dir / "logs" / "failures.csv", index=False)
    return raw, training_log, summary, failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "exp2_global_vs_local.yaml"))
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
        "device_used": str(get_device(config)),
    }

    start = time.time()
    raw, training_log, summary, failures = run_experiment(config, run_dir)
    metadata["end_time"] = datetime.now().isoformat()
    metadata["runtime_seconds"] = time.time() - start
    metadata["num_raw_rows"] = int(len(raw))
    metadata["num_training_log_rows"] = int(len(training_log))
    metadata["num_summary_rows"] = int(len(summary))
    metadata["num_failures"] = int(len(failures))
    with open(run_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"Saved Exp2 run to {run_dir}")


if __name__ == "__main__":
    main()
