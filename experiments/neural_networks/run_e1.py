r"""E1 runner (Tier A): train a Bayesian net by ELBO for one (arch, q-family, slab, seed)
and write the JSON-ready measurement (MI, budgets, sparsity).

Example:
  python run_e1.py --arch lenet --qfamily spikeslab --slab gaussian \
                   --epochs 15 --kl_weight 1.0 --seed 1 \
                   --results_file results/lenet_spikeslab_s1.json
"""

from __future__ import annotations

import argparse
import json
import os

import torch

from distributions.priors import SpikeSlabPrior
from layers.bayes_model import build_model
from data import IN_CHANNELS, NUM_CLASSES
from engine import run_e1_single


def pick_device(req):
    if req == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if req in ("auto", "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if req == "cuda":
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device("cpu")


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", default="lenet", choices=["lenet", "mlp"])
    p.add_argument("--dataset", default="mnist")
    p.add_argument("--qfamily", default="spikeslab",
                   choices=["gaussian", "spikeslab", "hardconcrete", "horseshoe"])
    p.add_argument("--slab", default="gaussian",
                   choices=["gaussian", "laplace", "studentt", "horseshoe"])
    p.add_argument("--pi", type=float, default=0.5, help="prior slab weight; atom mass = 1-pi")
    p.add_argument("--kl_weight", type=float, default=1.0)
    p.add_argument("--anneal_epochs", type=int, default=5)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--threshold", type=float, default=1e-3, help="below-eps sparsity threshold")
    p.add_argument("--mi_samples", type=int, default=4, help="MC weight draws for the MI estimator")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--results_file", default="results/e1_run.json")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main():
    args = build_args()
    device = pick_device(args.device)
    prior = SpikeSlabPrior(pi=args.pi, slab=args.slab)
    model = build_model(args.arch, args.qfamily,
                        num_classes=NUM_CLASSES[args.dataset],
                        in_channels=IN_CHANNELS[args.dataset])
    print(f"Run: arch={args.arch} q={args.qfamily} slab={args.slab} pi={args.pi} "
          f"seed={args.seed} device={device}")
    result = run_e1_single(model, prior, args, device)

    os.makedirs(os.path.dirname(os.path.abspath(args.results_file)), exist_ok=True)
    with open(args.results_file, "w") as f:
        json.dump(result, f, indent=2)

    m = result["mass_index"]; s = result["sparsity"]; b = result["budget"]
    print(f"  -> acc={result['eval']['test_acc']:.2f}%  nll={result['eval']['test_nll']:.4f}")
    print(f"  -> MI1={m['mi1_hat']}  atom={m['atom_detected']}  "
          f"exact_zero={s['exact_zero_frac']:.3f}  below_eps={s['below_thresh_frac']:.3f}")
    gf = "inf" if b["gamma_forward_is_inf"] else f"{b['gamma_forward']:.3f}"
    print(f"  -> Gamma(0) forward = {gf}   [{result['predicted_class']}]")
    print(f"  saved {args.results_file}")


if __name__ == "__main__":
    main()
