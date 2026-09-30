r"""Part 1 (Gadi): train the four small Bayesian NNs and emit per-coordinate
variational marginal statistics as an .npz for the their-style figure script
(local_mass_nn.py --mode trained).

For each variational family we train a LeNet-5 BNN on MNIST under a spike-and-slab
prior, then extract the per-weight marginal parameters and save them (subsampled):

  Gaussian      -> g_mu,  g_sg
  Horseshoe     -> h_mu,  h_sg,  h_nu
  Spike-and-slab-> ss_mu, ss_sg, ss_gamma
  Hard-concrete -> hc_theta, hc_p0, hc_sg

Run (Gadi):  python train_marginals.py --device cuda \
                 --data_dir /scratch/nu66/sm5828/pruning/data --out marginals.npz
"""

from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import get_dataloaders, set_seed, IN_CHANNELS, NUM_CLASSES
from distributions.priors import SpikeSlabPrior
from layers.bayes_model import build_model
from layers.bayes_layers import bayes_layers
from engine import train_vi
from qfamilies.hardconcrete_l0 import _CONST as HC_CONST
from run_metadata import write_metadata

# per-family KL weight (continuous families and spike-slab calibrated at 1e-3;
# hard-concrete's expected-L0 needs a larger weight to engage its gates)
OP_KLW = {"gaussian": 1e-3, "horseshoe": 1e-3, "spikeslab": 1e-3, "hardconcrete": 0.1}


def pick_device(req):
    if req == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if req in ("auto", "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if req == "cuda":
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device("cpu")


@torch.no_grad()
def extract_stats(model, family, subsample, rng):
    """Concatenate per-coordinate marginal params over all Bayesian weight tensors."""
    mu, sg, extra = [], [], []
    for L in bayes_layers(model):
        q = L.qweight
        if family in ("gaussian", "horseshoe"):
            mu.append(q.mu.detach().flatten().cpu())
            sg.append((F.softplus(q.rho) + 1e-8).detach().flatten().cpu())
        elif family == "spikeslab":
            mu.append(q.mu.detach().flatten().cpu())
            sg.append((F.softplus(q.rho) + 1e-8).detach().flatten().cpu())
            extra.append(torch.sigmoid(q.gate_logit).detach().flatten().cpu())   # gamma
        elif family == "hardconcrete":
            theta = q.theta.detach().flatten().cpu()
            p0 = torch.sigmoid(HC_CONST - q.loga).detach().flatten().cpu()        # P(z=0)
            mu.append(theta); extra.append(p0)
    mu = torch.cat(mu).numpy()
    out = {"mu": mu}
    if sg:
        out["sg"] = torch.cat(sg).numpy()
    if extra:
        out["extra"] = torch.cat(extra).numpy()
    # subsample coordinates for a compact npz
    n = mu.size
    if subsample and subsample < n:
        idx = rng.choice(n, size=subsample, replace=False)
        for k in out:
            out[k] = out[k][idx]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", default="lenet", choices=["lenet", "mlp"])
    ap.add_argument("--dataset", default="mnist")
    ap.add_argument("--pi", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--anneal_epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--subsample", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--data_dir", default="./data/raw")
    ap.add_argument("--device", default="cuda", choices=["auto", "cpu", "mps", "cuda"])
    ap.add_argument("--out", default="marginals.npz")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    device = pick_device(args.device)
    rng = np.random.default_rng(args.seed)
    prior = SpikeSlabPrior(pi=args.pi, slab="gaussian")
    saved = {}

    for family in ["gaussian", "horseshoe", "spikeslab", "hardconcrete"]:
        set_seed(args.seed)
        train_loader, _val, _test, n_train = get_dataloaders(
            args.dataset, args.batch_size, args.val_frac, args.data_dir)
        model = build_model(args.arch, family,
                            num_classes=NUM_CLASSES[args.dataset],
                            in_channels=IN_CHANNELS[args.dataset])
        targs = SimpleNamespace(epochs=args.epochs, anneal_epochs=args.anneal_epochs,
                                kl_weight=OP_KLW[family], lr=args.lr, verbose=args.verbose)
        print(f"[{family}] training (kl_weight={OP_KLW[family]}) ...", flush=True)
        tr = train_vi(model, prior, train_loader, n_train, device, targs)
        st = extract_stats(model, family, args.subsample, rng)
        if family == "gaussian":
            saved["g_mu"], saved["g_sg"] = st["mu"], st["sg"]
        elif family == "horseshoe":
            saved["h_mu"], saved["h_sg"] = st["mu"], st["sg"]
            saved["h_nu"] = np.float64(getattr(model_qnu(model), "nu", 3.0))
        elif family == "spikeslab":
            saved["ss_mu"], saved["ss_sg"], saved["ss_gamma"] = st["mu"], st["sg"], st["extra"]
        elif family == "hardconcrete":
            theta, p0 = st["mu"], st["extra"]
            saved["hc_theta"], saved["hc_p0"] = theta, p0
            saved["hc_sg"] = np.maximum(0.02, 0.5 * np.abs(theta))   # active-part scale
        print(f"[{family}] done: nll={tr['final_nll']:.4f}  coords saved={st['mu'].size}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, **saved)
    write_metadata(args.out, args, mode="weight_training", device_used=str(device),
                   kl_weight_by_family=OP_KLW, split_seed=42,
                   hardconcrete_active_scale="max(0.02, 0.5 * abs(theta)); effective Gaussian approximation")
    print(f"saved marginals -> {args.out}  keys={sorted(saved.keys())}")


def model_qnu(model):
    """Return a HorseshoeQ instance from the model (for nu)."""
    for L in bayes_layers(model):
        return L.qweight
    return SimpleNamespace(nu=3.0)


if __name__ == "__main__":
    main()
