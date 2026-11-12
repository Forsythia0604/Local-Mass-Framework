r"""Structured (channel-gate) marginals for the their-style figures at ResNet scale.

Two sources of trained gates:
  --arch resnet56 : attach gates to the FLOPP pretrained ResNet-56 (frozen base),
                    train the gates on CIFAR-10, extract per-channel gate marginals.
  --arch resnet50 : load a Tier-C gnet checkpoint (ResNet-50/ImageNet, gates already
                    trained on the frozen pretrained base) and extract gate marginals
                    -- no ImageNet data or GPU needed.

The measured object per prunable channel is the gate value z (channel multiplier). An
atom-capable gate places an atom at 0 (channel off) -> MI_pow = inf; a continuous gate
sits near 1 with no atom -> no local mass at 0. Output is an .npz consumed by
local_mass_nn.py --mode trained (gate marginals use the same q-family parameterisation).
"""

from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # e1_forward_budget package
sys.path.insert(0, os.path.join(HERE, "flopp"))    # FLOPP ResNet-56
from distributions.priors import SpikeSlabPrior
from structured.gated_model import GatedNet
from structured.engine_structured import train_elbo
from qfamilies.hardconcrete_l0 import _CONST as HC_CONST
from data import get_dataloaders, set_seed
from run_metadata import write_metadata

OP_KLW = {"gaussian": 1e-3, "horseshoe": 1e-3, "spikeslab": 1e-2, "hardconcrete": 1e-2}


def pick_device(req):
    if req == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if req in ("auto", "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if req == "cuda":
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device("cpu")


def flopp_resnet56(num_classes=10, ckpt=None):
    from resnet56_flopp import ResNet56
    m = ResNet56(num_classes=num_classes)
    if ckpt:
        sd = torch.load(ckpt, map_location="cpu")
        sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
        m.load_state_dict(sd, strict=True)
    return m


@torch.no_grad()
def extract_gate_stats(gnet, family):
    """Concatenate per-channel gate marginal params over all gates."""
    mu, sg, extra = [], [], []
    for k in gnet._order:
        q = gnet.gates[k]
        if family in ("gaussian", "horseshoe"):
            mu.append(q.mu.detach().flatten().cpu())
            sg.append((F.softplus(q.rho) + 1e-8).detach().flatten().cpu())
        elif family == "spikeslab":
            mu.append(q.mu.detach().flatten().cpu())
            sg.append((F.softplus(q.rho) + 1e-8).detach().flatten().cpu())
            extra.append(torch.sigmoid(q.gate_logit).detach().flatten().cpu())
        elif family == "hardconcrete":
            C = q.loga.numel()
            mu.append(torch.ones(C))                                  # gate "on" value ~1
            extra.append(torch.sigmoid(HC_CONST - q.loga).detach().flatten().cpu())  # P(z=0)
    out = {"mu": torch.cat(mu).numpy()}
    if sg:
        out["sg"] = torch.cat(sg).numpy()
    if extra:
        out["extra"] = torch.cat(extra).numpy()
    return out


def train_cifar_gates(family, args, device):
    set_seed(args.seed)
    base = flopp_resnet56(num_classes=10, ckpt=os.path.join(HERE, "flopp", "cifar10_resnet56.pth"))
    gnet = GatedNet(base, family, num_classes=10)
    prior = SpikeSlabPrior(pi=args.pi, slab="gaussian", loc=1.0)
    train_loader, _v, _t, n_train = get_dataloaders("cifar10", args.batch_size, 0.05, args.data_dir)
    targs = SimpleNamespace(epochs=args.epochs, warmup_epochs=0, anneal_epochs=max(1, args.epochs // 2),
                            kl_weight=OP_KLW[family], lr=args.lr, gate_lr=args.gate_lr,
                            weight_decay=5e-4, verbose=args.verbose)
    print(f"[{family}] training gates on CIFAR-10 (frozen base, kl={OP_KLW[family]}) ...", flush=True)
    train_elbo(gnet, prior, train_loader, n_train, device, targs, train_base=False)
    return extract_gate_stats(gnet, family)


def load_imagenet_gates(family, ckpt_path):
    import torchvision
    base = torchvision.models.resnet50(weights=None)
    gnet = GatedNet(base, family, num_classes=1000)
    ck = torch.load(ckpt_path, map_location="cpu")
    gnet.load_state_dict(ck["gnet"] if "gnet" in ck else ck, strict=True)
    print(f"[{family}] loaded gates from {ckpt_path}  ({len(gnet.gates)} gates)", flush=True)
    return extract_gate_stats(gnet, family)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", choices=["resnet56", "resnet50"], default="resnet56")
    ap.add_argument("--families", nargs="+",
                    default=["gaussian", "horseshoe", "spikeslab", "hardconcrete"])
    ap.add_argument("--pi", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--gate_lr", type=float, default=0.05)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--data_dir", default="./data/raw")
    ap.add_argument("--ckpt_dir", default=os.path.join(HERE, "imagenet_ckpts"),
                    help="dir of r50_<family>_p1_s<seed>_gnet.pt (for --arch resnet50)")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    ap.add_argument("--out", default="structured_marginals.npz")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    device = pick_device(args.device)
    saved = {}
    key = {"gaussian": "g", "horseshoe": "h", "spikeslab": "ss", "hardconcrete": "hc"}
    for fam in args.families:
        if args.arch == "resnet56":
            st = train_cifar_gates(fam, args, device)
        else:
            ckpt = os.path.join(args.ckpt_dir, f"r50_{fam}_p1_s{args.seed}_gnet.pt")
            st = load_imagenet_gates(fam, ckpt)
        p = key[fam]
        if fam in ("gaussian", "horseshoe"):
            saved[f"{p}_mu"], saved[f"{p}_sg"] = st["mu"], st["sg"]
            if fam == "horseshoe":
                saved["h_nu"] = np.float64(3.0)
        elif fam == "spikeslab":
            saved["ss_mu"], saved["ss_sg"], saved["ss_gamma"] = st["mu"], st["sg"], st["extra"]
        elif fam == "hardconcrete":
            saved["hc_theta"], saved["hc_p0"] = st["mu"], st["extra"]
            saved["hc_sg"] = np.full_like(st["mu"], 0.15)
        atom = (float(np.mean(1 - st["extra"])) if fam == "spikeslab"
                else float(np.mean(st["extra"])) if fam == "hardconcrete" else 0.0)
        print(f"[{fam}] channels={st['mu'].size}  atom mass={atom:.3f}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, **saved)
    if args.arch == "resnet56":
        write_metadata(args.out, args, mode="gate_training", device_used=str(device),
                       kl_weight_by_family=OP_KLW, split_seed=42,
                       anneal_epochs=max(1, args.epochs // 2),
                       hardconcrete_active_scale="0.15; effective Gaussian approximation")
    else:
        write_metadata(args.out, args, mode="checkpoint_extraction", device_used="cpu",
                       note="Arguments describe extraction, not the checkpoints' original training configuration.",
                       hardconcrete_active_scale="0.15; effective Gaussian approximation")
    print(f"saved structured marginals -> {args.out}  keys={sorted(saved.keys())}")


if __name__ == "__main__":
    main()
