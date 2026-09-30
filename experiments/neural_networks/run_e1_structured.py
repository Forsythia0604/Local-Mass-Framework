r"""Structured group-VI E1 runner (Tier B/C).

Tier B (single phase):
  python run_e1_structured.py --arch resnet56 --dataset cifar10 --qfamily spikeslab \
        --epochs 120 --kl_weight 1e-3 --seed 1 --results_file results/cifar_ss_s1.json

Tier C ImageNet:
  # phase 1 (learn gates on frozen pretrained base):
  python run_e1_structured.py --arch resnet50 --dataset imagenet --phase 1 \
        --qfamily spikeslab --epochs 8 --kl_weight 1e-3 --pretrained \
        --data_dir $DATA_DIR --results_file .../r50_ss_p1.json
  # phase 2 (KD fine-tune, resumable):
  python run_e1_structured.py --arch resnet50 --dataset imagenet --phase 2 \
        --qfamily spikeslab --pretrained --data_dir $DATA_DIR \
        --p1_ckpt .../r50_ss_p1_gnet.pt --resume --results_file .../r50_ss_p2.json
"""

from __future__ import annotations

import argparse
import json
import os

import torch

from distributions.priors import SpikeSlabPrior
from data import NUM_CLASSES
from qfamilies import predicted_class
from structured.gated_model import build_gated
from structured.engine_structured import train_elbo, eval_structured, kd_finetune


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
    p.add_argument("--arch", default="resnet56", choices=["resnet56", "resnet50"])
    p.add_argument("--dataset", default="cifar10")
    p.add_argument("--qfamily", default="spikeslab",
                   choices=["gaussian", "spikeslab", "hardconcrete", "horseshoe"])
    p.add_argument("--slab", default="gaussian")
    p.add_argument("--slab_loc", type=float, default=1.0,
                   help="slab centre for channel gates (1.0 = gate 'on'); sparsity comes from the atom")
    p.add_argument("--pi", type=float, default=0.5)
    p.add_argument("--phase", type=int, default=0, help="0=single (Tier B); 1/2 = Tier C phases")
    p.add_argument("--pretrained", action="store_true", help="load ImageNet-pretrained base (Tier C)")
    p.add_argument("--kl_weight", type=float, default=1e-3)
    p.add_argument("--warmup_epochs", type=int, default=20,
                   help="epochs training the base with gates frozen-on before beta ramps (Tier B)")
    p.add_argument("--anneal_epochs", type=int, default=60, help="epochs over which beta ramps 0->kl_weight")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--gate_lr", type=float, default=0.05)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--val_frac", type=float, default=0.05)
    p.add_argument("--threshold", type=float, default=1e-3)
    p.add_argument("--mi_samples", type=int, default=8)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--device", default="cuda", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--results_file", default="results/e1_structured.json")
    p.add_argument("--verbose", action="store_true")
    # phase-2 KD
    p.add_argument("--p1_ckpt", default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--ft_lr", type=float, default=0.01)
    p.add_argument("--ft_epochs", type=int, default=90)
    p.add_argument("--kd_T", type=float, default=4.0)
    p.add_argument("--kd_alpha", type=float, default=0.1)
    p.add_argument("--stop_at_epoch", type=int, default=0)
    return p.parse_args()


def main():
    args = build_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.results_file)), exist_ok=True)
    from data import get_dataloaders, set_seed
    device = pick_device(args.device)
    set_seed(args.seed)
    slab_kw = {"loc": args.slab_loc} if args.slab == "gaussian" else {}
    prior = SpikeSlabPrior(pi=args.pi, slab=args.slab, **slab_kw)
    nc = NUM_CLASSES[args.dataset]
    gnet = build_gated(args.arch, args.qfamily, num_classes=nc, pretrained=args.pretrained)
    print(f"Structured run: arch={args.arch} ds={args.dataset} q={args.qfamily} "
          f"phase={args.phase} gates={len(gnet.gates)} device={device}")

    train_loader, val_loader, test_loader, n_train = get_dataloaders(
        args.dataset, args.batch_size, args.val_frac, args.data_dir)

    ckpt = args.results_file.replace(".json", "_gnet.pt")

    if args.phase == 2:
        # KD fine-tune from a phase-1 gate checkpoint
        import torchvision
        teacher = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        if args.p1_ckpt and os.path.exists(args.p1_ckpt):
            gnet.load_state_dict(torch.load(args.p1_ckpt, map_location=device)["gnet"])
            print(f"loaded phase-1 gates from {args.p1_ckpt}")
        ev = kd_finetune(gnet, teacher, train_loader, test_loader, device, args, ckpt)
        result = {"meta": vars(args), "eval": ev, "phase": 2}
    else:
        # Tier B single-phase, or Tier C phase 1 (freeze pretrained base, train gates)
        train_base = not (args.phase == 1 and args.pretrained)
        tr = train_elbo(gnet, prior, train_loader, n_train, device, args, train_base=train_base)
        ev = eval_structured(gnet, test_loader, device)
        mi, budget, sparsity = gnet.measure(prior, mi_samples=args.mi_samples, threshold=args.threshold)
        torch.save({"gnet": gnet.state_dict()}, ckpt)
        result = {
            "meta": {**vars(args), "arch": args.arch, "dataset": args.dataset, "qfamily": args.qfamily,
                     "slab": args.slab, "pi": args.pi, "kl_weight": args.kl_weight,
                     "epochs": args.epochs, "seed": args.seed, "phase": args.phase,
                     "vi_scope": "structured", "device": str(device)},
            "train": tr, "eval": ev, "sparsity": sparsity,
            "mass_index": mi, "budget": budget,
            "predicted_class": predicted_class(args.qfamily),
        }

    with open(args.results_file, "w") as f:
        json.dump(result, f, indent=2, default=str)
    if "sparsity" in result:
        m, s, b = result["mass_index"], result["sparsity"], result["budget"]
        gf = "inf" if b["gamma_forward_is_inf"] else f"{b['gamma_forward']:.3f}"
        print(f"  -> acc={result['eval']['test_acc']:.2f}%  "
              f"MI1={m['mi1_hat']} atom={m['atom_detected']} "
              f"struct_sparsity={s['structural_sparsity']:.3f}  Gamma(0) = {gf}")
    print(f"  saved {args.results_file}")


if __name__ == "__main__":
    main()
