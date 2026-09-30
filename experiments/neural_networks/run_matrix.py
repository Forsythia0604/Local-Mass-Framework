r"""Tier-A run matrix driver: launches run_e1 over (family, seed, kl_weight) and writes
one JSON per run into an output dir.  Two blocks:

  * dichotomy : 4 families x seeds at each family's operating-point kl_weight
                -> the headline finite/inf-budget + MI table.
  * frontier  : spike-slab (and optionally hard-concrete) over a kl_weight sweep x seeds
                -> the sparsity-accuracy curve.

Sequential (MPS is single-GPU); ~2 min/run at 15 epochs.  Run:
  python run_matrix.py --out results/tierA --seeds 1 2 3 --epochs 15
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

# operating-point kl_weight per family (calibrated on LeNet/MNIST so acc ~ 99%)
OP_KLW = {"gaussian": 1e-3, "horseshoe": 1e-3, "spikeslab": 1e-3, "hardconcrete": 0.5}
FRONTIER_KLW = {
    "spikeslab": [3e-4, 1e-3, 3e-3, 1e-2],
    "hardconcrete": [0.1, 0.5, 1.0, 2.0],
}
PY = sys.executable


def run_one(out_dir, arch, qfamily, slab, kl_weight, seed, epochs, anneal, data_dir, device, tag):
    rf = os.path.join(out_dir, f"{tag}.json")
    if os.path.exists(rf):
        print(f"skip (exists): {tag}")
        return
    cmd = [PY, "run_e1.py", "--arch", arch, "--qfamily", qfamily, "--slab", slab,
           "--kl_weight", str(kl_weight), "--seed", str(seed), "--epochs", str(epochs),
           "--anneal_epochs", str(anneal), "--device", device, "--data_dir", data_dir,
           "--results_file", rf]
    print(f">>> {tag}")
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/tierA")
    ap.add_argument("--arch", default="lenet")
    ap.add_argument("--slab", default="gaussian")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--anneal", type=int, default=7)
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--block", default="all", choices=["all", "dichotomy", "frontier"])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.block in ("all", "dichotomy"):
        for q in ["gaussian", "spikeslab", "hardconcrete", "horseshoe"]:
            for s in args.seeds:
                run_one(args.out, args.arch, q, args.slab, OP_KLW[q], s,
                        args.epochs, args.anneal, args.data_dir, args.device,
                        tag=f"dich_{q}_s{s}")

    if args.block in ("all", "frontier"):
        for q, klws in FRONTIER_KLW.items():
            for kw in klws:
                for s in args.seeds:
                    run_one(args.out, args.arch, q, args.slab, kw, s,
                            args.epochs, args.anneal, args.data_dir, args.device,
                            tag=f"front_{q}_kw{kw}_s{s}")

    print("matrix complete.")


if __name__ == "__main__":
    main()
