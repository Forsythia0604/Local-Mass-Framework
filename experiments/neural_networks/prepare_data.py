r"""Pre-stage datasets for Gadi (compute nodes have NO internet).

Run this ONCE on a machine WITH internet (e.g. your Mac), then scp the resulting
data dir to Gadi.  The Gadi jobs then load with download already satisfied.

  # on the Mac:
  python prepare_data.py --data_dir ./gadi_data --datasets mnist cifar10 cifar100
  scp -r ./gadi_data sm5828@gadi.nci.org.au:/scratch/nu66/sm5828/bayes/data

The split seed (42) is fixed in data.py, so the seeded train/val split is identical
on Gadi.
"""

from __future__ import annotations

import argparse

from torchvision import datasets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./gadi_data")
    ap.add_argument("--datasets", nargs="+", default=["mnist", "cifar10", "cifar100"])
    args = ap.parse_args()
    for d in args.datasets:
        print(f"downloading {d} -> {args.data_dir}")
        if d == "mnist":
            datasets.MNIST(args.data_dir, train=True, download=True)
            datasets.MNIST(args.data_dir, train=False, download=True)
        elif d == "fashion":
            datasets.FashionMNIST(args.data_dir, train=True, download=True)
            datasets.FashionMNIST(args.data_dir, train=False, download=True)
        elif d == "cifar10":
            datasets.CIFAR10(args.data_dir, train=True, download=True)
            datasets.CIFAR10(args.data_dir, train=False, download=True)
        elif d == "cifar100":
            datasets.CIFAR100(args.data_dir, train=True, download=True)
            datasets.CIFAR100(args.data_dir, train=False, download=True)
        else:
            raise ValueError(f"unknown dataset '{d}'")
    print("done. scp the data_dir to Gadi scratch.")


if __name__ == "__main__":
    main()
