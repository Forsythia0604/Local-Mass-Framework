r"""Dataloaders + seeding.

Transforms, normalisation constants and the seeded train/val split mirror FLOPP's
`sims_paper/engine.py` (get_dataloaders, set_seed) verbatim so Tier-A/B results are
directly comparable; reproduced here to avoid importing a module named `engine`
(which would collide with this package's engine.py) and its penalty/oracle baggage.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

_SPLIT_SEED = 42  # FLOPP uses a fixed split seed independent of the run seed


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_dataloaders(dataset="mnist", batch_size=128, val_frac=0.1, data_dir="./data"):
    if dataset in ("mnist", "fashion"):
        tf = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((0.5,), (0.5,))])
        cls = datasets.MNIST if dataset == "mnist" else datasets.FashionMNIST
        full = cls(data_dir, train=True, download=True, transform=tf)
        test = cls(data_dir, train=False, download=True, transform=tf)
    elif dataset == "cifar10":
        mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        tf_tr = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(), transforms.Normalize(mean, std)])
        tf_te = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        full = datasets.CIFAR10(data_dir, train=True, download=True, transform=tf_tr)
        test = datasets.CIFAR10(data_dir, train=False, download=True, transform=tf_te)
    elif dataset == "cifar100":
        mean, std = (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
        tf_tr = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(), transforms.Normalize(mean, std)])
        tf_te = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        full = datasets.CIFAR100(data_dir, train=True, download=True, transform=tf_tr)
        test = datasets.CIFAR100(data_dir, train=False, download=True, transform=tf_te)
    elif dataset == "imagenet":
        # ILSVRC2012 at 224x224; data_dir has train/ and validation/ (Gadi jobfs).
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        tf_tr = transforms.Compose([transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(), transforms.Normalize(mean, std)])
        tf_te = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                                    transforms.ToTensor(), transforms.Normalize(mean, std)])
        full = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=tf_tr)
        test = datasets.ImageFolder(os.path.join(data_dir, "validation"), transform=tf_te)
        n_val = int(len(full) * val_frac)
        n_train = len(full) - n_val
        tr, va = random_split(full, [n_train, n_val],
                              generator=torch.Generator().manual_seed(_SPLIT_SEED))
        train = DataLoader(tr, batch_size=batch_size, shuffle=True, num_workers=12,
                           pin_memory=True, persistent_workers=True)
        val = DataLoader(va, batch_size=256, shuffle=False, num_workers=8,
                         pin_memory=True, persistent_workers=True)
        test = DataLoader(test, batch_size=256, shuffle=False, num_workers=8,
                          pin_memory=True, persistent_workers=True)
        return train, val, test, n_train
    else:
        raise ValueError(f"unknown dataset '{dataset}'")

    n_val = int(len(full) * val_frac)
    n_train = len(full) - n_val
    tr, va = random_split(full, [n_train, n_val],
                          generator=torch.Generator().manual_seed(_SPLIT_SEED))
    train = DataLoader(tr, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val = DataLoader(va, batch_size=512, shuffle=False, num_workers=2, pin_memory=True)
    test = DataLoader(test, batch_size=512, shuffle=False, num_workers=2, pin_memory=True)
    return train, val, test, n_train


IN_CHANNELS = {"mnist": 1, "fashion": 1, "cifar10": 3, "cifar100": 3, "imagenet": 3}
NUM_CLASSES = {"mnist": 10, "fashion": 10, "cifar10": 10, "cifar100": 100, "imagenet": 1000}
