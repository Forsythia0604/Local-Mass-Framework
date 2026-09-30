r"""Bayesian models for Tier A: LeNet-5-Caffe (MNIST) and a tiny MLP.

LeNet-5-Caffe matches the architecture used by Jantre et al. (thesis l.1503, Table T2),
the empirical anchor for the prior-MI / sparsity ranking.  Every weight tensor is a
variational q-family of the chosen kind; the whole net is trained by ELBO.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bayes_layers import BayesConv2d, BayesLinear


class BayesLeNet5(nn.Module):
    """LeNet-5-Caffe: conv(20,5) - pool - conv(50,5) - pool - fc(800->500) - fc(500->10)."""

    def __init__(self, qfamily, num_classes=10, in_channels=1, q_kw=None):
        super().__init__()
        self.conv1 = BayesConv2d(in_channels, 20, 5, qfamily, q_kw=q_kw)
        self.conv2 = BayesConv2d(20, 50, 5, qfamily, q_kw=q_kw)
        # MNIST 28x28 -> conv1 24x24 -> pool 12x12 -> conv2 8x8 -> pool 4x4 -> 50*4*4=800
        self.fc1 = BayesLinear(50 * 4 * 4, 500, qfamily, q_kw=q_kw)
        self.fc2 = BayesLinear(500, num_classes, qfamily, q_kw=q_kw)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class BayesMLP(nn.Module):
    """Tiny MLP for fast smoke tests (MNIST flattened)."""

    def __init__(self, qfamily, in_dim=784, hidden=300, num_classes=10, q_kw=None):
        super().__init__()
        self.fc1 = BayesLinear(in_dim, hidden, qfamily, q_kw=q_kw)
        self.fc2 = BayesLinear(hidden, num_classes, qfamily, q_kw=q_kw)

    def forward(self, x):
        x = x.flatten(1)
        return self.fc2(F.relu(self.fc1(x)))


def build_model(arch, qfamily, num_classes=10, in_channels=1, q_kw=None):
    if arch == "lenet":
        return BayesLeNet5(qfamily, num_classes=num_classes, in_channels=in_channels, q_kw=q_kw)
    if arch == "mlp":
        return BayesMLP(qfamily, num_classes=num_classes, q_kw=q_kw)
    raise ValueError(f"unknown arch '{arch}' for Tier A (use 'lenet' or 'mlp')")
