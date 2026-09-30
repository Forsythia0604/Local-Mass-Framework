r"""Bayesian Linear / Conv2d layers: each weight tensor is a variational q-family.

Weight-space sampling (mean-field) is used uniformly across all four families so the
atom families (spike-slab, hard-concrete) compose the same way as the continuous ones;
this is adequate at LeNet/MNIST scale (local reparameterization is a later optimisation
and does not apply to the gate-based families).

Biases are kept as ordinary point parameters (the sparsity question concerns weights);
they carry no KL term.  Each layer exposes `.kl(prior)` summing the weight q-family KL,
and `.qweight` so the measurement code can reach the variational family.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qfamilies import make_qfamily


class BayesLinear(nn.Module):
    def __init__(self, in_features, out_features, qfamily, bias=True, q_kw=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.qweight = make_qfamily(qfamily, (out_features, in_features), **(q_kw or {}))
        self.qweight.deterministic = False
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x):
        w = self.qweight.deterministic_weight() if self.qweight.deterministic else self.qweight.rsample()
        return F.linear(x, w, self.bias)

    def kl(self, prior):
        return self.qweight.kl(prior)


class BayesConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, qfamily,
                 stride=1, padding=0, bias=True, q_kw=None):
        super().__init__()
        ks = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        shape = (out_channels, in_channels, ks[0], ks[1])
        self.qweight = make_qfamily(qfamily, shape, **(q_kw or {}))
        self.qweight.deterministic = False
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def forward(self, x):
        w = self.qweight.deterministic_weight() if self.qweight.deterministic else self.qweight.rsample()
        return F.conv2d(x, w, self.bias, stride=self.stride, padding=self.padding)

    def kl(self, prior):
        return self.qweight.kl(prior)


def model_kl(model, prior):
    """Sum KL over all Bayesian layers in a model."""
    total = 0.0
    for m in model.modules():
        if isinstance(m, (BayesLinear, BayesConv2d)):
            total = total + m.kl(prior)
    return total


def bayes_layers(model):
    return [m for m in model.modules() if isinstance(m, (BayesLinear, BayesConv2d))]


def set_deterministic(model, flag):
    """Toggle deterministic (eval) forward: use each q-family's deterministic_weight
    (exact zeros for atom families) instead of a stochastic sample."""
    for m in bayes_layers(model):
        m.qweight.deterministic = bool(flag)
