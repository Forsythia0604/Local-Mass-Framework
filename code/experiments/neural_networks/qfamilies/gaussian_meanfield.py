r"""Family 1: Gaussian mean-field q_i = N(mu_i, sigma_i^2).  The NEGATIVE control.

Continuous, no atom -> forward budget Gamma(0) = inf against a spike-and-slab prior
(Thm "prop:finite").  Predicted: cannot keep exact zeros; MI1(q,0) -> 1.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import QFamily, gauss_logpdf


class GaussianMeanField(QFamily):
    has_atom = False

    def __init__(self, shape, mu_init=0.0, sigma_init=0.05):
        super().__init__(shape)
        self.mu = nn.Parameter(torch.full(self.shape, float(mu_init))
                               + 0.01 * torch.randn(self.shape))
        # softplus(rho) = sigma_init  =>  rho = log(exp(sigma)-1)
        rho0 = math.log(math.expm1(sigma_init))
        self.rho = nn.Parameter(torch.full(self.shape, float(rho0)))

    @property
    def sigma(self):
        return F.softplus(self.rho) + 1e-8

    def rsample(self):
        eps = torch.randn_like(self.mu)
        self._w = self.mu + self.sigma * eps
        return self._w

    def kl(self, prior):
        """KL(N || spike-slab) is intractable -> single-sample MC with cached w:
        E_q[log q(w) - log p_cont(w)], p_cont = a.c. part of the prior."""
        w = self._w
        log_q = gauss_logpdf(w, self.mu, self.sigma)
        log_p = prior.log_prob_continuous(w)
        return (log_q - log_p).sum()

    def deterministic_weight(self):
        return self.mu.detach()

    def density_at_0_elementwise(self):
        s = self.sigma.detach()
        return torch.exp(gauss_logpdf(torch.zeros_like(self.mu), self.mu.detach(), s))

    def atom_mass_mean(self):
        return 0.0
