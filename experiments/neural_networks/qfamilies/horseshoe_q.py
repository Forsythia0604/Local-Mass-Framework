r"""Family 4: horseshoe / Gaussian-scale-mixture q (continuous, NO atom).

We use the canonical *tractable* GSM: a mean-field Student-t,
    q_i = StudentT(nu; loc=mu_i, scale=sigma_i),
which is exactly a Gaussian scale mixture (Gaussian with inverse-gamma scale), has a
closed-form density, heavier-than-Gaussian tails, and a finite positive density at 0
-> MI1(q,0) = 1 (thesis Table at l.538, Student-t row).  Like the horseshoe it is a
continuous GSM with NO atom, so against a spike-and-slab prior the forward budget is
Gamma(0) = inf (Thm "prop:finite") and sparsity is predicted to collapse -- the second
negative control, complementing the Gaussian one with a heavy tail.

Reparameterized sampling: T = Z / sqrt(W/nu), Z ~ N(0,1), W ~ chi2(nu); w = mu + sigma*T.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import QFamily, student_logpdf


class HorseshoeQ(QFamily):
    has_atom = False

    def __init__(self, shape, mu_init=0.0, sigma_init=0.05, nu=3.0):
        super().__init__(shape)
        self.nu = float(nu)
        self.mu = nn.Parameter(torch.full(self.shape, float(mu_init))
                               + 0.01 * torch.randn(self.shape))
        rho0 = math.log(math.expm1(sigma_init))
        self.rho = nn.Parameter(torch.full(self.shape, float(rho0)))

    @property
    def sigma(self):
        return F.softplus(self.rho) + 1e-8

    def _sample_standard_t(self):
        z = torch.randn(self.shape, device=self.mu.device)
        # W ~ chi2(nu) = Gamma(nu/2, 2); reparameterizable via torch.distributions
        g = torch.distributions.Gamma(self.nu / 2.0, 0.5)
        w = g.rsample(self.shape).to(self.mu.device).clamp_min(1e-8)
        return z / torch.sqrt(w / self.nu)

    def rsample(self):
        t = self._sample_standard_t()
        self._w = self.mu + self.sigma * t
        return self._w

    def kl(self, prior):
        """MC KL with analytic Student-t log_q vs the prior's a.c. part."""
        w = self._w
        log_q = student_logpdf(w, self.mu, self.sigma, self.nu)
        log_p = prior.log_prob_continuous(w)
        return (log_q - log_p).sum()

    def deterministic_weight(self):
        return self.mu.detach()

    def density_at_0_elementwise(self):
        mu = self.mu.detach()
        return torch.exp(student_logpdf(torch.zeros_like(mu), mu, self.sigma.detach(), self.nu))

    def atom_mass_mean(self):
        return 0.0
