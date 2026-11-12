r"""Family 2: spike-and-slab q_i = (1 - gamma_i) delta_0 + gamma_i N(mu_i, sigma_i^2).

ATOM-capable: gamma_i = sigmoid(logit_i) is the inclusion prob.  Against a spike-and-
slab prior this gives a finite forward budget (Thm "prop:finite") -> exact zeros and
MI1(q,0) = inf.  Training uses a hard-concrete relaxation of the Bernoulli gate so the
gate is differentiable; evaluation uses the hard gate (gamma > 0.5) -> exact zeros.

KL decomposes (independent gate + slab):
    KL(q||p) = KL_Bern(gamma || pi)  +  gamma * KL(N(mu,sigma) || slab),
with the slab KL closed-form for a Gaussian slab and single-sample MC otherwise.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import QFamily, gauss_logpdf

_BETA = 2.0 / 3.0  # concrete temperature for the relaxed gate


class SpikeSlabQ(QFamily):
    has_atom = True

    def __init__(self, shape, mu_init=0.0, sigma_init=0.05, gamma_init=0.5):
        super().__init__(shape)
        self.mu = nn.Parameter(torch.full(self.shape, float(mu_init))
                               + 0.01 * torch.randn(self.shape))
        rho0 = math.log(math.expm1(sigma_init))
        self.rho = nn.Parameter(torch.full(self.shape, float(rho0)))
        logit0 = math.log(gamma_init / (1.0 - gamma_init))
        self.gate_logit = nn.Parameter(torch.full(self.shape, float(logit0)))

    @property
    def sigma(self):
        return F.softplus(self.rho) + 1e-8

    @property
    def gamma(self):
        return torch.sigmoid(self.gate_logit)

    def _relaxed_gate(self):
        u = torch.rand_like(self.gate_logit).clamp(1e-6, 1 - 1e-6)
        logit_u = torch.log(u) - torch.log1p(-u)
        return torch.sigmoid((self.gate_logit + logit_u) / _BETA)

    def rsample(self):
        z = self._relaxed_gate()
        slab = self.mu + self.sigma * torch.randn_like(self.mu)
        self._w = z * slab
        return self._w

    def kl(self, prior):
        gamma = self.gamma.clamp(1e-6, 1 - 1e-6)
        pi = prior.pi
        # Bernoulli KL(gamma || pi)
        kl_bern = (gamma * (torch.log(gamma) - math.log(pi))
                   + (1 - gamma) * (torch.log1p(-gamma) - math.log(1 - pi)))
        # slab KL, weighted by inclusion gamma
        slab = prior.slab
        if slab.name == "gaussian":
            s_p = slab.s
            loc = getattr(slab, "loc", 0.0)   # slab centre (1 for channel gates)
            kl_slab = (math.log(s_p) - torch.log(self.sigma)
                       + (self.sigma ** 2 + (self.mu - loc) ** 2) / (2 * s_p ** 2) - 0.5)
        else:
            w = self.mu + self.sigma * torch.randn_like(self.mu)  # MC under slab q
            kl_slab = gauss_logpdf(w, self.mu, self.sigma) - slab.log_density(w)
        return (kl_bern + gamma * kl_slab).sum()

    def measure_sample(self):
        """Hard Bernoulli gate -> exact zeros (the actual spike-and-slab measure q),
        so the MI estimator sees the atom (the relaxed gate used in training does not)."""
        gamma = self.gamma.detach()
        z = (torch.rand_like(gamma) < gamma).float()
        slab = self.mu.detach() + self.sigma.detach() * torch.randn_like(self.mu)
        return z * slab

    def deterministic_weight(self):
        keep = (self.gamma.detach() > 0.5).float()
        return keep * self.mu.detach()

    def atom_mass_mean(self):
        return float((1.0 - self.gamma.detach()).mean().item())

    # atom family -> no continuous density at 0
    def density_at_0_elementwise(self):
        return None
