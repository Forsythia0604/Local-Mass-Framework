r"""Priors used in E1: spike-and-slab with four slab choices.

The E1 prior (premise of Theorem "prop:finite") is the spike-and-slab measure

    p = (1 - pi) * delta_0  +  pi * S ,

which has an *atom* of mass (1 - pi) at 0, so MI1(p,0) = inf (Prop "prop:plainresult").
S is the slab; we provide Gaussian, Laplace, Student-t and Horseshoe slabs, whose
density-at-0 and tail order match the thesis catalogue (Sec. "sec:distributions",
Table at l.534).

Each slab exposes:
  * log_density(x)  : torch, the log Lebesgue density (used in MC KL during training);
  * density_at_0    : float (or +inf for the horseshoe log-spike).

The spike-and-slab prior exposes:
  * atom_mass               : 1 - pi (the singular part at 0);
  * cont_density_at_0       : pi * slab.density_at_0 (the a.c. part at 0);
  * log_prob_continuous(x)  : log(pi) + slab.log_density(x), the log of the a.c.
                              part density -- the reference for KL(q||p) when q is
                              continuous (the atom is p-singular, contributing the
                              singularity penalty handled in budgets/qfamilies).

Slabs are torch modules so densities differentiate; scalar density-at-0 values are
plain floats so the numpy numerical core (budgets gate) needs no torch.
"""

from __future__ import annotations

import math

import torch

INF = float("inf")
_LOG2PI = math.log(2.0 * math.pi)


# ---------------------------------------------------------------------------
# Slabs
# ---------------------------------------------------------------------------
class Slab:
    name = "slab"

    def log_density(self, x):  # torch -> torch
        raise NotImplementedError

    @property
    def density_at_0(self):  # float
        raise NotImplementedError


class GaussianSlab(Slab):
    name = "gaussian"

    def __init__(self, scale=1.0, loc=0.0):
        self.s = float(scale)
        self.loc = float(loc)   # slab centre; 0 for weight VI (Tier A), 1 for channel gates (Tier B/C)

    def log_density(self, x):
        return -0.5 * _LOG2PI - math.log(self.s) - 0.5 * ((x - self.loc) / self.s) ** 2

    @property
    def density_at_0(self):
        return math.exp(-0.5 * (self.loc / self.s) ** 2) / (self.s * math.sqrt(2.0 * math.pi))


class LaplaceSlab(Slab):
    name = "laplace"

    def __init__(self, b=1.0):
        self.b = float(b)

    def log_density(self, x):
        return -math.log(2.0 * self.b) - x.abs() / self.b

    @property
    def density_at_0(self):
        return 1.0 / (2.0 * self.b)


class StudentTSlab(Slab):
    name = "studentt"

    def __init__(self, nu=3.0, scale=1.0):
        self.nu = float(nu)
        self.s = float(scale)
        self._logC = (
            math.lgamma((self.nu + 1.0) / 2.0)
            - math.lgamma(self.nu / 2.0)
            - 0.5 * math.log(self.nu * math.pi)
            - math.log(self.s)
        )

    def log_density(self, x):
        z = x / self.s
        return self._logC - 0.5 * (self.nu + 1.0) * torch.log1p(z * z / self.nu)

    @property
    def density_at_0(self):
        return math.exp(self._logC)


class HorseshoeSlab(Slab):
    """Horseshoe marginal via the GSM x|lam ~ N(0, tau^2 lam^2), lam ~ C^+(1).

    The marginal has NO closed form; Carvalho-Polson-Scott (2010) give the tight
    bounds  (K/2) log(1 + 2/(x/tau)^2) < rho(x) < K log(1 + 4/(x/tau)^2),
    K = 1/sqrt(2 pi^3) (thesis l.523, *corrected* -- not a closed form).
    We use the upper bound as a differentiable surrogate density (it has the right
    log-spike at 0 and x^-2 tail).  density_at_0 = +inf (log-spike, MI1=1, MI2=1).
    """

    name = "horseshoe"
    _K = 1.0 / math.sqrt(2.0 * math.pi ** 3)

    def __init__(self, tau=1.0):
        self.tau = float(tau)

    def log_density(self, x):
        z2 = (x / self.tau) ** 2
        rho = self._K * torch.log1p(4.0 / (z2 + 1e-12)) / self.tau
        return torch.log(rho + 1e-30)

    @property
    def density_at_0(self):
        return INF


_SLABS = {
    "gaussian": GaussianSlab,
    "laplace": LaplaceSlab,
    "studentt": StudentTSlab,
    "horseshoe": HorseshoeSlab,
}


def make_slab(name, **kw):
    return _SLABS[name](**kw)


# ---------------------------------------------------------------------------
# Spike-and-slab prior
# ---------------------------------------------------------------------------
class SpikeSlabPrior:
    r"""p = (1 - pi) delta_0 + pi * S, with an atom of mass (1 - pi) at 0."""

    def __init__(self, pi=0.5, slab="gaussian", **slab_kw):
        if not (0.0 < pi < 1.0):
            raise ValueError("pi must be in (0,1) for a genuine spike-and-slab")
        self.pi = float(pi)
        self.slab = make_slab(slab, **slab_kw) if isinstance(slab, str) else slab

    @property
    def atom_mass(self):
        return 1.0 - self.pi

    @property
    def cont_density_at_0(self):
        """Density of the a.c. part at 0 = pi * slab(0) (+inf for horseshoe)."""
        d0 = self.slab.density_at_0
        return INF if d0 == INF else self.pi * d0

    def log_prob_continuous(self, x):
        """log of the a.c.-part density: log(pi) + slab.log_density(x)."""
        return math.log(self.pi) + self.slab.log_density(x)

    def __repr__(self):
        return f"SpikeSlab(pi={self.pi}, slab={self.slab.name})"
