r"""Family 3: hard-concrete / L0 gates (Louizos, Welling & Kingma 2018).

ATOM-capable via a stochastic gate z with a literal point mass at 0.  A deterministic
base weight theta is multiplied by a hard-concrete gate:

    u ~ U(0,1);  s = sigmoid((log u - log(1-u) + loga)/beta);
    s_bar = s*(zeta - gamma) + gamma;   z = clamp(s_bar, 0, 1);   w = theta * z.

P(z = 0) = sigmoid(loga - beta * log(-gamma/zeta)) > 0  -> exact zeros at eval.
The "KL" / complexity term is the expected number of active gates (expected-L0):
    E|z>0| = sum sigmoid(loga - beta*log(-gamma/zeta)) ... i.e. (1 - P(z=0)) per weight.
This plays the prior-regulariser role in the ELBO (weighted by kl_weight).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

_BETA = 2.0 / 3.0
_GAMMA = -0.1
_ZETA = 1.1
_CONST = _BETA * math.log(-_GAMMA / _ZETA)  # offset in P(z=0)/P(z!=0)


class HardConcreteL0(nn.Module):
    """Not a QFamily subclass (no slab/Gaussian posterior), but exposes the same
    interface used by the Bayesian layers and the measurement code."""

    has_atom = True

    def __init__(self, shape, theta_init=0.05, loga_init=0.0, gate_mode=False):
        super().__init__()
        self.shape = tuple(shape)
        self.numel = int(torch.tensor(self.shape).prod().item())
        self.gate_mode = gate_mode
        if gate_mode:
            # pure gate z in [0,1] multiplying a fixed unit base; loga_init>0 => start "on"
            self.register_buffer("theta", torch.ones(self.shape))
            if loga_init == 0.0:
                loga_init = 2.0
        else:
            self.theta = nn.Parameter(theta_init * torch.randn(self.shape))
        self.loga = nn.Parameter(torch.full(self.shape, float(loga_init)))
        self._w = None

    def _sample_z(self):
        u = torch.rand(self.shape, device=self.loga.device).clamp(1e-6, 1 - 1e-6)
        s = torch.sigmoid((torch.log(u) - torch.log1p(-u) + self.loga) / _BETA)
        s_bar = s * (_ZETA - _GAMMA) + _GAMMA
        return s_bar.clamp(0.0, 1.0)

    def rsample(self):
        self._w = self.theta * self._sample_z()
        return self._w

    def kl(self, prior):
        """Expected-L0 complexity = sum P(z != 0) = sum sigmoid(loga - beta*log(-gamma/zeta))."""
        p_active = torch.sigmoid(self.loga - _CONST)
        return p_active.sum()

    def _test_gate(self):
        # deterministic eval gate (Louizos): clamp(sigmoid(loga)*(zeta-gamma)+gamma, 0,1)
        s = torch.sigmoid(self.loga)
        s_bar = s * (_ZETA - _GAMMA) + _GAMMA
        return s_bar.clamp(0.0, 1.0)

    def deterministic_weight(self):
        return (self.theta * self._test_gate()).detach()

    @torch.no_grad()
    def exact_zero_fraction(self, eps=1e-8):
        w = self.deterministic_weight()
        return float((w.abs() <= eps).float().mean().item())

    def atom_mass_mean(self):
        p_zero = torch.sigmoid(_CONST - self.loga).detach()  # P(z=0)
        return float(p_zero.mean().item())

    def density_at_0_elementwise(self):
        return None  # atom family

    @torch.no_grad()
    def sample_flat(self, n_samples=1):
        outs = []
        for _ in range(n_samples):
            outs.append(self.rsample().detach().abs().flatten())
        return torch.cat(outs).cpu().numpy()
