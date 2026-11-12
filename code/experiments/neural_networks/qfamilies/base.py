r"""Variational family base class + the four E1 families' shared utilities.

A QFamily owns the variational parameters for ONE weight tensor of a given shape
and provides, uniformly across the four families:

  rsample()                 one reparameterized weight sample (used in the forward pass);
                            caches it so kl() can form the single-sample ELBO estimator.
  kl(prior)                 KL(q || p) summed over the tensor (closed-form where possible,
                            else single-sample MC with the cached rsample).
  deterministic_weight()    the eval/test weight, with *exact zeros* for atom families.
  exact_zero_fraction(eps)  fraction of deterministic weights with |w| <= eps.
  atom_mass_mean()          mean over elements of P(weight = 0)  (0 for continuous families).
  has_atom                  whether q places an atom at 0 (the E1 discriminator).
  sample_flat(n)            n i.i.d. |w| draws flattened -> fuel for the empirical MI estimator.

The E1 prediction (Thm "theo:budget behaviour" + "prop:finite"): atom families
(spike-slab, hard-concrete) keep exact zeros and large/inf MI1; continuous families
(Gaussian, Student-t/horseshoe GSM) have Gamma(0)=inf and MI1 -> 1.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

_LOG2PI = math.log(2.0 * math.pi)


def gauss_logpdf(x, mu, sigma):
    return -0.5 * _LOG2PI - torch.log(sigma) - 0.5 * ((x - mu) / sigma) ** 2


def student_logpdf(x, mu, sigma, nu):
    logC = (
        math.lgamma((nu + 1.0) / 2.0)
        - math.lgamma(nu / 2.0)
        - 0.5 * math.log(nu * math.pi)
    )
    z = (x - mu) / sigma
    return logC - torch.log(sigma) - 0.5 * (nu + 1.0) * torch.log1p(z * z / nu)


class QFamily(nn.Module):
    has_atom: bool = False

    def __init__(self, shape):
        super().__init__()
        self.shape = tuple(shape)
        self.numel = int(torch.tensor(self.shape).prod().item())
        self._w = None  # cached last rsample

    # --- to be implemented by subclasses ---
    def rsample(self):
        raise NotImplementedError

    def kl(self, prior):
        raise NotImplementedError

    def deterministic_weight(self):
        raise NotImplementedError

    def measure_sample(self):
        """A draw from the *actual* variational measure q for MI estimation.

        Defaults to rsample(); atom families that train with a relaxed gate must
        override to use the hard gate so the atom (exact zeros) is present, matching
        the deterministic eval semantics.
        """
        return self.rsample()

    def density_at_0_elementwise(self):
        """Per-element q-density at 0 (torch tensor) or None for atom families."""
        return None

    def atom_mass_mean(self):
        return 0.0

    # --- shared ---
    @torch.no_grad()
    def exact_zero_fraction(self, eps=1e-8):
        w = self.deterministic_weight()
        return float((w.abs() <= eps).float().mean().item())

    @torch.no_grad()
    def sample_flat(self, n_samples=1):
        """Return |w| draws (flattened, numpy) to feed the empirical MI estimator.

        n_samples independent reparameterized draws of the whole tensor are stacked,
        giving n_samples * numel scalar radii.
        """
        outs = []
        for _ in range(n_samples):
            outs.append(self.measure_sample().detach().abs().flatten())
        return torch.cat(outs).cpu().numpy()
