r"""Group-gated network: one variational gate per prunable channel (Tier B/C).

A gate is a q-family over a (C,) vector that multiplies a conv's output channels.
The *group atom* is a whole channel switched off (gate = 0) -- the structured reading
of the atom-matching theorem (Thm "prop:finite"): only gate families that can represent
an atom (spike-slab, hard-concrete) keep channels exactly off (finite forward budget);
continuous gates (Gaussian, horseshoe) have Gamma(0)=inf and cannot.

Prunable channels (reusing the standard ResNet structured choice, as in FLOPP):
  * BasicBlock  -> gate conv1 output channels.
  * Bottleneck  -> gate conv1 and conv2 output channels.
Stem, downsample and block-output convs are left ungated (residual-add constrained).

Gates start "on" so a trained / pretrained base net is not destroyed at init; KL
pressure then switches channels off.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qfamilies import make_qfamily, predicted_class
from mass_index import estimate_mi
from budgets import forward_budget, reverse_budget
from distributions.tsallis import INF

# gate-appropriate inits ("on" at start)
_GATE_INIT = {
    "gaussian":     dict(mu_init=1.0, sigma_init=0.1),
    "horseshoe":    dict(mu_init=1.0, sigma_init=0.1, nu=3.0),
    "spikeslab":    dict(mu_init=1.0, sigma_init=0.1, gamma_init=0.95),
    "hardconcrete": dict(gate_mode=True, loga_init=2.0),
}


def _prunable_convs(model):
    """Yield (name, conv_module, out_channels) for the gated inner convs."""
    out = []
    for name, m in model.named_modules():
        cls = m.__class__.__name__
        if cls == "BasicBlock":
            out.append((f"{name}.conv1", m.conv1, m.conv1.out_channels))
        elif cls == "Bottleneck":
            out.append((f"{name}.conv1", m.conv1, m.conv1.out_channels))
            out.append((f"{name}.conv2", m.conv2, m.conv2.out_channels))
    return out


class GatedNet(nn.Module):
    def __init__(self, base, qfamily, num_classes=10):
        super().__init__()
        self.base = base
        self.qfamily = qfamily
        self.gates = nn.ModuleDict()
        self._order = []          # gate keys in registration order
        self._handles = []
        init = _GATE_INIT[qfamily]

        for name, conv, C in _prunable_convs(base):
            key = name.replace(".", "__")
            gate = make_qfamily(qfamily, (C,), **init)
            gate.deterministic = False
            self.gates[key] = gate
            self._order.append(key)
            self._handles.append(conv.register_forward_hook(self._make_hook(key)))

    def _make_hook(self, key):
        def hook(module, inp, out):
            gate = self.gates[key]
            z = gate.deterministic_weight() if gate.deterministic else gate.rsample()
            return out * z.view(1, -1, 1, 1)
        return hook

    def forward(self, x):
        return self.base(x)

    def set_deterministic(self, flag):
        for k in self._order:
            self.gates[k].deterministic = bool(flag)

    def kl(self, prior):
        total = 0.0
        for k in self._order:
            total = total + self.gates[k].kl(prior)
        return total

    # ---- E1 structured measurement ----
    @torch.no_grad()
    def measure(self, prior, mi_samples=8, threshold=1e-3):
        # structural sparsity: fraction of channels switched off
        off = tot = 0
        for k in self._order:
            z = self.gates[k].deterministic_weight().abs()
            off += int((z <= 1e-8).sum().item()); tot += z.numel()
        sparsity = {
            "threshold_eps": threshold,
            "channels_total": tot,
            "channels_off": off,
            "structural_sparsity": off / tot,
        }
        # pooled empirical MI over gate values
        radii = np.concatenate([self.gates[k].sample_flat(mi_samples) for k in self._order])
        mi = estimate_mi(radii, d=1, k=1, with_log=True)
        mi_out = {kk: mi[kk] for kk in
                  ("alpha_hat", "beta_hat", "mi1_hat", "mi2_hat", "atom_detected", "atom_mass_hat")}

        # forward/reverse budget at the group level (alpha=1)
        atom_p = prior.atom_mass
        p0 = prior.cont_density_at_0
        any_inf = False
        gfwd = []
        for k in self._order:
            g = self.gates[k]
            atom_q = float(g.atom_mass_mean())
            d0 = g.density_at_0_elementwise()
            q0 = float(d0.mean().item()) if d0 is not None else 0.0
            gf, fl = forward_budget(p0, q0, 1.0, atom_p=atom_p, atom_q=atom_q)
            gfwd.append(gf); any_inf = any_inf or fl
        gamma_forward = INF if any_inf else float(np.mean([g for g in gfwd if np.isfinite(g)] or [0.0]))
        budget = {
            "alpha": 1.0, "atom_p": atom_p,
            "atom_q_mean": float(np.mean([self.gates[k].atom_mass_mean() for k in self._order])),
            "gamma_forward": (None if gamma_forward == INF else gamma_forward),
            "gamma_forward_is_inf": bool(gamma_forward == INF),
            "q_has_atom": bool(getattr(self.gates[self._order[0]], "has_atom", False)),
        }
        return mi_out, budget, sparsity

    @torch.no_grad()
    def channel_keep_masks(self):
        """Per-gate boolean keep-mask (channel on) -- for physical pruning / FLOPs."""
        self.set_deterministic(True)
        masks = {k: (self.gates[k].deterministic_weight().abs() > 1e-8).cpu()
                 for k in self._order}
        self.set_deterministic(False)
        return masks


def build_gated(arch, qfamily, num_classes, pretrained=False):
    if arch == "resnet56":
        from structured.resnet_cifar import resnet56
        base = resnet56(num_classes=num_classes)
    elif arch == "resnet50":
        import torchvision
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        base = torchvision.models.resnet50(weights=weights)
    else:
        raise ValueError(f"unknown structured arch '{arch}'")
    return GatedNet(base, qfamily, num_classes=num_classes)
