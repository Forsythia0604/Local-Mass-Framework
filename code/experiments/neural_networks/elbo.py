r"""ELBO objective (thesis Sec. "sec:vi-unified").

Maximise  E_q[log p(D|theta)] - KL(q||p)  <=>  minimise per-datapoint

    loss = NLL(mean over batch) + beta * KL / dataset_size,

where beta = kl_weight sweeps the sparsity-accuracy frontier (the E2-style knob,
also used here to make the spike-and-slab families actually sparsify).  KL annealing
linearly ramps beta over the first `anneal_epochs` epochs for stable optimisation.
"""

from __future__ import annotations

import torch.nn.functional as F

from layers.bayes_layers import model_kl


def elbo_loss(logits, targets, kl, kl_weight, dataset_size):
    nll = F.cross_entropy(logits, targets)
    loss = nll + kl_weight * kl / dataset_size
    return loss, nll


def anneal_beta(base_beta, epoch, anneal_epochs):
    if anneal_epochs <= 0:
        return base_beta
    return base_beta * min(1.0, (epoch + 1) / anneal_epochs)


def compute_kl(model, prior):
    return model_kl(model, prior)
