r"""Train / eval / measure for E1 (Tier A).

train_vi   : ELBO optimisation with KL annealing (AMP on CUDA only).
eval_vi    : test accuracy + NLL using deterministic (exact-zero) eval weights.
measure    : the E1 verdict per run --
               - empirical MI1(q,0) pooled over all weights (mass_index.estimate_mi),
               - per-layer forward/reverse budget Gamma(0) (closed form / inf-flag),
               - exact-zero / below-eps sparsity,
             returning the JSON-ready dict described in the plan.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from data import get_dataloaders, set_seed
from elbo import elbo_loss, anneal_beta, compute_kl
from layers.bayes_layers import bayes_layers, set_deterministic
from qfamilies import predicted_class
from mass_index import estimate_mi
from budgets import forward_budget, reverse_budget
from distributions.tsallis import INF


def train_vi(model, prior, train_loader, n_train, device, args):
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    use_amp = (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for epoch in range(args.epochs):
        beta = anneal_beta(args.kl_weight, epoch, args.anneal_epochs)
        run_nll = run_kl = nb = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            if use_amp:
                with torch.cuda.amp.autocast():
                    logits = model(x)
                    kl = compute_kl(model, prior)
                    loss, nll = elbo_loss(logits, y, kl, beta, n_train)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                logits = model(x)
                kl = compute_kl(model, prior)
                loss, nll = elbo_loss(logits, y, kl, beta, n_train)
                loss.backward()
                opt.step()
            run_nll += nll.item(); run_kl += float(kl.item()); nb += 1
        if args.verbose:
            print(f"  epoch {epoch+1}/{args.epochs}  nll={run_nll/nb:.4f}  "
                  f"kl={run_kl/nb:.1f}  beta={beta:.3g}")
    return {"final_nll": run_nll / nb, "kl_final": run_kl / nb}


@torch.no_grad()
def eval_vi(model, loader, device):
    model.eval()
    set_deterministic(model, True)
    correct = total = 0
    nll_sum = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        nll_sum += torch.nn.functional.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    set_deterministic(model, False)
    return {"test_acc": 100.0 * correct / total, "test_nll": nll_sum / total}


@torch.no_grad()
def measure(model, prior, args):
    """E1 measurement: MI, budgets, sparsity. Returns (mass_index, budget, sparsity) dicts."""
    layers = bayes_layers(model)

    # --- sparsity (exact-zero + below-eps over all weights) ---
    eps = args.threshold
    zeros = below = total = 0
    for L in layers:
        w = L.qweight.deterministic_weight().abs()
        zeros += int((w <= 1e-8).sum().item())
        below += int((w <= eps).sum().item())
        total += w.numel()
    sparsity = {
        "threshold_eps": eps,
        "exact_zero_frac": zeros / total,
        "below_thresh_frac": below / total,
        "params_total": total,
    }

    # --- empirical MI1(q,0) pooled over all weight draws ---
    radii = np.concatenate([L.qweight.sample_flat(args.mi_samples) for L in layers])
    mi = estimate_mi(radii, d=1, k=1, with_log=True)
    mi_out = {k: mi[k] for k in
              ("alpha_hat", "beta_hat", "mi1_hat", "mi2_hat", "atom_detected", "atom_mass_hat")}
    mi_out["radii"] = mi["radii"]
    mi_out["ball_mass"] = mi["ball_mass"]

    # --- per-layer forward/reverse budget Gamma(0) (alpha=1) ---
    alpha = 1.0
    atom_p = prior.atom_mass          # 1 - pi : the prior's atom at 0
    p0 = prior.cont_density_at_0      # a.c. density of the prior at 0 (may be +inf)
    gfwd_layers, grev_layers = [], []
    any_inf_fwd = False
    for L in layers:
        q = L.qweight
        atom_q = float(q.atom_mass_mean())
        d0 = q.density_at_0_elementwise()
        q0 = float(d0.mean().item()) if d0 is not None else 0.0
        gf, fflag = forward_budget(p0, q0, alpha, atom_p=atom_p, atom_q=atom_q)
        gr, _ = reverse_budget(p0, q0, alpha, atom_p=atom_p, atom_q=atom_q)
        gfwd_layers.append(gf); grev_layers.append(gr)
        any_inf_fwd = any_inf_fwd or fflag
    # forward budget is inf iff some layer's q fails to match the prior atom (continuous q)
    gamma_forward = INF if any_inf_fwd else float(np.mean([g for g in gfwd_layers if np.isfinite(g)] or [0.0]))
    budget = {
        "alpha": alpha,
        "atom_p": atom_p,
        "atom_q_mean": float(np.mean([L.qweight.atom_mass_mean() for L in layers])),
        "p0_prior_cont_density": (None if p0 == INF else p0),
        "q_has_atom": bool(getattr(layers[0].qweight, "has_atom", False)),
        "gamma_forward": (None if gamma_forward == INF else gamma_forward),
        "gamma_forward_is_inf": bool(gamma_forward == INF),
        "gamma_reverse_per_layer": [None if g == INF else g for g in grev_layers],
    }
    return mi_out, budget, sparsity


def run_e1_single(model, prior, args, device):
    """Full single run: train -> eval -> measure -> assemble JSON-ready result."""
    t0 = time.time()
    set_seed(args.seed)
    train_loader, val_loader, test_loader, n_train = get_dataloaders(
        args.dataset, args.batch_size, args.val_frac, args.data_dir)

    tr = train_vi(model, prior, train_loader, n_train, device, args)
    ev = eval_vi(model, test_loader, device)
    mi, budget, sparsity = measure(model, prior, args)

    result = {
        "meta": {
            "arch": args.arch, "dataset": args.dataset, "qfamily": args.qfamily,
            "slab": args.slab, "pi": args.pi, "kl_weight": args.kl_weight,
            "epochs": args.epochs, "seed": args.seed, "device": str(device),
            "vi_scope": "full",
        },
        "train": tr,
        "eval": ev,
        "sparsity": sparsity,
        "mass_index": mi,
        "budget": budget,
        "predicted_class": predicted_class(args.qfamily),
        "wall_clock_seconds": time.time() - t0,
    }
    return result
