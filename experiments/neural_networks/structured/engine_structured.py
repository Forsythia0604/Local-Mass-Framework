r"""Train / eval for structured group-VI (Tier B/C).

Tier B (ResNet-56/CIFAR): single phase -- train base weights + gates jointly by ELBO.
Tier C (ResNet-50/ImageNet): Phase 1 freezes the pretrained base and trains only the
gates (cheap, selects channels); Phase 2 fixes the gates (deterministic masks) and
KD-fine-tunes the base, with resumable checkpoints for the 48 h walltime.
"""

from __future__ import annotations

import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import get_dataloaders, set_seed
from elbo import elbo_loss, anneal_beta


def make_optimizer(gnet, args, train_base=True):
    gate_params, base_params = [], []
    gate_ids = set()
    for g in gnet.gates.values():
        for p in g.parameters():
            gate_ids.add(id(p)); gate_params.append(p)
    if train_base:
        base_params = [p for p in gnet.base.parameters() if id(p) not in gate_ids]
    groups = [{"params": gate_params, "lr": args.gate_lr, "weight_decay": 0.0}]
    if train_base and base_params:
        groups.append({"params": base_params, "lr": args.lr, "weight_decay": args.weight_decay})
    return torch.optim.SGD(groups, momentum=0.9, nesterov=True)


def train_elbo(gnet, prior, train_loader, n_train, device, args, train_base=True):
    gnet.to(device).train()
    if not train_base:
        for p in gnet.base.parameters():
            p.requires_grad_(False)
    opt = make_optimizer(gnet, args, train_base=train_base)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    use_amp = (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Warmup + slow anneal. For `warmup` epochs the gates are frozen ON (beta=0) and only
    # the base trains, so per-channel importance is established first; then beta ramps
    # linearly over `anneal` epochs. This lets the per-channel inclusion probs separate --
    # important channels resist the KL pull, unimportant ones drop -- so the gammas cross
    # the 0.5 prune threshold at *different* beta, giving a graceful sparsity frontier
    # instead of an all-at-once collapse. No base warmup when the base is frozen (Tier C
    # phase 1: the pretrained base already provides channel importance).
    warmup = getattr(args, "warmup_epochs", 0) if train_base else 0
    anneal = max(1, getattr(args, "anneal_epochs", 1))

    def beta_at(epoch):
        if epoch < warmup:
            return 0.0
        e = epoch - warmup
        return args.kl_weight if e >= anneal else args.kl_weight * (e + 1) / anneal

    for epoch in range(args.epochs):
        gates_active = epoch >= warmup
        for g in gnet.gates.values():
            for p in g.parameters():
                p.requires_grad_(gates_active)
        beta = beta_at(epoch)
        rn = rk = nb = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            if use_amp:
                with torch.cuda.amp.autocast():
                    logits = gnet(x)
                    kl = gnet.kl(prior) if beta > 0 else logits.new_zeros(())
                    nll = F.cross_entropy(logits, y)
                    loss = nll + beta * kl   # structured: gate KL is NOT /N (Louizos-style direct penalty)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                logits = gnet(x)
                kl = gnet.kl(prior) if beta > 0 else logits.new_zeros(())
                nll = F.cross_entropy(logits, y)
                loss = nll + beta * kl
                loss.backward(); opt.step()
            rn += nll.item(); rk += float(kl.item()); nb += 1
        sched.step()
        if args.verbose:
            phase = "warmup" if epoch < warmup else "anneal"
            print(f"  epoch {epoch+1}/{args.epochs} [{phase}] nll={rn/nb:.4f} kl={rk/nb:.1f} beta={beta:.3g}")
    return {"final_nll": rn / nb, "kl_final": rk / nb}


@torch.no_grad()
def eval_structured(gnet, loader, device):
    gnet.eval(); gnet.set_deterministic(True)
    correct = total = 0; nll = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = gnet(x)
        nll += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(1) == y).sum().item(); total += y.size(0)
    gnet.set_deterministic(False)
    return {"test_acc": 100.0 * correct / total, "test_nll": nll / total}


# ---------------------------------------------------------------------------
# Phase 2 (Tier C): KD fine-tune with fixed gates + resumable checkpoints
# ---------------------------------------------------------------------------
def kd_loss(student_logits, teacher_logits, targets, T=4.0, alpha=0.1):
    soft = F.kl_div(F.log_softmax(student_logits / T, 1),
                    F.softmax(teacher_logits / T, 1), reduction="batchmean") * (T * T)
    hard = F.cross_entropy(student_logits, targets)
    return alpha * hard + (1 - alpha) * soft


def kd_finetune(gnet, teacher, train_loader, test_loader, device, args, ckpt_path):
    """Fix gates (deterministic), fine-tune base with KD. Resumable."""
    gnet.to(device); teacher.to(device).eval()
    gnet.set_deterministic(True)              # gates fixed -> masks frozen
    for g in gnet.gates.values():
        for p in g.parameters():
            p.requires_grad_(False)
    for p in gnet.base.parameters():
        p.requires_grad_(True)

    opt = torch.optim.SGD(gnet.base.parameters(), lr=args.ft_lr, momentum=0.9,
                          weight_decay=args.weight_decay, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.ft_epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    start_epoch = 0

    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        gnet.load_state_dict(ck["gnet"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        start_epoch = ck["epoch"] + 1
        print(f"resumed from {ckpt_path} at epoch {start_epoch}")

    for epoch in range(start_epoch, args.ft_epochs):
        gnet.train(); gnet.set_deterministic(True)
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                s = gnet(x)
                with torch.no_grad():
                    t = teacher(x)
                loss = kd_loss(s, t, y, T=args.kd_T, alpha=args.kd_alpha)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sched.step()
        torch.save({"epoch": epoch, "gnet": gnet.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "scaler": scaler.state_dict()}, ckpt_path)
        if args.verbose:
            acc = eval_structured(gnet, test_loader, device)["test_acc"]
            print(f"  KD epoch {epoch+1}/{args.ft_epochs} acc={acc:.2f}%")
        if args.stop_at_epoch and (epoch + 1) >= args.stop_at_epoch:
            print(f"stopping at epoch {epoch+1} (resume later)"); break
    return eval_structured(gnet, test_loader, device)
