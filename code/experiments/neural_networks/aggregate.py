r"""Aggregate Tier-A run JSONs into mean +/- std tables, a LaTeX table, and figures.

Outputs (into --fig_dir / --tex_out):
  * dichotomy table  : per family -- MI1, atom?, Gamma(0) (finite/inf), exact-zero frac, acc, NLL.
  * frontier figure  : test acc vs exact-zero fraction over the kl_weight sweep.
  * headline figure  : MI1 (capped) and exact-zero fraction by family.
The LaTeX table is written for inclusion by Results/e1_results.tex.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import defaultdict

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FAMILY_ORDER = ["gaussian", "horseshoe", "spikeslab", "hardconcrete"]
FAMILY_LABEL = {"gaussian": "Gaussian MF", "horseshoe": "Horseshoe (GSM)",
                "spikeslab": "Spike-and-slab", "hardconcrete": "Hard-concrete L0"}


def _load(out_dir):
    runs = []
    for fp in glob.glob(os.path.join(out_dir, "*.json")):
        with open(fp) as f:
            runs.append(json.load(f))
    return runs


def _ms(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    if not xs:
        return float("nan"), float("nan")
    return float(np.mean(xs)), float(np.std(xs))


def _sparsity_frac(r):
    """Tier A reports exact_zero_frac (per weight); Tier B/C structural_sparsity (per channel)."""
    s = r["sparsity"]
    return s.get("exact_zero_frac", s.get("structural_sparsity"))


def dichotomy_table(runs):
    by_fam = defaultdict(list)
    for r in runs:
        if "sparsity" in r and r["meta"].get("seed") is not None:
            by_fam[r["meta"]["qfamily"]].append(r)
    rows = []
    for fam in FAMILY_ORDER:
        rs = by_fam.get(fam, [])
        if not rs:
            continue
        acc_m, acc_s = _ms([r["eval"]["test_acc"] for r in rs])
        nll_m, nll_s = _ms([r["eval"]["test_nll"] for r in rs])
        ez_m, ez_s = _ms([_sparsity_frac(r) for r in rs])
        mi_vals = [r["mass_index"]["mi1_hat"] for r in rs]
        atom = all(r["mass_index"]["atom_detected"] for r in rs)
        mi_disp = "$\\infty$" if atom else f"{np.mean([m for m in mi_vals if np.isfinite(m)]):.2f}"
        g_inf = all(r["budget"]["gamma_forward_is_inf"] for r in rs)
        g_disp = "$\\infty$" if g_inf else f"{_ms([r['budget']['gamma_forward'] for r in rs])[0]:.3f}"
        rows.append({
            "family": fam, "atom": atom, "mi1": mi_disp, "gamma": g_disp,
            "ez": (ez_m, ez_s), "acc": (acc_m, acc_s), "nll": (nll_m, nll_s),
            "pred": rs[0]["predicted_class"],
        })
    return rows


def write_latex_table(rows, path):
    lines = [
        r"\begin{tabular}{l c c c c c}",
        r"\toprule",
        r"$q$-family & atom? & $\mathrm{MI}_1(q,0)$ & $\Gamma(0)$ & exact-zero \% & test acc.\ \% \\",
        r"\midrule",
    ]
    for r in rows:
        ez = f"{100*r['ez'][0]:.1f}$\\pm${100*r['ez'][1]:.1f}"
        acc = f"{r['acc'][0]:.2f}$\\pm${r['acc'][1]:.2f}"
        atom = "yes" if r["atom"] else "no"
        lines.append(f"{FAMILY_LABEL[r['family']]} & {atom} & {r['mi1']} & {r['gamma']} & {ez} & {acc} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {path}")


def headline_figure(rows, path):
    fams = [r["family"] for r in rows]
    ez = [100 * r["ez"][0] for r in rows]
    ez_e = [100 * r["ez"][1] for r in rows]
    mi_cap = [12 if r["atom"] else float(r["mi1"]) for r in rows]
    colors = ["#d65f5f" if r["pred"] == "continuous" else "#4c72b0" for r in rows]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    x = np.arange(len(fams))
    a1.bar(x, ez, yerr=ez_e, color=colors)
    a1.set_xticks(x); a1.set_xticklabels([FAMILY_LABEL[f] for f in fams], rotation=20, ha="right")
    a1.set_ylabel("exact-zero fraction (%)"); a1.set_title("Sparsity preserved")
    a2.bar(x, mi_cap, color=colors)
    a2.axhline(1.0, ls="--", c="k", lw=0.8)
    a2.set_xticks(x); a2.set_xticklabels([FAMILY_LABEL[f] for f in fams], rotation=20, ha="right")
    a2.set_ylabel(r"$\mathrm{MI}_1(q,0)$ (capped at 12 = $\infty$)"); a2.set_title("Mass Index at 0")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    print(f"wrote {path}")


def frontier_figure(runs, path):
    pts = defaultdict(list)  # (family, kw) -> [(ez, acc)]
    for r in runs:
        if "sparsity" not in r or "kl_weight" not in r["meta"]:
            continue
        fam = r["meta"]["qfamily"]
        kw = r["meta"]["kl_weight"]
        pts[(fam, kw)].append((_sparsity_frac(r), r["eval"]["test_acc"]))
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for fam in ["spikeslab", "hardconcrete"]:
        curve = []
        for (f, kw), vals in sorted(pts.items(), key=lambda kv: kv[0][1]):
            if f != fam:
                continue
            ez = np.mean([v[0] for v in vals]); acc = np.mean([v[1] for v in vals])
            ez_e = np.std([v[0] for v in vals]); acc_e = np.std([v[1] for v in vals])
            curve.append((ez, acc, ez_e, acc_e))
        if curve:
            curve.sort()
            ez, acc, eze, acce = zip(*curve)
            ax.errorbar([100 * e for e in ez], acc, xerr=[100 * e for e in eze], yerr=acce,
                        marker="o", capsize=3, label=FAMILY_LABEL[fam])
    ax.set_xlabel("exact-zero fraction (%)"); ax.set_ylabel("test accuracy (%)")
    ax.set_title("Sparsity-accuracy frontier (kl-weight sweep)")
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/tierA")
    ap.add_argument("--fig_dir", default="../../Results/figures")
    ap.add_argument("--tex_out", default="../../Results/tab_dichotomy.tex")
    args = ap.parse_args()
    runs = _load(args.out)
    print(f"loaded {len(runs)} runs from {args.out}")
    os.makedirs(args.fig_dir, exist_ok=True)

    rows = dichotomy_table(runs)
    for r in rows:
        print(f"  {r['family']:14s} atom={r['atom']!s:5s} MI1={r['mi1']:8s} "
              f"Gamma={r['gamma']:8s} ez={100*r['ez'][0]:5.1f}% acc={r['acc'][0]:.2f}%")
    write_latex_table(rows, args.tex_out)
    headline_figure(rows, os.path.join(args.fig_dir, "e1_headline.png"))
    frontier_figure(runs, os.path.join(args.fig_dir, "e1_frontier.png"))


if __name__ == "__main__":
    main()
