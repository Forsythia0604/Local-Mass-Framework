r"""Multi-seed ImageNet gate figures + table for Experiment 5.

Reads the per-seed gate marginals produced by

    structured_marginals.py --arch resnet50 --seed <s> --out marginals/imagenet_marginals_s<s>.npz

builds the small-ball and directional local RE-KL curves for each seed with the same
machinery as local_mass_nn.py, and plots the mean curve over seeds in the same style.
The seed-to-seed spread of these curves is ~1e-5, so no band is drawn; the spread is
reported numerically in the table caption instead.

    python imagenet_multiseed.py --seeds 1 2 3
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

import local_mass_nn as M
from local_mass_nn import (FIG_WIDTH_IN, PRIOR_COLOR, POST_COLOR,
                           rekl_forward, rekl_reverse, power_mass_index,
                           SpikeSlabPrior, configure_matplotlib, ROOT)
import matplotlib.pyplot as plt

ORDER = ["Gaussian", "Horseshoe", "Spike-and-slab", "Hard-concrete"]
STYLES = {"Gaussian": "-", "Horseshoe": "--", "Spike-and-slab": "-", "Hard-concrete": "--"}
COLORS = {"Gaussian": PRIOR_COLOR, "Horseshoe": "#2ca02c",
          "Spike-and-slab": POST_COLOR, "Hard-concrete": "#9467bd"}


def load_seeds(seeds, mdir):
    return {s: M.trained_marginals(str(Path(mdir) / f"imagenet_marginals_s{s}.npz")) for s in seeds}


def mean_curve(per_seed):
    a = np.array(per_seed, dtype=float)
    return a.mean(axis=0), a.std(axis=0, ddof=1) if a.shape[0] > 1 else np.zeros(a.shape[1])


def write_summary(summary, seeds, out, tag):
    """Export the already-computed per-seed values and four-family summary."""
    rows = []
    lines = [r"\begin{tabular}{lcccc}", r"\toprule",
             r"Family & Atom? & $\mathrm{MI}_{\mathrm{pow}}$ & Atom mean & Atom SD \\",
             r"\midrule"]
    for name in ORDER:
        values = summary[name]
        for i, seed in enumerate(seeds):
            rows.append({"family": name, "seed": seed, "has_atom": values["has_atom"],
                         "mi_pow": values["mi"][i], "finite_radius_slope": values["slope"][i],
                         "atom_mass": values["atom"][i]})
        mi = (r"$\infty$" if all(math.isinf(x) for x in values["mi"])
              else f"{np.mean(values['mi']):.2f}")
        sd = f"{values['atom'].std(ddof=1):.6f}" if len(seeds) > 1 else "--"
        atom = "yes" if values["has_atom"] else "no"
        lines.append(f"{name} & {atom} & {mi} & {values['atom'].mean():.6f} & {sd} " + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (out / f"table_nn_mi{tag}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (out / f"table_nn_mi{tag}_runs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--marg_dir", default=str(ROOT / "data" / "marginals"))
    ap.add_argument("--pi", type=float, default=0.2)
    ap.add_argument("--slab_loc", type=float, default=1.0)
    ap.add_argument("--grid_max", type=float, default=1.2)
    ap.add_argument("--tag", default="_imagenet")
    ap.add_argument("--output-dir", type=Path,
                    default=ROOT / "results" / "generated" / "exp04_05" / "figures",
                    help="Directory for newly generated outputs.")
    args = ap.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        ap.error("--seeds must not contain duplicates")
    missing = [str(Path(args.marg_dir) / f"imagenet_marginals_s{s}.npz") for s in args.seeds
               if not (Path(args.marg_dir) / f"imagenet_marginals_s{s}.npz").is_file()]
    if missing:
        ap.error("Missing marginal input(s): " + ", ".join(missing))

    configure_matplotlib()
    out = args.output_dir; out.mkdir(parents=True, exist_ok=True)
    prior = SpikeSlabPrior(pi=args.pi, s=1.0, loc=args.slab_loc)
    radii = np.geomspace(1e-3, args.grid_max, 24)
    fams_by_seed = load_seeds(args.seeds, args.marg_dir)

    # ---------------- small-ball mass ----------------
    fig, ax = plt.subplots(1, 1, figsize=(FIG_WIDTH_IN, 1.7), constrained_layout=True)
    ax.loglog(radii, [prior.ball_mass(r) for r in radii], color="0.4", lw=1.0,
              ls="-.", label="prior (spike-slab)")
    summary = {}
    for name in ORDER:
        curves = [[fams_by_seed[s][name].ball_mass(r) for r in radii] for s in args.seeds]
        mu, sd = mean_curve(curves)
        ax.loglog(radii, mu, STYLES[name], color=COLORS[name], label=name)
        mis, slopes, atoms = [], [], []
        for s in args.seeds:
            q = fams_by_seed[s][name]
            mi, sl = power_mass_index(radii, np.array([q.ball_mass(r) for r in radii]))
            mis.append(mi); slopes.append(sl)
            atoms.append(q.atom_mass() if q.has_atom else 0.0)
        summary[name] = dict(has_atom=fams_by_seed[args.seeds[0]][name].has_atom,
                             mi=mis, slope=np.array(slopes), atom=np.array(atoms),
                             curve_rel_spread=float(np.max(sd / np.maximum(mu, 1e-300))))
    ax.set_xlabel("radius r"); ax.set_ylabel(r"small-ball mass $q(B_r(0))$")
    ax.set_xlim(radii[0], radii[-1])
    ax.legend(frameon=False, loc="lower right", handlelength=1.6, ncol=1)
    ax.tick_params(which="both", direction="out", length=2.5); ax.grid(False)
    fig.savefig(out / f"fig_nn_small_ball{args.tag}.pdf")
    fig.savefig(out / f"fig_nn_small_ball{args.tag}.png", dpi=600)
    plt.close(fig)

    # ---------------- directional local RE-KL (same three curves as MNIST/CIFAR) --------
    fig, ax = plt.subplots(1, 1, figsize=(FIG_WIDTH_IN, 1.55), constrained_layout=True)
    def avg(fn, fam):
        return mean_curve([[fn(fams_by_seed[s][fam], prior, r) for r in radii] for s in args.seeds])[0]
    ax.loglog(radii, avg(rekl_forward, "Gaussian"), color=PRIOR_COLOR, label=r"$q\|p$ (Gaussian)")
    ax.loglog(radii, avg(rekl_reverse, "Gaussian"), color=POST_COLOR, ls="--", label=r"$p\|q$ (Gaussian)")
    ax.loglog(radii, avg(rekl_reverse, "Spike-and-slab"), color="#2ca02c", ls=":",
              label=r"$p\|q$ (spike-slab)")
    ax.set_xlabel("radius r"); ax.set_ylabel(r"local RE-KL  $\bar D_\alpha$")
    ax.set_xlim(radii[0], radii[-1])
    ax.legend(frameon=False, loc="center left", handlelength=1.8)
    ax.tick_params(which="both", direction="out", length=2.5); ax.grid(False)
    fig.savefig(out / f"fig_nn_directional_rekl{args.tag}.pdf")
    fig.savefig(out / f"fig_nn_directional_rekl{args.tag}.png", dpi=600)
    plt.close(fig)

    # ---------------- report ----------------
    print("prior atom mass = %.2f   seeds = %s   channels/gate-set fixed" % (1 - args.pi, args.seeds))
    print("%-16s %-6s %-10s %-22s %s" % ("family", "atom", "MI_pow", "atom mass (mean+-sd)", "max rel. curve spread"))
    for name in ORDER:
        d = summary[name]
        mi = "inf" if all(math.isinf(x) for x in d["mi"]) else "%.2f" % float(np.mean(d["mi"]))
        am = "%.5f +- %.5f" % (d["atom"].mean(), d["atom"].std(ddof=1))
        print("%-16s %-6s %-10s %-22s %.2e" % (name, "yes" if d["has_atom"] else "no",
                                               mi, am, d["curve_rel_spread"]))
    write_summary(summary, args.seeds, out, args.tag)
    print("figures + table + per-seed CSV -> %s" % out)


if __name__ == "__main__":
    main()
