r"""Local mass and directional RE-KL of Bayesian neural-network variational posteriors.

A new experiment in the style of the AISTATS paper's Section 6 (Local-Mass-Framework),
extending its controlled small-ball / RE-KL illustrations from a d=5 UCI logistic
regression to the variational posteriors of a Bayesian neural network. The measured
objects are exactly those of the paper:

  * small-ball mass  q(B_r(0))  ->  Power Mass Index MI_pow (finite-radius slope), and
  * directional RE-KL  Dbar_alpha(q||p ; B_r)  vs  Dbar_alpha(p||q ; B_r)  as r -> 0.

We compare four mean-field variational families against a spike-and-slab prior (atom at 0):
Gaussian and Student-t/horseshoe (continuous; MI_pow = 1) vs spike-and-slab and
hard-concrete gates (atom-capable; MI_pow = inf). Because q is mean-field and sparsity is
coordinatewise, the local mass at 0 is read per coordinate (a 1-D small-ball); the joint
ball is degenerate at network scale.

Plotting conventions (single-column width, fonts, colours, geomspace grids, finite-radius
slopes) match the paper's figure script for drop-in consistency.

Modes:
  --mode synthetic : representative trained-like marginals (runs instantly; for figure dev).
  --mode trained   : load per-coordinate marginal stats from a Bayesian-NN run (npz).

Stack matches the paper: numpy, scipy, matplotlib.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np
from scipy.stats import norm, t as student_t
from scipy.integrate import quad

ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "results" / ".mplconfig"))
(ROOT / "results" / ".mplconfig").mkdir(parents=True, exist_ok=True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

# ---- paper plotting conventions (copied from their local_mass_experiments.py) ----
FIG_WIDTH_IN = 3.35
PRIOR_COLOR = "#1f77b4"
POST_COLOR = "#d95f02"
ALPHA_REKL = 0.5  # RE-KL order; alpha in (0,1), the Hellinger point
TAG = ""          # filename suffix (e.g. "_cifar", "_imagenet")


def configure_matplotlib() -> None:
    plt.rcParams.update({
        "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
        "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "lines.linewidth": 1.2, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.015,
    })


def f_alpha(x, alpha=ALPHA_REKL):
    """Tsallis / RE-KL generator f_alpha(x) = (x^a - a x + a - 1)/(a-1), f_alpha(0)=1."""
    x = np.asarray(x, dtype=float)
    return (np.power(x, alpha) - alpha * x + alpha - 1.0) / (alpha - 1.0)


# ===========================================================================
#  Prior and variational-family marginals (per coordinate, 1-D)
# ===========================================================================
class SpikeSlabPrior:
    """p = (1-pi) delta_0 + pi N(loc, s^2): atom of mass (1-pi) at 0.

    loc=0 for per-weight VI (sparsity at 0); loc=1 for channel gates ("on" slab at 1).
    """
    has_atom = True

    def __init__(self, pi=0.5, s=1.0, loc=0.0):
        self.pi, self.s, self.loc = float(pi), float(s), float(loc)

    @property
    def atom_mass(self):
        return 1.0 - self.pi

    def cont_density(self, w):  # a.c. part density: pi * phi(w;loc,s)
        return self.pi * norm.pdf(w, self.loc, self.s)

    def ball_mass(self, r):  # (1-pi) + pi P(N(loc,s) in [-r,r])
        return (1.0 - self.pi) + self.pi * (norm.cdf((r - self.loc) / self.s)
                                            - norm.cdf((-r - self.loc) / self.s))


class GaussianQ:
    """Gaussian mean-field marginal, parameters pooled over coordinates (mu_i, sg_i)."""
    name, has_atom = "Gaussian", False

    def __init__(self, mu, sg):
        self.mu = np.asarray(mu, float); self.sg = np.asarray(sg, float)

    def atom_mass(self):
        return 0.0

    def cont_density(self, w):  # mixture over coordinates
        return np.mean(norm.pdf(w, self.mu, self.sg))

    def ball_mass(self, r):
        return float(np.mean(norm.cdf((r - self.mu) / self.sg) - norm.cdf((-r - self.mu) / self.sg)))


class StudentTQ:
    """Student-t (horseshoe/GSM) marginal: continuous, heavier tail, MI_pow=1."""
    name, has_atom = "Horseshoe", False

    def __init__(self, mu, sg, nu=3.0):
        self.mu = np.asarray(mu, float); self.sg = np.asarray(sg, float); self.nu = float(nu)

    def atom_mass(self):
        return 0.0

    def cont_density(self, w):
        return np.mean(student_t.pdf((w - self.mu) / self.sg, self.nu) / self.sg)

    def ball_mass(self, r):
        z_hi = (r - self.mu) / self.sg; z_lo = (-r - self.mu) / self.sg
        return float(np.mean(student_t.cdf(z_hi, self.nu) - student_t.cdf(z_lo, self.nu)))


class SpikeSlabQ:
    """q = (1-g) delta_0 + g N(mu, sg^2): atom-capable, MI_pow=inf."""
    name, has_atom = "Spike-and-slab", True

    def __init__(self, mu, sg, gamma):
        self.mu = np.asarray(mu, float); self.sg = np.asarray(sg, float)
        self.gamma = np.asarray(gamma, float)

    def atom_mass(self):
        return float(np.mean(1.0 - self.gamma))

    def cont_density(self, w):  # a.c. part = mean_i g_i phi(w;mu_i,sg_i)
        return np.mean(self.gamma * norm.pdf(w, self.mu, self.sg))

    def ball_mass(self, r):
        slab = norm.cdf((r - self.mu) / self.sg) - norm.cdf((-r - self.mu) / self.sg)
        return float(np.mean((1.0 - self.gamma) + self.gamma * slab))


class HardConcreteQ:
    """Hard-concrete gate marginal: atom P(z=0) at 0, MI_pow=inf (effective atom)."""
    name, has_atom = "Hard-concrete", True

    def __init__(self, theta, p_zero, sg_eff):
        self.theta = np.asarray(theta, float); self.p0 = np.asarray(p_zero, float)
        self.sg = np.asarray(sg_eff, float)

    def atom_mass(self):
        return float(np.mean(self.p0))

    def cont_density(self, w):  # active part ~ (1-p0) N(theta, sg_eff^2)
        return np.mean((1.0 - self.p0) * norm.pdf(w, self.theta, self.sg))

    def ball_mass(self, r):
        active = norm.cdf((r - self.theta) / self.sg) - norm.cdf((-r - self.theta) / self.sg)
        return float(np.mean(self.p0 + (1.0 - self.p0) * active))


# ===========================================================================
#  Directional RE-KL on the ball B_r(0), normalised (their Dbar)
# ===========================================================================
def rekl_forward(q, p, r, alpha=ALPHA_REKL):
    """Dbar_alpha(q||p ; B_r) = D_alpha(q||p;B_r)/p(B_r).  (q relative to prior p)"""
    def integrand(w):
        pc = p.cont_density(w)
        if pc <= 0:
            return 0.0
        return f_alpha(q.cont_density(w) / pc, alpha) * pc
    ac, _ = quad(integrand, -r, r, limit=100, points=[0.0])
    # atom of p at 0: contributes f_alpha(q_atom/p_atom) * p_atom
    qa, pa = q.atom_mass(), p.atom_mass
    atom_term = f_alpha(qa / pa, alpha) * pa if pa > 0 else 0.0
    # q singular wrt p: q's support is R (covered by p's a.c. part) -> none here
    return (ac + atom_term) / p.ball_mass(r)


def rekl_reverse(q, p, r, alpha=ALPHA_REKL):
    """Dbar_alpha(p||q ; B_r) = D_alpha(p||q;B_r)/q(B_r).  (prior p relative to q)"""
    def integrand(w):
        qc = q.cont_density(w)
        if qc <= 0:
            return 0.0
        return f_alpha(p.cont_density(w) / qc, alpha) * qc
    ac, _ = quad(integrand, -r, r, limit=100, points=[0.0])
    pa, qa = p.atom_mass, q.atom_mass()
    if qa > 0:                      # atoms shared at 0 -> a.c., finite atom term
        atom_term = f_alpha(pa / qa, alpha) * qa
    else:                           # p atom singular wrt continuous q -> recession penalty
        atom_term = alpha / (1.0 - alpha) * pa
    return (ac + atom_term) / q.ball_mass(r)


# ===========================================================================
#  Marginal stats: synthetic (representative) or trained (npz from a BNN run)
# ===========================================================================
def synthetic_marginals(seed=0, n=4000):
    """Representative trained-like per-coordinate marginals for the four families."""
    rng = np.random.default_rng(seed)
    mu = 0.02 * rng.standard_normal(n)          # weights centred near 0
    sg = np.full(n, 0.05)                        # posterior scale
    gamma = rng.uniform(0.2, 0.6, n)             # spike-slab inclusion (=> ~60% atom)
    p_zero = rng.uniform(0.3, 0.8, n)            # hard-concrete P(z=0)
    return {
        "Gaussian": GaussianQ(mu, sg),
        "Horseshoe": StudentTQ(mu, sg, nu=3.0),
        "Spike-and-slab": SpikeSlabQ(mu, sg, gamma),
        "Hard-concrete": HardConcreteQ(mu, p_zero, np.full(n, 0.05)),
    }


def trained_marginals(npz_path):
    """Load marginal stats saved by a run; include only the families present."""
    z = np.load(npz_path)
    fams = {}
    if "g_mu" in z:
        fams["Gaussian"] = GaussianQ(z["g_mu"], z["g_sg"])
    if "h_mu" in z:
        fams["Horseshoe"] = StudentTQ(z["h_mu"], z["h_sg"], nu=float(z["h_nu"]) if "h_nu" in z else 3.0)
    if "ss_mu" in z:
        fams["Spike-and-slab"] = SpikeSlabQ(z["ss_mu"], z["ss_sg"], z["ss_gamma"])
    if "hc_theta" in z:
        fams["Hard-concrete"] = HardConcreteQ(z["hc_theta"], z["hc_p0"], z["hc_sg"])
    return fams


# ===========================================================================
#  Mass Index from finite-radius slopes (their convention)
# ===========================================================================
def power_mass_index(radii, ball_mass):
    """MI_pow ~ d / slope at the smallest radii; plateau (slope~0) => inf."""
    lr, lm = np.log(radii), np.log(np.maximum(ball_mass, 1e-300))
    slopes = np.diff(lm) / np.diff(lr)
    s = float(np.mean(slopes[:4]))             # small-radius slope (d=1 per coordinate)
    return (math.inf if s < 0.15 else 1.0 / s), s


# ===========================================================================
#  Figures (paper style)
# ===========================================================================
def fig_small_ball(fams, prior, radii, out, write_png=True):
    fig, ax = plt.subplots(1, 1, figsize=(FIG_WIDTH_IN, 1.7), constrained_layout=True)
    ax.loglog(radii, [prior.ball_mass(r) for r in radii], color="0.4", lw=1.0,
              ls="-.", label="prior (spike-slab)")
    styles = {"Gaussian": "-", "Horseshoe": "--", "Spike-and-slab": "-", "Hard-concrete": "--"}
    colors = {"Gaussian": PRIOR_COLOR, "Horseshoe": "#2ca02c",
              "Spike-and-slab": POST_COLOR, "Hard-concrete": "#9467bd"}
    for name, q in fams.items():
        ax.loglog(radii, [q.ball_mass(r) for r in radii], styles[name], color=colors[name], label=name)
    ax.set_xlabel("radius r"); ax.set_ylabel(r"small-ball mass $q(B_r(0))$")
    ax.set_xlim(radii[0], radii[-1])
    ax.legend(frameon=False, loc="lower right", handlelength=1.6, ncol=1)
    ax.tick_params(which="both", direction="out", length=2.5); ax.grid(False)
    fig.savefig(out / f"fig_nn_small_ball{TAG}.pdf")
    if write_png:
        fig.savefig(out / f"fig_nn_small_ball{TAG}.png", dpi=600)
    plt.close(fig)


def fig_directional(fams, prior, radii, out, write_png=True):
    fig, ax = plt.subplots(1, 1, figsize=(FIG_WIDTH_IN, 1.55), constrained_layout=True)
    g, ss = fams["Gaussian"], fams["Spike-and-slab"]
    fwd_g = [rekl_forward(g, prior, r) for r in radii]
    rev_g = [rekl_reverse(g, prior, r) for r in radii]
    rev_ss = [rekl_reverse(ss, prior, r) for r in radii]
    ax.loglog(radii, fwd_g, color=PRIOR_COLOR, label=r"$q\|p$ (Gaussian)")
    ax.loglog(radii, rev_g, color=POST_COLOR, ls="--", label=r"$p\|q$ (Gaussian)")
    ax.loglog(radii, rev_ss, color="#2ca02c", ls=":", label=r"$p\|q$ (spike-slab)")
    ax.set_xlabel("radius r"); ax.set_ylabel(r"local RE-KL  $\bar D_\alpha$")
    ax.set_xlim(radii[0], radii[-1])
    ax.legend(frameon=False, loc="center left", handlelength=1.8)
    ax.tick_params(which="both", direction="out", length=2.5); ax.grid(False)
    fig.savefig(out / f"fig_nn_directional_rekl{TAG}.pdf")
    if write_png:
        fig.savefig(out / f"fig_nn_directional_rekl{TAG}.png", dpi=600)
    plt.close(fig)


def write_mi_table(fams, radii, out):
    lines = [r"\begin{tabular}{lcc}", r"\toprule",
             r"variational family & atom? & $\MI_{\mathrm{pow}}(q,0)$ \\", r"\midrule"]
    rows = []
    for name, q in fams.items():
        bm = np.array([q.ball_mass(r) for r in radii])
        mi, slope = power_mass_index(radii, bm)
        atom = "yes" if q.has_atom else "no"
        mid = r"$\infty$" if math.isinf(mi) else f"{mi:.2f}"
        lines.append(f"{name} & {atom} & {mid} \\\\")
        rows.append((name, atom, mid, slope))
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / f"table_nn_mi{TAG}.tex").write_text("\n".join(lines) + "\n")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["synthetic", "trained"], default="synthetic")
    ap.add_argument("--npz", default=None, help="trained marginal stats (.npz) for --mode trained")
    ap.add_argument("--pi", type=float, default=0.5, help="prior slab weight; atom mass = 1-pi")
    ap.add_argument("--slab_scale", type=float, default=1.0)
    ap.add_argument("--slab_loc", type=float, default=0.0,
                    help="prior slab centre: 0 for weights, 1 for channel gates")
    ap.add_argument("--grid_max", type=float, default=0.3, help="max radius (use ~1.2 for gates)")
    ap.add_argument("--tag", default="", help="suffix for output filenames (e.g. _cifar)")
    ap.add_argument("--no-png", action="store_true")
    ap.add_argument("--output-dir", type=Path,
                    default=ROOT / "results" / "generated" / "exp04_05" / "figures",
                    help="Directory for newly generated outputs.")
    args = ap.parse_args()
    if args.mode == "trained" and not args.npz:
        ap.error("--npz is required with --mode trained")
    if args.mode == "trained" and not Path(args.npz).is_file():
        ap.error(f"Marginal input does not exist: {args.npz}")

    configure_matplotlib()
    global TAG
    TAG = args.tag
    out = args.output_dir; out.mkdir(parents=True, exist_ok=True)
    prior = SpikeSlabPrior(pi=args.pi, s=args.slab_scale, loc=args.slab_loc)
    fams = synthetic_marginals() if args.mode == "synthetic" else trained_marginals(args.npz)

    radii = np.geomspace(1e-3, args.grid_max, 24)   # geomspace grid (their convention)
    fig_small_ball(fams, prior, radii, out, write_png=not args.no_png)
    fig_directional(fams, prior, radii, out, write_png=not args.no_png)
    rows = write_mi_table(fams, radii, out)

    print(f"mode={args.mode}  prior: spike-slab pi={args.pi} (atom mass {1-args.pi:.2f})")
    print(f"{'family':16s}{'atom':>6s}{'MI_pow':>8s}{'slope':>8s}")
    for name, atom, mid, slope in rows:
        print(f"{name:16s}{atom:>6s}{mid.replace('$','').replace(chr(92)+'infty','inf'):>8s}{slope:>8.3f}")
    print(f"figures + table -> {out}")


if __name__ == "__main__":
    main()
