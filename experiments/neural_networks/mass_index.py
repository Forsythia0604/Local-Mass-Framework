r"""Empirical Mass-Index estimator (thesis "fast-read" theorem theo:fast read).

Given samples from a measure, we estimate how the small-ball mass grows,

    p(B_r(0)) ~ r^alpha (log(1/r))^beta   as r -> 0,

and read off the Mass Indices (k = 1 at a finite point 0, k = 0 at infinity;
d = ambient/group dimension):

    MI1(p,0) = d^k / alpha,        MI2(p,0) = d^k * beta.

Discriminating an *atom* (MI1 = inf) from a continuous density (MI1 = 1) or a
log-spike (horseshoe: MI1 = 1, MI2 = 1, *no* atom) is the crux:

  * An atom at 0 makes p(B_r(0)) -> p({0}) > 0 as r -> 0, i.e. a *plateau* on the
    log-log curve (slope alpha -> 0).  We detect it primarily by the fraction of
    (near-)exact zeros, and secondarily by a vanishing fitted slope.
  * A horseshoe log-spike has density -> inf but mass -> 0 (slope ~ 1); it must
    NOT be misread as an atom.  The (log 1/r)^beta term in the fit recovers
    MI2 ~ 1 here, and the "plateau" test guards against a false atom.

Pure numpy: no torch, no training -- the self-consistency gates run in ms.
"""

from __future__ import annotations

import numpy as np

from distributions.tsallis import INF

__all__ = [
    "empirical_ball_mass",
    "estimate_mi",
    "self_consistency_tests",
]

_EPS_ZERO = 1e-10      # |theta| <= this counts as an exact zero (literal atom)
_ATOM_FLOOR = 5e-3     # exact-zero fraction above this => declare an atom
_SLOPE_FLOOR = 0.15    # fitted alpha below this (with bounded mass) => atom-like


def empirical_ball_mass(radii_samples, radii):
    """p_hat(B_r(0)) = fraction of |theta| (or ||theta_group||) <= r, for each r.

    radii_samples : 1-D array of non-negative radii (|theta| in 1-D; group norms otherwise).
    radii         : 1-D array of query radii r.
    Returns array of the same length as `radii`.
    """
    s = np.sort(np.asarray(radii_samples, dtype=np.float64))
    r = np.asarray(radii, dtype=np.float64)
    # number of samples <= r  via searchsorted on the sorted array
    counts = np.searchsorted(s, r, side="right")
    return counts / s.size


def _build_radii_grid(radii_samples, n_radii=24, lo_q=0.02, hi_q=0.35, r_cap=0.36):
    """Geometric grid of query radii based on the *nonzero* radius scale.

    Uses quantiles of the strictly-positive radii so that families with a large
    exact-zero atom still get a sensible grid over their continuous (slab) part.
    Caps r_max below 1/e so that log(1/r) > 1 for the MI2 (log-correction) fit.
    """
    s = np.asarray(radii_samples, dtype=np.float64)
    pos = s[s > _EPS_ZERO]
    if pos.size < 50:
        return None
    r_min = max(np.quantile(pos, lo_q), 1e-6)
    r_max = min(np.quantile(pos, hi_q), r_cap)
    if not (r_max > r_min):
        return None
    return np.geomspace(r_min, r_max, n_radii)


def estimate_mi(radii_samples, d=1, k=1, with_log=True, n_radii=24):
    """Estimate the Mass Index at 0 from samples.

    Parameters
    ----------
    radii_samples : 1-D array of |theta| (1-D coords) or ||theta_group|| (groups).
    d : ambient/group dimension (the d in d^k).
    k : 1 at a finite point (here 0), 0 at infinity.
    with_log : also fit the (log 1/r)^beta term to recover MI2.

    Returns dict: alpha_hat, beta_hat, mi1_hat, mi2_hat, atom_detected,
                  atom_mass_hat, radii (list), ball_mass (list).
    """
    s = np.asarray(radii_samples, dtype=np.float64)
    n = s.size
    dk = float(d) ** k

    exact_zero_frac = float(np.mean(s <= _EPS_ZERO))

    # --- literal-atom fast path ---------------------------------------------
    if exact_zero_frac > _ATOM_FLOOR:
        return {
            "alpha_hat": 0.0,
            "beta_hat": float("nan"),
            "mi1_hat": INF,
            "mi2_hat": float("nan"),
            "atom_detected": True,
            "atom_mass_hat": exact_zero_frac,
            "radii": [],
            "ball_mass": [],
        }

    grid = _build_radii_grid(s, n_radii=n_radii)
    if grid is None:
        return {
            "alpha_hat": float("nan"), "beta_hat": float("nan"),
            "mi1_hat": float("nan"), "mi2_hat": float("nan"),
            "atom_detected": False, "atom_mass_hat": exact_zero_frac,
            "radii": [], "ball_mass": [],
        }

    mass = empirical_ball_mass(s, grid)
    good = mass > 0
    grid, mass = grid[good], mass[good]
    if grid.size < 4:
        return {
            "alpha_hat": float("nan"), "beta_hat": float("nan"),
            "mi1_hat": float("nan"), "mi2_hat": float("nan"),
            "atom_detected": False, "atom_mass_hat": exact_zero_frac,
            "radii": grid.tolist(), "ball_mass": mass.tolist(),
        }

    logr = np.log(grid)
    logm = np.log(mass)
    w = n * mass  # ~ point count in B_r : variance-stabilising weight

    # weighted design: [1, log r] and optionally [.., log(log(1/r))]
    cols = [np.ones_like(logr), logr]
    use_log = with_log and np.all(grid < np.exp(-1.0))
    if use_log:
        loglog = np.log(np.log(1.0 / grid))
        cols.append(loglog)
    X = np.vstack(cols).T
    W = np.diag(w)
    # solve (X' W X) b = X' W y
    XtW = X.T @ W
    beta_coef, *_ = np.linalg.lstsq(XtW @ X, XtW @ logm, rcond=None)

    alpha_hat = float(beta_coef[1])
    beta_log = float(beta_coef[2]) if use_log else 0.0

    # --- soft-plateau guard: vanishing slope with bounded mass => atom-like ---
    atom_like = (alpha_hat < _SLOPE_FLOOR) and (mass.min() > _ATOM_FLOOR)
    if atom_like:
        return {
            "alpha_hat": alpha_hat, "beta_hat": beta_log,
            "mi1_hat": INF, "mi2_hat": float("nan"),
            "atom_detected": True, "atom_mass_hat": float(mass.min()),
            "radii": grid.tolist(), "ball_mass": mass.tolist(),
        }

    mi1 = dk / alpha_hat if alpha_hat > 1e-9 else INF
    mi2 = dk * beta_log
    return {
        "alpha_hat": alpha_hat,
        "beta_hat": beta_log,
        "mi1_hat": float(mi1),
        "mi2_hat": float(mi2),
        "atom_detected": False,
        "atom_mass_hat": exact_zero_frac,
        "radii": grid.tolist(),
        "ball_mass": mass.tolist(),
    }


# ===========================================================================
#  Self-consistency gates (thesis-anchored)
# ===========================================================================

def _gaussian_radii(n, rng):
    return np.abs(rng.standard_normal(n))


def _spike_slab_radii(n, pi, rng):
    """(1-pi) delta_0 + pi N(0,1): exact zeros for the spike."""
    keep = rng.random(n) < pi
    x = np.zeros(n)
    x[keep] = rng.standard_normal(keep.sum())
    return np.abs(x)


def _radial_power_radii(n, d, beta, R, rng):
    """Sample ||x|| for rho(x) ∝ ||x||^{-beta} on the ball of radius R in R^d.

    Radial CDF P(||x|| <= r) ∝ r^{d-beta}, so ||x|| = R * U^{1/(d-beta)}.
    Proper iff 0 <= beta < d; then MI1 = d/(d-beta).
    """
    u = rng.random(n)
    return R * u ** (1.0 / (d - beta))


def _horseshoe_radii(n, rng):
    """Horseshoe marginal via the GSM definition: x|lam ~ N(0, lam^2), lam ~ C^+(1).

    Half-Cauchy lam = |tan(pi/2 * U)|.  Density has a log-spike at 0 (MI1=1, MI2=1)
    but NO atom -- the estimator must not flag an atom here.
    """
    u = rng.random(n)
    lam = np.abs(np.tan(np.pi / 2.0 * u))
    x = lam * rng.standard_normal(n)
    return np.abs(x)


def self_consistency_tests(n=400_000, seed=0, verbose=True):
    """Run the five thesis-anchored estimator gates; return a pass/fail dict.

    1. N(0,1)                 -> MI1 ~ 1,            no atom.
    2. spike-and-slab pi=0.3  -> atom, MI1 = inf,   atom_mass ~ 0.7.
    3. radial power-law       -> MI1 ~ d/(d-beta).
    4. horseshoe              -> MI1 ~ 1, MI2 ~ 1,  NO false atom.
    """
    rng = np.random.default_rng(seed)
    results = {}

    # 1. Gaussian
    r = estimate_mi(_gaussian_radii(n, rng), d=1, k=1)
    results["gaussian_MI1"] = {
        "mi1_hat": r["mi1_hat"], "atom": r["atom_detected"],
        "pass": (not r["atom_detected"]) and abs(r["mi1_hat"] - 1.0) < 0.12,
    }

    # 2. spike-and-slab
    r = estimate_mi(_spike_slab_radii(n, 0.3, rng), d=1, k=1)
    results["spikeslab_atom"] = {
        "mi1_hat": r["mi1_hat"], "atom": r["atom_detected"],
        "atom_mass_hat": r["atom_mass_hat"],
        "pass": r["atom_detected"] and (r["mi1_hat"] == INF)
                and abs(r["atom_mass_hat"] - 0.7) < 0.03,
    }

    # 3. radial power-law, a few (d, beta)
    rp = {}
    ok = True
    for d, beta in [(2, 0.5), (3, 1.0), (3, 2.0)]:
        r = estimate_mi(_radial_power_radii(n, d, beta, R=1.0, rng=rng), d=d, k=1)
        target = d / (d - beta)
        rel = abs(r["mi1_hat"] - target) / target
        rp[f"d{d}_b{beta}"] = {"mi1_hat": r["mi1_hat"], "target": target, "rel": rel}
        ok = ok and (rel < 0.12) and (not r["atom_detected"])
    results["radial_power"] = {"detail": rp, "pass": ok}

    # 4. horseshoe
    r = estimate_mi(_horseshoe_radii(n, rng), d=1, k=1)
    results["horseshoe_noatom"] = {
        "mi1_hat": r["mi1_hat"], "mi2_hat": r["mi2_hat"], "atom": r["atom_detected"],
        "pass": (not r["atom_detected"]) and abs(r["mi1_hat"] - 1.0) < 0.2
                and (r["mi2_hat"] > 0.4),
    }

    results["all_pass"] = all(v["pass"] for k_, v in results.items() if isinstance(v, dict) and "pass" in v)

    if verbose:
        for name, v in results.items():
            if name == "all_pass":
                continue
            print(f"  [{'PASS' if v['pass'] else 'FAIL'}] {name}: "
                  + ", ".join(f"{kk}={vv}" for kk, vv in v.items() if kk not in ("pass", "detail")))
            if "detail" in v:
                for dk_, dv in v["detail"].items():
                    print(f"        {dk_}: MI1={dv['mi1_hat']:.3f} target={dv['target']:.3f} rel={dv['rel']:.3f}")
        print(f"  ALL PASS: {results['all_pass']}")
    return results


if __name__ == "__main__":
    self_consistency_tests()
