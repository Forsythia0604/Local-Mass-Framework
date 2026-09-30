r"""Forward/reverse budgets and the budget<->KL recovery gate (thesis Ch. 4-5).

Forward budget at a finite point (Definition "define:budget", Prop "theo:both"):

    Gamma(0)  = lim_{r->0} D_alpha(p || q; B_r(0)) / q(B_r(0))   = f_alpha(p0/q0),
    Gamma*(0) = lim_{r->0} D_alpha(q || p; B_r(0)) / p(B_r(0))   = f_alpha(q0/p0),

for continuous p, q with densities p0=p(0), q0=q(0).  The atom-mismatch rule of
Theorem "prop:finite": if the prior p has an atom at 0 (p({0})>0) and q has none,
then Gamma(0) = inf -- a purely continuous q cannot preserve the prior's atom.

The TKL divergence at alpha=1 recovers KL on the whole space (Remark at l.1809):

    D_1(q||p; R^d) = \int f_1(dq/dp) dp = KL(q||p),

which is the implementation sanity check the thesis itself asks for (Exp. 1,
l.2243/2310).  `integrated_budget_recovers_kl` verifies this numerically for
Gaussians, and also checks the closed-form continuous budget f_alpha(p0/q0).

Pure numpy.
"""

from __future__ import annotations

import math

import numpy as np

from distributions.tsallis import INF, f_alpha_scalar

__all__ = ["forward_budget", "reverse_budget", "integrated_budget_recovers_kl"]


def forward_budget(p0, q0, alpha, atom_p=0.0, atom_q=0.0):
    """Forward budget Gamma(0). Returns (value, is_inf_flag).

    Three regimes (Definition "define:budget", Thm "prop:finite"):
      * prior atom, q continuous  (atom_p>0, atom_q=0)  -> +inf   [the E1 negative case]
      * matching atoms            (atom_p>0, atom_q>0)  -> finite, f_alpha(atom_p/atom_q)
        (near 0 the small-ball mass is dominated by the atoms; the budget is the
         Tsallis value of the atom-mass ratio -- finite because both atoms are > 0)
      * continuous-continuous     (atom_p=0, atom_q=0)  -> f_alpha(p0/q0)
    """
    p_has_atom = atom_p > 0.0
    q_has_atom = atom_q > 0.0
    if p_has_atom and not q_has_atom:
        return INF, True
    if p_has_atom and q_has_atom:
        return f_alpha_scalar(atom_p / atom_q, alpha), False
    # continuous-continuous
    if q0 == 0.0:
        return (INF, True) if p0 > 0.0 else (0.0, False)
    if not np.isfinite(p0):
        return INF, True
    return f_alpha_scalar(p0 / q0, alpha), False


def reverse_budget(p0, q0, alpha, atom_p=0.0, atom_q=0.0):
    """Reverse budget Gamma*(0) = f_alpha(q0/p0) (atom-mass ratio in the atom case)."""
    p_has_atom = atom_p > 0.0
    q_has_atom = atom_q > 0.0
    if q_has_atom and not p_has_atom:
        return INF, True
    if p_has_atom and q_has_atom:
        return f_alpha_scalar(atom_q / atom_p, alpha), False
    if p0 == 0.0:
        return (INF, True) if q0 > 0.0 else (0.0, False)
    if not np.isfinite(q0):
        return INF, True
    return f_alpha_scalar(q0 / p0, alpha), False


# ---------------------------------------------------------------------------
# Analytic + numeric Gaussian helpers for the recovery gate
# ---------------------------------------------------------------------------
def _gauss_logpdf(x, mu, s):
    return -0.5 * math.log(2 * math.pi) - math.log(s) - 0.5 * ((x - mu) / s) ** 2


def _kl_gauss(mu_q, s_q, mu_p, s_p):
    """Closed-form KL(N_q || N_p) for 1-D Gaussians."""
    return (
        math.log(s_p / s_q)
        + (s_q ** 2 + (mu_q - mu_p) ** 2) / (2 * s_p ** 2)
        - 0.5
    )


def _f1(x):
    """Tsallis f_1(x) = x log x - x + 1, vectorised, f_1(0)=1."""
    x = np.asarray(x, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.where(x > 0, x * np.log(x) - x + 1.0, 1.0)
    return v


def integrated_budget_recovers_kl(verbose=True):
    r"""Gate 5: numeric TKL (alpha=1) integral recovers analytic KL, both directions,
    and the closed-form continuous budget matches f_alpha(p0/q0).

    Uses two 1-D Gaussians q=N(mu_q,s_q), p=N(mu_p,s_p).  The RN derivative is
    R^q_p(x) = q(x)/p(x) and  D_1(q||p;R) = \int f_1(q/p) p dx  should equal KL(q||p).
    """
    mu_q, s_q = 0.10, 0.80
    mu_p, s_p = 0.00, 1.00

    # fine grid covering both densities
    xs = np.linspace(-12.0, 12.0, 400_001)
    dx = xs[1] - xs[0]
    qx = np.exp(-0.5 * math.log(2 * math.pi) - math.log(s_q) - 0.5 * ((xs - mu_q) / s_q) ** 2)
    px = np.exp(-0.5 * math.log(2 * math.pi) - math.log(s_p) - 0.5 * ((xs - mu_p) / s_p) ** 2)

    # D_1(q||p; R) = \int f_1(q/p) p dx
    R_qp = qx / px
    D1_qp = np.sum(_f1(R_qp) * px) * dx
    KL_qp = _kl_gauss(mu_q, s_q, mu_p, s_p)

    # D_1(p||q; R) = \int f_1(p/q) q dx  (forward direction)
    R_pq = px / qx
    D1_pq = np.sum(_f1(R_pq) * qx) * dx
    KL_pq = _kl_gauss(mu_p, s_p, mu_q, s_q)

    err_qp = abs(D1_qp - KL_qp) / abs(KL_qp)
    err_pq = abs(D1_pq - KL_pq) / abs(KL_pq)

    # closed-form continuous budget at 0
    p0 = math.exp(_gauss_logpdf(0.0, mu_p, s_p))
    q0 = math.exp(_gauss_logpdf(0.0, mu_q, s_q))
    g_fwd, _ = forward_budget(p0, q0, alpha=1.0)
    g_fwd_ref = f_alpha_scalar(p0 / q0, 1.0)
    err_budget = abs(g_fwd - g_fwd_ref)

    # atom-mismatch sanity: prior atom, continuous q -> inf
    g_atom, is_inf = forward_budget(p0=p0, q0=q0, alpha=1.0, atom_p=0.5, atom_q=0.0)

    ok = (err_qp < 1e-3) and (err_pq < 1e-3) and (err_budget < 1e-9) and is_inf
    if verbose:
        print(f"  D1(q||p)={D1_qp:.6f}  KL(q||p)={KL_qp:.6f}  rel_err={err_qp:.2e}")
        print(f"  D1(p||q)={D1_pq:.6f}  KL(p||q)={KL_pq:.6f}  rel_err={err_pq:.2e}")
        print(f"  continuous budget f_1(p0/q0)={g_fwd:.6f}  (err={err_budget:.1e})")
        print(f"  atom-mismatch forward budget is +inf: {is_inf}")
        print(f"  [{'PASS' if ok else 'FAIL'}] budget<->KL recovery")
    return {
        "D1_qp": D1_qp, "KL_qp": KL_qp, "rel_err_qp": err_qp,
        "D1_pq": D1_pq, "KL_pq": KL_pq, "rel_err_pq": err_pq,
        "budget_err": err_budget, "atom_is_inf": is_inf, "pass": ok,
    }


if __name__ == "__main__":
    integrated_budget_recovers_kl()
