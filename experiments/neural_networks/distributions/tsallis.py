r"""Tsallis function family f_alpha and the continuous-case forward budget.

Implements Definition "define:Tsallis" of the thesis:

    f_alpha(x) = (x^alpha - alpha*x + (alpha - 1)) / (alpha - 1),   alpha != 1, x >= 0
    f_1(x)     = x*log(x) - x + 1,                                  alpha  = 1, x >  0
    f_alpha(0) = 1   (all alpha)

Key properties (Proposition "proposition:Tsallis") used elsewhere:
  * f_alpha(x) >= 0, with equality iff x = 1 (unique minimiser).
  * f_alpha is strictly convex on (0, inf), decreasing on [0,1), increasing on (1,inf).
  * alpha -> f_alpha(x) is non-decreasing.

The continuous-density forward budget at a finite point (Proposition "theo:both")
is  Gamma(0) = f_alpha(p0 / q0), and it is +inf when q0 = 0 while p0 > 0
(the atom-mismatch case of Theorem "prop:finite").

Everything is plain numpy so the numerical core has no torch dependency and the
self-consistency gates run in milliseconds.
"""

from __future__ import annotations

import numpy as np

__all__ = ["f_alpha", "f_alpha_scalar", "budget_continuous", "INF"]

INF = float("inf")


def f_alpha(x, alpha):
    """Vectorised Tsallis function f_alpha(x).

    Parameters
    ----------
    x : array_like, values in [0, inf].  Negative inputs are invalid (domain is x>=0).
    alpha : float > 0.

    Returns
    -------
    np.ndarray (same broadcast shape as x), values in [0, inf].

    Conventions (extended arithmetic of Remark "define:extended"):
      f_alpha(0) = 1 for every alpha; for alpha > 1, f_alpha(+inf) = +inf;
      for 0 < alpha < 1, x^alpha -> +inf as well so f_alpha(+inf) = +inf only
      through the -alpha*x term -- handled by the limit below.
    """
    x = np.asarray(x, dtype=np.float64)
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {alpha}")
    if np.any(x < 0):
        raise ValueError("f_alpha domain is x >= 0")

    out = np.empty_like(x)
    finite = np.isfinite(x)
    inf_mask = ~finite

    if abs(alpha - 1.0) < 1e-12:
        # f_1(x) = x log x - x + 1, with f_1(0) := 1 (limit x log x -> 0 gives 0-0+1=1).
        xf = x[finite]
        with np.errstate(divide="ignore", invalid="ignore"):
            val = np.where(xf > 0, xf * np.log(xf) - xf + 1.0, 1.0)
        out[finite] = val
        out[inf_mask] = INF  # x log x dominates
    else:
        xf = x[finite]
        val = (np.power(xf, alpha) - alpha * xf + (alpha - 1.0)) / (alpha - 1.0)
        # enforce exact f_alpha(0) = 1 (avoids tiny round-off; (alpha-1)/(alpha-1)=1)
        val = np.where(xf == 0.0, 1.0, val)
        out[finite] = val
        # x -> +inf : for alpha>1, x^alpha dominates -> +inf; for alpha<1,
        # the -alpha*x term dominates and (alpha-1)<0 so ratio -> +inf as well.
        out[inf_mask] = INF

    return out


def f_alpha_scalar(x, alpha):
    """Scalar convenience wrapper returning a Python float."""
    return float(f_alpha(np.asarray(x, dtype=np.float64), alpha))


def budget_continuous(p0, q0, alpha):
    """Forward budget Gamma(0) for continuous p, q with densities p0=p(0), q0=q(0).

    Proposition "theo:both":  Gamma(0) = f_alpha(p0 / q0).
    Theorem "prop:finite" (atom mismatch): if q0 = 0 (q has no density / mass at 0)
    while p0 > 0, the budget is +inf -- a purely continuous q cannot match an atom.

    Returns (gamma, is_inf_flag).
    """
    p0 = float(p0)
    q0 = float(q0)
    if q0 == 0.0:
        if p0 == 0.0:
            # 0/0 is undefined in the extended arithmetic; treat as no constraint.
            return 0.0, False
        return INF, True
    if not np.isfinite(p0):  # p has a log-spike / atom-like density blow-up
        return INF, True
    return f_alpha_scalar(p0 / q0, alpha), False
