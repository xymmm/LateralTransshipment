"""
normal_demand.py
────────────────
Shared primitives for the normal-demand two-location transshipment study.

  1. Discretised normal demand PMF on non-negative integers  (used by the SDPs
     and by the Monte-Carlo simulation, so both face an identical process).
  2. Exact first-order loss function  L(x, mu, sigma)  and its complement
     Lhat(x, mu, sigma)  for a normal variable  (used by the receding-horizon
     refinement step, which needs the exact expected cost of a fixed plan).
  3. Rossi, Tarim, Prestwich & Hnich (2014, AMC 231:489-502) minimax
     piecewise-linear lower-bound coefficients for the standard normal loss
     function  (used by MILP-1).  Table 1 of that paper, keyed by the number
     of partition regions  N.
"""

import numpy as np
from scipy.stats import norm
from scipy.special import ndtr

_INV_SQRT_2PI = 0.3989422804014327

def _pdf(z):
    return _INV_SQRT_2PI * np.exp(-0.5 * z * z)

def _cdf(z):
    return ndtr(z)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Discretised normal demand
# ─────────────────────────────────────────────────────────────────────────────

def normal_pmf(mu: float, sigma: float, tail: float = 1e-6) -> np.ndarray:
    """
    Non-negative integer PMF obtained by discretising N(mu, sigma):
        P(d = k) = Phi((k+0.5 - mu)/sigma) - Phi((k-0.5 - mu)/sigma),  k >= 1,
        P(d = 0) = Phi(( 0.5 - mu)/sigma),
    truncated at the tail quantile and renormalised.  Returns pmf indexed by
    k = 0, 1, ..., k_max.
    """
    k_max = int(np.ceil(mu + sigma * norm.ppf(1.0 - tail)))
    k_max = max(k_max, 1)
    k = np.arange(0, k_max + 1)
    upper = norm.cdf((k + 0.5 - mu) / sigma)
    lower = norm.cdf((k - 0.5 - mu) / sigma)
    lower[0] = 0.0                      # fold the (-inf, 0.5) mass onto d = 0
    pmf = upper - lower
    pmf = np.clip(pmf, 0.0, None)
    return pmf / pmf.sum()


# ─────────────────────────────────────────────────────────────────────────────
# 2. Exact normal loss functions  (Rossi 2014, Lemmas 8-9)
# ─────────────────────────────────────────────────────────────────────────────

def loss_complement(x, mu, sigma):
    """Lhat(x) = E[max(x - D, 0)],  holding side,  D ~ N(mu, sigma)."""
    if sigma <= 0:
        return np.maximum(x - mu, 0.0)
    z = (x - mu) / sigma
    return sigma * (_pdf(z) + z * _cdf(z))


def loss(x, mu, sigma):
    """L(x) = E[max(D - x, 0)],  penalty side,  D ~ N(mu, sigma)."""
    if sigma <= 0:
        return np.maximum(mu - x, 0.0)
    z = (x - mu) / sigma
    return sigma * (_pdf(z) - z * (1.0 - _cdf(z)))


# ─────────────────────────────────────────────────────────────────────────────
# 3. Rossi (2014) Table 1 minimax coefficients for the standard normal
#    lower bound.  N = number of partition regions; the resulting piecewise
#    linear bound has N + 1 segments.  prob[k] = p_k = Pr(Omega_k),
#    cmean[k] = E[Z | Omega_k].  Both are symmetric about zero and
#    sum(prob) = 1,  sum(prob * cmean) = 0.
# ─────────────────────────────────────────────────────────────────────────────

_TABLE1 = {
    1: (np.array([1.0]),
        np.array([0.0])),
    3: (np.array([0.287833, 0.424333, 0.287833]),
        np.array([-1.18505, 0.0, 1.18505])),
    4: (np.array([0.187555, 0.312445, 0.312445, 0.187555]),
        np.array([-1.43535, -0.415223, 0.415223, 1.43535])),
    5: (np.array([0.132411, 0.234913, 0.265353, 0.234913, 0.132411]),
        np.array([-1.61805, -0.691424, 0.0, 0.691424, 1.61805])),
    9: (np.array([0.0503306, 0.0988444, 0.129004, 0.146037, 0.151568,
                  0.146037, 0.129004, 0.0988444, 0.0503306]),
        np.array([-2.05996, -1.30127, -0.8004, -0.384597, 0.0,
                  0.384597, 0.8004, 1.30127, 2.05996])),
    10: (np.array([0.0420611, 0.0836356, 0.110743, 0.127682, 0.135878,
                   0.135878, 0.127682, 0.110743, 0.0836356, 0.0420611]),
         np.array([-2.13399, -1.39768, -0.9182, -0.526575, -0.17199,
                   0.17199, 0.526575, 0.9182, 1.39768, 2.13399])),
}


def rossi_coeffs(n_regions: int):
    """
    Return (P_cum, M_cum) each of length n_regions, where
        P_cum[i] = sum_{k<=i} p_k,
        M_cum[i] = sum_{k<=i} p_k * E[Z|Omega_k].
    The i-th holding lower-bound segment (i = 1..n_regions) is then
        Lhat_lb^i(x; mu, sigma) = P_cum[i]*x - P_cum[i]*mu - M_cum[i]*sigma.
    """
    if n_regions not in _TABLE1:
        raise ValueError(f"no tabulated coefficients for N={n_regions}; "
                         f"available: {sorted(_TABLE1)}")
    p, m = _TABLE1[n_regions]
    return np.cumsum(p), np.cumsum(p * m)


# ─────────────────────────────────────────────────────────────────────────────
# self-check
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # PMF sanity
    pmf = normal_pmf(8.0, 2.0)
    print(f"normal_pmf(8,2): support 0..{len(pmf)-1}, "
          f"sum={pmf.sum():.10f}, mean={np.dot(np.arange(len(pmf)),pmf):.4f}")

    # exact loss vs Table 1 lower bound at a few points (max error ~ 0.0059*sigma
    # for N=10 regions, per Rossi Table 1)
    mu, sigma = 20.0, 5.0
    Pc, Mc = rossi_coeffs(10)
    print("\n x     Lhat_exact   Lhat_lb(N=10)   gap")
    for x in [10, 15, 20, 25, 30]:
        exact = loss_complement(x, mu, sigma)
        segs = Pc * x - Pc * mu - Mc * sigma
        lb = max(0.0, segs.max())
        print(f"{x:3d}    {exact:9.4f}     {lb:9.4f}    {exact-lb:7.4f}")
    print(f"\n Rossi max error bound for N=10 regions: 0.00588597*sigma = "
          f"{0.00588597*sigma:.4f}")
