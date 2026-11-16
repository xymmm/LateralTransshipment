"""
NOTINUSE_guessingOptimalStructure/verify_counterexample.py : independent check of the counterexample to the threshold property.

    python verify_counterexample.py

Instance: two periods, Poisson demand with expected demands (1, 30) at location 1 and (1, 12) at location 2,
K = 57, R = 2, z = 1, v = 0.5, h = 1, b = 20.

The script does not import lt_sdp or any other file of the project. It computes the two-period problem by direct
enumeration: the last period in closed form of the recursion, then, for the post-transshipment states (x1, X2) of
period 1, the cheapest reorder decision with Q1 = 0 and the cheapest with Q1 > 0. It does this twice: with the
demand truncated at the 1 - 1e-4 quantile and renormalised, as in the manuscript, and with the tail cut at 1e-12.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import poisson

K, R, z, v, h, b = 57.0, 2.0, 1.0, 0.5, 1.0, 20.0
MU = {1: (1, 30), 2: (1, 12)}            # location -> expected demand in periods 1 and 2
LO, HI = -25, 160                        # inventory levels kept
X2 = 3                                   # inventory of location 2 after the transshipment
OFF = -LO


def pmf(mean, tail):
    m = int(poisson.ppf(1 - tail, mean))
    k = np.arange(m + 1)
    return poisson.pmf(k, mean) / poisson.cdf(m, mean)


def loss(p):
    """l(y) = E[h (y - d)^+ + b (d - y)^+] for y = LO..HI."""
    y = np.arange(LO, HI + 1)[:, None]
    d = np.arange(len(p))[None, :]
    return (np.maximum(y - d, 0) * h + np.maximum(d - y, 0) * b) @ p


def solve(tail):
    lev = np.arange(LO, HI + 1)
    n = len(lev)
    p = {(j, t): pmf(MU[j][t - 1], tail) for j in (1, 2) for t in (1, 2)}
    # ---- period 2, second stage: one (s, S) problem per location
    c = {}
    for j in (1, 2):
        f = z * lev + loss(p[(j, 2)])
        above = np.full(n, np.inf)
        above[:-1] = np.minimum.accumulate(f[::-1])[::-1][1:]          # min of f over levels strictly above
        c[j] = np.minimum(f, K + above) - z * lev
    Cq2 = c[1][:, None] + c[2][None, :]
    # ---- period 2, first stage, for the opening states that period 1 can reach
    lo2, hi2 = -20, 75
    C2 = np.full((hi2 - lo2 + 1, hi2 - lo2 + 1), np.inf)
    for i1 in range(lo2, hi2 + 1):
        for i2 in range(lo2, hi2 + 1):
            W = np.arange(-max(i2, 0), max(i1, 0) + 1)
            cost = np.where(W != 0, R + v * np.abs(W), 0.0) + Cq2[i1 - W + OFF, i2 + W + OFF]
            C2[i1 - lo2, i2 - lo2] = cost.min()
    # ---- period 1: expected cost after reordering up to (y1, y2)
    ymax = 70
    l1, l2 = loss(p[(1, 1)]), loss(p[(2, 1)])
    p1, p2 = p[(1, 1)], p[(2, 1)]

    def after(y1, y2):
        e = 0.0
        for d1, q1 in enumerate(p1):
            for d2, q2 in enumerate(p2):
                e += q1 * q2 * C2[y1 - d1 - lo2, y2 - d2 - lo2]
        return l1[y1 + OFF] + l2[y2 + OFF] + e

    A = {(y1, y2): after(y1, y2) for y1 in range(-3, ymax + 1) for y2 in range(X2, ymax + 1)}
    rows = []
    x2 = X2
    for x1 in range(-3, 13):
        no = min((K * (y2 > x2) + z * (y2 - x2) + A[(x1, y2)], 0, y2 - x2) for y2 in range(x2, ymax + 1))
        yes = min((K + K * (y2 > x2) + z * (y1 - x1) + z * (y2 - x2) + A[(y1, y2)], y1 - x1, y2 - x2)
                  for y1 in range(x1 + 1, ymax + 1) for y2 in range(x2, ymax + 1))
        rows.append((x1, no, yes))
    return rows


if __name__ == "__main__":
    for tail, label in ((1e-4, "demand truncated at the 1 - 1e-4 quantile, as in the manuscript"),
                        (1e-12, "demand tail cut at 1e-12")):
        print(label)
        print("  state      cost without reorder at 1 (Q1, Q2)     cost with reorder at 1 (Q1, Q2)     difference   location 1")
        for x1, no, yes in solve(tail):
            diff = no[0] - yes[0]
            print(f"  ({x1:3d}, {X2})   {no[0]:10.4f}  ({no[1]:2d}, {no[2]:2d})                  {yes[0]:10.4f}  ({yes[1]:2d}, {yes[2]:2d})"
                  f"               {diff:8.4f}   {'reorders' if diff > 1e-9 else 'does not reorder'}")
        print()
