"""
sdp_normal.py
─────────────
SDP-1 (joint) and SDP-2 (two-stage) for the two-location lateral-transshipment
lot-sizing problem under NORMAL demand.

The backward-induction core (build_EC and the state/action loops) is the one
verified for the Poisson version; only the demand law changes to a discretised
normal, and the state grid is sized automatically from the demand so that the
grid-exit renormalisation in build_EC is negligible at the states of interest.

Both solvers share build_EC / hp / trans_cost / order_cost from SDP1solver.py.
"""

import numpy as np
from normal_demand import normal_pmf
from SDP1solver import build_EC, hp, trans_cost, order_cost


def auto_grid(mu1, mu2, cv, z_lo=4.0, z_hi=4.0):
    """
    Size the inventory grid from the demand.

    max_inv covers the largest plausible order-up-to level: the maximum over t
    of the remaining cumulative mean demand plus a z_hi-sigma cushion.
    min_inv covers the deepest plausible backorder: a z_lo-sigma cushion below
    zero on the heaviest single-period demand.
    """
    mu1 = np.asarray(mu1, float); mu2 = np.asarray(mu2, float)
    T = len(mu1)
    sig1 = cv * mu1; sig2 = cv * mu2
    # remaining cumulative demand from each t to T, per location
    rem1 = np.array([mu1[t:].sum() for t in range(T)])
    rem2 = np.array([mu2[t:].sum() for t in range(T)])
    rsig1 = np.array([np.sqrt((sig1[t:] ** 2).sum()) for t in range(T)])
    rsig2 = np.array([np.sqrt((sig2[t:] ** 2).sum()) for t in range(T)])
    max_inv = int(np.ceil(max((rem1 + z_hi * rsig1).max(),
                              (rem2 + z_hi * rsig2).max())))
    lo = max((mu1 + z_lo * sig1).max(), (mu2 + z_lo * sig2).max())
    min_inv = -int(np.ceil(lo))
    return min_inv, max_inv


class _NormalSolverBase:
    def __init__(self, mu1, mu2, cv, K, z, R, v, h, b,
                 min_inv=None, max_inv=None, tail=1e-6):
        assert len(mu1) == len(mu2)
        self.T = len(mu1)
        self.mu1 = np.asarray(mu1, float); self.mu2 = np.asarray(mu2, float)
        self.cv = cv
        self.K, self.z = K, z
        self.R, self.v = R, v
        self.h, self.b = h, b
        if min_inv is None or max_inv is None:
            gmin, gmax = auto_grid(mu1, mu2, cv)
            min_inv = gmin if min_inv is None else min_inv
            max_inv = gmax if max_inv is None else max_inv
        self.min_inv, self.max_inv = min_inv, max_inv
        self.N = max_inv - min_inv + 1
        self.pmf1 = [normal_pmf(m, cv * m, tail) for m in self.mu1]
        self.pmf2 = [normal_pmf(m, cv * m, tail) for m in self.mu2]


class SDP1Solver(_NormalSolverBase):
    """Joint backward induction over (W, Q1, Q2)."""

    def solve(self, verbose=False):
        self.C = np.zeros((self.T + 1, self.N, self.N))
        self.optA = np.zeros((self.T, self.N, self.N, 3), dtype=int)
        for t in range(self.T - 1, -1, -1):
            if verbose:
                print(f"  SDP-1 | period {t+1}/{self.T}", flush=True)
            EC = build_EC(self.C[t + 1], self.pmf1[t], self.pmf2[t],
                          self.h, self.b, self.min_inv, self.N)
            for i1i in range(self.N):
                i1 = i1i + self.min_inv
                for i2i in range(self.N):
                    i2 = i2i + self.min_inv
                    best, best_act = np.inf, (0, 0, 0)
                    for W in range(min(0, -i2), max(0, i1) + 1):
                        p1 = i1 - W - self.min_inv
                        p2 = i2 + W - self.min_inv
                        if not (0 <= p1 < self.N and 0 <= p2 < self.N):
                            continue
                        tc = trans_cost(W, self.R, self.v)
                        q1_hi = self.max_inv - (i1 - W)
                        q2_hi = self.max_inv - (i2 + W)
                        if q1_hi < 0 or q2_hi < 0:
                            continue
                        q1 = np.arange(0, q1_hi + 1); q2 = np.arange(0, q2_hi + 1)
                        s1 = p1 + q1; s2 = p2 + q2
                        m1 = s1 < self.N; q1 = q1[m1]; s1 = s1[m1]
                        m2 = s2 < self.N; q2 = q2[m2]; s2 = s2[m2]
                        if not (len(q1) and len(q2)):
                            continue
                        oc1 = order_cost(q1, self.K, self.z)
                        oc2 = order_cost(q2, self.K, self.z)
                        grid = tc + oc1[:, None] + oc2[None, :] + EC[np.ix_(s1, s2)]
                        idx = np.argmin(grid); val = grid.flat[idx]
                        if val < best:
                            best = val
                            r, c = np.unravel_index(idx, grid.shape)
                            best_act = (W, int(q1[r]), int(q2[c]))
                    self.C[t, i1i, i2i] = best
                    self.optA[t, i1i, i2i, :] = best_act
        return self

    def value(self, t, i1, i2):
        return float(self.C[t, i1 - self.min_inv, i2 - self.min_inv])

    def action(self, t, i1, i2):
        a = self.optA[t, i1 - self.min_inv, i2 - self.min_inv]
        return int(a[0]), int(a[1]), int(a[2])


class SDP2Solver(_NormalSolverBase):
    """Two-stage backward induction: order stage then transship stage."""

    def solve(self, verbose=False):
        self.CnOrder = np.zeros((self.T + 1, self.N, self.N))
        self.CnTransship = np.zeros((self.T + 1, self.N, self.N))
        self.optQA = np.zeros((self.T, self.N, self.N), dtype=int)
        self.optQB = np.zeros((self.T, self.N, self.N), dtype=int)
        self.optW = np.zeros((self.T, self.N, self.N), dtype=int)
        for t in range(self.T - 1, -1, -1):
            if verbose:
                print(f"  SDP-2 | period {t+1}/{self.T}", flush=True)
            EC = build_EC(self.CnTransship[t + 1], self.pmf1[t], self.pmf2[t],
                          self.h, self.b, self.min_inv, self.N)
            # stage A: ordering, over post-transship state
            for i1i in range(self.N):
                i1 = i1i + self.min_inv
                for i2i in range(self.N):
                    i2 = i2i + self.min_inv
                    q1_hi = self.max_inv - i1; q2_hi = self.max_inv - i2
                    if q1_hi < 0 or q2_hi < 0:
                        self.CnOrder[t, i1i, i2i] = np.inf; continue
                    q1 = np.arange(0, q1_hi + 1); q2 = np.arange(0, q2_hi + 1)
                    s1 = i1i + q1; s2 = i2i + q2
                    m1 = s1 < self.N; q1 = q1[m1]; s1 = s1[m1]
                    m2 = s2 < self.N; q2 = q2[m2]; s2 = s2[m2]
                    if not (len(q1) and len(q2)):
                        self.CnOrder[t, i1i, i2i] = np.inf; continue
                    oc1 = order_cost(q1, self.K, self.z)
                    oc2 = order_cost(q2, self.K, self.z)
                    grid = oc1[:, None] + oc2[None, :] + EC[np.ix_(s1, s2)]
                    idx = np.argmin(grid)
                    self.CnOrder[t, i1i, i2i] = grid.flat[idx]
                    r, c = np.unravel_index(idx, grid.shape)
                    self.optQA[t, i1i, i2i] = int(q1[r])
                    self.optQB[t, i1i, i2i] = int(q2[c])
            # stage B: transshipment, over pre-transship state
            for i1i in range(self.N):
                i1 = i1i + self.min_inv
                for i2i in range(self.N):
                    i2 = i2i + self.min_inv
                    best, best_W = np.inf, 0
                    for W in range(min(0, -i2), max(0, i1) + 1):
                        p1 = i1i - W; p2 = i2i + W
                        if not (0 <= p1 < self.N and 0 <= p2 < self.N):
                            continue
                        cost = trans_cost(W, self.R, self.v) + self.CnOrder[t, p1, p2]
                        if cost < best:
                            best, best_W = cost, W
                    self.CnTransship[t, i1i, i2i] = best
                    self.optW[t, i1i, i2i] = best_W
        return self

    def value(self, t, i1, i2):
        return float(self.CnTransship[t, i1 - self.min_inv, i2 - self.min_inv])

    def action_W(self, t, i1, i2):
        return int(self.optW[t, i1 - self.min_inv, i2 - self.min_inv])

    def action_Q(self, t, i1, i2):
        r = i1 - self.min_inv; c = i2 - self.min_inv
        return int(self.optQA[t, r, c]), int(self.optQB[t, r, c])
