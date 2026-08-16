"""
SDP_comparePassed-1 — Joint stochastic dynamic programme for two-location lateral transshipment.
Reference: Ma, Rossi & Archibald (2022), equations (1)–(3).

State       (i1, i2)         pre-transshipment inventory at both locations.
Action      (W, Q1, Q2)      transshipment quantity and order quantities, chosen jointly.
Value fn    C[t, i1, i2]     minimum expected total cost from period t.
"""

import numpy as np
from scipy.stats import poisson as _sp_poisson


# ─────────────────────────────────────────────────────────────────────────────
# Shared utility functions (also imported by SDP2stagesSolver.py)
# ─────────────────────────────────────────────────────────────────────────────

def poisson_pmf(lam: float, tail: float = 1e-4) -> np.ndarray:
    """Truncated, renormalised Poisson PMF with mean lam."""
    k_max = int(_sp_poisson.ppf(1.0 - tail, lam))
    k     = np.arange(k_max + 1)
    pmf   = _sp_poisson.pmf(k, lam)
    return pmf / pmf.sum()


def hp(inv: np.ndarray, h: float, b: float) -> np.ndarray:
    """Holding (inv > 0) or penalty (inv < 0) cost."""
    return np.where(inv >= 0, h * inv, -b * inv)


def trans_cost(W: int, R: float, v: float) -> float:
    return (R + v * abs(W)) if W != 0 else 0.0


def order_cost(Q: np.ndarray, K: float, z: float) -> np.ndarray:
    return np.where(Q > 0, K + z * Q, 0.0)


def build_EC(C_next: np.ndarray,
             pmf1: np.ndarray, pmf2: np.ndarray,
             h: float, b: float,
             min_inv: int, N: int) -> np.ndarray:
    """
    Expected-cost matrix EC[s1_idx, s2_idx].

    For each stock-level pair (s1, s2) — where s_j = ĩ_j + Q_j is the
    inventory *before* demand — this gives the expected sum of:
        immediate holding/penalty cost  +  C_next(s1−d1, s2−d2).

    Demand realisations that push either closing inventory outside [0, N-1]
    (the state-space grid) are excluded; the sum is renormalised by the
    total probability mass of the valid realisations.

    Index convention:  inv_value = idx + min_inv,  so idx 0 ↔ min_inv.
    """
    K1, K2   = len(pmf1), len(pmf2)
    inv_vals = np.arange(min_inv, min_inv + N)
    hp_vals  = hp(inv_vals, h, b)          # shape (N,)

    cost_sum = np.zeros((N, N))
    prob_sum = np.zeros((N, N))

    for k1 in range(K1):
        for k2 in range(K2):
            p = pmf1[k1] * pmf2[k2]
            if p < 1e-15:
                continue
            # Closing index for loc j:  c_j = s_j_idx − k_j  ∈ [0, N−1]
            # ⟹  s_j_idx ∈ [k_j, N−1]
            s1_lo, s1_hi = k1, N - 1
            s2_lo, s2_hi = k2, N - 1
            if s1_lo > s1_hi or s2_lo > s2_hi:
                continue
            c1 = np.arange(s1_lo - k1, s1_hi - k1 + 1)   # [0, N−1−k1]
            c2 = np.arange(s2_lo - k2, s2_hi - k2 + 1)   # [0, N−1−k2]

            imm   = hp_vals[c1][:, None] + hp_vals[c2][None, :]   # (n1, n2)
            fut   = C_next[np.ix_(c1, c2)]                         # (n1, n2)

            cost_sum[s1_lo:s1_hi+1, s2_lo:s2_hi+1] += p * (imm + fut)
            prob_sum[s1_lo:s1_hi+1, s2_lo:s2_hi+1] += p

    EC      = np.full((N, N), np.inf)
    valid   = prob_sum > 0
    EC[valid] = cost_sum[valid] / prob_sum[valid]
    return EC


# ─────────────────────────────────────────────────────────────────────────────
# SDP_comparePassed-1 solver
# ─────────────────────────────────────────────────────────────────────────────

class SDP1Solver:
    """
    Backward induction for SDP_comparePassed-1.

    For every (state, period), all feasible (W, Q1, Q2) triples are enumerated
    jointly.  Within each W, the inner optimisation over (Q1, Q2) is fully
    vectorised via numpy.

    Parameters
    ----------
    lam1, lam2  : demand means per period for locations 1 and 2.
    K, z        : fixed and unit ordering cost.
    R, v        : fixed and unit transshipment cost.
    h, b        : holding and penalty cost per unit.
    min_inv     : lower bound on the state-space grid.
    max_inv     : upper bound on the state-space grid.
                  Stock level after ordering is capped at max_inv, matching
                  the LT2Backwards2Stages_slow.java convention.
    tail        : truncation tail for the Poisson PMF.
    """

    def __init__(self, lam1, lam2, K, z, R, v, h, b,
                 min_inv: int, max_inv: int, tail: float = 1e-4):
        assert len(lam1) == len(lam2), "demand series must have equal length"
        self.T         = len(lam1)
        self.K, self.z = K, z
        self.R, self.v = R, v
        self.h, self.b = h, b
        self.min_inv   = min_inv
        self.max_inv   = max_inv
        self.N         = max_inv - min_inv + 1

        self.pmf1 = [poisson_pmf(l, tail) for l in lam1]
        self.pmf2 = [poisson_pmf(l, tail) for l in lam2]

        # C[T] = 0  (terminal boundary condition, Eq. 3)
        self.C    = np.zeros((self.T + 1, self.N, self.N))
        # Optimal action store: axis-3 holds (W, Q1, Q2)
        self.optA = np.zeros((self.T, self.N, self.N, 3), dtype=int)

    # ── solver ───────────────────────────────────────────────────────────── #

    def solve(self, verbose: bool = True) -> "SDP1Solver":
        """Run backward induction; return self for chaining."""
        for t in range(self.T - 1, -1, -1):
            if verbose:
                print(f"  SDP_comparePassed-1 | period {t+1}/{self.T} ...", flush=True)

            # Build EC once per period (shared across all states and W values)
            EC = build_EC(self.C[t + 1], self.pmf1[t], self.pmf2[t],
                          self.h, self.b, self.min_inv, self.N)

            for i1_idx in range(self.N):
                i1 = i1_idx + self.min_inv
                for i2_idx in range(self.N):
                    i2 = i2_idx + self.min_inv

                    best     = np.inf
                    best_act = (0, 0, 0)

                    # Feasibility: W ∈ [min(0,−i2), max(0,i1)]
                    for W in range(min(0, -i2), max(0, i1) + 1):
                        i1p = i1 - W    # post-transshipment inventory
                        i2p = i2 + W

                        # Post-transship state must remain on the grid
                        p1 = i1p - self.min_inv
                        p2 = i2p - self.min_inv
                        if not (0 <= p1 < self.N and 0 <= p2 < self.N):
                            continue

                        tc = trans_cost(W, self.R, self.v)

                        # Stock level = post-transship inv + order quantity
                        # Capped at max_inv so s_idx ∈ [0, N−1]
                        q1_hi = self.max_inv - i1p
                        q2_hi = self.max_inv - i2p
                        if q1_hi < 0 or q2_hi < 0:
                            continue

                        q1 = np.arange(0, q1_hi + 1)
                        q2 = np.arange(0, q2_hi + 1)
                        s1 = p1 + q1    # stock-level indices
                        s2 = p2 + q2

                        # Safety clip (handles numerical edge cases)
                        m1 = s1 < self.N;  q1 = q1[m1];  s1 = s1[m1]
                        m2 = s2 < self.N;  q2 = q2[m2];  s2 = s2[m2]
                        if not (len(q1) and len(q2)):
                            continue

                        oc1 = order_cost(q1, self.K, self.z)   # (nQ1,)
                        oc2 = order_cost(q2, self.K, self.z)   # (nQ2,)

                        # Cost grid: shape (nQ1, nQ2)
                        grid = (tc
                                + oc1[:, None]
                                + oc2[None, :]
                                + EC[np.ix_(s1, s2)])

                        idx = np.argmin(grid)
                        val = grid.flat[idx]
                        if val < best:
                            best = val
                            r, c = np.unravel_index(idx, grid.shape)
                            best_act = (W, int(q1[r]), int(q2[c]))

                    self.C[t, i1_idx, i2_idx]       = best
                    self.optA[t, i1_idx, i2_idx, :] = best_act

        return self

    # ── queries ──────────────────────────────────────────────────────────── #

    def value(self, t: int, i1: int, i2: int) -> float:
        """Expected total cost from period t given pre-transshipment state (i1, i2)."""
        return float(self.C[t, i1 - self.min_inv, i2 - self.min_inv])

    def action(self, t: int, i1: int, i2: int) -> tuple:
        """Optimal (W, Q1, Q2) at period t in state (i1, i2)."""
        a = self.optA[t, i1 - self.min_inv, i2 - self.min_inv]
        return int(a[0]), int(a[1]), int(a[2])


# ─────────────────────────────────────────────────────────────────────────────
# Standalone verification
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Parameters matching the paper's 4-period Poisson test set
    lam1         = [4, 6, 8, 6]
    lam2         = [4, 6, 8, 6]
    K, z         = 20.0, 1.0
    R, v         = 5.0, 0.5
    h, b         = 1.0, 5.0
    min_inv      = -10
    max_inv      = 20

    print("═" * 60)
    print("SDP_comparePassed-1  ·  4-period Poisson example")
    print(f"  lam1 = {lam1}")
    print(f"  lam2 = {lam2}")
    print(f"  K={K}  z={z}  R={R}  v={v}  h={h}  b={b}")
    print(f"  inventory grid : [{min_inv}, {max_inv}]  (N={max_inv-min_inv+1})")
    print("═" * 60)
    print()

    solver = SDP1Solver(lam1, lam2, K, z, R, v, h, b, min_inv, max_inv)
    solver.solve()

    # ── Value function at representative states ──────────────────────── #
    print()
    print("Value function C[t, i1, i2] (expected total cost):")
    states = [(0, 0), (5, 5), (-5, 5), (10, -5), (0, 10)]
    header = f"  {'state':>10}  " + "  ".join(f"{'t='+str(t+1):>10}" for t in range(4))
    print(header)
    print("  " + "-" * (len(header) - 2))
    for (i1, i2) in states:
        vals = "  ".join(f"{solver.value(t, i1, i2):10.4f}" for t in range(4))
        print(f"  ({i1:3d},{i2:3d})     {vals}")

    # ── Optimal actions at (0, 0) ────────────────────────────────────── #
    print()
    print("Optimal action (W, Q1, Q2) and value at state (0, 0):")
    print(f"  {'period':>8}  {'W':>6}  {'Q1':>6}  {'Q2':>6}  {'C value':>12}")
    print("  " + "-" * 48)
    for t in range(4):
        W, Q1, Q2 = solver.action(t, 0, 0)
        print(f"  t={t+1:>6}  {W:>6}  {Q1:>6}  {Q2:>6}  {solver.value(t,0,0):12.4f}")

    # ── Slice of value function for i2 = 0, period 1 ────────────────── #
    print()
    print("C[t=1, i1, i2=0]  (period 1, location-2 inventory fixed at 0):")
    print("  " + "  ".join(f"i1={i:3d} → {solver.value(0, i, 0):8.3f}"
                            for i in range(min_inv, max_inv + 1, 5)))
