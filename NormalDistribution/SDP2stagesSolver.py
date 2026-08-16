"""
SDP_comparePassed-2 — Two-stage backward induction for two-location lateral transshipment.
Reference: Ma, Rossi & Archibald (2022), equations (4)–(7).

Within each period t the backward recursion runs two sub-stages:

  Stage A  (ordering, computed first in backward induction)
    CnOrder[t, i1_idx, i2_idx]
        Minimum expected cost from the ordering sub-stage of period t,
        given the POST-transshipment inventory (ĩ1, ĩ2).
        Future cost = CnTransship[t+1] (next period's start-of-period value).

  Stage B  (transshipment, computed second)
    CnTransship[t, i1_idx, i2_idx]
        Minimum expected cost from the START of period t,
        given the PRE-transshipment inventory (i1, i2).
        Future cost within this period = CnOrder[t] at the post-transship state.

Physical timeline inside period t:
    [start] → transship W → order (Q1,Q2) → demand realised → [period t+1]

Equivalence with SDP_comparePassed-1:
    C_SDP1[t] = CnTransship[t]  for all t  (proved by backward induction;
    both value functions satisfy the same Bellman equation with the same
    terminal condition).  The two formulations differ only in the order in
    which the joint minimum over (W, Q1, Q2) is evaluated.
"""

import numpy as np
from SDP1solver import (poisson_pmf, trans_cost, order_cost,
                         build_EC, SDP1Solver)


# ─────────────────────────────────────────────────────────────────────────────
# SDP_comparePassed-2 solver
# ─────────────────────────────────────────────────────────────────────────────

class SDP2Solver:
    """
    Two-stage backward induction for SDP_comparePassed-2.

    Parameters: same as SDP1Solver.
    """

    def __init__(self, lam1, lam2, K, z, R, v, h, b,
                 min_inv: int, max_inv: int, tail: float = 1e-4):
        assert len(lam1) == len(lam2)
        self.T         = len(lam1)
        self.K, self.z = K, z
        self.R, self.v = R, v
        self.h, self.b = h, b
        self.min_inv   = min_inv
        self.max_inv   = max_inv
        self.N         = max_inv - min_inv + 1

        self.pmf1 = [poisson_pmf(l, tail) for l in lam1]
        self.pmf2 = [poisson_pmf(l, tail) for l in lam2]

        # Terminal boundary: both value functions = 0 at t = T  (Eq. 7)
        self.CnOrder     = np.zeros((self.T + 1, self.N, self.N))
        self.CnTransship = np.zeros((self.T + 1, self.N, self.N))

        # Optimal decisions (stored for inspection)
        self.optQA = np.zeros((self.T, self.N, self.N), dtype=int)
        self.optQB = np.zeros((self.T, self.N, self.N), dtype=int)
        self.optW  = np.zeros((self.T, self.N, self.N), dtype=int)

    # ── solver ───────────────────────────────────────────────────────────── #

    def solve(self, verbose: bool = True) -> "SDP2Solver":
        """Run two-stage backward induction; return self."""
        for t in range(self.T - 1, -1, -1):
            if verbose:
                print(f"  SDP_comparePassed-2 | period {t+1}/{self.T} ...", flush=True)

            # ── Stage A: ordering ─────────────────────────────────────────
            # EC is built using CnTransship[t+1] as future cost, which is the
            # start-of-period value of the NEXT period (after its transshipment
            # decision has been made).
            EC = build_EC(self.CnTransship[t + 1], self.pmf1[t], self.pmf2[t],
                          self.h, self.b, self.min_inv, self.N)

            for i1_idx in range(self.N):
                i1 = i1_idx + self.min_inv
                for i2_idx in range(self.N):
                    i2 = i2_idx + self.min_inv

                    q1_hi = self.max_inv - i1   # stock level capped at max_inv
                    q2_hi = self.max_inv - i2
                    if q1_hi < 0 or q2_hi < 0:
                        self.CnOrder[t, i1_idx, i2_idx] = np.inf
                        continue

                    q1 = np.arange(0, q1_hi + 1)
                    q2 = np.arange(0, q2_hi + 1)
                    s1 = i1_idx + q1    # stock-level indices
                    s2 = i2_idx + q2

                    m1 = s1 < self.N;  q1 = q1[m1];  s1 = s1[m1]
                    m2 = s2 < self.N;  q2 = q2[m2];  s2 = s2[m2]
                    if not (len(q1) and len(q2)):
                        self.CnOrder[t, i1_idx, i2_idx] = np.inf
                        continue

                    oc1 = order_cost(q1, self.K, self.z)   # (nQ1,)
                    oc2 = order_cost(q2, self.K, self.z)   # (nQ2,)

                    # Cost grid over (Q1, Q2): shape (nQ1, nQ2)
                    grid = oc1[:, None] + oc2[None, :] + EC[np.ix_(s1, s2)]

                    idx = np.argmin(grid)
                    self.CnOrder[t, i1_idx, i2_idx] = grid.flat[idx]
                    r, c = np.unravel_index(idx, grid.shape)
                    self.optQA[t, i1_idx, i2_idx] = int(q1[r])
                    self.optQB[t, i1_idx, i2_idx] = int(q2[c])

            # ── Stage B: transshipment ────────────────────────────────────
            # For each pre-transshipment state, the future cost is CnOrder[t]
            # evaluated at the post-transshipment state (ĩ1, ĩ2).
            for i1_idx in range(self.N):
                i1 = i1_idx + self.min_inv
                for i2_idx in range(self.N):
                    i2 = i2_idx + self.min_inv

                    best   = np.inf
                    best_W = 0

                    for W in range(min(0, -i2), max(0, i1) + 1):
                        p1 = i1_idx - W    # post-transship indices
                        p2 = i2_idx + W
                        if not (0 <= p1 < self.N and 0 <= p2 < self.N):
                            continue
                        cost = trans_cost(W, self.R, self.v) + self.CnOrder[t, p1, p2]
                        if cost < best:
                            best   = cost
                            best_W = W

                    self.CnTransship[t, i1_idx, i2_idx] = best
                    self.optW[t, i1_idx, i2_idx]         = best_W

        return self

    # ── queries ──────────────────────────────────────────────────────────── #

    def value(self, t: int, i1: int, i2: int) -> float:
        """Expected total cost from period t, pre-transshipment state (i1, i2)."""
        return float(self.CnTransship[t, i1 - self.min_inv, i2 - self.min_inv])

    def order_value(self, t: int, i1: int, i2: int) -> float:
        """CnOrder at period t, post-transshipment state (i1, i2)."""
        return float(self.CnOrder[t, i1 - self.min_inv, i2 - self.min_inv])

    def action_W(self, t: int, i1: int, i2: int) -> int:
        return int(self.optW[t, i1 - self.min_inv, i2 - self.min_inv])

    def action_Q(self, t: int, i1: int, i2: int) -> tuple:
        r = i1 - self.min_inv;  c = i2 - self.min_inv
        return int(self.optQA[t, r, c]), int(self.optQB[t, r, c])


# ─────────────────────────────────────────────────────────────────────────────
# Standalone run: solve both SDPs and compare
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    lam1         = [4, 6, 8, 6]
    lam2         = [4, 6, 8, 6]
    K, z         = 20.0, 1.0
    R, v         = 5.0,  0.5
    h, b         = 1.0,  5.0
    min_inv      = -10
    max_inv      = 20

    print("═" * 64)
    print("SDP_comparePassed-1 vs SDP_comparePassed-2  ·  4-period Poisson example")
    print(f"  lam1 = {lam1}  lam2 = {lam2}")
    print(f"  K={K}  z={z}  R={R}  v={v}  h={h}  b={b}")
    print(f"  inventory grid : [{min_inv}, {max_inv}]  (N={max_inv-min_inv+1})")
    print("═" * 64)

    print("\nSolving SDP_comparePassed-2 ...")
    s2 = SDP2Solver(lam1, lam2, K, z, R, v, h, b, min_inv, max_inv)
    s2.solve()

    print("\nSolving SDP_comparePassed-1 ...")
    s1 = SDP1Solver(lam1, lam2, K, z, R, v, h, b, min_inv, max_inv)
    s1.solve()

    # ── Equivalence check ─────────────────────────────────────────────── #
    print()
    print("═" * 64)
    print("EQUIVALENCE CHECK  |C_SDP1[t] − CnTransship_SDP2[t]|")
    print("═" * 64)
    all_close = True
    for t in range(4):
        diff = np.abs(s1.C[t] - s2.CnTransship[t])
        flag = "  ✓" if diff.max() < 1e-8 else "  ✗"
        print(f"  period {t+1}:  max = {diff.max():.3e}"
              f"   mean = {diff.mean():.3e}"
              f"   99th pct = {np.percentile(diff, 99):.3e}{flag}")
        if diff.max() >= 1e-8:
            all_close = False
    print()
    if all_close:
        print("  → Value functions are numerically identical. Equivalence confirmed.")
    else:
        print("  → Differences exceed numerical tolerance. Investigate further.")

    # ── Value function comparison at representative states ─────────────── #
    check_states = [(0, 0), (5, 5), (-5, 5), (10, -5), (0, 10), (-5, -5)]
    print()
    print("Value function at representative states (period 1 = t=0):")
    print(f"  {'state':>10}  {'SDP_comparePassed-1':>12}  {'SDP_comparePassed-2':>12}  {'|diff|':>10}")
    print("  " + "-" * 50)
    for (i1, i2) in check_states:
        if not (min_inv <= i1 <= max_inv and min_inv <= i2 <= max_inv):
            continue
        v1 = s1.value(0, i1, i2)
        v2 = s2.value(0, i1, i2)
        print(f"  ({i1:3d},{i2:3d})     {v1:12.6f}  {v2:12.6f}  {abs(v1-v2):10.2e}")

    # ── Value function across all four periods at (0, 0) ─────────────── #
    print()
    print("Value function C[t, 0, 0] across all periods:")
    print(f"  {'period':>8}  {'SDP_comparePassed-1 C[t]':>14}  {'SDP_comparePassed-2 Transship':>16}"
          f"  {'SDP_comparePassed-2 Order':>13}  {'|diff|':>10}")
    print("  " + "-" * 72)
    for t in range(4):
        v1   = s1.value(t, 0, 0)
        v2t  = s2.value(t, 0, 0)          # CnTransship
        v2o  = s2.order_value(t, 0, 0)    # CnOrder at same state
        print(f"  t={t+1:>6}  {v1:14.6f}  {v2t:16.6f}  {v2o:13.6f}  {abs(v1-v2t):10.2e}")

    # ── Optimal actions at (0, 0) ─────────────────────────────────────── #
    print()
    print("Optimal actions at state (0, 0):")
    print(f"  {'period':>8}  {'SDP_comparePassed-1 (W,Q1,Q2)':>18}  {'SDP_comparePassed-2 (W,Q1,Q2)':>18}")
    print("  " + "-" * 50)
    for t in range(4):
        W1, Q1a, Q1b = s1.action(t, 0, 0)
        W2            = s2.action_W(t, 0, 0)
        # Post-transship state for SDP_comparePassed-2 action query
        i1p = 0 - W2;  i2p = 0 + W2
        Q2a, Q2b = s2.action_Q(t, i1p, i2p)
        print(f"  t={t+1:>6}    ({W1:3d}, {Q1a:3d}, {Q1b:3d})         "
              f"  ({W2:3d}, {Q2a:3d}, {Q2b:3d})")

    # ── Intermediate CnOrder values (useful for debugging) ─────────────── #
    print()
    print("CnOrder[t] (ordering stage) vs CnTransship[t] (full period) at (0,0):")
    print("  CnOrder should be ≤ CnTransship since transshipment can only help.")
    print(f"  {'period':>8}  {'CnOrder(0,0)':>14}  {'CnTransship(0,0)':>18}  {'diff':>10}")
    print("  " + "-" * 58)
    for t in range(4):
        o = s2.order_value(t, 0, 0)
        tr = s2.value(t, 0, 0)
        print(f"  t={t+1:>6}  {o:14.6f}  {tr:18.6f}  {tr - o:10.6f}")

    # ── Full value-function slice for inspection ──────────────────────── #
    print()
    print("CnTransship (SDP_comparePassed-2) = C (SDP_comparePassed-1) at period 1, i2 fixed at 0:")
    i2_fixed = 0
    i1_range = range(min_inv, max_inv + 1, 3)
    print(f"  {'i1':>5}", end="")
    for i1 in i1_range:
        print(f"  {i1:8d}", end="")
    print()
    print(f"  {'SDP_comparePassed-1':>5}", end="")
    for i1 in i1_range:
        print(f"  {s1.value(0, i1, i2_fixed):8.3f}", end="")
    print()
    print(f"  {'SDP_comparePassed-2':>5}", end="")
    for i1 in i1_range:
        print(f"  {s2.value(0, i1, i2_fixed):8.3f}", end="")
    print()
