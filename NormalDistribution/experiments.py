"""
experiments.py
──────────────
Produces the numbers for the paper:

  exp1  SDP-1 vs SDP-2  — equivalence (Delta_ref = 0) and computational effort.
  exp2  Total optimality gap of the receding-horizon policy, and the
        refinement ablation that isolates the value of the transshipment
        refinement (a proxy for Delta_eval).
  exp3  Linearisation sweep over the number of partition regions N, and a
        direct reading of Delta_lin as MILP objective vs the exact expected
        cost of the same open-loop plan.

Run a single experiment with:  python experiments.py {1|2|3}
"""

import sys, time
import numpy as np
from normal_demand import normal_pmf, loss, loss_complement
from sdp_normal import SDP1Solver, SDP2Solver
from milp1 import solve_milp1
from rh import RHPolicy


def exact_static_cost(i1_0, i2_0, plan, mu1, mu2, cv, K, z, R, v, h, b):
    """
    Exact expected total cost of a fixed open-loop static plan, using the exact
    normal loss functions.  This is the honest cost of the plan that MILP-1
    scores with its piecewise-linear (Jensen) lower bound.
    """
    mu1 = np.asarray(mu1, float); mu2 = np.asarray(mu2, float)
    sig1 = cv * mu1; sig2 = cv * mu2
    Wp, Wm, Q1, Q2 = plan["Wp"], plan["Wm"], plan["Q1"], plan["Q2"]
    W = Wp - Wm
    T = len(mu1)
    muC1 = np.cumsum(mu1); muC2 = np.cumsum(mu2)
    sigC1 = np.sqrt(np.cumsum(sig1 ** 2)); sigC2 = np.sqrt(np.cumsum(sig2 ** 2))
    cumQ1 = np.cumsum(Q1); cumQ2 = np.cumsum(Q2); cumW = np.cumsum(W)
    tol = 1e-3   # MILP slack (~1e-6) must not be mistaken for a real setup
    cost = 0.0
    for t in range(T):
        if abs(W[t]) > tol:
            cost += R + v * abs(W[t])
        if Q1[t] > tol:
            cost += K + z * Q1[t]
        if Q2[t] > tol:
            cost += K + z * Q2[t]
        X1 = i1_0 + cumQ1[t] - cumW[t]
        X2 = i2_0 + cumQ2[t] + cumW[t]
        cost += (h * loss_complement(X1, muC1[t], sigC1[t])
                 + b * loss(X1, muC1[t], sigC1[t])
                 + h * loss_complement(X2, muC2[t], sigC2[t])
                 + b * loss(X2, muC2[t], sigC2[t]))
    return float(cost)


# ─────────────────────────────────────────────────────────────────────────────
def exp1():
    """SDP-1 vs SDP-2: equivalence and effort, over instances of growing grid."""
    K, z, R, v, h, b = 40.0, 1.0, 15.0, 0.5, 1.0, 8.0
    cv = 0.30
    instances = [
        ("I1", [4, 5, 6, 5], [4, 5, 6, 5]),
        ("I2", [6, 7, 8, 7], [6, 7, 8, 7]),
        ("I3", [8, 9, 10, 9], [8, 9, 10, 9]),
    ]
    print("SDP-1 vs SDP-2  (cv=0.30, K=40 z=1 R=15 v=0.5 h=1 b=8)")
    print(f"{'inst':>5}{'T':>3}{'N':>5}{'SDP-1 (s)':>11}{'SDP-2 (s)':>11}"
          f"{'speedup':>9}{'max|C1-C2|':>12}")
    for name, mu1, mu2 in instances:
        t = time.time(); s2 = SDP2Solver(mu1, mu2, cv, K, z, R, v, h, b).solve()
        t2 = time.time() - t
        t = time.time(); s1 = SDP1Solver(mu1, mu2, cv, K, z, R, v, h, b).solve()
        t1 = time.time() - t
        d = np.abs(s1.C[0] - s2.CnTransship[0]).max()
        print(f"{name:>5}{s1.T:>3}{s1.N:>5}{t1:>11.2f}{t2:>11.2f}"
              f"{t1/t2:>9.1f}{d:>12.2e}")


# ─────────────────────────────────────────────────────────────────────────────
def exp2():
    """Total gap and refinement ablation on a symmetric and an asymmetric case."""
    K, z, R, v, h, b = 40.0, 1.0, 15.0, 0.5, 1.0, 8.0
    cv = 0.30
    cases = [
        ("symmetric",  [6, 7, 8, 7], [6, 7, 8, 7], [(0, 0), (8, -4)]),
        ("asymmetric", [10, 10, 4, 4], [4, 4, 10, 10], [(0, 0), (10, -2)]),
    ]
    n_paths = 2500
    for label, mu1, mu2, starts in cases:
        print(f"\n=== {label} demand  mu1={mu1}  mu2={mu2} ===")
        sdp = SDP1Solver(mu1, mu2, cv, K, z, R, v, h, b).solve()
        pol_r = RHPolicy(mu1, mu2, cv, K, z, R, v, h, b, refine=True)
        pol_n = RHPolicy(mu1, mu2, cv, K, z, R, v, h, b, refine=False)
        print(f"{'state':>10}{'opt (SDP)':>11}{'RH refine':>20}"
              f"{'RH no-refine':>20}{'gap%':>8}{'refine saves':>14}")
        for st in starts:
            opt = sdp.value(0, *st)
            mr, hr, _ = pol_r.evaluate(st, n_paths=n_paths, seed=7)
            mn, hn, _ = pol_n.evaluate(st, n_paths=n_paths, seed=7)
            gap = 100.0 * (mr - opt) / opt
            print(f"{str(st):>10}{opt:>11.3f}{mr:>13.3f}+/-{hr:>4.2f}"
                  f"{mn:>13.3f}+/-{hn:>4.2f}{gap:>8.2f}{mn-mr:>14.3f}")


# ─────────────────────────────────────────────────────────────────────────────
def exp3():
    """Linearisation sweep: RH cost and direct Delta_lin vs number of regions."""
    K, z, R, v, h, b = 40.0, 1.0, 15.0, 0.5, 1.0, 8.0
    cv = 0.30
    mu1 = [6, 7, 8, 7]; mu2 = [6, 7, 8, 7]; i0 = (8, -4)
    sdp = SDP1Solver(mu1, mu2, cv, K, z, R, v, h, b).solve()
    opt = sdp.value(0, *i0)
    print(f"Linearisation sweep  mu1={mu1} i0={i0}  SDP optimum = {opt:.3f}\n")
    print(f"{'N regions':>10}{'segments':>10}{'MILP obj':>11}"
          f"{'exact plan cost':>17}{'Delta_lin':>11}{'RH refine gap%':>16}")
    for N in [3, 4, 5, 9, 10]:
        plan = solve_milp1(i0[0], i0[1], mu1, mu2, cv, K, z, R, v, h, b,
                           n_regions=N)
        exact = exact_static_cost(i0[0], i0[1], plan, mu1, mu2, cv,
                                  K, z, R, v, h, b)
        pol = RHPolicy(mu1, mu2, cv, K, z, R, v, h, b, n_regions=N, refine=True)
        mr, hr, _ = pol.evaluate(i0, n_paths=1500, seed=7)
        gap = 100.0 * (mr - opt) / opt
        print(f"{N:>10}{N+1:>10}{plan['obj']:>11.3f}{exact:>17.3f}"
              f"{exact-plan['obj']:>11.3f}{gap:>16.2f}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "1"
    {"1": exp1, "2": exp2, "3": exp3}[which]()
