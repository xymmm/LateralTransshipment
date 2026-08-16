"""
milp1.py
────────
MILP-1: the two-location static-uncertainty-strategy model for the hybrid
transshipment lot-sizing problem, with the first-order loss function replaced
by the Rossi (2014) piecewise-linear lower bound.

This is the model embedded at each period of the receding horizon.  It is
solved over a residual horizon of length T from a given opening inventory
(i1_0, i2_0), and returns the open-loop plan (Wp, Wm, Q1, Q2) for that horizon.

Formulation (paper eqs 8-16), with the transshipment-availability constraint
(15) expressed through the expected on-hand proxy Hbar_{t-1} rather than the
big-M linearisation of Remark 1:

  min  sum_t [ R d_t + v(Wp_t + Wm_t)
               + sum_j ( h Hj_t + b Bj_t + K gj_t + z Qj_t ) ]
  s.t. X1_t = i1_0 + sum_{s<=t} (Q1_s - Wp_s + Wm_s)
       X2_t = i2_0 + sum_{s<=t} (Q2_s + Wp_s - Wm_s)
       Hj_t >= Pcum_i Xj_t - Pcum_i muC_j,t - Mcum_i sigC_j,t     i=1..N
       Bj_t >= (same) - Xj_t + muC_j,t                            i=1..N
       Qj_t <= M gj_t ,   Wp_t + Wm_t <= M d_t
       Wp_t <= onhand1_{t-1} ,   Wm_t <= onhand2_{t-1}
"""

import numpy as np
import pulp
from normal_demand import rossi_coeffs


def solve_milp1(i1_0, i2_0, mu1, mu2, cv,
                K, z, R, v, h, b,
                n_regions=10, msg=False):
    """
    Solve MILP-1 over the residual horizon.

    mu1, mu2 : demand means for the residual periods 1..T (period 1 = current).
    Returns dict with per-period plan arrays and the objective value.
    """
    mu1 = np.asarray(mu1, float); mu2 = np.asarray(mu2, float)
    T = len(mu1)
    sig1 = cv * mu1; sig2 = cv * mu2
    Pcum, Mcum = rossi_coeffs(n_regions)
    nreg = len(Pcum)

    # cumulative demand moments over the residual horizon
    muC1 = np.cumsum(mu1); muC2 = np.cumsum(mu2)
    sigC1 = np.sqrt(np.cumsum(sig1 ** 2)); sigC2 = np.sqrt(np.cumsum(sig2 ** 2))

    Mbig = float(muC1[-1] + muC2[-1] + 10.0 * (sigC1[-1] + sigC2[-1])
                 + abs(i1_0) + abs(i2_0) + 10.0)

    m = pulp.LpProblem("MILP1", pulp.LpMinimize)
    rng = range(T)
    Wp = [pulp.LpVariable(f"Wp_{t}", lowBound=0) for t in rng]
    Wm = [pulp.LpVariable(f"Wm_{t}", lowBound=0) for t in rng]
    Q1 = [pulp.LpVariable(f"Q1_{t}", lowBound=0) for t in rng]
    Q2 = [pulp.LpVariable(f"Q2_{t}", lowBound=0) for t in rng]
    g1 = [pulp.LpVariable(f"g1_{t}", cat="Binary") for t in rng]
    g2 = [pulp.LpVariable(f"g2_{t}", cat="Binary") for t in rng]
    dl = [pulp.LpVariable(f"d_{t}", cat="Binary") for t in rng]
    H1 = [pulp.LpVariable(f"H1_{t}", lowBound=0) for t in rng]
    H2 = [pulp.LpVariable(f"H2_{t}", lowBound=0) for t in rng]
    B1 = [pulp.LpVariable(f"B1_{t}", lowBound=0) for t in rng]
    B2 = [pulp.LpVariable(f"B2_{t}", lowBound=0) for t in rng]

    # cumulative available inventory (before demand)
    X1 = [i1_0 + pulp.lpSum(Q1[s] - Wp[s] + Wm[s] for s in range(t + 1)) for t in rng]
    X2 = [i2_0 + pulp.lpSum(Q2[s] + Wp[s] - Wm[s] for s in range(t + 1)) for t in rng]

    m += pulp.lpSum(
        R * dl[t] + v * (Wp[t] + Wm[t])
        + h * H1[t] + b * B1[t] + K * g1[t] + z * Q1[t]
        + h * H2[t] + b * B2[t] + K * g2[t] + z * Q2[t]
        for t in rng)

    for t in rng:
        for i in range(nreg):
            base1 = Pcum[i] * X1[t] - Pcum[i] * muC1[t] - Mcum[i] * sigC1[t]
            base2 = Pcum[i] * X2[t] - Pcum[i] * muC2[t] - Mcum[i] * sigC2[t]
            m += H1[t] >= base1
            m += H2[t] >= base2
            m += B1[t] >= base1 - X1[t] + muC1[t]
            m += B2[t] >= base2 - X2[t] + muC2[t]
        m += Q1[t] <= Mbig * g1[t]
        m += Q2[t] <= Mbig * g2[t]
        m += Wp[t] + Wm[t] <= Mbig * dl[t]
        # transshipment availability via expected on-hand at t-1
        onhand1 = max(0.0, i1_0) if t == 0 else H1[t - 1]
        onhand2 = max(0.0, i2_0) if t == 0 else H2[t - 1]
        m += Wp[t] <= onhand1
        m += Wm[t] <= onhand2

    m.solve(pulp.PULP_CBC_CMD(msg=msg))
    val = lambda x: float(pulp.value(x))
    return {
        "status": pulp.LpStatus[m.status],
        "obj": float(pulp.value(m.objective)),
        "Wp": np.array([val(Wp[t]) for t in rng]),
        "Wm": np.array([val(Wm[t]) for t in rng]),
        "W":  np.array([val(Wp[t]) - val(Wm[t]) for t in rng]),
        "Q1": np.array([val(Q1[t]) for t in rng]),
        "Q2": np.array([val(Q2[t]) for t in rng]),
    }


if __name__ == "__main__":
    # asymmetric opening inventory to trigger transshipment in the plan
    res = solve_milp1(i1_0=12, i2_0=-4,
                      mu1=[6, 7, 8, 7], mu2=[6, 7, 8, 7], cv=0.30,
                      K=40, z=1, R=15, v=0.5, h=1, b=8, n_regions=10)
    print("status:", res["status"])
    print("obj   :", round(res["obj"], 4))
    print("W  (loc1->loc2 +):", np.round(res["W"], 3))
    print("Q1:", np.round(res["Q1"], 3))
    print("Q2:", np.round(res["Q2"], 3))
