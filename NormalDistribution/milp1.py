"""
milp1.py
────────
MILP-1: two-location static-uncertainty-strategy model for the hybrid
transshipment lot-sizing problem, with the first-order loss function replaced
by the Rossi (2014) piecewise-linear (Jensen) lower bound.

Solved in-process with scipy.optimize.milp (HiGHS).  Same interface as the
PuLP reference in milp1_pulp_ref.py, against which it is cross-checked.

Variables per residual period t (11 each):
    Wp Wm Q1 Q2 H1 H2 B1 B2   (continuous >= 0)
    g1 g2 d                    (binary)
where X_j,t = i_j0 + sum_{s<=t}(Q_j,s -/+ (Wp_s - Wm_s)) is substituted inline.
"""

import os, contextlib
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from normal_demand import rossi_coeffs


@contextlib.contextmanager
def _suppress_c_stderr():
    """Silence HiGHS' occasional C-level stdout/stderr chatter during solve."""
    saved = (os.dup(1), os.dup(2))
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1); os.dup2(devnull, 2); os.close(devnull)
    try:
        yield
    finally:
        os.dup2(saved[0], 1); os.dup2(saved[1], 2)
        os.close(saved[0]); os.close(saved[1])

_NV = 11  # variables per period
_Wp, _Wm, _Q1, _Q2, _H1, _H2, _B1, _B2, _g1, _g2, _d = range(_NV)


def solve_milp1(i1_0, i2_0, mu1, mu2, cv,
                K, z, R, v, h, b, n_regions=10, msg=False):
    mu1 = np.asarray(mu1, float); mu2 = np.asarray(mu2, float)
    T = len(mu1)
    sig1 = cv * mu1; sig2 = cv * mu2
    Pcum, Mcum = rossi_coeffs(n_regions); nreg = len(Pcum)
    muC1 = np.cumsum(mu1); muC2 = np.cumsum(mu2)
    sigC1 = np.sqrt(np.cumsum(sig1 ** 2)); sigC2 = np.sqrt(np.cumsum(sig2 ** 2))
    Mbig = float(muC1[-1] + muC2[-1] + 10.0 * (sigC1[-1] + sigC2[-1])
                 + abs(i1_0) + abs(i2_0) + 10.0)

    n = _NV * T

    def col(t, which):
        return _NV * t + which

    # objective
    c = np.zeros(n)
    for t in range(T):
        c[col(t, _Wp)] = v; c[col(t, _Wm)] = v
        c[col(t, _Q1)] = z; c[col(t, _Q2)] = z
        c[col(t, _H1)] = h; c[col(t, _H2)] = h
        c[col(t, _B1)] = b; c[col(t, _B2)] = b
        c[col(t, _g1)] = K; c[col(t, _g2)] = K; c[col(t, _d)] = R

    rows = []; ub = []

    def add(row, bound):
        rows.append(row); ub.append(bound)

    # helper: cumulative X coefficients into a row for loc j up to period t
    def put_cumX(row, t, coef, loc):
        # X_loc,t = i0 + sum_{s<=t}(Q_loc,s -/+ (Wp_s - Wm_s))
        for s in range(t + 1):
            if loc == 1:
                row[col(s, _Q1)] += coef * 1.0
                row[col(s, _Wp)] += coef * (-1.0)
                row[col(s, _Wm)] += coef * (+1.0)
            else:
                row[col(s, _Q2)] += coef * 1.0
                row[col(s, _Wp)] += coef * (+1.0)
                row[col(s, _Wm)] += coef * (-1.0)

    for t in range(T):
        for i in range(nreg):
            P = Pcum[i]; Mc = Mcum[i]
            # H1 >= P*X1 - P*muC1 - Mc*sigC1   ->  P*X1 - H1 <= P*muC1 + Mc*sigC1
            r = np.zeros(n); put_cumX(r, t, P, 1); r[col(t, _H1)] -= 1.0
            add(r, P * muC1[t] + Mc * sigC1[t] - P * i1_0)
            r = np.zeros(n); put_cumX(r, t, P, 2); r[col(t, _H2)] -= 1.0
            add(r, P * muC2[t] + Mc * sigC2[t] - P * i2_0)
            # B1 >= (P-1)*X1 - P*muC1 - Mc*sigC1 + muC1
            r = np.zeros(n); put_cumX(r, t, P - 1.0, 1); r[col(t, _B1)] -= 1.0
            add(r, P * muC1[t] + Mc * sigC1[t] - muC1[t] - (P - 1.0) * i1_0)
            r = np.zeros(n); put_cumX(r, t, P - 1.0, 2); r[col(t, _B2)] -= 1.0
            add(r, P * muC2[t] + Mc * sigC2[t] - muC2[t] - (P - 1.0) * i2_0)
        # setup linking:  Q - Mbig*g <= 0 ; Wp+Wm - Mbig*d <= 0
        r = np.zeros(n); r[col(t, _Q1)] = 1.0; r[col(t, _g1)] = -Mbig; add(r, 0.0)
        r = np.zeros(n); r[col(t, _Q2)] = 1.0; r[col(t, _g2)] = -Mbig; add(r, 0.0)
        r = np.zeros(n); r[col(t, _Wp)] = 1.0; r[col(t, _Wm)] = 1.0
        r[col(t, _d)] = -Mbig; add(r, 0.0)
        # availability:  Wp[t] <= onhand1_{t-1}, Wm[t] <= onhand2_{t-1}
        if t == 0:
            r = np.zeros(n); r[col(t, _Wp)] = 1.0; add(r, max(0.0, i1_0))
            r = np.zeros(n); r[col(t, _Wm)] = 1.0; add(r, max(0.0, i2_0))
        else:
            r = np.zeros(n); r[col(t, _Wp)] = 1.0; r[col(t - 1, _H1)] = -1.0; add(r, 0.0)
            r = np.zeros(n); r[col(t, _Wm)] = 1.0; r[col(t - 1, _H2)] = -1.0; add(r, 0.0)

    A = np.array(rows); bu = np.array(ub)
    constraints = LinearConstraint(A, -np.inf, bu)

    lb = np.zeros(n); ubv = np.full(n, np.inf)
    integrality = np.zeros(n)
    for t in range(T):
        for gcol in (_g1, _g2, _d):
            integrality[col(t, gcol)] = 1
            ubv[col(t, gcol)] = 1.0
    bounds = Bounds(lb, ubv)

    with _suppress_c_stderr():
        res = milp(c=c, constraints=constraints, integrality=integrality,
                   bounds=bounds, options={"disp": msg})
    if not res.success:
        raise RuntimeError(f"MILP-1 failed: {res.message}")
    x = res.x
    gv = lambda t, w: float(x[col(t, w)])
    Wp = np.array([gv(t, _Wp) for t in range(T)])
    Wm = np.array([gv(t, _Wm) for t in range(T)])
    Q1 = np.array([gv(t, _Q1) for t in range(T)])
    Q2 = np.array([gv(t, _Q2) for t in range(T)])
    return {"status": "Optimal", "obj": float(res.fun),
            "Wp": Wp, "Wm": Wm, "W": Wp - Wm, "Q1": Q1, "Q2": Q2}


if __name__ == "__main__":
    res = solve_milp1(12, -4, [6, 7, 8, 7], [6, 7, 8, 7], 0.30,
                      K=40, z=1, R=15, v=0.5, h=1, b=8, n_regions=10)
    print("status:", res["status"], " obj:", round(res["obj"], 4))
    print("W :", np.round(res["W"], 3))
    print("Q1:", np.round(res["Q1"], 3), " Q2:", np.round(res["Q2"], 3))
