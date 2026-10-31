"""
milp1.py : loss functions of cumulative demand and model MILP-1 of Section 4.1, solved with Gurobi.

Contents
    cumulative_pmfs        pmf of d^j_{k..t} for t = k..T, the convolution of the truncated period pmfs of lt_sdp
    Loss                   exact first order loss function L(x, w) and its complement L^(x, w) of one pmf
    partition_coefficients the piecewise linear lower bound of Rossi et al. (2014) for one pmf
    MILP1                  MILP-1 over the residual horizon k..T; one model per k, re-solved for each opening inventory

Model, as in Section 4.1 of the paper (equations milp-obj to nonneg), at period k
    opening inventory   I^j_{k-1} = observed inventory i^j_{k-1}
    demand              cumulated from period k: d^j_{k..t} replaces d^j_{1..t}, t = k..T
    variables           Q^j_t, W+_t, W-_t integer >= 0;  gamma^j_t, delta^1_t, delta^2_t binary;
                        H^j_t, B^j_t >= 0;  I^j_t free (expected closing inventory)
    objective           sum_t  R (delta^1_t + delta^2_t) + v (W+_t + W-_t) + sum_j [h H^j_t + b B^j_t + K gamma^j_t + z Q^j_t]
    (pwl-H)             H^j_t >= X^j_t P_i - S_i                         i = 1..N
    (pwl-B)             B^j_t >= X^j_t P_i - S_i - X^j_t + dbar^j_{k..t}  i = 1..N
                        with X^j_t = I^j_t + dbar^j_{k..t},  P_i = sum_{l<=i} p_l,  S_i = sum_{l<=i} p_l E[d^j_{k..t} | Omega_l]
    (flow1), (flow2)    I^1_{t-1} - W+_t + W-_t + Q^1_t - dbar^1_t = I^1_t,   I^2_{t-1} + W+_t - W-_t + Q^2_t - dbar^2_t = I^2_t
    (link-Q)            gamma^j_t = 0  =>  Q^j_t = 0
    (link-W)            delta^1_t = 0  =>  W+_t = 0,   delta^2_t = 0  =>  W-_t = 0,   delta^1_t + delta^2_t <= 1
    (trans-bound)       delta^1_t = 1  =>  W+_t <= I^1_{t-1},   delta^2_t = 1  =>  W-_t <= I^2_{t-1}
The logical constraints are Gurobi indicator constraints, so the model has no big-M.

Relation to the OPL model LT_opl_models/LT_MILP_C.mod and LT_MILP.java
    * Same objective and cost terms, same transshipment domain on the expected opening inventory.
    * Uncertainty is cumulated from the first period of the residual horizon, as in the paper, instead of
      from the last replenishment through P1[j][t] and mean_matrix[j][t].
    * Corrected: lamda_matrix[j][t][p] inside the sum over k should be indexed by k; each location uses its
      own coefficients (LT_RQ_RecedingHorizon built them from location A only); cumulative means are not
      truncated to integers (possionPiecewisePartitions.convolution returned an int).
    * Two transshipment binaries delta^1, delta^2 instead of one transshipDecision; Q and W integer.

Partition of the piecewise linear bound
    Poisson: N subregions of equal probability 1/N, as LT_MILP.java and possionPiecewisePartitions.java.
    Normal : the probabilities p_l of the optimal partition of Rossi et al. (2014) for N = 10, as RSmilp_normal.java.
    The conditional expectations E[d | Omega_l] are computed exactly from the pmf of d^j_{k..t} that the SDP uses,
    by cutting its quantile function at the cumulative probabilities P_i. For Poisson this is the limit of the
    sampling estimate in possionPiecewisePartitions.conditionalExpectationGivenPartitions; for the normal case it
    is the discrete counterpart of mu + sigma * E[Z | Omega_l]. Any such partition gives a lower bound (Jensen).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

import gurobipy as gp
from gurobipy import GRB

# Rossi, Tarim, Prestwich & Hnich (2014), standard normal, N = 10; the values in RSmilp_normal.java
ROSSI_N10_PROB = [0.04206108420763477, 0.0836356495308449, 0.11074334596058821, 0.1276821455299152,
                  0.13587777477101692, 0.13587777477101692, 0.1276821455299152, 0.11074334596058821,
                  0.0836356495308449, 0.04206108420763477]
ROSSI_N10_MEAN = [-2.133986195498256, -1.3976822972668839, -0.918199946431143, -0.5265753462727588,
                  -0.17199013069262026, 0.17199013069262026, 0.5265753462727588, 0.918199946431143,
                  1.3976822972668839, 2.133986195498256]


# ----------------------------------------------------------------------------------------------------
# cumulative demand and loss functions
# ----------------------------------------------------------------------------------------------------
def cumulative_pmfs(period_pmfs: Sequence[np.ndarray], k: int) -> List[np.ndarray]:
    """pmf of d_{k..t} for t = k..T, with periods counted from 0 here: element s is d_{k..k+s}."""
    out, acc = [], None
    for p in period_pmfs[k:]:
        acc = np.asarray(p, float) if acc is None else np.convolve(acc, p)
        out.append(acc)
    return out


class Loss:
    """Exact L^(x) = E[max(x - w, 0)] and L(x) = E[max(w - x, 0)] of a pmf on 0..m, for integer x."""

    def __init__(self, pmf: np.ndarray):
        p = np.asarray(pmf, float)
        w = np.arange(len(p))
        self.m = len(p) - 1
        self.F = np.cumsum(p)                    # P(w <= x)
        self.G = np.cumsum(w * p)                # E[w ; w <= x]
        self.mean = float(self.G[-1])

    def complement(self, x) -> np.ndarray:
        """L^(x) = sum_{w <= x} (x - w) p(w)."""
        x = np.asarray(x)
        xi = np.clip(x, 0, self.m).astype(int)
        inside = x * self.F[xi] - self.G[xi]
        return np.where(x < 0, 0.0, np.where(x > self.m, x - self.mean, inside))

    def loss(self, x) -> np.ndarray:
        """L(x) = L^(x) - x + E[w]."""
        x = np.asarray(x)
        return self.complement(x) - x + self.mean


def partition_probabilities(distribution: str, partitions: int = 10) -> np.ndarray:
    dist = distribution.lower()
    if dist == "poisson":
        return np.full(partitions, 1.0 / partitions)
    if dist == "normal":
        if partitions != 10:
            raise ValueError("the Rossi et al. (2014) partition is tabulated here for N = 10 only")
        return np.array(ROSSI_N10_PROB)
    raise ValueError("distribution must be 'poisson' or 'normal'")


def partition_coefficients(pmf: np.ndarray, probs: np.ndarray):
    """P_i = p_1 + ... + p_i and S_i = sum_{l <= i} p_l E[w | Omega_l], i = 1..N, for the quantile partition
    of the pmf at the cumulative probabilities P_i. S_i is the integral of the quantile function from 0 to P_i."""
    p = np.asarray(pmf, float)
    w = np.arange(len(p))
    Fprev = np.concatenate(([0.0], np.cumsum(p)[:-1]))     # P(w < x)
    P = np.cumsum(probs)
    P[-1] = 1.0
    S = np.array([np.sum(w * np.clip(Pi - Fprev, 0.0, p)) for Pi in P])
    S[-1] = float(np.sum(w * p))                            # last piece is exactly X - E[w]
    return P, S


def conditional_means(pmf: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """E[w | Omega_l], l = 1..N; used only for checks against the Java coefficients."""
    P, S = partition_coefficients(pmf, probs)
    return np.diff(np.concatenate(([0.0], S))) / np.asarray(probs)


# ----------------------------------------------------------------------------------------------------
# MILP-1
# ----------------------------------------------------------------------------------------------------
@dataclass
class MILP1Solution:
    """Optimal decisions pi^_k over periods k..T, element s refers to period k + s."""
    k: int                      # first period of the residual horizon, counted from 1
    opening: tuple              # (i^1_{k-1}, i^2_{k-1})
    W: np.ndarray               # W^_t = W+_t - W-_t
    Q1: np.ndarray
    Q2: np.ndarray
    obj: float
    seconds: float


class MILP1:
    """MILP-1 over the residual horizon k..T. The model is built once; solve() changes only the opening inventory."""

    def __init__(self, k: int, pmfs1: Sequence[np.ndarray], pmfs2: Sequence[np.ndarray],
                 K: float, z: float, R: float, v: float, h: float, b: float,
                 probs: np.ndarray, threads: int = 1, env: Optional[gp.Env] = None):
        """k counted from 1; pmfs1, pmfs2 are the period pmfs of the whole horizon (lt_sdp.demand_pmf)."""
        self.k = k
        T = len(pmfs1)
        L = T - k + 1
        self.L = L
        m = gp.Model(f"MILP1_k{k}", env=env) if env is not None else gp.Model(f"MILP1_k{k}")
        m.Params.OutputFlag = 0
        m.Params.Threads = threads
        self.model = m

        dbar = [[float(np.dot(np.arange(len(p)), p)) for p in pm[k - 1:]] for pm in (pmfs1, pmfs2)]
        cum = [cumulative_pmfs(pm, k - 1) for pm in (pmfs1, pmfs2)]
        J, S_ = (0, 1), range(L)

        self.I0 = m.addVars(J, lb=0.0, ub=0.0, name="I0")                  # bounds set in solve()
        self.Q = m.addVars(J, S_, vtype=GRB.INTEGER, lb=0, name="Q")
        g = m.addVars(J, S_, vtype=GRB.BINARY, name="gamma")
        self.Wp = m.addVars(S_, vtype=GRB.INTEGER, lb=0, name="Wplus")
        self.Wm = m.addVars(S_, vtype=GRB.INTEGER, lb=0, name="Wminus")
        d1 = m.addVars(S_, vtype=GRB.BINARY, name="delta1")
        d2 = m.addVars(S_, vtype=GRB.BINARY, name="delta2")
        I = m.addVars(J, S_, lb=-GRB.INFINITY, name="I")
        H = m.addVars(J, S_, lb=0.0, name="H")
        B = m.addVars(J, S_, lb=0.0, name="B")
        sign = {0: -1.0, 1: 1.0}                                           # location 1 sends W+, location 2 receives it

        for s in S_:
            for j in J:
                prev = self.I0[j] if s == 0 else I[j, s - 1]
                m.addConstr(prev + sign[j] * (self.Wp[s] - self.Wm[s]) + self.Q[j, s] - dbar[j][s] == I[j, s],
                            name=f"flow{j + 1}_{s}")
                dcum = float(np.dot(np.arange(len(cum[j][s])), cum[j][s]))
                X = I[j, s] + dcum
                P, S = partition_coefficients(cum[j][s], probs)
                for i in range(len(P)):
                    m.addConstr(H[j, s] >= P[i] * X - S[i], name=f"pwlH{j + 1}_{s}_{i}")
                    m.addConstr(B[j, s] >= P[i] * X - S[i] - X + dcum, name=f"pwlB{j + 1}_{s}_{i}")
                m.addGenConstrIndicator(g[j, s], False, self.Q[j, s] == 0, name=f"linkQ{j + 1}_{s}")
            m.addGenConstrIndicator(d1[s], False, self.Wp[s] == 0, name=f"linkW1_{s}")
            m.addGenConstrIndicator(d2[s], False, self.Wm[s] == 0, name=f"linkW2_{s}")
            m.addConstr(d1[s] + d2[s] <= 1, name=f"oneway_{s}")
            prev1 = self.I0[0] if s == 0 else I[0, s - 1]
            prev2 = self.I0[1] if s == 0 else I[1, s - 1]
            m.addGenConstrIndicator(d1[s], True, self.Wp[s] - prev1 <= 0, name=f"bound1_{s}")
            m.addGenConstrIndicator(d2[s], True, self.Wm[s] - prev2 <= 0, name=f"bound2_{s}")

        m.setObjective(gp.quicksum(R * (d1[s] + d2[s]) + v * (self.Wp[s] + self.Wm[s])
                                   + gp.quicksum(h * H[j, s] + b * B[j, s] + K * g[j, s] + z * self.Q[j, s] for j in J)
                                   for s in S_), GRB.MINIMIZE)
        m.update()

    def solve(self, i1: int, i2: int) -> MILP1Solution:
        t0 = time.perf_counter()
        for j, x in ((0, i1), (1, i2)):
            self.I0[j].LB = x
            self.I0[j].UB = x
        self.model.optimize()
        if self.model.Status != GRB.OPTIMAL:
            raise RuntimeError(f"MILP-1 at k={self.k}, opening ({i1}, {i2}): Gurobi status {self.model.Status}")
        r = lambda var: int(round(var.X))
        W = np.array([r(self.Wp[s]) - r(self.Wm[s]) for s in range(self.L)])
        Q1 = np.array([r(self.Q[0, s]) for s in range(self.L)])
        Q2 = np.array([r(self.Q[1, s]) for s in range(self.L)])
        return MILP1Solution(self.k, (i1, i2), W, Q1, Q2, self.model.ObjVal, time.perf_counter() - t0)
