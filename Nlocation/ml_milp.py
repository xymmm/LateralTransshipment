"""
ml_milp.py : the mixed integer linear model for J locations under the static strategy (N-main.tex, Section 5).

All quantities of periods k..T are fixed at period k, from the observed opening inventory. Solved with Gurobi.

    opening inventory   I^j_{k-1} = observed inventory i^j_{k-1}
    variables           Q^j_t integer >= 0, gamma^j_t binary                 reorder at location j
                        W^{jm}_t integer >= 0, delta^{jm}_t binary           shipment from j to m
                        H^j_t, B^j_t >= 0, I^j_t free                        expected on-hand, backorders, closing inventory
    objective           sum_t { sum_{j != m} [R delta^{jm}_t + v W^{jm}_t] + sum_j [h H^j_t + b B^j_t + K gamma^j_t + z Q^j_t] }
    (pwl-H)             H^j_t >= P_s X^j_t - S_s,                 s = 0..N,  X^j_t = I^j_t + E[d^j_{k..t}]
    (pwl-B)             B^j_t >= P_s X^j_t - S_s - I^j_t,         s = 0..N
                        with P_0 = S_0 = 0, P_s = p_1 + .. + p_s and S_s = sum_{n <= s} p_n E[d^j_{k..t} | Omega_n]
    (flow)              I^j_{t-1} - sum_m W^{jm}_t + sum_m W^{mj}_t + Q^j_t - dbar^j_t = I^j_t
    (link-Q)            gamma^j_t = 0   =>  Q^j_t = 0
    (link-W)            delta^{jm}_t = 0  =>  W^{jm}_t = 0
    (one-role)          delta^{jm}_t + delta^{lj}_t <= 1                     a location does not ship and receive in a period
    (trans-bound)       delta^{jm}_t = 1  =>  sum_l W^{jl}_t <= I^j_{t-1}    shipments limited by the expected opening stock

The piece s = 0 of (pwl-B) is  B^j_t >= -I^j_t.
    The bound on the complementary loss function is max over s = 0..N of P_s X - S_s, and its piece s = 0 is 0.
    The loss function is the complementary loss function minus X plus the expected demand, so the same piece
    gives B >= E[d] - X = -I. Without it the model undervalues the backorders whenever the expected inventory
    is negative. milp1.MILP1 of the two-location code has the pieces s = 1..N only; zero_piece=False gives that
    model, which ml_checks.py uses to compare the two models. zero_piece=True is the default.

With J = 2 and zero_piece=False this is MILP-1 of milp1.py: W^{12} = W+, W^{21} = W-, (one-role) is
delta^1 + delta^2 <= 1 and (trans-bound) is W+ <= I^1_{t-1}, W- <= I^2_{t-1}.

Coefficients
    The model needs, for every cumulative demand d^j_{k..t}, its expected value and the coefficients P_s, S_s.
    They depend on the demand distributions alone. ml_milpRL.py computes them once before the first period
    (class Precomputation) and passes them in as `pre`. Without `pre` the model computes them itself.

    from ml_milp import solve_static
    plan = solve_static([[1, 1, 1], [1, 2, 1], [2, 1, 1]], K=5, z=1, R=1, v=0.5, h=1, b=5)
    plan.Q, plan.W, plan.obj, plan.seconds     the static plan from the opening inventory (0, 0, 0)

Real-valued demand: step < 1 (0.5, 0.25, 0.1, ...)
    Demand values are multiples of step, pmfs[j][t][k] = P(d^j_t = k * step); Q and W stay integers. The model
    itself does not change: only the expected demands and the coefficients S_s are computed from the real demand
    values. The cells Omega_n are cells of the quantile function, so a probability atom can be split between
    two cells. The opening inventory may be any real vector, also with step = 1. Because W is an integer,
    (trans-bound) lets location j ship floor(max(I^j_{k-1}, 0)) units in period k.
    distribution="gamma" (with cv) is a nonnegative real-valued demand; pmfs=... gives any demand law.
        plan = solve_static(means, K=5, z=1, R=1, v=0.5, h=1, b=5, distribution="gamma", cv=0.5, step=0.1,
                            opening=(2.6, 0.7, -1.2))
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Sequence

import numpy as np

import gurobipy as gp
from gurobipy import GRB

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from milp1 import partition_coefficients, partition_probabilities  # noqa: E402
from ml_common import per_unit, period_pmfs  # noqa: E402

__all__ = ["MILP", "MILPSolution", "solve_static", "gurobi_env", "cell_probabilities"]


def cell_probabilities(distribution: str, partitions: int = 10) -> np.ndarray:
    """Probabilities p_n of the cells Omega_n: those of Rossi et al. (2014) for normal demand, as in the paper,
    and equal probabilities for every other demand."""
    if str(distribution).lower() == "normal":
        return partition_probabilities("normal", partitions)
    return np.full(partitions, 1.0 / partitions)


def gurobi_env() -> gp.Env:
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    return env


class _Coefficients:
    """The coefficients of the model, computed here when no precomputation is passed in."""

    def __init__(self, pmfs, probs, step: float = 1.0):
        self.pmfs = [[np.asarray(p, float) for p in pm] for pm in pmfs]
        self.probs = np.asarray(probs, float)
        self.unit = 1.0 / per_unit(step)                                   # a demand of k steps is k * unit
        self.dbar = [[self.unit * float(np.dot(np.arange(len(p)), p)) for p in pm] for pm in self.pmfs]

    def entries(self, j: int, k: int):
        out, acc = [], None
        for p in self.pmfs[j][k - 1:]:
            acc = p if acc is None else np.convolve(acc, p)                # distribution of d^j_{k..t}
            P, S = partition_coefficients(acc, self.probs)                 # S in steps
            out.append(SimpleNamespace(mean=self.unit * float(np.dot(np.arange(len(acc)), acc)), P=P, S=self.unit * S))
        return out


@dataclass
class MILPSolution:
    """Optimal decisions over periods k..T; index s refers to period k + s, locations are counted from 0."""
    k: int
    opening: tuple
    W: np.ndarray               # W[s, j, m] units from j to m
    Q: np.ndarray               # Q[s, j]
    obj: float
    seconds: float

    def net_inflow(self) -> np.ndarray:
        """net[s, j] = sum_m W[s, m, j] - sum_m W[s, j, m]."""
        return self.W.sum(axis=1) - self.W.sum(axis=2)


class MILP:
    """The model over the residual horizon k..T. It is built once; solve() changes only the opening inventory."""

    def __init__(self, k: int, pmfs: Sequence[Sequence[np.ndarray]],
                 K: float, z: float, R: float, v: float, h: float, b: float,
                 probs: np.ndarray, threads: int = 1, env: Optional[gp.Env] = None,
                 time_limit: Optional[float] = None, mip_gap: Optional[float] = None,
                 zero_piece: bool = True, pre=None, step: float = 1.0):
        """k counted from 1; pmfs[j] are the period pmfs of location j over the whole horizon, on the multiples
        of step. pre: the precomputation of ml_milpRL.py for the same pmfs, probs and step."""
        self.k = k
        self.J = J = len(pmfs)
        T = len(pmfs[0])
        self.L = L = T - k + 1
        self.zero_piece = zero_piece
        pre = _Coefficients(pmfs, probs, step) if pre is None else pre
        ent = [pre.entries(j, k) for j in range(J)]
        t_build = time.perf_counter()
        m = gp.Model(f"MILP_J{J}_k{k}", env=env) if env is not None else gp.Model(f"MILP_J{J}_k{k}")
        m.Params.OutputFlag = 0
        m.Params.Threads = threads
        if time_limit is not None:
            m.Params.TimeLimit = time_limit
        if mip_gap is not None:
            m.Params.MIPGap = mip_gap
        self.model = m

        dbar = [pre.dbar[j][k - 1:] for j in range(J)]
        locs, S_ = range(J), range(L)
        arcs = [(j, mm) for j in locs for mm in locs if j != mm]

        self.I0 = m.addVars(locs, lb=0.0, ub=0.0, name="I0")                 # bounds set in solve()
        self.Q = m.addVars(locs, S_, vtype=GRB.INTEGER, lb=0, name="Q")
        g = m.addVars(locs, S_, vtype=GRB.BINARY, name="gamma")
        self.W = m.addVars(arcs, S_, vtype=GRB.INTEGER, lb=0, name="W")
        d = m.addVars(arcs, S_, vtype=GRB.BINARY, name="delta")
        I = m.addVars(locs, S_, lb=-GRB.INFINITY, name="I")
        H = m.addVars(locs, S_, lb=0.0, name="H")
        B = m.addVars(locs, S_, lb=0.0, name="B")

        for s in S_:
            for j in locs:
                prev = self.I0[j] if s == 0 else I[j, s - 1]
                out = gp.quicksum(self.W[j, mm, s] for mm in locs if mm != j)
                inn = gp.quicksum(self.W[mm, j, s] for mm in locs if mm != j)
                m.addConstr(prev - out + inn + self.Q[j, s] - dbar[j][s] == I[j, s], name=f"flow{j + 1}_{s}")
                dcum = ent[j][s].mean
                X = I[j, s] + dcum
                P, S = ent[j][s].P, ent[j][s].S
                for i in range(len(P)):
                    m.addConstr(H[j, s] >= P[i] * X - S[i], name=f"pwlH{j + 1}_{s}_{i}")
                    m.addConstr(B[j, s] >= P[i] * X - S[i] - X + dcum, name=f"pwlB{j + 1}_{s}_{i}")
                m.addGenConstrIndicator(g[j, s], False, self.Q[j, s] == 0, name=f"linkQ{j + 1}_{s}")
            # the constraints below are added in the sequence of milp1.MILP1, so that with J = 2 and
            # zero_piece=False the two models are the same model for the solver
            for j, mm in arcs:
                m.addGenConstrIndicator(d[j, mm, s], False, self.W[j, mm, s] == 0, name=f"linkW{j + 1}{mm + 1}_{s}")
            done = set()
            for j in locs:                                   # location j ships to mm and receives from l
                for mm in locs:
                    for l in locs:
                        if mm == j or l == j or frozenset(((j, mm), (l, j))) in done:
                            continue
                        done.add(frozenset(((j, mm), (l, j))))
                        m.addConstr(d[j, mm, s] + d[l, j, s] <= 1, name=f"role{j + 1}_{mm + 1}_{l + 1}_{s}")
            for j, mm in arcs:
                prev = self.I0[j] if s == 0 else I[j, s - 1]
                out = gp.quicksum(self.W[j, l, s] for l in locs if l != j)
                m.addGenConstrIndicator(d[j, mm, s], True, out - prev <= 0, name=f"bound{j + 1}{mm + 1}_{s}")
        if zero_piece:                                       # the piece s = 0 of (pwl-B), added last
            for s in S_:
                for j in locs:
                    m.addConstr(B[j, s] >= -I[j, s], name=f"pwlB{j + 1}_{s}_zero")

        m.setObjective(gp.quicksum(gp.quicksum(R * d[j, mm, s] + v * self.W[j, mm, s] for j, mm in arcs)
                                   + gp.quicksum(h * H[j, s] + b * B[j, s] + K * g[j, s] + z * self.Q[j, s] for j in locs)
                                   for s in S_), GRB.MINIMIZE)
        m.update()
        self.seconds_build = time.perf_counter() - t_build

    def solve(self, opening: Sequence[float]) -> MILPSolution:
        """opening: the observed inventory, any real vector."""
        t0 = time.perf_counter()
        for j, x in enumerate(opening):
            self.I0[j].LB = float(x)
            self.I0[j].UB = float(x)
        self.model.optimize()
        if self.model.SolCount == 0 or self.model.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
            raise RuntimeError(f"MILP at k={self.k}, opening {tuple(opening)}: Gurobi status {self.model.Status}")
        J, L = self.J, self.L
        W = np.zeros((L, J, J), dtype=np.int64)
        Q = np.zeros((L, J), dtype=np.int64)
        for s in range(L):
            for j in range(J):
                Q[s, j] = int(round(self.Q[j, s].X))
                for mm in range(J):
                    if mm != j:
                        W[s, j, mm] = int(round(self.W[j, mm, s].X))
        return MILPSolution(self.k, tuple(opening), W, Q, self.model.ObjVal, time.perf_counter() - t0)


def solve_static(demand_means: Sequence[Sequence[float]], K: float, z: float, R: float, v: float, h: float, b: float,
                 opening: Optional[Sequence[float]] = None, distribution: str = "poisson", cv=None, tail: float = 1e-4,
                 partitions: int = 10, zero_piece: bool = True, threads: int = 1,
                 time_limit: Optional[float] = None, mip_gap: Optional[float] = None,
                 step: float = 1.0, pmfs=None) -> MILPSolution:
    """One solve of the model over the whole horizon from the opening inventory (default: all zero): the static
    plan. Its objective approximates the expected cost of the plan with the piecewise linear bounds; it is not
    the cost of the rolling-horizon algorithm, which ml_milpRL.py estimates."""
    explicit = pmfs is not None
    pmfs = period_pmfs(demand_means, distribution, cv, tail, step, pmfs)
    J = len(pmfs)
    opening = (0,) * J if opening is None else tuple(opening)
    env = gurobi_env()
    try:
        mdl = MILP(1, pmfs, K, z, R, v, h, b, cell_probabilities("explicit" if explicit else distribution, partitions),
                   threads=threads, env=env, time_limit=time_limit, mip_gap=mip_gap, zero_piece=zero_piece, step=step)
        sol = mdl.solve(opening)
        mdl.model.dispose()
    finally:
        env.dispose()
    return sol
