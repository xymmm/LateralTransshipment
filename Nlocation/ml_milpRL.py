"""
ml_milpRL.py : the static model of ml_milp.py applied under a rolling horizon, for J locations
               (Algorithm 1 of N-main.tex, Section 6).

    Step 0   PRECOMPUTATION, once, before the first period                         class Precomputation
             everything that depends on the demand distributions alone
    then, in every period k with the observed inventory i_{k-1}:                   class MilpRLPolicy
    Step 1   solve the static model over k..T from i_{k-1}                         -> plan
    Step 2   keep the reorder quantities of period k
    Step 3   refine the transshipment of period k:
                 min over the feasible net inflows delta of   U(delta) + sum_j g^j_k(i^j + delta^j)
             g^j_k(x) is the exact expected holding and backorder cost of location j over k..T from the
             post-transshipment inventory x, with the reorders and the later transshipments of the plan fixed:
                 g^j_k(x) = sum_{t=k}^T [ h L^(x + base^j_t, d^j_{k..t}) + b L(x + base^j_t, d^j_{k..t}) ],
                 base^j_t = sum_{l=k}^t Q^j_l + sum_{l=k+1}^t (net inflow of j in period l).
             The search lists the net inflows for J <= 3 and solves a small integer program for larger J.
    The expected total cost of the algorithm is estimated by simulation.           function milpRL

Times are kept apart
    seconds_precompute   step 0
    seconds_build        making the solver models, one per period k
    seconds_milp         step 1, all decisions computed
    seconds_refine       step 3, all decisions computed
    seconds_decision_k1  step 1 plus step 3 of the decision at k = 1, the longest residual horizon

    from ml_milpRL import milpRL
    r = milpRL([[1, 1, 1], [1, 2, 1], [2, 1, 1]], K=5, z=1, R=1, v=0.5, h=1, b=5)
    r.etc, r.half_width, r.n_paths, r.first_decision, r.seconds_precompute, r.seconds_decision_k1

Real-valued demand: step < 1 (0.5, 0.25, 0.1, ...)
    Demand values are multiples of step, pmfs[j][t][k] = P(d^j_t = k * step); reorders and shipments stay whole
    units. The observed inventory is then a real vector, and the opening inventory may be any real vector,
    also with step = 1.
    Step 0   the same entries, with the loss functions evaluated at real arguments (class LatticeLoss).
    Step 3   location j can ship s^j = floor(max(i^j, 0)) units, so its post-transshipment level is one of
             i^j - s^j, i^j - s^j + 1, ..., i^j - s^j + S with S the sum of the s^j. The search uses g^j_k at
             these levels, by listing the net inflows or by the integer program, exactly as with integers.
    distribution="gamma" (with cv) is a nonnegative real-valued demand; pmfs=... gives any demand law.
        r = milpRL(means, K=5, z=1, R=1, v=0.5, h=1, b=5, distribution="gamma", cv=0.5, step=0.1,
                   initial_inventory=(2.6, 0.7, -1.2))
"""
from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

import gurobipy as gp
from gurobipy import GRB

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from milp1 import Loss, partition_coefficients, partition_probabilities  # noqa: E402
from ml_common import (SimResult, feasible_net_inflows, min_shipments_rows, per_unit, period_pmfs,  # noqa: E402
                       shipment_plan, simulate, transshipment_cost)
from ml_milp import MILP, cell_probabilities, gurobi_env  # noqa: E402

__all__ = ["Precomputation", "LatticeLoss", "clear_store", "best_transshipment", "MilpRLPolicy", "MilpRLResult",
           "milpRL"]


# ====================================================================================================
# STEP 0: PRECOMPUTATION
# ====================================================================================================
# For every location j and every pair of periods k <= t the algorithm needs the distribution of the cumulative
# demand d^j_{k..t}, and from it
#     the expected value                                  (model)
#     the coefficients P_s and S_s of the linear pieces   (model)
#     the two loss functions at every multiple of step    (refinement)
# These depend on the demand distributions alone, not on the inventory and not on the cost parameters. For
# Poisson demand they form one entry for each element of the matrices of cumulative rates
# Lambda^j_{k,t} = lambda^j_k + .. + lambda^j_t. They are computed once and kept in _STORE. Two cumulative
# demands made of the same period distributions in the same sequence share one entry: within a location, between
# locations and between instances solved in the same run.
_STORE: dict = {}


def clear_store():
    """Forget the stored entries, so that the next Precomputation computes everything again."""
    _STORE.clear()


class LatticeLoss:
    """Exact loss function L(x) = E[max(d - x, 0)] and its complement L^(x) = E[max(x - d, 0)] of a demand d on
    the multiples of step (pmf[k] = P(d = k * step), m = 1 / step), at real arguments x. Both functions are
    linear between two neighbouring multiples of step, so the value at any other argument is exact as well.
    With step = 1 and integer arguments this is milp1.Loss."""

    def __init__(self, pmf: np.ndarray, m: int = 1):
        self.m = m
        self._steps = Loss(pmf)                                    # in steps, at whole numbers of steps
        self.mean = self._steps.mean / m

    def _at(self, x, fn) -> np.ndarray:
        a = np.asarray(x, dtype=float) * self.m                    # the argument in steps
        r = np.rint(a)
        if np.all(np.abs(a - r) < 1e-9):                           # multiples of step
            return fn(r.astype(np.int64)) / self.m
        lo = np.floor(a)
        w = a - lo
        lo = lo.astype(np.int64)
        return ((1.0 - w) * fn(lo) + w * fn(lo + 1)) / self.m

    def complement(self, x) -> np.ndarray:
        return self._at(x, self._steps.complement)

    def loss(self, x) -> np.ndarray:
        return self._at(x, self._steps.loss)


@dataclass
class _Entry:
    pmf: np.ndarray             # distribution of the cumulative demand, on the multiples of step
    mean: float
    P: np.ndarray               # P_s, s = 1..N
    S: np.ndarray               # S_s, s = 1..N
    loss: LatticeLoss           # exact loss function and its complement


class Precomputation:
    """prepare() computes every entry; `seconds` is the time spent, `computed` and `reused` count the entries."""

    def __init__(self, pmfs: Sequence[Sequence[np.ndarray]], probs: np.ndarray, step: float = 1.0):
        self.pmfs = [[np.asarray(p, float) for p in pm] for pm in pmfs]
        self.J, self.T = len(self.pmfs), len(self.pmfs[0])
        self.probs = np.asarray(probs, float)
        self.m = per_unit(step)
        self.unit = 1.0 / self.m                                   # a demand of k steps is k * unit
        self.dbar = [[self.unit * float(np.dot(np.arange(len(p)), p)) for p in pm] for pm in self.pmfs]   # expected period demands
        self._key0 = (self.probs.tobytes(), self.m)
        self._keys = [[p.tobytes() for p in pm] for pm in self.pmfs]
        self._table: dict = {}                    # (j, k) -> list over t = k..T of _Entry
        self.seconds = 0.0
        self.computed = 0
        self.reused = 0

    def entries(self, j: int, k: int):
        """Entries of d^j_{k..t} for t = k..T; location j from 0, period k from 1."""
        hit = self._table.get((j, k))
        if hit is not None:
            return hit
        t0 = time.perf_counter()
        out, key, acc = [], self._key0, None
        for t in range(k - 1, self.T):
            key = key + (self._keys[j][t],)
            e = _STORE.get(key)
            if e is None:
                acc = self.pmfs[j][t] if acc is None else np.convolve(acc, self.pmfs[j][t])
                P, S = partition_coefficients(acc, self.probs)                  # S in steps
                e = _STORE[key] = _Entry(acc, self.unit * float(np.dot(np.arange(len(acc)), acc)), P, self.unit * S,
                                         LatticeLoss(acc, self.m))
                self.computed += 1
            else:
                self.reused += 1
            acc = e.pmf
            out.append(e)
        self._table[(j, k)] = out
        self.seconds += time.perf_counter() - t0
        return out

    def prepare(self):
        for j in range(self.J):
            for k in range(1, self.T + 1):
                self.entries(j, k)
        return self


# ====================================================================================================
# STEP 3: the search over the transshipments of one period
# ====================================================================================================
def _enumerate(state, tables, R, v):
    """Lists the set D(state) and returns the first minimum of U(delta) + sum_j tables[j](state^j + delta^j)."""
    J = len(state)
    i = np.asarray(state, dtype=float)
    s = np.floor(np.maximum(i, 0.0)).astype(np.int64)       # whole units on hand
    S = int(s.sum())
    f = [np.asarray(tables[j](i[j] - s[j] + np.arange(S + 1)), dtype=float) for j in range(J)]
    D = feasible_net_inflows(s)
    val = v * np.maximum(D, 0).sum(axis=1) + R * min_shipments_rows(D)
    for j in range(J):
        val = val + f[j][D[:, j] + s[j]]
    r = int(np.argmin(val))                                 # first minimum
    return tuple(int(x) for x in D[r]), float(val[r])


def _integer_program(state, tables, R, v, env=None):
    """The same minimum by a small integer program with the variables W^{jm} and delta^{jm} of one period; each
    tables[j] enters as a piecewise linear function with a breakpoint at every level that location j can reach,
    i^j - s^j + 0, 1, ..., S. The shipments are integers, so the solution sits on these breakpoints."""
    J = len(state)
    i = [float(x) for x in state]
    s = [int(math.floor(max(x, 0.0))) for x in i]           # whole units on hand
    S = sum(s)
    m = gp.Model("refine", env=env) if env is not None else gp.Model("refine")
    m.Params.OutputFlag = 0
    m.Params.Threads = 1
    m.Params.MIPGap = 0.0
    arcs = [(j, l) for j in range(J) for l in range(J) if j != l and s[j] > 0]
    W = m.addVars(arcs, vtype=GRB.INTEGER, lb=0, name="W")
    d = m.addVars(arcs, vtype=GRB.BINARY, name="delta")
    x = {}
    for j in range(J):
        xs = i[j] - s[j] + np.arange(S + 1, dtype=float)
        ys = np.asarray(tables[j](xs), dtype=float)
        x[j] = m.addVar(lb=float(xs[0]), ub=float(xs[-1]), name=f"x{j + 1}")
        if len(xs) > 1:
            m.setPWLObj(x[j], xs.tolist(), ys.tolist())
        else:
            m.ObjCon = m.ObjCon + float(ys[0])
        out = gp.quicksum(W[j, l] for l in range(J) if (j, l) in W)
        inn = gp.quicksum(W[l, j] for l in range(J) if (l, j) in W)
        m.addConstr(x[j] == i[j] - out + inn)
        if s[j] > 0:
            m.addConstr(out <= s[j])
    for (j, l) in arcs:
        m.addConstr(W[j, l] <= s[j] * d[j, l])
        d[j, l].Obj = R
        W[j, l].Obj = v
    m.ModelSense = GRB.MINIMIZE
    m.optimize()
    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"refinement at state {tuple(i)}: Gurobi status {m.Status}")
    delta = tuple(int(round(x[j].X - i[j])) for j in range(J))
    m.dispose()
    # the value of the returned integer action, not the objective value reported by the solver
    val = transshipment_cost(delta, R, v) + sum(float(tables[j](np.array([i[j] + delta[j]]))[0]) for j in range(J))
    return delta, float(val)


def best_transshipment(state: Sequence[float], tables: Sequence[Callable], R: float, v: float,
                       method: str = "auto", env=None) -> Tuple[Tuple[int, ...], float]:
    """min over the feasible net inflows delta of U(delta) + sum_j tables[j](state^j + delta^j).
    state is a real vector; tables[j] maps an array of inventory levels to values. Returns (delta, value).
    method: "enumerate", "milp", or "auto" = enumerate for J <= 3 and milp otherwise."""
    if method == "auto":
        method = "enumerate" if len(state) <= 3 else "milp"
    if method == "enumerate":
        return _enumerate(state, tables, R, v)
    if method == "milp":
        return _integer_program(state, tables, R, v, env)
    raise ValueError("method must be 'auto', 'enumerate' or 'milp'")


# ====================================================================================================
# THE DECISION RULE: steps 1 to 3 in one period
# ====================================================================================================
class MilpRLPolicy:
    """decide(k, state) -> (net inflow of the transshipment, reorder quantities) at period k (from 1) and
    observed inventory state. A decision is computed once for each (k, state) and then kept."""

    def __init__(self, pmfs: Sequence[Sequence[np.ndarray]],
                 K: float, z: float, R: float, v: float, h: float, b: float,
                 distribution: str, partitions: int = 10, refine: bool = True, search: str = "auto",
                 threads: int = 1, env=None, time_limit: Optional[float] = None, mip_gap: Optional[float] = None,
                 zero_piece: bool = True, step: float = 1.0):
        self.pmfs = [list(p) for p in pmfs]
        self.J = len(self.pmfs)
        self.T = len(self.pmfs[0])
        self.K, self.z, self.R, self.v, self.h, self.b = K, z, R, v, h, b
        self.step, self.m = step, per_unit(step)
        self.probs = cell_probabilities(distribution, partitions)
        self.refine, self.search = refine, search
        self.threads, self.env = threads, env
        self.time_limit, self.mip_gap = time_limit, mip_gap
        self.zero_piece = zero_piece
        self.pre = Precomputation(self.pmfs, self.probs, step)              # step 0, run by prepare()
        self._milp: Dict[int, MILP] = {}
        self.cache: Dict[tuple, tuple] = {}
        self.plans: Dict[tuple, object] = {}
        self.time_build = 0.0           # making the solver models
        self.time_milp = 0.0            # step 1
        self.time_refine = 0.0          # step 3
        self.decision_times = []        # (k, seconds) of each decision computed: step 1 plus step 3
        self.last = {}                  # times of the last decision computed

    # -- step 0 ----------------------------------------------------------------------------------------
    def prepare(self):
        self.pre.prepare()
        return self

    @property
    def time_precompute(self) -> float:
        return self.pre.seconds

    # -- the model of period k, built once -------------------------------------------------------------
    def milp(self, k: int) -> MILP:
        if k not in self._milp:
            for j in range(self.J):                  # entries of the precomputation; no time if prepare() was run
                self.pre.entries(j, k)
            self._milp[k] = MILP(k, self.pmfs, self.K, self.z, self.R, self.v, self.h, self.b, self.probs,
                                 threads=self.threads, env=self.env, time_limit=self.time_limit,
                                 mip_gap=self.mip_gap, zero_piece=self.zero_piece, pre=self.pre, step=self.step)
            self.time_build += self._milp[k].seconds_build
        return self._milp[k]

    def tables(self, k: int, plan):
        """The functions g^j_k of the module docstring, one per location."""
        Ls = [[e.loss for e in self.pre.entries(j, k)] for j in range(self.J)]
        net = plan.net_inflow()
        later = np.vstack([np.zeros((1, self.J), dtype=np.int64), np.cumsum(net[1:], axis=0)])
        base = np.cumsum(plan.Q, axis=0) + later                              # base[s, j]

        def make(j):
            def g(x):
                x = np.asarray(x)
                val = np.zeros(x.shape)
                for s in range(len(base)):
                    val = val + self.h * Ls[j][s].complement(x + base[s, j]) + self.b * Ls[j][s].loss(x + base[s, j])
                return val
            return g

        return [make(j) for j in range(self.J)]

    # -- steps 1 to 3 ----------------------------------------------------------------------------------
    def _state(self, state):
        """Integers when step = 1 and the inventory is integer, as in the paper; real numbers otherwise."""
        if self.m == 1 and all(float(x) == int(x) for x in state):
            return tuple(int(x) for x in state)
        return tuple(float(x) for x in state)

    def decide(self, k: int, state: Sequence[float]):
        state = self._state(state)
        key = (k,) + state
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        mdl = self.milp(k)
        plan = mdl.solve(state)                                               # step 1
        self.time_milp += plan.seconds
        t_ref = 0.0
        if self.refine:                                                       # step 3
            t1 = time.perf_counter()
            delta, _ = best_transshipment(state, self.tables(k, plan), self.R, self.v, self.search, self.env)
            t_ref = time.perf_counter() - t1
            self.time_refine += t_ref
        else:
            delta = tuple(int(x) for x in plan.net_inflow()[0])
        dec = (delta, tuple(int(q) for q in plan.Q[0]))                       # step 2: the reorders of period k
        self.cache[key] = dec
        self.plans[key] = plan
        self.decision_times.append((k, plan.seconds + t_ref))
        try:
            gap = float(mdl.model.MIPGap)
        except (AttributeError, gp.GurobiError):
            gap = float("nan")
        self.last = dict(k=k, seconds_milp=plan.seconds, seconds_refine=t_ref, mip_gap=gap)
        return dec

    def dispose(self):
        for mdl in self._milp.values():
            mdl.model.dispose()
        self._milp = {}


# ====================================================================================================
# THE ALGORITHM: precomputation, then the decision rule along simulated demand paths
# ====================================================================================================
@dataclass
class MilpRLResult:
    etc: float                       # estimated expected total cost from the opening inventory
    half_width: float                # half-width of the confidence interval
    n_paths: int
    precision_attained: bool
    first_decision: tuple            # (net inflow, reorder quantities) at the opening inventory
    first_shipments: list            # the shipments of that decision: (from, to, quantity), locations from 1
    seconds_precompute: float        # step 0
    seconds_build: float             # making the solver models
    seconds_milp: float              # step 1, all decisions
    seconds_refine: float            # step 3, all decisions
    seconds_total: float             # wall time of the whole call
    n_decisions: int                 # distinct (k, state) pairs at which a decision was computed
    seconds_per_decision: float      # mean of step 1 plus step 3 over these decisions
    seconds_decision_k1: float       # step 1 plus step 3 at k = 1
    entries_computed: int            # cumulative demands computed in step 0
    entries_reused: int
    policy: MilpRLPolicy = field(repr=False)
    sim: SimResult = field(repr=False)


def milpRL(demand_means: Sequence[Sequence[float]],
           K: float, z: float, R: float, v: float, h: float, b: float,
           distribution: str = "poisson", cv=None, tail: float = 1e-4,
           initial_inventory: Optional[Sequence[float]] = None, partitions: int = 10,
           refine: bool = True, search: str = "auto",
           rel_precision: float = 1e-3, confidence: float = 0.95, seed: int = 1234,
           batch_size: int = 1000, min_paths: int = 1000, max_paths: int = 1_000_000,
           n_paths: Optional[int] = None, threads: int = 1,
           time_limit: Optional[float] = None, mip_gap: Optional[float] = None,
           zero_piece: bool = True, verbose: bool = False, step: float = 1.0, pmfs=None) -> MilpRLResult:
    """Demand paths are added until the half-width of the confidence interval is at most rel_precision times the
    estimated expected total cost, or max_paths is reached; with n_paths given, exactly n_paths paths are used.
    step: demand values are multiples of step. initial_inventory: any real vector. pmfs: an explicit demand law,
    pmfs[j][t][k] = P(d^j_t = k * step), in the place of demand_means, distribution and cv."""
    t0 = time.perf_counter()
    explicit = pmfs is not None
    pmfs = period_pmfs(demand_means, distribution, cv, tail, step, pmfs)
    J = len(pmfs)
    i0 = (0,) * J if initial_inventory is None else tuple(initial_inventory)
    env = gurobi_env()
    policy = None
    try:
        policy = MilpRLPolicy(pmfs, K, z, R, v, h, b, "explicit" if explicit else distribution, partitions,
                              refine=refine, search=search, threads=threads, env=env, time_limit=time_limit,
                              mip_gap=mip_gap, zero_piece=zero_piece, step=step)
        policy.prepare()                                                      # step 0
        first = policy.decide(1, i0)
        sim = simulate(policy, pmfs, i0, K, z, R, v, h, b, seed=seed, rel_precision=rel_precision,
                       confidence=confidence, batch_size=batch_size, min_paths=min_paths, max_paths=max_paths,
                       n_paths=n_paths, verbose=verbose, step=step)
    finally:
        if policy is not None:
            policy.dispose()                         # the decisions in policy.cache remain usable
        env.dispose()
    dt = np.array([s for _, s in policy.decision_times])
    dt1 = np.array([s for k, s in policy.decision_times if k == 1])
    ships = [(j + 1, m + 1, q) for j, m, q in shipment_plan(first[0])]
    return MilpRLResult(etc=sim.mean, half_width=sim.half_width, n_paths=sim.n, precision_attained=sim.attained,
                        first_decision=first, first_shipments=ships, seconds_precompute=policy.time_precompute,
                        seconds_build=policy.time_build, seconds_milp=policy.time_milp,
                        seconds_refine=policy.time_refine, seconds_total=time.perf_counter() - t0,
                        n_decisions=len(dt), seconds_per_decision=float(dt.mean()),
                        seconds_decision_k1=float(dt1.mean()), entries_computed=policy.pre.computed,
                        entries_reused=policy.pre.reused, policy=policy, sim=sim)
