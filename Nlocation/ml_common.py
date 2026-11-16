"""
ml_common.py : pieces shared by the solvers for J locations.

    demand          per_unit, lattice_pmf, period_pmfs, largest_demand, total_demand_quantile
    numerics        setup, expectation, residue_classes
    transshipment   all_transshipments      every vector W of the set R(i), for ml_SDP1.py and ml_SDP2.py
                    feasible_net_inflows    the set D(i) of net inflows, for ml_SDP2quick.py and ml_milpRL.py
                    min_shipments, min_shipments_rows, shipment_plan, transshipment_cost
    results         MLResult, OptimalPolicy
    simulation      demand_batch, run_paths, simulate, half_width

Conventions, the same in every solver
    * Locations j = 1..J, periods t = 1..T. demand_means has one sequence of T expected demands per location.
    * step, 1 by default: demand values and inventory levels are multiples of step, and step is 1, 1/2, 1/3, ...
      (0.5, 0.25, 0.1, ...). Reorder and shipment quantities are whole units whatever the step. With step = 1
      everything is an integer, as in the paper. With step < 1 demand and inventory take real values:
          pmfs[j][t][k] = P(demand of location j in period t = k * step),
          a location with the level x can ship floor(max(x, 0)) units.
      The levels with the same fractional parts form a class. Reorders and shipments are whole units, so they
      stay within a class, and within a class the two stages are the integer problem on the whole-unit parts of
      the levels (residue_classes). Demand is what moves the inventory from one class to another.
    * The inventory level of a location is a multiple of step whose whole-unit part lies in
      min_inventory..max_inventory, both integers: the levels are min_inventory, min_inventory + step, ...,
      max_inventory + 1 - step. With step = 1 they are the integers min_inventory..max_inventory.
    * cap=True, the default: a reorder cannot raise the whole-unit part of a level above max_inventory, so
      max_inventory is the largest stock that a location can hold. cap=False is the convention of lt_sdp.py, in
      which the level after reordering may exceed max_inventory by up to max_quantity; it is used to compare
      with the two-location code and needs step = 1.
    * Demand is truncated at its 1 - tail quantile and renormalised. A demand outcome that takes a closing level
      below min_inventory, or above max_inventory, is dropped and the remaining probabilities are renormalised,
      as in lt_sdp.py.
    * A transshipment is described by its net inflow delta = (delta^1, ..., delta^J): delta^j is what location j
      receives minus what it ships. shipment_plan(delta) lists the shipments of a cheapest transshipment.

lt_sdp.py is not modified; its pmf and transition helpers are imported.
"""
from __future__ import annotations

import itertools
import math
import sys
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import gamma as gamma_law
from scipy.stats import norm, poisson
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TwoLocation.lt_sdp import _tabulate, _transition, demand_pmf  # noqa: E402


# ----------------------------------------------------------------------------------------------------
# demand
# ----------------------------------------------------------------------------------------------------
def per_unit(step: float = 1.0) -> int:
    """Number of steps in one unit of supply. Demand values and inventory levels are multiples of step, reorder
    and shipment quantities are whole units, so 1 / step must be a whole number."""
    m = int(round(1.0 / float(step)))
    if m < 1 or abs(m * float(step) - 1.0) > 1e-9:
        raise ValueError("step must be 1, 1/2, 1/3, ... (for example 0.5, 0.25, 0.1): "
                         "one unit of supply is a whole number of steps")
    return m


def lattice_pmf(mean: float, distribution: str = "poisson", cv: Optional[float] = None, tail: float = 1e-4,
                step: float = 1.0) -> np.ndarray:
    """p[k] = P(demand = k * step), truncated at the 1 - tail quantile and renormalised.

    poisson   whole-number demand. With step < 1 the probabilities sit on the whole numbers.
    normal    standard deviation cv * mean. A value is rounded to the nearest multiple of step and the values
              that round below zero are removed. With step = 1 this is lt_sdp.demand_pmf.
    gamma     nonnegative demand with standard deviation cv * mean, rounded to the nearest multiple of step.
    With step = 1 and Poisson or normal demand the result is exactly the pmf of lt_sdp.demand_pmf."""
    m = per_unit(step)
    dist = distribution.lower()
    if m == 1 and dist in ("poisson", "normal"):
        return demand_pmf(mean, dist, cv, tail)
    if dist == "poisson":
        p = demand_pmf(mean, "poisson", None, tail)
        out = np.zeros((len(p) - 1) * m + 1)
        out[::m] = p
        return out
    if dist not in ("normal", "gamma"):
        raise ValueError("distribution must be 'poisson', 'normal' or 'gamma'")
    if cv is None:
        raise ValueError(f"distribution='{dist}' needs cv; the standard deviation is cv * mean")
    if mean <= 0:
        return np.array([1.0])
    law = norm(mean, cv * mean) if dist == "normal" else gamma_law(1.0 / cv ** 2, scale=mean * cv ** 2)
    top = int(math.floor(law.ppf(1 - tail) * m + 0.5))
    k = np.arange(top + 1)
    num = law.cdf((k + 0.5) / m) - law.cdf((k - 0.5) / m)
    return num / (law.cdf((top + 0.5) / m) - law.cdf(-0.5 / m))


def period_pmfs(demand_means: Optional[Sequence[Sequence[float]]], distribution="poisson", cv=None, tail=1e-4,
                step: float = 1.0, pmfs=None):
    """Truncated and renormalised period pmfs: pmfs[j][t][k] = P(d^j_t = k * step). With step = 1 and Poisson
    or normal demand they are built exactly as lt_sdp builds them.
    pmfs, if given, is an explicit demand law on the multiples of step and is returned as arrays; demand_means,
    distribution and cv are then not used."""
    per_unit(step)
    if pmfs is not None:
        out = [[np.asarray(p, dtype=float) for p in pm] for pm in pmfs]
        for pm in out:
            for p in pm:
                if p.ndim != 1 or len(p) == 0 or np.any(p < 0) or abs(float(p.sum()) - 1.0) > 1e-9:
                    raise ValueError("every pmfs[j][t] must be a vector of probabilities that sum to 1")
        return out
    out = []
    for means in demand_means:
        T = len(means)
        cvs = [cv] * T if (cv is None or np.ndim(cv) == 0) else list(cv)
        if len(cvs) != T:
            raise ValueError("cv must be a scalar or have one value per period")
        out.append([lattice_pmf(means[t], distribution, cvs[t], tail, step) for t in range(T)])
    return out


def largest_demand(demand_means, distribution="poisson", cv=None, tail=1e-4, step: float = 1.0, pmfs=None):
    """D: the largest truncated demand of a period at any location, in units."""
    m = per_unit(step)
    top = max(len(p) - 1 for pm in period_pmfs(demand_means, distribution, cv, tail, step, pmfs) for p in pm)
    return top if m == 1 else top / m


def total_demand_quantile(demand_means, distribution="poisson", cv=None, tail=1e-4, step: float = 1.0, pmfs=None):
    """The 1 - tail quantile of the total demand of all locations over the horizon, in units."""
    m = per_unit(step)
    if m == 1 and pmfs is None and distribution.lower() in ("poisson", "normal"):
        means = np.array([float(x) for mm in demand_means for x in mm])
        if distribution.lower() == "poisson":
            return int(poisson.ppf(1 - tail, means.sum()))
        cvs = np.array([float(c) for _ in demand_means
                        for c in ([cv] * len(demand_means[0]) if np.ndim(cv) == 0 else cv)])
        return int(math.ceil(norm.ppf(1 - tail, means.sum(), math.sqrt(((cvs * means) ** 2).sum()))))
    acc = np.array([1.0])                                   # distribution of the total demand, by convolution
    for pm in period_pmfs(demand_means, distribution, cv, tail, step, pmfs):
        for p in pm:
            acc = np.convolve(acc, p)
    k = min(int(np.searchsorted(np.cumsum(acc), 1 - tail, side="left")), len(acc) - 1)
    return k if m == 1 else k / m


# ----------------------------------------------------------------------------------------------------
# numerics shared by the three SDPs
# ----------------------------------------------------------------------------------------------------
def setup(demand_means, min_inventory, max_inventory, K, z, R, v, h, b, distribution, cv, tail, max_quantity, cap,
          step: float = 1.0, pmfs=None):
    """Returns J, T, grid, n, qmax, c_q, Y, hb, pmfs, params, m, lattice.
    grid      the whole-unit inventory levels min_inventory..max_inventory, n of them
    c_q[Q]    reorder cost of Q units
    m         steps per unit (1 / step)
    lattice   with m = 1 the grid; with m > 1 the indices 0..m*n-1 of the levels min_inventory + index * step
    Y, hb     the levels after reordering, and the holding or backorder cost of every closing level, both on
              the lattice: they are the arguments of expectation()
    pmfs      pmfs[j][t][k] = P(d^j_t = k * step)"""
    m = per_unit(step)
    explicit = pmfs is not None
    pmfs = period_pmfs(demand_means, distribution, cv, tail, step, pmfs)
    J = len(pmfs)
    T = len(pmfs[0])
    if J < 2 or any(len(pm) != T for pm in pmfs):
        raise ValueError("demand_means needs one sequence of the same length per location, for at least 2 locations")
    if min_inventory > 0 or max_inventory < 0:
        raise ValueError("the inventory levels must contain 0")
    if int(min_inventory) != min_inventory or int(max_inventory) != max_inventory:
        raise ValueError("min_inventory and max_inventory are whole units")
    min_inventory, max_inventory = int(min_inventory), int(max_inventory)
    grid = np.arange(min_inventory, max_inventory + 1)
    n = len(grid)
    qmax = n if max_quantity is None else int(max_quantity)
    q = np.arange(qmax + 1)
    c_q = np.where(q > 0, K + z * q, 0.0)
    if m == 1:
        lattice = grid
        Y = grid if cap else np.arange(min_inventory, max_inventory + qmax + 1)
        hb = np.where(grid >= 0, h * grid, -b * grid)
    else:
        if not cap:
            raise ValueError("cap=False is the convention of lt_sdp.py and needs step = 1")
        lattice = np.arange(m * n)
        level = (min_inventory * m + lattice) / m                 # the real inventory levels
        Y = lattice
        hb = np.where(level >= 0, h * level, -b * level)
    params = dict(demand_means=None if demand_means is None else [list(mm) for mm in demand_means],
                  min_inventory=min_inventory, max_inventory=max_inventory, K=K, z=z, R=R, v=v, h=h, b=b,
                  distribution="explicit pmfs" if explicit else distribution, cv=cv,
                  tail=tail, max_quantity=max_quantity, cap=cap, step=step)
    return J, T, grid, n, qmax, c_q, Y, hb, pmfs, params, m, lattice


def residue_classes(m: int, J: int):
    """Index slices of the m^J classes of inventory levels with the same fractional parts, on arrays indexed by
    the lattice. Within a class the whole-unit parts of the levels are the grid min_inventory..max_inventory, and
    a location with the whole-unit part g can ship max(g, 0) units, so the routines for integer levels apply
    to each class without change. With m = 1 there is one class, which holds all levels."""
    for r in itertools.product(range(m), repeat=J):
        yield tuple(slice(a, None, m) for a in r)


def expectation(pmfs_t, Y, grid, hb, C_next):
    """E[y^1, ..., y^J]: expected holding and backorder cost of the period plus the expected value of the next
    stage, given the levels y after reordering. This is the term f + E[C_{t+1}] of the formulations."""
    J, n = len(pmfs_t), len(grid)
    F = np.zeros((n,) * J) if C_next is None else C_next.copy()
    for j in range(J):
        F += hb.reshape([-1 if a == j else 1 for a in range(J)])
    A, mass = F, 1.0
    for j in range(J):
        M = _transition(pmfs_t[j], Y, grid)                                   # M[y, e] = P(d^j = y - e)
        with np.errstate(all="ignore"):
            A = np.moveaxis(np.tensordot(M, A, axes=(1, j)), 0, j)
        mass = mass * M.sum(1).reshape([-1 if a == j else 1 for a in range(J)])
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(mass > 0, A / mass, np.inf)


def reorder_cost_table(c_q, nq: int, J: int) -> np.ndarray:
    """Cq[Q^1, ..., Q^J] = sum_j c(Q^j) for Q^j = 0..nq-1."""
    Cq = np.zeros((nq,) * J)
    for j in range(J):
        Cq = Cq + c_q[:nq].reshape([-1 if a == j else 1 for a in range(J)])
    return Cq


def reorder_minimum(E, x_idx, Cq, n: int, cap: bool, win=None):
    """min over the reorder vectors q of sum_j c(Q^j) + E[x + q] at one post-transshipment state x, given by
    its indices x_idx. Every q is evaluated. Returns (value, q) with the first minimum.
    cap=True: q may raise each level up to max_inventory. cap=False: win is the sliding window view of E."""
    if cap:
        m = [min(n - int(a), Cq.shape[0]) for a in x_idx]
        block = E[tuple(slice(int(a), int(a) + mm) for a, mm in zip(x_idx, m))] + Cq[tuple(slice(0, mm) for mm in m)]
    else:
        block = win[tuple(int(a) for a in x_idx)] + Cq
    k = int(np.argmin(block))
    return float(block.flat[k]), np.unravel_index(k, block.shape)


# ----------------------------------------------------------------------------------------------------
# transshipment, as in the formulation: every vector W of R(i)
# ----------------------------------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _outflows(stock: int, others: int):
    """All ways in which one location with `stock` units on hand ships to its `others` destinations."""
    return tuple(o for o in itertools.product(range(stock + 1), repeat=others) if sum(o) <= stock)


def all_transshipments(state, R: float, v: float):
    """Every transshipment W = (W^{jm}) of the set R(state): sum_m W^{jm} <= max(state^j, 0) for every j.
    Returns one row per W: its net inflow (array of shape (rows, J)) and its cost sum_{j != m} u(W^{jm}).
    The list includes the W in which a location ships and receives, and several W share one net inflow."""
    J = len(state)
    D = np.zeros((1, J), dtype=np.int64)
    u = np.zeros(1)
    for j in range(J):
        outs = _outflows(max(int(state[j]), 0), J - 1)
        Dj = np.zeros((len(outs), J), dtype=np.int64)
        uj = np.zeros(len(outs))
        dest = [m for m in range(J) if m != j]
        for r, o in enumerate(outs):
            Dj[r, j] = -sum(o)
            for m, w in zip(dest, o):
                Dj[r, m] += w
                uj[r] += (R + v * w) if w > 0 else 0.0
        D = (D[:, None, :] + Dj[None, :, :]).reshape(-1, J)
        u = (u[:, None] + uj[None, :]).reshape(-1)
    return D, u


def count_transshipments(state) -> int:
    """Number of vectors W in R(state)."""
    J = len(state)
    return math.prod(math.comb(max(int(s), 0) + J - 1, J - 1) for s in state)


# ----------------------------------------------------------------------------------------------------
# transshipment by its net inflow (N-main.tex, Lemma 1 and Corollary 1)
# ----------------------------------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _best_partition(vals: Tuple[int, ...]):
    """vals: nonzero integers with sum 0. Returns the partition into the largest number of groups with sum 0,
    as a tuple of tuples of positions in vals."""
    n = len(vals)
    sums = [0] * (1 << n)
    for mask in range(1, 1 << n):
        low = mask & -mask
        sums[mask] = sums[mask ^ low] + vals[low.bit_length() - 1]
    f = [-1] * (1 << n)                       # f[mask] = largest number of groups of a zero-sum mask
    choice = [0] * (1 << n)                   # the group that contains the lowest position of mask
    f[0] = 0
    for mask in range(1, 1 << n):
        if sums[mask] != 0:
            continue
        low = mask & -mask
        f[mask], choice[mask] = 1, mask
        sub = (mask - 1) & mask
        while sub:
            if (sub & low) and sums[sub] == 0 and 1 + f[mask ^ sub] > f[mask]:
                f[mask], choice[mask] = 1 + f[mask ^ sub], sub
            sub = (sub - 1) & mask
    groups, mask = [], (1 << n) - 1
    while mask:
        g = choice[mask]
        groups.append(tuple(p for p in range(n) if g >> p & 1))
        mask ^= g
    return tuple(groups)


def min_shipments(delta: Sequence[int]) -> int:
    """Smallest number of shipments that realises the net inflow delta: p - g of Lemma 1."""
    nz = tuple(int(d) for d in delta if d != 0)
    if len(nz) <= 3:
        return max(len(nz) - 1, 0)
    return len(nz) - len(_best_partition(nz))


def min_shipments_rows(D: np.ndarray) -> np.ndarray:
    """min_shipments of every row of D. With at most five locations two groups with zero net inflow exist
    if and only if at least four entries are nonzero and two of the nonzero entries sum to zero, and three
    groups need six locations, so the count is vectorised; with more locations the rows are counted one by one."""
    D = np.asarray(D)
    J = D.shape[1]
    nz = D != 0
    p = nz.sum(axis=1)
    out = np.maximum(p - 1, 0)
    if J <= 3:
        return out
    if J <= 5:
        pair = np.zeros(len(D), dtype=bool)
        for a in range(J):
            for b in range(a + 1, J):
                pair |= nz[:, a] & nz[:, b] & (D[:, a] + D[:, b] == 0)
        return np.where((p >= 4) & pair, p - 2, out)
    return np.array([min_shipments(d) for d in D], dtype=np.int64)


def transshipment_cost(delta: Sequence[int], R: float, v: float) -> float:
    """U(delta): cost of the cheapest transshipment with the net inflow delta."""
    return v * sum(int(d) for d in delta if d > 0) + R * min_shipments(delta)


def shipment_plan(delta: Sequence[int]) -> List[Tuple[int, int, int]]:
    """A cheapest transshipment for the net inflow delta: list of (from j, to m, quantity), locations from 0."""
    idx = [j for j, d in enumerate(delta) if d != 0]
    if not idx:
        return []
    vals = tuple(int(delta[j]) for j in idx)
    if sum(vals) != 0:
        raise ValueError("the net inflow must sum to 0")
    plan = []
    for group in _best_partition(vals):
        send = [[idx[p], -vals[p]] for p in group if vals[p] < 0]
        recv = [[idx[p], vals[p]] for p in group if vals[p] > 0]
        a = b = 0
        while a < len(send) and b < len(recv):
            q = min(send[a][1], recv[b][1])
            plan.append((send[a][0], recv[b][0], q))
            send[a][1] -= q
            recv[b][1] -= q
            a += send[a][1] == 0
            b += recv[b][1] == 0
    return plan


def feasible_net_inflows(state: Sequence[int], hi: Optional[int] = None) -> np.ndarray:
    """The set D(i): the net inflows that are feasible at one state, one per row. A location ships at most its
    stock on hand and receives at most the stock of the others. Rows are listed with every coordinate
    descending, so that with J = 2 the sequence is W = -delta^1 ascending, as in lt_sdp.
    hi, if given, removes the rows that raise a location above hi."""
    i = np.asarray(state, dtype=np.int64)
    J = len(i)
    s = np.maximum(i, 0)
    S = int(s.sum())
    axes = [np.arange(S - s[j], -s[j] - 1, -1) for j in range(J - 1)]
    mesh = np.meshgrid(*axes, indexing="ij")
    cols = [m.ravel() for m in mesh]
    last = -sum(cols)
    D = np.stack(cols + [last], axis=1)
    ok = last >= -s[-1]
    if hi is not None:
        ok &= np.all(i[None, :] + D <= hi, axis=1)
    return D[ok]


# ----------------------------------------------------------------------------------------------------
# results
# ----------------------------------------------------------------------------------------------------
@dataclass
class MLResult:
    """Value function and optimal policy of an SDP. Arrays are indexed [t, a^1, ..., a^J], t = 0..T-1, where
    a^j = (i^j - min_inventory) / step is the lattice index of the level i^j. With step = 1 it is i^j - min.

    value(state, period)     C_period(state); the expected total cost from the opening inventory is value()
    action(state, period)    optimal (net inflow of the transshipment, reorder quantities), both in whole units
    A state is a tuple of inventory levels, each a multiple of step.
    """
    method: str
    J: int
    grid: np.ndarray                         # whole-unit inventory levels min..max
    C: np.ndarray                            # value at the start-of-period state
    seconds: float
    params: dict = field(repr=False)
    C_order: Optional[np.ndarray] = field(default=None, repr=False)   # value on the post-transshipment state (SDP-2)
    act_delta: Optional[np.ndarray] = field(default=None, repr=False)  # optimal net inflow at every state (SDP1, SDP2)
    act_Q: Optional[np.ndarray] = field(default=None, repr=False)      # optimal reorders (SDP1: at the state; SDP2: at the post-transshipment state)
    q_on_post: bool = False
    argq: Optional[list] = field(default=None, repr=False)             # SDP2quick: nested minimisers of the second stage
    deltas: Optional[np.ndarray] = field(default=None, repr=False)     # SDP2quick with first_stage="net": net inflows scanned
    delta_cost: Optional[np.ndarray] = field(default=None, repr=False)
    delta_index: Optional[np.ndarray] = field(default=None, repr=False)  # the same, optimal row at every state if tracked
    seconds_stage: dict = field(default_factory=dict, repr=False)
    step: float = 1.0                        # demand values and inventory levels are multiples of step

    @property
    def min_inventory(self) -> int:
        return int(self.grid[0])

    @property
    def max_inventory(self) -> int:
        return int(self.grid[-1])

    @property
    def per_unit(self) -> int:
        return per_unit(self.step)

    @property
    def states(self) -> int:
        return (len(self.grid) * self.per_unit) ** self.J

    def _idx(self, state):
        """Lattice index of every level of the state."""
        m = self.per_unit
        if m == 1:
            return tuple(int(s) - self.min_inventory for s in state)
        idx = []
        for s in state:
            a = (float(s) - self.min_inventory) * m
            r = int(round(a))
            if abs(a - r) > 1e-6:
                raise ValueError(f"the inventory level {s} is not a multiple of the step {self.step}")
            idx.append(r)
        return tuple(idx)

    def value(self, state: Optional[Sequence[float]] = None, period: int = 1) -> float:
        """C_period(state), period counted from 1. The default state is (0, ..., 0)."""
        state = (0,) * self.J if state is None else state
        return float(self.C[(period - 1,) + self._idx(state)])

    # -- SDP2quick: second stage ------------------------------------------------------------------------
    def reorders(self, post_state: Sequence[float], period: int = 1) -> Tuple[int, ...]:
        """Optimal reorder quantities at a post-transshipment state."""
        if self.argq is None:
            return tuple(int(x) for x in self.act_Q[(period - 1,) + self._idx(post_state)])
        args = self.argq[period - 1]
        idx = list(self._idx(post_state))
        m = self.per_unit
        Q = []
        for j in range(self.J):
            q = int(args[j][tuple(idx)])
            Q.append(q)
            idx[j] += m * q                       # later axes are looked up at the level after the reorder of axis j
        return tuple(Q)

    # -- SDP2quick: first stage -------------------------------------------------------------------------
    def first_stage(self, state: Sequence[float], period: int = 1):
        """(optimal net inflow, value) at one state, by listing the set D(state): the first minimum in the
        sequence of feasible_net_inflows. The value equals value(state, period)."""
        lo, hi, m = self.min_inventory, self.max_inventory, self.per_unit
        a = np.asarray(self._idx(state), dtype=np.int64)                      # lattice indices of the state
        whole = lo + a // m                                                   # whole-unit parts of the levels
        D = feasible_net_inflows(whole, hi)
        R, v = self.params["R"], self.params["v"]
        val = (v * np.maximum(D, 0).sum(axis=1) + R * min_shipments_rows(D)
               + self.C_order[period - 1][tuple((a[None, :] + m * D).T)])
        r = int(np.argmin(val))
        return tuple(int(d) for d in D[r]), float(val[r])

    def action(self, state: Optional[Sequence[float]] = None, period: int = 1):
        """Optimal (net inflow, reorder quantities) at a start-of-period state."""
        if state is None:
            state = (0,) * self.J
        elif self.per_unit == 1:
            state = tuple(int(s) for s in state)
        else:
            state = tuple(float(s) for s in state)
        key = (period - 1,) + self._idx(state)
        if self.act_delta is not None:                       # SDP1 and SDP2 store the actions
            delta = tuple(int(x) for x in self.act_delta[key])
            if not self.q_on_post:
                return delta, tuple(int(x) for x in self.act_Q[key])
        elif self.delta_index is not None:
            delta = tuple(int(d) for d in self.deltas[self.delta_index[key]])
        else:
            delta = self.first_stage(state, period)[0]
        post = tuple(s + d for s, d in zip(state, delta))
        return delta, self.reorders(post, period)

    def actions_all(self, period: int = 1):
        """SDP2quick with track=True: (delta, Q) of every state as integer arrays of shape (J, n, ..., n), with
        n the number of levels of one location on the lattice."""
        if self.delta_index is None:
            raise ValueError("solve with first_stage='net' and track=True to obtain the actions of all states at once")
        m = self.per_unit
        t, n, J = period - 1, len(self.grid) * m, self.J
        delta = np.moveaxis(self.deltas[self.delta_index[t]], -1, 0)          # (J, n, ..., n)
        idx = list(np.indices((n,) * J) + m * delta)                          # post-transshipment indices
        Q = np.zeros((J,) + (n,) * J, dtype=np.int64)
        for j in range(J):
            Q[j] = self.argq[t][j][tuple(idx)]
            idx[j] = idx[j] + m * Q[j]
        return delta, Q


class OptimalPolicy:
    """The optimal policy of an MLResult, for the simulation. A state outside its inventory levels is refused;
    with clip=True it takes the action of the nearest state of the range instead, and `outside` counts them."""

    def __init__(self, res: MLResult, clip: bool = False):
        self.res = res
        self.lo, self.hi = res.min_inventory, res.max_inventory
        self.clip = clip
        self.outside = 0
        self.cache: Dict[tuple, tuple] = {}

    def decide(self, k: int, state: Sequence[float]):
        m = self.res.per_unit
        if m > 1:
            return self._decide_lattice(k, state, m)
        state = tuple(int(x) for x in state)
        key = (k,) + state
        hit = self.cache.get(key)
        if hit is None:
            look = state
            if not all(self.lo <= x <= self.hi for x in state):
                if not self.clip:
                    raise IndexError(f"state {state} at period {k} is outside the inventory levels [{self.lo}, {self.hi}]")
                self.outside += 1
                look = tuple(min(max(x, self.lo), self.hi) for x in state)
            hit = self.cache[key] = self.res.action(look, period=k)
        return hit

    def _decide_lattice(self, k: int, state, m: int):
        """The same with step < 1: the state is looked up by the lattice indices of its levels."""
        idx = self.res._idx(state)
        key = (k,) + idx
        hit = self.cache.get(key)
        if hit is None:
            top = len(self.res.grid) * m - 1
            look = idx
            if not all(0 <= a <= top for a in idx):
                if not self.clip:
                    raise IndexError(f"state {tuple(state)} at period {k} is outside the inventory levels "
                                     f"[{self.lo}, {self.hi + 1})")
                self.outside += 1
                look = tuple(min(max(a, 0), top) for a in idx)
            hit = self.cache[key] = self.res.action(tuple(self.lo + a / m for a in look), period=k)
        return hit


# ----------------------------------------------------------------------------------------------------
# simulation of a policy: any object with decide(k, state) -> (net inflow, reorder quantities)
# ----------------------------------------------------------------------------------------------------
@dataclass
class SimResult:
    mean: float                  # estimated expected total cost
    half_width: float            # half-width of the confidence interval
    n: int                       # number of demand paths
    attained: bool               # precision reached
    costs: np.ndarray            # total cost of each path, in path sequence
    seconds: float


def demand_batch(pmfs: Sequence[Sequence[np.ndarray]], n: int, seed: int, batch: int):
    """Demand paths of batch `batch`: one array (n, T) per location. Batch m of an instance uses the generator
    seeded with (seed, m); with J = 2 the paths are those of simulate.demand_batch of the two-location code."""
    rng = np.random.default_rng([seed, batch])
    T = len(pmfs[0])
    out = []
    for pm in pmfs:
        U = rng.random((n, T))
        D = np.empty((n, T), dtype=np.int64)
        for t in range(T):
            cdf = np.cumsum(pm[t])
            D[:, t] = np.minimum(np.searchsorted(cdf, U[:, t], side="left"), len(cdf) - 1)
        out.append(D)
    return out


def run_paths(policy, D: Sequence[np.ndarray], i0, K, z, R, v, h, b, step: float = 1.0) -> np.ndarray:
    """Total cost of each path. A decision is asked once per distinct state and period.
    One period: opening inventory i -> (delta, Q) = policy.decide(k, i); closing x = i + delta + Q - d;
    cost = U(delta) + sum_j c(Q^j) + sum_j [h max(x^j, 0) + b max(-x^j, 0)].
    D holds the demands in steps (as demand_batch returns them). The inventory of a path is kept as a whole
    number of steps from the opening inventory, so no rounding error accumulates. The opening inventory may be
    any real vector; the policy is given the real inventory levels (integers when step = 1 and the opening
    inventory is integer)."""
    m = per_unit(step)
    J = len(D)
    n, T = D[0].shape
    i0 = np.asarray(i0, dtype=float)
    start = np.round(i0 * m)
    offset = i0 - start / m                                                   # zero if i0 is a multiple of step
    integer = m == 1 and not offset.any()
    x = np.tile(start.astype(np.int64), (n, 1))                               # (n, J), in steps
    cost = np.zeros(n)
    u_cache = {}
    for k in range(1, T + 1):
        states, inv = np.unique(x, axis=0, return_inverse=True)
        inv = inv.ravel()
        delta = np.empty((len(states), J), dtype=np.int64)
        Q = np.empty((len(states), J), dtype=np.int64)
        u = np.empty(len(states))
        for r, st in enumerate(states):
            level = tuple(int(a) for a in st) if integer else tuple(float(a) / m + o for a, o in zip(st, offset))
            dl, q = policy.decide(k, level)
            delta[r], Q[r] = dl, q
            if dl not in u_cache:
                u_cache[dl] = transshipment_cost(dl, R, v)
            u[r] = u_cache[dl]
        dem = np.stack([D[j][:, k - 1] for j in range(J)], axis=1)
        q = Q[inv]
        x = x + m * (delta[inv] + q) - dem
        lev = x if integer else x / m + offset                                # closing inventory levels
        cost += (u[inv] + np.where(q > 0, K + z * q, 0.0).sum(axis=1)
                 + np.where(lev >= 0, h * lev, -b * lev).sum(axis=1))
    return cost


def half_width(costs: np.ndarray, confidence: float = 0.95) -> float:
    n = len(costs)
    return float(student_t.ppf(0.5 + confidence / 2, n - 1) * costs.std(ddof=1) / np.sqrt(n))


def simulate(policy, pmfs, i0, K, z, R, v, h, b, seed: int = 1234,
             rel_precision: float = 1e-3, confidence: float = 0.95,
             batch_size: int = 1000, min_paths: int = 1000, max_paths: int = 1_000_000,
             n_paths: Optional[int] = None, verbose: bool = False, step: float = 1.0) -> SimResult:
    """Paths are added in batches until the half-width of the confidence interval of the mean path cost is at
    most rel_precision times the mean, or max_paths is reached. With n_paths given, exactly the first n_paths
    paths of the seed are simulated, which evaluates several policies on the same demand paths.
    pmfs[j][t][k] = P(d^j_t = k * step); the opening inventory i0 may be any real vector."""
    t0 = time.perf_counter()
    chunks, n, m = [], 0, 0
    while True:
        size = batch_size if n_paths is None else min(batch_size, n_paths - n)
        D = demand_batch(pmfs, batch_size, seed, m)                           # full batch, so paths do not depend on size
        chunks.append(run_paths(policy, [d[:size] for d in D], i0, K, z, R, v, h, b, step))
        n += size
        m += 1
        costs = np.concatenate(chunks)
        hw = half_width(costs, confidence)
        mean = float(costs.mean())
        if verbose:
            print(f"  paths {n:>8d}   mean {mean:.4f}   half-width {hw / mean:.5%}", flush=True)
        if n_paths is not None:
            if n >= n_paths:
                return SimResult(mean, hw, n, hw <= rel_precision * abs(mean), costs, time.perf_counter() - t0)
            continue
        if n >= min_paths and hw <= rel_precision * abs(mean):
            return SimResult(mean, hw, n, True, costs, time.perf_counter() - t0)
        if n >= max_paths:
            return SimResult(mean, hw, n, False, costs, time.perf_counter() - t0)
