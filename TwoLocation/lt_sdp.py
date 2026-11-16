"""
Exact SDP solvers for the two-location lateral transshipment problem.

    SDP1  simultaneous formulation, one minimisation over the joint action (W, QA, QB);
          follows LateralTransshipment_slow/LT2locationsBackwards.java, method computeLTinstance
    SDP2  decoupled formulation, orders on the post-transshipment state first, then transshipment;
          follows LateralTransshipment/LTbackwards_2stages.java, method solveInstance

Both functions take exactly the same arguments and implement the operationally corrected Java code, so they
return the same value function at every state and period (Proposition 1), up to floating point rounding.

Model, as in the Java code
    * State (i1, i2) is the inventory at the start of a period, integer grid [min_inventory, max_inventory].
    * W > 0 moves W units from location 1 to location 2, W < 0 moves |W| units from location 2 to location 1.
      Only physical stock can be moved: -max(i2, 0) <= W <= max(i1, 0). The post-transshipment state
      (i1 - W, i2 + W) must lie in the grid.
    * Orders QA, QB each run over 0..max_quantity, independently of the state, as in LTbackwards_2stages and
      in the single-location sS / sQ code. The default max_quantity is the number of grid levels, which is
      the Java default stateSpaceSize.
    * Costs: transshipment R + v|W| for W != 0, ordering K + z Q for Q > 0, holding h and backorder b on the
      closing inventory of each period, terminal value 0.

Truncation and renormalisation, as in sdp/util/Demand.java and every SDP in sdp-master
    * Demand in a period is truncated at its 1 - tail quantile and the probabilities are renormalised:
        distribution="poisson": maxDemand = smallest k with F(k) >= 1 - tail, p_k = f(k) / F(maxDemand)
        distribution="normal" : sd = cv * mean, maxDemand = Math.round(F^{-1}(1 - tail)),
                                p_k = (F(k + .5) - F(k - .5)) / (F(maxDemand + .5) - F(-.5))
      for k = 0..maxDemand, following tabulateProbabilityDiscrete / tabulateProbabilityContinuous.
    * Outcomes whose closing state leaves the grid are dropped and the expected holding, backorder and future
      cost is divided by the probability mass of the outcomes that remain. The grid should be wide enough
      for this to matter only far from the states of interest; orders_beyond_grid() reports how often an
      optimal order-up-to level leaves the grid, which should be 0 for a well chosen grid.

Corrections relative to the Java code, applied throughout (code review of 29 Oct 2026)
    * Unit ordering cost is z. LT2locationsBackwards.computeActionCost charged instance.v.
    * Demand support is 0..maxDemand in both models. LT2locationsBackwards.generatePMF stopped at maxDemand - 1.
    * Ordering and transshipment costs are deterministic, so they are added to the conditional expectation
      and are not divided by the in-grid probability mass. Both Java models divided them as well, and they did
      so differently (SDP-1 divided both costs, SDP-2 only the ordering cost).
    * W covers the full feasible range in SDP2. LTbackwards_2stages looped W = min..max+1, which capped
      transshipment from location 2 to location 1 at |min_inventory|.
    * The order range in SDP1 no longer depends on the pre-transshipment state, which LT2locationsBackwards
      used in its cases 2 and 3; both models use 0..max_quantity.
    * Ties are broken by the first action in the Java enumeration order of each model.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional, Sequence, Union

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import norm, poisson

__all__ = ["SDP1", "SDP2", "SDPResult", "demand_pmf"]


# ----------------------------------------------------------------------------------------------------
# results
# ----------------------------------------------------------------------------------------------------
@dataclass
class SDPResult:
    """Value function and optimal policy. Arrays are indexed [t, i1 - min, i2 - min] with t = 0..T-1."""
    method: str
    grid: np.ndarray                    # inventory levels min..max
    C: np.ndarray                       # expected total cost from period t+1 to T, start-of-period state
    W: np.ndarray                       # optimal transshipment at the start-of-period state
    QA: np.ndarray                      # optimal order at location 1, placed after the optimal transshipment
    QB: np.ndarray                      # optimal order at location 2, placed after the optimal transshipment
    seconds: float
    params: dict = field(repr=False)
    C_order: Optional[np.ndarray] = field(default=None, repr=False)   # SDP2: second stage value on the post-transshipment state
    Q_order: Optional[np.ndarray] = field(default=None, repr=False)   # SDP2: optimal (QA, QB) on the post-transshipment state

    @property
    def min_inventory(self) -> int:
        return int(self.grid[0])

    def _idx(self, i1: int, i2: int):
        return i1 - self.min_inventory, i2 - self.min_inventory

    def value(self, i1: int = 0, i2: int = 0, period: int = 1) -> float:
        """C_period(i1, i2), with period counted from 1 as in the paper."""
        a, b = self._idx(i1, i2)
        return float(self.C[period - 1, a, b])

    def action(self, i1: int = 0, i2: int = 0, period: int = 1):
        """Optimal (W, QA, QB) at state (i1, i2) in the given period, counted from 1."""
        a, b = self._idx(i1, i2)
        t = period - 1
        return int(self.W[t, a, b]), int(self.QA[t, a, b]), int(self.QB[t, a, b])

    def orders_beyond_grid(self) -> int:
        """Number of (period, state) pairs whose optimal order-up-to level exceeds max_inventory."""
        hi = int(self.grid[-1])
        x1 = self.grid[None, :, None] - self.W
        x2 = self.grid[None, None, :] + self.W
        return int(np.sum((x1 + self.QA > hi) | (x2 + self.QB > hi)))


# ----------------------------------------------------------------------------------------------------
# demand: truncation at the 1 - tail quantile and renormalisation
# ----------------------------------------------------------------------------------------------------
def demand_pmf(mean: float, distribution: str = "poisson", cv: Optional[float] = None, tail: float = 1e-4) -> np.ndarray:
    """Probabilities of demand 0..maxDemand, as tabulated in LTbackwards_2stages."""
    dist = distribution.lower()
    if dist == "poisson":
        m = int(poisson.ppf(1 - tail, mean))
        k = np.arange(m + 1)
        return poisson.pmf(k, mean) / poisson.cdf(m, mean)
    if dist == "normal":
        if cv is None:
            raise ValueError("distribution='normal' needs cv; the standard deviation is cv * mean")
        sd = cv * mean
        m = int(math.floor(norm.ppf(1 - tail, mean, sd) + 0.5))          # Math.round
        k = np.arange(m + 1)
        num = norm.cdf(k + 0.5, mean, sd) - norm.cdf(k - 0.5, mean, sd)
        return num / (norm.cdf(m + 0.5, mean, sd) - norm.cdf(-0.5, mean, sd))
    raise ValueError("distribution must be 'poisson' or 'normal'")


def _tabulate(means, distribution, cv, tail):
    T = len(means)
    if cv is None or np.ndim(cv) == 0:
        cvs = [cv] * T
    else:
        cvs = list(cv)
        if len(cvs) != T:
            raise ValueError("cv must be a scalar or have one value per period")
    return [demand_pmf(means[t], distribution, cvs[t], tail) for t in range(T)]


# ----------------------------------------------------------------------------------------------------
# shared pieces
# ----------------------------------------------------------------------------------------------------
def _setup(demand_mean1, demand_mean2, min_inventory, max_inventory, K, z, R, v, h, b,
           distribution, cv, tail, max_quantity):
    if len(demand_mean1) != len(demand_mean2):
        raise ValueError("demand_mean1 and demand_mean2 must have the same length")
    if min_inventory > 0 or max_inventory < 0:
        raise ValueError("the grid must contain 0")
    grid = np.arange(min_inventory, max_inventory + 1)
    n = len(grid)
    qmax = n if max_quantity is None else int(max_quantity)
    q = np.arange(qmax + 1)
    c_q = np.where(q > 0, K + z * q, 0.0)
    Y = np.arange(min_inventory, max_inventory + qmax + 1)               # order-up-to levels
    hb = np.where(grid >= 0, h * grid, -b * grid)
    pA = _tabulate(demand_mean1, distribution, cv, tail)
    pB = _tabulate(demand_mean2, distribution, cv, tail)
    params = dict(demand_mean1=list(demand_mean1), demand_mean2=list(demand_mean2),
                  min_inventory=min_inventory, max_inventory=max_inventory, K=K, z=z, R=R, v=v, h=h, b=b,
                  distribution=distribution, cv=cv, tail=tail, max_quantity=max_quantity)
    return len(demand_mean1), grid, n, qmax, c_q, Y, hb, pA, pB, params


def _transition(p, Y, grid):
    """M[y, e] = P(demand = y - e) for order-up-to level y and closing level e in the grid."""
    d = Y[:, None] - grid[None, :]
    ok = (d >= 0) & (d < len(p))
    return np.where(ok, p[np.clip(d, 0, len(p) - 1)], 0.0)


def _conditional_expectation(pA, pB, Y, grid, hb, C_next):
    """E[y1, y2] = expected holding, backorder and future cost given order-up-to levels (y1, y2),
    over the demand outcomes that keep both closing levels in the grid, divided by their probability mass."""
    MA, MB = _transition(pA, Y, grid), _transition(pB, Y, grid)
    F = hb[:, None] + hb[None, :]
    if C_next is not None:
        F = F + C_next
    S = MA @ F @ MB.T
    mass = np.outer(MA.sum(1), MB.sum(1))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(mass > 0, S / mass, np.inf)


def _u(W, R, v):
    return np.where(W != 0, R + v * np.abs(W), 0.0)


# ----------------------------------------------------------------------------------------------------
# SDP1
# ----------------------------------------------------------------------------------------------------
def _transshipments_java_order(i1, i2, lo, hi):
    """Feasible W in the enumeration order of LT2locationsBackwards.generateFeasibleActions."""
    if i1 <= 0 and i2 <= 0:          # case 4
        W = np.array([0])
    elif i1 <= 0 < i2:               # case 3: t = 0..i2, W = -t
        W = -np.arange(0, i2 + 1)
    elif i1 > 0 >= i2:               # case 2: t = 0..i1, W = t
        W = np.arange(0, i1 + 1)
    else:                            # case 1: t = -i2..i1, W = t
        W = np.arange(-i2, i1 + 1)
    x1, x2 = i1 - W, i2 + W
    keep = (x1 >= lo) & (x1 <= hi) & (x2 >= lo) & (x2 <= hi)
    return W[keep]


def SDP1(demand_mean1: Sequence[float], demand_mean2: Sequence[float],
         min_inventory: int, max_inventory: int,
         K: float, z: float, R: float, v: float, h: float, b: float,
         distribution: str = "poisson", cv: Union[None, float, Sequence[float]] = None,
         tail: float = 1e-4, max_quantity: Optional[int] = None,
         no_initial_transship: bool = False, no_initial_order: bool = False,
         verbose: bool = False) -> SDPResult:
    """Simultaneous formulation: C_t(i) = min over (W, QA, QB) of u(W) + c(QA) + c(QB) + E[...]."""
    t_start = time.perf_counter()
    T, grid, n, qmax, c_q, Y, hb, pA, pB, params = _setup(
        demand_mean1, demand_mean2, min_inventory, max_inventory, K, z, R, v, h, b,
        distribution, cv, tail, max_quantity)
    lo, hi = min_inventory, max_inventory
    C_pair = c_q[:, None] + c_q[None, :]

    C = np.zeros((T, n, n))
    Wopt = np.zeros((T, n, n), dtype=int)
    QAopt = np.zeros((T, n, n), dtype=int)
    QBopt = np.zeros((T, n, n), dtype=int)

    for t in range(T - 1, -1, -1):
        tp = time.perf_counter()
        E = _conditional_expectation(pA[t], pB[t], Y, grid, hb, None if t == T - 1 else C[t + 1])
        win = sliding_window_view(E, (qmax + 1, qmax + 1))               # win[x1, x2, QA, QB] = E[x1 + QA, x2 + QB]
        for a in range(n):
            i1 = int(grid[a])
            for bb in range(n):
                i2 = int(grid[bb])
                W = np.array([0]) if (t == 0 and no_initial_transship) else _transshipments_java_order(i1, i2, lo, hi)
                block = win[i1 - W - lo, i2 + W - lo]                        # (len(W), QA, QB)
                if t == 0 and no_initial_order:
                    block = block[:, :1, :1]
                    cost = _u(W, R, v)[:, None, None] + block
                else:
                    cost = _u(W, R, v)[:, None, None] + C_pair[None] + block
                k = int(np.argmin(cost))                                     # first minimum in (W, QA, QB) order
                kw, ka, kb = np.unravel_index(k, cost.shape)
                C[t, a, bb] = cost[kw, ka, kb]
                Wopt[t, a, bb], QAopt[t, a, bb], QBopt[t, a, bb] = W[kw], ka, kb
        if verbose:
            print(f"SDP1 period {t + 1}: {time.perf_counter() - tp:.2f}s")

    params.update(no_initial_transship=no_initial_transship, no_initial_order=no_initial_order)
    return SDPResult("SDP1", grid, C, Wopt, QAopt, QBopt, time.perf_counter() - t_start, params)


# ----------------------------------------------------------------------------------------------------
# SDP2
# ----------------------------------------------------------------------------------------------------
def SDP2(demand_mean1: Sequence[float], demand_mean2: Sequence[float],
         min_inventory: int, max_inventory: int,
         K: float, z: float, R: float, v: float, h: float, b: float,
         distribution: str = "poisson", cv: Union[None, float, Sequence[float]] = None,
         tail: float = 1e-4, max_quantity: Optional[int] = None,
         no_initial_transship: bool = False, no_initial_order: bool = False,
         verbose: bool = False) -> SDPResult:
    """Decoupled formulation: second stage over (QA, QB) on the post-transshipment state, then first stage over W."""
    t_start = time.perf_counter()
    T, grid, n, qmax, c_q, Y, hb, pA, pB, params = _setup(
        demand_mean1, demand_mean2, min_inventory, max_inventory, K, z, R, v, h, b,
        distribution, cv, tail, max_quantity)
    lo, hi = min_inventory, max_inventory
    C_pair = c_q[:, None] + c_q[None, :]
    i1 = grid[:, None] + 0 * grid[None, :]
    i2 = 0 * grid[:, None] + grid[None, :]

    C = np.zeros((T, n, n))
    C_order = np.zeros((T, n, n))
    Q_order = np.zeros((T, n, n, 2), dtype=int)
    Wopt = np.zeros((T, n, n), dtype=int)

    for t in range(T - 1, -1, -1):
        tp = time.perf_counter()
        E = _conditional_expectation(pA[t], pB[t], Y, grid, hb, None if t == T - 1 else C[t + 1])
        win = sliding_window_view(E, (qmax + 1, qmax + 1))
        # second stage: orders on the post-transshipment state, QA outer loop and QB inner loop
        for a in range(n):
            if t == 0 and no_initial_order:
                C_order[t, a] = win[a, :n, 0, 0]
                Q_order[t, a] = 0
                continue
            flat = (C_pair[None] + win[a, :n]).reshape(n, -1)
            k = np.argmin(flat, axis=1)
            C_order[t, a] = flat[np.arange(n), k]
            Q_order[t, a, :, 0], Q_order[t, a, :, 1] = np.divmod(k, qmax + 1)
        # first stage: transshipment, W ascending, strict "<" keeps the first minimum
        best = np.full((n, n), np.inf)
        bestW = np.zeros((n, n), dtype=int)
        W_range = [0] if (t == 0 and no_initial_transship) else range(-(n - 1), n)
        for W in W_range:
            x1, x2 = i1 - W, i2 + W
            ok = (W <= np.maximum(i1, 0)) & (W >= -np.maximum(i2, 0)) & (x1 >= lo) & (x1 <= hi) & (x2 >= lo) & (x2 <= hi)
            if not ok.any():
                continue
            val = np.full((n, n), np.inf)
            val[ok] = float(_u(np.array(W), R, v)) + C_order[t][x1[ok] - lo, x2[ok] - lo]
            better = val < best
            best = np.where(better, val, best)
            bestW = np.where(better, W, bestW)
        C[t], Wopt[t] = best, bestW
        if verbose:
            print(f"SDP2 period {t + 1}: {time.perf_counter() - tp:.2f}s")

    QA = np.zeros((T, n, n), dtype=int)
    QB = np.zeros((T, n, n), dtype=int)
    for t in range(T):
        xa, xb = i1 - Wopt[t] - lo, i2 + Wopt[t] - lo
        QA[t], QB[t] = Q_order[t, xa, xb, 0], Q_order[t, xa, xb, 1]

    params.update(no_initial_transship=no_initial_transship, no_initial_order=no_initial_order)
    return SDPResult("SDP2", grid, C, Wopt, QA, QB, time.perf_counter() - t_start, params,
                     C_order=C_order, Q_order=Q_order)