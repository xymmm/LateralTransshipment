"""
sdp_single.py : exact SDP for one location without transshipment, following sS/sS.java (method solveInstance).

    SDP_single(demand_mean, min_inventory, max_inventory, K, z, h, b, distribution="poisson", cv=None, ...)

The arguments have the same meaning as in lt_sdp.SDP2, for one location. The recursion is

    C_t(i) = min_{Q in 0..max_quantity} { c(Q) + E[ h max(0, i + Q - d_t) + b max(0, d_t - i - Q) + C_{t+1}(i + Q - d_t) ] },
    C_{T+1} = 0,

on the integer grid [min_inventory, max_inventory], with the same conventions as lt_sdp:
    * demand truncated at its 1 - tail quantile and renormalised (lt_sdp.demand_pmf);
    * closing levels outside the grid are dropped and the expectation is divided by the in-grid mass; the ordering
      cost is deterministic and is not divided (sS.java divided it as well, the correction made in lt_sdp);
    * orders 0..max_quantity, by default the number of grid levels;
    * ties broken by the smallest order, as the strict "<" of sS.getOptimalAction.

Without transshipment the two locations of the lateral transshipment problem are independent, so
SDP2(..., R = infinity) equals SDP_single(location 1) + SDP_single(location 2) at every state; tests2.py checks this.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Sequence, Union

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from lt_sdp import _tabulate, _transition

__all__ = ["SDP_single", "SingleResult"]


@dataclass
class SingleResult:
    """Arrays indexed [t, i - min_inventory], t = 0..T-1."""
    grid: np.ndarray
    C: np.ndarray                    # optimal expected cost from period t+1 to T at opening inventory i
    Q: np.ndarray                    # optimal order at opening inventory i
    seconds: float
    params: dict = field(repr=False)

    @property
    def min_inventory(self) -> int:
        return int(self.grid[0])

    def value(self, i: int = 0, period: int = 1) -> float:
        return float(self.C[period - 1, i - self.min_inventory])

    def action(self, i: int = 0, period: int = 1) -> int:
        return int(self.Q[period - 1, i - self.min_inventory])


def SDP_single(demand_mean: Sequence[float], min_inventory: int, max_inventory: int,
               K: float, z: float, h: float, b: float,
               distribution: str = "poisson", cv: Union[None, float, Sequence[float]] = None,
               tail: float = 1e-4, max_quantity: Optional[int] = None) -> SingleResult:
    t_start = time.perf_counter()
    if min_inventory > 0 or max_inventory < 0:
        raise ValueError("the grid must contain 0")
    grid = np.arange(min_inventory, max_inventory + 1)
    n = len(grid)
    qmax = n if max_quantity is None else int(max_quantity)
    q = np.arange(qmax + 1)
    c_q = np.where(q > 0, K + z * q, 0.0)
    Y = np.arange(min_inventory, max_inventory + qmax + 1)                  # order-up-to levels
    hb = np.where(grid >= 0, h * grid, -b * grid)
    pmfs = _tabulate(demand_mean, distribution, cv, tail)
    T = len(demand_mean)

    C = np.zeros((T, n))
    Q = np.zeros((T, n), dtype=int)
    for t in range(T - 1, -1, -1):
        M = _transition(pmfs[t], Y, grid)                                  # M[y, e] = P(d_t = y - e)
        f = hb + (C[t + 1] if t < T - 1 else 0.0)
        mass = M.sum(1)
        with np.errstate(all="ignore"):
            E = np.where(mass > 0, (M @ f) / mass, np.inf)                 # E[y]
        cost = c_q[None, :] + sliding_window_view(E, qmax + 1)[:n]         # cost[i, Q] = c(Q) + E[i + Q]
        Q[t] = np.argmin(cost, axis=1)                                      # first minimum
        C[t] = cost[np.arange(n), Q[t]]
    params = dict(demand_mean=list(demand_mean), min_inventory=min_inventory, max_inventory=max_inventory,
                  K=K, z=z, h=h, b=b, distribution=distribution, cv=cv, tail=tail, max_quantity=max_quantity)
    return SingleResult(grid, C, Q, time.perf_counter() - t_start, params)
