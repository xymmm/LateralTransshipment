"""
ml_SDP1.py : SDP-1, the simultaneous formulation for J locations (N-main.tex, Section 2).

    C_t(i) = min over a = <w, q> in A(i) = R(i) x N^J of
                 sum_{j != m} u(W^{jm}) + sum_j c(Q^j) + f_t(i, a) + E[ C_{t+1}(i') ],        C_{T+1} = 0,
    R(i) = { w : sum_m W^{jm} <= max(i^j, 0) for every j },      i' = i + Delta(w) + q - d_t.

The code follows the formulation and uses none of the results of Section 4 and Appendix A.
    * At every state i it lists every transshipment w of R(i): every quantity W^{jm}, also the w in which a
      location ships and receives (ml_common.all_transshipments).
    * For every w it evaluates every reorder vector q. Several w of one state lead to the same
      post-transshipment state i + Delta(w); the minimum over q is taken once for each of these states. It is
      taken again at every other state i, which is what distinguishes SDP-1 from SDP-2.
The number of w grows quickly with the stock on hand: a location with s units and J - 1 destinations has
C(s + J - 1, J - 1) ways to ship. SDP-1 is therefore for small instances.

    from ml_SDP1 import SDP1
    res = SDP1([[1, 1, 1], [1, 2, 1], [2, 1, 1]], min_inventory=-4, max_inventory=8, K=5, z=1, R=1, v=0.5, h=1, b=5)
    res.value()      expected total cost from the opening inventory (0, 0, 0)
    res.action()     optimal (net inflow of the transshipment, reorder quantities) at the opening inventory

Real-valued demand: step < 1 (0.5, 0.25, 0.1, ...)
    Demand values and inventory levels are then multiples of step; the quantities W^{jm} and Q^j stay whole
    units, and R(i) allows location j to ship floor(max(i^j, 0)) units. The levels with the same fractional
    parts form a class that no reorder or shipment leaves, so the minimisation below is carried out class by
    class on the whole-unit parts of the levels; the expectation over the demand links the classes.
    distribution="gamma" (with cv) is a nonnegative real-valued demand; pmfs=... gives any demand law on the
    multiples of step, pmfs[j][t][k] = P(d^j_t = k * step). A state is a tuple of real levels:
        res = SDP1(means, -4, 8, K=5, z=1, R=1, v=0.5, h=1, b=5, distribution="gamma", cv=0.5, step=0.5)
        res.value((1.5, 0, -0.5))
"""
from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from ml_common import (MLResult, all_transshipments, expectation, reorder_cost_table, reorder_minimum,  # noqa: E402
                       residue_classes, setup)

__all__ = ["SDP1"]


def SDP1(demand_means: Sequence[Sequence[float]], min_inventory: int, max_inventory: int,
         K: float, z: float, R: float, v: float, h: float, b: float,
         distribution: str = "poisson", cv=None, tail: float = 1e-4, max_quantity: Optional[int] = None,
         cap: bool = True, verbose: bool = False, step: float = 1.0, pmfs=None) -> MLResult:
    t_start = time.perf_counter()
    J, T, grid, n, qmax, c_q, Y, hb, pmfs, params, m, lattice = setup(
        demand_means, min_inventory, max_inventory, K, z, R, v, h, b, distribution, cv, tail, max_quantity, cap,
        step, pmfs)
    lo = min_inventory
    Cq = reorder_cost_table(c_q, min(qmax + 1, n) if cap else qmax + 1, J)    # sum_j c(Q^j)
    shape = (n,) * J                                                         # whole-unit levels of one class
    full = (m * n,) * J                                                      # all levels; equal to shape if step = 1
    C = np.zeros((T,) + full)
    act_delta = np.zeros((T,) + full + (J,), dtype=np.int64)
    act_Q = np.zeros((T,) + full + (J,), dtype=np.int64)

    for t in range(T - 1, -1, -1):
        tp = time.perf_counter()
        # E[y] = f_t + E[C_{t+1}] for the levels y after reordering
        E_all = expectation([pmfs[j][t] for j in range(J)], Y, lattice, hb, None if t == T - 1 else C[t + 1])
        for cls in residue_classes(m, J):                                     # one class of fractional parts
            E = E_all if m == 1 else np.ascontiguousarray(E_all[cls])
            C_cls, delta_cls, Q_cls = C[t][cls], act_delta[t][cls], act_Q[t][cls]      # views of this class
            win = None if cap else sliding_window_view(E, (qmax + 1,) * J)
            for idx in itertools.product(range(n), repeat=J):                 # every state i of the class
                state = grid[list(idx)]                                       # whole-unit parts of its levels
                D, u = all_transshipments(state, R, v)                        # every w of R(i): net inflow, cost
                post = state[None, :] + D - lo                                # indices of i + Delta(w)
                ok = np.all(post <= n - 1, axis=1)                            # stay within the inventory levels
                u, post = u[ok], post[ok]
                keys, inv = np.unique(np.ravel_multi_index(post.T, shape), return_inverse=True)
                cheapest = np.full(len(keys), np.inf)                         # cheapest w for each state i + Delta(w)
                np.minimum.at(cheapest, inv.ravel(), u)
                best, best_x, best_q = np.inf, None, None
                for key, cost_w in zip(keys, cheapest):
                    x = np.unravel_index(key, shape)
                    value_q, q = reorder_minimum(E, x, Cq, n, cap, win)       # every reorder vector q
                    if cost_w + value_q < best:
                        best, best_x, best_q = cost_w + value_q, x, q
                C_cls[idx] = best
                delta_cls[idx] = np.array(best_x) - np.array(idx)
                Q_cls[idx] = best_q
        if verbose:
            print(f"SDP1 period {t + 1}: {time.perf_counter() - tp:.2f}s", flush=True)

    return MLResult("SDP1", J, grid, C, time.perf_counter() - t_start, params, act_delta=act_delta, act_Q=act_Q,
                    step=step)
