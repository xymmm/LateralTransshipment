"""
ml_SDP2.py : SDP-2, the two-stage formulation for J locations (N-main.tex, Section 3).

    first stage     C~_t(i)  = min over w in R(i) of  sum_{j != m} u(W^{jm}) + C~~_t(i + Delta(w))
    second stage    C~~_t(x) = min over q in N^J of   sum_j c(Q^j) + f~_t(x, q) + E[ C~_{t+1}(x + q - d_t) ]
    C~_{T+1} = 0,   R(i) = { w : sum_m W^{jm} <= max(i^j, 0) for every j }.

The code follows the formulation and uses none of the results of Section 4 and Appendix A.
    * Second stage: at every post-transshipment state x every reorder vector q is evaluated. This is done once
      for each x and kept, which is the saving over SDP-1.
    * First stage: at every state i every transshipment w of R(i) is listed: every quantity W^{jm}, also the w
      in which a location ships and receives (ml_common.all_transshipments).
ml_SDP2quick.py computes the same value function with Lemma 1 of Section 4 and Lemmas 2 and 3 of Appendix A.

    from ml_SDP2 import SDP2
    res = SDP2([[1, 1, 1], [1, 2, 1], [2, 1, 1]], min_inventory=-4, max_inventory=8, K=5, z=1, R=1, v=0.5, h=1, b=5)
    res.value()      expected total cost from the opening inventory (0, 0, 0)
    res.action()     optimal (net inflow of the transshipment, reorder quantities) at the opening inventory

Real-valued demand: step < 1 (0.5, 0.25, 0.1, ...)
    Demand values and inventory levels are then multiples of step; the quantities W^{jm} and Q^j stay whole
    units, and R(i) allows location j to ship floor(max(i^j, 0)) units. The levels with the same fractional
    parts form a class that no reorder or shipment leaves, so the minimisation below is carried out class by
    class on the whole-unit parts of the levels; the expectation over the demand links the classes.
    distribution="gamma" (with cv) is a nonnegative real-valued demand; pmfs=... gives any demand law on the
    multiples of step, pmfs[j][t][k] = P(d^j_t = k * step). A state is a tuple of real levels:
        res = SDP2(means, -4, 8, K=5, z=1, R=1, v=0.5, h=1, b=5, distribution="gamma", cv=0.5, step=0.5)
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

__all__ = ["SDP2"]


def SDP2(demand_means: Sequence[Sequence[float]], min_inventory: int, max_inventory: int,
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
    C = np.zeros((T,) + full)                                                # C~_t
    C_order = np.zeros((T,) + full)                                          # C~~_t
    act_delta = np.zeros((T,) + full + (J,), dtype=np.int64)                 # optimal Delta(w) at every state
    act_Q = np.zeros((T,) + full + (J,), dtype=np.int64)                     # optimal q at every post-transshipment state
    sec = dict(expectation=0.0, reorder=0.0, transshipment=0.0)

    for t in range(T - 1, -1, -1):
        t0 = time.perf_counter()
        # E[y] = f~_t + E[C~_{t+1}] for the levels y after reordering
        E_all = expectation([pmfs[j][t] for j in range(J)], Y, lattice, hb, None if t == T - 1 else C[t + 1])
        sec["expectation"] += time.perf_counter() - t0
        t_second = t_first = 0.0
        for cls in residue_classes(m, J):                                     # one class of fractional parts
            t1 = time.perf_counter()
            E = E_all if m == 1 else np.ascontiguousarray(E_all[cls])
            Co_cls, Q_cls = C_order[t][cls], act_Q[t][cls]                    # views of this class
            C_cls, delta_cls = C[t][cls], act_delta[t][cls]
            win = None if cap else sliding_window_view(E, (qmax + 1,) * J)

            # second stage: every reorder vector q at every post-transshipment state x
            for x in itertools.product(range(n), repeat=J):
                Co_cls[x], Q_cls[x] = reorder_minimum(E, x, Cq, n, cap, win)
            t2 = time.perf_counter()

            # first stage: every transshipment w of R(i) at every state i
            Co = np.ascontiguousarray(Co_cls).ravel()
            for idx in itertools.product(range(n), repeat=J):
                state = grid[list(idx)]                                       # whole-unit parts of the levels
                D, u = all_transshipments(state, R, v)                        # every w of R(i): net inflow, cost
                post = state[None, :] + D - lo                                # indices of i + Delta(w)
                ok = np.all(post <= n - 1, axis=1)                            # stay within the inventory levels
                D, u, post = D[ok], u[ok], post[ok]
                val = u + Co[np.ravel_multi_index(post.T, shape)]
                r = int(np.argmin(val))                                       # first minimum
                C_cls[idx] = val[r]
                delta_cls[idx] = D[r]
            t3 = time.perf_counter()
            t_second += t2 - t1
            t_first += t3 - t2
        sec["reorder"] += t_second
        sec["transshipment"] += t_first
        if verbose:
            print(f"SDP2 period {t + 1}: second stage {t_second:.2f}s, first stage {t_first:.2f}s", flush=True)

    return MLResult("SDP2", J, grid, C, time.perf_counter() - t_start, params, C_order=C_order,
                    act_delta=act_delta, act_Q=act_Q, q_on_post=True, seconds_stage=sec, step=step)
