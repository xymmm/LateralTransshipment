"""
ml_SDP2quick.py : SDP-2 for J locations with three results of N-main.tex: Lemma 1 (Section 4) and Lemmas 2 and 3
                  (Appendix A).

It returns the value function of ml_SDP2.py (and of ml_SDP1.py). The two stages are computed as follows.

Second stage, one location at a time (Lemma 2)
    The reorder costs are separable, so
        min_{Q^1,..,Q^J} sum_j c(Q^j) + E(x + Q)  =  min_{Q^1} { c(Q^1) + min_{Q^2} { c(Q^2) + ... } }.
    For n inventory levels this needs about J n^(J+1) evaluations per period, against n^(2J) for all vectors q.
    E stays a function of all J levels: the locations are not separated.

First stage, by the net inflow (Lemma 1 and Corollary 1)
    The second stage depends on the transshipment w through its net inflow delta alone, and the cheapest w with
    the net inflow delta costs
        U(delta) = v * sum_j max(delta^j, 0) + R * (p - g),
    p the number of locations with delta^j != 0 and g the largest number of groups with zero net inflow into
    which they divide (p - g = p - 1 for J <= 3). The first stage is a minimum over the set D(i) of net inflows.
    first_stage="net" lists every net inflow: about n^(2J-1) evaluations per period.

First stage, one shipment at a time (Lemma 3)
    first_stage="shipments", the default. The single-shipment operator
        (S F)(x) = min{ F(x), min over j != m and 1 <= q <= max(x^j, 0) of R + v q + F(x - q e_j + q e_m) }
    is applied J - 1 times to the second-stage value. For one pair (j, m) the minimum over q is a running
    minimum along the line x^j + x^m = constant, so one application needs about J (J - 1) n^J operations.

    from ml_SDP2quick import SDP2quick
    res = SDP2quick([[1, 1, 1], [1, 2, 1], [2, 1, 1]], min_inventory=-4, max_inventory=8, K=5, z=1, R=1, v=0.5, h=1, b=5)
    res.value()      expected total cost from the opening inventory (0, 0, 0)
    res.action()     optimal (net inflow of the transshipment, reorder quantities) at the opening inventory

track=True (with first_stage="net") stores the optimal net inflow of every state, for MLResult.actions_all.
Otherwise the optimal net inflow of a state is found when it is asked for (MLResult.first_stage).

Real-valued demand: step < 1 (0.5, 0.25, 0.1, ...)
    Demand values and inventory levels are then multiples of step; shipments and reorders stay whole units, and
    a location with the level x can ship floor(max(x, 0)) units. The levels with the same fractional parts form
    a class that no reorder or shipment leaves. The three results above are therefore applied class by class,
    with the routines for integer levels, and the expectation over the demand links the classes. The work
    stays proportional to the number of states, which is ((max_inventory - min_inventory + 1) / step)^J.
    distribution="gamma" (with cv) is a nonnegative real-valued demand; pmfs=... gives any demand law on the
    multiples of step, pmfs[j][t][k] = P(d^j_t = k * step). A state is a tuple of real levels:
        res = SDP2quick(means, -4, 8, K=5, z=1, R=1, v=0.5, h=1, b=5, distribution="gamma", cv=0.5, step=0.1)
        res.value((1.3, 0, -0.5)),  res.action((1.3, 0, -0.5))
"""
from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from ml_common import MLResult, expectation, min_shipments_rows, residue_classes, setup  # noqa: E402

__all__ = ["SDP2quick", "reorder_stage", "single_shipment", "first_stage_shipments", "net_inflows"]


# ----------------------------------------------------------------------------------------------------
# second stage, one location at a time (Lemma 2)
# ----------------------------------------------------------------------------------------------------
def reorder_stage(E, n, c_q):
    """min over Q of sum_j c(Q^j) + E[x + Q] for every x, one location at a time, the last location first.
    Returns the value on the n^J post-transshipment states and the minimisers args[j] (see MLResult.reorders)."""
    J = E.ndim
    A, args = E, [None] * J
    for j in range(J - 1, -1, -1):
        A = np.moveaxis(A, j, 0)
        nY = A.shape[0]
        best = np.full((n,) + A.shape[1:], np.inf)
        arg = np.zeros(best.shape, dtype=np.int16)
        for q in range(len(c_q)):
            m = min(n, nY - q)                                                # states x with x + q among the levels
            if m <= 0:
                break
            val = c_q[q] + A[q:q + m]
            better = val < best[:m]                                           # strict "<" keeps the smallest Q
            np.copyto(best[:m], val, where=better)
            np.copyto(arg[:m], np.int16(q), where=better)
        A = np.moveaxis(best, 0, j)
        args[j] = np.moveaxis(arg, 0, j)
    return np.ascontiguousarray(A), args


# ----------------------------------------------------------------------------------------------------
# first stage, one shipment at a time (Lemma 3)
# ----------------------------------------------------------------------------------------------------
def single_shipment(V, j, m, grid, R, v):
    """min over 1 <= q <= max(x^j, 0) of R + v q + V(x - q e_j + q e_m) at every state x; inf where no shipment
    from j to m is feasible. With y = x^j - q >= 0 the level of j after the shipment, the minimum is
        R + v x^j + min over 0 <= y < x^j of [ V(.., y, .., x^j + x^m - y, ..) - v y ],
    a running minimum along the line x^j + x^m = constant."""
    n = len(grid)
    A = np.moveaxis(V, (j, m), (-2, -1))                                      # [..., level of j, level of m]
    H = np.full(A.shape[:-2] + (n, 2 * n - 1), np.inf)                         # [..., y, y + level index of m]
    yi, bi = np.indices((n, n))
    H[..., yi, yi + bi] = A - v * grid[:, None]
    H[..., :int(-grid[0]), :] = np.inf                                         # the shipping location keeps y >= 0
    P = np.minimum.accumulate(H, axis=-2)
    out = R + v * grid[:, None] + P[..., np.maximum(yi - 1, 0), yi + bi]       # yi, bi: levels of j and m before
    out = np.where((grid >= 1)[:, None], out, np.inf)
    return np.moveaxis(out, (-2, -1), (j, m))


def first_stage_shipments(Co, grid, R, v):
    """At most J - 1 rounds; a round applies the single-shipment operator for every pair of locations. A round
    uses the values already improved in the same round, which only adds sequences of shipments and so does not
    change the result. The rounds stop early when a round changes nothing."""
    J = Co.ndim
    best = Co.copy()
    for _ in range(J - 1):
        before = best.copy()
        for j in range(J):
            for m in range(J):
                if j != m:
                    np.minimum(best, single_shipment(best, j, m, grid, R, v), out=best)
        if np.array_equal(best, before):
            break
    return best


# ----------------------------------------------------------------------------------------------------
# first stage, every net inflow (Corollary 1)
# ----------------------------------------------------------------------------------------------------
def net_inflows(J: int, lo: int, hi: int) -> np.ndarray:
    """All net inflows delta that are feasible at some state of [lo, hi]^J, one per row. A location can ship at
    most hi units and receive at most hi - lo units. Every coordinate is listed descending, so that with J = 2
    the sequence is W = -delta^1 ascending, as in lt_sdp.SDP2."""
    rng = range(hi - lo, -hi - 1, -1)
    rows = []
    for head in itertools.product(rng, repeat=J - 1):
        last = -sum(head)
        if -hi <= last <= hi - lo:
            rows.append(head + (last,))
    return np.array(rows, dtype=np.int64).reshape(-1, J)


def _slices(delta, lo: int, n: int):
    """Index slices (state, post state) of the states at which the net inflow delta is feasible, or None."""
    src, dst = [], []
    for d in delta:
        d = int(d)
        if d < 0:                                # ships a = -d units: on-hand stock i >= a, post state i - a >= 0
            a = -d
            if a - lo >= n:
                return None
            src.append(slice(a - lo, n))
            dst.append(slice(-lo, n - a))
        elif d > 0:                              # receives d units: post state i + d <= max_inventory
            if d >= n:
                return None
            src.append(slice(0, n - d))
            dst.append(slice(d, n))
        else:
            src.append(slice(None))
            dst.append(slice(None))
    return tuple(src), tuple(dst)


# ----------------------------------------------------------------------------------------------------
def SDP2quick(demand_means: Sequence[Sequence[float]], min_inventory: int, max_inventory: int,
              K: float, z: float, R: float, v: float, h: float, b: float,
              distribution: str = "poisson", cv=None, tail: float = 1e-4, max_quantity: Optional[int] = None,
              cap: bool = True, first_stage: str = "shipments", track: bool = False,
              verbose: bool = False, step: float = 1.0, pmfs=None) -> MLResult:
    t_start = time.perf_counter()
    J, T, grid, n, qmax, c_q, Y, hb, pmfs, params, m, lattice = setup(
        demand_means, min_inventory, max_inventory, K, z, R, v, h, b, distribution, cv, tail, max_quantity, cap,
        step, pmfs)
    lo, hi = min_inventory, max_inventory
    N = m * n                                             # levels of one location; n whole-unit levels per class
    if first_stage not in ("shipments", "net"):
        raise ValueError("first_stage must be 'shipments' or 'net'")
    if track and first_stage != "net":
        raise ValueError("track=True needs first_stage='net'")
    params["first_stage"] = first_stage

    deltas = cost = sl = None
    if first_stage == "net":
        deltas = net_inflows(J, lo, hi)
        cost = v * np.maximum(deltas, 0).sum(axis=1) + R * min_shipments_rows(deltas)      # U(delta)
        sl = [_slices(d, lo, n) for d in deltas]
        keep = [s is not None for s in sl]
        deltas, cost, sl = deltas[keep], cost[keep], [s for s in sl if s is not None]

    C = np.zeros((T,) + (N,) * J)
    C_order = np.zeros((T,) + (N,) * J)
    argq: list = [None] * T
    delta_index = np.zeros((T,) + (N,) * J, dtype=np.int32) if track else None
    sec = dict(expectation=0.0, reorder=0.0, transshipment=0.0)

    for t in range(T - 1, -1, -1):
        t0 = time.perf_counter()
        E = expectation([pmfs[j][t] for j in range(J)], Y, lattice, hb, None if t == T - 1 else C[t + 1])
        sec["expectation"] += time.perf_counter() - t0
        args_t = None if m == 1 else [np.zeros((N,) * J, dtype=np.int16) for _ in range(J)]
        for cls in residue_classes(m, J):                                     # one class of fractional parts
            t1 = time.perf_counter()
            Co, args = reorder_stage(E if m == 1 else np.ascontiguousarray(E[cls]), n, c_q)   # second stage (Lemma 2)
            if m == 1:
                del E
            t2 = time.perf_counter()
            if first_stage == "shipments":                                    # first stage (Lemma 3)
                best = first_stage_shipments(Co, grid, R, v)
            else:                                                             # first stage (Corollary 1)
                best = np.full((n,) * J, np.inf)
                if track:
                    arg = np.zeros((n,) * J, dtype=np.int32)
                    for r, (src, dst) in enumerate(sl):
                        val = cost[r] + Co[dst]
                        better = val < best[src]                               # strict "<" keeps the first minimum
                        np.copyto(best[src], val, where=better)
                        np.copyto(arg[src], np.int32(r), where=better)
                    delta_index[t][cls] = arg
                else:
                    for r, (src, dst) in enumerate(sl):
                        np.minimum(best[src], cost[r] + Co[dst], out=best[src])
            C_order[t][cls] = Co
            C[t][cls] = best
            if m == 1:
                argq[t] = args
            else:
                for j in range(J):
                    args_t[j][cls] = args[j]
            t3 = time.perf_counter()
            sec["reorder"] += t2 - t1
            sec["transshipment"] += t3 - t2
        if m > 1:
            argq[t] = args_t
            del E
        if verbose:
            print(f"SDP2quick period {t + 1}: {time.perf_counter() - t0:.2f}s", flush=True)

    return MLResult("SDP2quick", J, grid, C, time.perf_counter() - t_start, params, C_order=C_order, argq=argq,
                    deltas=deltas, delta_cost=cost, delta_index=delta_index, seconds_stage=sec, step=step)
