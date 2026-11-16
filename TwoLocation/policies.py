"""
policies.py : the decision rule of Algorithm 1 at period k, and the SDP-2 optimal policy for comparison.

RecedingHorizonPolicy.decide(k, i1, i2)
    1. solve MILP-1 over k..T from the observed inventory (i1, i2)                       -> pi^_k
    2. keep the order quantities of period k:  Q~^j_k = Q^^j_k
    3. refine the transshipment by (alg.refine1):
           W~_k = argmin_{W in R_k(i)}  u(|W|) + C^pi_k(<i1 - W, i2 + W>)
       where C^pi_k(i~) is the expected cost over k..T from the post-transshipment state i~ with the decisions
       fixed at pi^_k (the transshipment of period k excluded):
           C^pi_k(i~) = sum_{t=k+1}^T u(|W^_t|) + sum_{t=k}^T sum_j [ c(Q^^j_t) + h L^(X^j_t, d^j_{k..t}) + b L(X^j_t, d^j_{k..t}) ]
           X^1_t = i~^1 + sum_{l=k}^t Q^^1_l - sum_{l=k+1}^t W^_l,   X^2_t = i~^2 + sum_{l=k}^t Q^^2_l + sum_{l=k+1}^t W^_l
       with the exact loss functions of the cumulative demand (milp1.Loss). Candidates W are scanned in ascending
       order and the first minimum is kept, as in the first stage of lt_sdp.SDP2.
    R_k(i) = { W : -max(i2, 0) <= W <= max(i1, 0) }, Section 3.1.

The decision depends only on (k, i1, i2), so it is computed once per state and stored; repeated visits of the same
state in the simulation reuse it. This is a cache of the same rule, not a change of the rule.

refine=False gives the variant without refinement (W~_k = W^_k), useful to measure what the refinement adds.

OptimalPolicy wraps an lt_sdp.SDPResult and returns its optimal action at (k, i1, i2).

QuasiMyopicPolicy is the benchmark built on Glazebrook, Paterson, Rauscher & Archibald (2015): the transshipment is
chosen as if no transshipment will ever be made again, i.e. one policy improvement step from the policy without
transshipment. Without transshipment the two locations are independent and their optimal costs are the single-location
values V^1_k, V^2_k of sdp_single.SDP_single, so at period k
       W_k = argmin_{W in R_k(i)}  u(|W|) + V^1_k(i1 - W) + V^2_k(i2 + W),
and each location then orders by its own optimal single-location policy at the post-transshipment inventory.
Candidates W are scanned in ascending order and the first minimum is kept.
"""
from __future__ import annotations

import time
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from milp1 import MILP1, Loss, cumulative_pmfs, partition_probabilities


def _u(W, R, v):
    W = np.asarray(W)
    return np.where(W != 0, R + v * np.abs(W), 0.0)


def _c(Q, K, z):
    Q = np.asarray(Q)
    return np.where(Q > 0, K + z * Q, 0.0)


class RecedingHorizonPolicy:
    def __init__(self, pmfs1: Sequence[np.ndarray], pmfs2: Sequence[np.ndarray],
                 K: float, z: float, R: float, v: float, h: float, b: float,
                 distribution: str, partitions: int = 10, refine: bool = True, threads: int = 1, env=None):
        self.pmfs = (list(pmfs1), list(pmfs2))
        self.T = len(pmfs1)
        self.K, self.z, self.R, self.v, self.h, self.b = K, z, R, v, h, b
        self.probs = partition_probabilities(distribution, partitions)
        self.refine = refine
        self.threads, self.env = threads, env
        self._milp: Dict[int, MILP1] = {}
        self._loss: Dict[int, tuple] = {}
        self.cache: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {}
        self.plans: Dict[Tuple[int, int, int], object] = {}
        self.time_build = 0.0           # building the MILP-1 models and loss tables
        self.time_milp = 0.0            # Gurobi solves
        self.time_refine = 0.0          # one-dimensional searches
        self.decision_times = []        # wall time of each decision computed (cache misses), with its k

    # -- pieces built once per k ----------------------------------------------------------------------
    def milp(self, k: int) -> MILP1:
        if k not in self._milp:
            t0 = time.perf_counter()
            self._milp[k] = MILP1(k, *self.pmfs, self.K, self.z, self.R, self.v, self.h, self.b,
                                  self.probs, threads=self.threads, env=self.env)
            self.time_build += time.perf_counter() - t0
        return self._milp[k]

    def losses(self, k: int):
        """Loss objects of d^j_{k..t}, t = k..T, for j = 1, 2."""
        if k not in self._loss:
            t0 = time.perf_counter()
            self._loss[k] = tuple([Loss(p) for p in cumulative_pmfs(pm, k - 1)] for pm in self.pmfs)
            self.time_build += time.perf_counter() - t0
        return self._loss[k]

    # -- C^pi_k ----------------------------------------------------------------------------------------
    def value_fixed(self, k: int, plan, post1, post2) -> np.ndarray:
        """C^pi_k(<post1, post2>) for arrays of post-transshipment states (closed form, loss functions)."""
        post1, post2 = np.atleast_1d(post1), np.atleast_1d(post2)
        L1, L2 = self.losses(k)
        Wlater = np.concatenate(([0], np.cumsum(plan.W[1:])))          # sum_{l=k+1}^t W^_l
        base1 = np.cumsum(plan.Q1) - Wlater
        base2 = np.cumsum(plan.Q2) + Wlater
        const = float(np.sum(_u(plan.W[1:], self.R, self.v)) + np.sum(_c(plan.Q1, self.K, self.z))
                      + np.sum(_c(plan.Q2, self.K, self.z)))
        val = np.full(post1.shape, const)
        for s in range(len(plan.W)):
            for Ls, x in ((L1[s], post1 + base1[s]), (L2[s], post2 + base2[s])):
                val = val + self.h * Ls.complement(x) + self.b * Ls.loss(x)
        return val

    # -- decision rule ---------------------------------------------------------------------------------
    def decide(self, k: int, i1: int, i2: int) -> Tuple[int, int, int]:
        """(W~_k, Q~^1_k, Q~^2_k) at period k (counted from 1) and opening inventory (i1, i2)."""
        key = (k, int(i1), int(i2))
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        t0 = time.perf_counter()
        plan = self.milp(k).solve(int(i1), int(i2))
        self.time_milp += plan.seconds
        if self.refine:
            t1 = time.perf_counter()
            Wc = np.arange(-max(i2, 0), max(i1, 0) + 1)                  # R_k(i), ascending
            val = _u(Wc, self.R, self.v) + self.value_fixed(k, plan, i1 - Wc, i2 + Wc)
            W = int(Wc[int(np.argmin(val))])                               # first minimum
            self.time_refine += time.perf_counter() - t1
        else:
            W = int(plan.W[0])
        dec = (W, int(plan.Q1[0]), int(plan.Q2[0]))
        self.cache[key] = dec
        self.plans[key] = plan
        self.decision_times.append((k, time.perf_counter() - t0))
        return dec


class OptimalPolicy:
    """Optimal action of SDP-2 (or SDP-1) at (k, i1, i2). States outside the SDP grid are counted and refused."""

    def __init__(self, res):
        self.res = res
        self.lo, self.hi = int(res.grid[0]), int(res.grid[-1])

    def decide(self, k: int, i1: int, i2: int) -> Tuple[int, int, int]:
        if not (self.lo <= i1 <= self.hi and self.lo <= i2 <= self.hi):
            raise IndexError(f"state ({i1}, {i2}) at period {k} is outside the SDP grid [{self.lo}, {self.hi}]")
        return self.res.action(i1, i2, period=k)


class QuasiMyopicPolicy:
    """Benchmark: one policy improvement step from the optimal policy without transshipment (see the module docstring).
    sol1, sol2 are sdp_single.SingleResult on the same grid; states whose post-transshipment inventory would leave the
    grid are refused."""

    def __init__(self, sol1, sol2, R: float, v: float):
        if sol1.min_inventory != sol2.min_inventory or len(sol1.grid) != len(sol2.grid):
            raise ValueError("the two single-location solutions must use the same grid")
        self.s1, self.s2 = sol1, sol2
        self.lo, self.hi = int(sol1.grid[0]), int(sol1.grid[-1])
        self.R, self.v = R, v
        self.cache: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {}

    def decide(self, k: int, i1: int, i2: int) -> Tuple[int, int, int]:
        key = (k, int(i1), int(i2))
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        lo, hi, t = self.lo, self.hi, k - 1
        Wc = np.arange(-max(i2, 0), max(i1, 0) + 1)                       # R_k(i), ascending
        x1, x2 = i1 - Wc, i2 + Wc
        ok = (x1 >= lo) & (x1 <= hi) & (x2 >= lo) & (x2 <= hi)
        if not ok.any():
            raise IndexError(f"state ({i1}, {i2}) at period {k} is outside the grid [{lo}, {hi}]")
        Wc, x1, x2 = Wc[ok], x1[ok], x2[ok]
        val = _u(Wc, self.R, self.v) + self.s1.C[t, x1 - lo] + self.s2.C[t, x2 - lo]
        j = int(np.argmin(val))                                            # first minimum
        dec = (int(Wc[j]), int(self.s1.Q[t, x1[j] - lo]), int(self.s2.Q[t, x2[j] - lo]))
        self.cache[key] = dec
        return dec
