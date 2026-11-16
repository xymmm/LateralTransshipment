"""
NOTINUSE_guessingOptimalStructure/common.py : instances and exact solves for the study of the optimal policy NOTINUSE_guessingOptimalStructure.

Nothing in the parent folder is modified. lt_sdp.SDP2 is called as it stands, and its private helpers _setup and
_conditional_expectation are imported to recompute the expected cost after reordering, which SDP2 does not return.

Notation, as in the paper
    i  = (i1, i2)   opening state of a period
    x  = (x1, x2)   post-transshipment state, x = (i1 - W, i2 + W)
    y  = (y1, y2)   inventory levels after reordering, y = x + Q
    Ct  = C~_t(i)   value of SDP-2 at the opening state              (res.C)
    Cq  = C~~_t(x)  second-stage value at the post-transshipment state (res.C_order)
    H_t(y) = z (y1 + y2) + E[ holding + backorder + C~_{t+1}(y - d) ], so that
    C~~_t(x) = -z (x1 + x2) + min_{y >= x} { K 1(y1 > x1) + K 1(y2 > x2) + H_t(y) }.

Range of inventory levels
    The analysis window [lo, hi] is the adequate range of the paper (test1_experiments.adequate_grid).
    SDP-2 is solved on [lo - T D, hi], where D is the largest truncated demand of a period, so that no closing
    inventory reachable from the window within T periods falls below the range. lt_sdp drops the demand outcomes
    that leave the range and renormalises, and with this choice nothing is dropped on the window.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from lt_sdp import SDP2, _conditional_expectation, _setup          # noqa: E402  (read only, never edited)
from sdp_single import SDP_single                                    # noqa: E402
from test1_experiments import adequate_grid, max_demand, pattern     # noqa: E402

OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)

M_LEVEL = 7
BASE_COSTS = dict(z=1.0, v=0.5, h=1.0, b=5.0)


def relation_means(name: str, T: int):
    """Expected demands of the two locations, as in test bed II (test3_experiments.relation_means)."""
    P = lambda p: pattern(p, T, M_LEVEL)
    low = lambda m: [max(1, int(np.floor(x / 3 + 0.5))) for x in m]
    table = {"STAT-STAT": (P("STAT"), P("STAT")),
             "LCY1-LCY2": (P("LCY1"), P("LCY2")),
             "STAT-STAT3": (P("STAT"), low(P("STAT")))}
    return table[name]


@dataclass
class Instance:
    relation: str
    T: int
    K: float
    R: float
    dist: str = "poisson"
    cv: float | None = None

    @property
    def name(self) -> str:
        return f"{self.relation}_T{self.T}_K{self.K:g}_R{self.R:g}"

    @property
    def label(self) -> str:
        rel = self.relation.replace("-", "–").replace("STAT3", "STAT/3")
        return f"{rel}, T = {self.T}, K = {self.K:g}, R = {self.R:g}, Poisson"

    @property
    def costs(self) -> dict:
        return dict(K=float(self.K), R=float(self.R), **BASE_COSTS)

    def means(self):
        return relation_means(self.relation, self.T)


FIRST_ROUND = [Instance(rel, 4, K, R)
               for rel in ("STAT-STAT", "LCY1-LCY2", "STAT-STAT3")
               for (K, R) in ((20, 20), (50, 2))]


@dataclass
class Solution:
    inst: Instance
    mu1: list
    mu2: list
    lo: int                 # analysis window
    hi: int
    lo_w: int               # range on which SDP-2 was solved
    hi_w: int
    D: int
    C: np.ndarray           # [t, i1 - lo_w, i2 - lo_w]   C~_t
    Cq: np.ndarray          # [t, x1 - lo_w, x2 - lo_w]   C~~_t
    W: np.ndarray           # optimal transshipment at the opening state
    Q: np.ndarray           # [t, x1, x2, 2] optimal reorders at the post-transshipment state
    H: np.ndarray           # [t, y1 - lo_w, y2 - lo_w]   H_t on the range of inventory levels
    seconds: float

    @property
    def T(self):
        return self.inst.T

    def win(self):
        """Slice of the analysis window inside the arrays."""
        return slice(self.lo - self.lo_w, self.hi - self.lo_w + 1)

    @property
    def levels(self):
        return np.arange(self.lo, self.hi + 1)


def solve(inst: Instance, hi_factor: float = 1.0, cache: bool = True) -> Solution:
    """SDP-2 on [lo - T D, ceil(hi_factor * hi)], with H_t recomputed from the value function."""
    tag = "" if hi_factor == 1.0 else f"_hi{hi_factor:g}"
    path = os.path.join(OUT, f"sol_{inst.name}{tag}.npz")
    mu1, mu2 = inst.means()
    if cache and os.path.exists(path):
        d = np.load(path)
        return Solution(inst, mu1, mu2, int(d["lo"]), int(d["hi"]), int(d["lo_w"]), int(d["hi_w"]), int(d["D"]),
                        d["C"], d["Cq"], d["W"], d["Q"], d["H"], float(d["seconds"]))
    costs = inst.costs
    lo, hi = adequate_grid(mu1, mu2, inst.dist, inst.cv, costs)
    D = max_demand(mu1, mu2, inst.dist, inst.cv)
    lo_w, hi_w = lo - inst.T * D, int(np.ceil(hi_factor * hi))
    res = SDP2(mu1, mu2, lo_w, hi_w, distribution=inst.dist, cv=inst.cv, **costs)
    # H_t(y) on the range, from the same helpers and the same arguments as SDP2
    T, grid, n, qmax, c_q, Y, hb, pA, pB, _ = _setup(mu1, mu2, lo_w, hi_w, costs["K"], costs["z"], costs["R"],
                                                     costs["v"], costs["h"], costs["b"], inst.dist, inst.cv, 1e-4, None)
    H = np.zeros((T, n, n))
    for t in range(T):
        E = _conditional_expectation(pA[t], pB[t], Y, grid, hb, None if t == T - 1 else res.C[t + 1])
        H[t] = E[:n, :n] + costs["z"] * (grid[:, None] + grid[None, :])
    sol = Solution(inst, mu1, mu2, lo, hi, lo_w, hi_w, D, res.C, res.C_order, res.W, res.Q_order, H, res.seconds)
    if cache:
        np.savez_compressed(path, lo=lo, hi=hi, lo_w=lo_w, hi_w=hi_w, D=D, C=sol.C, Cq=sol.Cq, W=sol.W, Q=sol.Q,
                            H=sol.H, seconds=sol.seconds)
    return sol


def single_location(sol: Solution, loc: int):
    """Optimal single-location policy of location loc (1 or 2) on the same range, without transshipment."""
    c = sol.inst.costs
    mu = sol.mu1 if loc == 1 else sol.mu2
    return SDP_single(mu, sol.lo_w, sol.hi_w, c["K"], c["z"], c["h"], c["b"], distribution=sol.inst.dist, cv=sol.inst.cv)
