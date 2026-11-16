"""
NOTINUSE_guessingOptimalStructure/threshold_variants.py : three versions of the threshold property on the counterexample family.

    python threshold_variants.py

For every instance and period it counts, for location 1 and location 2,
    all      rows of the post-transshipment state space in which the location strictly reorders at some inventory
             and strictly does not reorder at a lower one                       (the property of the former conjecture)
    reached  the same, restricted to the post-transshipment states that the optimal first stage reaches from some
             opening state of the window
    opening  rows of the opening state space in which the location reorders after the optimal transshipment at
             some inventory and does not reorder at a lower one, the other opening inventory fixed
All comparisons are strict with the tolerance TOL of structure_tests.py.
"""
from __future__ import annotations

import multiprocessing as mp
import os

import numpy as np

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from common import solve
from conjecture_search import Instance3
from structure_tests import TOL, second_stage_values, transshipment


def variants(sol, t):
    w, lev, lo, lo_w = sol.win(), sol.levels, sol.lo, sol.lo_w
    m = len(lev)
    vals = second_stage_values(sol, t)                       # whole range
    W = transshipment(sol, t)
    x1 = lev[:, None] - W
    x2 = lev[None, :] + W
    reached = np.zeros(vals[0].shape, dtype=bool)
    reached[x1 - lo_w, x2 - lo_w] = True
    out = {}
    for loc in (1, 2):
        with_r = np.minimum(vals[loc], vals[3])
        without = np.minimum(vals[0], vals[3 - loc])
        sr, sn = with_r < without - TOL, without < with_r - TOL            # strict reorder, strict no reorder
        # opening states: decision at the post-transshipment state of the optimal first stage
        o_r, o_n = sr[x1 - lo_w, x2 - lo_w], sn[x1 - lo_w, x2 - lo_w]
        a_r, a_n, rc = sr[w, w], sn[w, w], reached[w, w]
        if loc == 2:
            a_r, a_n, rc, o_r, o_n = a_r.T, a_n.T, rc.T, o_r.T, o_n.T
        cnt = dict(all=0, reached=0, opening=0)
        ex = {}
        for k in range(m):
            for name, R_, N_ in (("all", a_r[:, k], a_n[:, k]), ("reached", a_r[:, k] & rc[:, k], a_n[:, k] & rc[:, k]),
                                 ("opening", o_r[:, k], o_n[:, k])):
                r, n = np.flatnonzero(R_), np.flatnonzero(N_)
                if len(r) and len(n) and n.min() < r.max():
                    cnt[name] += 1
                    ex.setdefault(name, dict(other=int(lev[k]), no_reorder_at=int(lev[n.min()]), reorder_at=int(lev[r.max()])))
        out[loc] = (cnt, ex)
    return out


def run(inst):
    sol = solve(inst, cache=False)
    return inst.name, [variants(sol, t) for t in range(sol.T)]


if __name__ == "__main__":
    todo = [Instance3("ASYM2", 2, float(K), 2, d, c, b=20.0, mu1=[1, 30], mu2=[1, 12])
            for (d, c), Ks in ((("poisson", None), range(54, 61)), (("normal", 0.1), range(43, 55, 2)),
                               (("normal", 0.3), range(56, 66, 2))) for K in Ks]
    with mp.get_context("spawn").Pool(10) as pool:
        res = pool.map(run, todo)
    tot = dict(all=0, reached=0, opening=0)
    for name, per in res:
        for t, v in enumerate(per, 1):
            for loc in (1, 2):
                cnt, ex = v[loc]
                if any(cnt.values()):
                    print(name, "t", t, "location", loc, cnt, ex)
                for k in tot:
                    tot[k] += cnt[k] > 0
    print("instance-period-locations with a violation:", tot, "of", len(res) * 2 * 2)
