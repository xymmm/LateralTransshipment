"""
NOTINUSE_guessingOptimalStructure/conjecture_search.py : targeted search for a counterexample to the threshold conjecture.

    python conjecture_search.py [--workers 8]

Why these instances. A proof of condition (Q) would have to exclude the following situation. Both locations hold
enough stock for the current period, so one more unit at location 1 only adds holding cost. A reorder at location 1
in the current period nevertheless pays when it lets location 1 supply location 2 by a transshipment in the next
period, which saves one fixed reorder cost K at the price of R + v W and one period of holding. If this saving is
slightly negative at the level that just covers the demand of the period, it turns positive at higher levels,
because waiting with more stock costs holding while the reorder-up-to level stays the same. Location 1 would then
reorder at a higher inventory and not at a lower one. With demand 7 per period at both locations, z = 1, v = 0.5,
h = 1 and R = 2 the saving at the level 7 is about K - R - 7 v - 14 h = K - 19.5, so K is swept around this value,
with nearly deterministic and with Poisson demand, for horizons of 2, 3 and 4 periods.

Writes out/conjecture_search.csv (one row per instance and period) and prints every violation.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import itertools
import multiprocessing as mp
from dataclasses import dataclass, field

import numpy as np

from common import OUT, Instance, solve
from structure_tests import TOL, second_stage_values, test_second_stage

CSV = os.path.join(OUT, "conjecture_search_{}.csv")


@dataclass
class Instance3(Instance):
    z: float = 1.0
    v: float = 0.5
    h: float = 1.0
    b: float = 5.0
    mu1: list = field(default_factory=list)
    mu2: list = field(default_factory=list)

    @property
    def name(self) -> str:
        d = self.dist if self.cv is None else f"{self.dist}{self.cv:g}"
        m = "-".join(map(str, self.mu1)) + "_" + "-".join(map(str, self.mu2))
        return f"S_{m}_{d}_K{self.K:g}_R{self.R:g}_z{self.z:g}_v{self.v:g}_h{self.h:g}_b{self.b:g}"

    @property
    def costs(self) -> dict:
        return dict(K=float(self.K), R=float(self.R), z=self.z, v=self.v, h=self.h, b=self.b)

    def means(self):
        return list(self.mu1), list(self.mu2)


def q_check(sol, t):
    """Violations of condition (Q) for g^1 and g^2 on the window: list of (location, own, other, excess)."""
    w, lev = sol.win(), sol.levels
    K, z = sol.inst.costs["K"], sol.inst.costs["z"]
    vals = second_stage_values(sol, t)[:, w, w]
    base = z * (lev[:, None] + lev[None, :])
    out = []
    for loc in (1, 2):
        g = np.minimum(vals[0], vals[3 - loc]) + base
        if loc == 2:
            g = g.T
        pm = np.full_like(g, np.inf)
        pm[1:] = np.minimum.accumulate(g, axis=0)[:-1]
        sm = np.full_like(g, np.inf)
        sm[:-1] = np.flip(np.minimum.accumulate(np.flip(g, 0), axis=0), 0)[1:]
        exc = np.minimum(g - pm, g - K - sm)
        exc = np.where(np.isfinite(exc), exc, -np.inf)
        for a, b in np.argwhere(exc > TOL):
            out.append((loc, int(lev[a]), int(lev[b]), float(exc[a, b])))
    return out


def run(inst):
    try:
        sol = solve(inst, cache=False)
    except Exception as e:
        return [dict(instance=inst.name, error=repr(e))]
    rows = []
    for t in range(sol.T):
        sec = test_second_stage(sol, t)
        thr = sec["loc1"]["threshold_violations"] + sec["loc2"]["threshold_violations"]
        q = q_check(sol, t)
        worst = max(q, key=lambda e: e[3]) if q else None
        rows.append(dict(instance=inst.name, T=sol.T, t=t + 1, dist=inst.dist, cv="" if inst.cv is None else inst.cv,
                         mu1=" ".join(map(str, sol.mu1)), mu2=" ".join(map(str, sol.mu2)), **inst.costs,
                         lo=sol.lo, hi=sol.hi, threshold_violations=len(thr), Q_violations=len(q),
                         worst=str(worst) if worst else "", first_threshold=str(thr[0]) if thr else ""))
    return rows


def instances(family="asym"):
    """sym : equal demands of 7 at both locations, K around R + 7 v + 14 h. No violation was found: the other location
             reorders for both at least as cheaply, which caps the cost of waiting.
       asym: two periods, expected demands (7, 20) at location 1 and (7, 3) at location 2, b = 20. A reorder of
             location 2 for both locations is then dearer than one of location 1, and the cap is lost.
       asym2: expected demands (1, 30) and (1, 12), which widens the range of K.
       kler: the demands of asym and asym2 with K <= R + v."""
    out = []
    if family == "sym":
        for T, (d, c), R, K in itertools.product([2, 3, 4], [("normal", 0.01), ("normal", 0.1), ("normal", 0.2), ("poisson", None)],
                                                 [2, 5], np.arange(12, 24.01, 0.5)):
            out.append(Instance3("STAT-STAT", T, float(K), R, d, c, mu1=[7] * T, mu2=[7] * T))
    if family == "asym":
        for (d, c), K in itertools.product([("normal", 0.01), ("normal", 0.03), ("normal", 0.05), ("normal", 0.1),
                                            ("normal", 0.2), ("normal", 0.3), ("poisson", None)], np.arange(20, 32.01, 0.25)):
            out.append(Instance3("ASYM", 2, float(K), 2, d, c, b=20.0, mu1=[7, 20], mu2=[7, 3]))
    if family == "asym2":
        # larger second-period demands widen the range of K: the deterministic calculation gives 39 < K <= 50
        for (d, c), K in itertools.product([("poisson", None), ("normal", 0.1), ("normal", 0.2), ("normal", 0.3)],
                                           np.arange(36, 54.01, 1.0)):
            out.append(Instance3("ASYM2", 2, float(K), 2, d, c, b=20.0, mu1=[1, 30], mu2=[1, 12]))
        return out
    if family == "kler":
        # the same demands with K <= R + v, where a reorder for both locations cannot pay
        for (mu1, mu2, d, c), K, dR in itertools.product(
                [([1, 30], [1, 12], "poisson", None), ([1, 30], [1, 12], "normal", 0.1),
                 ([7, 20], [7, 3], "poisson", None), ([7, 20], [7, 3], "normal", 0.01)],
                [20, 30, 40, 50, 57, 60], [-0.5, 0.0, 10.0]):
            out.append(Instance3("KLER", 2, float(K), float(K + dR), d, c, b=20.0, mu1=mu1, mu2=mu2))
        return out
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--family", default="asym", choices=["sym", "asym", "asym2", "kler"])
    a = ap.parse_args()
    todo = instances(a.family)
    print(len(todo), "instances", flush=True)
    fields = None
    n_bad = 0
    with mp.get_context("spawn").Pool(a.workers) as pool, open(CSV.format(a.family), "w", newline="") as f:
        wr = None
        for k, rows in enumerate(pool.imap_unordered(run, todo), 1):
            if "error" in rows[0]:
                print("ERROR", rows[0], flush=True)
                continue
            if wr is None:
                wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                wr.writeheader()
            for r in rows:
                wr.writerow(r)
                if r["threshold_violations"] or r["Q_violations"]:
                    n_bad += 1
                    print("VIOLATION", r["instance"], "t", r["t"], "threshold", r["threshold_violations"], r["first_threshold"],
                          "Q", r["Q_violations"], r["worst"], flush=True)
            f.flush()
    print("instance-periods with a violation:", n_bad)


if __name__ == "__main__":
    main()
