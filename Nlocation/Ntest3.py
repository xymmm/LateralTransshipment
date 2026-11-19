"""
Ntest3.py : test bed B. The heuristic against the quasi-myopic policy and the policy without transshipment,
for two to six locations.

SDP-2 cannot be solved with more than two locations at these demand levels (Ntest1.py and the one-hour rule of
Ntest2.py), so this test bed has no optimum. It compares three policies on the same demand paths:
    NT    no transshipment: every location follows its optimal single-location policy and nothing is transshipped
    QM    quasi-myopic, after Glazebrook et al. (2015): at the start of every period it chooses the transshipment
          that minimises the transshipment cost plus the sum of the optimal single-location costs of all locations
          from the inventories after the transshipment; each location then reorders by its optimal single-location
          policy
    ALG   the heuristic (Algorithm 1, ml_milpRL.milpRL)
The instances are those of the block core of Ntest2.py at T = 4, for more locations: locations 1, 3, 5, ... have
the first pattern of a relation and locations 2, 4, 6, ... the second. With J = 2 an instance of the block core is
an instance of Ntest2.py, so Ntest3analysis.py can add the optimum there.

Run this file. Results are appended to results_Ntest3/<block>.csv, one row per instance. An instance that is
already in the file is skipped, so a run can be stopped and started again. Ntest3analysis.py makes the tables.

Blocks
    core    for every J in LOCATIONS the factorial of Ntest2.py at T = 4:
                K in {20, 30, 50}  x  R in {2, 5, 10, 20}  x  4 distributions  x  6 relations     288 per J
    large   for every J in LARGE_LOCATIONS a part of that factorial, because one decision takes seconds there:
                K in {20, 30, 50}  x  R in {2, 5, 10, 20}  x  Poisson  x  LARGE_RELATIONS          24 per J
    pivot   around the base instance (T = 4, Poisson, LCY1-LCY2, K = 30, R = 5):
                for every J in LOCATIONS:      b in {2, 10, 20};  v in {0, 1, 2};  z in {0, 2};  demand level in {5, 10, 15}
                for every J in T8_LOCATIONS:   T = 8 for the six relations, on T8_PATHS[J] demand paths
    time    computation times of the base instance on TIME_PATHS demand paths, TIME_WORKERS at the same time:
                T = 4 for every J in TIME_LOCATIONS, T = 8 for every J in T8_LOCATIONS

Numbers of locations and demand paths
    One decision of the heuristic solves the model over the remaining periods. Measured on the base instance with
    T = 4, the decision of period 1 takes 0.2 s with 3 locations, 0.8 s with 4, 17 s with 5 and 108 s with 6, and
    the decisions of period 2 take 0.16 s, 0.3 s, 3.3 s and 14 s. With J >= 3 nearly every demand path needs its own
    decisions. PATHS therefore falls with J, and the blocks core and pivot stop at 4 locations.

What is run for one instance
    1. the single-location problem of every location, by sdp_single.SDP_single    -> NT exactly, and the tables of QM
    2. NT, QM and ALG on the same demand paths (seed SEED)
Measures, all in per cent of the exact cost of NT
    saving of QM and of ALG   NT minus the policy, on the same paths
    ALG minus QM              negative when the heuristic is cheaper, with the half-width of its 95% confidence
                              interval from the paired differences
Times: one decision of ALG at k = 1 (the longest residual horizon) and on average, one decision of QM, and the
single-location problems. sdp_states is the number of states that SDP-2 would need per period on the inventory
levels of Ntest2.py; SDP-2 is not run here.

Inventory levels
    The heuristic has no inventory levels: its model takes any inventory. The single-location problem of location
    j is solved on the levels -M_j..H, where M_j is the largest demand that the location can have over the horizon
    and H is the sum of the M_j. No state outside these levels is reached from the opening inventory 0 under NT or
    QM: a location never orders above M_j, so the stock of the whole system stays below H, and a location is never
    more than M_j short. A state outside the levels would stop the instance with an error.

The two comparison policies are small classes below. They call the current solvers only: SDP_single for the
single-location problems and ml_milpRL.best_transshipment for the search over the transshipments.

Settings for clicking "Run" in PyCharm are below; command-line arguments override them:
    python Ntest3.py --block core --workers 8
    python Ntest3.py --quick                    a few instances, to check that everything runs
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")                         # one processor core per instance

# ====================================================================================================
# settings for clicking "Run" in PyCharm; command-line arguments override them
RUN_BLOCK = "all"        # "core", "large", "pivot", "time" or "all"
RUN_QUICK = False        # True: a few instances per block, to check that everything runs
N_WORKERS = 8            # instances solved at the same time in the blocks core, large and pivot
LOCATIONS = (2, 3, 4)                          # numbers of locations J in the blocks core and pivot
LARGE_LOCATIONS = (5, 6)                       # numbers of locations J in the block large
LARGE_RELATIONS = ("LCY1-LCY2", "STAT-STAT/3")
PATHS = {2: 5000, 3: 2000, 4: 1000, 5: 300, 6: 100}    # demand paths of an instance with J locations, T = 4
T8_LOCATIONS = (2, 3)                          # numbers of locations with T = 8, in the blocks pivot and time
T8_PATHS = {2: 5000, 3: 100}                   # demand paths of an instance with T = 8
TIME_LOCATIONS = (2, 3, 4, 5, 6)               # numbers of locations in the block time, T = 4
TIME_PATHS = 20
TIME_WORKERS = 4         # fewer at the same time, so that the times are less disturbed
MILP_TIME_LIMIT = None   # seconds for one solve of the model; None: no limit
# ====================================================================================================

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for _p in (HERE.parent / "TwoLocation", HERE.parent, HERE):        # milp1, lt_sdp and sdp_single are in TwoLocation
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from Ntest2 import (BASE, PIVOT, RELATIONS, SEED, TAIL, core_instances, costs_of, demand_of,  # noqa: E402
                    instance_id, largest_demands, means_text, pivot_instances, run_instances, sdp_levels)

OUT_DIR = HERE / "results_Ntest3"


# ----------------------------------------------------------------------------------------------------
# the instances
# ----------------------------------------------------------------------------------------------------
def block_instances(block: str, quick: bool = False):
    rows = []
    if block == "core":
        for J in LOCATIONS:
            rows += [dict(r, paths=PATHS[J]) for r in core_instances(J, horizons=(BASE["T"],))]
        if quick:
            rows = [dict(r, paths=40) for r in rows if r["K"] == 30 and r["R"] == 5
                    and r["dist"] == "poisson" and r["relation"] in ("LCY1-LCY2", "STAT-STAT/3")]
    elif block == "large":
        for J in LARGE_LOCATIONS:
            rows += [dict(r, paths=PATHS[J])
                     for r in core_instances(J, horizons=(BASE["T"],), distributions=(("poisson", None),),
                                             relations=LARGE_RELATIONS, block="large")]
        if quick:
            rows = [dict(r, paths=4) for r in rows if r["J"] == 5 and r["K"] == 50 and r["R"] == 5][:1]
    elif block == "pivot":
        for J in LOCATIONS:
            rows += [dict(r, paths=PATHS[J]) for r in pivot_instances(J, PIVOT)]
        for J in T8_LOCATIONS:
            rows += [dict(BASE, block="pivot", factor="T", J=J, T=8, relation=rel, paths=T8_PATHS[J])
                     for rel in RELATIONS]
        if quick:
            rows = [dict(r, paths=40) for r in rows if r["J"] == 3 and r["factor"] in ("b", "level")]
    elif block == "time":
        rows = [dict(BASE, block="time", factor="", J=J, T=4, paths=TIME_PATHS) for J in TIME_LOCATIONS]
        rows += [dict(BASE, block="time", factor="", J=J, T=8, paths=TIME_PATHS) for J in T8_LOCATIONS]
        if quick:
            rows = [dict(r, paths=10) for r in rows if r["J"] <= 4 and r["T"] == 4]
    rows.sort(key=lambda r: (-r["T"], -r["J"]))               # the long instances first
    return rows


# ----------------------------------------------------------------------------------------------------
# the comparison policies: decide(k, state) -> (net inflow of every location, reorder quantities)
# ----------------------------------------------------------------------------------------------------
class SingleLocations:
    """The single-location problem of every location: optimal costs V^j_k(x) and optimal reorders."""

    def __init__(self, means, dist, cv, costs):
        from ml_common import period_pmfs
        from sdp_single import SDP_single
        t0 = time.perf_counter()
        self.J = len(means)
        self.M = largest_demands(period_pmfs(means, dist, cv, TAIL))
        self.lo = [-m for m in self.M]
        self.hi = sum(self.M)
        self.sol = [SDP_single(means[j], self.lo[j], self.hi, costs["K"], costs["z"], costs["h"], costs["b"],
                               distribution=dist, cv=cv, tail=TAIL, max_quantity=2 * self.M[j]) for j in range(self.J)]
        self.seconds = time.perf_counter() - t0

    def cost_without_transshipment(self) -> float:
        """Exact expected total cost of NT from the opening inventory 0."""
        return float(sum(s.value(0) for s in self.sol))

    def _index(self, j: int, x):
        idx = np.rint(np.asarray(x, dtype=float)).astype(np.int64) - self.lo[j]
        if idx.min() < 0 or idx.max() > self.hi - self.lo[j]:
            raise IndexError(f"location {j + 1}: an inventory level is outside {self.lo[j]}..{self.hi}")
        return idx

    def table(self, j: int, k: int):
        """x -> V^j_k(x), for an array of inventory levels x."""
        C = self.sol[j].C[k - 1]
        return lambda x: C[self._index(j, x)]

    def reorder(self, j: int, k: int, x: int) -> int:
        return int(self.sol[j].Q[k - 1, self._index(j, x)])


class NoTransshipment:
    def __init__(self, single: SingleLocations):
        self.single = single
        self.zero = (0,) * single.J

    def decide(self, k: int, state):
        return self.zero, tuple(self.single.reorder(j, k, x) for j, x in enumerate(state))


class QuasiMyopic:
    """The transshipment that minimises U(delta) + sum_j V^j_k(i^j + delta^j), then the single-location reorders
    at the inventories after the transshipment. The search is ml_milpRL.best_transshipment: it lists the net
    inflows for J <= 3 and solves a small integer program for larger J, and it keeps the first minimum."""

    def __init__(self, single: SingleLocations, R: float, v: float, env=None):
        self.single, self.R, self.v, self.env = single, R, v, env
        self.cache = {}
        self.seconds = []                                  # time of each decision computed

    def decide(self, k: int, state):
        from ml_milpRL import best_transshipment
        state = tuple(int(x) for x in state)
        key = (k,) + state
        hit = self.cache.get(key)
        if hit is None:
            t0 = time.perf_counter()
            tables = [self.single.table(j, k) for j in range(self.single.J)]
            delta, _ = best_transshipment(state, tables, self.R, self.v, "auto", self.env)
            Q = tuple(self.single.reorder(j, k, x + d) for j, (x, d) in enumerate(zip(state, delta)))
            hit = self.cache[key] = (tuple(int(d) for d in delta), Q)
            self.seconds.append(time.perf_counter() - t0)
        return hit


# ----------------------------------------------------------------------------------------------------
# one instance
# ----------------------------------------------------------------------------------------------------
FIELDS = ["id", "block", "factor", "J", "T", "dist", "cv", "relation", "K", "z", "R", "v", "h", "b", "level", "means",
          "paths", "etc_nt_exact", "etc_nt", "etc_qm", "etc_alg", "hw_nt", "hw_qm", "hw_alg",
          "saving_qm_pct", "saving_qm_hw_pct", "saving_alg_pct", "saving_alg_hw_pct",
          "alg_minus_qm_pct", "alg_minus_qm_hw_pct", "first_decision_qm", "first_decision_alg",
          "seconds_single", "decisions_qm", "seconds_per_decision_qm", "seconds_qm",
          "decisions_alg", "seconds_per_decision_alg", "seconds_decision_k1_alg", "seconds_precompute_alg",
          "seconds_milp_alg", "seconds_refine_alg", "seconds_alg", "single_levels_top", "sdp_levels", "sdp_states"]


def run_instance(inst: dict) -> dict:
    from ml_common import half_width, period_pmfs, simulate
    from ml_milp import gurobi_env
    from ml_milpRL import milpRL
    J, costs, n = inst["J"], costs_of(inst), int(inst["paths"])
    means, dist, cv = demand_of(inst)
    row = dict(inst, id=instance_id(inst), means=means_text(means))
    pmfs = period_pmfs(means, dist, cv, TAIL)
    i0 = (0,) * J

    single = SingleLocations(means, dist, cv, costs)                           # 1. the single-location problems
    base = single.cost_without_transshipment()
    nt = simulate(NoTransshipment(single), pmfs, i0, **costs, seed=SEED, n_paths=n)

    env = gurobi_env()                                                         # 2. the quasi-myopic policy
    try:
        policy = QuasiMyopic(single, costs["R"], costs["v"], env)
        t0 = time.perf_counter()
        qm = simulate(policy, pmfs, i0, **costs, seed=SEED, n_paths=n)
        seconds_qm = time.perf_counter() - t0
        first_qm = policy.decide(1, i0)
    finally:
        env.dispose()

    r = milpRL(means, **costs, distribution=dist, cv=cv, tail=TAIL, seed=SEED, n_paths=n,   # 3. the heuristic
               time_limit=MILP_TIME_LIMIT)
    alg = r.sim

    pct = 100.0 / base
    lo, hi = sdp_levels(means, dist, cv)
    row.update(etc_nt_exact=base, etc_nt=nt.mean, etc_qm=qm.mean, etc_alg=alg.mean,
               hw_nt=nt.half_width, hw_qm=qm.half_width, hw_alg=alg.half_width,
               saving_qm_pct=pct * float((nt.costs - qm.costs).mean()),
               saving_qm_hw_pct=pct * half_width(nt.costs - qm.costs),
               saving_alg_pct=pct * float((nt.costs - alg.costs).mean()),
               saving_alg_hw_pct=pct * half_width(nt.costs - alg.costs),
               alg_minus_qm_pct=pct * float((alg.costs - qm.costs).mean()),
               alg_minus_qm_hw_pct=pct * half_width(alg.costs - qm.costs),
               first_decision_qm=str(first_qm), first_decision_alg=str(r.first_decision),
               seconds_single=single.seconds, decisions_qm=len(policy.seconds),
               seconds_per_decision_qm=float(np.mean(policy.seconds)), seconds_qm=seconds_qm,
               decisions_alg=r.n_decisions, seconds_per_decision_alg=r.seconds_per_decision,
               seconds_decision_k1_alg=r.seconds_decision_k1, seconds_precompute_alg=r.seconds_precompute,
               seconds_milp_alg=r.seconds_milp, seconds_refine_alg=r.seconds_refine, seconds_alg=r.seconds_total,
               single_levels_top=single.hi, sdp_levels=hi - lo + 1, sdp_states=str((hi - lo + 1) ** J))
    return row


def _describe(row) -> str:
    return (f"J={row['J']} T={row['T']} {row['dist']} cv={row['cv']} {row['relation']} K={row['K']} R={row['R']} "
            f"b={row['b']} v={row['v']} z={row['z']} level={row['level']}: saving QM {row['saving_qm_pct']:.2f}%, "
            f"ALG {row['saving_alg_pct']:.2f}%, ALG - QM {row['alg_minus_qm_pct']:+.2f}% "
            f"(+-{row['alg_minus_qm_hw_pct']:.2f}); one decision ALG {row['seconds_per_decision_alg']:.3f}s, "
            f"QM {row['seconds_per_decision_qm']:.4f}s; heuristic {row['seconds_alg']:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", default=RUN_BLOCK, choices=("core", "large", "pivot", "time", "all"))
    ap.add_argument("--quick", action="store_true", default=RUN_QUICK)
    ap.add_argument("--workers", type=int, default=N_WORKERS)
    a = ap.parse_args()
    out = Path(str(OUT_DIR) + ("_quick" if a.quick else ""))
    for block in (("core", "pivot", "large", "time") if a.block == "all" else (a.block,)):
        workers = min(a.workers, TIME_WORKERS) if block == "time" else a.workers
        run_instances(block_instances(block, a.quick), run_instance, out / f"{block}.csv", FIELDS, workers, _describe)
    print("results ->", out)


if __name__ == "__main__":
    main()
