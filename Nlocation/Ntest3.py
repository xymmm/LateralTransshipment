"""
Ntest3.py : the main test bed. The heuristic against the quasi-myopic policy and the policy without
transshipment, for two to ten locations.

SDP-2 cannot be solved with more than two locations at these demand levels, so this test bed has no optimum. It
compares three policies on the same demand paths:
    NT    no transshipment: every location follows its optimal single-location policy and nothing is transshipped
    QM    quasi-myopic, after Glazebrook et al. (2015): at the start of every period it chooses the transshipment
          that minimises the transshipment cost plus the sum of the optimal single-location costs of all locations
          from the inventories after the transshipment; each location then reorders by its optimal single-location
          policy
    ALG   the heuristic (Algorithm 1, ml_milpRL.milpRL)

Run this file as a script (not in the Python Console: the parallel processes cannot start there). Results are
appended to results_Ntest3_v2/<block>.csv, one row per instance. A metadata file records the settings, source
hashes and library versions. Resuming requires the same configuration; incompatible files are never appended.
The older results_Ntest3 folders are left unchanged. Ntest3analysis.py makes the tables and figures.

Factors
    number of locations J      2, 3, 4 in every block; 5, 6, 8 and 10 in the block large
    fixed reorder cost K       20, 30, 50
    fixed transshipment cost R 2, 5, 10, 20                              K/R from 1 to 25
    demand distribution        Poisson; normal with coefficient of variation 0.1, 0.2, 0.3
    demand patterns            ten patterns at an average of 7 units per period (PATTERNS below)
    relation between the patterns of the locations: locations 1, 3, 5, ... have the first pattern of a relation
    and locations 2, 4, 6, ... the second
        identical        STAT-STAT, LCY1-LCY1
        opposite         LCY1-LCY2, SIN1-SIN1r, LCY2-LCY2r      (a trailing r reverses the pattern in time)
        unequal levels   STAT-STAT/3                            (/3: one third of every expected demand, at least 1)
        all different    MIX: location j has the j-th pattern of LCY1, LCY2, SIN1, SIN2, STAT, RAND, EMP1, ...
                         (with J = 2 this is LCY1-LCY2, so MIX starts at J = 3)
        the other pairs  the 17 further pairs of the ten patterns, among them EMP1 to EMP4 (block patterns)
    horizon T                  4; 8 in the block horizon
    demand level               average expected demand per period: 7; 5, 10 and 15 in the block level
    penalty cost b             5; 2, 10, 20 in the block pivot
    unit transshipment cost v  0.5; 0, 1, 2 in the block pivot
    unit reorder cost z        1; 0, 2 in the block pivot
    holding cost h = 1, opening inventory 0

Blocks
    core      complete factorial at T = 4 and demand level 7, for J = 2, 3, 4
                  K x R x 4 distributions x 7 relations (6 with J = 2)               288 + 336 + 336 instances
    large     J = 5, 6, 8, 10, where one decision takes seconds: K x R x the five cells LARGE_CELLS  60 per J
    patterns  the 17 other pairs x 4 distributions x the two cost settings COSTS_2, J = 2, 3, 4     136 per J
    pivot     one factor at a time around two base instances (Poisson, LCY1-LCY2, COSTS_2):
                  b in {2, 10, 20}, v in {0, 1, 2}, z in {0, 2}, J = 2, 3, 4                         16 per J
    level     demand level 5, 10, 15 x K x R x LEVEL_RELATIONS, Poisson, J = 2, 3, 4                 72 per J
    horizon   T = 8, Poisson: J = 2 with K x R x 6 relations (72); J = 3 with K x R x LEVEL_RELATIONS (24)
    time      computation times of the base instance on TIME_PATHS demand paths, TIME_WORKERS at the same time:
                  T = 4 for J = 2 to 6, T = 8 for J = 2 and 3

Numbers of locations, horizons, demand paths and the time limit
    One decision of the heuristic solves the model over the remaining periods. Measured on the base instance with
    T = 4, the decision of period 1 takes 0.2 s with 3 locations, 0.8 s with 4, 17 s with 5 and 108 s with 6, and a
    decision of period 2 takes 0.16 s, 0.3 s, 3.3 s and 14 s. With 7 or 8 locations, and with 4 locations at
    T = 8, the model of period 1 was not solved after 30 minutes. With J >= 3 nearly every demand path needs its
    own decisions. PATHS therefore falls with J, the complete factorial stops at 4 locations, and T = 8 stops at
    3 locations.
    Every solve of the model is stopped after MILP_TIME_LIMIT seconds, and the heuristic then uses the best plan
    found, if an incumbent exists; otherwise the instance is recorded as failed. The limit applies to the planning
    MILP, not to the refinement. In a pilot on three instances with 6 locations a
    limit of 30 seconds gave the same costs as no limit. The column decisions_at_limit counts the decisions of an
    instance returned with the solver's TIME_LIMIT status. Solver status counts and mean/maximum final gaps are
    also recorded. The block time has no limit, so that it reports the full times.

What is run for one instance
    1. the single-location problem of every location, by sdp_single.SDP_single    -> NT exactly, and the tables of QM
    2. NT, QM and ALG on the same demand paths (seed SEED)
Measures
    saving of QM and of ALG   NT minus the policy, in per cent of the exact cost of NT
    reduction against QM      QM minus ALG, in per cent of the cost of QM: positive when the heuristic is cheaper;
                              with a paired delta-method confidence interval for the ratio of expected costs
    paired cost difference    QM minus ALG, with a Student-t interval; used to classify significantly better/worse
    sampling                  fixed PATHS/T8_PATHS budgets, not adaptive precision stopping. The relative half-width
                              and whether the 0.1% target is attained are recorded separately for each policy.
Times: a full uncached decision of ALG at k = 1 and on average, including model construction and refinement;
one decision of QM, and the
single-location problems. sdp_states is the number of states that SDP-2 would need per period; SDP-2 is not run.

Demand
    Demand is integer: Poisson, or a normal distribution discretised to the integers. Both are truncated at their
    1 - TAIL quantile and renormalised (ml_common.period_pmfs). A pattern at another demand level is the pattern
    of level 7 multiplied by level / 7 and rounded by the largest-remainder rule (rescale below).

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
RUN_BLOCK = "all"        # "core", "large", "patterns", "pivot", "level", "horizon", "time" or "all"
RUN_QUICK = False        # True: a few instances per block, to check that everything runs
N_WORKERS = 8            # instances solved at the same time
LOCATIONS = (2, 3, 4)                          # numbers of locations J in all blocks but large and time
LARGE_LOCATIONS = (5, 6, 8, 10)                # numbers of locations J in the block large
PATHS = {2: 5000, 3: 2000, 4: 1000, 5: 300, 6: 100, 8: 60, 10: 40}     # demand paths with J locations, T = 4
T8_LOCATIONS = (2, 3)                          # numbers of locations with T = 8, in the blocks horizon and time
T8_PATHS = {2: 5000, 3: 100}                   # demand paths of an instance with T = 8
TIME_LOCATIONS = (2, 3, 4, 5, 6)               # numbers of locations in the block time, T = 4
TIME_PATHS = 20
TIME_WORKERS = 4         # fewer at the same time, so that the times are less disturbed
MILP_TIME_LIMIT = 30     # seconds for one solve of the model in all blocks but time; None: no limit
# ====================================================================================================

import argparse
import csv
import hashlib
import importlib.metadata
import itertools
import json
import math
import multiprocessing as mp
import platform
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for _p in (HERE.parent / "TwoLocation", HERE.parent, HERE):        # milp1, lt_sdp and sdp_single are in TwoLocation
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

OUT_DIR = HERE / "results_Ntest3_v2"
SEED = 1234
TAIL = 1e-4                  # truncation of the demand
CONFIDENCE = 0.95
REL_PRECISION = 1e-3         # reported precision target; fixed sample budgets are retained
SCHEMA_VERSION = 2

# ----------------------------------------------------------------------------------------------------
# demand patterns: expected demand of each period at an average of 7 units, as in the paper
# ----------------------------------------------------------------------------------------------------
PATTERNS = {
    4: {"LCY1": [2, 5, 10, 11], "LCY2": [6, 13, 7, 2], "SIN1": [13, 7, 1, 7], "SIN2": [9, 7, 5, 7],
        "STAT": [7, 7, 7, 7], "RAND": [12, 4, 7, 5], "EMP1": [8, 7, 8, 5], "EMP2": [10, 8, 5, 5],
        "EMP3": [9, 6, 10, 3], "EMP4": [7, 8, 5, 8]},
    8: {"LCY1": [2, 2, 4, 7, 9, 10, 11, 11], "LCY2": [4, 8, 12, 13, 10, 5, 2, 2], "SIN1": [11, 13, 11, 7, 3, 1, 3, 7],
        "SIN2": [8, 9, 8, 7, 6, 5, 6, 7], "STAT": [7, 7, 7, 7, 7, 7, 7, 7], "RAND": [7, 5, 8, 10, 11, 6, 3, 6],
        "EMP1": [5, 11, 11, 7, 4, 4, 8, 6], "EMP2": [7, 8, 5, 9, 5, 9, 6, 7], "EMP3": [3, 7, 8, 6, 8, 8, 8, 8],
        "EMP4": [5, 5, 5, 4, 5, 10, 12, 10]},
}
PATTERN_NAMES = ("LCY1", "LCY2", "SIN1", "SIN2", "STAT", "RAND", "EMP1", "EMP2", "EMP3", "EMP4")
LEVEL = 7                    # average expected demand per period of the patterns above


def rescale(values, level: int):
    """The pattern `values` of level LEVEL at another level: every expected demand is multiplied by level / LEVEL
    and rounded by the largest-remainder rule, so that the average is exactly `level`. Equal remainders go to the
    earlier period, and every period keeps an expected demand of at least 1."""
    exact = [x * level / LEVEL for x in values]
    out = [int(math.floor(x)) for x in exact]
    by_remainder = sorted(range(len(values)), key=lambda t: (-(exact[t] - out[t]), t))
    for t in by_remainder[:level * len(values) - sum(out)]:
        out[t] += 1
    for t in range(len(out)):                                # a period at 0 takes one unit from the largest period
        if out[t] == 0:
            out[max(range(len(out)), key=lambda u: out[u])] -= 1
            out[t] = 1
    return out


def pattern(name: str, T: int, level: int = LEVEL):
    """Expected demands of one location. STAT, LCY1, ...: a pattern of PATTERNS, at the given demand level.
    A trailing r lists it in reverse order. A trailing /3 gives one third of every expected demand, at least 1."""
    if name.endswith("/3"):
        return [max(1, int(math.floor(x / 3 + 0.5))) for x in pattern(name[:-2], T, level)]
    if name.endswith("r") and name[:-1] in PATTERNS[T]:
        return pattern(name[:-1], T, level)[::-1]
    values = PATTERNS[T][name]
    return list(values) if level == LEVEL else rescale(values, level)


def relation_means(relation: str, T: int, J: int = 2, level: int = LEVEL):
    """Expected demands of J locations. "P-Q": locations 1, 3, 5, ... have the pattern P and locations 2, 4, 6, ...
    the pattern Q. "MIX": location j has the j-th pattern of PATTERN_NAMES."""
    if relation == "MIX":
        return [pattern(PATTERN_NAMES[j % len(PATTERN_NAMES)], T, level) for j in range(J)]
    first, second = relation.split("-")
    return [pattern(first if j % 2 == 0 else second, T, level) for j in range(J)]


# ----------------------------------------------------------------------------------------------------
# the instances
# ----------------------------------------------------------------------------------------------------
K_LEVELS = (20, 30, 50)
R_LEVELS = (2, 5, 10, 20)
DISTRIBUTIONS = (("poisson", None), ("normal", 0.1), ("normal", 0.2), ("normal", 0.3))
RELATIONS = ("STAT-STAT", "LCY1-LCY1", "LCY1-LCY2", "SIN1-SIN1r", "LCY2-LCY2r", "STAT-STAT/3")
BASE = dict(T=4, dist="poisson", cv=None, relation="LCY1-LCY2", K=30, z=1, R=5, v=0.5, h=1, b=5, level=LEVEL)
COSTS_2 = ((30, 5), (50, 5))                   # (K, R) of the blocks patterns and pivot: K/R = 6 and 10
PIVOT = dict(b=(2, 10, 20), v=(0, 1, 2), z=(0, 2))
LEVELS = (5, 10, 15)                           # demand levels of the block level, next to 7
LEVEL_RELATIONS = ("LCY1-LCY2", "STAT-STAT/3")
LARGE_CELLS = (("poisson", None, "STAT-STAT"), ("poisson", None, "LCY1-LCY2"), ("poisson", None, "STAT-STAT/3"),
               ("poisson", None, "MIX"), ("normal", 0.3, "LCY1-LCY2"))
PAIRS = ([f"{p}-{p}" for p in PATTERN_NAMES]                 # the 20 pairs of the earlier test bed
         + [f"{p}-{PATTERN_NAMES[(n + 1) % len(PATTERN_NAMES)]}" for n, p in enumerate(PATTERN_NAMES)])
PATTERN_PAIRS = tuple(p for p in PAIRS if p not in RELATIONS)      # 17: those that the block core does not have
ID_KEYS = ("block", "J", "T", "dist", "cv", "relation", "K", "z", "R", "v", "h", "b", "level")


def relations_of(J: int):
    """The relations of the block core: MIX from three locations on."""
    return RELATIONS + (("MIX",) if J >= 3 else ())


def factorial(block: str, J: int, T: int = 4, distributions=DISTRIBUTIONS, relations=RELATIONS, costs=None,
              level: int = LEVEL, paths=None):
    """Instances of a block: distributions x relations x (K, R)."""
    costs = tuple(itertools.product(K_LEVELS, R_LEVELS)) if costs is None else costs
    n = (T8_PATHS if T == 8 else PATHS)[J] if paths is None else paths
    return [dict(BASE, block=block, factor="", J=J, T=T, dist=d, cv=c, relation=rel, K=K, R=R, level=level, paths=n,
                 limit=MILP_TIME_LIMIT)
            for (d, c), rel, (K, R) in itertools.product(distributions, relations, costs)]


def block_instances(block: str, quick: bool = False):
    rows = []
    poisson = (("poisson", None),)
    if block == "core":
        for J in LOCATIONS:
            rows += factorial("core", J, relations=relations_of(J))
        if quick:
            rows = [dict(r, paths=40) for r in rows if (r["K"], r["R"]) == (30, 5) and r["dist"] == "poisson"
                    and r["relation"] in ("LCY1-LCY2", "STAT-STAT/3", "MIX")]
    elif block == "large":
        for J in LARGE_LOCATIONS:
            for d, c, rel in LARGE_CELLS:
                rows += factorial("large", J, distributions=((d, c),), relations=(rel,))
        if quick:
            rows = [dict(r, paths=4) for r in rows if r["J"] == 5 and (r["K"], r["R"]) == (50, 5)
                    and r["relation"] == "MIX"]
    elif block == "patterns":
        for J in LOCATIONS:
            rows += factorial("patterns", J, relations=PATTERN_PAIRS, costs=COSTS_2)
        if quick:
            rows = [dict(r, paths=40) for r in rows if r["J"] == 3 and r["dist"] == "poisson" and r["K"] == 50
                    and r["relation"] in ("EMP1-EMP2", "EMP4-EMP4")]
    elif block == "pivot":
        for J in LOCATIONS:
            for K, R in COSTS_2:
                rows += [dict(BASE, block="pivot", factor=name, J=J, K=K, R=R, paths=PATHS[J], limit=MILP_TIME_LIMIT,
                              **{name: x}) for name, levels in PIVOT.items() for x in levels]
        if quick:
            rows = [dict(r, paths=40) for r in rows if r["J"] == 3 and r["K"] == 50 and r["factor"] in ("b", "v")]
    elif block == "level":
        for J in LOCATIONS:
            for level in LEVELS:
                rows += [dict(r, factor="level") for r in
                         factorial("level", J, distributions=poisson, relations=LEVEL_RELATIONS, level=level)]
        if quick:
            rows = [dict(r, paths=40) for r in rows if r["J"] == 3 and (r["K"], r["R"]) == (50, 5)
                    and r["relation"] == "LCY1-LCY2"]
    elif block == "horizon":
        for J in T8_LOCATIONS:
            rels = RELATIONS if J == 2 else LEVEL_RELATIONS
            rows += [dict(r, factor="T") for r in factorial("horizon", J, T=8, distributions=poisson, relations=rels)]
        if quick:
            rows = [dict(r, paths=60) for r in rows if r["J"] == 2 and (r["K"], r["R"]) == (50, 5)
                    and r["relation"] == "LCY1-LCY2"]
    elif block == "time":
        rows = [dict(BASE, block="time", factor="", J=J, T=4, paths=TIME_PATHS, limit=None) for J in TIME_LOCATIONS]
        rows += [dict(BASE, block="time", factor="", J=J, T=8, paths=TIME_PATHS, limit=None) for J in T8_LOCATIONS]
        if quick:
            rows = [dict(r, paths=10) for r in rows if r["J"] <= 4 and r["T"] == 4]
    rows.sort(key=lambda r: (-r["T"], -r["J"]))               # the long instances first
    return rows


BLOCKS = ("core", "patterns", "pivot", "level", "horizon", "large", "time")


def instance_id(inst: dict) -> str:
    return "|".join(str(inst[k]) for k in ID_KEYS)


def costs_of(inst: dict) -> dict:
    return {k: inst[k] for k in ("K", "z", "R", "v", "h", "b")}


def demand_of(inst: dict):
    """(expected demands of every location, distribution, coefficient of variation)."""
    return relation_means(inst["relation"], inst["T"], inst["J"], inst["level"]), inst["dist"], inst["cv"]


def means_text(means) -> str:
    return " ; ".join(" ".join(str(x) for x in m) for m in means)


def largest_demands(pmfs):
    """For every location the largest demand that it can have over the horizon: the sum over the periods of the
    largest value of its truncated demand."""
    return [sum(len(p) - 1 for p in pm) for pm in pmfs]


# ----------------------------------------------------------------------------------------------------
# the comparison policies: decide(k, state) -> (net inflow of every location, reorder quantities)
# ----------------------------------------------------------------------------------------------------
class SingleLocations:
    """The single-location problem of every location: optimal costs V^j_k(x) and optimal reorders."""

    def __init__(self, means, dist, cv, costs):
        from ml_common import period_pmfs
        from TwoLocation.sdp_single import SDP_single
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
    inflows for J <= 3 and solves a small integer program for larger J. This network adaptation permits
    multiple simultaneous shipments; enumeration keeps the first minimum, while MILP ties are solver-dependent."""

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
          "paths", "limit", "decisions_at_limit", "etc_nt_exact", "etc_nt", "etc_qm", "etc_alg", "hw_nt", "hw_qm", "hw_alg",
          "saving_qm_pct", "saving_qm_hw_pct", "saving_alg_pct", "saving_alg_hw_pct",
          "reduction_qm_pct", "reduction_qm_hw_pct", "first_decision_qm", "first_decision_alg",
          "seconds_single", "decisions_qm", "seconds_per_decision_qm", "seconds_qm",
          "decisions_alg", "seconds_per_decision_alg", "seconds_decision_k1_alg", "seconds_precompute_alg",
          "seconds_milp_alg", "seconds_refine_alg", "seconds_alg", "single_levels_top", "sdp_levels", "sdp_states"]
FIELDS += ["sampling_mode", "confidence", "rel_precision", "relative_hw_nt", "relative_hw_qm", "relative_hw_alg",
           "precision_attained_nt", "precision_attained_qm", "precision_attained_alg",
           "difference_qm_alg", "difference_qm_alg_hw", "seconds_build_alg", "solver_status_counts",
           "mip_gap_mean", "mip_gap_max", "mip_gap_at_limit_max", "mip_gap_unavailable"]


def paired_reduction(qm_costs, alg_costs, confidence=CONFIDENCE):
    """Percentage reduction 100 E[QM-ALG]/E[QM], with a paired delta-method half-width.

    For the ratio r, the estimated influence values are (QM-ALG)-r*QM. Using only QM-ALG would incorrectly
    treat the estimated denominator as fixed. The raw paired-difference interval is kept separately for tests.
    """
    from ml_common import half_width
    qm = np.asarray(qm_costs, dtype=float)
    alg = np.asarray(alg_costs, dtype=float)
    if qm.ndim != 1 or qm.shape != alg.shape or len(qm) < 2:
        raise ValueError("paired costs must be equal-length vectors with at least two observations")
    mean_qm = float(qm.mean())
    if not np.isfinite(qm).all() or not np.isfinite(alg).all() or mean_qm <= 0:
        raise ValueError("costs must be finite and the mean QM cost must be positive")
    difference = qm - alg
    ratio = float(difference.mean()) / mean_qm
    hw = 100.0 * half_width(difference - ratio * qm, confidence) / mean_qm
    return 100.0 * ratio, hw, float(difference.mean()), half_width(difference, confidence)


def solver_summary(reports):
    """Count actual solve outcomes, never infer a timeout from elapsed decision time."""
    from gurobipy import GRB
    counts = Counter(int(r["status"]) for r in reports)
    gaps = [float(r["mip_gap"]) for r in reports if np.isfinite(r["mip_gap"])]
    limited = [float(r["mip_gap"]) for r in reports
               if r["status"] == GRB.TIME_LIMIT and np.isfinite(r["mip_gap"])]
    return dict(decisions_at_limit=counts[GRB.TIME_LIMIT], solver_status_counts=json.dumps(dict(counts), sort_keys=True),
                mip_gap_mean=float(np.mean(gaps)) if gaps else float("nan"),
                mip_gap_max=max(gaps) if gaps else float("nan"),
                mip_gap_at_limit_max=max(limited) if limited else float("nan"),
                mip_gap_unavailable=len(reports) - len(gaps))


def run_instance(inst: dict) -> dict:
    from ml_common import half_width, period_pmfs, simulate, total_demand_quantile
    from ml_milp import gurobi_env
    from ml_milpRL import milpRL
    J, costs, n = inst["J"], costs_of(inst), int(inst["paths"])
    means, dist, cv = demand_of(inst)
    row = dict(inst, id=instance_id(inst), means=means_text(means))
    pmfs = period_pmfs(means, dist, cv, TAIL)
    i0 = (0,) * J

    single = SingleLocations(means, dist, cv, costs)                           # 1. the single-location problems
    base = single.cost_without_transshipment()
    nt = simulate(NoTransshipment(single), pmfs, i0, **costs, seed=SEED, n_paths=n,
                  confidence=CONFIDENCE, rel_precision=REL_PRECISION)

    env = gurobi_env()                                                         # 2. the quasi-myopic policy
    try:
        policy = QuasiMyopic(single, costs["R"], costs["v"], env)
        t0 = time.perf_counter()
        qm = simulate(policy, pmfs, i0, **costs, seed=SEED, n_paths=n,
                      confidence=CONFIDENCE, rel_precision=REL_PRECISION)
        seconds_qm = time.perf_counter() - t0
        first_qm = policy.decide(1, i0)
    finally:
        env.dispose()

    limit = inst.get("limit")
    r = milpRL(means, **costs, distribution=dist, cv=cv, tail=TAIL, seed=SEED, n_paths=n,   # 3. the heuristic
               time_limit=limit, confidence=CONFIDENCE, rel_precision=REL_PRECISION)
    alg = r.sim
    row.update(solver_summary(r.policy.decision_reports))
    reduction, reduction_hw, difference, difference_hw = paired_reduction(qm.costs, alg.costs)
    row.update(sampling_mode="fixed", confidence=CONFIDENCE, rel_precision=REL_PRECISION,
               difference_qm_alg=difference, difference_qm_alg_hw=difference_hw, seconds_build_alg=r.seconds_build)
    for name, sim in (("nt", nt), ("qm", qm), ("alg", alg)):
        row[f"relative_hw_{name}"] = sim.half_width / abs(sim.mean) if sim.mean else float("nan")
        row[f"precision_attained_{name}"] = bool(sim.attained)

    pct = 100.0 / base                                       # per cent of the exact cost without transshipment
    levels = int(total_demand_quantile(means, dist, cv, TAIL)) + max(single.M) + 1     # levels SDP-2 would need
    row.update(etc_nt_exact=base, etc_nt=nt.mean, etc_qm=qm.mean, etc_alg=alg.mean,
               hw_nt=nt.half_width, hw_qm=qm.half_width, hw_alg=alg.half_width,
               saving_qm_pct=pct * float((nt.costs - qm.costs).mean()),
               saving_qm_hw_pct=pct * half_width(nt.costs - qm.costs, CONFIDENCE),
               saving_alg_pct=pct * float((nt.costs - alg.costs).mean()),
               saving_alg_hw_pct=pct * half_width(nt.costs - alg.costs, CONFIDENCE),
               reduction_qm_pct=reduction, reduction_qm_hw_pct=reduction_hw,
               first_decision_qm=str(first_qm), first_decision_alg=str(r.first_decision),
               seconds_single=single.seconds, decisions_qm=len(policy.seconds),
               seconds_per_decision_qm=float(np.mean(policy.seconds)), seconds_qm=seconds_qm,
               decisions_alg=r.n_decisions, seconds_per_decision_alg=r.seconds_per_decision,
               seconds_decision_k1_alg=r.seconds_decision_k1, seconds_precompute_alg=r.seconds_precompute,
               seconds_milp_alg=r.seconds_milp, seconds_refine_alg=r.seconds_refine, seconds_alg=r.seconds_total,
               single_levels_top=single.hi, sdp_levels=levels, sdp_states=str(levels ** J))
    return row


# ----------------------------------------------------------------------------------------------------
# the driver
# ----------------------------------------------------------------------------------------------------
def experiment_metadata(instances, workers):
    """Everything that affects the simulation, solver outcomes, sampling or reported times."""
    sources = [HERE / name for name in ("Ntest3.py", "ml_milp.py", "ml_milpRL.py", "ml_common.py")]
    sources += [HERE.parent / "TwoLocation" / name for name in ("sdp_single.py", "lt_sdp.py", "milp1.py")]
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "gurobipy")}
    shared = dict(schema_version=SCHEMA_VERSION, seed=SEED, tail=TAIL, confidence=CONFIDENCE,
                  rel_precision=REL_PRECISION, sampling_mode="fixed", python=platform.python_version(),
                  platform=platform.platform(), versions=versions,
                  thread_settings={key: os.environ.get(key) for key in
                                   ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")},
                  source_hashes={str(p.relative_to(HERE.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sources})
    return dict(shared=shared, workers=workers, fields=FIELDS,
                instances=sorted(instances, key=instance_id))


def prepare_results(instances, path: Path, workers: int):
    """Validate the run manifest and CSV before any results are appended; return completed IDs."""
    metadata_path = path.with_suffix(".meta.json")
    expected = experiment_metadata(instances, workers)
    if metadata_path.exists():
        if json.loads(metadata_path.read_text()) != expected:
            raise ValueError(f"Configuration changed for {path}. Use a new --output-dir; existing results are preserved.")
    elif path.exists():
        raise ValueError(f"{path} has no experiment metadata. Use a new --output-dir; legacy results are preserved.")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = metadata_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n")
        temporary.replace(metadata_path)
    done = set()
    if path.exists() and path.stat().st_size:
        allowed = {instance_id(i) for i in instances}
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != FIELDS:
                raise ValueError(f"Incompatible CSV columns in {path}; use a new --output-dir.")
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(f"Incomplete or malformed row in {path}; inspect the file before resuming.")
                key = row["id"]
                if key not in allowed or key in done:
                    raise ValueError(f"Unexpected or duplicate instance ID in {path}: {key}")
                done.add(key)
    return done


def _work(inst):
    try:
        return run_instance(inst), None
    except Exception as e:                                   # the run goes on; the instance is reported
        return None, f"{instance_id(inst)}: {type(e).__name__}: {e}"


def run_instances(instances, path: Path, workers: int):
    """Runs every instance that is not yet in the file `path`, `workers` at a time, and appends one row each."""
    done = prepare_results(instances, path, workers)
    todo = [i for i in instances if instance_id(i) not in done]
    print(f"{path.name}: {len(instances)} instances, {len(done)} already in the file, {len(todo)} to run, "
          f"{workers} at the same time", flush=True)
    if not todo:
        return
    t0 = time.perf_counter()
    new = not path.exists() or path.stat().st_size == 0
    failed = []
    with open(path, "a", newline="") as f, mp.get_context("spawn").Pool(workers) as pool:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for n, (row, err) in enumerate(pool.imap_unordered(_work, todo), 1):
            elapsed = (time.perf_counter() - t0) / 60
            if err is not None:
                failed.append(err)
                print(f"[{n}/{len(todo)}] FAILED {err}", flush=True)
                continue
            w.writerow({k: row.get(k, "") for k in FIELDS})
            f.flush()
            print(f"[{n}/{len(todo)}] {_describe(row)}   {elapsed:.1f} min", flush=True)
    if failed:
        (path.parent / (path.stem + "_failed.txt")).write_text("\n".join(failed) + "\n")
        print(f"{len(failed)} instances failed; see {path.stem}_failed.txt. Run again to repeat them.", flush=True)
    elif (path.parent / (path.stem + "_failed.txt")).exists():
        (path.parent / (path.stem + "_failed.txt")).unlink()


def _describe(row) -> str:
    return (f"J={row['J']} T={row['T']} {row['dist']} cv={row['cv']} {row['relation']} K={row['K']} R={row['R']} "
            f"b={row['b']} v={row['v']} z={row['z']} level={row['level']}: saving QM {row['saving_qm_pct']:.2f}%, "
            f"ALG {row['saving_alg_pct']:.2f}%; reduction against QM {row['reduction_qm_pct']:+.2f}% "
            f"(+-{row['reduction_qm_hw_pct']:.2f}); one decision ALG {row['seconds_per_decision_alg']:.3f}s, "
            f"QM {row['seconds_per_decision_qm']:.4f}s; heuristic {row['seconds_alg']:.0f}s; "
            f"ALG relative half-width {100 * row['relative_hw_alg']:.2f}%, "
            f"target attained: {row['precision_attained_alg']}; time-limit solves {row['decisions_at_limit']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", default=RUN_BLOCK, choices=BLOCKS + ("all",))
    ap.add_argument("--quick", action="store_true", default=RUN_QUICK)
    ap.add_argument("--workers", type=int, default=N_WORKERS)
    ap.add_argument("--output-dir", type=Path, help="results folder; defaults to results_Ntest3_v2 (or its _quick folder)")
    a = ap.parse_args()
    out = a.output_dir if a.output_dir is not None else Path(str(OUT_DIR) + ("_quick" if a.quick else ""))
    for block in (BLOCKS if a.block == "all" else (a.block,)):
        workers = min(a.workers, TIME_WORKERS) if block == "time" else a.workers
        run_instances(block_instances(block, a.quick), out / f"{block}.csv", workers)
    print("results ->", out)


if __name__ == "__main__":
    main()
