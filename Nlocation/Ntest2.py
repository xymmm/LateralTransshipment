"""
Ntest2.py : test bed A. The heuristic against the optimum of SDP-2.

Question: how far is the expected total cost of the heuristic (Algorithm 1, ml_milpRL.milpRL) from the optimal
expected total cost (ml_SDP2.SDP2)? Every instance is defined for J = 2, 3 and 4 locations. An instance is deleted
when SDP-2 cannot be solved within one hour (see "The one-hour rule" below); the deleted instances are listed in
results_Ntest2/<block>_deleted.csv. Ntest3.py takes the same instances without SDP-2.

Run this file. Results are appended to results_Ntest2/<block>.csv, one row per instance. An instance that is
already in the file is skipped, so a run can be stopped and started again. Ntest2analysis.py makes the tables.

Blocks, for every J in LOCATIONS
    core      complete factorial, 576 instances per J
                  K in {20, 30, 50}  x  R in {2, 5, 10, 20}
                  demand distribution: Poisson, normal with coefficient of variation 0.1, 0.2, 0.3
                  horizon T in {4, 8}
                  relation between the demand patterns of the locations (6)
                      identical        STAT-STAT, LCY1-LCY1
                      opposite         LCY1-LCY2, SIN1-SIN1r, LCY2-LCY2r   (a trailing r reverses the pattern in time)
                      unequal levels   STAT-STAT/3                         (/3: one third of the level, at least 1)
                  z = 1, v = 0.5, h = 1, b = 5, demand level 7
    patterns  the other pairs of the ten demand patterns, among them EMP1 to EMP4, 884 instances per J
                  the ten identical pairs and the ten pairs of neighbours of the earlier test bed, without the
                  three that are relations of the block core: 17 pairs
                  T = 4: every pair with all K, R and distributions; T = 8: every pair with K = 30, R = 5
    pivot     one factor at a time around the base instance, 10 instances per J
                  base: T = 4, Poisson, LCY1-LCY2, K = 30, R = 5, z = 1, v = 0.5, h = 1, b = 5, demand level 7
                  b in {2, 10, 20};  v in {0, 1, 2};  z in {0, 2};  demand level in {5, 10, 15}

Demand
    The expected demands are the fixed integer patterns of the paper at an average of 7 units per period
    (PATTERNS below). Locations 1, 3, ... have the first pattern of a relation and locations 2, 4, ... the second.
    The demand level is the average expected demand per period of a pattern. It is 7 in the blocks core and
    patterns, and 5, 10 and 15 in the block pivot; pattern() rescales the pattern of level 7 to another level.
    Demand is integer: Poisson, or a normal distribution discretised to the integers. Both are truncated at their
    1 - TAIL quantile and renormalised (ml_common.period_pmfs).

What is run for one instance
    1. SDP-2 on the inventory levels below                               -> optimal expected total cost C
    2. the heuristic, simulated until the half-width of the 95% confidence interval is at most 0.1% of the
       estimate (or MAX_PATHS paths)                                     -> gap = (ETC - C) / C
    3. the optimal policy of SDP-2 on the same demand paths              -> paired gap, with its confidence interval
    4. the optimal single-location policies, without transshipment       -> exact cost; its gap is the value of
                                                                            transshipment in the instance

Inventory levels of SDP-2: the integers lo..hi at every location
    hi   the 1 - TAIL quantile of the total demand of all locations over the horizon: all expected demands plus
         a buffer, so one location can hold the stock of the whole system
    lo   minus the largest demand that one location can have over the horizon; no state below it can be reached
         from the opening inventory, so no demand outcome is dropped on a path from the opening inventory
    Check: every instance with T = 4, and every instance with T = 8, K = 50 and R = 2, is solved again with both
    ends moved out by 25%. The file reports the relative change of the optimal cost (change_widened).

The one-hour rule
    SDP-2 lists every shipment vector of every state in every period. In Ntest1.py it listed 6.2e9 vectors in
    223 seconds, which is 2.8e7 vectors per second, so one hour allows about 1e11 vectors (VECTOR_LIMIT). An
    instance with more shipment vectors over its horizon is deleted without a run. An instance that is run and
    needs more than one hour for SDP-2 is marked in the column over_hour and left out by Ntest2analysis.py.

Settings for clicking "Run" in PyCharm are below; command-line arguments override them:
    python Ntest2.py --block core --workers 8
    python Ntest2.py --quick                    a few instances, to check that everything runs
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")                         # one processor core per instance

# ====================================================================================================
# settings for clicking "Run" in PyCharm; command-line arguments override them
RUN_BLOCK = "all"        # "core", "patterns", "pivot" or "all"
RUN_QUICK = False        # True: a few instances per block, to check that everything runs
N_WORKERS = 8            # instances solved at the same time
LOCATIONS = (2, 3, 4)    # numbers of locations J; an instance is deleted when SDP-2 needs more than one hour
# ====================================================================================================

import argparse
import csv
import itertools
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _p in (HERE.parent / "TwoLocation", HERE.parent, HERE):        # milp1, lt_sdp and sdp_single are in TwoLocation
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

OUT_DIR = HERE / "results_Ntest2"
SEED = 1234
TAIL = 1e-4                  # truncation of the demand and buffer of the inventory levels
REL_PRECISION = 1e-3         # the simulation of the heuristic stops at a half-width of 0.1% of the estimate
MAX_PATHS = 200_000          # or at this many demand paths
WIDEN = 0.25                 # the check of the inventory levels moves both ends out by this share
SDP_SECONDS = 3600           # an instance is deleted when SDP-2 needs more than this
VECTORS_PER_SECOND = 2.8e7   # shipment vectors that SDP-2 lists per second, measured in Ntest1.py
VECTOR_LIMIT = SDP_SECONDS * VECTORS_PER_SECOND

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
    """Expected demands of J locations for a relation "P-Q": locations 1, 3, 5, ... have the pattern P and
    locations 2, 4, 6, ... the pattern Q. With J = 2 this is the pair of the paper."""
    first, second = relation.split("-")
    return [pattern(first if j % 2 == 0 else second, T, level) for j in range(J)]


# ----------------------------------------------------------------------------------------------------
# the instances
# ----------------------------------------------------------------------------------------------------
K_LEVELS = (20, 30, 50)
R_LEVELS = (2, 5, 10, 20)
DISTRIBUTIONS = (("poisson", None), ("normal", 0.1), ("normal", 0.2), ("normal", 0.3))
HORIZONS = (4, 8)
RELATIONS = ("STAT-STAT", "LCY1-LCY1", "LCY1-LCY2", "SIN1-SIN1r", "LCY2-LCY2r", "STAT-STAT/3")
BASE = dict(T=4, dist="poisson", cv=None, relation="LCY1-LCY2", K=30, z=1, R=5, v=0.5, h=1, b=5, level=LEVEL)
PIVOT = dict(b=(2, 10, 20), v=(0, 1, 2), z=(0, 2), level=(5, 10, 15))
PAIRS = ([f"{p}-{p}" for p in PATTERN_NAMES]                 # the 20 pairs of the earlier test bed
         + [f"{p}-{PATTERN_NAMES[(n + 1) % len(PATTERN_NAMES)]}" for n, p in enumerate(PATTERN_NAMES)])
PATTERN_PAIRS = tuple(p for p in PAIRS if p not in RELATIONS)      # 17: those that the block core does not have
ID_KEYS = ("block", "J", "T", "dist", "cv", "relation", "K", "z", "R", "v", "h", "b", "level")


def core_instances(J: int = 2, horizons=HORIZONS, distributions=DISTRIBUTIONS, relations=RELATIONS,
                   k_levels=K_LEVELS, r_levels=R_LEVELS, block: str = "core"):
    """The complete factorial."""
    return [dict(BASE, block=block, factor="", J=J, T=T, dist=d, cv=c, relation=rel, K=K, R=R)
            for T, (d, c), rel, K, R in itertools.product(horizons, distributions, relations, k_levels, r_levels)]


def pattern_instances(J: int = 2):
    """The other pairs of patterns: all K, R and distributions at T = 4, the base K and R at T = 8."""
    rows = core_instances(J, horizons=(4,), relations=PATTERN_PAIRS, block="patterns")
    rows += core_instances(J, horizons=(8,), relations=PATTERN_PAIRS, k_levels=(BASE["K"],), r_levels=(BASE["R"],),
                           block="patterns")
    return rows


def pivot_instances(J: int = 2, pivot=PIVOT):
    """One factor at a time around the base instance."""
    return [dict(BASE, block="pivot", factor=name, J=J, **{name: x}) for name, levels in pivot.items() for x in levels]


def instance_id(inst: dict) -> str:
    return "|".join(str(inst[k]) for k in ID_KEYS)


def costs_of(inst: dict) -> dict:
    return {k: inst[k] for k in ("K", "z", "R", "v", "h", "b")}


def demand_of(inst: dict):
    """(expected demands of every location, distribution, coefficient of variation)."""
    return relation_means(inst["relation"], inst["T"], inst["J"], inst["level"]), inst["dist"], inst["cv"]


def means_text(means) -> str:
    return " ; ".join(" ".join(str(x) for x in m) for m in means)


# ----------------------------------------------------------------------------------------------------
# inventory levels, the one-hour rule, and the cost without transshipment
# ----------------------------------------------------------------------------------------------------
def largest_demands(pmfs):
    """For every location the largest demand that it can have over the horizon: the sum over the periods of the
    largest value of its truncated demand."""
    return [sum(len(p) - 1 for p in pm) for pm in pmfs]


def sdp_levels(means, dist, cv):
    """(lo, hi) of every location for SDP-2; see the top of the file."""
    from ml_common import period_pmfs, total_demand_quantile
    lo = -max(largest_demands(period_pmfs(means, dist, cv, TAIL)))
    hi = int(total_demand_quantile(means, dist, cv, TAIL))
    return lo, hi


def shipment_vectors(J: int, T: int, lo: int, hi: int) -> int:
    """Number of shipment vectors that SDP-2 lists over the horizon, exact: a location with s units on hand and
    J - 1 destinations ships in C(s + J - 1, J - 1) ways, at every state of every period."""
    return T * sum(math.comb(max(x, 0) + J - 1, J - 1) for x in range(lo, hi + 1)) ** J


def size_of(inst: dict) -> dict:
    """Inventory levels, states per period and shipment vectors over the horizon of an instance."""
    means, dist, cv = demand_of(inst)
    lo, hi = sdp_levels(means, dist, cv)
    return dict(lo=lo, hi=hi, levels=hi - lo + 1, states=(hi - lo + 1) ** inst["J"],
                vectors=shipment_vectors(inst["J"], inst["T"], lo, hi))


def no_transshipment_cost(means, dist, cv, costs) -> float:
    """Exact expected total cost when every location follows its optimal single-location policy and nothing is
    transshipped: the sum of the single-location optimal costs from the opening inventory 0."""
    from ml_common import period_pmfs
    from TwoLocation.sdp_single import SDP_single
    total = 0.0
    for mu, M in zip(means, largest_demands(period_pmfs(means, dist, cv, TAIL))):
        total += SDP_single(mu, -M, M, costs["K"], costs["z"], costs["h"], costs["b"], distribution=dist, cv=cv,
                            tail=TAIL).value(0)
    return total


# ----------------------------------------------------------------------------------------------------
# one instance
# ----------------------------------------------------------------------------------------------------
FIELDS = ["id", "block", "factor", "J", "T", "dist", "cv", "relation", "K", "z", "R", "v", "h", "b", "level", "means",
          "lo", "hi", "levels", "states", "vectors", "seconds_sdp2", "over_hour", "etc_opt", "first_decision_opt",
          "etc_alg", "hw_alg", "paths", "attained", "gap_pct", "gap_hw_pct", "first_decision_alg",
          "etc_opt_same_paths", "paired_gap_pct", "paired_gap_hw_pct",
          "etc_nt", "gap_nt_pct",
          "seconds_alg", "seconds_precompute", "decisions", "seconds_per_decision", "seconds_decision_k1",
          "lo_widened", "hi_widened", "etc_opt_widened", "change_widened"]
DELETED_FIELDS = ["id", "block", "factor", "J", "T", "dist", "cv", "relation", "K", "z", "R", "v", "h", "b", "level",
                  "means", "lo", "hi", "levels", "states", "vectors", "hours_estimated"]


def check_levels(inst: dict) -> bool:
    """The instances on which SDP-2 is solved again on wider inventory levels."""
    return inst["T"] == 4 or (inst["K"] == max(K_LEVELS) and inst["R"] == min(R_LEVELS))


def run_instance(inst: dict, max_paths: int = MAX_PATHS) -> dict:
    from ml_common import OptimalPolicy, half_width, period_pmfs, simulate
    from ml_milpRL import milpRL
    from ml_SDP2 import SDP2
    J, costs = inst["J"], costs_of(inst)
    means, dist, cv = demand_of(inst)
    row = dict(inst, id=instance_id(inst), means=means_text(means), **size_of(inst))
    lo, hi = row["lo"], row["hi"]

    s = SDP2(means, lo, hi, distribution=dist, cv=cv, tail=TAIL, **costs)      # 1. the optimum
    C = s.value()
    row.update(seconds_sdp2=s.seconds, over_hour=int(s.seconds > SDP_SECONDS), etc_opt=C,
               first_decision_opt=str(s.action()))

    r = milpRL(means, **costs, distribution=dist, cv=cv, tail=TAIL, seed=SEED,  # 2. the heuristic
               rel_precision=REL_PRECISION, max_paths=max_paths)
    row.update(etc_alg=r.etc, hw_alg=r.half_width, paths=r.n_paths, attained=r.precision_attained,
               gap_pct=100 * (r.etc - C) / C, gap_hw_pct=100 * r.half_width / C,
               first_decision_alg=str(r.first_decision), seconds_alg=r.seconds_total,
               seconds_precompute=r.seconds_precompute, decisions=r.n_decisions,
               seconds_per_decision=r.seconds_per_decision, seconds_decision_k1=r.seconds_decision_k1)

    pmfs = period_pmfs(means, dist, cv, TAIL)                                  # 3. the optimal policy, same paths
    opt = simulate(OptimalPolicy(s), pmfs, (0,) * J, **costs, seed=SEED, n_paths=r.n_paths)
    diff = r.sim.costs - opt.costs
    row.update(etc_opt_same_paths=opt.mean, paired_gap_pct=100 * float(diff.mean()) / C,
               paired_gap_hw_pct=100 * half_width(diff) / C)
    del s

    nt = no_transshipment_cost(means, dist, cv, costs)                         # 4. without transshipment
    row.update(etc_nt=nt, gap_nt_pct=100 * (nt - C) / C)

    if check_levels(inst):                                                     # wider inventory levels
        lo_w, hi_w = math.floor(lo * (1 + WIDEN)), math.ceil(hi * (1 + WIDEN))
        wide = SDP2(means, lo_w, hi_w, distribution=dist, cv=cv, tail=TAIL, **costs).value()
        row.update(lo_widened=lo_w, hi_widened=hi_w, etc_opt_widened=wide, change_widened=abs(wide - C) / C)
    return row


# ----------------------------------------------------------------------------------------------------
# the driver, also used by Ntest3.py
# ----------------------------------------------------------------------------------------------------
def _work(args):
    function, inst, kwargs = args
    try:
        return function(inst, **kwargs), None
    except Exception as e:                                   # the run goes on; the instance is reported
        return None, f"{instance_id(inst)}: {type(e).__name__}: {e}"


def run_instances(instances, function, path: Path, fields, workers: int, describe, kwargs=None):
    """Runs function(inst) for every instance that is not yet in the file `path`, `workers` at a time, and
    appends one row per instance. describe(row) is the text printed when an instance is finished."""
    path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if path.exists():
        with open(path, newline="") as f:
            done = {r["id"] for r in csv.DictReader(f)}
    todo = [i for i in instances if instance_id(i) not in done]
    print(f"{path.name}: {len(instances)} instances, {len(done)} already in the file, {len(todo)} to run, "
          f"{workers} at the same time", flush=True)
    if not todo:
        return
    t0 = time.perf_counter()
    new = not path.exists()
    failed = []
    with open(path, "a", newline="") as f, mp.get_context("spawn").Pool(workers) as pool:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        jobs = [(function, i, kwargs or {}) for i in todo]
        for n, (row, err) in enumerate(pool.imap_unordered(_work, jobs), 1):
            elapsed = (time.perf_counter() - t0) / 60
            if err is not None:
                failed.append(err)
                print(f"[{n}/{len(todo)}] FAILED {err}", flush=True)
                continue
            w.writerow({k: row.get(k, "") for k in fields})
            f.flush()
            print(f"[{n}/{len(todo)}] {describe(row)}   {elapsed:.1f} min", flush=True)
    if failed:
        (path.parent / (path.stem + "_failed.txt")).write_text("\n".join(failed) + "\n")
        print(f"{len(failed)} instances failed; see {path.stem}_failed.txt. Run again to repeat them.", flush=True)


def _describe(row) -> str:
    return (f"J={row['J']} T={row['T']} {row['dist']} cv={row['cv']} {row['relation']} K={row['K']} R={row['R']} "
            f"b={row['b']} v={row['v']} z={row['z']} level={row['level']}: gap {row['gap_pct']:.3f}% "
            f"(paired {row['paired_gap_pct']:.3f}%), no transshipment {row['gap_nt_pct']:.3f}%, "
            f"{row['states']:,} states, SDP-2 {row['seconds_sdp2']:.0f}s, heuristic {row['seconds_alg']:.0f}s")


def block_instances(block: str, quick: bool = False):
    """All instances of a block, for every J in LOCATIONS, before the one-hour rule."""
    make = dict(core=core_instances, patterns=pattern_instances, pivot=pivot_instances)[block]
    rows = [r for J in LOCATIONS for r in make(J)]
    if quick:
        rows = [r for r in rows if r["K"] == 30 and r["R"] == 5 and r["dist"] == "poisson" and r["J"] <= 3
                and r["relation"] in ("LCY1-LCY2", "STAT-STAT/3", "EMP1-EMP2")
                and (block != "pivot" or r["factor"] in ("b", "level"))]
    rows.sort(key=lambda r: -r["T"])                         # the long instances first
    return rows


def one_hour_rule(instances, path: Path):
    """Splits the instances into those that are run and those that are deleted because SDP-2 would list more
    than VECTOR_LIMIT shipment vectors. The deleted ones are written to `path`."""
    kept, deleted = [], []
    for inst in instances:
        size = size_of(inst)
        if size["vectors"] > VECTOR_LIMIT:
            means, _, _ = demand_of(inst)
            deleted.append(dict(inst, id=instance_id(inst), means=means_text(means), **size,
                                hours_estimated=f"{size['vectors'] / VECTORS_PER_SECOND / 3600:.3g}"))
        else:
            kept.append(inst)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=DELETED_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(deleted)
    for J in sorted({i["J"] for i in instances}):
        n = sum(1 for i in instances if i["J"] == J)
        d = [x for x in deleted if x["J"] == J]
        text = f"  J = {J}: {n} instances, {n - len(d)} are run, {len(d)} are deleted by the one-hour rule"
        if d:
            text += (f" (their shipment vectors over the horizon: from {min(x['vectors'] for x in d):.1e} to "
                     f"{max(x['vectors'] for x in d):.1e}; the limit is {VECTOR_LIMIT:.1e})")
        print(text, flush=True)
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", default=RUN_BLOCK, choices=("core", "patterns", "pivot", "all"))
    ap.add_argument("--quick", action="store_true", default=RUN_QUICK)
    ap.add_argument("--workers", type=int, default=N_WORKERS)
    a = ap.parse_args()
    out = Path(str(OUT_DIR) + ("_quick" if a.quick else ""))
    kwargs = dict(max_paths=3000) if a.quick else {}
    for block in (("core", "patterns", "pivot") if a.block == "all" else (a.block,)):
        print(f"block {block}", flush=True)
        kept = one_hour_rule(block_instances(block, a.quick), out / f"{block}_deleted.csv")
        run_instances(kept, run_instance, out / f"{block}.csv", FIELDS, a.workers, _describe, kwargs)
    print("results ->", out)


if __name__ == "__main__":
    main()
