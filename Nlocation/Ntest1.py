"""
Ntest1.py : computational effort of SDP-1 and SDP-2 (Section 2.3 and Table 1 of the paper).

Run this file. It writes its results to results_Ntest1/ next to it. A new run replaces that folder.
    runs.csv          one row per solver and instance: status, time, time of each stage, peak memory
    instances.csv     one row per instance: inventory levels, number of states, work counts, optimal cost,
                      largest difference between the two value functions, check of the inventory levels
    table_effort.tex  the rows of Table 1
    log.txt           what was run and when

Solvers, each in its own file
    SDP1        ml_SDP1.py        at every state every shipment vector w; the reorder minimisation is repeated at
                                  every state for each post-transshipment state that the state can reach
    SDP2        ml_SDP2.py        the reorder minimisation once per post-transshipment state; at every state every
                                  shipment vector w

Instances: integer demand and integer quantities
    J locations, T = 4 periods, opening inventory 0, Poisson demand.
    The expected demand of location j in period t is 0.5 mu if j + t is even and 1.5 mu if j + t is odd, so every
    location has the expected demand T mu over the horizon and adjacent locations have opposite patterns.
    Costs of the test bed of the paper: K = 20, z = 1, R = 10, v = 0.5, h = 1, b = 5.

Inventory levels of every location: the integers min_inventory..max_inventory
    max_inventory   the 1 - TAIL quantile of the total demand of all locations over the horizon. It is the sum of
                    all expected demands plus a buffer, and one location can hold the stock of the whole system.
    min_inventory   minus the largest demand that one location can have over the horizon, which is the sum over the
                    periods of the largest values of its truncated demand. No state below it can be reached from
                    the opening inventory, so no demand outcome is dropped on a path from the opening inventory.
    Check: an instance that SDP2 solves in at most CHECK_BELOW seconds is solved again by SDP2 on levels with both
    ends moved out by 25%. instances.csv reports the two optimal costs.

Time limit and what is not run
    A run is stopped after TIME_LIMIT seconds. A solver that reaches the limit is not run on the instances with the
    same number of locations and a larger demand. It is also not run on an instance on which it lists at least as
    many shipment vectors per period as on an instance on which it reached the limit.
    The time of a run is the time of the solver, from its first line to its last. A run shorter than REPEAT_BELOW
    seconds is made REPEATS times and the median time is kept.

Parallel runs
    Every run is one process that uses one processor core. Up to WORKERS runs are carried out at the same time.

Work counts of one period, exact
    states                  (number of inventory levels)^J
    shipment vectors        the vectors w that both solvers list: the sum over the states of the size of the set
                            W(i), before the highest inventory level is applied. Table 1: transshipment candidates.
    reorder minimisations   of SDP1: the different post-transshipment states within the inventory levels that a
                            state can reach, summed over the states. SDP2 carries out one per state.

Repeating the summary without repeating the runs
    Run once with KEEP_VALUES = True, then run with REUSE_RUNS = True: the runs recorded in runs.csv are kept and
    only the missing ones are made.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"                                   # one processor core per run

import csv
import json
import math
import multiprocessing as mp
import resource
import shutil
import sys
import threading
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for _p in (str(HERE.parent), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ml_common import all_transshipments, period_pmfs, total_demand_quantile  # noqa: E402

# ----------------------------------------------------------------------------------------------------
# settings
# ----------------------------------------------------------------------------------------------------
RUN_QUICK = False                 # True: two demand levels and a limit of 20 seconds, to check that everything runs
REUSE_RUNS = False                # True: keep the runs recorded in runs.csv and make only the missing ones
KEEP_VALUES = False               # True: keep the value functions in results_Ntest1/values for a later REUSE_RUNS
LOCATIONS = (2, 3, 4)             # numbers of locations J
MU = (0.25, 0.5, 1, 2, 4, 8)      # average expected demand of one location in one period
T = 4                             # periods
COSTS = dict(K=20, z=1, R=10, v=0.5, h=1, b=5)
TAIL = 1e-4                       # truncation of the demand and buffer of the inventory levels
TIME_LIMIT = 1800                 # seconds for one run
REPEAT_BELOW, REPEATS = 10, 3     # a run shorter than REPEAT_BELOW seconds is made REPEATS times; the median is kept
VALUE_BYTES = 100e6               # a larger value function is not kept for the comparison between the solvers
WORKERS = 7                       # runs carried out at the same time
WIDEN = 0.25                      # the check of the inventory levels moves both ends out by this share
CHECK_BELOW = 60                  # the levels are checked on the instances that SDP2 solves in at most this many seconds
SOLVERS = ("SDP1", "SDP2")
OUT_DIR = HERE / "results_Ntest1"

if RUN_QUICK or "--quick" in sys.argv:
    MU, TIME_LIMIT, OUT_DIR = (0.25, 0.5), 20, HERE / "results_Ntest1_quick"
REUSE_RUNS = REUSE_RUNS or "--reuse" in sys.argv
KEEP_VALUES = KEEP_VALUES or "--keep" in sys.argv

VALUES = OUT_DIR / "values"       # value functions of the runs; removed at the end unless KEEP_VALUES


# ----------------------------------------------------------------------------------------------------
# instances
# ----------------------------------------------------------------------------------------------------
def demand_means(J: int, mu: float):
    """Expected demand of location j in period t: 0.5 mu if j + t is even, 1.5 mu if j + t is odd."""
    return [[mu * (0.5 if (j + t) % 2 == 0 else 1.5) for t in range(1, T + 1)] for j in range(1, J + 1)]


def inventory_range(means):
    """(min_inventory, max_inventory) of every location; see the top of the file."""
    pmfs = period_pmfs(means, "poisson", None, TAIL)
    lo = -max(sum(len(p) - 1 for p in pm) for pm in pmfs)
    hi = int(total_demand_quantile(means, "poisson", None, TAIL))
    return lo, hi


def widened_range(lo: int, hi: int):
    """Both ends moved out by the share WIDEN."""
    return math.floor(lo * (1 + WIDEN)), math.ceil(hi * (1 + WIDEN))


def _convolve(a, b):
    """Convolution with exact integers."""
    out = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x:
            for k, y in enumerate(b):
                out[i + k] += x * y
    return out


def work_counts(J: int, lo: int, hi: int):
    """(states, shipment vectors, reorder minimisations of SDP1) of one period, exact.
    shipment vectors: a location with s units on hand and J - 1 destinations ships in C(s + J - 1, J - 1) ways.
    reorder minimisations of SDP1: a transshipment moves location j from the level x by d units with
    -max(x, 0) <= d <= hi - x. c[d + hi] counts the levels at which d is possible. A pair of a state and one of
    its post-transshipment states is one choice of d per location with sum 0, which is one entry of the J-fold
    convolution of c."""
    levels = range(lo, hi + 1)
    vectors = sum(math.comb(max(x, 0) + J - 1, J - 1) for x in levels) ** J
    c = [0] * (2 * hi - lo + 1)
    for x in levels:
        for d in range(-max(x, 0), hi - x + 1):
            c[d + hi] += 1
    acc = c
    for _ in range(J - 1):
        acc = _convolve(acc, c)
    return len(levels) ** J, vectors, acc[J * hi]


@lru_cache(maxsize=None)
def _instance(J: int, mu: float):
    means = demand_means(J, mu)
    lo, hi = inventory_range(means)
    states, vectors, minimisations = work_counts(J, lo, hi)
    return dict(J=J, T=T, mu=mu, mean_low=0.5 * mu, mean_high=1.5 * mu, total_mean=J * T * mu,
                min_inventory=lo, max_inventory=hi, levels=hi - lo + 1, states=states,
                shipment_vectors=vectors, reorder_minimisations_SDP1=minimisations)


def instance(J: int, mu: float) -> dict:
    """Inventory levels and work counts of one instance."""
    return dict(_instance(J, mu))


def _name(solver: str, J: int, mu: float) -> str:
    return f"{solver}_J{J}_mu{mu:g}"


def _values_kept(J: int, mu: float) -> bool:
    """Whether the value function of this instance is small enough to be kept for the comparison."""
    return _instance(J, mu)["states"] * T * 8 <= VALUE_BYTES


# ----------------------------------------------------------------------------------------------------
# one run, in its own process
# ----------------------------------------------------------------------------------------------------
def _solve(solver: str, J: int, mu: float, path: str):
    from ml_SDP1 import SDP1
    from ml_SDP2 import SDP2
    means = demand_means(J, mu)
    lo, hi = inventory_range(means)
    solve = {"SDP1": SDP1, "SDP2": SDP2}[solver]
    res = solve(means, lo, hi, tail=TAIL, **COSTS)
    seconds = [res.seconds]
    while seconds[0] < REPEAT_BELOW and len(seconds) < REPEATS:          # a short run is repeated
        seconds.append(solve(means, lo, hi, tail=TAIL, **COSTS).seconds)
    if _values_kept(J, mu):
        np.save(path + ".npy", res.C)
    info = dict(seconds=float(np.median(seconds)), runs=len(seconds), value=res.value(),
                peak_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20,
                **{"seconds_" + k: s for k, s in res.seconds_stage.items()})
    with open(path + ".tmp", "w") as f:
        json.dump(info, f)
    os.replace(path + ".tmp", path + ".json")              # the file exists only when the run is complete


def _check_levels(J: int, mu: float, path: str):
    """The instance by SDP2 on inventory levels with both ends moved out."""
    from ml_SDP2 import SDP2
    means = demand_means(J, mu)
    lo_w, hi_w = widened_range(*inventory_range(means))
    wide = SDP2(means, lo_w, hi_w, tail=TAIL, **COSTS)
    info = dict(value_widened=wide.value(), min_inventory_widened=lo_w, max_inventory_widened=hi_w)
    with open(path + ".tmp", "w") as f:
        json.dump(info, f)
    os.replace(path + ".tmp", path + ".json")


# ----------------------------------------------------------------------------------------------------
# the runs
# ----------------------------------------------------------------------------------------------------
_lock = threading.Lock()
_slots = threading.Semaphore(WORKERS)
_processes = []
_rows = []                        # rows of runs.csv
_previous = {}                    # runs of an earlier call, with REUSE_RUNS
_limit_vectors = {}               # solver -> smallest number of shipment vectors at which it reached the time limit
RUN_FIELDS = ["solver", "J", "mu", "status", "seconds", "runs", "seconds_expectation", "seconds_reorder",
              "seconds_transshipment", "peak_mb", "value", "wall_seconds", "note"]
_NUMBERS = ("seconds", "seconds_expectation", "seconds_reorder", "seconds_transshipment", "peak_mb", "value",
            "wall_seconds")


def log(text: str):
    line = f"{time.strftime('%H:%M:%S')}  {text}"
    with _lock:
        print(line, flush=True)
        with open(OUT_DIR / "log.txt", "a") as f:
            f.write(line + "\n")


def _read_runs(path) -> dict:
    out = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            row = {k: v for k, v in r.items() if v != ""}
            if row["solver"] not in SOLVERS:               # a solver of an earlier version of this file
                continue
            row["J"], row["mu"] = int(row["J"]), float(row["mu"])
            for k in _NUMBERS:
                if k in row:
                    row[k] = float(row[k])
            if "runs" in row:
                row["runs"] = int(row["runs"])
            row.setdefault("note", "")
            out[(row["solver"], row["J"], row["mu"])] = row
    return out


def _record(row: dict):
    with _lock:
        _rows.append(row)
        with open(OUT_DIR / "runs.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=RUN_FIELDS)
            w.writeheader()
            w.writerows(sorted(_rows, key=lambda r: (r["J"], r["mu"], SOLVERS.index(r["solver"]))))


def _process(target, args, limit: float):
    """Runs target(*args) in its own process. Returns (finished, wall seconds)."""
    p = mp.Process(target=target, args=args, daemon=True)
    with _lock:
        _processes.append(p)
    t0 = time.perf_counter()
    p.start()
    p.join(limit)
    finished = not p.is_alive()
    if not finished:
        p.terminate()
        p.join()
    return finished and p.exitcode == 0, time.perf_counter() - t0


def run(solver: str, J: int, mu: float) -> dict:
    """One solver on one instance, with the time limit."""
    path = str(VALUES / _name(solver, J, mu))
    with _slots:
        log(f"start    {solver:9s} J={J} mu={mu:g}")
        finished, wall = _process(_solve, (solver, J, mu, path), TIME_LIMIT + 60)   # 60 s for the start of the process
    row = dict(solver=solver, J=J, mu=mu, wall_seconds=round(wall, 2), note="")
    if finished and os.path.exists(path + ".json"):
        with open(path + ".json") as f:
            info = json.load(f)
        row.update({k: info.get(k) for k in RUN_FIELDS if k in info})
        row["status"] = "solved" if info["seconds"] <= TIME_LIMIT else "time limit"
    else:
        row["status"] = "time limit" if wall >= TIME_LIMIT else "failed"
    if row["status"] != "solved":
        for ext in (".npy", ".json", ".tmp"):
            if os.path.exists(path + ext):
                os.remove(path + ext)
    log(f"{row['status']:8s} {solver:9s} J={J} mu={mu:g}   {row.get('seconds', wall):.2f} s")
    return row


def _reusable(solver: str, J: int, mu: float):
    """The run of an earlier call, if it can stand for a new one."""
    row = _previous.get((solver, J, mu))
    if row is None:
        return None
    if row["status"] == "time limit":
        return row
    if row["status"] == "solved" and (not _values_kept(J, mu) or (VALUES / (_name(solver, J, mu) + ".npy")).exists()):
        return row
    return None


def chain(solver: str, locations):
    """The instances of one solver for the given numbers of locations, the demand levels ascending."""
    for J in locations:
        stopped = False
        for mu in MU:
            inst = _instance(J, mu)
            vectors = inst["shipment_vectors"]
            row = _reusable(solver, J, mu)
            if row is not None:
                log(f"kept     {solver:9s} J={J} mu={mu:g}   {row['status']}")
            else:
                note = None
                if stopped:
                    note = "a lower demand level reached the time limit"
                elif vectors >= _limit_vectors.get(solver, math.inf):
                    note = "at least as many shipment vectors as an instance at the time limit"
                if note:
                    _record(dict(solver=solver, J=J, mu=mu, status="not run", note=note))
                    log(f"not run  {solver:9s} J={J} mu={mu:g}   {note}")
                    continue
                row = run(solver, J, mu)
            _record(row)
            if row["status"] != "solved":
                stopped = True
                if row["status"] == "time limit":
                    with _lock:
                        _limit_vectors[solver] = min(vectors, _limit_vectors.get(solver, math.inf))


def all_runs():
    """Each solver takes three locations before four, so that the rule on the shipment vectors applies to four."""
    plans = [(s, (2,)) for s in SOLVERS if 2 in LOCATIONS]
    rest = tuple(J for J in LOCATIONS if J != 2)
    plans += [(s, rest) for s in SOLVERS if rest]
    threads = [threading.Thread(target=chain, args=plan) for plan in plans]
    for th in threads:
        th.start()
    for th in threads:
        th.join()


# ----------------------------------------------------------------------------------------------------
# after the runs: agreement of the value functions, check of the inventory levels, table
# ----------------------------------------------------------------------------------------------------
def _check_thread(J, mu, out):
    path = str(VALUES / _name("check", J, mu))
    with _slots:
        log(f"start    check     J={J} mu={mu:g}")
        finished, wall = _process(_check_levels, (J, mu, path), TIME_LIMIT)
    if finished and os.path.exists(path + ".json"):
        with open(path + ".json") as f:
            out[(J, mu)] = json.load(f)
        os.remove(path + ".json")
    log(f"{'checked' if (J, mu) in out else 'no check':8s} check     J={J} mu={mu:g}   {wall:.2f} s")


def summarise():
    status = {(r["solver"], r["J"], r["mu"]): r for r in _rows}
    checks, threads = {}, []
    for J in LOCATIONS:
        for mu in MU:
            r = status[("SDP2", J, mu)]
            if r["status"] == "solved" and r["seconds"] <= CHECK_BELOW:
                threads.append(threading.Thread(target=_check_thread, args=(J, mu, checks)))
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    rows = []
    for J in LOCATIONS:
        for mu in MU:
            row = instance(J, mu)
            solved = [s for s in SOLVERS if status[(s, J, mu)]["status"] == "solved"]
            row["solved_by"] = " ".join(solved)
            for s in SOLVERS:
                r = status[(s, J, mu)]
                row["seconds_" + s] = r.get("seconds") if r["status"] == "solved" else r["status"]
            if solved:
                row["value"] = status[(solved[-1], J, mu)]["value"]
                files = {s: VALUES / (_name(s, J, mu) + ".npy") for s in solved}
                C = {s: np.load(p) for s, p in files.items() if p.exists()}
                row["compared"] = " ".join(C)
                if len(C) > 1:
                    row["max_difference"] = max(float(np.max(np.abs(C[a] - C[b]))) for a in C for b in C if a < b)
            row.update(checks.get((J, mu), {}))
            rows.append(row)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(OUT_DIR / "instances.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    write_table(rows)
    return rows


def _count(x: int) -> str:
    """An exact count for the table: digits with thin spaces up to 9,999,999, then two significant digits."""
    if x < 10 ** 7:
        return f"{x:,}".replace(",", "\\,")
    e = int(math.floor(math.log10(x)))
    m = x / 10 ** e
    if round(m, 1) >= 10:
        m, e = m / 10, e + 1
    return f"${m:.1f}\\times10^{{{e}}}$"


def _time(x) -> str:
    if x == "time limit":
        return "limit"
    if x == "not run" or x is None or x == "failed":
        return "--"
    x = float(x)
    if x < 0.1:
        return f"{x:.3f}"
    return f"{x:.2f}" if x < 10 else (f"{x:.1f}" if x < 100 else f"{x:,.0f}".replace(",", "\\,"))


def write_table(rows):
    """The rows of Table 1: one row per instance."""
    lines = []
    for r in rows:
        lines.append(" & ".join([
            str(r["J"]), f"{r['mu']:g}", f"$[{r['min_inventory']},{r['max_inventory']}]$", _count(r["states"]),
            _count(r["shipment_vectors"]), _count(r["reorder_minimisations_SDP1"])]
            + [_time(r["seconds_" + s]) for s in SOLVERS]) + r" \\")
    with open(OUT_DIR / "table_effort.tex", "w") as f:
        f.write("% J & mu & levels & states per period & transshipment candidates & reorder minimisations in SDP-1 & "
                + " & ".join(SOLVERS) + "\n")
        f.write("\n".join(lines) + "\n")


def self_check():
    """The exact work counts against a direct count on small ranges: at every state the shipment vectors are
    listed and the post-transshipment states within the inventory levels are counted, as SDP1 does."""
    for J, lo, hi in ((2, -3, 5), (3, -2, 4), (4, -1, 2)):
        states, vectors, minimisations = work_counts(J, lo, hi)
        direct_v = direct_m = 0
        for state in np.ndindex((hi - lo + 1,) * J):
            s = np.array(state) + lo
            D, _ = all_transshipments(s, 0.0, 0.0)
            direct_v += len(D)
            post = s[None, :] + D
            direct_m += len(np.unique(post[np.all(post <= hi, axis=1)], axis=0))
        if (vectors, minimisations) != (direct_v, direct_m):
            raise AssertionError(f"work counts differ for J={J}: {(vectors, minimisations)} "
                                 f"against {(direct_v, direct_m)}")


def main():
    if REUSE_RUNS and (OUT_DIR / "runs.csv").exists():
        _previous.update(_read_runs(OUT_DIR / "runs.csv"))
    elif OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    VALUES.mkdir(parents=True, exist_ok=True)
    self_check()
    log(f"Ntest1: J in {LOCATIONS}, mu in {MU}, T = {T}, costs {COSTS}, time limit {TIME_LIMIT} s, "
        f"{WORKERS} runs at the same time" + (f", {len(_previous)} earlier runs on record" if _previous else ""))
    for J in LOCATIONS:
        for mu in MU:
            r = instance(J, mu)
            log(f"instance J={J} mu={mu:g}: levels [{r['min_inventory']}, {r['max_inventory']}], "
                f"{r['states']:,} states, {r['shipment_vectors']:.3e} shipment vectors, "
                f"{r['reorder_minimisations_SDP1']:.3e} reorder minimisations of SDP1 per period")
    t0 = time.perf_counter()
    rows = []
    try:
        all_runs()
        rows = summarise()
    finally:
        for p in _processes:
            if p.is_alive():
                p.terminate()
    if not KEEP_VALUES:
        shutil.rmtree(VALUES, ignore_errors=True)
    log(f"finished in {time.perf_counter() - t0:.0f} s")
    for r in rows:
        log(f"J={r['J']} mu={r['mu']:g}: " + ", ".join(f"{s} {r['seconds_' + s]}" for s in SOLVERS)
            + f", value {r.get('value')}, largest difference {r.get('max_difference')}, "
              f"widened {r.get('value_widened')}")


if __name__ == "__main__":
    main()
