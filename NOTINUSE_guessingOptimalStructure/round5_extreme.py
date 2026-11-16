"""
NOTINUSE_guessingOptimalStructure/round5_extreme.py : stress test of the NOTINUSE_guessingOptimalStructure theorem on extreme instances, and search for further
regularities of the optimal policy.

    python round5_extreme.py [--workers 8] [--sets corners random]

Writes out/round5.csv (one row per instance and period) and out/round5_examples.json (every violation found).
Instances already in the csv are skipped, so a run can be resumed.

Instance sets, all with z = 1
    corners   every combination of the extreme values K in {0.5, 300}, R in {0.1, 300}, v in {0.01, 1},
              b in {1.05, 100}, h in {0.05, 10}, with three demand settings at T = 3                  96 instances
    random2   costs drawn log-uniformly, K in [0.5, 300], R in [0.1, 300], v in [0.01, 1], b in [1.05, 100],
              h in [0.05, 10]; horizon 2 to 5; a demand setting drawn from DEMANDS; Poisson demand or normal
              demand with a coefficient of variation in {0.02, 0.1, 0.3, 0.5}. The expected demands are scaled
              down until (3 + T) D <= 200, where D is the largest truncated demand of a period, so that every
              instance is solved in about a minute                                                    600 instances
    (random   the first version of this set without the scaling; 57 instances were solved before the run was
              stopped because single instances took hours; they are kept in out/round5_part1.csv)

Checks of the proved statements: V1 to V4 of round2.check_period, and V5t for the last period. Each must be 0.
    V5t compares the thresholds and levels of both locations with the (s, S) values of Lemma 7, computed here
    from the demand distribution alone. It is tie-aware: s is the largest level with f(x) > K + f(S) + TOL.
    (V5 of round2.check_period compares with the single-location solver, which reorders in a tie. It is kept in
    the csv, and a value above 0 that V5t does not confirm is a tie.)
Tests of the threshold property: Q1 and Q2 of round2.check_period on every post-transshipment state of the window;
    Q1_reached on the post-transshipment states that the optimal first stage reaches from an opening state of the
    window; Q1_opening on the opening states (threshold_variants.variants).
Further regularities, none of them proved (counts of exceptions)
    E1  the reorder-up-to curve S^j_t(x) increases somewhere in the inventory x of the other location
    E2  the curve falls by more than one unit per unit of x somewhere
    E3  a transshipment level D^+_t(n) or D^-_t(n) decreases somewhere in the total inventory n
    E4  a level rises by more than one unit per unit of n somewhere
    E5  the sender reorders after a transshipment without shipping all its stock on hand
    E6  C~~_t violates supermodularity at some state; E7: it violates submodularity at some state
    regions  number of states of each of the ten types of action
All comparisons use the tolerance TOL of structure_tests.py, and ties never count as exceptions.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import itertools
import json
import multiprocessing as mp
import time
from dataclasses import dataclass

import numpy as np

from common import OUT, max_demand, single_location, solve
from conjecture_search import Instance3
from lt_sdp import demand_pmf                              # read only
from round2 import check_period
from structure_tests import TOL, test_second_stage, transshipment
from threshold_variants import variants

CSV = os.path.join(OUT, "round5b.csv")
EXAMPLES = os.path.join(OUT, "round5b_examples.json")

DEMANDS = {                                    # name -> expected demands of location 1 and of location 2 per period
    "low":      ([0.3] * 5, [0.6] * 5),
    "equal":    ([6] * 5, [6] * 5),
    "unequal":  ([1] * 5, [14] * 5),
    "spike":    ([1, 1, 18, 1, 1], [1, 18, 1, 1, 18]),
    "ramp":     ([2, 5, 9, 14, 20], [20, 14, 9, 5, 2]),
    "end":      ([1, 1, 1, 1, 25], [1, 1, 1, 1, 10]),
    "start":    ([25, 1, 1, 1, 1], [10, 1, 1, 1, 1]),
    "pulse":    ([12, 0.2, 12, 0.2, 12], [0.2, 12, 0.2, 12, 0.2]),
    "one-idle": ([0.1] * 5, [10] * 5),
    "counter":  ([1, 30, 1, 30, 1], [1, 12, 1, 12, 1]),
}


@dataclass
class Instance5(Instance3):
    set_name: str = "random"
    tag: str = ""

    @property
    def name(self) -> str:
        return f"{self.set_name}_{self.tag}"


def instances(sets):
    out = []
    if "corners" in sets:
        k = 0
        for dem, K, R, v, b, h in itertools.product(["equal", "unequal", "spike"], [0.5, 300.0], [0.1, 300.0],
                                                    [0.01, 1.0], [1.05, 100.0], [0.05, 10.0]):
            m1, m2 = DEMANDS[dem]
            out.append(Instance5(dem, 3, K, R, "poisson", None, z=1.0, v=v, h=h, b=b, mu1=m1[:3], mu2=m2[:3],
                                 set_name="corners", tag=f"{k:03d}"))
            k += 1
    if "random" in sets or "random2" in sets:
        scaled = "random2" in sets
        rng = np.random.default_rng(20261113 if scaled else 20261112)
        names = list(DEMANDS)
        dists = [("poisson", None), ("normal", 0.02), ("normal", 0.1), ("normal", 0.3), ("normal", 0.5)]
        lu = lambda a, b: float(np.exp(rng.uniform(np.log(a), np.log(b))))
        for k in range(600):
            dem = names[rng.integers(len(names))]
            d, c = dists[rng.integers(len(dists))]
            T = int(rng.integers(2, 6))
            K, R, v, b, h = lu(0.5, 300), lu(0.1, 300), lu(0.01, 1.0), lu(1.05, 100), lu(0.05, 10)
            K, R, v, b, h = (round(x, 3) for x in (K, R, v, b, h))
            m1, m2 = DEMANDS[dem]
            if d == "normal":                                   # the discretised normal needs a mean of at least 1
                m1, m2 = [max(1, x) for x in m1], [max(1, x) for x in m2]
            m1, m2 = list(m1[:T]), list(m2[:T])
            floor = 1.0 if d == "normal" else 0.1
            while scaled and (3 + T) * max_demand(m1, m2, d, c) > 200:
                m1 = [max(floor, round(0.8 * x, 1)) for x in m1]
                m2 = [max(floor, round(0.8 * x, 1)) for x in m2]
            out.append(Instance5(dem, T, K, R, d, c, z=1.0, v=v, h=h, b=b, mu1=m1, mu2=m2,
                                 set_name="random2" if scaled else "random", tag=f"{k:03d}"))
    return out


# ---------------------------------------------------------------------------------------------------------
def _tie_range(col, lev):
    """Smallest and largest minimiser of a one-dimensional array, with the tolerance TOL."""
    m = col.min()
    idx = np.flatnonzero(col <= m + TOL)
    return int(lev[idx[0]]), int(lev[idx[-1]])


def extra(sol, t):
    c = sol.inst.costs
    v = c["v"]
    w, lev, lo, hi, lo_w = sol.win(), sol.levels, sol.lo, sol.hi, sol.lo_w
    m = len(lev)
    Hw, Cq = sol.H[t][w, w], sol.Cq[t][w, w]
    W = transshipment(sol, t)
    x1 = lev[:, None] - W
    x2 = lev[None, :] + W
    Qp = sol.Q[t][x1 - lo_w, x2 - lo_w]                      # reorders that follow the optimal transshipment
    q1, q2 = Qp[..., 0] > 0, Qp[..., 1] > 0
    row = {}

    # ten types of action at the opening states of the window
    typ = {"none": (W == 0) & ~q1 & ~q2, "r1": (W == 0) & q1 & ~q2, "r2": (W == 0) & ~q1 & q2, "r12": (W == 0) & q1 & q2,
           "s12": (W > 0) & ~q1 & ~q2, "s21": (W < 0) & ~q1 & ~q2,
           "s12_recv": (W > 0) & q2 & ~q1, "s21_recv": (W < 0) & q1 & ~q2,
           "s12_send": (W > 0) & q1 & ~q2, "s21_send": (W < 0) & q2 & ~q1}
    for k, msk in typ.items():
        row[f"n_{k}"] = int(msk.sum())
    row["n_types"] = int(sum(1 for msk in typ.values() if msk.any()))

    # E5 : does the sender ship all its stock when it reorders afterwards?  the same count for the other types
    i1 = lev[:, None] + 0 * W
    i2 = lev[None, :] + 0 * W
    all12, all21 = (W > 0) & (W == i1), (W < 0) & (-W == i2)
    row["E5"] = int((typ["s12_send"] & ~all12).sum() + (typ["s21_send"] & ~all21).sum())
    row["recv_ships_all"] = int((typ["s12_recv"] & all12).sum() + (typ["s21_recv"] & all21).sum())
    row["only_ships_all"] = int((typ["s12"] & all12).sum() + (typ["s21"] & all21).sum())

    # E1, E2 : reorder-up-to curves, on the inventories of the other location at which the curve is used
    Qx = sol.Q[t][w, w]
    patx = (Qx[..., 0] > 0) * 1 + (Qx[..., 1] > 0) * 2
    e1 = e2 = used_pairs = 0
    for loc, H, pat_only in ((1, Hw, patx == 1), (2, Hw.T, (patx == 2).T)):
        rng = [_tie_range(H[:, k], lev) for k in range(m)]
        used = [bool(pat_only[:, k].any()) and lo < rng[k][0] and rng[k][1] < hi for k in range(m)]
        for k in range(m - 1):
            if used[k] and used[k + 1]:
                used_pairs += 1
                e1 += rng[k + 1][0] > rng[k][1]
                e2 += rng[k + 1][1] < rng[k][0] - 1
    row.update(E1=int(e1), E2=int(e2), curve_pairs=int(used_pairs))

    # E3, E4 : transshipment levels along the lines, on the lines on which the level is used
    e3 = e4 = line_pairs = 0
    prev = {"+": None, "-": None}
    for n in range(2 * lo, 2 * hi + 1):
        a = np.arange(max(0, n - 2 * lo - (m - 1)), min(m - 1, n - 2 * lo) + 1)
        b = n - 2 * lo - a
        cur = {"+": None, "-": None}
        if len(a) >= 2:
            i1l, G, Wl = lev[a], Cq[a, b], W[a, b]
            for sgn, dom, phi, ship in (("+", i1l >= 0, G - v * i1l, Wl > 0), ("-", i1l <= n, G + v * i1l, Wl < 0)):
                d = np.flatnonzero(dom)
                if len(d) and ship.any():
                    mval = phi[d].min()
                    tie = d[phi[d] <= mval + TOL]
                    dmin, dmax = int(i1l[tie[0]]), int(i1l[tie[-1]])
                    if i1l[0] < dmin and dmax < i1l[-1]:              # not at an end of the line inside the window
                        cur[sgn] = (dmin, dmax)
        for sgn in "+-":
            if cur[sgn] is not None and prev[sgn] is not None:
                line_pairs += 1
                e3 += cur[sgn][1] < prev[sgn][0]
                e4 += cur[sgn][0] > prev[sgn][1] + 1
            prev[sgn] = cur[sgn]
    row.update(E3=int(e3), E4=int(e4), line_pairs=int(line_pairs))

    # E6, E7 : supermodularity and submodularity of the second-stage value on the window
    d = Cq[1:, 1:] + Cq[:-1, :-1] - Cq[1:, :-1] - Cq[:-1, 1:]
    scale = 1e-9 * max(1.0, float(np.abs(Cq).max()))
    row.update(E6=int((d < -TOL - scale).sum()), E7=int((d > TOL + scale).sum()), cells=int(d.size))
    return row


def last_period_check(sol):
    """V5t: thresholds and levels of the last period against Lemma 7, computed from the demand distribution."""
    c = sol.inst.costs
    K, z, h, b = c["K"], c["z"], c["h"], c["b"]
    t = sol.T - 1
    sec = test_second_stage(sol, t)
    lev = np.arange(sol.lo_w, sol.hi_w + 1)
    bad, ex = 0, []
    for loc, mu in ((1, sol.mu1), (2, sol.mu2)):
        cv = sol.inst.cv
        p = demand_pmf(mu[t], sol.inst.dist, cv[t] if np.ndim(cv) else cv, 1e-4)
        d = np.arange(len(p))
        f = z * lev + (np.maximum(lev[:, None] - d[None, :], 0) * h + np.maximum(d[None, :] - lev[:, None], 0) * b) @ p
        fmin = f.min()
        mins = lev[f <= fmin + TOL]
        S = int(mins[0])
        strict = lev[(lev < S) & (f - K - fmin > TOL)]
        s = int(strict.max()) if len(strict) else None
        want = s if (s is not None and s >= sol.lo) else None
        thr = set(sec[f"loc{loc}"]["threshold"])
        Sv = set(y for lst in sec[f"loc{loc}"]["S_only"] for y in lst)
        if thr != {want} or not Sv <= set(int(x) for x in mins):
            bad += 1
            ex.append(dict(check="V5t", t=t + 1, location=loc, lemma=(s, S), thresholds=sorted(thr, key=str),
                           levels=sorted(Sv)))
    return bad, ex


def run(inst, hi_factor=1.0):
    """hi_factor > 1 solves on a range whose upper end is hi_factor times the adequate one and extends the window to it.
    It is used when a reorder-up-to level of a window state lies above the adequate range, which the consistency
    assertion of structure_tests.test_second_stage detects."""
    t0 = time.perf_counter()
    sol = solve(inst, hi_factor=hi_factor, cache=False)
    if hi_factor != 1.0:
        sol.hi = sol.hi_w
    single = [single_location(sol, 1), single_location(sol, 2)]
    rows, examples = [], []
    for t in range(sol.T):
        row, ex = check_period(sol, t, single)
        c = inst.costs
        head = dict(instance=inst.name, set=inst.set_name, demand=inst.relation, dist=inst.dist,
                    cv="" if inst.cv is None else inst.cv, T=inst.T, K=c["K"], R=c["R"], z=c["z"], v=c["v"], h=c["h"],
                    b=c["b"], mu1=" ".join(f"{x:g}" for x in sol.mu1), mu2=" ".join(f"{x:g}" for x in sol.mu2),
                    lo=sol.lo, hi=sol.hi, lo_w=sol.lo_w, t=t + 1)
        head.update(row)
        head.update(extra(sol, t))
        var = variants(sol, t)
        for name in ("all", "reached", "opening"):
            head[f"Q1_{name}"] = int(var[1][0][name] + var[2][0][name])
        for loc in (1, 2):
            for name in ("reached", "opening"):
                if name in var[loc][1]:
                    ex.append(dict(check=f"Q1_{name}", t=t + 1, location=loc, **var[loc][1][name]))
        if t == sol.T - 1:
            head["V5t"], ex5 = last_period_check(sol)
            ex += ex5
        else:
            head["V5t"] = ""
        rows.append(head)
        for e in ex:
            e["instance"] = inst.name
        examples += ex
    for r in rows:
        r["seconds"] = round(time.perf_counter() - t0, 1)
    return rows, examples


def _work(inst):
    for f in (1.0, 1.6, 2.5):
        try:
            rows, ex = run(inst, f)
            for r in rows:
                r["hi_factor"] = f
            return rows, ex
        except AssertionError:                              # a level above the range: widen the upper end and repeat
            continue
        except Exception as e:                              # report and continue
            return [dict(instance=inst.name, error=repr(e))], []
    return [dict(instance=inst.name, error="AssertionError() with every hi_factor")], []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--sets", nargs="+", default=["corners", "random2"])
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    todo = instances(a.sets)
    done = set()
    if os.path.exists(CSV):
        with open(CSV) as f:
            done = {r["instance"] for r in csv.DictReader(f)}
    todo = [i for i in todo if i.name not in done]
    if a.limit:
        todo = todo[:a.limit]
    print(f"{len(todo)} instances to run, {len(done)} done", flush=True)
    examples = json.load(open(EXAMPLES)) if os.path.exists(EXAMPLES) else []
    errors = []
    fields = None
    if os.path.exists(CSV):
        with open(CSV) as f:
            fields = csv.DictReader(f).fieldnames
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for k, (rows, ex) in enumerate(pool.imap_unordered(_work, todo), 1):
            if "error" in rows[0]:
                errors.append(rows[0])
                print("ERROR", rows[0], flush=True)
                with open(os.path.join(OUT, "round5b_errors.json"), "w") as f:
                    json.dump(errors, f, indent=1)
                continue
            if fields is None:
                fields = list(rows[0].keys())
                with open(CSV, "w", newline="") as f:
                    csv.DictWriter(f, fieldnames=fields).writeheader()
            with open(CSV, "a", newline="") as f:
                wr = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                for r in rows:
                    wr.writerow(r)
            examples += ex
            with open(EXAMPLES, "w") as f:
                json.dump(examples, f, indent=1)
            print(f"{k}/{len(todo)} {rows[0]['instance']} {rows[0]['seconds']}s", flush=True)
    print("finished", flush=True)


if __name__ == "__main__":
    main()
