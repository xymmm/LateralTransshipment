"""
NOTINUSE_guessingOptimalStructure/round2.py : numerical verification of the NOTINUSE_guessingOptimalStructure theorem and tests of the threshold conjecture.

    python round2.py [--workers 8] [--sets A B C]

Writes out/round2.csv (one row per instance and period) and out/round2_examples.json (every violation found).
Instances already in the csv are skipped, so a run can be resumed.

Instance sets
    A  test bed II at T = 4: K in {20, 30, 50} x R in {2, 5, 10, 20} x {Poisson, normal 0.1, normal 0.3}
       x six relations between the demand patterns; z = 1, v = 0.5, h = 1, b = 5                      216 instances
    B  test bed II at T = 8: (K, R) in {(20, 20), (20, 2), (50, 20), (50, 2)} x {Poisson, normal 0.3}
       x six relations                                                                                48 instances
    D  outside the test beds, T = 4: (K, R, z, v, b) in {(100,2,1,.5,5), (5,20,1,.5,5), (20,5,1,1,10), (30,10,2,.5,3),
       (50,50,1,.5,20)} x {Poisson, normal 0.3} x SIN1-SIN2, RAND-RAND, EMP1-EMP2, LCY2-SIN1            40 instances
    C  other cost parameters at T = 4: b = 3, (z, v) = (2, 1), K in {10, 30}, R in {5, 20},
       {Poisson, normal 0.2}, relations STAT-STAT, LCY1-LCY2, STAT-STAT3, RAND-EMP1                    32 instances

Checks of proved statements (each count must be 0)
    V1  a strict transshipment followed by reorders at both locations
    V2  a strict transshipment after which the sender reorders, with Q <= |W| or K < R + v |W|
    V3  a reorder whose level differs in value from the unconstrained level although the state lies below it
    V4  a transshipment from a state beyond the transshipment level of its line that does not reach this level in
        value; a state beyond the level without transshipment although the level is cheaper by more than R;
        a line with D^- > D^+
    V5  last period: thresholds or levels that depend on the other location or differ from the single-location (s, S)
Tests of the conjecture
    Q1  threshold in the own inventory (structure_tests.test_second_stage)
    Q2  condition (Q): g(x) <= max{ g(x'), K + g(y) } for x' < x < y, where g is the cost of location j as a function
        of its level after reordering, the other location responding optimally
Descriptive counts
    lines i1 + i2 = n whose states without transshipment are not one interval; minimal constant of K-convexity of
    C~_t and C~~_t in the coordinates.
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

from common import OUT, Instance, M_LEVEL, pattern, single_location, solve
from structure_tests import (TOL, kappa_lines, reorder_pattern, second_stage_values, test_first_stage,
                             test_second_stage, transshipment)

CSV = os.path.join(OUT, "round2.csv")
EXAMPLES = os.path.join(OUT, "round2_examples.json")


@dataclass
class Instance2(Instance):
    z: float = 1.0
    v: float = 0.5
    h: float = 1.0
    b: float = 5.0
    set_name: str = "A"

    @property
    def name(self) -> str:
        d = self.dist if self.cv is None else f"{self.dist}{self.cv:g}"
        return (f"{self.set_name}_{self.relation}_{d}_T{self.T}_K{self.K:g}_R{self.R:g}"
                f"_z{self.z:g}_v{self.v:g}_b{self.b:g}")

    @property
    def costs(self) -> dict:
        return dict(K=float(self.K), R=float(self.R), z=self.z, v=self.v, h=self.h, b=self.b)

    def means(self):
        p1, p2 = self.relation.split("-")
        m1 = pattern(p1, self.T, M_LEVEL)
        if p2.endswith("r"):
            m2 = pattern(p2[:-1], self.T, M_LEVEL)[::-1]
        elif p2.endswith("3"):
            m2 = [max(1, int(np.floor(x / 3 + 0.5))) for x in pattern(p2[:-1], self.T, M_LEVEL)]
        else:
            m2 = pattern(p2, self.T, M_LEVEL)
        return m1, m2


RELATIONS = ["STAT-STAT", "LCY1-LCY1", "LCY1-LCY2", "SIN1-SIN1r", "LCY2-LCY2r", "STAT-STAT3"]


def instances(sets):
    out = []
    if "A" in sets:
        for (d, c), rel, K, R in itertools.product([("poisson", None), ("normal", 0.1), ("normal", 0.3)], RELATIONS,
                                                   [20, 30, 50], [2, 5, 10, 20]):
            out.append(Instance2(rel, 4, K, R, d, c, set_name="A"))
    if "B" in sets:
        for (d, c), rel, (K, R) in itertools.product([("poisson", None), ("normal", 0.3)], RELATIONS,
                                                     [(20, 20), (20, 2), (50, 20), (50, 2)]):
            out.append(Instance2(rel, 8, K, R, d, c, set_name="B"))
    if "C" in sets:
        for (d, c), rel, K, R in itertools.product([("poisson", None), ("normal", 0.2)],
                                                   ["STAT-STAT", "LCY1-LCY2", "STAT-STAT3", "RAND-EMP1"], [10, 30], [5, 20]):
            out.append(Instance2(rel, 4, K, R, d, c, z=2.0, v=1.0, b=3.0, set_name="C"))
    if "D" in sets:        # cost settings and pattern pairs outside the test beds of the manuscript
        for (d, c), rel, (K, R, z, v, b) in itertools.product(
                [("poisson", None), ("normal", 0.3)], ["SIN1-SIN2", "RAND-RAND", "EMP1-EMP2", "LCY2-SIN1"],
                [(100, 2, 1.0, 0.5, 5.0), (5, 20, 1.0, 0.5, 5.0), (20, 5, 1.0, 1.0, 10.0), (30, 10, 2.0, 0.5, 3.0),
                 (50, 50, 1.0, 0.5, 20.0)]):
            out.append(Instance2(rel, 4, K, R, d, c, z=z, v=v, b=b, set_name="D"))
    return out


# ---------------------------------------------------------------------------------------------------------
def check_period(sol, t, single):
    c = sol.inst.costs
    K, R, z, v = c["K"], c["R"], c["z"], c["v"]
    w, lev, lo, lo_w = sol.win(), sol.levels, sol.lo, sol.lo_w
    m = len(lev)
    Hw, Cq = sol.H[t][w, w], sol.Cq[t][w, w]
    W = transshipment(sol, t)
    x1 = lev[:, None] - W
    x2 = lev[None, :] + W
    Qp = sol.Q[t][x1 - lo_w, x2 - lo_w]
    pat = (Qp[..., 0] > 0) * 1 + (Qp[..., 1] > 0) * 2
    row, ex = {}, []

    # V1, V2 : combinations of actions
    row["n_states"] = m * m
    row["n_transship"] = int((W != 0).sum())
    row["V1"] = int(((W != 0) & (pat == 3)).sum())
    s1 = (W > 0) & (Qp[..., 0] > 0)
    s2 = (W < 0) & (Qp[..., 1] > 0)
    bad1 = s1 & ~((Qp[..., 0] > W) & (K >= R + v * W - TOL))
    bad2 = s2 & ~((Qp[..., 1] > -W) & (K >= R - v * W - TOL))
    row["n_sender_reorders"] = int(s1.sum() + s2.sum())
    row["n_receiver_reorders"] = int(((W > 0) & (Qp[..., 1] > 0)).sum() + ((W < 0) & (Qp[..., 0] > 0)).sum())
    row["V2"] = int(bad1.sum() + bad2.sum())

    # V3 : reorder-up-to levels on the post-transshipment states of the window
    Qx = sol.Q[t][w, w]
    patx = (Qx[..., 0] > 0) * 1 + (Qx[..., 1] > 0) * 2
    y1 = lev[:, None] + Qx[..., 0]
    y2 = lev[None, :] + Qx[..., 1]
    inside = (y1 <= sol.hi) & (y2 <= sol.hi)
    Hy = np.where(inside, Hw[np.clip(y1, lo, sol.hi) - lo, np.clip(y2, lo, sol.hi) - lo], np.nan)
    kS = np.unravel_index(int(np.argmin(Hw)), Hw.shape)
    Sbar = (int(lev[kS[0]]), int(lev[kS[1]]))
    below = (lev[:, None] < Sbar[0]) & (lev[None, :] < Sbar[1])
    both = patx == 3
    S1 = lev[np.argmin(Hw, axis=0)]                       # S^1(x2), smallest minimiser on the window
    S2 = lev[np.argmin(Hw, axis=1)]                       # S^2(x1)
    H1 = Hw.min(axis=0)[None, :] + 0 * Hw
    H2 = Hw.min(axis=1)[:, None] + 0 * Hw
    only1, only2 = patx == 1, patx == 2
    below1 = lev[:, None] < S1[None, :]
    below2 = lev[None, :] < S2[:, None]
    v3 = (both & below & ~(np.abs(Hy - Hw[kS]) <= TOL)) | (only1 & below1 & ~(np.abs(Hy - H1) <= TOL)) \
        | (only2 & below2 & ~(np.abs(Hy - H2) <= TOL))
    row.update(V3=int(v3.sum()), n_both=int(both.sum()), n_both_below=int((both & below).sum()),
               n_only=int((only1 | only2).sum()), n_only_below=int((only1 & below1).sum() + (only2 & below2).sum()),
               levels_outside_window=int((~inside & (patx > 0)).sum()), Sbar1=Sbar[0], Sbar2=Sbar[1])

    # V4 : transshipment levels on every line i1 + i2 = n
    v4a = v4b = v4c = 0
    n_plus = n_plus_beyond = n_minus = n_minus_beyond = 0
    for n in range(2 * lo, 2 * sol.hi + 1):
        a = np.arange(max(0, n - 2 * lo - (m - 1)), min(m - 1, n - 2 * lo) + 1)
        b = n - 2 * lo - a
        if len(a) < 2:
            continue
        i1 = lev[a]
        G = Cq[a, b]
        Wl = W[a, b]
        dom_p = np.flatnonzero(i1 >= 0)                  # the sender keeps a non-negative inventory
        dom_m = np.flatnonzero(i1 <= n)
        Dp = Dm = None
        if len(dom_p):
            phi_p = G - v * i1
            kp = dom_p[int(np.argmin(phi_p[dom_p]))]
            Dp = int(i1[kp])
            ship = np.flatnonzero(Wl > 0)
            n_plus += len(ship)
            beyond = ship[i1[ship] > Dp]
            n_plus_beyond += len(beyond)
            tgt = a[beyond] - Wl[beyond]                 # index of the post-transshipment i1 on the window axis
            got = Cq[tgt, n - 2 * lo - tgt] - v * lev[tgt]
            bad = np.abs(got - phi_p[kp]) > TOL
            v4a += int(bad.sum())
            for k in beyond[bad][:1]:
                ex.append(dict(check="V4a", t=t + 1, state=(int(i1[k]), int(n - i1[k])), W=int(Wl[k]), level=Dp))
            stay = np.flatnonzero((Wl == 0) & (i1 > Dp))
            lazy = stay[phi_p[stay] - phi_p[kp] > R + TOL]
            v4b += len(lazy)
        if len(dom_m):
            phi_m = G + v * i1
            km = dom_m[int(np.argmin(phi_m[dom_m]))]
            Dm = int(i1[km])
            ship = np.flatnonzero(Wl < 0)
            n_minus += len(ship)
            beyond = ship[i1[ship] < Dm]
            n_minus_beyond += len(beyond)
            tgt = a[beyond] - Wl[beyond]
            got = Cq[tgt, n - 2 * lo - tgt] + v * lev[tgt]
            bad = np.abs(got - phi_m[km]) > TOL
            v4a += int(bad.sum())
            for k in beyond[bad][:1]:
                ex.append(dict(check="V4a", t=t + 1, state=(int(i1[k]), int(n - i1[k])), W=int(Wl[k]), level=Dm))
            stay = np.flatnonzero((Wl == 0) & (i1 < Dm))
            lazy = stay[phi_m[stay] - phi_m[km] > R + TOL]
            v4b += len(lazy)
        if Dp is not None and Dm is not None and Dm > Dp:
            # largest minimiser of phi^- against smallest of phi^+ : compare the sets, a tie is not a violation
            tie_m = dom_m[np.abs(phi_m[dom_m] - phi_m[km]) <= TOL]
            tie_p = dom_p[np.abs(phi_p[dom_p] - phi_p[kp]) <= TOL]
            if i1[tie_m].min() > i1[tie_p].max():
                v4c += 1
                ex.append(dict(check="V4c", t=t + 1, n=int(n), D_minus=Dm, D_plus=Dp))
    row.update(V4a=v4a, V4b=v4b, V4c=v4c, n_ship_12=n_plus, n_ship_12_beyond=n_plus_beyond,
               n_ship_21=n_minus, n_ship_21_beyond=n_minus_beyond)

    # Q1 : threshold in the own inventory;   V5 : last period
    sec = test_second_stage(sol, t)
    q1 = 0
    for loc in (1, 2):
        vio = sec[f"loc{loc}"]["threshold_violations"]
        q1 += len(vio)
        for e in vio[:2]:
            ex.append(dict(check="Q1", t=t + 1, location=loc, **e))
    row["Q1"] = q1
    row["S_only_multivalued"] = int(sum(1 for loc in (1, 2) for s in sec[f"loc{loc}"]["S_only"] if len(s) > 1))
    for loc in (1, 2):
        thr = [x for x in sec[f"loc{loc}"]["threshold"] if x is not None]
        row[f"thr{loc}_min"], row[f"thr{loc}_max"] = (min(thr), max(thr)) if thr else ("", "")
        ups = sum(1 for p, q in zip(thr, thr[1:]) if q > p)
        row[f"thr{loc}_increases"] = ups
    if t == sol.T - 1:
        v5 = 0
        for loc in (1, 2):
            s = single[loc - 1]
            r = np.flatnonzero(s.Q[t] > 0)
            # the single-location policy may not reorder at any level of the range, for example when K is large
            ss, SS = (int(s.grid[r.max()]), int(s.grid[r.max()] + s.Q[t][r.max()])) if len(r) else (None, None)
            thr = set(x for x in sec[f"loc{loc}"]["threshold"])
            Sv = set(y for lst in sec[f"loc{loc}"]["S_only"] for y in lst)
            # no reorder at any state of the window agrees with the lemma when the threshold lies below the window
            below_window = thr == {None} and not Sv and (ss is None or ss < sol.lo)
            if not below_window and (thr != {ss} or (Sv and Sv != {SS})):
                v5 += 1
                ex.append(dict(check="V5", t=t + 1, location=loc, single=(ss, SS), thresholds=sorted(thr, key=str),
                               levels=sorted(Sv)))
        row["V5"] = v5
    else:
        row["V5"] = ""

    # Q2 : condition (Q) for g
    vals = second_stage_values(sol, t)[:, w, w]
    base = z * (lev[:, None] + lev[None, :])
    q2 = 0
    worst = 0.0
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
        bad = exc > TOL
        q2 += int(bad.any(axis=0).sum())
        worst = max(worst, float(exc.max()))
        if bad.any():
            k = np.unravel_index(int(np.argmax(exc)), exc.shape)
            ex.append(dict(check="Q2", t=t + 1, location=loc, own=int(lev[k[0]]), other=int(lev[k[1]]), excess=float(exc[k])))
    row["Q2"] = q2
    row["Q2_worst_excess"] = worst

    # descriptive
    first = test_first_stage(sol, t)
    row["lines"] = first["lines"]
    row["interval_violations"] = len(first["F1_violations"])
    row["max_targets"] = max(first["targets_max"].values())
    for name, F in (("Cq", Cq), ("C", sol.C[t][w, w])):
        row[f"kappa_{name}"] = max(kappa_lines(F)[0], kappa_lines(F.T)[0])
    return row, ex


def run(inst):
    t0 = time.perf_counter()
    sol = solve(inst, cache=False)
    single = [single_location(sol, 1), single_location(sol, 2)]
    rows, examples = [], []
    for t in range(sol.T):
        row, ex = check_period(sol, t, single)
        c = inst.costs
        head = dict(instance=inst.name, set=inst.set_name, relation=inst.relation, dist=inst.dist,
                    cv="" if inst.cv is None else inst.cv, T=inst.T, K=c["K"], R=c["R"], z=c["z"], v=c["v"], h=c["h"],
                    b=c["b"], lo=sol.lo, hi=sol.hi, lo_w=sol.lo_w, t=t + 1,
                    C1_00=float(sol.C[0][-sol.lo_w, -sol.lo_w]))
        head.update(row)
        rows.append(head)
        for e in ex:
            e["instance"] = inst.name
        examples += ex
    for r in rows:
        r["seconds"] = round(time.perf_counter() - t0, 1)
    return rows, examples


def _work(inst):
    try:
        return run(inst)
    except Exception as e:                                  # report and continue
        return [dict(instance=inst.name, error=repr(e))], []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--sets", nargs="+", default=["A", "C", "B"])
    a = ap.parse_args()
    todo = instances(a.sets)
    done = set()
    if os.path.exists(CSV):
        with open(CSV) as f:
            done = {r["instance"] for r in csv.DictReader(f)}
    todo = [i for i in todo if i.name not in done]
    todo.sort(key=lambda i: i.T)
    print(f"{len(todo)} instances to run, {len(done)} done", flush=True)
    examples = json.load(open(EXAMPLES)) if os.path.exists(EXAMPLES) else []
    fields = None
    if os.path.exists(CSV):
        with open(CSV) as f:
            fields = csv.DictReader(f).fieldnames
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for k, (rows, ex) in enumerate(pool.imap_unordered(_work, todo), 1):
            if "error" in rows[0]:
                print("ERROR", rows[0], flush=True)
                continue
            if fields is None:
                fields = list(rows[0].keys())
                with open(CSV, "w", newline="") as f:
                    csv.DictWriter(f, fieldnames=fields).writeheader()
            with open(CSV, "a", newline="") as f:
                wr = csv.DictWriter(f, fieldnames=fields)
                for r in rows:
                    wr.writerow(r)
            examples += ex
            with open(EXAMPLES, "w") as f:
                json.dump(examples, f, indent=1)
            print(f"{k}/{len(todo)} {rows[0]['instance']} {rows[0]['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
