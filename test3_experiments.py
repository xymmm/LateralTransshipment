"""
test3_experiments.py : Algorithm 1 against the quasi-myopic benchmark of Glazebrook et al. (2015), and the role of the
coupling between the two locations.

Benchmarks, all evaluated against the exact SDP-2 value C_1(0,0)
    QM  quasi-myopic: one policy improvement step from the optimal policy without transshipment
        (policies.QuasiMyopicPolicy, single-location values from sdp_single.SDP_single)
    NT  no transshipment: each location follows its optimal single-location policy and never transships; its expected
        cost is exact, V^1_1(0) + V^2_1(0), so it needs no simulation. NT is not a competitor; NT - SDP-2 measures the
        value of transshipment, and QM - SDP-2 the value of planning orders and transshipments jointly.
QM is simulated on the same demand paths as Algorithm 1 (seed 1234, the same number of paths), so differences between
the two are differences over identical paths.

Blocks
    A  QM and NT for the 1440 instances of test2 block A; reads results_test2/A.csv, no MILP is solved
    B  QM and NT for the 40 instances of test2 block B; reads results_test2/B.csv
    D  coupling test bed, 432 new instances, full factorial, every instance reported:
           K in {20, 30, 50} x R in {2, 5, 10, 20}
           demand relation between the locations (6):
               identical      STAT/STAT, LCY1/LCY1
               opposite       LCY1/LCY2, SIN1/SIN1 reversed, LCY2/LCY2 reversed
               unequal level  STAT/STAT at one third of the level
           distribution Poisson, normal cv 0.1, normal cv 0.3;  T in {4, 8}
           z = 1, v = 0.5, h = 1, b = 5, opening inventory (0, 0)
       Each instance runs SDP-2 on the adequate grid of test1, Algorithm 1, the SDP-2 policy and QM on the same paths,
       and NT.
Demand means come from the fixed integer table test1_experiments.DEMAND_PATTERNS at demand level 7; a reversed pattern
lists the same means in reverse order, and the lower level is max(1, round(mean / 3)), so all means are integers.

Usage
    set the options below and click Run, or: python test3_experiments.py --block D --workers 8
Results are appended to results_test3/<block>.csv; instances already present are skipped, so a run can be resumed.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

# ====================================================================================================
# settings for clicking "Run" in PyCharm; command-line arguments override them
RUN_BLOCK = "Aplus"      # "A", "Aplus", "B", "D", or "all" (= A, Aplus, B, D)
RUN_QUICK = False        # True: a few instances per block, to check that everything runs
N_WORKERS = 8            # parallel processes; this test compares costs, not computing times
# ====================================================================================================

import argparse
import csv
import itertools
import multiprocessing as mp
import time

import numpy as np
import pandas as pd

from algorithm import algorithm, gurobi_env, period_pmfs
from lt_sdp import SDP2
from policies import OptimalPolicy, QuasiMyopicPolicy
from sdp_single import SDP_single
from simulate import half_width, simulate
from test1_experiments import adequate_grid, pattern

OUT_DIR = "results_test3"
TEST2_DIR = "results_test2"
SEED = 1234
M_LEVEL = 7

D_K = [20, 30, 50]
D_R = [2, 5, 10, 20]
D_DISTRIBUTIONS = [("poisson", None), ("normal", 0.1), ("normal", 0.3)]
D_HORIZONS = [4, 8]
D_COSTS = dict(z=1, v=0.5, h=1, b=5)
D_RELATIONS = ["identical STAT/STAT", "identical LCY1/LCY1", "opposite LCY1/LCY2", "opposite SIN1/SIN1rev",
               "opposite LCY2/LCY2rev", "unequal STAT/STATlow"]


def relation_means(name, T):
    P = lambda p: pattern(p, T, M_LEVEL)
    low = lambda m: [max(1, int(np.floor(x / 3 + 0.5))) for x in m]
    table = {"identical STAT/STAT": (P("STAT"), P("STAT")),
             "identical LCY1/LCY1": (P("LCY1"), P("LCY1")),
             "opposite LCY1/LCY2": (P("LCY1"), P("LCY2")),
             "opposite SIN1/SIN1rev": (P("SIN1"), P("SIN1")[::-1]),
             "opposite LCY2/LCY2rev": (P("LCY2"), P("LCY2")[::-1]),
             "unequal STAT/STATlow": (P("STAT"), low(P("STAT")))}
    return table[name]


def d_instances(quick=False):
    rows = [dict(block="D", T=T, dist=d, cv=c, relation=rel, K=K, R=R, **D_COSTS)
            for T, (d, c), rel, K, R in itertools.product(D_HORIZONS, D_DISTRIBUTIONS, D_RELATIONS, D_K, D_R)]
    return rows[:2] + [r for r in rows if r["T"] == 8][:1] if quick else rows


def d_id(inst):
    return "|".join(str(inst[k]) for k in ("block", "dist", "cv", "T", "relation", "K", "z", "R", "v", "h", "b"))


# -- the two benchmarks ------------------------------------------------------------------------------------
def benchmarks(mu1, mu2, dist, cv, costs, lo, hi, C, n_paths, alg_costs=None, etc_alg=None):
    """QM simulated on the first n_paths paths of SEED, and the exact NT value, both at opening inventory (0, 0)."""
    t0 = time.perf_counter()
    s1 = SDP_single(mu1, lo, hi, costs["K"], costs["z"], costs["h"], costs["b"], distribution=dist, cv=cv)
    s2 = SDP_single(mu2, lo, hi, costs["K"], costs["z"], costs["h"], costs["b"], distribution=dist, cv=cv)
    t_single = time.perf_counter() - t0
    nt = s1.value(0) + s2.value(0)
    row = dict(etc_nt=nt, gap_nt=100 * (nt - C) / C, seconds_single_sdp=t_single)
    pm1, pm2 = period_pmfs(mu1, dist, cv), period_pmfs(mu2, dist, cv)
    pol = QuasiMyopicPolicy(s1, s2, costs["R"], costs["v"])
    try:
        sim = simulate(pol, pm1, pm2, (0, 0), **costs, seed=SEED, n_paths=int(n_paths))
    except IndexError:
        row.update(qm_outside_grid=1)
        return row
    row.update(etc_qm=sim.mean, gap_qm=100 * (sim.mean - C) / C, first_decision_qm=str(pol.decide(1, 0, 0)),
               qm_outside_grid=0, seconds_qm_sim=sim.seconds)
    if alg_costs is not None:                                # paired difference over identical paths, with its CI
        diff = alg_costs - sim.costs
        row.update(alg_minus_qm_pct=100 * float(diff.mean()) / C, alg_minus_qm_hw_pct=100 * half_width(diff) / C)
    elif etc_alg is not None:                                # same paths, means only (path costs of test2 not stored)
        row.update(alg_minus_qm_pct=100 * (etc_alg - sim.mean) / C)
    return row


# -- blocks A and B: benchmarks on the instances of test2 -------------------------------------------------
def run_existing(r):
    cv = None if pd.isna(r["cv"]) else float(r["cv"])
    mu1 = [int(x) for x in str(r["mu1"]).split()]
    mu2 = [int(x) for x in str(r["mu2"]).split()]
    costs = {k: float(r[k]) for k in ("K", "z", "R", "v", "h", "b")}
    row = dict(id=r["id"], gap_alg=r["gap_pct"], paired_gap_alg=r["paired_gap_pct"])
    row.update(benchmarks(mu1, mu2, r["dist"], cv, costs, int(r["lo"]), int(r["hi"]), float(r["C00_sdp2"]),
                          int(r["paths_alg"]), etc_alg=float(r["etc_alg"])))
    if "etc_qm" in row and not pd.isna(r["etc_opt_same_paths"]):
        row["paired_gap_qm"] = 100 * (row["etc_qm"] - float(r["etc_opt_same_paths"])) / float(r["C00_sdp2"])
    return row


# -- block D: the coupling test bed -----------------------------------------------------------------------
def run_d(inst, env):
    costs = {k: inst[k] for k in ("K", "z", "R", "v", "h", "b")}
    mu1, mu2 = relation_means(inst["relation"], inst["T"])
    dist, cv = inst["dist"], inst["cv"]
    row = dict(id=d_id(inst), **inst, mu1=" ".join(map(str, mu1)), mu2=" ".join(map(str, mu2)))
    lo, hi = adequate_grid(mu1, mu2, dist, cv, costs)
    s = SDP2(mu1, mu2, lo, hi, distribution=dist, cv=cv, **costs)
    C = s.value()
    row.update(lo=lo, hi=hi, N=hi - lo + 1, seconds_sdp2=s.seconds, C00_sdp2=C, action_sdp2=str(s.action()))

    r = algorithm(mu1, mu2, **costs, distribution=dist, cv=cv, seed=SEED, env=env)
    row.update(etc_alg=r.etc, hw_alg=r.half_width, paths_alg=r.n_paths, attained_alg=r.precision_attained,
               gap_alg=100 * (r.etc - C) / C, hw_alg_pct=100 * r.half_width / C, first_decision_alg=str(r.first_decision),
               seconds_alg_total=r.seconds_total, decisions_alg=r.n_decisions,
               seconds_per_decision=r.seconds_per_decision, seconds_per_decision_k1=r.seconds_per_decision_k1)
    for mdl in r.policy._milp.values():
        mdl.model.dispose()

    pm1, pm2 = period_pmfs(mu1, dist, cv), period_pmfs(mu2, dist, cv)
    try:
        opt = simulate(OptimalPolicy(s), pm1, pm2, (0, 0), **costs, seed=SEED, n_paths=r.n_paths)
        row.update(etc_opt_same_paths=opt.mean, paired_gap_alg=100 * float((r.sim.costs - opt.costs).mean()) / C,
                   opt_outside_grid=0)
    except IndexError:
        opt = None
        row.update(opt_outside_grid=1)

    b = benchmarks(mu1, mu2, dist, cv, costs, lo, hi, C, r.n_paths, alg_costs=r.sim.costs)
    row.update(b)
    if opt is not None and "etc_qm" in b:
        row["paired_gap_qm"] = 100 * (b["etc_qm"] - opt.mean) / C
    return row


# -- driver -----------------------------------------------------------------------------------------------
FIELDS_EXISTING = ["id", "gap_alg", "paired_gap_alg", "etc_nt", "gap_nt", "etc_qm", "gap_qm", "paired_gap_qm",
                   "alg_minus_qm_pct", "first_decision_qm", "qm_outside_grid", "seconds_single_sdp", "seconds_qm_sim"]
FIELDS_D = ["id", "block", "T", "dist", "cv", "relation", "K", "z", "R", "v", "h", "b", "mu1", "mu2",
            "lo", "hi", "N", "seconds_sdp2", "C00_sdp2", "action_sdp2",
            "etc_alg", "hw_alg", "paths_alg", "attained_alg", "gap_alg", "hw_alg_pct", "first_decision_alg",
            "seconds_alg_total", "decisions_alg", "seconds_per_decision", "seconds_per_decision_k1",
            "etc_opt_same_paths", "paired_gap_alg", "opt_outside_grid",
            "etc_nt", "gap_nt", "etc_qm", "gap_qm", "paired_gap_qm", "alg_minus_qm_pct", "alg_minus_qm_hw_pct",
            "first_decision_qm", "qm_outside_grid", "seconds_single_sdp", "seconds_qm_sim"]

_ENV = None


def _init_worker():
    global _ENV
    _ENV = gurobi_env()


def _work(args):
    block, item = args
    try:
        return (run_d(item, _ENV) if block == "D" else run_existing(item)), None
    except Exception as e:
        key = d_id(item) if block == "D" else item["id"]
        return None, f"{key}: {type(e).__name__}: {e}"


def run_block(block, quick, out_dir, workers):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{block}.csv")
    fields = FIELDS_D if block == "D" else FIELDS_EXISTING
    done = set(pd.read_csv(path)["id"]) if os.path.exists(path) else set()
    if block == "D":
        items = [i for i in d_instances(quick) if d_id(i) not in done]
    else:
        src = os.path.join(TEST2_DIR, f"{block}.csv")
        if not os.path.exists(src):
            print(f"block {block}: {src} not found, run test2 block {block} first")
            return
        d = pd.read_csv(src)
        items = [r.to_dict() for _, r in d.iterrows() if r["id"] not in done]
        if quick:
            items = items[:3]
    print(f"block {block}: {len(items)} instances to run, {workers} process(es), results in {path}", flush=True)
    if not items:
        return
    t0 = time.perf_counter()
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f, mp.get_context("spawn").Pool(workers, initializer=_init_worker) as pool:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        for n, (row, err) in enumerate(pool.imap_unordered(_work, [(block, i) for i in items]), 1):
            if err is not None:
                print(f"[{block} {n}/{len(items)}] FAILED {err}", flush=True)
                continue
            w.writerow({k: row.get(k, "") for k in fields})
            f.flush()
            if block == "D":
                print(f"[D {n}/{len(items)}] T={row['T']} {row['dist']} cv={row['cv']} {row['relation']} K={row['K']} "
                      f"R={row['R']}  gap: Alg {row['gap_alg']:.3f}%  QM {row.get('gap_qm', float('nan')):.3f}%  "
                      f"NT {row['gap_nt']:.3f}%  elapsed {(time.perf_counter() - t0) / 3600:.1f}h", flush=True)
            elif n % 50 == 0 or n == len(items):
                print(f"[{block} {n}/{len(items)}] elapsed {(time.perf_counter() - t0) / 60:.1f} min", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", default=RUN_BLOCK)
    ap.add_argument("--quick", action="store_true", default=RUN_QUICK)
    ap.add_argument("--workers", type=int, default=N_WORKERS)
    ap.add_argument("--out", default=OUT_DIR)
    a = ap.parse_args()
    out = a.out + ("_quick" if a.quick else "")
    for blk in (["A", "Aplus", "B", "D"] if a.block == "all" else [a.block]):
        run_block(blk, a.quick, out, a.workers)


if __name__ == "__main__":
    main()
