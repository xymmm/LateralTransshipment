"""
test2_experiments.py : Algorithm 1 against the exact SDP-2, 100 instances, one call of algorithm() per instance.

Questions
    Q1  How far is the expected total cost of Algorithm 1 from the optimum computed by SDP-2?
    Q2  How do the computing times compare, for each demand distribution?
          SDP-2: the whole backward induction, which must be completed before the first decision.
          Algorithm 1: one decision, i.e. one MILP-1 solve and one refinement, made online at each period.
    Q3  From which problem size on is SDP-2 no longer practical while Algorithm 1 remains so?
        The size of SDP-2 is set by the number of inventory levels N per location (time grows with T N^4),
        and N follows the demand level, so the demand level is scaled at a fixed horizon.

Blocks (100 instances; C is optional and runs only when named)
    A  60 instances: T = 8, demand level 7, 3 distributions (Poisson; normal with cv 0.1 and 0.3)
       x 20 pattern pairs (10 identical, 10 different, as in test1).                           -> Q1, Q2
    B  40 instances: T = 8, Poisson and normal cv 0.3, the means of A multiplied by s in {2, 3, 4, 5}
       x 5 identical pattern pairs (LCY1, LCY2, SIN1, STAT, EMP1).                              -> Q3, and Q1 at larger sizes
    C  timing only, no simulation of the expected cost: Poisson, s in {6, 8, 10, 12}, LCY2 and SIN1.
       SDP-2 is solved once on the grid [-D, 2D] (D the largest period demand), and Algorithm 1 makes its decisions
       along 10 demand paths. This extends Q3 to sizes at which simulating the expected cost is too long.
    Costs K = 20, z = 1, R = 10, v = 0.5, h = 1, b = 5 (test1 default); opening inventory (0, 0).
    Demand means are the fixed integers of test1_experiments.DEMAND_PATTERNS for (M, T) = (7, 8); blocks B and C use
    integer multiples of them, so all means stay integer.
    T = 8 because the time to solve MILP-1 grows quickly with the length of the residual horizon (about 0.5 s at
    8 periods and 5 s at 12 periods on one thread), and the simulation solves it at thousands of states per instance.

For each instance
    1. grid: the adequate_grid rule of test1 (no optimal order leaves the grid, and widening the grid by 25% changes
       C_1(0,0) by less than 1e-6 in relative terms)
    2. SDP-2 on that grid: time and C_1(0,0)
    3. algorithm(): ETC of Algorithm 1 by simulation until the 95% half-width is at most 0.1% of the estimate
    4. the SDP-2 optimal policy simulated on the same demand paths (common random numbers): a check of the
       simulation, and a paired estimate of the gap with a much narrower confidence interval
    gap = 100 (ETC - C_1(0,0)) / C_1(0,0)

Usage
    set RUN_BLOCK and RUN_QUICK below and click Run, or: python test2_experiments.py --block A
Results are appended to results_test2/<block>.csv, one row per instance; instances already in the csv are skipped,
so an interrupted run can be resumed.
"""
from __future__ import annotations

import os

# one thread for numpy and for Gurobi, so that the two methods are timed on the same footing
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

# ====================================================================================================
# settings for clicking "Run" in PyCharm; command-line arguments override them
RUN_BLOCK = "all"        # "A", "B", "C" or "all" (= A and B)
RUN_QUICK = False        # True: 2 instances per block, to check that everything runs
RUN_NO_REFINE = False    # True: also simulate Algorithm 1 without the refinement step on the same paths
# ====================================================================================================

import argparse
import csv
import itertools
import time

import numpy as np

from algorithm import algorithm, gurobi_env, period_pmfs
from lt_sdp import SDP2
from policies import OptimalPolicy, RecedingHorizonPolicy
from simulate import demand_batch, half_width, run_paths, simulate
from test1_experiments import COSTS_DEFAULT, adequate_grid, max_demand, pattern, pattern_pairs

OUT_DIR = "results_test2"
T_MAIN, M_MAIN = 8, 7
DISTRIBUTIONS = [("poisson", None), ("normal", 0.1), ("normal", 0.3)]
SCALES = [2, 3, 4, 5]
B_PATTERNS = ["LCY1", "LCY2", "SIN1", "STAT", "EMP1"]
C_SCALES = [6, 8, 10, 12]
C_PATTERNS = ["LCY2", "SIN1"]
C_PATHS = 10
SEED = 1234


def instances(block, quick=False):
    if block == "A":
        prs = pattern_pairs()
        rows = [dict(dist=d, cv=c, scale=1, pat1=p1, pat2=p2) for (d, c), (p1, p2) in itertools.product(DISTRIBUTIONS, prs)]
    elif block == "B":
        rows = [dict(dist=d, cv=c, scale=s, pat1=p, pat2=p)
                for (d, c), s, p in itertools.product([("poisson", None), ("normal", 0.3)], SCALES, B_PATTERNS)]
    elif block == "C":
        rows = [dict(dist="poisson", cv=None, scale=sc, pat1=p, pat2=p) for sc, p in itertools.product(C_SCALES, C_PATTERNS)]
    else:
        raise ValueError(block)
    for r in (rows[:1] + rows[-1:] if quick else rows):
        yield dict(block=block, M=M_MAIN, T=T_MAIN, **r, **COSTS_DEFAULT)


def instance_id(inst):
    return "|".join(str(inst[k]) for k in ("block", "dist", "cv", "M", "T", "scale", "pat1", "pat2",
                                          "K", "z", "R", "v", "h", "b"))


FIELDS = ["id", "block", "dist", "cv", "M", "T", "scale", "pat1", "pat2", "mu1", "mu2", "K", "z", "R", "v", "h", "b",
          "lo", "hi", "N", "seconds_grid_search", "seconds_sdp2", "C00_sdp2", "action_sdp2",
          "etc_alg", "hw_alg", "paths_alg", "attained_alg", "gap_pct", "hw_pct",
          "first_decision_alg", "seconds_alg_total", "seconds_alg_milp", "seconds_alg_refine", "seconds_alg_build",
          "decisions_alg", "seconds_per_decision", "seconds_per_decision_k1",
          "etc_opt_same_paths", "paired_gap_pct", "paired_hw_pct", "opt_paths_outside_grid",
          "etc_noref", "gap_noref_pct", "paired_alg_minus_noref_pct", "paired_hw_alg_minus_noref_pct"]


def done_ids(path):
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {r["id"] for r in csv.DictReader(f)}


def append(path, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})


def paired(policy, pm1, pm2, costs, n, base_costs):
    """Simulate `policy` on the first n paths of SEED; return its mean and the paired difference to base_costs."""
    sim = simulate(policy, pm1, pm2, (0, 0), **costs, seed=SEED, n_paths=n)
    diff = base_costs - sim.costs
    return sim.mean, float(diff.mean()), half_width(diff)


def run_instance(inst, no_refine, env):
    costs = {k: inst[k] for k in ("K", "z", "R", "v", "h", "b")}
    mu1 = [inst["scale"] * x for x in pattern(inst["pat1"], inst["T"], inst["M"])]
    mu2 = [inst["scale"] * x for x in pattern(inst["pat2"], inst["T"], inst["M"])]
    dist, cv = inst["dist"], inst["cv"]
    row = dict(id=instance_id(inst), **inst, mu1=" ".join(map(str, mu1)), mu2=" ".join(map(str, mu2)))

    t0 = time.perf_counter()
    lo, hi = adequate_grid(mu1, mu2, dist, cv, costs)
    row.update(lo=lo, hi=hi, N=hi - lo + 1, seconds_grid_search=time.perf_counter() - t0)
    s = SDP2(mu1, mu2, lo, hi, distribution=dist, cv=cv, **costs)
    C = s.value()
    row.update(seconds_sdp2=s.seconds, C00_sdp2=C, action_sdp2=str(s.action()))

    r = algorithm(mu1, mu2, **costs, distribution=dist, cv=cv, seed=SEED, env=env)
    row.update(etc_alg=r.etc, hw_alg=r.half_width, paths_alg=r.n_paths, attained_alg=r.precision_attained,
               gap_pct=100 * (r.etc - C) / C, hw_pct=100 * r.half_width / C,
               first_decision_alg=str(r.first_decision), seconds_alg_total=r.seconds_total,
               seconds_alg_milp=r.seconds_milp, seconds_alg_refine=r.seconds_refine, seconds_alg_build=r.seconds_build,
               decisions_alg=r.n_decisions, seconds_per_decision=r.seconds_per_decision,
               seconds_per_decision_k1=r.seconds_per_decision_k1)

    for mdl in r.policy._milp.values():                     # free the Gurobi models of this instance
        mdl.model.dispose()

    pm1, pm2 = period_pmfs(mu1, dist, cv), period_pmfs(mu2, dist, cv)
    try:
        m, d, hw = paired(OptimalPolicy(s), pm1, pm2, costs, r.n_paths, r.sim.costs)
        row.update(etc_opt_same_paths=m, paired_gap_pct=100 * d / C, paired_hw_pct=100 * hw / C, opt_paths_outside_grid=0)
    except IndexError:
        row.update(opt_paths_outside_grid=1)

    if no_refine:
        pol = RecedingHorizonPolicy(pm1, pm2, **costs, distribution=dist, refine=False, env=env)
        m, d, hw = paired(pol, pm1, pm2, costs, r.n_paths, r.sim.costs)
        row.update(etc_noref=m, gap_noref_pct=100 * (m - C) / C,
                   paired_alg_minus_noref_pct=100 * d / C, paired_hw_alg_minus_noref_pct=100 * hw / C)
        for mdl in pol._milp.values():
            mdl.model.dispose()
    return row


def run_timing(inst, env):
    """Block C: SDP-2 time on the grid [-D, 2D] and the time of the decisions of Algorithm 1 along C_PATHS paths."""
    costs = {k: inst[k] for k in ("K", "z", "R", "v", "h", "b")}
    mu1 = [inst["scale"] * x for x in pattern(inst["pat1"], inst["T"], inst["M"])]
    mu2 = [inst["scale"] * x for x in pattern(inst["pat2"], inst["T"], inst["M"])]
    dist, cv = inst["dist"], inst["cv"]
    row = dict(id=instance_id(inst), **inst, mu1=" ".join(map(str, mu1)), mu2=" ".join(map(str, mu2)))
    D = max_demand(mu1, mu2, dist, cv)
    lo, hi = -D, 2 * D
    s = SDP2(mu1, mu2, lo, hi, distribution=dist, cv=cv, **costs)
    row.update(lo=lo, hi=hi, N=hi - lo + 1, seconds_sdp2=s.seconds, C00_sdp2=s.value(), action_sdp2=str(s.action()))
    del s
    pm1, pm2 = period_pmfs(mu1, dist, cv), period_pmfs(mu2, dist, cv)
    pol = RecedingHorizonPolicy(pm1, pm2, **costs, distribution=dist, env=env)
    D1, D2 = demand_batch(pm1, pm2, 1000, SEED, 0)
    run_paths(pol, D1[:C_PATHS], D2[:C_PATHS], (0, 0), **costs)
    dt = np.array([x for _, x in pol.decision_times])
    row.update(first_decision_alg=str(pol.cache[(1, 0, 0)]), decisions_alg=len(dt),
               seconds_per_decision=float(dt.mean()),
               seconds_per_decision_k1=float(np.mean([x for k, x in pol.decision_times if k == 1])))
    for mdl in pol._milp.values():
        mdl.model.dispose()
    return row


def run_block(block, quick, no_refine, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{block}.csv")
    done = done_ids(path)
    insts = [i for i in instances(block, quick) if instance_id(i) not in done]
    env = gurobi_env()
    print(f"block {block}: {len(insts)} instances to run, results in {path}")
    for n, inst in enumerate(insts, 1):
        t0 = time.perf_counter()
        if block == "C":
            row = run_timing(inst, env)
            append(path, row)
            print(f"[C {n}/{len(insts)}] s={inst['scale']} {inst['pat1']}  N={row['N']}  SDP-2 {row['seconds_sdp2']:.1f}s  "
                  f"decision {row['seconds_per_decision']:.3f}s, at k=1 {row['seconds_per_decision_k1']:.3f}s", flush=True)
            continue
        row = run_instance(inst, no_refine, env)
        append(path, row)
        print(f"[{block} {n}/{len(insts)}] {inst['dist']} cv={inst['cv']} s={inst['scale']} {inst['pat1']}-{inst['pat2']}  "
              f"N={row['N']}  SDP-2 {row['seconds_sdp2']:.1f}s  C={row['C00_sdp2']:.2f}  "
              f"ETC={row['etc_alg']:.2f}  gap={row['gap_pct']:.3f}% (+-{row['hw_pct']:.3f})  "
              f"paired={row.get('paired_gap_pct', float('nan')):.3f}%  "
              f"decision {1000 * row['seconds_per_decision']:.1f}ms  instance {time.perf_counter() - t0:.0f}s", flush=True)
    env.dispose()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", default=RUN_BLOCK)
    ap.add_argument("--quick", action="store_true", default=RUN_QUICK)
    ap.add_argument("--no-refine", action="store_true", default=RUN_NO_REFINE)
    ap.add_argument("--out", default=OUT_DIR)
    a = ap.parse_args()
    out = a.out + ("_quick" if a.quick else "")
    for blk in (["A", "B"] if a.block == "all" else [a.block]):
        run_block(blk, a.quick, a.no_refine, out)


if __name__ == "__main__":
    main()
