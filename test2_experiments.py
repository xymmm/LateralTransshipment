"""
test2_experiments.py : Algorithm 1 against the exact SDP-2. One call of algorithm() and one call of SDP2() per instance.

Questions
    Q1  How far is the expected total cost of Algorithm 1 from the optimum computed by SDP-2, and how does the gap
        depend on the horizon, the demand distribution, the demand pattern and the cost parameters?
    Q2  How do the computing times compare?
          SDP-2: the whole backward induction, which must be completed before the first decision.
          Algorithm 1: one decision, i.e. one MILP-1 solve and one refinement, made online at each period.
    Q3  From which problem size on is SDP-2 no longer practical while Algorithm 1 remains so? The size of SDP-2 is
        set by the number N of inventory levels per location (time grows with T N^4) and N follows the demand level,
        so the demand level is scaled at a fixed horizon.

Block A : main test bed, 1440 instances                                                    -> Q1, Q2
    horizon T               4, 8
    demand distribution     Poisson; normal with cv 0.1, 0.2, 0.3                (4)
    demand patterns         10 patterns at demand level 7, 20 pairs: 10 identical, 10 different as in test1
    cost parameters         K in {10, 20, 30}, R in {5, 10, 20}, b in {3, 5}, (z, v) in {(2, 1), (1, 0.5)}, h = 1,
                            the 36 settings of the 2022 version and of test1 block D
    The full factorial would have 2 x 4 x 20 x 36 = 5760 instances. Every (T, distribution, pattern pair) is run with
    all 9 combinations of K and R and with one of the 4 combinations of b and (z, v); that combination rotates with
    the pattern pair and the distribution, (pair index + distribution index) mod 4. Hence each (b, (z, v)) is used
    by 5 pattern pairs in every distribution and by 1 distribution for every pattern pair, and every level of every
    parameter has the same number of instances, half of them with T = 4 and half with T = 8:
        T: 720 each;  distribution: 360;  pattern of location 1: 144;  K, R: 480;  b, (z, v): 720;
        each of the 36 cost settings: 40 instances.
Block Aplus : horizon T = 12, 180 instances, results in results_test2/Aplus.csv            -> Q1 at a longer horizon
    The design of block A with T = 12 on 5 of its 20 pattern pairs: LCY1-LCY1, SIN1-SIN1, STAT-STAT, LCY1-LCY2,
    EMP1-EMP2; 4 distributions x 5 pairs x 9 (K, R), with (b, (z, v)) rotated exactly as in block A. Each of these
    instances has the same distribution, pair and costs as one instance of block A at T = 4 and one at T = 8, so the
    three horizons can be compared on identical settings. Block A and its csv are not changed. About 20-30 minutes
    per instance in one process, about 10 hours with 8 processes.
Block B : demand level, 40 instances                                                       -> Q3, and Q1 at larger sizes
    T = 8, Poisson and normal cv 0.3, the means multiplied by s in {2, 3, 4, 5}, 5 identical pairs
    (LCY1, LCY2, SIN1, STAT, EMP1), costs K = 20, z = 1, R = 10, v = 0.5, h = 1, b = 5.
Block C : computing time, 80 instances, full factorial                                      -> Q3
    distribution Poisson and normal cv 0.3; state space at 5 levels, with the demand means multiplied by
    s in {1, 2, 4, 6, 8} for Poisson and s in {1, 2, 3, 4, 5} for the normal, so that both distributions give about
    the same sizes, |I| = (number of inventory levels per location)^2 from about 85^2 to about 430^2;
    T in {4, 8, 12, 16}; patterns LCY2 and SIN1 at both locations; costs as in B; grid [-D, 2D] (D the largest
    period demand). Each instance records the complete SDP-2 solve, which must finish before the first decision,
    and the decisions of Algorithm 1 along 10 demand paths, the decision of period 1 at (0, 0) separately, since it
    looks over the whole horizon and is the slowest. Wall time and CPU time of the process are both recorded.
    No simulation of the expected cost. Instances can run in C_WORKERS parallel processes: SDP-2 and Algorithm 1
    of one instance always run in the same process one after the other; the CPU times are the cleaner measure when
    several processes share the machine. About 11 hours in one process. The earlier 12-instance version of block C
    is kept in results_test2/C_old_design.csv.
Demand means are the fixed integers of test1_experiments.DEMAND_PATTERNS for (M, T) = (7, 4) and (7, 8); blocks B
and C use integer multiples of them, and blocks Aplus and C also use (7, 12) and (7, 16). Opening inventory (0, 0)
throughout. Blocks with simulation stop at T = 12 because the time to solve MILP-1 grows quickly with the residual
horizon (one T = 12 instance takes about 20 minutes).

For each instance of blocks A and B
    1. grid: the adequate_grid rule of test1 (no optimal order leaves the grid, and widening the grid by 25% changes
       C_1(0,0) by less than 1e-6 in relative terms)
    2. SDP-2 on that grid: time and C_1(0,0)
    3. algorithm(): ETC of Algorithm 1 by simulation until the 95% half-width is at most 0.1% of the estimate
    4. the SDP-2 optimal policy simulated on the same demand paths (common random numbers): a check of the
       simulation, and a paired estimate of the gap with a much narrower confidence interval
    gap = 100 (ETC - C_1(0,0)) / C_1(0,0)

Running time on an M4 Pro, one process: about 10-20 s for an instance with T = 4 and 2-5 minutes with T = 8, so
block A takes about 40 hours in one process. Block A is therefore run by N_WORKERS processes in parallel, each on one
thread; blocks B and C measure computing time against the problem size and always run in one process.

Usage
    set the options below and click Run, or: python test2_experiments.py --block A --workers 8
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
RUN_BLOCK = "Aplus"      # "A", "Aplus", "B", "C", or "all" (= A, then B)
RUN_QUICK = False        # True: a few instances per block, to check that everything runs
RUN_NO_REFINE = False    # True: also simulate Algorithm 1 without the refinement step on the same paths
N_WORKERS = 8            # parallel processes for blocks A and Aplus (the M4 Pro has 12 cores); B always uses 1
C_WORKERS = 4            # parallel processes for block C (1 = cleanest timing, about 11 hours)
# ====================================================================================================

import argparse
import csv
import itertools
import multiprocessing as mp
import time

import numpy as np

from algorithm import algorithm, gurobi_env, period_pmfs
from lt_sdp import SDP2
from policies import OptimalPolicy, RecedingHorizonPolicy
from simulate import demand_batch, half_width, run_paths, simulate
from test1_experiments import COSTS_DEFAULT, adequate_grid, max_demand, pattern, pattern_pairs

OUT_DIR = "results_test2"
M_LEVEL = 7
SEED = 1234

# block A
A_HORIZONS = [4, 8]
A_DISTRIBUTIONS = [("poisson", None), ("normal", 0.1), ("normal", 0.2), ("normal", 0.3)]
A_KR = list(itertools.product([10, 20, 30], [5, 10, 20]))                 # (K, R), all 9 with every instance group
A_BZV = list(itertools.product([3, 5], [(2, 1), (1, 0.5)]))               # (b, (z, v)), one of 4, rotated
A_H = 1
# block Aplus: T = 12, the design of A on 5 of its pattern pairs (indices into test1 pattern_pairs())
AP_T = 12
AP_PAIRS = [0, 2, 4, 10, 16]                                              # LCY1-LCY1, SIN1-SIN1, STAT-STAT, LCY1-LCY2, EMP1-EMP2
# block B
B_T = 8
B_DISTRIBUTIONS = [("poisson", None), ("normal", 0.3)]
B_SCALES = [2, 3, 4, 5]
B_PATTERNS = ["LCY1", "LCY2", "SIN1", "STAT", "EMP1"]
# block C
C_DISTRIBUTIONS = [("poisson", None), ("normal", 0.3)]
C_SCALES = {"poisson": [1, 2, 4, 6, 8], "normal": [1, 2, 3, 4, 5]}
C_HORIZONS = [4, 8, 12, 16]
C_PATTERNS = ["LCY2", "SIN1"]
C_PATHS = 10


def instances(block, quick=False):
    rows = []
    if block == "A":
        prs = pattern_pairs()
        for T in A_HORIZONS:
            for (di, (d, c)), (pi, (p1, p2)) in itertools.product(enumerate(A_DISTRIBUTIONS), enumerate(prs)):
                b, (z, v) = A_BZV[(pi + di) % len(A_BZV)]
                for K, R in A_KR:
                    rows.append(dict(T=T, dist=d, cv=c, scale=1, pat1=p1, pat2=p2, K=K, z=z, R=R, v=v, h=A_H, b=b))
        if quick:
            rows = rows[:3] + [r for r in rows if r["T"] == 8][:1]
    elif block == "Aplus":
        prs = pattern_pairs()
        for (di, (d, c)), pi in itertools.product(enumerate(A_DISTRIBUTIONS), AP_PAIRS):
            p1, p2 = prs[pi]
            b, (z, v) = A_BZV[(pi + di) % len(A_BZV)]                       # same rotation as block A
            for K, R in A_KR:
                rows.append(dict(T=AP_T, dist=d, cv=c, scale=1, pat1=p1, pat2=p2, K=K, z=z, R=R, v=v, h=A_H, b=b))
        if quick:
            rows = rows[:2]
    elif block == "B":
        rows = [dict(T=B_T, dist=d, cv=c, scale=s, pat1=p, pat2=p, **COSTS_DEFAULT)
                for (d, c), s, p in itertools.product(B_DISTRIBUTIONS, B_SCALES, B_PATTERNS)]
        if quick:
            rows = rows[:1]
    elif block == "C":
        rows = [dict(T=T, dist=d, cv=c, scale=sc, pat1=p, pat2=p, **COSTS_DEFAULT)
                for d, c in C_DISTRIBUTIONS for sc in C_SCALES[d] for T in C_HORIZONS for p in C_PATTERNS]
        if quick:
            rows = rows[:1]
    else:
        raise ValueError(block)
    for r in rows:
        yield dict(block=block, M=M_LEVEL, **r)


def instance_id(inst):
    return "|".join(str(inst[k]) for k in ("block", "dist", "cv", "M", "T", "scale", "pat1", "pat2",
                                          "K", "z", "R", "v", "h", "b"))


FIELDS = ["id", "block", "dist", "cv", "M", "T", "scale", "pat1", "pat2", "pair_type", "mu1", "mu2",
          "K", "z", "R", "v", "h", "b",
          "lo", "hi", "N", "seconds_grid_search", "seconds_sdp2", "C00_sdp2", "action_sdp2",
          "etc_alg", "hw_alg", "paths_alg", "attained_alg", "gap_pct", "hw_pct",
          "first_decision_alg", "seconds_alg_total", "seconds_alg_milp", "seconds_alg_refine", "seconds_alg_build",
          "decisions_alg", "seconds_per_decision", "seconds_per_decision_k1",
          "etc_opt_same_paths", "paired_gap_pct", "paired_hw_pct", "opt_paths_outside_grid",
          "etc_noref", "gap_noref_pct", "paired_alg_minus_noref_pct", "paired_hw_alg_minus_noref_pct", "workers",
          "states", "cpu_sdp2", "cpu_decision_k1", "seconds_decision_k1", "cpu_per_decision"]


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


def means(inst):
    mu1 = [inst["scale"] * x for x in pattern(inst["pat1"], inst["T"], inst["M"])]
    mu2 = [inst["scale"] * x for x in pattern(inst["pat2"], inst["T"], inst["M"])]
    return mu1, mu2


def new_row(inst, mu1, mu2):
    return dict(id=instance_id(inst), **inst, pair_type="identical" if inst["pat1"] == inst["pat2"] else "different",
                mu1=" ".join(map(str, mu1)), mu2=" ".join(map(str, mu2)))


def paired(policy, pm1, pm2, costs, n, base_costs):
    """Simulate `policy` on the first n paths of SEED; return its mean and the paired difference base - policy."""
    sim = simulate(policy, pm1, pm2, (0, 0), **costs, seed=SEED, n_paths=n)
    diff = base_costs - sim.costs
    return sim.mean, float(diff.mean()), half_width(diff)


def run_instance(inst, no_refine, env):
    costs = {k: inst[k] for k in ("K", "z", "R", "v", "h", "b")}
    mu1, mu2 = means(inst)
    dist, cv = inst["dist"], inst["cv"]
    row = new_row(inst, mu1, mu2)

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
    """Block C: the complete SDP-2 solve on the grid [-D, 2D], and the decisions of Algorithm 1 along C_PATHS paths."""
    costs = {k: inst[k] for k in ("K", "z", "R", "v", "h", "b")}
    mu1, mu2 = means(inst)
    dist, cv = inst["dist"], inst["cv"]
    row = new_row(inst, mu1, mu2)
    D = max_demand(mu1, mu2, dist, cv)
    lo, hi = -D, 2 * D
    c0 = time.process_time()
    s = SDP2(mu1, mu2, lo, hi, distribution=dist, cv=cv, **costs)
    row.update(lo=lo, hi=hi, N=hi - lo + 1, states=(hi - lo + 1) ** 2, seconds_sdp2=s.seconds,
               cpu_sdp2=time.process_time() - c0, C00_sdp2=s.value(), action_sdp2=str(s.action()))
    del s
    pm1, pm2 = period_pmfs(mu1, dist, cv), period_pmfs(mu2, dist, cv)
    pol = RecedingHorizonPolicy(pm1, pm2, **costs, distribution=dist, env=env)
    c0, t0 = time.process_time(), time.perf_counter()
    first = pol.decide(1, 0, 0)                              # period 1 at (0, 0): MILP-1 over the whole horizon
    row.update(cpu_decision_k1=time.process_time() - c0, seconds_decision_k1=time.perf_counter() - t0)
    D1, D2 = demand_batch(pm1, pm2, 1000, SEED, 0)
    run_paths(pol, D1[:C_PATHS], D2[:C_PATHS], (0, 0), **costs)
    cpu_all = time.process_time() - c0
    dt = np.array([x for _, x in pol.decision_times])
    row.update(first_decision_alg=str(first), decisions_alg=len(dt), seconds_per_decision=float(dt.mean()),
               cpu_per_decision=cpu_all / len(dt), seconds_per_decision_k1=row["seconds_decision_k1"])
    for mdl in pol._milp.values():
        mdl.model.dispose()
    return row


# -- parallel workers (block A) ------------------------------------------------------------------------
_ENV = None


def _init_worker():
    global _ENV
    _ENV = gurobi_env()


def _work(args):
    inst, no_refine = args
    try:
        if inst["block"] == "C":
            return run_timing(inst, _ENV), None
        return run_instance(inst, no_refine, _ENV), None
    except Exception as e:                                   # report and continue with the other instances
        return None, f"{instance_id(inst)}: {type(e).__name__}: {e}"


def report(block, n, total, inst, row, t_inst):
    if block == "C":
        print(f"[C {n}/{total}] {inst['dist']} cv={inst['cv']} s={inst['scale']} T={inst['T']} {inst['pat1']}  "
              f"|I|={row['states']}  SDP-2 {row['seconds_sdp2']:.1f}s (CPU {row['cpu_sdp2']:.1f}s)  "
              f"decision {row['seconds_per_decision']:.3f}s, period 1 {row['seconds_decision_k1']:.3f}s", flush=True)
        return
    print(f"[{block} {n}/{total}] T={inst['T']} {inst['dist']} cv={inst['cv']} s={inst['scale']} "
          f"{inst['pat1']}-{inst['pat2']} K={inst['K']} z={inst['z']} R={inst['R']} v={inst['v']} b={inst['b']}  "
          f"|I|={row['N'] ** 2}  SDP-2 {row['seconds_sdp2']:.1f}s  C={row['C00_sdp2']:.2f}  ETC={row['etc_alg']:.2f}  "
          f"gap={row['gap_pct']:.3f}% (+-{row['hw_pct']:.3f})  paired={row.get('paired_gap_pct', float('nan')):.3f}%  "
          f"decision {1000 * row['seconds_per_decision']:.1f}ms  {t_inst}", flush=True)


def run_block(block, quick, no_refine, out_dir, workers):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{block}.csv")
    done = done_ids(path)
    insts = [i for i in instances(block, quick) if instance_id(i) not in done]
    workers = workers if block in ("A", "Aplus") else (C_WORKERS if block == "C" else 1)
    print(f"block {block}: {len(insts)} instances to run, {workers} process(es), results in {path}", flush=True)
    if not insts:
        return
    t_start = time.perf_counter()
    if workers == 1:
        env = gurobi_env()
        for n, inst in enumerate(insts, 1):
            t0 = time.perf_counter()
            row = run_timing(inst, env) if block == "C" else run_instance(inst, no_refine, env)
            row["workers"] = 1
            append(path, row)
            report(block, n, len(insts), inst, row, f"instance {time.perf_counter() - t0:.0f}s")
        env.dispose()
        return
    with mp.get_context("spawn").Pool(workers, initializer=_init_worker) as pool:
        for n, (row, err) in enumerate(pool.imap_unordered(_work, [(i, no_refine) for i in insts]), 1):
            if err is not None:
                print(f"[{block} {n}/{len(insts)}] FAILED {err}", flush=True)
                continue
            row["workers"] = workers
            append(path, row)
            inst = {k: row[k] for k in ("T", "dist", "cv", "scale", "pat1", "pat2", "K", "z", "R", "v", "b")}
            if block == "C":
                report(block, n, len(insts), inst, row, "")
                continue
            report(block, n, len(insts), inst, row, f"elapsed {(time.perf_counter() - t_start) / 3600:.1f}h")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", default=RUN_BLOCK)
    ap.add_argument("--quick", action="store_true", default=RUN_QUICK)
    ap.add_argument("--no-refine", action="store_true", default=RUN_NO_REFINE)
    ap.add_argument("--workers", type=int, default=N_WORKERS)
    ap.add_argument("--out", default=OUT_DIR)
    a = ap.parse_args()
    out = a.out + ("_quick" if a.quick else "")
    for blk in (["A", "B"] if a.block == "all" else [a.block]):
        run_block(blk, a.quick, a.no_refine, out, a.workers)


if __name__ == "__main__":
    main()
