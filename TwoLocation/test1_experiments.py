"""
test1_experiments.py : SDP-1 versus SDP-2, accuracy and computational effort.

Questions
    Q1  Does SDP-2 reproduce SDP-1 exactly across distributions, horizons, demand scales and patterns?
    Q2  How much computation does SDP-2 save, and does the measured speed-up follow the operation count?
    Q3  How far can SDP-2 go after SDP-1 has become impractical?

Blocks
    A  main test bed, T = 4: 3 distributions x demand level M in {4, 7, 10} x 20 pattern pairs = 180 instances
    B  horizon: T in {4, 8, 12, 16, 20, 52}, M = 7, Poisson and Normal(cv = 0.3), 10 identical pairs = 120 instances
    C  mechanism: one instance (LCY2, M = 4, adequate at N = 45), grid widened step by step,
       N in {45, 60, 80, 100, 120, 160}; the grid only changes the size of the problem here
    D  cost robustness of Q1: the 36 cost settings of the 2022 version x 4 pattern pairs, M = 7, T = 4
    E  SDP-2 only on larger instances, M in {14, 21} at T = 4 and T in {20, 52} at M = 7;
       the SDP-1 time is extrapolated from the operation count

Demand means are fixed integers in the table DEMAND_PATTERNS below, one entry per (M, T) used in the experiments;
pattern(name, T, M) reads them. python test1_experiments.py --show-patterns writes a figure of all
ten patterns and prints their means, e.g. --show-patterns --M 7 --T 4 8 16.
Every row of the result csv files also records the means of both locations in the columns mu1 and mu2.

Usage
    python test1_experiments.py --block A          (or B, C, D, E, or all)
    python test1_experiments.py --block all --quick   (smoke test on a few small instances)
Results are appended to results_test1_int/<block>.csv one row per instance, so an interrupted run can be resumed:
instances already present in the csv are skipped.
"""
from __future__ import annotations

import os

# one BLAS thread, so that timings compare the two algorithms and not the thread pool
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

# ====================================================================================================
# settings for clicking "Run" in PyCharm; command-line arguments override them
RUN_BLOCK = "all"     # "A", "B", "D" or "all" (= A, B, D); "C" and "E" are optional and run only when named
RUN_QUICK = False     # True: a few small instances per block, about 10 minutes; False: the full study
# ====================================================================================================

import argparse
import csv
import itertools
import math
import time
from statistics import median

import numpy as np

from lt_sdp import SDP1, SDP2, demand_pmf

# ----------------------------------------------------------------------------------------------------
# demand patterns
# ----------------------------------------------------------------------------------------------------
# Ten expected demand patterns, as described in Ma, Rossi & Archibald (2022, EJOR 298, Section 5):
#   LCY1  life cycle, from launch to maturity through growth (logistic growth to a plateau)
#   LCY2  life cycle, from growth through maturity into decline (bell shape peaking at 40% of the horizon)
#   SIN1  sinusoidal, stronger oscillation (amplitude 0.8 of the level; cycle of 4 periods for T < 8, else 8)
#   SIN2  sinusoidal, weaker oscillation (amplitude 0.3 of the level, same cycle)
#   STAT  stationary
#   RAND  random (uniform between 0 and 2 times the level)
#   EMP1-EMP4  empirical-type patterns following Strijbosch, Syntetos, Boylan & Janssen (2011): one sample path
#              of an ARIMA(0,1,1) process each, with smoothing constants 0.2, 0.4, 0.6 and 0.8.
#
# The values are FIXED integers, listed for every combination of demand level M and horizon T used in the
# experiments. They were obtained once from the shapes described above: the shape was scaled to average M,
# rounded to integers by the largest-remainder rule so that the average demand per period is exactly M, and
# every period was given a mean of at least 1. Ties between equal remainders went to the earlier period.
PATTERN_NAMES = ["LCY1", "LCY2", "SIN1", "SIN2", "STAT", "RAND", "EMP1", "EMP2", "EMP3", "EMP4"]

DEMAND_PATTERNS = {
    (4, 4): {
        "LCY1": [1, 3, 6, 6],
        "LCY2": [4, 7, 4, 1],
        "SIN1": [7, 4, 1, 4],
        "SIN2": [5, 4, 3, 4],
        "STAT": [4, 4, 4, 4],
        "RAND": [7, 2, 4, 3],
        "EMP1": [4, 4, 5, 3],
        "EMP2": [6, 5, 2, 3],
        "EMP3": [5, 4, 6, 1],
        "EMP4": [4, 4, 3, 5],
    },
    (7, 4): {
        "LCY1": [2, 5, 10, 11],
        "LCY2": [6, 13, 7, 2],
        "SIN1": [13, 7, 1, 7],
        "SIN2": [9, 7, 5, 7],
        "STAT": [7, 7, 7, 7],
        "RAND": [12, 4, 7, 5],
        "EMP1": [8, 7, 8, 5],
        "EMP2": [10, 8, 5, 5],
        "EMP3": [9, 6, 10, 3],
        "EMP4": [7, 8, 5, 8],
    },
    (10, 4): {
        "LCY1": [3, 7, 14, 16],
        "LCY2": [8, 18, 11, 3],
        "SIN1": [18, 10, 2, 10],
        "SIN2": [13, 10, 7, 10],
        "STAT": [10, 10, 10, 10],
        "RAND": [17, 6, 9, 8],
        "EMP1": [11, 11, 12, 6],
        "EMP2": [14, 12, 7, 7],
        "EMP3": [13, 9, 14, 4],
        "EMP4": [10, 11, 8, 11],
    },
    (14, 4): {
        "LCY1": [3, 10, 20, 23],
        "LCY2": [12, 25, 15, 4],
        "SIN1": [25, 14, 3, 14],
        "SIN2": [18, 14, 10, 14],
        "STAT": [14, 14, 14, 14],
        "RAND": [24, 9, 13, 10],
        "EMP1": [16, 15, 16, 9],
        "EMP2": [20, 17, 9, 10],
        "EMP3": [18, 13, 20, 5],
        "EMP4": [15, 15, 11, 15],
    },
    (21, 4): {
        "LCY1": [5, 15, 30, 34],
        "LCY2": [17, 38, 23, 6],
        "SIN1": [38, 21, 4, 21],
        "SIN2": [27, 21, 15, 21],
        "STAT": [21, 21, 21, 21],
        "RAND": [36, 13, 20, 15],
        "EMP1": [24, 22, 25, 13],
        "EMP2": [30, 25, 14, 15],
        "EMP3": [27, 19, 30, 8],
        "EMP4": [22, 23, 16, 23],
    },
    (7, 8): {
        "LCY1": [2, 2, 4, 7, 9, 10, 11, 11],
        "LCY2": [4, 8, 12, 13, 10, 5, 2, 2],
        "SIN1": [11, 13, 11, 7, 3, 1, 3, 7],
        "SIN2": [8, 9, 8, 7, 6, 5, 6, 7],
        "STAT": [7, 7, 7, 7, 7, 7, 7, 7],
        "RAND": [7, 5, 8, 10, 11, 6, 3, 6],
        "EMP1": [5, 11, 11, 7, 4, 4, 8, 6],
        "EMP2": [7, 8, 5, 9, 5, 9, 6, 7],
        "EMP3": [3, 7, 8, 6, 8, 8, 8, 8],
        "EMP4": [5, 5, 5, 4, 5, 10, 12, 10],
    },
    (7, 12): {
        "LCY1": [2, 2, 2, 3, 5, 7, 9, 10, 11, 11, 11, 11],
        "LCY2": [3, 6, 9, 11, 13, 12, 10, 8, 5, 3, 2, 2],
        "SIN1": [10, 11, 9, 6, 3, 1, 3, 6, 9, 11, 9, 6],
        "SIN2": [8, 8, 8, 7, 5, 5, 5, 7, 8, 8, 8, 7],
        "STAT": [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
        "RAND": [2, 3, 14, 2, 4, 10, 5, 13, 13, 1, 6, 11],
        "EMP1": [11, 4, 1, 3, 10, 9, 6, 4, 5, 10, 11, 10],
        "EMP2": [5, 9, 6, 10, 7, 5, 6, 8, 5, 8, 9, 6],
        "EMP3": [1, 4, 6, 5, 8, 3, 9, 12, 10, 7, 10, 9],
        "EMP4": [6, 8, 4, 5, 7, 9, 7, 7, 8, 7, 6, 10],
    },
    (7, 16): {
        "LCY1": [2, 2, 2, 2, 3, 4, 6, 7, 9, 10, 10, 11, 11, 11, 11, 11],
        "LCY2": [3, 5, 7, 9, 11, 12, 13, 12, 11, 9, 7, 4, 3, 2, 2, 2],
        "SIN1": [11, 13, 11, 7, 3, 1, 3, 7, 11, 13, 11, 7, 3, 1, 3, 7],
        "SIN2": [8, 9, 8, 7, 6, 5, 6, 7, 8, 9, 8, 7, 6, 5, 6, 7],
        "STAT": [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
        "RAND": [2, 8, 5, 7, 5, 17, 4, 10, 5, 5, 8, 4, 9, 6, 15, 2],
        "EMP1": [3, 6, 9, 2, 1, 11, 3, 12, 7, 7, 11, 9, 8, 9, 10, 4],
        "EMP2": [5, 7, 2, 6, 7, 5, 7, 10, 12, 11, 7, 6, 9, 8, 7, 3],
        "EMP3": [10, 8, 8, 9, 4, 5, 2, 7, 17, 9, 4, 13, 3, 5, 1, 7],
        "EMP4": [3, 5, 4, 4, 7, 8, 6, 8, 6, 6, 7, 9, 9, 9, 10, 11],
    },
    (7, 20): {
        "LCY1": [2, 2, 2, 2, 2, 3, 4, 5, 6, 8, 9, 9, 10, 10, 11, 11, 11, 11, 11, 11],
        "LCY2": [3, 4, 6, 8, 9, 11, 12, 13, 13, 12, 11, 9, 8, 6, 4, 3, 2, 2, 2, 2],
        "SIN1": [10, 12, 10, 6, 3, 1, 3, 6, 10, 12, 10, 6, 3, 1, 3, 6, 10, 12, 10, 6],
        "SIN2": [8, 9, 8, 7, 5, 5, 5, 7, 8, 9, 8, 7, 5, 5, 5, 7, 8, 9, 8, 7],
        "STAT": [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
        "RAND": [2, 14, 6, 2, 1, 10, 8, 6, 7, 4, 3, 4, 13, 8, 5, 9, 10, 14, 4, 10],
        "EMP1": [5, 6, 8, 7, 8, 7, 4, 10, 6, 12, 7, 9, 7, 5, 7, 6, 6, 7, 5, 8],
        "EMP2": [2, 1, 2, 9, 6, 3, 1, 4, 1, 1, 5, 9, 7, 13, 10, 10, 15, 11, 15, 15],
        "EMP3": [3, 6, 1, 6, 7, 1, 4, 3, 5, 2, 5, 3, 6, 8, 9, 13, 13, 18, 17, 10],
        "EMP4": [4, 5, 6, 5, 5, 7, 6, 8, 10, 9, 9, 9, 10, 9, 8, 8, 6, 5, 4, 7],
    },
    (7, 52): {
        "LCY1": [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 6, 7, 7, 8, 8, 8, 9, 9,
                 9, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11],
        "LCY2": [3, 3, 4, 4, 5, 5, 6, 6, 7, 8, 9, 9, 10, 10, 11, 12, 12, 12, 13, 13, 13, 13, 13, 12, 12,
                 12, 11, 11, 10, 10, 9, 8, 8, 7, 6, 6, 5, 4, 4, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        "SIN1": [11, 12, 11, 7, 3, 1, 3, 7, 11, 12, 11, 7, 3, 1, 3, 7, 11, 12, 11, 7, 3, 1, 3, 7, 11, 12,
                 10, 7, 3, 1, 3, 7, 10, 12, 10, 7, 3, 1, 3, 7, 10, 12, 10, 7, 3, 1, 3, 7, 10, 12, 10, 7],
        "SIN2": [8, 9, 8, 7, 6, 5, 6, 7, 8, 9, 8, 7, 6, 5, 6, 7, 8, 9, 8, 7, 6, 5, 6, 7, 8, 9, 8, 7, 6, 5,
                 6, 7, 8, 9, 8, 7, 5, 5, 5, 7, 8, 9, 8, 7, 5, 5, 5, 7, 8, 9, 8, 7],
        "STAT": [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
        "RAND": [8, 7, 9, 1, 2, 7, 4, 1, 9, 1, 1, 3, 8, 5, 13, 1, 1, 3, 4, 3, 13, 7, 5, 13, 14, 3, 8, 1,
                 2, 7, 14, 12, 5, 8, 13, 15, 7, 2, 11, 6, 16, 11, 15, 9, 9, 3, 13, 3, 15, 7, 4, 2],
        "EMP1": [24, 2, 18, 2, 2, 5, 7, 14, 2, 3, 11, 17, 2, 10, 2, 2, 12, 11, 7, 13, 5, 22, 12, 3, 4, 2,
                 11, 4, 6, 2, 8, 9, 6, 2, 2, 13, 6, 5, 4, 5, 10, 9, 6, 3, 2, 2, 19, 4, 2, 2, 6, 2],
        "EMP2": [9, 8, 10, 6, 4, 4, 4, 4, 6, 3, 2, 3, 2, 1, 1, 3, 1, 1, 5, 1, 3, 1, 4, 3, 2, 3, 2, 5, 3,
                 3, 4, 9, 9, 6, 8, 8, 8, 10, 9, 10, 10, 12, 12, 13, 14, 14, 16, 18, 16, 19, 14, 18],
        "EMP3": [5, 6, 8, 8, 7, 7, 6, 9, 8, 10, 8, 6, 7, 7, 6, 8, 7, 7, 8, 6, 6, 6, 5, 5, 7, 6, 6, 7, 7,
                 7, 8, 7, 7, 7, 6, 6, 7, 8, 6, 8, 7, 9, 9, 8, 7, 7, 7, 7, 7, 8, 6, 6],
        "EMP4": [7, 7, 4, 2, 1, 1, 4, 2, 1, 1, 3, 2, 4, 2, 5, 4, 6, 8, 7, 5, 6, 9, 10, 11, 10, 9, 10, 9,
                 8, 10, 11, 13, 11, 10, 8, 10, 10, 9, 7, 9, 6, 10, 11, 11, 11, 9, 3, 8, 10, 7, 4, 8],
    },
}


def pattern(name: str, T: int, M: float) -> list:
    """Integer mean demand of pattern `name` over T periods at demand level M, from the fixed table DEMAND_PATTERNS."""
    key = (int(M), int(T))
    if key not in DEMAND_PATTERNS:
        raise ValueError(f"no fixed pattern for (M, T) = {key}; available: {sorted(DEMAND_PATTERNS)}")
    return list(DEMAND_PATTERNS[key][name])


def pattern_pairs():
    """10 identical pairs, then 10 different pairs (pattern k with pattern k + 1, cyclically)."""
    same = [(p, p) for p in PATTERN_NAMES]
    cross = [(PATTERN_NAMES[k], PATTERN_NAMES[(k + 1) % 10]) for k in range(10)]
    return same + cross


DISTRIBUTIONS = [("poisson", None), ("normal", 0.1), ("normal", 0.3)]
COSTS_DEFAULT = dict(K=20, z=1, R=10, v=0.5, h=1, b=5)
TAIL = 1e-4


# ----------------------------------------------------------------------------------------------------
# grid, operation counts, comparison
# ----------------------------------------------------------------------------------------------------
def max_demand(mu1, mu2, dist, cv, tail=TAIL):
    return max(len(demand_pmf(m, dist, cv, tail)) - 1 for m in list(mu1) + list(mu2))


def adequate_grid(mu1, mu2, dist, cv, costs, tail=TAIL, widen=1.25, rtol=1e-6, max_iter=10):
    """Smallest grid in the sequence [-D, 2D], then x1.25 at both ends, ..., that passes two checks:
    no optimal order-up-to level leaves the grid, and widening the grid by 25% at both ends changes
    C_1(0, 0) by less than rtol in relative terms. The checks use SDP2, which equals SDP1."""
    D = max_demand(mu1, mu2, dist, cv, tail)
    lo, hi = -D, 2 * D
    r = SDP2(mu1, mu2, lo, hi, distribution=dist, cv=cv, tail=tail, **costs)
    for _ in range(max_iter):
        lo2, hi2 = math.floor(lo * widen), math.ceil(hi * widen)
        r2 = SDP2(mu1, mu2, lo2, hi2, distribution=dist, cv=cv, tail=tail, **costs)
        stable = abs(r.value() - r2.value()) <= rtol * abs(r2.value())
        if r.orders_beyond_grid() == 0 and stable:
            return lo, hi
        lo, hi, r = lo2, hi2, r2
    raise RuntimeError("no adequate grid found; increase max_iter")


def operation_counts(lo, hi, T, qmax=None):
    """Bellman evaluations over the horizon.
    SDP-1: sum over states of |R(i)| (QA, QB) pairs. SDP-2: |I| (QA, QB) pairs plus sum over states of |R(i)|."""
    g = np.arange(lo, hi + 1)
    n = len(g)
    q2 = ((n if qmax is None else qmax) + 1) ** 2
    i1, i2 = np.meshgrid(g, g, indexing="ij")
    w_lo = np.maximum(-np.maximum(i2, 0), lo - i2, )          # x2 = i2 + W >= lo  and  W >= -max(i2, 0)
    w_lo = np.maximum(w_lo, i1 - hi)                            # x1 = i1 - W <= hi
    w_hi = np.minimum(np.maximum(i1, 0), hi - i2)               # W <= max(i1, 0)  and  x2 <= hi
    w_hi = np.minimum(w_hi, i1 - lo)                            # x1 >= lo
    sumR = int(np.maximum(w_hi - w_lo + 1, 0).sum())
    ops1 = T * sumR * q2
    ops2 = T * (n * n * q2 + sumR)
    return ops1, ops2, sumR / (n * n)


def timed(fun, reps, **kw):
    times, res = [], None
    for _ in range(reps):
        t0 = time.perf_counter()
        res = fun(**kw)
        times.append(time.perf_counter() - t0)
    return res, median(times)


def compare(row, mu1, mu2, lo, hi, dist, cv, costs, reps1, reps2, run_sdp1=True):
    kw = dict(demand_mean1=mu1, demand_mean2=mu2, min_inventory=lo, max_inventory=hi,
              distribution=dist, cv=cv, tail=TAIL, **costs)
    SDP2(**kw)                                                  # warm-up
    r2, t2 = timed(SDP2, reps2, **kw)
    T = len(mu1)
    ops1, ops2, meanR = operation_counts(lo, hi, T)
    row.update(N=hi - lo + 1, lo=lo, hi=hi, mean_R=round(meanR, 2), ops_sdp1=ops1, ops_sdp2=ops2,
               predicted_speedup=ops1 / ops2, time_sdp2=t2, C00_sdp2=r2.value(),
               beyond_grid=r2.orders_beyond_grid())
    if run_sdp1:
        r1, t1 = timed(SDP1, reps1, **kw)
        same = (r1.W == r2.W) & (r1.QA == r2.QA) & (r1.QB == r2.QB)
        row.update(time_sdp1=t1, measured_speedup=t1 / t2, C00_sdp1=r1.value(),
                   max_abs_diff=float(np.max(np.abs(r1.C - r2.C))),
                   rel_diff_00=abs(r1.value() - r2.value()) / abs(r1.value()),
                   identical_actions=float(same.mean()), differing_actions=int((~same).sum()))
    else:
        row.update(time_sdp1_extrapolated=t2 * ops1 / ops2)
    return row


# ----------------------------------------------------------------------------------------------------
# instance lists
# ----------------------------------------------------------------------------------------------------
def instances(block, quick=False):
    pairs = pattern_pairs()
    if block == "A":
        levels = [4, 7] if quick else [4, 7, 10]
        prs = pairs[:2] + pairs[10:12] if quick else pairs
        for (dist, cv), M, (p1, p2) in itertools.product(DISTRIBUTIONS, levels, prs):
            yield dict(dist=dist, cv=cv, M=M, T=4, pat1=p1, pat2=p2, **COSTS_DEFAULT)
    elif block == "B":
        Ts = [4, 8] if quick else [4, 8, 12, 16, 20, 52]
        fams = PATTERN_NAMES[:2] if quick else PATTERN_NAMES
        for (dist, cv), T, p in itertools.product([("poisson", None), ("normal", 0.3)], Ts, fams):
            yield dict(dist=dist, cv=cv, M=7, T=T, pat1=p, pat2=p, **COSTS_DEFAULT)
    elif block == "C":
        Ns = [45, 60] if quick else [45, 60, 80, 100, 120, 160]
        for N in Ns:
            yield dict(dist="poisson", cv=None, M=4, T=4, pat1="LCY2", pat2="LCY2", N_fixed=N, **COSTS_DEFAULT)
    elif block == "D":
        grid = itertools.product([10, 20, 30], [5, 10, 20], [3, 5], [(2, 1), (1, 0.5)])
        prs = [pairs[0], pairs[4], pairs[10], pairs[15]]           # LCY1-LCY1, STAT-STAT, LCY1-LCY2, EMP1-EMP2
        settings = list(grid)[:4] if quick else list(grid)
        for (K, R, b, (z, v)), (p1, p2) in itertools.product(settings, prs[:2] if quick else prs):
            yield dict(dist="poisson", cv=None, M=7, T=4, pat1=p1, pat2=p2, K=K, z=z, R=R, v=v, h=1, b=b)
    elif block == "E":
        cases = [(14, 4), (21, 4), (7, 20), (7, 52)]
        for (M, T), p in itertools.product(cases[:2] if quick else cases, ["LCY2", "SIN1"]):
            yield dict(dist="poisson", cv=None, M=M, T=T, pat1=p, pat2=p, **COSTS_DEFAULT)


def instance_id(inst):
    keys = ("dist", "cv", "M", "T", "pat1", "pat2", "K", "z", "R", "v", "h", "b", "N_fixed")
    return "|".join(str(inst.get(k)) for k in keys)


def done_ids(path):
    if not os.path.exists(path):
        return set()
    with open(path, newline="") as f:
        return {r["id"] for r in csv.DictReader(f)}


def append(path, row, fields):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


FIELDS = ["id", "block", "dist", "cv", "M", "T", "pat1", "pat2", "pair_type", "mu1", "mu2", "K", "z", "R", "v", "h", "b",
          "N", "lo", "hi", "mean_R", "beyond_grid", "ops_sdp1", "ops_sdp2", "predicted_speedup",
          "time_sdp1", "time_sdp2", "measured_speedup", "time_sdp1_extrapolated",
          "C00_sdp1", "C00_sdp2", "max_abs_diff", "rel_diff_00", "identical_actions", "differing_actions"]


def run_block(block, out_dir, quick, reps1, reps2):
    path = os.path.join(out_dir, f"{block}.csv")
    done = done_ids(path)
    insts = list(instances(block, quick))
    print(f"block {block}: {len(insts)} instances, {len(done)} already in {path}", flush=True)
    for k, inst in enumerate(insts, 1):
        iid = instance_id(inst)
        if iid in done:
            continue
        T = inst["T"]
        mu1, mu2 = pattern(inst["pat1"], T, inst["M"]), pattern(inst["pat2"], T, inst["M"])
        costs = {c: inst[c] for c in ("K", "z", "R", "v", "h", "b")}
        if "N_fixed" in inst:
            N = inst["N_fixed"]
            lo = -round(N / 3)                                   # one third of the levels below zero
            hi = lo + N - 1
        else:
            lo, hi = adequate_grid(mu1, mu2, inst["dist"], inst["cv"], costs)
        row = dict(id=iid, block=block, pair_type="same" if inst["pat1"] == inst["pat2"] else "different",
                   mu1=" ".join(map(str, mu1)), mu2=" ".join(map(str, mu2)), **inst)
        t0 = time.perf_counter()
        compare(row, mu1, mu2, lo, hi, inst["dist"], inst["cv"], costs, reps1, reps2, run_sdp1=(block != "E"))
        append(path, row, FIELDS)
        msg = (f"[{block} {k}/{len(insts)}] {inst['dist']} cv={inst['cv']} M={inst['M']} T={T} "
               f"{inst['pat1']}-{inst['pat2']} N={row['N']}  SDP2 {row['time_sdp2']:.2f}s")
        if "time_sdp1" in row:
            msg += (f"  SDP1 {row['time_sdp1']:.2f}s  speed-up {row['measured_speedup']:.1f} "
                    f"(predicted {row['predicted_speedup']:.1f})  max|diff| {row['max_abs_diff']:.1e}")
        else:
            msg += f"  SDP1 extrapolated {row['time_sdp1_extrapolated']:.0f}s"
        print(msg + f"  [{time.perf_counter() - t0:.1f}s]", flush=True)


def show_patterns(out, M=7, horizons=(4, 12)):
    """Figure and printed values of the ten patterns at demand level M for each horizon in `horizons`."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(out, exist_ok=True)
    f, axes = plt.subplots(len(horizons), 10, figsize=(16, 2.2 * len(horizons)), sharey=True, squeeze=False)
    for r, T in enumerate(horizons):
        for c, name in enumerate(PATTERN_NAMES):
            ax = axes[r, c]
            ax.bar(range(1, T + 1), pattern(name, T, M), color="C0")
            ax.set_title(f"{name}, T={T}", fontsize=8)
            ax.tick_params(labelsize=7)
    f.tight_layout()
    path = os.path.join(out, "demand_patterns.png")
    f.savefig(path.replace(".png", f"_M{M:g}.png"), dpi=150)
    path = path.replace(".png", f"_M{M:g}.png")
    print(f"figure: {path}")
    for T in horizons:
        for name in PATTERN_NAMES:
            print(f"T={T:2d} {name}: {pattern(name, T, M)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", default=RUN_BLOCK, choices=["A", "B", "C", "D", "E", "all"])
    ap.add_argument("--out", default="results_test1_int")
    ap.add_argument("--quick", action=argparse.BooleanOptionalAction, default=RUN_QUICK,
                    help="a few small instances per block, for a first check")
    ap.add_argument("--reps-sdp1", type=int, default=1, help="timed runs of SDP1 per instance, median reported")
    ap.add_argument("--reps-sdp2", type=int, default=3, help="timed runs of SDP2 per instance, median reported")
    ap.add_argument("--show-patterns", action="store_true", help="plot and print the ten demand patterns and exit")
    ap.add_argument("--M", type=float, default=7, help="demand level for --show-patterns")
    ap.add_argument("--T", type=int, nargs="+", default=[4, 12], help="horizons for --show-patterns")
    a = ap.parse_args()
    if a.show_patterns:
        show_patterns(a.out, M=a.M, horizons=tuple(a.T))
        return
    out = a.out + ("_quick" if a.quick else "")
    os.makedirs(out, exist_ok=True)
    for blk in (["A", "B", "D"] if a.block == "all" else [a.block]):
        run_block(blk, out, a.quick, a.reps_sdp1, a.reps_sdp2)


if __name__ == "__main__":
    main()