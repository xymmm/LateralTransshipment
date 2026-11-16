"""
simulate.py : Monte Carlo estimate of the expected total cost of a policy, with common random numbers and the
stopping rule of Algorithm 1 (half-width of the 95% confidence interval at most 0.1% of the estimate).

Dynamics and costs of one period, as LT_RQ_RecedingHorizon.singleSimulationRH and lt_sdp
    opening inventory (i1, i2) -> decision (W, Q1, Q2) from policy.decide(k, i1, i2)
    cost u(W) + c(Q1) + c(Q2) + h max(x, 0) + b max(-x, 0) at each location, x the closing inventory
    closing inventory  x1 = i1 - W + Q1 - d1,  x2 = i2 + W + Q2 - d2

Demand
    Sampled by inversion from the truncated and renormalised period pmfs of lt_sdp.demand_pmf, so the simulated
    system is the one whose optimal cost SDP-2 computes. Batch number m of an instance always uses the random
    generator seeded with (seed, m): two policies simulated with the same seed see the same demand paths.

Stopping rule
    Paths are added in batches. After each batch the Student-t confidence interval of the mean path cost is
    computed (as umontreal.ssj Tally.setConfidenceIntervalStudent in the Java code); the run stops when its
    half-width is at most rel_precision times the mean, with at least min_paths paths, or at max_paths.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy.stats import t as student_t


@dataclass
class SimResult:
    mean: float
    half_width: float
    n: int
    attained: bool               # precision reached before max_paths
    costs: np.ndarray            # total cost of each path, in path order
    seconds: float


def demand_batch(pmfs1: Sequence[np.ndarray], pmfs2: Sequence[np.ndarray], n: int, seed: int, batch: int):
    """Demand paths of batch `batch`: arrays (n, T) for locations 1 and 2."""
    rng = np.random.default_rng([seed, batch])
    T = len(pmfs1)
    out = []
    for pmfs in (pmfs1, pmfs2):
        U = rng.random((n, T))
        D = np.empty((n, T), dtype=np.int64)
        for t in range(T):
            cdf = np.cumsum(pmfs[t])
            D[:, t] = np.minimum(np.searchsorted(cdf, U[:, t], side="left"), len(cdf) - 1)
        out.append(D)
    return out[0], out[1]


def run_paths(policy, D1: np.ndarray, D2: np.ndarray, i0, K, z, R, v, h, b) -> np.ndarray:
    """Total cost of each path. Decisions are asked once per distinct state and period."""
    n, T = D1.shape
    x1 = np.full(n, int(i0[0]), dtype=np.int64)
    x2 = np.full(n, int(i0[1]), dtype=np.int64)
    cost = np.zeros(n)
    for k in range(1, T + 1):
        states, inv = np.unique(np.stack([x1, x2], axis=1), axis=0, return_inverse=True)
        inv = inv.ravel()
        dec = np.array([policy.decide(k, int(a), int(c)) for a, c in states], dtype=np.int64)
        W, Q1, Q2 = dec[inv, 0], dec[inv, 1], dec[inv, 2]
        x1 = x1 - W + Q1 - D1[:, k - 1]
        x2 = x2 + W + Q2 - D2[:, k - 1]
        cost += (np.where(W != 0, R + v * np.abs(W), 0.0)
                 + np.where(Q1 > 0, K + z * Q1, 0.0) + np.where(Q2 > 0, K + z * Q2, 0.0)
                 + np.where(x1 >= 0, h * x1, -b * x1) + np.where(x2 >= 0, h * x2, -b * x2))
    return cost


def half_width(costs: np.ndarray, confidence: float = 0.95) -> float:
    n = len(costs)
    return float(student_t.ppf(0.5 + confidence / 2, n - 1) * costs.std(ddof=1) / np.sqrt(n))


def simulate(policy, pmfs1, pmfs2, i0, K, z, R, v, h, b, seed: int = 1234,
             rel_precision: float = 1e-3, confidence: float = 0.95,
             batch_size: int = 1000, min_paths: int = 1000, max_paths: int = 1_000_000,
             n_paths: Optional[int] = None, verbose: bool = False) -> SimResult:
    """Simulate until the stopping rule holds; with n_paths given, simulate exactly that many paths
    (the first n_paths of the same seed), which is used to evaluate a second policy on the same paths."""
    t0 = time.perf_counter()
    chunks, n, m = [], 0, 0
    while True:
        size = batch_size if n_paths is None else min(batch_size, n_paths - n)
        D1, D2 = demand_batch(pmfs1, pmfs2, batch_size, seed, m)          # full batch, so paths do not depend on size
        chunks.append(run_paths(policy, D1[:size], D2[:size], i0, K, z, R, v, h, b))
        n += size
        m += 1
        costs = np.concatenate(chunks)
        hw = half_width(costs, confidence)
        mean = float(costs.mean())
        if verbose:
            print(f"  paths {n:>8d}   mean {mean:.4f}   half-width {hw / mean:.5%}")
        if n_paths is not None:
            if n >= n_paths:
                return SimResult(mean, hw, n, hw <= rel_precision * abs(mean), costs, time.perf_counter() - t0)
            continue
        if n >= min_paths and hw <= rel_precision * abs(mean):
            return SimResult(mean, hw, n, True, costs, time.perf_counter() - t0)
        if n >= max_paths:
            return SimResult(mean, hw, n, False, costs, time.perf_counter() - t0)
