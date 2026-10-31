"""
algorithm.py : Algorithm 1 of the paper, the single entry point for the experiments.

    res = algorithm(demand_mean1, demand_mean2, K, z, R, v, h, b, distribution="poisson", cv=None, ...)

The arguments demand_mean1 ... tail are those of lt_sdp.SDP2, so one parameter set drives both.
The function simulates the receding-horizon policy of Algorithm 1 from the opening inventory i_0:

    repeat
        sample demand paths at both locations                       (simulate.py, common random numbers)
        for k = 1..T
            solve MILP-1 over k..T from i_{k-1}; keep Q^^j_k         (milp1.py)
            refine the transshipment by (alg.refine1)                (policies.py)
            update i_k with the decisions and the sampled demand
    until the half-width of the 95% confidence interval is at most 0.1% of the estimated expected total cost

and returns the estimate of the expected total cost (ETC), its confidence interval, the decision of period 1 at i_0,
and the computing times.

Example
    from algorithm import algorithm
    r = algorithm([4, 6, 8, 6], [4, 6, 8, 6], K=20, z=1, R=10, v=0.5, h=1, b=5)
    print(r.etc, r.half_width, r.first_decision)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Sequence, Union

import numpy as np

import gurobipy as gp

from lt_sdp import demand_pmf
from policies import RecedingHorizonPolicy
from simulate import SimResult, simulate


@dataclass
class AlgorithmResult:
    etc: float                       # estimated expected total cost from i_0
    half_width: float                # half-width of the confidence interval
    n_paths: int
    precision_attained: bool
    first_decision: tuple            # (W~_1, Q~^1_1, Q~^2_1) at i_0, the decision a manager implements now
    seconds_total: float             # wall time of the whole call
    seconds_milp: float              # Gurobi solves
    seconds_refine: float            # one-dimensional searches
    seconds_build: float             # building the MILP-1 models and loss tables
    n_decisions: int                 # distinct (k, state) pairs at which a decision was computed
    seconds_per_decision: float      # mean wall time of one decision (MILP-1 solve plus refinement)
    seconds_per_decision_k1: float   # the same at k = 1, the longest residual horizon
    policy: RecedingHorizonPolicy = field(repr=False)
    sim: SimResult = field(repr=False)


def period_pmfs(demand_mean, distribution="poisson", cv=None, tail=1e-4):
    """Truncated and renormalised period pmfs, exactly as lt_sdp builds them."""
    T = len(demand_mean)
    cvs = [cv] * T if (cv is None or np.ndim(cv) == 0) else list(cv)
    return [demand_pmf(demand_mean[t], distribution, cvs[t], tail) for t in range(T)]


def gurobi_env() -> gp.Env:
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    return env


def algorithm(demand_mean1: Sequence[float], demand_mean2: Sequence[float],
              K: float, z: float, R: float, v: float, h: float, b: float,
              distribution: str = "poisson", cv: Union[None, float, Sequence[float]] = None, tail: float = 1e-4,
              initial_inventory=(0, 0), partitions: int = 10, refine: bool = True,
              rel_precision: float = 1e-3, confidence: float = 0.95, seed: int = 1234,
              batch_size: int = 1000, min_paths: int = 1000, max_paths: int = 1_000_000,
              threads: int = 1, env: Optional[gp.Env] = None, verbose: bool = False) -> AlgorithmResult:
    t0 = time.perf_counter()
    p1 = period_pmfs(demand_mean1, distribution, cv, tail)
    p2 = period_pmfs(demand_mean2, distribution, cv, tail)
    own_env = env is None
    env = gurobi_env() if own_env else env
    policy = None
    try:
        policy = RecedingHorizonPolicy(p1, p2, K, z, R, v, h, b, distribution, partitions,
                                       refine=refine, threads=threads, env=env)
        first = policy.decide(1, *initial_inventory)
        sim = simulate(policy, p1, p2, initial_inventory, K, z, R, v, h, b, seed=seed,
                       rel_precision=rel_precision, confidence=confidence, batch_size=batch_size,
                       min_paths=min_paths, max_paths=max_paths, verbose=verbose)
    finally:
        if own_env:                                  # free Gurobi; the stored decisions in policy.cache remain usable
            for mdl in (policy._milp.values() if policy is not None else []):
                mdl.model.dispose()
            env.dispose()
    dt = np.array([s for _, s in policy.decision_times])
    dt1 = np.array([s for k, s in policy.decision_times if k == 1])
    return AlgorithmResult(etc=sim.mean, half_width=sim.half_width, n_paths=sim.n, precision_attained=sim.attained,
                           first_decision=first, seconds_total=time.perf_counter() - t0,
                           seconds_milp=policy.time_milp, seconds_refine=policy.time_refine,
                           seconds_build=policy.time_build, n_decisions=len(dt),
                           seconds_per_decision=float(dt.mean()), seconds_per_decision_k1=float(dt1.mean()),
                           policy=policy, sim=sim)
