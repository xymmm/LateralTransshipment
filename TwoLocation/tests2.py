"""
tests2.py : checks of milp1.py, policies.py and simulate.py. Run before the experiments: python tests2.py

    1  partition coefficients: Poisson against the Java sampling estimate (possionPiecewisePartitions, lambda = 14,
       N = 5, target values in its testExpectedValues); normal against mu + sigma E[Z | Omega_l] of Rossi et al.
    2  loss functions: Loss.complement and Loss.loss against direct sums
    3  MILP-1 against enumeration on small instances (k = 1 and k = 2, with and without transshipment)
    4  MILP-1 is a lower bound: its objective never exceeds the exact expected cost of its own decisions
    5  C^pi_k of policies.py against an independent forward recursion with the decisions fixed
    6  simulator: the simulated cost of the SDP-2 optimal policy agrees with the SDP-2 value
    7  sdp_single: without transshipment (R very large) SDP-2 equals the sum of the two single-location SDPs
"""
from __future__ import annotations

import itertools
import math

import numpy as np
from scipy.stats import norm, poisson

from algorithm import gurobi_env, period_pmfs
from lt_sdp import SDP2
from milp1 import MILP1, Loss, ROSSI_N10_MEAN, ROSSI_N10_PROB, conditional_means, cumulative_pmfs, partition_coefficients, partition_probabilities
from policies import OptimalPolicy, RecedingHorizonPolicy
from simulate import simulate

ENV = gurobi_env()
OK = []


def check(name, cond, info=""):
    OK.append(bool(cond))
    print(f"[{'pass' if cond else 'FAIL'}] {name} {info}")


# 1 -------------------------------------------------------------------------------------------------------
def test_partitions():
    pmf = poisson.pmf(np.arange(200), 14)
    got = conditional_means(pmf / pmf.sum(), np.full(5, 0.2))
    java = np.array([8.99635, 11.918, 13.841, 15.8981, 19.4344])        # sampling estimate, 100000 samples
    check("1a Poisson conditional means vs Java (lambda 14, N 5)", np.max(np.abs(got - java)) < 0.05,
          f"max diff {np.max(np.abs(got - java)):.4f}  ours {np.round(got, 4)}")
    mu, sd = 70.0, 21.0                                                # a cumulative demand, cv 0.3
    k = np.arange(0, 250)
    pmf = norm.cdf(k + .5, mu, sd) - norm.cdf(k - .5, mu, sd)
    pmf /= pmf.sum()
    got = conditional_means(pmf, np.array(ROSSI_N10_PROB))
    ref = mu + sd * np.array(ROSSI_N10_MEAN)
    check("1b normal conditional means vs mu + sigma E[Z|Omega]", np.max(np.abs(got - ref)) / sd < 0.02,
          f"max diff / sigma {np.max(np.abs(got - ref)) / sd:.4f}")
    P, S = partition_coefficients(pmf, np.array(ROSSI_N10_PROB))
    check("1c last piece is X - E[d]", abs(P[-1] - 1) < 1e-15 and abs(S[-1] - np.dot(k, pmf)) < 1e-12)


# 2 -------------------------------------------------------------------------------------------------------
def test_loss():
    pmf = period_pmfs([5, 9], "poisson")
    c = cumulative_pmfs(pmf, 0)[1]
    L = Loss(c)
    w = np.arange(len(c))
    xs = np.arange(-5, len(c) + 10)
    d1 = max(abs(L.complement(x) - np.dot(np.maximum(x - w, 0), c)) for x in xs)
    d2 = max(abs(L.loss(x) - np.dot(np.maximum(w - x, 0), c)) for x in xs)
    check("2  loss functions against direct sums", max(d1, d2) < 1e-10, f"max diff {max(d1, d2):.1e}")


# 3, 4 ----------------------------------------------------------------------------------------------------
def pwl_value(X, cum, probs):
    """Tightest value of H and B allowed by (pwl-H), (pwl-B) and H, B >= 0 at expected inventory X."""
    P, S = partition_coefficients(cum, probs)
    dcum = float(np.dot(np.arange(len(cum)), cum))
    Xa = np.asarray(X, float)[..., None]
    Hb = np.maximum(0, np.max(P * Xa - S, axis=-1))
    Bb = np.maximum(0, np.max(P * Xa - S - Xa + dcum, axis=-1))
    return Hb, Bb


def brute_force_milp(k, i0, pm1, pm2, K, z, R, v, h, b, probs, qmax):
    """Enumerate MILP-1 for a residual horizon of two periods (k = T - 1) with integer Q in 0..qmax."""
    L = len(pm1) - k + 1
    assert L == 2
    dbar = [[float(np.dot(np.arange(len(p)), p)) for p in pm[k - 1:]] for pm in (pm1, pm2)]
    cum = [cumulative_pmfs(pm, k - 1) for pm in (pm1, pm2)]
    q = np.arange(qmax + 1)
    cq = np.where(q > 0, K + z * q, 0.0)
    best = math.inf
    for W1 in range(-max(i0[1], 0), max(i0[0], 0) + 1):
        uW1 = (R + v * abs(W1)) if W1 else 0.0
        for a, c in itertools.product(q, q):
            I1 = i0[0] - W1 + a - dbar[0][0]
            I2 = i0[1] + W1 + c - dbar[1][0]
            H1, B1 = pwl_value(I1 + dbar[0][0], cum[0][0], probs)
            H2, B2 = pwl_value(I2 + dbar[1][0], cum[1][0], probs)
            first = uW1 + cq[a] + cq[c] + h * (H1 + H2) + b * (B1 + B2)
            if first >= best:
                continue
            for W2 in range(-math.floor(max(I2, 0)), math.floor(max(I1, 0)) + 1):
                uW2 = (R + v * abs(W2)) if W2 else 0.0
                X1 = (I1 - W2 + q)[:, None] + 0 * q[None, :] - dbar[0][1] + dbar[0][0] + dbar[0][1]
                X2 = 0 * q[:, None] + (I2 + W2 + q)[None, :] + dbar[1][0]
                Ha, Ba = pwl_value(X1, cum[0][1], probs)
                Hc, Bc = pwl_value(X2, cum[1][1], probs)
                tot = first + uW2 + cq[:, None] + cq[None, :] + h * (Ha + Hc) + b * (Ba + Bc)
                best = min(best, float(tot.min()))
    return best


def test_milp():
    costs = dict(K=20, z=1, R=5, v=0.5, h=1, b=5)
    for dist, cv, mu1, mu2 in (("poisson", None, [3, 5, 2], [2, 4, 6]), ("normal", 0.3, [4, 3, 6], [6, 2, 3])):
        pm1, pm2 = period_pmfs(mu1, dist, cv), period_pmfs(mu2, dist, cv)
        probs = partition_probabilities(dist)
        for i0 in ((0, 0), (12, -3), (-2, 9)):
            m = MILP1(2, pm1, pm2, **costs, probs=probs, env=ENV)
            sol = m.solve(*i0)
            bf = brute_force_milp(2, i0, pm1, pm2, **costs, probs=probs, qmax=25)
            check(f"3  MILP-1 = enumeration, {dist}, k=2, i={i0}", abs(sol.obj - bf) < 1e-6 * max(1, abs(bf)),
                  f"gurobi {sol.obj:.6f} enum {bf:.6f}  plan W={sol.W} Q1={sol.Q1} Q2={sol.Q2}")
            m.model.dispose()


def test_lower_bound():
    costs = dict(K=20, z=1, R=10, v=0.5, h=1, b=5)
    mu1, mu2 = [4, 9, 12, 6, 3, 8], [7, 3, 5, 11, 9, 2]
    for dist, cv in (("poisson", None), ("normal", 0.3)):
        pm1, pm2 = period_pmfs(mu1, dist, cv), period_pmfs(mu2, dist, cv)
        pol = RecedingHorizonPolicy(pm1, pm2, **costs, distribution=dist, env=ENV)
        worst = 0.0
        for k, i0 in itertools.product((1, 3, 5), ((0, 0), (15, -5), (-4, 20), (30, 30))):
            plan = pol.milp(k).solve(*i0)
            W = plan.W[0]
            exact = (((costs["R"] + costs["v"] * abs(W)) if W else 0.0)
                     + float(pol.value_fixed(k, plan, i0[0] - W, i0[1] + W)[0]))
            worst = max(worst, plan.obj - exact)
            gap = (exact - plan.obj) / exact
        check(f"4  MILP-1 objective <= exact cost of its decisions, {dist}", worst <= 1e-7,
              f"max(obj - exact) = {worst:.2e}")


# 5 -------------------------------------------------------------------------------------------------------
def forward_fixed(k, plan, post, pmfs, h, b, K, z, R, v):
    """Independent evaluation: propagate the distribution of the inventory at each location separately
    under the fixed decisions, the later transshipments executed in full on every demand outcome."""
    total = sum((R + v * abs(w)) if w else 0.0 for w in plan.W[1:])
    for j, Q, sgn in ((0, plan.Q1, -1), (1, plan.Q2, +1)):
        lo = -2000
        dist = np.zeros(6000)
        dist[post[j] - lo] = 1.0
        for s in range(len(plan.W)):
            shift = int(Q[s] + (sgn * plan.W[s] if s > 0 else 0))
            total += (K + z * Q[s]) if Q[s] > 0 else 0.0
            dist = np.roll(dist, shift)
            dist = np.convolve(dist, pmfs[j][k - 1 + s][::-1])[len(pmfs[j][k - 1 + s]) - 1:]   # x - d
            dist = np.concatenate((dist, np.zeros(6000 - len(dist))))
            x = np.arange(len(dist)) + lo
            total += float(np.dot(dist, np.where(x >= 0, h * x, -b * x)))
    return total


def test_value_fixed():
    costs = dict(K=20, z=1, R=10, v=0.5, h=1, b=5)
    mu1, mu2 = [4, 9, 12, 6, 3], [7, 3, 5, 11, 9]
    pm1, pm2 = period_pmfs(mu1, "poisson"), period_pmfs(mu2, "poisson")
    pol = RecedingHorizonPolicy(pm1, pm2, **costs, distribution="poisson", env=ENV)
    worst = 0.0
    for k, i0 in ((1, (0, 0)), (2, (20, -6)), (4, (3, 14))):
        plan = pol.milp(k).solve(*i0)
        for W in range(-max(i0[1], 0), max(i0[0], 0) + 1):
            post = (i0[0] - W, i0[1] + W)
            a = float(pol.value_fixed(k, plan, post[0], post[1])[0])
            c = forward_fixed(k, plan, post, (pm1, pm2), **costs)
            worst = max(worst, abs(a - c) / c)
    check("5  C^pi_k closed form = forward recursion with decisions fixed", worst < 1e-10, f"max rel diff {worst:.1e}")


# 6 -------------------------------------------------------------------------------------------------------
def test_simulator():
    costs = dict(K=20, z=1, R=10, v=0.5, h=1, b=5)
    mu1, mu2 = [4, 6, 8, 6], [7, 3, 2, 5]
    for dist, cv in (("poisson", None), ("normal", 0.3)):
        res = SDP2(mu1, mu2, -30, 60, distribution=dist, cv=cv, **costs)
        pm1, pm2 = period_pmfs(mu1, dist, cv), period_pmfs(mu2, dist, cv)
        sim = simulate(OptimalPolicy(res), pm1, pm2, (0, 0), **costs, seed=99)
        zscore = (sim.mean - res.value()) / (sim.half_width / 1.96)
        check(f"6  simulated SDP-2 policy = SDP-2 value, {dist}", abs(zscore) < 3,
              f"sim {sim.mean:.4f} +- {sim.half_width:.4f} ({sim.n} paths), SDP-2 {res.value():.4f}, z = {zscore:.2f}")


# 7 -------------------------------------------------------------------------------------------------------
def test_single():
    from sdp_single import SDP_single
    costs = dict(K=20, z=1, h=1, b=5)
    mu1, mu2 = [4, 6, 8, 6], [7, 3, 2, 5]
    for dist, cv in (("poisson", None), ("normal", 0.3)):
        s = SDP2(mu1, mu2, -20, 60, R=1e9, v=1, distribution=dist, cv=cv, **costs)
        a = SDP_single(mu1, -20, 60, distribution=dist, cv=cv, **costs)
        c = SDP_single(mu2, -20, 60, distribution=dist, cv=cv, **costs)
        diff = np.max(np.abs(s.C - (a.C[:, :, None] + c.C[:, None, :])))
        same = np.mean((s.QA == a.Q[:, :, None]) & (s.QB == c.Q[:, None, :]))
        check(f"7  SDP-2 without transshipment = two single-location SDPs, {dist}", diff < 1e-9 and same == 1,
              f"max diff {diff:.1e}, identical orders {same:.0%}")


if __name__ == "__main__":
    test_partitions()
    test_loss()
    test_milp()
    test_lower_bound()
    test_value_fixed()
    test_simulator()
    test_single()
    print(f"\n{sum(OK)} of {len(OK)} checks passed")
