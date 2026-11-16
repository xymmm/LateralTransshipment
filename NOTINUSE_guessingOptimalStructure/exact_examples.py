"""
NOTINUSE_guessingOptimalStructure/exact_examples.py : counterexamples of the paper in exact rational arithmetic.

    python exact_examples.py            all examples of the paper
    python exact_examples.py search     search of the deterministic two-period family over K

The script imports no other file of the project. It solves problems with one or two periods and demands with
finite support. Every number is a fractions.Fraction, so every comparison is exact.

Method.
  last period   Lemma 7 of the paper gives the second stage in closed form: location j reorders up to S^j
                if and only if f^j(x) > K + f^j(S^j), with f^j(y) = z y + l^j(y) convex.
  first stage   minimum over the finite set of feasible transshipments.
  period 1 of a two-period problem
                the levels after reordering are enumerated up to a cap Y. The script then checks, for every
                state it evaluates, the bound
                    cost >= z (Y+1-x^j) + h (Y+1-mu^j_1) + h (Y+1+x^k - sum of all expected demands)
                for every decision with y^j > Y, which follows from l(y) >= h (y - mu) and from
                C_2(i) >= h (i^1+i^2-mu^1_2-mu^2_2). If the bound exceeds the best cost found, no level
                above the cap can be optimal. The assertion fails otherwise.
"""
from __future__ import annotations

import sys
from fractions import Fraction as F
from functools import lru_cache

INF = None


def point(d):
    return {int(d): F(1)}


def truncated_poisson(mean: int, m: int):
    """Poisson probabilities on 0..m, renormalised. They are rational for an integer mean."""
    w, cur = [], F(1)
    for k in range(m + 1):
        if k:
            cur = cur * mean / k
        w.append(cur)
    s = sum(w)
    return {k: w[k] / s for k in range(m + 1)}


class Problem:
    """Two locations, T in {1, 2}. pmf[j][t] is a dict demand -> probability for location j and period t (0-based)."""

    def __init__(self, pmf1, pmf2, K, R, z, v, h, b, cap=100):
        self.pmf = (pmf1, pmf2)
        self.T = len(pmf1)
        assert self.T in (1, 2) and len(pmf2) == self.T
        self.K, self.R, self.z, self.v, self.h, self.b = map(F, (K, R, z, v, h, b))
        self.cap = cap
        self.mu = [[sum(d * p for d, p in pm[t].items()) for t in range(self.T)] for pm in self.pmf]
        self._ell = {}
        self._last = {}
        self._S = {}
        self._Clast = {}
        self._A = {}
        self.best_bound_margin = None

    # ---------------------------------------------------------------- one-period pieces
    def ell(self, j, t, y):
        key = (j, t, y)
        if key not in self._ell:
            h, b = self.h, self.b
            self._ell[key] = sum(p * (h * (y - d) if y >= d else b * (d - y)) for d, p in self.pmf[j][t].items())
        return self._ell[key]

    def f(self, j, y):
        return self.z * y + self.ell(j, self.T - 1, y)

    def S_last(self, j):
        """Smallest minimiser of the convex function f^j. f^j decreases on y <= 0 because z < b."""
        if j not in self._S:
            y = 0
            while self.f(j, y + 1) < self.f(j, y):
                y += 1
            self._S[j] = y
        return self._S[j]

    def c_last(self, j, x):
        """Second stage of the last period for one location: (cost, level after reordering)."""
        key = (j, x)
        if key not in self._last:
            S = self.S_last(j)
            stay = self.ell(j, self.T - 1, x)
            if x < S and self.f(j, x) > self.K + self.f(j, S):
                self._last[key] = (self.K + self.z * (S - x) + self.ell(j, self.T - 1, S), S)
            else:
                self._last[key] = (stay, x)
        return self._last[key]

    def Cq_last(self, x1, x2):
        return self.c_last(0, x1)[0] + self.c_last(1, x2)[0]

    def first_stage(self, Cq, i1, i2):
        """(value, list of optimal W) of min over W of u(|W|) + Cq(i1 - W, i2 + W)."""
        best, arg = None, []
        for W in range(-max(0, i2), max(0, i1) + 1):
            c = Cq(i1 - W, i2 + W) + (self.R + self.v * abs(W) if W else 0)
            if best is None or c < best:
                best, arg = c, [W]
            elif c == best:
                arg.append(W)
        return best, arg

    def C_last(self, i1, i2):
        key = (i1, i2)
        if key not in self._Clast:
            self._Clast[key] = self.first_stage(self.Cq_last, i1, i2)[0]
        return self._Clast[key]

    # ---------------------------------------------------------------- period 1 of a two-period problem
    def A(self, y1, y2):
        """L_1(y) + E[C_2(y - d_1)] ."""
        key = (y1, y2)
        if key not in self._A:
            e = F(0)
            for d1, p1 in self.pmf[0][0].items():
                for d2, p2 in self.pmf[1][0].items():
                    e += p1 * p2 * self.C_last(y1 - d1, y2 - d2)
            self._A[key] = self.ell(0, 0, y1) + self.ell(1, 0, y2) + e
        return self._A[key]

    def H1(self, y1, y2):
        return self.z * (y1 + y2) + self.A(y1, y2)

    def second_stage_1(self, x1, x2):
        """Second stage of period 1 at the post-transshipment state x.
        Returns {pattern: (cost, y1, y2)} for the patterns '00', '10', '01', '11' (1 = the location reorders)."""
        assert self.T == 2
        K, z, Y = self.K, self.z, self.cap
        out = {"00": (self.A(x1, x2), x1, x2)}
        b10 = min(((K + z * (y1 - x1) + self.A(y1, x2), y1, x2) for y1 in range(x1 + 1, Y + 1)), key=lambda e: e[0])
        b01 = min(((K + z * (y2 - x2) + self.A(x1, y2), x1, y2) for y2 in range(x2 + 1, Y + 1)), key=lambda e: e[0])
        b11 = min(((2 * K + z * (y1 - x1) + z * (y2 - x2) + self.A(y1, y2), y1, y2)
                   for y1 in range(x1 + 1, Y + 1) for y2 in range(x2 + 1, Y + 1)), key=lambda e: e[0])
        out.update({"10": b10, "01": b01, "11": b11})
        best = min(e[0] for e in out.values())
        tot = sum(self.mu[j][t] for j in (0, 1) for t in (0, 1))
        for j, (xj, xk) in enumerate(((x1, x2), (x2, x1))):
            bound = z * (Y + 1 - xj) + self.h * (Y + 1 - self.mu[j][0]) + self.h * (Y + 1 + xk - tot)
            assert bound > best, ("cap too small", x1, x2, float(bound), float(best))
            m = bound - best
            self.best_bound_margin = m if self.best_bound_margin is None else min(self.best_bound_margin, m)
        return out

    @lru_cache(maxsize=None)
    def Cq1(self, x1, x2):
        return min(e[0] for e in self.second_stage_1(x1, x2).values())

    def reorder_split(self, x1, x2, loc):
        """(cost of the best decision in which `loc` does not reorder, cost of the best in which it reorders,
        decision without, decision with)."""
        s = self.second_stage_1(x1, x2)
        a, bb = ("00", "01"), ("10", "11")
        if loc == 2:
            a, bb = ("00", "10"), ("01", "11")
        no = min((s[k] for k in a), key=lambda e: e[0])
        yes = min((s[k] for k in bb), key=lambda e: e[0])
        return no, yes


def fmt(x, nd=4):
    return f"{float(x):.{nd}f}"


def frac(x):
    x = F(x)
    return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"


# ====================================================================== examples
def threshold_table(P, x2, xs, label):
    print(label)
    print("  x1 | without reorder at 1 (levels) | with reorder at 1 (levels) | difference | location 1")
    signs = []
    for x1 in xs:
        no, yes = P.reorder_split(x1, x2, 1)
        diff = no[0] - yes[0]
        signs.append((x1, diff))
        print(f"  {x1:3d} | {fmt(no[0])} = {frac(no[0]) if no[0].denominator <= 4 else '...'} ({no[1]},{no[2]}) | "
              f"{fmt(yes[0])} ({yes[1]},{yes[2]}) | {fmt(diff)} "
              f"{'= ' + frac(diff) if diff.denominator <= 4 else ''} | "
              f"{'reorders' if diff > 0 else ('does not reorder' if diff < 0 else 'TIE')}")
    return signs


def no_transshipment_check(P, x2, xs):
    """Opening states (i1, x2): is W = 0 the unique optimal transshipment in period 1?"""
    res = []
    for i1 in xs:
        val, arg = P.first_stage(P.Cq1, i1, x2)
        res.append((i1, arg))
    return res


def deterministic_threshold(K=45, verbose=True):
    P = Problem([point(1), point(30)], [point(1), point(12)], K=K, R=2, z=1, v=F(1, 2), h=1, b=20, cap=100)
    if verbose:
        signs = threshold_table(P, 3, range(-2, 14), f"\n== Deterministic demand (1,30) and (1,12), K = {K}: period 1, states (x1, 3)")
        print("  optimal transshipments at the opening states (i1, 3):", no_transshipment_check(P, 3, range(-2, 14)))
        print("  smallest margin of the cap bound:", fmt(P.best_bound_margin, 2))
    else:
        signs = [(x1, (lambda ny: ny[0][0] - ny[1][0])(P.reorder_split(x1, 3, 1))) for x1 in range(-2, 14)]
    return P, signs


def poisson_threshold():
    pm1 = [truncated_poisson(1, 6), truncated_poisson(30, 52)]
    pm2 = [truncated_poisson(1, 6), truncated_poisson(12, 27)]
    P = Problem(pm1, pm2, K=57, R=2, z=1, v=F(1, 2), h=1, b=20, cap=100)
    signs = threshold_table(P, 3, range(0, 11),
                            "\n== Truncated Poisson demand, means (1,30) and (1,12), support up to 6, 52, 6 and 27, K = 57: period 1, states (x1, 3)")
    print("  optimal transshipments at the opening states (i1, 3):", no_transshipment_check(P, 3, range(-1, 12)))
    print("  smallest margin of the cap bound:", fmt(P.best_bound_margin, 2))
    return P, signs


def violates_threshold(signs):
    """True if some state strictly reorders above a state that strictly does not."""
    no = [x for x, d in signs if d < 0]
    yes = [x for x, d in signs if d > 0]
    return bool(no and yes and min(no) < max(yes))


def one_period_example():
    """T = 1, deterministic demand 3 at both locations, K = 20, R = 2, z = 1, v = 1/2, h = 1, b = 20."""
    P = Problem([point(3)], [point(3)], K=20, R=2, z=1, v=F(1, 2), h=1, b=20)
    print("\n== One period, deterministic demand (3) and (3), K = 20, R = 2, b = 20")
    print("  second stage of one location, c(x), x = -3..7:", [frac(P.c_last(0, x)[0]) for x in range(-3, 8)])
    print("  reorder-up-to level and levels at which the location reorders:", P.S_last(0),
          [x for x in range(-3, 8) if P.c_last(0, x)[1] != x])
    n = 3
    print(f"  line n = {n}:  i1 | G(i1) | value of the first stage | optimal W")
    for i1 in range(-4, 9):
        val, arg = P.first_stage(P.Cq_last, i1, n - i1)
        print(f"     {i1:3d} | {frac(P.Cq_last(i1, n - i1)):>5} | {frac(val):>5} | {arg}")
    a, b0, c = P.C_last(-1, 0), P.C_last(0, 0), P.C_last(6, 0)
    print("  C(-1,0), C(0,0), C(6,0) =", frac(a), frac(b0), frac(c),
          "   K + C(6,0) - C(0,0) - 6 (C(0,0) - C(-1,0)) =", frac(P.K + c - b0 - 6 * (b0 - a)))
    return P


def two_period_details():
    P, _ = deterministic_threshold(F(89, 2))
    print("  period 2: state | value | optimal W | levels after reordering")
    for i in [(0, 2), (3, 2), (8, 2), (9, 2), (10, 2), (40, 2)]:
        val, arg = P.first_stage(P.Cq_last, *i)
        W = arg[0]
        x = (i[0] - W, i[1] + W)
        print(f"     {i} | {frac(val)} | {arg} | ({P.c_last(0, x[0])[1]}, {P.c_last(1, x[1])[1]})")
    print("  g(y1) = H_1(y1, 3) or K + min over y2 > 3 of H_1(y1, y2), whichever is smaller, y1 = 0..12 and 41:")
    for y1 in list(range(0, 13)) + [40, 41, 42]:
        s = P.second_stage_1(y1, 3)
        g = min(s["00"][0], s["01"][0]) + P.z * (y1 + 3)
        print(f"     g({y1}) = {frac(g)}")
    print("  minimisers of H_1(., x2) over y1 = -5..100, the level S^1_1(x2):")
    for x2 in range(2, 15):
        vals = {y1: P.H1(y1, x2) for y1 in range(-5, P.cap + 1)}
        m = min(vals.values())
        print(f"     x2 = {x2:2d}: {[y for y, val in vals.items() if val == m]}   minimum {frac(m)}")
    # a tie: with K = 45 the two alternatives cost the same at the state (3, 3)
    P45 = Problem([point(1), point(30)], [point(1), point(12)], K=45, R=2, z=1, v=F(1, 2), h=1, b=20, cap=100)
    no, yes = P45.reorder_split(3, 3, 1)
    val, arg = P45.first_stage(P45.Cq1, 3, 3)
    print(f"\n== The same instance with K = 45, state (3, 3) of period 1: without reorder {frac(no[0])} at levels {no[1:]},"
          f" with reorder {frac(yes[0])} at levels {yes[1:]}; optimal transshipments {arg}, value {frac(val)}")
    return P


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "search":
        for K2 in range(60, 121):                      # K in steps of 1/2
            K = F(K2, 2)
            _, signs = deterministic_threshold(K, verbose=False)
            pat = "".join("R" if d > 0 else ("n" if d < 0 else "=") for _, d in signs)
            print(f"K = {float(K):5.1f}  x1 = -2..13: {pat}  {'VIOLATION' if violates_threshold(signs) else ''}")
        sys.exit()
    one_period_example()
    two_period_details()
    poisson_threshold()
