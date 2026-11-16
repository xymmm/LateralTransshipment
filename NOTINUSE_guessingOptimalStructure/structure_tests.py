"""
NOTINUSE_guessingOptimalStructure/structure_tests.py : numerical tests of the NOTINUSE_guessingOptimalStructure of the optimal policy of SDP-2.

    python structure_tests.py            runs the first-round instances and writes out/tests_<instance>.json
                                         and out/tests_summary.md

Every test is run on the analysis window [lo, hi]^2 (see common.py) and is tie-aware: an action is called
strictly optimal when the best alternative without it is worse by more than TOL.

Second stage, C~~_t(x) on the post-transshipment state x
    S1  threshold in the own inventory: for a fixed x2, if location 1 strictly reorders at x1 it does not strictly
        prefer not to reorder at any lower x1 (and the same for location 2)
    S2  the threshold s^1_t(x2) and the reorder-up-to level S^1_t(x2) as functions of the inventory of the other
        location, with the single-location (s, S) of the problem without transshipment for comparison
First stage, C~_t(i) on the opening state i, along each line i1 + i2 = n
    F1  the states without transshipment form one interval of the line
    F2  transshipments from 2 to 1 lie below this interval in i1 and transshipments from 1 to 2 above it
    F3  number of distinct post-transshipment states reached per line and direction
    F4  actions that occur together: transshipment direction x reorder pattern
K-convexity
    minimal constant kappa for which f(x + a) + kappa >= f(x) + (a / b) (f(x) - f(x - b)) for all x, a > 0, b > 0
    along a lattice direction d. f is K(d)-convex along d when kappa <= K(d), with K(d) = K times the number of
    positive components of d (K-convexity in R^n of Gallego and Sethi with K_0 = 0, K_1 = K_2 = K).
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from common import FIRST_ROUND, OUT, Solution, single_location, solve

TOL = 1e-8
DIRECTIONS = [(1, 0), (0, 1), (1, 1), (1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2), (1, 4), (4, 1)]


# ---------------------------------------------------------------------------------------------------------
# second stage
# ---------------------------------------------------------------------------------------------------------
def _above(A, axis):
    """min of A over indices strictly above along axis."""
    B = np.flip(np.minimum.accumulate(np.flip(A, axis), axis=axis), axis)
    out = np.full_like(A, np.inf)
    if axis == 0:
        out[:-1, :] = B[1:, :]
    else:
        out[:, :-1] = B[:, 1:]
    return out


def second_stage_values(sol: Solution, t: int):
    """Values of the four reorder patterns at every post-transshipment state of the range:
    v[0] none, v[1] location 1 only, v[2] location 2 only, v[3] both."""
    K, z = sol.inst.costs["K"], sol.inst.costs["z"]
    H = sol.H[t]
    lev = np.arange(sol.lo_w, sol.hi_w + 1)
    base = z * (lev[:, None] + lev[None, :])
    return np.stack([H - base, K + _above(H, 0) - base, K + _above(H, 1) - base,
                     2 * K + _above(_above(H, 0), 1) - base])


def reorder_pattern(sol: Solution, t: int):
    """Recorded reorder pattern of SDP-2 at the post-transshipment state: 0 none, 1 location 1, 2 location 2, 3 both."""
    Q = sol.Q[t]
    return (Q[..., 0] > 0) * 1 + (Q[..., 1] > 0) * 2


def test_second_stage(sol: Solution, t: int):
    w = sol.win()
    lev = sol.levels
    v = second_stage_values(sol, t)[:, w, w]
    best = v.min(0)
    assert np.abs(best - sol.Cq[t][w, w]).max() < 1e-9
    out = {}
    for loc in (1, 2):
        with_r = np.minimum(v[loc], v[3])                 # location loc reorders
        without = np.minimum(v[0], v[3 - loc])            # it does not
        strict_r = with_r < without - TOL
        strict_n = without < with_r - TOL
        if loc == 2:                                      # put the own inventory on axis 0
            strict_r, strict_n = strict_r.T, strict_n.T
        # S1: no strict "do not reorder" below a strict "reorder" in the own inventory
        viol = []
        thr = []
        for k in range(len(lev)):                         # k indexes the inventory of the other location
            r, n = np.flatnonzero(strict_r[:, k]), np.flatnonzero(strict_n[:, k])
            thr.append(int(lev[r.max()]) if len(r) else None)
            if len(r) and len(n) and n.min() < r.max():
                viol.append(dict(other=int(lev[k]), not_reorder_at=int(lev[n.min()]), reorder_at=int(lev[r.max()])))
        # reorder-up-to level where only this location reorders, from the recorded policy
        pat = reorder_pattern(sol, t)[w, w]
        y = (lev[:, None] + sol.Q[t][w, w][..., 0]) if loc == 1 else (lev[None, :] + sol.Q[t][w, w][..., 1])
        if loc == 2:
            pat, y = pat.T, y.T
        S_only, S_both = [], set()
        for k in range(len(lev)):
            vals = sorted(set(int(a) for a in y[pat[:, k] == loc, k]))
            S_only.append(vals)
        out[f"loc{loc}"] = dict(threshold_violations=viol, threshold=thr, S_only=S_only)
    pat = reorder_pattern(sol, t)[w, w]
    y1 = lev[:, None] + sol.Q[t][w, w][..., 0]
    y2 = lev[None, :] + sol.Q[t][w, w][..., 1]
    out["S_both"] = sorted(set((int(a), int(b)) for a, b in zip(y1[pat == 3], y2[pat == 3])))
    out["pattern_counts"] = [int((pat == c).sum()) for c in range(4)]
    return out


def single_sS(sol: Solution, loc: int, t: int):
    s = single_location(sol, loc)
    lev = s.grid
    r = np.flatnonzero(s.Q[t] > 0)
    r = r[(lev[r] >= sol.lo)]
    if not len(r):
        return None, None
    i = r.max()
    return int(lev[i]), int(lev[i] + s.Q[t][i])


# ---------------------------------------------------------------------------------------------------------
# first stage
# ---------------------------------------------------------------------------------------------------------
def transshipment(sol: Solution, t: int):
    """W on the window, set to 0 where the transshipment is not strictly better than none."""
    w = sol.win()
    strict = sol.C[t][w, w] < sol.Cq[t][w, w] - TOL
    return np.where(strict, sol.W[t][w, w], 0)


def test_first_stage(sol: Solution, t: int):
    lev = sol.levels
    m = len(lev)
    W = transshipment(sol, t)
    w = sol.win()
    Cq = sol.Cq[t][w, w]
    F1, F2 = [], []
    n_lines = 0
    targets = {"+": [], "-": []}
    multi = {"+": [], "-": []}
    band = []
    max_local_min = 0
    for n in range(2 * sol.lo, 2 * sol.hi + 1):
        a = np.arange(max(0, n - sol.lo - (m - 1) - sol.lo), min(m - 1, n - 2 * sol.lo) + 1)   # index of i1
        b = n - 2 * sol.lo - a                                                               # index of i2
        if len(a) < 3:
            continue
        n_lines += 1
        Wl = W[a, b]
        i1 = lev[a]
        none, pos, neg = np.flatnonzero(Wl == 0), np.flatnonzero(Wl > 0), np.flatnonzero(Wl < 0)
        if len(none):
            inside = np.flatnonzero(Wl[none.min():none.max() + 1] != 0)
            if len(inside):
                k = none.min() + inside[0]
                F1.append(dict(n=int(n), state=(int(i1[k]), int(n - i1[k])), W=int(Wl[k]),
                               no_transshipment_at=(int(i1[none.min()]), int(i1[none.max()]))))
            band.append((int(n), int(i1[none.min()]), int(i1[none.max()])))
        lo_none = none.min() if len(none) else (pos.min() if len(pos) else len(a))
        hi_none = none.max() if len(none) else (neg.max() if len(neg) else -1)
        if (len(neg) and neg.max() > lo_none) or (len(pos) and pos.min() < hi_none):
            F2.append(int(n))
        for key, idx in (("+", pos), ("-", neg)):
            if len(idx):
                tg = sorted(set(int(x) for x in (i1[idx] - Wl[idx])))
                targets[key].append(len(tg))
                if len(tg) > 1:
                    multi[key].append(dict(n=int(n), post_x1=tg))
        g = Cq[a, b]
        strict_min = (g[1:-1] < g[:-2] - TOL) & (g[1:-1] < g[2:] - TOL)
        max_local_min = max(max_local_min, int(strict_min.sum()))
    # actions that occur together on the window
    x1 = lev[:, None] - W - sol.lo_w
    x2 = lev[None, :] + W - sol.lo_w
    pat = reorder_pattern(sol, t)[x1, x2]
    direction = np.sign(W)
    table = {f"{d:+d}|{p}": int(((direction == d) & (pat == p)).sum()) for d in (-1, 0, 1) for p in range(4)}
    return dict(lines=n_lines, F1_violations=F1, F2_violations=F2,
                targets_lines={k: len(v) for k, v in targets.items()},
                targets_single={k: int(sum(1 for c in v if c == 1)) for k, v in targets.items()},
                targets_max={k: (max(v) if v else 0) for k, v in targets.items()},
                targets_multi_examples={k: v[:3] for k, v in multi.items()},
                max_strict_local_minima_on_a_line=max_local_min, together=table,
                n_transship=int((W != 0).sum()), W_min=int(W.min()), W_max=int(W.max()))


# ---------------------------------------------------------------------------------------------------------
# K-convexity
# ---------------------------------------------------------------------------------------------------------
def kappa_lines(F):
    """Minimal constant of K-convexity of the columns of F (lines along axis 0), with the worst triple.
    Returns (kappa, (x, a, b, column))."""
    m = F.shape[0]
    s = np.full(F.shape, -np.inf)
    sb = np.zeros(F.shape, dtype=int)
    for b in range(1, m):
        cand = (F[b:] - F[:-b]) / b
        better = cand > s[b:]
        s[b:] = np.where(better, cand, s[b:])
        sb[b:] = np.where(better, b, sb[b:])
    best, arg = -np.inf, None
    for a in range(1, m - 1):
        val = F[1:m - a] - F[1 + a:] + a * s[1:m - a]          # x = 1 .. m - a - 1
        k = np.unravel_index(int(np.argmax(val)), val.shape)
        if val[k] > best:
            x, col = k[0] + 1, (k[1] if val.ndim > 1 else 0)
            best, arg = float(val[k]), (int(x), int(a), int(sb[x, col] if val.ndim > 1 else sb[x]), int(col))
    return best, arg


def kappa_direction(F, d, lev):
    """Minimal constant along the lattice direction d on the window, with the worst triple of states."""
    d1, d2 = d
    m = F.shape[0]
    if d == (1, 0):
        k, (x, a, b, col) = kappa_lines(F)
        return k, dict(x=(int(lev[x]), int(lev[col])), a=a, b=b)
    if d == (0, 1):
        k, (x, a, b, col) = kappa_lines(F.T)
        return k, dict(x=(int(lev[col]), int(lev[x])), a=a, b=b)
    best, arg = -np.inf, None
    for i0 in range(m):
        for j0 in range(m):
            if 0 <= i0 - d1 < m and 0 <= j0 - d2 < m:
                continue                                         # not the start of a line
            L = min((m - 1 - i0) // d1 if d1 > 0 else (i0 // -d1 if d1 < 0 else 10 ** 9),
                    (m - 1 - j0) // d2 if d2 > 0 else (j0 // -d2 if d2 < 0 else 10 ** 9)) + 1
            if L < 3:
                continue
            idx = np.arange(L)
            f = F[i0 + idx * d1, j0 + idx * d2]
            k, (x, a, b, _) = kappa_lines(f[:, None])
            if k > best:
                best, arg = k, dict(x=(int(lev[i0 + x * d1]), int(lev[j0 + x * d2])), a=a, b=b)
    return best, arg


def test_k_convexity(sol: Solution, t: int):
    w = sol.win()
    lev = sol.levels
    K = sol.inst.costs["K"]
    out = {}
    for name, F in (("H", sol.H[t][w, w]), ("Cq", sol.Cq[t][w, w]), ("C", sol.C[t][w, w])):
        res = {}
        for d in DIRECTIONS:
            Kd = K * ((d[0] > 0) + (d[1] > 0))
            k, arg = kappa_direction(F, d, lev)
            x, a, b = arg["x"], arg["a"], arg["b"]
            pts = [(x[0] - b * d[0], x[1] - b * d[1]), x, (x[0] + a * d[0], x[1] + a * d[1])]
            vals = [float(F[p[0] - sol.lo, p[1] - sol.lo]) for p in pts]
            res[f"{d[0]},{d[1]}"] = dict(kappa=k, K_d=Kd, holds=bool(k <= Kd + TOL), points=pts, values=vals, a=a, b=b)
        out[name] = res
    # the transshipment direction, both orientations, for information
    for d in ((1, -1), (-1, 1)):
        k, arg = kappa_direction(sol.Cq[t][w, w], d, lev)
        out[f"Cq_along_{d[0]},{d[1]}"] = dict(kappa=k, R=sol.inst.costs["R"], **arg)
    return out


# ---------------------------------------------------------------------------------------------------------
def run(inst):
    sol = solve(inst)
    res = dict(instance=inst.name, label=inst.label, mu1=sol.mu1, mu2=sol.mu2, window=[sol.lo, sol.hi],
               range=[sol.lo_w, sol.hi_w], D=sol.D, C1_00=float(sol.C[0][-sol.lo_w, -sol.lo_w]), periods=[])
    for t in range(sol.T):
        r = dict(t=t + 1, second=test_second_stage(sol, t), first=test_first_stage(sol, t),
                 kconv=test_k_convexity(sol, t),
                 single=[single_sS(sol, 1, t), single_sS(sol, 2, t)])
        res["periods"].append(r)
    # single-location value functions as a check of the K-convexity routine (they must be K-convex)
    chk = []
    for loc in (1, 2):
        s = single_location(sol, loc)
        for t in range(sol.T):
            k, _ = kappa_lines(s.C[t][sol.win()][:, None])
            chk.append(k)
    res["single_location_kappa_max"] = float(max(chk))
    with open(os.path.join(OUT, f"tests_{inst.name}.json"), "w") as f:
        json.dump(res, f, indent=1)
    return res


def _mono(seq):
    v = [x for x in seq if x is not None]
    up = sum(1 for a, b in zip(v, v[1:]) if b > a)
    down = sum(1 for a, b in zip(v, v[1:]) if b < a)
    return up, down


def summarise(results):
    L = ["# Structure tests, first round", "",
         "Generated by structure_tests.py. Every number is computed on the analysis window of the instance.", ""]
    for res in results:
        K = None
        L += [f"## {res['label']}", "",
              f"Expected demands {res['mu1']} and {res['mu2']}; window {res['window']}; "
              f"C_1(0,0) = {res['C1_00']:.4f}; check of the routine on the single-location values: "
              f"largest kappa {res['single_location_kappa_max']:.4f}.", ""]
        L += ["| t | single (s,S) loc 1 / loc 2 | S1 viol. | threshold s1(x2): range, ups, downs | S_only loc 1: range | "
              "threshold s2(x1): range, ups, downs | S_only loc 2: range | S_both | F1 viol. | F2 viol. | "
              "lines with one target 1->2 | 2->1 | max targets | local minima |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for p in res["periods"]:
            s, f = p["second"], p["first"]
            cells = [str(p["t"]), f"{tuple(p['single'][0])} / {tuple(p['single'][1])}"]
            nv = len(s["loc1"]["threshold_violations"]) + len(s["loc2"]["threshold_violations"])
            cells.append(str(nv))
            for loc in ("loc1", "loc2"):
                thr = [x for x in s[loc]["threshold"] if x is not None]
                up, down = _mono(s[loc]["threshold"])
                So = [y for v in s[loc]["S_only"] for y in v]
                cells.append(f"{min(thr)}..{max(thr)}, {up}, {down}" if thr else "never")
                cells.append(f"{min(So)}..{max(So)}" if So else "-")
            cells.append(str(s["S_both"][:3]) + ("..." if len(s["S_both"]) > 3 else ""))
            cells += [str(len(f["F1_violations"])), str(len(f["F2_violations"])),
                      f"{f['targets_single']['+']}/{f['targets_lines']['+']}",
                      f"{f['targets_single']['-']}/{f['targets_lines']['-']}",
                      str(max(f["targets_max"].values())), str(f["max_strict_local_minima_on_a_line"])]
            L.append("| " + " | ".join(cells) + " |")
        L += ["", "Actions that occur together on the window (number of opening states):", "",
              "| t | no transshipment: none / 1 / 2 / both reorder | 1->2: none / 1 / 2 / both | 2->1: none / 1 / 2 / both |",
              "|---|---|---|---|"]
        for p in res["periods"]:
            j = p["first"]["together"]
            L.append(f"| {p['t']} | " + " | ".join(" / ".join(str(j[f"{d:+d}|{c}"]) for c in range(4)) for d in (0, 1, -1)) + " |")
        L += ["", "Minimal constant kappa of K-convexity along each direction (holds when kappa <= K(d)):", "",
              "| t | function | " + " | ".join(f"({d[0]},{d[1]})" for d in DIRECTIONS) + " |",
              "|---|---|" + "---|" * len(DIRECTIONS)]
        for p in res["periods"]:
            for name in ("H", "Cq", "C"):
                row = []
                for d in DIRECTIONS:
                    e = p["kconv"][name][f"{d[0]},{d[1]}"]
                    row.append(f"{e['kappa']:.2f}" + ("" if e["holds"] else f" **> {e['K_d']:g}**"))
                L.append(f"| {p['t']} | {name} | " + " | ".join(row) + " |")
        L.append("")
    with open(os.path.join(OUT, "tests_summary.md"), "w") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    all_res = []
    for inst in FIRST_ROUND:
        r = run(inst)
        all_res.append(r)
        print("done", inst.name)
    summarise(all_res)
    print("written", os.path.join(OUT, "tests_summary.md"))
