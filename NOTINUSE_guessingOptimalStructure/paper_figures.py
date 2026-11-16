"""
NOTINUSE_guessingOptimalStructure/paper_figures.py : the levels of the NOTINUSE_guessingOptimalStructure theorem drawn on the policy maps.

    python paper_figures.py      writes paper/fig_levels_<instance>.pdf and .png

Left panel, post-transshipment state: reorder regions of the second stage, the point S_t (both reorder), the curve
S^1_t(x2) of location 1 and the curve S^2_t(x1) of location 2. A level is drawn only where the policy uses it:
S^j_t where location j reorders alone for some state, D^+_t(n) and D^-_t(n) on the lines with a transshipment
in that direction.
Right panel, opening state: transshipment regions of the first stage and the two transshipment levels
D^+_t(n) and D^-_t(n) of every line i1 + i2 = n.
All levels are computed from H_t and from the second-stage value as in the theorem, not read from the policy.
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from common import FIRST_ROUND, HERE, solve
from policy_maps import INK, INK2, PATTERN_COLOURS, PATTERN_LABELS, RULE, SURFACE, _axes, _edges
from structure_tests import reorder_pattern, transshipment

PAPER = os.path.join(HERE, "paper")
os.makedirs(PAPER, exist_ok=True)
SHIP_COLOURS = ["#cde2fb", "#f0efec", "#f6c6c5"]            # 2 to 1, none, 1 to 2


def levels(sol, t):
    w, lev, lo = sol.win(), sol.levels, sol.lo
    m = len(lev)
    v = sol.inst.costs["v"]
    Hw, Cq = sol.H[t][w, w], sol.Cq[t][w, w]
    k = np.unravel_index(int(np.argmin(Hw)), Hw.shape)
    Sbar = (lev[k[0]], lev[k[1]])
    S1 = lev[np.argmin(Hw, axis=0)]
    S2 = lev[np.argmin(Hw, axis=1)]
    Dp, Dm = [], []
    for n in range(2 * lo, 2 * sol.hi + 1):
        a = np.arange(max(0, n - 2 * lo - (m - 1)), min(m - 1, n - 2 * lo) + 1)
        b = n - 2 * lo - a
        i1, G = lev[a], Cq[a, b]
        dp, dm = np.flatnonzero(i1 >= 0), np.flatnonzero(i1 <= n)
        if len(dp):
            x = i1[dp[int(np.argmin((G - v * i1)[dp]))]]
            Dp.append((x, n - x))
        if len(dm):
            x = i1[dm[int(np.argmin((G + v * i1)[dm]))]]
            Dm.append((x, n - x))
    # keep each level only where the policy uses it
    pat = reorder_pattern(sol, t)[w, w]
    S1 = np.where((pat == 1).any(axis=0), S1, np.nan).astype(float)
    S2 = np.where((pat == 2).any(axis=1), S2, np.nan).astype(float)
    W = transshipment(sol, t)
    tot = lev[:, None] + lev[None, :]
    used_p = set(int(x) for x in tot[W > 0])
    used_m = set(int(x) for x in tot[W < 0])
    Dp = np.array([p for p in Dp if p[0] + p[1] in used_p], dtype=float)
    Dm = np.array([p for p in Dm if p[0] + p[1] in used_m], dtype=float)
    return Sbar, S1, S2, Dp, Dm


def figure(sol, t=0):
    lev = sol.levels
    e = _edges(lev)
    w = sol.win()
    Sbar, S1, S2, Dp, Dm = levels(sol, t)
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 5.0), constrained_layout=True)
    ax = axs[0]
    ax.pcolormesh(e, e, reorder_pattern(sol, t)[w, w].T, cmap=ListedColormap(PATTERN_COLOURS),
                  norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], 4), rasterized=True)
    ax.plot(S1, lev, color=INK, lw=1.4)
    ax.plot(lev, S2, color=INK, lw=1.4, ls=(0, (4, 2)))
    ax.plot([Sbar[0]], [Sbar[1]], marker="*", ms=13, mfc="#ffffff", mec=INK, mew=1.0, ls="")
    _axes(ax, lev, "post-transshipment inventory $\\tilde i^1$", "post-transshipment inventory $\\tilde i^2$", True)
    ax.set_title("Second stage: reorder regions and reorder-up-to levels", fontsize=9, color=INK)
    h = [Patch(facecolor=c, edgecolor=RULE, lw=0.5, label=l) for c, l in zip(PATTERN_COLOURS, PATTERN_LABELS)]
    h += [Line2D([], [], color=INK, lw=1.4, label="$S^1_t(\\tilde i^2)$"),
          Line2D([], [], color=INK, lw=1.4, ls=(0, (4, 2)), label="$S^2_t(\\tilde i^1)$"),
          Line2D([], [], marker="*", ms=11, mfc="#ffffff", mec=INK, ls="", label="$\\bar S_t$")]
    ax.legend(handles=h, loc="upper right", fontsize=7, frameon=True, framealpha=0.92, edgecolor=RULE)
    ax = axs[1]
    W = transshipment(sol, t)
    ax.pcolormesh(e, e, np.sign(W).T, cmap=ListedColormap(SHIP_COLOURS), norm=BoundaryNorm([-1.5, -0.5, 0.5, 1.5], 3),
                  rasterized=True)
    ax.plot(Dp[:, 0], Dp[:, 1], color="#8a1a19", lw=1.4)
    ax.plot(Dm[:, 0], Dm[:, 1], color="#104281", lw=1.4, ls=(0, (4, 2)))
    _axes(ax, lev, "opening inventory $i^1$", "opening inventory $i^2$", True)
    ax.set_title("First stage: transshipment regions and transshipment levels", fontsize=9, color=INK)
    h = [Patch(facecolor=SHIP_COLOURS[2], edgecolor=RULE, lw=0.5, label="transshipment from 1 to 2"),
         Patch(facecolor=SHIP_COLOURS[0], edgecolor=RULE, lw=0.5, label="transshipment from 2 to 1"),
         Patch(facecolor=SHIP_COLOURS[1], edgecolor=RULE, lw=0.5, label="no transshipment"),
         Line2D([], [], color="#8a1a19", lw=1.4, label="$D^+_t(n)$"),
         Line2D([], [], color="#104281", lw=1.4, ls=(0, (4, 2)), label="$D^-_t(n)$")]
    ax.legend(handles=h, loc="upper right", fontsize=7, frameon=True, framealpha=0.92, edgecolor=RULE)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(PAPER, f"fig_levels_{sol.inst.name}.{ext}"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    for k in (0, 1):
        figure(solve(FIRST_ROUND[k]), t=0)
        print("figure", FIRST_ROUND[k].name)
