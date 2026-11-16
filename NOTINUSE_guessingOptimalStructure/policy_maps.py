"""
NOTINUSE_guessingOptimalStructure/policy_maps.py : maps of the optimal policy of SDP-2 over the state space, one column per period.

    python policy_maps.py        writes out/policy_<instance>.png and .pdf for the first-round instances

Rows of each figure
    1  opening state i = (i1, i2): the colour is the reorder pattern after the optimal transshipment, and the
       hatching marks the opening states from which a transshipment is made and its direction
    2  post-transshipment state x = (x1, x2): the colour is the reorder pattern of the second stage, and the
       dots are the inventory levels y = x + Q reached after reordering
    3  opening state i: the transshipment quantity W (positive from location 1 to 2), and the dots are the
       post-transshipment states reached by a transshipment

A transshipment is drawn only where it is strictly better than none (tolerance TOL of structure_tests.py).
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from common import FIRST_ROUND, OUT, Solution, solve
from structure_tests import reorder_pattern, transshipment

INK, INK2, RULE, SURFACE = "#0b0b0b", "#52514e", "#c9c7bf", "#fcfcfb"
PATTERN_COLOURS = ["#e6e4dd", "#2a78d6", "#eb6834", "#1baf7a"]          # none, location 1, location 2, both
PATTERN_LABELS = ["no reorder", "location 1 reorders", "location 2 reorders", "both reorder"]
DIVERGING = LinearSegmentedColormap.from_list("w", ["#104281", "#5598e7", "#cde2fb", "#f0efec",
                                                    "#f6c6c5", "#e34948", "#8a1a19"])

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": RULE, "axes.linewidth": 0.6,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK2, "text.color": INK,
                     "hatch.color": INK, "hatch.linewidth": 0.5, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})


def _edges(lev):
    return np.arange(lev[0] - 0.5, lev[-1] + 1.0)


def _axes(ax, lev, xlabel, ylabel, first):
    ax.set_aspect("equal")
    ax.set_xlim(lev[0] - 0.5, lev[-1] + 0.5)
    ax.set_ylim(lev[0] - 0.5, lev[-1] + 0.5)
    ax.axhline(0, color=INK2, lw=0.4, ls=(0, (2, 2)))
    ax.axvline(0, color=INK2, lw=0.4, ls=(0, (2, 2)))
    ax.set_xlabel(xlabel)
    if first:
        ax.set_ylabel(ylabel)
    ax.tick_params(length=2, width=0.5)


def figure(sol: Solution):
    lev = sol.levels
    e = _edges(lev)
    w = sol.win()
    T = sol.T
    fig, axs = plt.subplots(3, T, figsize=(3.05 * T + 0.6, 9.9), sharex=True, sharey=True, constrained_layout=True)
    cmap = ListedColormap(PATTERN_COLOURS)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], 4)
    Wall = [transshipment(sol, t) for t in range(T)]
    wmax = max(1, max(int(np.abs(W).max()) for W in Wall))
    wnorm = TwoSlopeNorm(vmin=-wmax, vcenter=0, vmax=wmax)
    im3 = None
    for t in range(T):
        W = Wall[t]
        pat_x = reorder_pattern(sol, t)                       # on the whole range, post-transshipment state
        x1 = lev[:, None] - W
        x2 = lev[None, :] + W
        pat_i = pat_x[x1 - sol.lo_w, x2 - sol.lo_w]           # pattern after the optimal transshipment
        # row 1
        ax = axs[0, t]
        ax.pcolormesh(e, e, pat_i.T, cmap=cmap, norm=norm, rasterized=True)
        for mask, hatch in ((W > 0, "////"), (W < 0, "\\\\\\\\")):
            if mask.any():
                # one hatched rectangle per run of cells in each row of the map
                for j in range(len(lev)):
                    col = np.flatnonzero(mask[:, j])
                    if not len(col):
                        continue
                    for run in np.split(col, np.flatnonzero(np.diff(col) > 1) + 1):
                        ax.add_patch(Rectangle((lev[run[0]] - 0.5, lev[j] - 0.5), len(run), 1.0, fill=False,
                                               hatch=hatch, lw=0, edgecolor=INK, alpha=0.55))
        ax.set_title(f"period {t + 1}:  expected demands ({sol.mu1[t]}, {sol.mu2[t]})", fontsize=8.5, color=INK)
        _axes(ax, lev, "", "opening inventory $i^2$", t == 0)
        # row 2
        ax = axs[1, t]
        ax.pcolormesh(e, e, pat_x[w, w].T, cmap=cmap, norm=norm, rasterized=True)
        Q = sol.Q[t][w, w]
        r = (Q[..., 0] > 0) | (Q[..., 1] > 0)
        y1 = (lev[:, None] + Q[..., 0])[r]
        y2 = (lev[None, :] + Q[..., 1])[r]
        pts = np.unique(np.stack([y1, y2], 1), axis=0)
        ax.scatter(pts[:, 0], pts[:, 1], s=7, c=INK, edgecolors=SURFACE, linewidths=0.4, zorder=3)
        _axes(ax, lev, "", "post-transshipment inventory $\\tilde i^2$", t == 0)
        # row 3
        ax = axs[2, t]
        im3 = ax.pcolormesh(e, e, W.T, cmap=DIVERGING, norm=wnorm, rasterized=True)
        m = W != 0
        if m.any():
            tg = np.unique(np.stack([x1[m], x2[m]], 1), axis=0)
            ax.scatter(tg[:, 0], tg[:, 1], s=6, c=INK, edgecolors=SURFACE, linewidths=0.35, zorder=3)
        _axes(ax, lev, "inventory at location 1", "opening inventory $i^2$", t == 0)
    handles = [Patch(facecolor=c, edgecolor=RULE, lw=0.5, label=l) for c, l in zip(PATTERN_COLOURS, PATTERN_LABELS)]
    handles += [Patch(facecolor="none", edgecolor=INK, hatch="////", lw=0.5, label="transshipment from 1 to 2"),
                Patch(facecolor="none", edgecolor=INK, hatch="\\\\\\\\", lw=0.5, label="transshipment from 2 to 1"),
                Line2D([], [], marker="o", ls="", ms=3.5, mfc=INK, mec=SURFACE, mew=0.4,
                       label="row 2: levels after reordering;  row 3: states after a transshipment")]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False, fontsize=8)
    cb = fig.colorbar(im3, ax=axs[2, :], shrink=0.85, pad=0.01, aspect=30)
    cb.set_label("transshipment quantity $W_t$  (positive: from location 1 to 2)", color=INK2)
    cb.outline.set_linewidth(0.4)
    c = sol.inst.costs
    fig.suptitle(f"Optimal policy of SDP-2.  {sol.inst.label},  z = {c['z']:g}, v = {c['v']:g}, h = {c['h']:g}, b = {c['b']:g}\n"
                 "Row 1: opening state, reorder pattern after the optimal transshipment.   "
                 "Row 2: second stage on the post-transshipment state.   Row 3: transshipment quantity.",
                 fontsize=9.5, color=INK)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"policy_{sol.inst.name}.{ext}"), dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    for inst in FIRST_ROUND:
        figure(solve(inst))
        print("figure", inst.name)
