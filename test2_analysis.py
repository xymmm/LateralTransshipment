"""
test2_analysis.py : tables and figures from the csv files written by test2_experiments.py.

    Q1  optimality gap of Algorithm 1
          table  test2_Q1_gap          by horizon and distribution (blocks A and Aplus), and by demand scale (block B)
          table  test2_T_matched       T = 4, 8, 12 on the 180 settings that block Aplus shares with block A
          table  test2_pivot           average gap pivoting each parameter, as Tables 1 and 2 of the 2022 version:
                                       horizon, distribution, demand pattern of location 1, identical or different
                                       patterns, K, R, b, (z, v); one column per horizon, and "all (T=4, 8)" over
                                       block A only, since block Aplus covers 5 of the 20 pattern pairs
          figure test2_box_pattern     box plots of the gap by demand pattern of location 1, one panel per horizon
          figure test2_box_cost        box plots of the gap by K, R, b and (z, v)
          figure test2_box_dist        box plots of the gap by distribution, one panel per horizon
    Q2  computing time
          table  test2_Q2_time         SDP-2 and one decision of Algorithm 1, by horizon and distribution (A, Aplus)
    Q3  computing time as the state space and the horizon grow (block C, full factorial)
          table  test2_Q3_time         SDP-2 complete solve and Algorithm 1 decisions for every distribution, state space
                                       level and T: wall time and CPU time
          table  test2_Q3_fit          fitted laws, log(time) regressed on log T and log |I| (SDP-2) and on T and
                                       log |I| (decision of period 1, MILP-1 over the whole horizon)
          figure test2_Q3_time         time against |I|, one line per T, SDP-2 and Algorithm 1 side by side
          figure test2_Q3_map          where each method fits a time budget in the (|I|, T) plane, from the fitted laws

|I| denotes the number of states (i1, i2) of SDP-2, i.e. the square of the number of inventory levels per location.
The comparison in Q2 and Q3 is between the complete SDP-2 solve, which must finish before the first decision can be
made, and the time Algorithm 1 needs to make the decision of one period; the simulation that estimates the expected
cost of Algorithm 1 serves the evaluation only and is not part of this comparison.

As in the 2022 version, the demand pattern is pivoted on location 1. The gap is 100 (ETC - C_1) / C_1.
Set RESULTS_DIR below and click Run, or: python test2_analysis.py --dir results_test2
Output goes to <dir>/tables (csv and booktabs LaTeX) and <dir>/figures (pdf and png).
Blocks whose csv does not exist yet are skipped, so the script can be run while the experiments are still running.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from test1_analysis import latex

RESULTS_DIR = "results_test2"
I_COL = "$|\\mathcal{I}|$"                 # number of states of SDP-2 = (inventory levels per location)^2
BUDGETS = [(60, "1 minute"), (3600, "1 hour"), (8 * 3600, "8 hours")]
PATTERN_ORDER = ["LCY1", "LCY2", "SIN1", "SIN2", "STAT", "RAND", "EMP1", "EMP2", "EMP3", "EMP4"]
DIST_ORDER = ["Poisson", "Normal, cv=0.1", "Normal, cv=0.2", "Normal, cv=0.3"]

FMT = {"instances": "{:.0f}", "$|\\mathcal{I}|$": "{:.0f}", "scale": "{:.0f}", "T": "{:.0f}",
       "mean gap (%)": "{:.3f}", "max gap (%)": "{:.3f}", "mean half-width (%)": "{:.3f}",
       "mean paired gap (%)": "{:.3f}", "max paired gap (%)": "{:.3f}",
       "T=4": "{:.3f}", "T=8": "{:.3f}", "T=12": "{:.3f}", "all (T=4, 8)": "{:.3f}",
       "SDP-2 (s)": "{:.2f}", "Algorithm 1, one decision (s)": "{:.3f}", "Algorithm 1, decision at k=1 (s)": "{:.3f}",
       "decisions per instance": "{:.0f}", "Algorithm 1, simulation (s)": "{:.1f}"}


def save(df, name, caption, out):
    df.to_csv(os.path.join(out, f"{name}.csv"), index=False)
    with open(os.path.join(out, f"{name}.tex"), "w") as f:
        f.write(latex(df, caption, f"tab:{name}", FMT))
    print(f"\n== {name}: {caption}")
    with pd.option_context("display.width", 220, "display.max_rows", 200):
        print(df.to_string(index=False))


def dist_label(r):
    return "Poisson" if r["dist"] == "poisson" else f"Normal, cv={r['cv']}"


def gap_row(keys, g):
    return {**keys, "instances": len(g),
            "mean gap (%)": g["gap_pct"].mean(), "max gap (%)": g["gap_pct"].max(),
            "mean half-width (%)": g["hw_pct"].mean(),
            "mean paired gap (%)": g["paired_gap_pct"].mean(), "max paired gap (%)": g["paired_gap_pct"].max()}


def time_row(keys, g):
    return {**keys, "instances": len(g), I_COL: (g["N"] ** 2).mean(),
            "SDP-2 (s)": g["seconds_sdp2"].mean(),
            "Algorithm 1, one decision (s)": g["seconds_per_decision"].mean(),
            "Algorithm 1, decision at k=1 (s)": g["seconds_per_decision_k1"].mean(),
            "decisions per instance": g["decisions_alg"].mean(),
            "Algorithm 1, simulation (s)": g["seconds_alg_total"].mean() if "seconds_alg_total" in g else np.nan}


def ordered(values, order):
    return [x for x in order if x in set(values)] + sorted(set(values) - set(order), key=str)


def pivot_table(A, AA):
    """Average gap for every level of every parameter: one column per horizon (blocks A and Aplus, AA) and one over
    block A only (A), whose design is balanced."""
    horizons = sorted(AA["T"].unique())
    params = [("horizon T", "T", sorted(A["T"].unique())),
              ("distribution", "distribution", ordered(A["distribution"].unique(), DIST_ORDER)),
              ("demand pattern of location 1", "pat1", ordered(A["pat1"].unique(), PATTERN_ORDER)),
              ("patterns of the two locations", "pair_type", ["identical", "different"]),
              ("fixed ordering cost K", "K", sorted(A["K"].unique())),
              ("fixed transshipment cost R", "R", sorted(A["R"].unique())),
              ("backorder cost b", "b", sorted(A["b"].unique())),
              ("unit costs (z, v)", "zv", ordered(A["zv"].unique(), [])),
              ]
    params[0] = ("horizon T", "T", horizons)
    rows = []
    for label, col, levels in params:
        for lev in levels:
            g, gA = AA[AA[col] == lev], A[A[col] == lev]
            if not len(g):
                continue
            r = {"parameter": label, "level": lev, "instances": len(g)}
            for T in horizons:
                gT = g[g["T"] == T]
                r[f"T={T}"] = gT["gap_pct"].mean() if len(gT) else np.nan
            r["all (T=4, 8)"] = gA["gap_pct"].mean() if len(gA) else np.nan
            rows.append(r)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=RESULTS_DIR)
    a = ap.parse_args()
    tab, fig = os.path.join(a.dir, "tables"), os.path.join(a.dir, "figures")
    os.makedirs(tab, exist_ok=True)
    os.makedirs(fig, exist_ok=True)

    def read(b):
        p = os.path.join(a.dir, f"{b}.csv")
        if not os.path.exists(p):
            return None
        d = pd.read_csv(p)
        if not len(d):
            return None
        d["distribution"] = d.apply(dist_label, axis=1)
        d["zv"] = "(" + d["z"].map("{:g}".format) + ", " + d["v"].map("{:g}".format) + ")"
        if "pair_type" not in d or d["pair_type"].isna().all():
            d["pair_type"] = np.where(d["pat1"] == d["pat2"], "identical", "different")
        return d

    A, B, Cb, Ap = read("A"), read("B"), read("C"), read("Aplus")
    AA = pd.concat([d for d in (A, Ap) if d is not None], ignore_index=True) if (A is not None or Ap is not None) else None
    for name, d in (("A", A), ("Aplus", Ap), ("B", B)):
        if d is not None:
            n = int((~d["attained_alg"].astype(bool)).sum())
            if n:
                print(f"warning: {n} instances of block {name} stopped at max_paths before the 0.1% precision")
            n = int((d["opt_paths_outside_grid"] == 1).sum())
            if n:
                print(f"note: in {n} instances of block {name} the SDP-2 policy left the grid, no paired estimate")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None
        print("matplotlib not installed, figures skipped")

    def savefig(f, name):
        f.tight_layout()
        for ext in ("pdf", "png"):
            f.savefig(os.path.join(fig, f"{name}.{ext}"), dpi=200)
        plt.close(f)
        print(f"figure: {fig}/{name}.pdf")

    def boxes(ax, data, labels, title):
        ax.boxplot(data, showfliers=True, widths=0.6, flierprops=dict(markersize=2))
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=45 if len(labels) > 4 else 0, fontsize=7, ha="right" if len(labels) > 4 else "center")
        ax.set_title(title, fontsize=8)
        ax.tick_params(axis="y", labelsize=7)

    # Q1 ----------------------------------------------------------------------------------------------
    rows = []
    if AA is not None:
        for T, gT in AA.groupby("T"):
            name = "Aplus" if T == 12 else "A"
            for dist in ordered(gT["distribution"].unique(), DIST_ORDER):
                rows.append(gap_row({"test set": name, "T": T, "distribution": dist}, gT[gT["distribution"] == dist]))
            rows.append(gap_row({"test set": name, "T": T, "distribution": "all"}, gT))
    if A is not None:
        rows.append(gap_row({"test set": "A", "T": "4 and 8", "distribution": "all"}, A))
    if B is not None:
        rows += [gap_row({"test set": f"B, scale {s}", "T": int(g["T"].iloc[0]), "distribution": k}, g)
                 for (k, s), g in B.groupby(["distribution", "scale"])]
    if rows:
        save(pd.DataFrame(rows), "test2_Q1_gap",
             "Optimality gap of Algorithm 1 relative to SDP-2, $100(\\mathrm{ETC}-C_1)/C_1$.", tab)

    if A is not None:
        save(pivot_table(A, AA), "test2_pivot",
             "Average percentage optimality gap of Algorithm 1 under pivoting parameters.", tab)

    if A is not None and Ap is not None:
        key = ["dist", "cv", "pat1", "pat2", "K", "z", "R", "v", "h", "b"]
        settings = Ap[key].drop_duplicates()
        M = AA.merge(settings, on=key, how="inner")
        rows = []
        for dist in ordered(M["distribution"].unique(), DIST_ORDER) + ["all"]:
            g = M if dist == "all" else M[M["distribution"] == dist]
            r = {"distribution": dist}
            for T in sorted(g["T"].unique()):
                gT = g[g["T"] == T]
                r[f"T={T}"] = gT["gap_pct"].mean()
                r[f"instances T={T}"] = len(gT)
            rows.append(r)
        save(pd.DataFrame(rows), "test2_T_matched",
             "Mean optimality gap (\\%) at T = 4, 8 and 12 on identical settings: 5 pattern pairs, all costs.", tab)

        if plt is not None:
            Ts = sorted(AA["T"].unique())
            f, axes = plt.subplots(1, len(Ts), figsize=(4.2 * len(Ts), 3), sharey=True, squeeze=False)
            for ax, T in zip(axes[0], Ts):
                g = AA[AA["T"] == T]
                labs = ordered(g["pat1"].unique(), PATTERN_ORDER)
                boxes(ax, [g.loc[g["pat1"] == p, "gap_pct"] for p in labs], labs, f"T = {T}")
            axes[0][0].set_ylabel("optimality gap (%)", fontsize=8)
            savefig(f, "test2_box_pattern")

            params = [("K", "K"), ("R", "R"), ("b", "b"), ("(z, v)", "zv")]
            f, axes = plt.subplots(1, len(params), figsize=(8.4, 2.8), sharey=True)
            for ax, (lab, col) in zip(axes, params):
                levs = sorted(A[col].unique(), key=str) if col == "zv" else sorted(A[col].unique())
                boxes(ax, [A.loc[A[col] == x, "gap_pct"] for x in levs], [str(x) for x in levs], lab)
            axes[0].set_ylabel("optimality gap (%)", fontsize=8)
            savefig(f, "test2_box_cost")

            f, axes = plt.subplots(1, len(Ts), figsize=(4.2 * len(Ts), 3), sharey=True, squeeze=False)
            for ax, T in zip(axes[0], Ts):
                g = AA[AA["T"] == T]
                labs = ordered(g["distribution"].unique(), DIST_ORDER)
                boxes(ax, [g.loc[g["distribution"] == d, "gap_pct"] for d in labs], labs, f"T = {T}")
            axes[0][0].set_ylabel("optimality gap (%)", fontsize=8)
            savefig(f, "test2_box_dist")

    # Q2 ----------------------------------------------------------------------------------------------
    if AA is not None:
        rows = [time_row({"T": T, "distribution": d}, g[g["distribution"] == d])
                for T, g in AA.groupby("T") for d in ordered(g["distribution"].unique(), DIST_ORDER)]
        save(pd.DataFrame(rows), "test2_Q2_time", "Computing time of SDP-2 and of one decision of Algorithm 1.", tab)
        if "workers" in AA and (AA["workers"] > 1).any():
            print("note: block A ran in parallel processes; its times include the effect of the shared machine")

    # Q3 ----------------------------------------------------------------------------------------------
    if Cb is not None and "cpu_sdp2" in Cb and Cb["cpu_sdp2"].notna().any():
        C3 = Cb[Cb["cpu_sdp2"].notna()].copy()
        C3["I"] = C3["N"] ** 2
        rows = []
        for (d, sc, T), g in C3.groupby(["distribution", "scale", "T"]):
            rows.append({"distribution": d, "scale": sc, "T": T, "instances": len(g), I_COL: g["I"].mean(),
                         "SDP-2 (s)": g["seconds_sdp2"].mean(), "SDP-2 CPU (s)": g["cpu_sdp2"].mean(),
                         "period 1 decision (s)": g["seconds_decision_k1"].mean(),
                         "period 1 decision CPU (s)": g["cpu_decision_k1"].mean(),
                         "mean decision (s)": g["seconds_per_decision"].mean(),
                         "mean decision CPU (s)": g["cpu_per_decision"].mean()})
        t = pd.DataFrame(rows)
        for c in ("SDP-2 (s)", "SDP-2 CPU (s)", "period 1 decision (s)", "period 1 decision CPU (s)",
                  "mean decision (s)", "mean decision CPU (s)"):
            FMT[c] = "{:.3f}"
        save(t, "test2_Q3_time", "Complete SDP-2 solve against the decisions of Algorithm 1, by state space and horizon.", tab)
        if (C3["workers"] > 1).any():
            print("note: block C ran in parallel processes; the fits and figures use CPU times")

        # fitted laws: SDP-2 log t = a + alpha log T + beta log|I|; period-1 decision log t = a + gamma T + delta log|I|
        def ols(y, X):
            X = np.column_stack([np.ones(len(y))] + X)
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            r2 = 1 - np.sum((y - X @ coef) ** 2) / np.sum((y - y.mean()) ** 2)
            return coef, r2
        g = C3[C3["cpu_sdp2"] > 0.5]                      # very short runs are dominated by overheads
        cs, r2s = ols(np.log(g["cpu_sdp2"].values), [np.log(g["T"].values), np.log(g["I"].values)])
        cd, r2d = ols(np.log(C3["cpu_decision_k1"].values), [C3["T"].values.astype(float), np.log(C3["I"].values)])
        fit = pd.DataFrame([
            {"model": "SDP-2: log t = a + alpha log T + beta log|I|", "a": cs[0], "alpha / gamma": cs[1],
             "beta / delta": cs[2], "R2": r2s, "instances": len(g)},
            {"model": "period 1 decision: log t = a + gamma T + delta log|I|", "a": cd[0], "alpha / gamma": cd[1],
             "beta / delta": cd[2], "R2": r2d, "instances": len(C3)}])
        for c in ("a", "alpha / gamma", "beta / delta", "R2"):
            FMT[c] = "{:.3f}"
        save(fit, "test2_Q3_fit", "Fitted computing-time laws (CPU time, block C).", tab)

        if plt is not None:
            Ts = sorted(C3["T"].unique())
            f, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
            for ax, col, title in ((axes[0], "cpu_sdp2", "SDP-2, complete solve"),
                                   (axes[1], "cpu_decision_k1", "Algorithm 1, decision of period 1")):
                for T, ls in zip(Ts, ("-", "--", "-.", ":")):
                    m = C3[C3["T"] == T].groupby("I")[col].mean()
                    ax.plot(m.index, m.values, ls, marker="o", markersize=3, label=f"T = {T}")
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlabel("number of states $|\\mathcal{I}|$", fontsize=8)
                ax.set_title(title, fontsize=8)
                ax.tick_params(labelsize=7)
            axes[0].set_ylabel("CPU time (s)", fontsize=8)
            axes[0].legend(frameon=False, fontsize=7)
            savefig(f, "test2_Q3_time")

            Ig = np.logspace(np.log10(C3["I"].min()), np.log10(C3["I"].max() * 25), 200)
            Tg = np.linspace(2, max(Ts) * 4, 200)
            II, TT = np.meshgrid(Ig, Tg)
            sdp = np.exp(cs[0] + cs[1] * np.log(TT) + cs[2] * np.log(II))
            dec = np.exp(cd[0] + cd[1] * TT + cd[2] * np.log(II))
            from matplotlib.lines import Line2D
            f, ax = plt.subplots(figsize=(5.5, 4))
            handles = []
            for (bud, lab), ls in zip(BUDGETS, (":", "--", "-")):
                ax.contour(II, TT, sdp, levels=[bud], colors="C3", linestyles=ls)
                ax.contour(II, TT, dec, levels=[bud], colors="C0", linestyles=ls)
                handles.append(Line2D([], [], color="0.3", linestyle=ls, label=f"budget {lab}"))
            handles += [Line2D([], [], color="C3", label="SDP-2, complete solve"),
                        Line2D([], [], color="C0", label="Algorithm 1, decision of period 1"),
                        Line2D([], [], color="k", marker="o", linestyle="", markersize=3, label="measured instances")]
            ax.scatter(C3["I"], C3["T"], s=8, color="k")
            ax.set_xscale("log")
            ax.set_xlabel("number of states $|\\mathcal{I}|$", fontsize=8)
            ax.set_ylabel("planning horizon $T$", fontsize=8)
            ax.legend(handles=handles, frameon=False, fontsize=6, loc="upper right")
            ax.tick_params(labelsize=7)
            savefig(f, "test2_Q3_map")


if __name__ == "__main__":
    main()
