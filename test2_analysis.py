"""
test2_analysis.py : tables and figures from the csv files written by test2_experiments.py.

    Q1  optimality gap of Algorithm 1
          table  test2_Q1_gap          by horizon and distribution (blocks A and Aplus), and by demand scale (block B)
          table  test2_T_matched       T = 4, 8, 12 on the 180 settings that block Aplus shares with block A
          table  test2_pivot           average gap pivoting each parameter, as Tables 1 and 2 of the 2022 version:
                                       horizon, distribution, demand pattern of location 1, identical or different
                                       patterns, K, R, b, (z, v); one column per horizon and one over all instances
                                       of blocks A and Aplus, which form one test set (Aplus covers 5 of the 20
                                       pattern pairs at T = 12)
          figure test2_box_pattern     box plots of the gap by demand pattern of location 1, one panel per horizon
          figure test2_box_cost        box plots of the gap by K, R, b and (z, v)
          figure test2_box_dist        box plots of the gap by distribution, one panel per horizon
    Q2  computing time
          table  test2_Q2_time         SDP-2 and one decision of Algorithm 1, by horizon and distribution (A, Aplus)
          table  test2_Q2_pivot        the same test set pivoted like test2_pivot: for every level of every parameter
                                       and every horizon, SDP-2 complete solve, one decision and decision at k=1
    Q3  computing time as the state space and the horizon grow (block C, full factorial)
          table  test2_Q3_time         SDP-2 complete solve and Algorithm 1 decisions for every distribution, state space
                                       level and T: wall time and CPU time
          table  test2_Q3_fit          fitted laws, log(time) regressed on log T and log |I| (SDP-2) and on T and
                                       log |I| (decision of period 1, MILP-1 over the whole horizon)
          figure test2_Q3_time         time against |I|, one line per T, SDP-2 and Algorithm 1 side by side
          figure test2_Q3_map          where each method fits a time budget in the (|I|, T) plane, from the fitted laws
          figure test2_Q3_map_simple   the same map for a budget of one hour only, with the four regions labelled
          table  test2_Q3_table        the table of the paper (longtable, label tab:comp.size): for each T, the CPU time
                                       of the complete SDP-2 solve, of the T decisions of Algorithm 1 along one demand
                                       path (block Chorizon, mean over its paths) and of the decision at t = 1
          table  test2_Q3_budget       in place of the map: by horizon, the largest |I| that SDP-2 solves within one
                                       hour and the fitted times of Algorithm 1 (test2_Q3_budget_paper.tex is the
                                       booktabs version, label tab:comp.budget)
          table  test2_Q3_period       mean decision time of Algorithm 1 in each period at T = 12 and 16 (Chorizon)

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


def pivot_params(A, AA):
    return [("distribution", "distribution", ordered(AA["distribution"].unique(), DIST_ORDER)),
            ("demand pattern of location 1", "pat1", ordered(AA["pat1"].unique(), PATTERN_ORDER)),
            ("patterns of the two locations", "pair_type", ["identical", "different"]),
            ("fixed ordering cost K", "K", sorted(AA["K"].unique())),
            ("fixed transshipment cost R", "R", sorted(AA["R"].unique())),
            ("backorder cost b", "b", sorted(AA["b"].unique())),
            ("unit costs (z, v)", "zv", ordered(AA["zv"].unique(), []))]


def time_pivot(AA):
    """Mean wall time for every level of every parameter and every horizon on blocks A and Aplus."""
    horizons = sorted(AA["T"].unique())
    cols = [("SDP-2", "seconds_sdp2"), ("one decision", "seconds_per_decision"), ("k=1", "seconds_per_decision_k1")]
    rows = []
    for label, col, levels in pivot_params(None, AA) + [("all instances", None, ["all"])]:
        for lev in levels:
            g = AA if col is None else AA[AA[col] == lev]
            r = {"parameter": label, "level": lev}
            for T in horizons:
                gT = g[g["T"] == T]
                for name, c in cols:
                    r[f"T={T} {name} (s)"] = gT[c].mean() if len(gT) else np.nan
            rows.append(r)
    return pd.DataFrame(rows)


def pivot_table(A, AA):
    """Average gap for every level of every parameter: one column per horizon and one over all instances of blocks
    A and Aplus (AA), which form one test set."""
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
            r["all"] = g["gap_pct"].mean()
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

    A, B, Cb, Ap, H = read("A"), read("B"), read("C"), read("Aplus"), read("Chorizon")
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
            f.savefig(os.path.join(fig, f"{name}.{ext}"), dpi=300)
        plt.close(f)
        print(f"figure: {fig}/{name}.pdf")

    # figures are drawn at the width of the text (about 6.5 in), so that 8 pt in the figure is 8 pt on the page
    TEXT_WIDTH = 6.5

    def short(d):
        return "Poisson" if d == "Poisson" else "$\\rho$ = " + d.split("=")[-1]

    def boxes(ax, data, labels, title):
        ax.boxplot(data, showfliers=True, widths=0.6, flierprops=dict(markersize=2, markeredgewidth=0.5),
                   medianprops=dict(color="C0", linewidth=1.2), boxprops=dict(linewidth=0.8),
                   whiskerprops=dict(linewidth=0.8), capprops=dict(linewidth=0.8))
        ax.set_xticks(range(1, len(labels) + 1))
        rot = 90 if len(labels) > 6 else 0
        ax.set_xticklabels(labels, rotation=rot, fontsize=7)
        ax.set_title(title, fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="y", linewidth=0.3, alpha=0.5)

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
            f, axes = plt.subplots(1, len(Ts), figsize=(TEXT_WIDTH, 2.5), sharey=True, squeeze=False)
            for ax, T in zip(axes[0], Ts):
                g = AA[AA["T"] == T]
                labs = ordered(g["pat1"].unique(), PATTERN_ORDER)
                boxes(ax, [g.loc[g["pat1"] == p, "gap_pct"] for p in labs], labs, f"T = {T}")
            axes[0][0].set_ylabel("optimality gap (%)", fontsize=8)
            savefig(f, "test2_box_pattern")

            params = [("K", "K"), ("R", "R"), ("b", "b"), ("(z, v)", "zv")]
            f, axes = plt.subplots(1, len(params), figsize=(TEXT_WIDTH, 2.3), sharey=True)
            for ax, (lab, col) in zip(axes, params):
                levs = sorted(AA[col].unique(), key=str) if col == "zv" else sorted(AA[col].unique())
                boxes(ax, [AA.loc[AA[col] == x, "gap_pct"] for x in levs], [str(x) for x in levs], lab)
            axes[0].set_ylabel("optimality gap (%)", fontsize=8)
            savefig(f, "test2_box_cost")

            f, axes = plt.subplots(1, len(Ts), figsize=(TEXT_WIDTH, 2.3), sharey=True, squeeze=False)
            for ax, T in zip(axes[0], Ts):
                g = AA[AA["T"] == T]
                labs = ordered(g["distribution"].unique(), DIST_ORDER)
                boxes(ax, [g.loc[g["distribution"] == d, "gap_pct"] for d in labs], [short(d) for d in labs], f"T = {T}")
            axes[0][0].set_ylabel("optimality gap (%)", fontsize=8)
            savefig(f, "test2_box_dist")

    # Q2 ----------------------------------------------------------------------------------------------
    if AA is not None:
        rows = [time_row({"T": T, "distribution": d}, g[g["distribution"] == d])
                for T, g in AA.groupby("T") for d in ordered(g["distribution"].unique(), DIST_ORDER)]
        save(pd.DataFrame(rows), "test2_Q2_time", "Computing time of SDP-2 and of one decision of Algorithm 1.", tab)
        save(time_pivot(AA), "test2_Q2_pivot",
             "Average computing time in seconds of SDP-2 and of Algorithm 1 under pivoting parameters.", tab)
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
        # the same quantities on T alone, which is what the budget table below needs
        ct, r2t = ols(np.log(C3["cpu_decision_k1"].values), [C3["T"].values.astype(float)])
        fit.loc[len(fit)] = {"model": "period 1 decision: log t = a + gamma T", "a": ct[0], "alpha / gamma": ct[1],
                             "beta / delta": np.nan, "R2": r2t, "instances": len(C3)}
        keys = ["dist", "cv", "scale", "T", "pat1"]
        C3h = C3.copy()
        ch = None
        if H is not None and "cpu_horizon_mean" in H and H["cpu_horizon_mean"].notna().any():
            Hm = H[H["cpu_horizon_mean"].notna()]
            C3h = C3h.merge(Hm[keys + ["cpu_horizon_mean", "cpu_horizon_max", "cpu_period_mean", "paths", "decisions",
                                        "cpu_decision_k1"]].rename(columns={"cpu_decision_k1": "cpu_k1_h"}),
                            on=keys, how="left")
            # the t = 1 column of the paper's table comes from the same run as the horizon column, so that the
            # horizon time, which contains the decision at t = 1, is never below it
            C3h["cpu_k1_tab"] = C3h["cpu_k1_h"].fillna(C3h["cpu_decision_k1"])
            g = C3h[C3h["cpu_horizon_mean"].notna()]
            ch, r2h = ols(np.log(g["cpu_horizon_mean"].values), [g["T"].values.astype(float)])
            fit.loc[len(fit)] = {"model": "horizon (T decisions on one path): log t = a + gamma T", "a": ch[0],
                                 "alpha / gamma": ch[1], "beta / delta": np.nan, "R2": r2h, "instances": len(g)}
            print(f"note: block Chorizon covers {len(g)} of the {len(C3)} instances of block C")
        else:
            C3h["cpu_horizon_mean"] = np.nan
            C3h["cpu_k1_tab"] = C3h["cpu_decision_k1"]
            print("note: results_test2/Chorizon.csv not found or empty; the horizon column of test2_Q3_table is empty")
        for c in ("a", "alpha / gamma", "beta / delta", "R2"):
            FMT[c] = "{:.3f}"
        save(fit, "test2_Q3_fit", "Fitted computing-time laws (CPU time, block C).", tab)

        # table of the paper: SDP-2 complete solve | Algorithm 1 over the horizon | Algorithm 1 decision at t = 1,
        # three columns for each T, rows by distribution and scale, averaged over the two patterns
        def sec(x):
            if pd.isna(x):
                return "--"
            if x >= 1000:
                return f"{x:,.0f}"
            if x >= 100:
                return f"{x:.0f}"
            if x >= 10:
                return f"{x:.1f}"
            if x >= 1:
                return f"{x:.2f}"
            if x >= 0.1:
                return f"{x:.2f}"
            return f"{x:.3f}"

        def longtable(caption, label, spec, header, body, ncols):
            head = ["\\toprule", header, "\\midrule"]
            lines = ["{\\footnotesize", "\\renewcommand{\\arraystretch}{0.9}", "\\setlength{\\tabcolsep}{3pt}",
                     f"\\begin{{longtable}}{{{spec}}}", f"\\caption{{{caption}}}", f"\\label{{{label}}} \\\\"]
            lines += head + ["\\endfirsthead", f"\\multicolumn{{{ncols}}}{{l}}{{\\textit{{Table~\\ref{{{label}}} continued}}}} \\\\"]
            lines += head + ["\\endhead", "\\midrule", f"\\multicolumn{{{ncols}}}{{r}}{{\\textit{{continued on next page}}}} \\\\",
                             "\\endfoot", "\\bottomrule", "\\endlastfoot"]
            lines += body + ["\\end{longtable}", "}"]
            return "\n".join(lines) + "\n"

        Ts = sorted(C3["T"].unique())
        header = (" & & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{$T={T}$}}" for T in Ts) + " \\\\\n"
                  + "".join(f"\\cmidrule(lr){{{3 + 3 * j}-{5 + 3 * j}}}" for j in range(len(Ts))) + "\n"
                  + "$s$ & " + I_COL + " & " + " & ".join("SDP-2 & Horizon & $t=1$" for _ in Ts) + " \\\\")
        body, recs = [], []
        for dl in ("Poisson", "Normal, cv=0.3"):
            d = C3h[C3h["distribution"] == dl]
            if not len(d):
                continue
            lab = "Poisson" if dl == "Poisson" else "Normal, $\\rho=" + dl.split("=")[1] + "$"
            body.append(f"\\multicolumn{{{2 + 3 * len(Ts)}}}{{l}}{{\\textit{{{lab}}}}} \\\\")
            for sc in sorted(d["scale"].unique()):
                x = d[d["scale"] == sc]
                I = x[x["T"] == min(Ts)]["I"].mean()
                cells, rec = [], {"distribution": dl, "scale": sc, I_COL: I}
                for T in Ts:
                    y = x[x["T"] == T]
                    vals = (y["cpu_sdp2"].mean(), y["cpu_horizon_mean"].mean(), y["cpu_k1_tab"].mean())
                    cells += [sec(v) for v in vals]
                    rec.update({f"SDP-2 T={T}": vals[0], f"horizon T={T}": vals[1], f"t=1 T={T}": vals[2]})
                body.append(f"\\quad {sc:g} & {I:,.0f} & " + " & ".join(cells) + " \\\\")
                recs.append(rec)
        cap = ("CPU time in seconds of the complete solve of SDP-2, of the $T$ decisions of Algorithm~\\ref{alg:heuristic} "
               "along one demand path and of its decision at $t=1$, by the size of the state space and the horizon. "
               "$|\\mathcal{I}|$ is the number of states at $T=4$, 8 and 16; at $T=12$ the pattern SIN1 has slightly fewer states.")
        with open(os.path.join(tab, "test2_Q3_table.tex"), "w") as f:
            f.write(longtable(cap, "tab:comp.size", "l" + "r" * (1 + 3 * len(Ts)), header, body, 2 + 3 * len(Ts)))
        pd.DataFrame(recs).to_csv(os.path.join(tab, "test2_Q3_table.csv"), index=False)
        print(f"\n== test2_Q3_table: {cap}")
        print("\n".join(body))

        # where each method is faster: smallest |I| at which Algorithm 1 beats SDP-2, by T
        print("\n== crossover: instances where the horizon time / the decision at t=1 of Algorithm 1 is below SDP-2")
        for T in Ts:
            y = C3h[C3h["T"] == T].sort_values("I")
            fh = y[y["cpu_horizon_mean"] < y["cpu_sdp2"]]["I"] if y["cpu_horizon_mean"].notna().any() else pd.Series(dtype=float)
            fk = y[y["cpu_decision_k1"] < y["cpu_sdp2"]]["I"]
            print(f"  T={T}: horizon faster in {len(fh)}/{len(y)} instances"
                  + (f", smallest |I| {fh.min():,.0f}" if len(fh) else "")
                  + f"; t=1 faster in {len(fk)}/{len(y)}" + (f", smallest |I| {fk.min():,.0f}" if len(fk) else ""))

        # budget table, in place of the map: for a budget of one hour, the largest state space SDP-2 solves and the
        # fitted times of Algorithm 1, by horizon; cells beyond one day are left out, the fit is estimated up to T = 16
        B_SEC, DAY = 3600.0, 86400.0
        rows = []
        for T in (4, 8, 12, 16, 20, 24, 28, 52):
            I_max = np.exp((np.log(B_SEC) - cs[0] - cs[1] * np.log(T)) / cs[2])
            tk = np.exp(ct[0] + ct[1] * T)
            th = np.exp(ch[0] + ch[1] * T) if ch is not None else np.nan
            rows.append({"T": T, "SDP-2: largest |I| within one hour": I_max, "inventory levels per location": np.sqrt(I_max),
                         "Algorithm 1: decision at t=1 (s)": tk if tk <= DAY else np.nan,
                         "Algorithm 1: horizon (s)": th if th <= DAY else np.nan})
        bud = pd.DataFrame(rows)
        FMT.update({"SDP-2: largest |I| within one hour": "{:,.0f}", "inventory levels per location": "{:.0f}",
                    "Algorithm 1: decision at t=1 (s)": "{:,.1f}", "Algorithm 1: horizon (s)": "{:,.1f}"})
        save(bud, "test2_Q3_budget", "Problem sizes within a budget of one hour, from the fitted relations.", tab)
        body = [f"{int(r['T'])} & {r['SDP-2: largest |I| within one hour']:,.0f} & {r['inventory levels per location']:.0f} & "
                f"{sec(r['Algorithm 1: decision at t=1 (s)'])} & {sec(r['Algorithm 1: horizon (s)'])} \\\\" for _, r in bud.iterrows()]
        with open(os.path.join(tab, "test2_Q3_budget_paper.tex"), "w") as f:
            f.write("\\begin{table}[H]\n\\centering\n\\footnotesize\n"
                    "\\caption{Problem sizes within a budget of one hour, from the fitted relations. The fit of "
                    "Algorithm~\\ref{alg:heuristic} is estimated on horizons up to $T=16$, and its times beyond one day are left out.}\n"
                    "\\label{tab:comp.budget}\n\\renewcommand{\\arraystretch}{0.9}\n\\begin{tabular}{rrrrr}\n\\toprule\n"
                    " & \\multicolumn{2}{c}{SDP-2 within one hour} & \\multicolumn{2}{c}{Algorithm~\\ref{alg:heuristic}, fitted time (s)} \\\\\n"
                    "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
                    "$T$ & largest $|\\mathcal{I}|$ & levels per location & $t=1$ & horizon \\\\\n\\midrule\n"
                    + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")

        # decision time by period: mean CPU time of the decisions of each period, at T = 16 and T = 12 (block Chorizon)
        if "cpu_period_mean" in C3h and C3h["cpu_period_mean"].notna().any():
            rows = []
            for T in (12, 16):
                y = C3h[(C3h["T"] == T) & C3h["cpu_period_mean"].notna()]
                if not len(y):
                    continue
                M = np.array([[float(v) for v in s.split()] for s in y["cpu_period_mean"]])
                rows.append({"T": T, **{f"t={k + 1}": M[:, k].mean() for k in range(T)}})
            per = pd.DataFrame(rows)
            for c in per.columns:
                if c != "T":
                    FMT[c] = "{:.3f}"
            save(per, "test2_Q3_period", "Mean CPU time (s) of the decision of Algorithm 1 in each period, block Chorizon.", tab)

        if plt is not None:
            Ts = sorted(C3["T"].unique())
            f, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.8), sharey=True)
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
            f, ax = plt.subplots(figsize=(0.75 * TEXT_WIDTH, 3.6))
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

            # simple map: one budget of one hour, four regions labelled in the plot
            B, Tmax = 3600.0, 60
            I1 = np.logspace(np.log10(4e3), np.log10(3e6), 400)
            T_sdp = np.exp((np.log(B) - cs[0] - cs[2] * np.log(I1)) / cs[1])       # SDP-2 reaches one hour
            T_alg = (np.log(B) - cd[0] - cd[2] * np.log(I1)) / cd[1]                # Algorithm 1 reaches one hour
            red, blue = "#c0392b", "#1f5f99"
            f, ax = plt.subplots(figsize=(0.75 * TEXT_WIDTH, 3.2))
            ax.fill_between(I1, 0, np.minimum(np.minimum(T_sdp, T_alg), Tmax), color="0.88", lw=0)
            ax.fill_between(I1, np.minimum(T_sdp, T_alg), np.minimum(T_alg, Tmax), where=T_alg > T_sdp,
                            color="#cfe0f1", lw=0)
            ax.fill_between(I1, np.minimum(T_alg, Tmax), np.minimum(T_sdp, Tmax), where=T_sdp > T_alg,
                            color="#f6d3d1", lw=0)
            ax.plot(I1, np.minimum(T_sdp, Tmax + 5), color=red, lw=1.4)
            ax.plot(I1, T_alg, color=blue, lw=1.4)
            ax.scatter(C3["I"], C3["T"], s=6, color="0.25", zorder=3, lw=0)
            ax.set_xscale("log")
            ax.set_xlim(4e3, 3e6)
            ax.set_ylim(0, Tmax)
            ax.set_xlabel("number of states $|\\mathcal{I}|$", fontsize=8)
            ax.set_ylabel("planning horizon $T$", fontsize=8)
            ax.text(2.2e4, 22.5, "both within one hour", ha="center", va="center", fontsize=7.5)
            ax.text(9.0e5, 12, "Algorithm 1 only", ha="center", va="center", fontsize=7.5, color=blue)
            ax.text(1.2e4, 48, "SDP-2 only", ha="center", va="center", fontsize=7.5, color=red)
            ax.text(9.0e5, 48, "out of time budget", ha="center", va="center", fontsize=7.5, color="0.35")
            ax.text(1.55e5, 57, "SDP-2, complete solve", ha="left", va="top", fontsize=7, color=red)
            ax.text(4.5e3, T_alg[0] + 1.3, "Algorithm 1, decision of period 1", ha="left", va="bottom", fontsize=7,
                    color=blue)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.tick_params(labelsize=7)
            savefig(f, "test2_Q3_map_simple")


if __name__ == "__main__":
    main()
