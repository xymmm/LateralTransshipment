"""
test3_analysis.py : Algorithm 1 against the quasi-myopic benchmark (QM), tables and figures.

    table  test3_overall       mean and median gap of Algorithm 1, QM and NT in test sets A, B and D, and the share
                               of instances in which Algorithm 1 has a lower expected cost than QM
    table  test3_A_value       test set A by the value of transshipment (NT gap) and by the value of joint planning
                               (QM gap); shows where each method is better
    table  test3_D_pivot       test set D, average gaps pivoting each factor: T, distribution, demand relation, K, R
    table  test3_D_KR          test set D, mean gap of Algorithm 1 and of QM for every (K, R)
    figure test3_gap_vs_value  mean gap of Algorithm 1 and QM against the NT gap, test bed I (A, Aplus) and test bed II (D)
    figure test3_box_ratio     box plots of the gaps in block D by K/R, Algorithm 1 next to QM

Gaps are 100 (ETC - C_1) / C_1 with C_1 the SDP-2 optimum at (0, 0). NT is the optimal policy without transshipment:
NT gap = value of transshipment, QM gap = value of planning orders and transshipments jointly.
Set RESULTS_DIR below and click Run, or: python test3_analysis.py
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from test1_analysis import latex

RESULTS_DIR = "results_test3"
TEST2_DIR = "results_test2"
DIST_ORDER = ["Poisson", "Normal, cv=0.1", "Normal, cv=0.2", "Normal, cv=0.3"]
VALUE_BINS = [-np.inf, 0.05, 0.5, 1, 2, 5, np.inf]
VALUE_LABELS = ["<= 0.05", "0.05-0.5", "0.5-1", "1-2", "2-5", "> 5"]
RATIO_BINS = [0, 2, 4.5, np.inf]
RATIO_LABELS = ["K/R <= 2", "2 < K/R <= 4", "K/R >= 5"]

FMT = {c: "{:.3f}" for c in ["Alg mean (%)", "QM mean (%)", "NT mean (%)", "Alg median (%)", "QM median (%)",
                             "NT median (%)", "Alg", "QM", "NT"]}
FMT.update({"instances": "{:.0f}", "Alg better (%)": "{:.1f}"})


def save(df, name, caption, out):
    df.to_csv(os.path.join(out, f"{name}.csv"), index=False)
    with open(os.path.join(out, f"{name}.tex"), "w") as f:
        f.write(latex(df, caption, f"tab:{name}", FMT))
    print(f"\n== {name}: {caption}")
    with pd.option_context("display.width", 220, "display.max_rows", 200):
        print(df.to_string(index=False))


def dist_label(r):
    return "Poisson" if r["dist"] == "poisson" else f"Normal, cv={r['cv']}"


def summary(keys, g):
    return {**keys, "instances": len(g),
            "Alg mean (%)": g["gap_alg"].mean(), "QM mean (%)": g["gap_qm"].mean(), "NT mean (%)": g["gap_nt"].mean(),
            "Alg median (%)": g["gap_alg"].median(), "QM median (%)": g["gap_qm"].median(),
            "NT median (%)": g["gap_nt"].median(),
            "Alg better (%)": 100 * (g["alg_minus_qm_pct"] < 0).mean()}


def pivot(d, params):
    rows = []
    for label, col, levels in params:
        for lev in levels:
            g = d[d[col] == lev]
            if len(g):
                r = summary({"parameter": label, "level": lev}, g)
                rows.append({k: r[k] for k in ("parameter", "level", "instances", "Alg mean (%)", "QM mean (%)",
                                               "NT mean (%)", "Alg better (%)")})
    return pd.DataFrame(rows)


def ordered(values, order):
    return [x for x in order if x in set(values)] + sorted(set(values) - set(order), key=str)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=RESULTS_DIR)
    ap.add_argument("--test2", default=TEST2_DIR)
    a = ap.parse_args()
    tab, fig = os.path.join(a.dir, "tables"), os.path.join(a.dir, "figures")
    os.makedirs(tab, exist_ok=True)
    os.makedirs(fig, exist_ok=True)

    def existing(block):
        p3, p2 = os.path.join(a.dir, f"{block}.csv"), os.path.join(a.test2, f"{block}.csv")
        if not (os.path.exists(p3) and os.path.exists(p2)):
            return None
        q = pd.read_csv(p3).drop(columns=["gap_alg", "paired_gap_alg"], errors="ignore")
        d = pd.read_csv(p2).merge(q, on="id")
        d["gap_alg"] = d["gap_pct"]
        return d

    A, B = existing("A"), existing("B")
    Ap = existing("Aplus")                       # blocks A and Aplus of test2 form one test set
    if Ap is not None:
        A = Ap if A is None else pd.concat([A, Ap], ignore_index=True)
    pD = os.path.join(a.dir, "D.csv")
    D = pd.read_csv(pD) if os.path.exists(pD) else None
    for d in (A, B, D):
        if d is not None:
            d["distribution"] = d.apply(dist_label, axis=1)
            if "qm_outside_grid" in d and (d["qm_outside_grid"] == 1).any():
                print(f"note: {int((d['qm_outside_grid'] == 1).sum())} instances where QM left the grid are excluded")
            d.drop(d[d["gap_qm"].isna()].index, inplace=True)

    # overall ----------------------------------------------------------------------------------------------
    rows = [summary({"test set": n}, d) for n, d in (("A" if Ap is None else "A+Aplus", A), ("B", B), ("D", D))
            if d is not None and len(d)]
    if rows:
        save(pd.DataFrame(rows), "test3_overall",
             "Optimality gap of Algorithm 1 (Alg), the quasi-myopic benchmark (QM) and no transshipment (NT).", tab)

    # A by the value of transshipment and of joint planning ------------------------------------------------
    if A is not None and len(A):
        rows = []
        for name, col in (("value of transshipment (NT gap, %)", "gap_nt"), ("value of joint planning (QM gap, %)", "gap_qm")):
            bins = pd.cut(A[col], VALUE_BINS, labels=VALUE_LABELS)
            for lab in VALUE_LABELS:
                g = A[bins == lab]
                if len(g):
                    r = summary({"grouped by": name, "range": lab}, g)
                    rows.append({k: r[k] for k in ("grouped by", "range", "instances", "Alg mean (%)", "QM mean (%)",
                                                   "NT mean (%)", "Alg better (%)")})
        save(pd.DataFrame(rows), "test3_A_value",
             "Test set A grouped by the value of transshipment and by the value of joint planning.", tab)

    # D pivots -----------------------------------------------------------------------------------------------
    if D is not None and len(D):
        D["ratio class"] = pd.cut(D["K"] / D["R"], RATIO_BINS, labels=RATIO_LABELS).astype(str)
        D["value of transshipment"] = pd.cut(D["gap_nt"], VALUE_BINS, labels=VALUE_LABELS).astype(str)
        params = [("horizon T", "T", sorted(D["T"].unique())),
                  ("distribution", "distribution", ordered(D["distribution"].unique(), DIST_ORDER)),
                  ("demand relation", "relation", list(dict.fromkeys(D["relation"]))),
                  ("fixed ordering cost K", "K", sorted(D["K"].unique())),
                  ("fixed transshipment cost R", "R", sorted(D["R"].unique())),
                  ("ratio K/R", "ratio class", RATIO_LABELS),
                  ("value of transshipment (NT gap, %)", "value of transshipment", VALUE_LABELS)]
        save(pivot(D, params), "test3_D_pivot", "Test set D: average gaps pivoting each factor.", tab)
        rows = []
        for (K, R), g in D.groupby(["K", "R"]):
            rows.append({"K": K, "R": R, "instances": len(g), "Alg": g["gap_alg"].mean(), "QM": g["gap_qm"].mean(),
                         "NT": g["gap_nt"].mean(), "Alg better (%)": 100 * (g["alg_minus_qm_pct"] < 0).mean()})
        save(pd.DataFrame(rows), "test3_D_KR", "Test set D: mean gap (%) for every combination of K and R.", tab)

    # figures --------------------------------------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, figures skipped")
        return

    def savefig(f, name):
        f.tight_layout()
        for ext in ("pdf", "png"):
            f.savefig(os.path.join(fig, f"{name}.{ext}"), dpi=300)
        plt.close(f)
        print(f"figure: {fig}/{name}.pdf")

    sets = [(n, d) for n, d in (("Test bed I", A), ("Test bed II", D))
            if d is not None and len(d)]
    if sets:
        f, axes = plt.subplots(1, len(sets), figsize=(6.5, 2.8), squeeze=False, sharey=True)
        for ax, (n, d) in zip(axes[0], sets):
            bins = pd.cut(d["gap_nt"], VALUE_BINS, labels=VALUE_LABELS)
            m = d.groupby(bins, observed=False)[["gap_alg", "gap_qm"]].mean()
            x = np.arange(len(VALUE_LABELS))
            ax.plot(x, m["gap_alg"].values, "o-", color="C0", label="Algorithm 1")
            ax.plot(x, m["gap_qm"].values, "s--", color="C3", label="quasi-myopic")
            ax.set_xticks(x)
            ax.set_xticklabels(VALUE_LABELS, fontsize=7)
            ax.set_xlabel("gap of the policy without transshipment (%)", fontsize=8)
            ax.set_title(n, fontsize=8)
            ax.tick_params(axis="y", labelsize=7)
        axes[0][0].set_ylabel("mean optimality gap (%)", fontsize=8)
        axes[0][0].legend(frameon=False, fontsize=7)
        savefig(f, "test3_gap_vs_value")
    if D is not None and len(D):
        ratio = D["K"] / D["R"]
        rs = sorted(ratio.unique())
        f, ax = plt.subplots(figsize=(6.5, 3.0))
        pos = np.arange(len(rs))
        ax.boxplot([D.loc[ratio == r, "gap_alg"] for r in rs], positions=pos - 0.18, widths=0.3,
                   flierprops=dict(markersize=2), patch_artist=True, boxprops=dict(facecolor="C0", alpha=0.4))
        ax.boxplot([D.loc[ratio == r, "gap_qm"] for r in rs], positions=pos + 0.18, widths=0.3,
                   flierprops=dict(markersize=2), patch_artist=True, boxprops=dict(facecolor="C3", alpha=0.4))
        ax.set_xticks(pos)
        ax.set_xticklabels([f"{r:g}" for r in rs], fontsize=7)
        ax.set_xlabel("ratio of the fixed reorder cost to the fixed transshipment cost, $K/R$", fontsize=8)
        ax.set_ylabel("optimality gap (%)", fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
        ax.legend([plt.Rectangle((0, 0), 1, 1, fc="C0", alpha=0.4), plt.Rectangle((0, 0), 1, 1, fc="C3", alpha=0.4)],
                  ["Algorithm 1", "quasi-myopic"], frameon=False, fontsize=7)
        savefig(f, "test3_box_ratio")

if __name__ == "__main__":
    main()
