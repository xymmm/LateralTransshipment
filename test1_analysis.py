"""
test1_analysis.py : the three answers from the csv files written by test1_experiments.py.

    Q1  Is SDP-2 exactly equal to SDP-1?                       table test1_Q1_equality   (blocks A, B, D)
    Q2  How much time does SDP-2 save?                         table test1_Q2_time       (block A)
    Q3  How large is the gap when the horizon T grows?         table test1_Q3_horizon    (block B)
                                                               figure test1_time_vs_T

Set RESULTS_DIR below and click Run, or: python test1_analysis.py --dir results_test1_int
Output goes to <dir>/tables (csv and booktabs LaTeX) and <dir>/figures (pdf and png).
Blocks whose csv does not exist yet are skipped.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

# folder to read when clicking "Run": "results_test1_int" after the full study, "results_test1_int_quick" after a quick run
RESULTS_DIR = "results_test1_int"


def latex(df: pd.DataFrame, caption: str, label: str, fmt: dict) -> str:
    """Minimal booktabs table, no dependency beyond pandas."""
    cols = list(df.columns)
    lines = ["\\begin{table}[H]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             "\\begin{tabular}{" + "c" * len(cols) + "}", "\\toprule",
             " & ".join(str(c).replace("_", "\\_").replace("%", "\\%") for c in cols) + " \\\\", "\\midrule"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v, f = r[c], fmt.get(c)
            num = isinstance(v, (int, float, np.integer, np.floating)) and not pd.isna(v)
            cells.append(f.format(v) if (f is not None and num) else str(v))
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


FMT = {"instances": "{:.0f}", "N": "{:.0f}", "T": "{:.0f}", "M": "{:g}",
       "SDP-1 (s)": "{:.2f}", "SDP-2 (s)": "{:.2f}", "SDP-1 / SDP-2": "{:.1f}", "SDP-1 - SDP-2 (s)": "{:.2f}",
       "max abs diff": "{:.1e}", "max rel diff at (0,0)": "{:.1e}",
       "instances with identical policy (%)": "{:.1f}", "states with a tie": "{:.0f}"}


def save(df, name, caption, out):
    df.to_csv(os.path.join(out, f"{name}.csv"), index=False)
    with open(os.path.join(out, f"{name}.tex"), "w") as f:
        f.write(latex(df, caption, f"tab:{name}", FMT))
    print(f"\n== {name}: {caption}")
    with pd.option_context("display.width", 200):
        print(df.to_string(index=False))


def dist_label(r):
    return "Poisson" if r["dist"] == "poisson" else f"Normal, cv={r['cv']}"


def equality_row(label, g):
    return {"test set": label,
            "instances": len(g),
            "max abs diff": g["max_abs_diff"].max(),
            "max rel diff at (0,0)": g["rel_diff_00"].max(),
            "instances with identical policy (%)": 100 * (g["differing_actions"] == 0).mean(),
            "states with a tie": int(g["differing_actions"].sum())}


def time_row(keys, g):
    return {**keys,
            "instances": len(g),
            "N": g["N"].mean(),
            "SDP-1 (s)": g["time_sdp1"].mean(),
            "SDP-2 (s)": g["time_sdp2"].mean(),
            "SDP-1 - SDP-2 (s)": (g["time_sdp1"] - g["time_sdp2"]).mean(),
            "SDP-1 / SDP-2": g["measured_speedup"].mean()}


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
        d = d[d["time_sdp1"].notna()].copy()                    # rows where both methods were run
        d["distribution"] = d.apply(dist_label, axis=1)
        return d

    A, B, D = read("A"), read("B"), read("D")

    # Q1: exact equality --------------------------------------------------------------------------------
    rows = []
    for label, d in (("A: distributions x demand levels, T = 4", A), ("B: horizons T = 4 to 52", B),
                     ("D: 36 cost settings, T = 4", D)):
        if d is not None and len(d):
            rows.append(equality_row(label, d))
    if rows:
        allrows = pd.concat([d for d in (A, B, D) if d is not None], ignore_index=True)
        rows.append(equality_row("all instances", allrows))
        save(pd.DataFrame(rows), "test1_Q1_equality",
             "Difference between the value functions of SDP-1 and SDP-2 over all periods and states. "
             "A different optimal action can only arise from a tie between actions of equal value.", tab)

    # Q2: time saved, block A ----------------------------------------------------------------------------
    if A is not None and len(A):
        t = pd.DataFrame([time_row({"distribution": k[0], "M": k[1]}, g)
                          for k, g in A.groupby(["distribution", "M"])])
        save(t, "test1_Q2_time", "Running time of SDP-1 and SDP-2, $T=4$, averages over 20 pattern pairs.", tab)

    # Q3: the gap as T grows, block B --------------------------------------------------------------------
    if B is not None and len(B):
        t = pd.DataFrame([time_row({"distribution": k[0], "T": k[1]}, g)
                          for k, g in B.groupby(["distribution", "T"])])
        save(t, "test1_Q3_horizon", "Running time of SDP-1 and SDP-2 as the horizon grows, $M=7$, "
                                    "averages over 10 demand patterns.", tab)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed, figure skipped")
            return
        f, ax = plt.subplots(figsize=(5, 3.4))
        for (dist, g), ls in zip(t.groupby("distribution"), ("-", "--")):
            ax.plot(g["T"], g["SDP-1 (s)"], ls, marker="o", color="C3", label=f"SDP-1, {dist}")
            ax.plot(g["T"], g["SDP-2 (s)"], ls, marker="s", color="C0", label=f"SDP-2, {dist}")
        ax.set_yscale("log")
        ax.set_xlabel("planning horizon $T$")
        ax.set_ylabel("running time (s)")
        ax.legend(frameon=False, fontsize=8)
        f.tight_layout()
        for ext in ("pdf", "png"):
            f.savefig(os.path.join(fig, f"test1_time_vs_T.{ext}"), dpi=200)
        print(f"\nfigure: {fig}/test1_time_vs_T.pdf")


if __name__ == "__main__":
    main()