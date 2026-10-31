"""
test2_analysis.py : tables and figure from the csv files written by test2_experiments.py.

    Q1  optimality gap of Algorithm 1                    table test2_Q1_gap      (block A, and block B by scale)
    Q2  computing time by demand distribution            table test2_Q2_time     (block A)
    Q3  computing time as the problem size grows         table test2_Q3_scale    (blocks A, B and C)
                                                         figure test2_time_vs_N

Set RESULTS_DIR below and click Run, or: python test2_analysis.py --dir results_test2
Output goes to <dir>/tables (csv and booktabs LaTeX) and <dir>/figures (pdf and png).
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from test1_analysis import latex

RESULTS_DIR = "results_test2"

FMT = {"instances": "{:.0f}", "N": "{:.0f}", "scale": "{:.0f}",
       "mean gap (%)": "{:.3f}", "max gap (%)": "{:.3f}", "mean half-width (%)": "{:.3f}",
       "mean paired gap (%)": "{:.3f}", "max paired gap (%)": "{:.3f}",
       "SDP-2 (s)": "{:.2f}", "Algorithm 1, one decision (s)": "{:.3f}", "Algorithm 1, decision at k=1 (s)": "{:.3f}",
       "decisions per instance": "{:.0f}", "Algorithm 1, simulation (s)": "{:.1f}"}


def save(df, name, caption, out):
    df.to_csv(os.path.join(out, f"{name}.csv"), index=False)
    with open(os.path.join(out, f"{name}.tex"), "w") as f:
        f.write(latex(df, caption, f"tab:{name}", FMT))
    print(f"\n== {name}: {caption}")
    with pd.option_context("display.width", 200):
        print(df.to_string(index=False))


def dist_label(r):
    return "Poisson" if r["dist"] == "poisson" else f"Normal, cv={r['cv']}"


def gap_row(keys, g):
    return {**keys, "instances": len(g),
            "mean gap (%)": g["gap_pct"].mean(), "max gap (%)": g["gap_pct"].max(),
            "mean half-width (%)": g["hw_pct"].mean(),
            "mean paired gap (%)": g["paired_gap_pct"].mean(), "max paired gap (%)": g["paired_gap_pct"].max()}


def time_row(keys, g):
    return {**keys, "instances": len(g), "N": g["N"].mean(),
            "SDP-2 (s)": g["seconds_sdp2"].mean(),
            "Algorithm 1, one decision (s)": g["seconds_per_decision"].mean(),
            "Algorithm 1, decision at k=1 (s)": g["seconds_per_decision_k1"].mean(),
            "decisions per instance": g["decisions_alg"].mean(),
            "Algorithm 1, simulation (s)": g["seconds_alg_total"].mean() if "seconds_alg_total" in g else np.nan}


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
        d["distribution"] = d.apply(dist_label, axis=1)
        return d

    A, B, Cb = read("A"), read("B"), read("C")
    not_attained = sum(int((~d["attained_alg"].astype(bool)).sum()) for d in (A, B) if d is not None)
    if not_attained:
        print(f"warning: {not_attained} instances stopped at max_paths before the 0.1% precision")

    # Q1 ----------------------------------------------------------------------------------------------
    rows = []
    if A is not None and len(A):
        rows += [gap_row({"test set": "A", "distribution": k}, g) for k, g in A.groupby("distribution")]
        rows.append(gap_row({"test set": "A", "distribution": "all"}, A))
    if B is not None and len(B):
        rows += [gap_row({"test set": f"B, scale {s}", "distribution": k}, g)
                 for (k, s), g in B.groupby(["distribution", "scale"])]
    if rows:
        save(pd.DataFrame(rows), "test2_Q1_gap",
             "Optimality gap of Algorithm 1 relative to SDP-2, $100(\\mathrm{ETC}-C_1)/C_1$.", tab)

    # Q2 ----------------------------------------------------------------------------------------------
    if A is not None and len(A):
        t = pd.DataFrame([time_row({"distribution": k}, g) for k, g in A.groupby("distribution")])
        save(t, "test2_Q2_time", f"Computing time of SDP-2 and of one decision of Algorithm 1, $T={int(A['T'].iloc[0])}$.", tab)

    # Q3 ----------------------------------------------------------------------------------------------
    if B is not None and len(B):
        AB = pd.concat([d for d in (A, B) if d is not None], ignore_index=True)
        AB = AB[(AB["dist"].eq("poisson") | AB["cv"].eq(0.3)) & AB["pat1"].eq(AB["pat2"])
                & AB["pat1"].isin(["LCY1", "LCY2", "SIN1", "STAT", "EMP1"])]
        rows = [time_row({"test set": "A, B", "distribution": k[0], "scale": k[1]}, g)
                for k, g in AB.groupby(["distribution", "scale"])]
        if Cb is not None and len(Cb):
            for k, g in Cb.groupby(["distribution", "scale"]):
                r = time_row({"test set": "C", "distribution": k[0], "scale": k[1]}, g)
                r["Algorithm 1, simulation (s)"] = np.nan                 # block C does not simulate the expected cost
                rows.append(r)
        t = pd.DataFrame(rows)
        save(t, "test2_Q3_scale", f"Computing time as the demand level grows, $T={int(B['T'].iloc[0])}$, demand means of "
                                  "test set A multiplied by the scale.", tab)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed, figure skipped")
            return
        f, ax = plt.subplots(figsize=(5, 3.4))
        for (dist, g), ls in zip(t.groupby("distribution"), ("-", "--")):
            ax.plot(g["N"], g["SDP-2 (s)"], ls, marker="o", color="C3", label=f"SDP-2, {dist}")
            ax.plot(g["N"], g["Algorithm 1, decision at k=1 (s)"], ls, marker="s", color="C0",
                    label=f"Algorithm 1, one decision, {dist}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("inventory levels per location $N$")
        ax.set_ylabel("computing time (s)")
        ax.legend(frameon=False, fontsize=7)
        f.tight_layout()
        for ext in ("pdf", "png"):
            f.savefig(os.path.join(fig, f"test2_time_vs_N.{ext}"), dpi=200)
        print(f"\nfigure: {fig}/test2_time_vs_N.pdf")


if __name__ == "__main__":
    main()
