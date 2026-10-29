"""
test1_analysis.py : tables and figures from the csv files written by test1_experiments.py.

Usage
    python test1_analysis.py                     (reads results_test1/)
    python test1_analysis.py --dir results_test1_quick
Output goes to <dir>/tables (csv and booktabs LaTeX) and <dir>/figures (pdf and png).
Only the blocks whose csv exists are processed.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd


def latex(df: pd.DataFrame, caption: str, label: str, fmt: dict | None = None) -> str:
    """Minimal booktabs table, no dependency beyond pandas."""
    fmt = fmt or {}
    cols = list(df.columns)
    lines = ["\\begin{table}[H]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             "\\begin{tabular}{" + "c" * len(cols) + "}", "\\toprule",
             " & ".join(str(c).replace("_", "\\_").replace("%", "\\%") for c in cols) + " \\\\", "\\midrule"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            f = fmt.get(c)
            if f is not None and isinstance(v, (int, float, np.integer, np.floating)) and not pd.isna(v):
                cells.append(f.format(v))
            else:
                cells.append(str(v))
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def save(df, name, caption, out, fmt=None):
    df.to_csv(os.path.join(out, f"{name}.csv"), index=False)
    with open(os.path.join(out, f"{name}.tex"), "w") as f:
        f.write(latex(df, caption, f"tab:{name}", fmt))
    print(f"\n== {name}: {caption}")
    print(df.to_string(index=False))


def dist_label(r):
    return "Poisson" if r["dist"] == "poisson" else f"Normal, cv={r['cv']}"


FMT = {"N": "{:.0f}", "SDP-1 (s)": "{:.2f}", "SDP-2 (s)": "{:.2f}", "measured speed-up": "{:.1f}",
       "predicted speed-up": "{:.1f}", "max abs diff": "{:.1e}", "identical actions (%)": "{:.2f}",
       "instances": "{:.0f}", "SDP-1 extrapolated (s)": "{:.0f}", "T": "{:.0f}"}


def grouped(df, keys):
    """groupby-apply that works with pandas versions before and after 2.2."""
    try:
        return df.groupby(keys).apply(summarise, include_groups=False).reset_index()
    except TypeError:
        return df.groupby(keys).apply(summarise).reset_index()


def summarise(g):
    return pd.Series({
        "instances": len(g),
        "N": g["N"].mean(),
        "SDP-1 (s)": g["time_sdp1"].mean(),
        "SDP-2 (s)": g["time_sdp2"].mean(),
        "measured speed-up": g["measured_speedup"].mean(),
        "predicted speed-up": g["predicted_speedup"].mean(),
        "max abs diff": g["max_abs_diff"].max(),
        "identical actions (%)": 100 * g["identical_actions"].min(),
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results_test1")
    a = ap.parse_args()
    tab = os.path.join(a.dir, "tables")
    fig = os.path.join(a.dir, "figures")
    os.makedirs(tab, exist_ok=True)
    os.makedirs(fig, exist_ok=True)
    read = lambda b: pd.read_csv(os.path.join(a.dir, f"{b}.csv")) if os.path.exists(os.path.join(a.dir, f"{b}.csv")) else None

    A = read("A")
    if A is not None:
        A["distribution"] = A.apply(dist_label, axis=1)
        t = grouped(A, ["distribution", "M"])
        save(t, "test1_main", "SDP-1 against SDP-2 on the main test bed, $T=4$, averages over the pattern pairs.", tab, FMT)
        t = grouped(A, ["pat1", "pair_type"])
        save(t, "test1_patterns", "SDP-1 against SDP-2 by demand pattern of location 1 and pairing type.", tab, FMT)

    B = read("B")
    if B is not None:
        B["distribution"] = B.apply(dist_label, axis=1)
        t = grouped(B, ["distribution", "T"])
        save(t, "test1_horizon", "SDP-1 against SDP-2 for growing planning horizons, M = 7.", tab, FMT)

    D = read("D")
    if D is not None:
        D["(z, v)"] = "(" + D["z"].astype(str) + ", " + D["v"].astype(str) + ")"
        rows = []
        for par in ["K", "R", "b", "(z, v)"]:
            for lev, g in D.groupby(par):
                rows.append({"parameter": par, "level": lev, "instances": len(g),
                             "max abs diff": g["max_abs_diff"].max(),
                             "identical actions (%)": 100 * g["identical_actions"].min()})
        save(pd.DataFrame(rows), "test1_costs", "Maximum difference between the value functions of SDP-1 and SDP-2 by cost parameter.", tab, FMT)

    E = read("E")
    if E is not None:
        t = E[["pat1", "M", "T", "N", "time_sdp2", "time_sdp1_extrapolated", "predicted_speedup"]].rename(columns={
            "pat1": "pattern", "time_sdp2": "SDP-2 (s)", "time_sdp1_extrapolated": "SDP-1 extrapolated (s)",
            "predicted_speedup": "predicted speed-up"})
        save(t, "test1_large", "SDP-2 on larger instances; SDP-1 time extrapolated from the operation count.", tab, FMT)

    C = read("C")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, figures skipped")
        return
    if C is not None:
        C = C.sort_values("N")
        f, ax = plt.subplots(figsize=(4.5, 3.2))
        ax.plot(C["N"], C["predicted_speedup"], "-", color="0.3", label="predicted (operation count)")
        ax.plot(C["N"], C["measured_speedup"], "o", color="C0", label="measured (time)")
        ax.set_xlabel("inventory levels per location $N$")
        ax.set_ylabel("speed-up of SDP-2 over SDP-1")
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False)
        f.tight_layout()
        for ext in ("pdf", "png"):
            f.savefig(os.path.join(fig, f"test1_speedup_vs_N.{ext}"), dpi=200)
        print(f"\nfigure: {fig}/test1_speedup_vs_N.pdf")
    allrows = pd.concat([d for d in (A, B, C, D) if d is not None and "measured_speedup" in d], ignore_index=True) if any(
        d is not None for d in (A, B, C, D)) else None
    if allrows is not None and len(allrows):
        f, ax = plt.subplots(figsize=(4.5, 3.2))
        ax.scatter(allrows["predicted_speedup"], allrows["measured_speedup"], s=10, color="C0")
        lim = [0, 1.05 * max(allrows["predicted_speedup"].max(), allrows["measured_speedup"].max())]
        ax.plot(lim, lim, "--", color="0.5", lw=1)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("predicted speed-up")
        ax.set_ylabel("measured speed-up")
        f.tight_layout()
        for ext in ("pdf", "png"):
            f.savefig(os.path.join(fig, f"test1_measured_vs_predicted.{ext}"), dpi=200)
        print(f"figure: {fig}/test1_measured_vs_predicted.pdf")


if __name__ == "__main__":
    main()
