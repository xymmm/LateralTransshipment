"""
Ntest2analysis.py : tables and figures of test bed A (Ntest2.py), from results_Ntest2/core.csv, patterns.csv
and pivot.csv.

Run this file after Ntest2.py. It writes into results_Ntest2/tables and results_Ntest2/figures:
    tables/gap_by_factor.csv, .tex    block core: instances, average and largest gap of the heuristic, and the
                                      average gap of the policy without transshipment, for every level of every
                                      factor, for three classes of K/R and for six classes of the value of
                                      transshipment
    tables/gap_patterns.csv, .tex     the 20 pairs of demand patterns at T = 4 (blocks core and patterns): the
                                      average gap for every pair under each distribution
    tables/gap_patterns_T8.csv, .tex  the same pairs at T = 8 with K = 30 and R = 5
    tables/gap_pivot.csv, .tex        block pivot: the gap at every level of b, v, z and the demand level, next to the base
    tables/summary.txt                the numbers quoted in the text: gaps, share of instances below 1%, paths,
                                      times, the check of the inventory levels, and the instances deleted by the
                                      one-hour rule
    figures/gap_by_KR.png             the gap against the ratio K/R, heuristic and no transshipment
    figures/gap_against_value.png     the gap of the heuristic against the value of transshipment
The gap of the heuristic is (ETC - C) / C in per cent, with C the optimal expected total cost of SDP-2. The value
of transshipment is the gap of the policy without transshipment. An instance whose SDP-2 run took more than one
hour (over_hour) is left out. All numbers are measured; nothing is fitted.

    python Ntest2analysis.py            the full results
    python Ntest2analysis.py --quick    the results of Ntest2.py --quick
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
KR_CLASSES = [("$K/R\\le2$", lambda x: x <= 2), ("$2<K/R\\le4$", lambda x: (x > 2) & (x <= 4)),
              ("$K/R\\ge5$", lambda x: x >= 5)]
VALUE_CLASSES = [("$\\le0.05$", -np.inf, 0.05), ("0.05--0.5", 0.05, 0.5), ("0.5--1", 0.5, 1), ("1--2", 1, 2),
                 ("2--5", 2, 5), ("$>5$", 5, np.inf)]
COLS = ["", "Instances", "Heuristic", "Largest", "Paired", "No transshipment", "Below 1% (%)"]
FORMATS = ["", "{:d}", "{:.2f}", "{:.2f}", "{:.2f}", "{:.2f}", "{:.0f}"]


def read(folder: Path, name: str) -> pd.DataFrame:
    path = folder / f"{name}.csv"
    if not path.exists():
        print(f"{path} not found")
        return pd.DataFrame()
    d = pd.read_csv(path)
    if "over_hour" in d:
        d = d[d["over_hour"].fillna(0) == 0].copy()
    d["distribution"] = np.where(d["dist"] == "poisson", "Poisson", "Normal, $\\rho=" + d["cv"].astype(str) + "$")
    d["KR"] = d["K"] / d["R"]
    return d


def line(label, d: pd.DataFrame) -> dict:
    """One row of a table: the measures of a set of instances."""
    return {"": str(label), "Instances": len(d), "Heuristic": d["gap_pct"].mean(), "Largest": d["gap_pct"].max(),
            "Paired": d["paired_gap_pct"].mean(), "No transshipment": d["gap_nt_pct"].mean(),
            "Below 1% (%)": 100 * (d["gap_pct"] < 1).mean()}


def to_tex(rows, path: Path, columns, formats):
    """Rows of a LaTeX table; a row with one key is a heading."""
    out = ["% " + " & ".join(columns)]
    for r in rows:
        if list(r) == ["heading"]:
            out.append("\\multicolumn{" + str(len(columns)) + "}{l}{\\textit{" + r["heading"] + "}}\\\\")
            continue
        cells = []
        for c, fmt in zip(columns, formats):
            x = r.get(c, "")
            cells.append(fmt.format(x) if isinstance(x, (int, float, np.integer, np.floating)) and fmt else str(x))
        out.append(" & ".join(cells) + " \\\\")
    path.write_text("\n".join(out) + "\n")


def save(rows, tables: Path, name: str, columns=COLS, formats=FORMATS):
    rows = [r for r in rows if r.get("Instances", 1) > 0]            # a class without instances is left out
    pd.DataFrame([r for r in rows if "heading" not in r])[columns].to_csv(tables / f"{name}.csv", index=False)
    to_tex(rows, tables / f"{name}.tex", columns, formats)
    return rows


def gap_by_factor(core: pd.DataFrame, tables: Path):
    rows = []
    groups = [("Number of locations $J$", "J"), ("Horizon $T$", "T"), ("Demand distribution", "distribution"),
              ("Demand patterns", "relation"), ("Fixed reorder cost $K$", "K"), ("Fixed transshipment cost $R$", "R")]
    for title, col in groups:
        if col == "J" and core["J"].nunique() == 1:
            continue
        rows.append({"heading": title})
        rows += [line(level, d) for level, d in core.groupby(col, sort=True)]
    rows.append({"heading": "Ratio of the fixed costs"})
    rows += [line(label, core[test(core["KR"])]) for label, test in KR_CLASSES]
    rows.append({"heading": "Gap of the policy without transshipment (\\%)"})
    rows += [line(label, core[(core["gap_nt_pct"] > lo) & (core["gap_nt_pct"] <= hi)]) for label, lo, hi in VALUE_CLASSES]
    rows += [{"heading": "All"}, line("All instances", core)]
    return save(rows, tables, "gap_by_factor")


def gap_patterns(d: pd.DataFrame, tables: Path, name: str):
    """Rows: the pairs of patterns. Columns: the distributions, their average and the value of transshipment."""
    if d.empty:
        return
    wide = d.pivot_table(index="relation", columns="distribution", values="gap_pct", sort=False)
    wide["Average"] = d.groupby("relation", sort=False)["gap_pct"].mean()
    wide["Largest"] = d.groupby("relation", sort=False)["gap_pct"].max()
    wide["No transshipment"] = d.groupby("relation", sort=False)["gap_nt_pct"].mean()
    wide["Instances"] = d.groupby("relation", sort=False)["gap_pct"].size()
    wide = wide.sort_index()
    total = {c: d[d["distribution"] == c]["gap_pct"].mean() for c in d["distribution"].unique()}
    total.update(Average=d["gap_pct"].mean(), Largest=d["gap_pct"].max(), **{"No transshipment": d["gap_nt_pct"].mean()},
                 Instances=len(d))
    wide.loc["All pairs"] = pd.Series(total)
    wide.to_csv(tables / f"{name}.csv")
    tex = ["% pair & " + " & ".join(str(c) for c in wide.columns)]
    for i, r in wide.iterrows():
        tex.append(str(i) + " & " + " & ".join(f"{int(x):d}" if c == "Instances" else f"{x:.2f}"
                                               for c, x in r.items()) + " \\\\")
    (tables / f"{name}.tex").write_text("\n".join(tex) + "\n")


def gap_pivot(core: pd.DataFrame, pivot: pd.DataFrame, tables: Path):
    """The pivot block: every level of b, v, z and the demand level next to the base instance."""
    rows = []
    for J in sorted(pivot["J"].unique()):
        base = core[(core["J"] == J) & (core["T"] == 4) & (core["dist"] == "poisson")
                    & (core["relation"] == "LCY1-LCY2") & (core["K"] == 30) & (core["R"] == 5)]
        for factor, title, value in (("b", "Penalty cost $b$", 5), ("v", "Unit transshipment cost $v$", 0.5),
                                     ("z", "Unit reorder cost $z$", 1), ("level", "Average expected demand per period", 7)):
            d = pivot[(pivot["factor"] == factor) & (pivot["J"] == J)]
            if d.empty:
                continue
            rows.append({"heading": title + (f", $J={J}$" if pivot["J"].nunique() > 1 else "")})
            levels = [(float(r[factor]), r, "") for _, r in d.iterrows()]
            levels += [(float(value), r, " (base)") for _, r in base.iterrows()]
            for x, r, mark in sorted(levels, key=lambda a: a[0]):
                rows.append({"": f"{x:g}{mark}", "Heuristic": r["gap_pct"], "Half-width": r["gap_hw_pct"],
                             "Paired": r["paired_gap_pct"], "No transshipment": r["gap_nt_pct"],
                             "States": int(r["states"])})
    if rows:
        save(rows, tables, "gap_pivot", ["", "Heuristic", "Half-width", "Paired", "No transshipment", "States"],
             ["", "{:.2f}", "{:.2f}", "{:.2f}", "{:.2f}", "{:d}"])


def summary(blocks: dict, folder: Path, tables: Path):
    out = []
    for name, d in blocks.items():
        deleted = folder / f"{name}_deleted.csv"
        if deleted.exists():
            x = pd.read_csv(deleted)
            for J, e in x.groupby("J"):
                out.append(f"block {name}, J = {J}: {len(e)} instances deleted by the one-hour rule; their shipment "
                           f"vectors over the horizon are between {e['vectors'].astype(float).min():.1e} and "
                           f"{e['vectors'].astype(float).max():.1e}, their states per period between "
                           f"{e['states'].astype(float).min():.1e} and {e['states'].astype(float).max():.1e}")
        if d.empty:
            continue
        g = d["gap_pct"]
        out += [f"block {name}: {len(d)} instances, J in {sorted(int(j) for j in d['J'].unique())}",
                f"  gap of the heuristic: average {g.mean():.3f}%, median {g.median():.3f}%, "
                f"95% of the instances below {g.quantile(0.95):.3f}%, largest {g.max():.3f}%",
                f"  instances with a gap below 1%: {(g < 1).sum()} of {len(d)}; below 0.5%: {(g < 0.5).sum()}",
                f"  paired gap on the same demand paths: average {d['paired_gap_pct'].mean():.3f}%, "
                f"largest half-width {d['paired_gap_hw_pct'].max():.3f}%",
                f"  gap of the policy without transshipment: average {d['gap_nt_pct'].mean():.3f}%, "
                f"from {d['gap_nt_pct'].min():.3f}% to {d['gap_nt_pct'].max():.3f}%",
                f"  demand paths of the heuristic: from {int(d['paths'].min()):,} to {int(d['paths'].max()):,}; "
                f"precision of 0.1% reached in {int(d['attained'].astype(str).eq('True').sum())} instances",
                f"  states of SDP-2 per period: from {int(d['states'].min()):,} to {int(d['states'].max()):,}; "
                f"inventory levels from [{int(d['lo'].max())}, {int(d['hi'].min())}] to [{int(d['lo'].min())}, "
                f"{int(d['hi'].max())}]"]
        for T, e in d.groupby("T"):
            out.append(f"  T = {T}: SDP-2 {e['seconds_sdp2'].mean():.1f} s on average (largest {e['seconds_sdp2'].max():.1f} s); "
                       f"one decision of the heuristic at k = 1: {e['seconds_decision_k1'].mean():.3f} s on average "
                       f"(largest {e['seconds_decision_k1'].max():.3f} s)")
        chk = d.dropna(subset=["change_widened"]) if "change_widened" in d else pd.DataFrame()
        if len(chk):
            out.append(f"  inventory levels moved out by 25% on {len(chk)} instances: largest relative change of the "
                       f"optimal cost {chk['change_widened'].max():.1e}")
    (tables / "summary.txt").write_text("\n".join(out) + "\n")
    print("\n".join(out))


def figures(core: pd.DataFrame, folder: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; no figures")
        return
    ratios = sorted(core["KR"].unique())
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(range(len(ratios)), [core[core["KR"] == r]["gap_nt_pct"].mean() for r in ratios], "s--", color="0.5",
            label="No transshipment")
    ax.boxplot([core[core["KR"] == r]["gap_pct"] for r in ratios], positions=range(len(ratios)), widths=0.6,
               flierprops=dict(markersize=3))
    ax.plot([], [], color="k", label="Heuristic")
    ax.set_xticks(range(len(ratios)), [f"{r:g}" for r in ratios])
    ax.set_xlabel("K / R")
    ax.set_ylabel("Gap (%)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(folder / "gap_by_KR.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    ax.scatter(core["gap_nt_pct"], core["gap_pct"], s=10, color="k", alpha=0.6)
    ax.set_xlabel("Gap of the policy without transshipment (%)")
    ax.set_ylabel("Gap of the heuristic (%)")
    fig.tight_layout()
    fig.savefig(folder / "gap_against_value.png", dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    folder = HERE / ("results_Ntest2_quick" if a.quick else "results_Ntest2")
    core, patterns, pivot = read(folder, "core"), read(folder, "patterns"), read(folder, "pivot")
    if core.empty:
        print("no results of the block core; run Ntest2.py first")
        return
    tables, figs = folder / "tables", folder / "figures"
    tables.mkdir(exist_ok=True)
    figs.mkdir(exist_ok=True)
    rows = gap_by_factor(core, tables)
    print(pd.DataFrame([r for r in rows if "heading" not in r]).round(3).to_string(index=False))
    pairs = pd.concat([core[~core["relation"].str.endswith("r") & ~core["relation"].str.endswith("/3")], patterns])
    gap_patterns(pairs[pairs["T"] == 4], tables, "gap_patterns")
    gap_patterns(pairs[(pairs["T"] == 8) & (pairs["K"] == 30) & (pairs["R"] == 5)], tables, "gap_patterns_T8")
    if not pivot.empty:
        gap_pivot(core, pivot, tables)
    summary(dict(core=core, patterns=patterns, pivot=pivot), folder, tables)
    figures(core, figs)
    print("tables ->", tables, "\nfigures ->", figs)


if __name__ == "__main__":
    main()
