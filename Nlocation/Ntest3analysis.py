"""
Ntest3analysis.py : tables and figures of test bed B (Ntest3.py), from results_Ntest3/core.csv, pivot.csv, time.csv.

Run this file after Ntest3.py. It writes into results_Ntest3/tables and results_Ntest3/figures:
    tables/by_locations.csv, .tex     block core: for every number of locations J the saving of the quasi-myopic
                                      policy and of the heuristic, their difference, and how often the heuristic
                                      is cheaper; then the block large with the same instances of the block core
    tables/difference_large.csv, .tex heuristic minus quasi-myopic on these instances, for every J
    tables/by_factor.csv, .tex        block core, all J together: the same measures for every level of every
                                      factor and for three classes of K/R
    tables/difference_by_J.csv, .tex  block core: heuristic minus quasi-myopic for every J (columns) and every
                                      level of K, R, K/R, distribution and relation (rows)
    tables/pivot.csv, .tex            block pivot: the measures at every level of b, v, z, the demand level and T, for every J
    tables/times.csv, .tex            block time: seconds of one decision and of the single-location problems, and
                                      the states that SDP-2 would need
    tables/two_locations.csv, .tex    J = 2 with the optimum of Ntest2.py, if results_Ntest2/core.csv exists:
                                      gaps of the three policies for the classes of K/R
    tables/summary.txt                the numbers quoted in the text
    figures/saving_by_J.png, difference_by_KR.png, time_by_J.png
Measures, all in per cent of the exact cost of the policy without transshipment
    saving              cost without transshipment minus cost of the policy, on the same demand paths
    heuristic - QM      negative when the heuristic is cheaper
    better              share of the instances in which the heuristic is cheaper on the simulated paths
    sig. better/worse   share in which the 95% confidence interval of the paired difference excludes zero
All numbers are measured; nothing is fitted.

    python Ntest3analysis.py            the full results
    python Ntest3analysis.py --quick    the results of Ntest3.py --quick
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
KR_CLASSES = [("$K/R\\le2$", lambda x: x <= 2), ("$2<K/R\\le4$", lambda x: (x > 2) & (x <= 4)),
              ("$K/R\\ge5$", lambda x: x >= 5)]
COLS = ["", "Instances", "Saving QM", "Saving heuristic", "Heuristic - QM", "Better (%)", "Sig. better (%)",
        "Sig. worse (%)"]
FORMATS = ["", "{:d}", "{:.2f}", "{:.2f}", "{:+.2f}", "{:.0f}", "{:.0f}", "{:.0f}"]


def read(folder: Path, name: str) -> pd.DataFrame:
    path = folder / f"{name}.csv"
    if not path.exists():
        print(f"{path} not found")
        return pd.DataFrame()
    d = pd.read_csv(path)
    d["distribution"] = np.where(d["dist"] == "poisson", "Poisson", "Normal, $\\rho=" + d["cv"].astype(str) + "$")
    d["KR"] = d["K"] / d["R"]
    return d


def line(label, d: pd.DataFrame) -> dict:
    diff, hw = d["alg_minus_qm_pct"], d["alg_minus_qm_hw_pct"]
    return {"": str(label), "Instances": len(d), "Saving QM": d["saving_qm_pct"].mean(),
            "Saving heuristic": d["saving_alg_pct"].mean(), "Heuristic - QM": diff.mean(),
            "Better (%)": 100 * (diff < 0).mean(), "Sig. better (%)": 100 * (diff + hw < 0).mean(),
            "Sig. worse (%)": 100 * (diff - hw > 0).mean()}


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


def by_locations(core, large, tables):
    """Block core for every J, then the instances of the block large with the same instances of the block core,
    so that the second part compares the same instances for every J."""
    rows = [{"heading": "Block core"}] + [line(f"{J}", d) for J, d in core.groupby("J")] + [line("All", core)]
    if len(large):
        same = core[(core["dist"] == "poisson") & core["relation"].isin(large["relation"].unique())]
        both = pd.concat([same, large])
        rows += [{"heading": "Block large and the same instances of block core"}]
        rows += [line(f"{J}", d) for J, d in both.groupby("J")]
    save(rows, tables, "by_locations")
    return rows


def by_factor(core, tables):
    rows = []
    for title, col in (("Demand distribution", "distribution"), ("Demand patterns", "relation"),
                       ("Fixed reorder cost $K$", "K"), ("Fixed transshipment cost $R$", "R")):
        rows.append({"heading": title})
        rows += [line(level, d) for level, d in core.groupby(col, sort=True)]
    rows.append({"heading": "Ratio of the fixed costs"})
    rows += [line(label, core[test(core["KR"])]) for label, test in KR_CLASSES]
    rows += [{"heading": "All"}, line("All instances", core)]
    save(rows, tables, "by_factor")
    return rows


def difference_by_J(core, tables):
    """Heuristic minus quasi-myopic: rows are the levels of the factors, columns the numbers of locations."""
    Js = sorted(core["J"].unique())
    rows = []

    def add(label, d):
        r = {"": str(label)}
        for J in Js:
            e = d[d["J"] == J]
            r[f"J={J}"] = e["alg_minus_qm_pct"].mean() if len(e) else np.nan
        rows.append(r)

    for title, col in (("Fixed reorder cost $K$", "K"), ("Fixed transshipment cost $R$", "R"),
                       ("Demand distribution", "distribution"), ("Demand patterns", "relation")):
        rows.append({"heading": title})
        for level, d in core.groupby(col, sort=True):
            add(level, d)
    rows.append({"heading": "Ratio of the fixed costs"})
    for label, test in KR_CLASSES:
        add(label, core[test(core["KR"])])
    rows.append({"heading": "All"})
    add("All instances", core)
    cols = [""] + [f"J={J}" for J in Js]
    save(rows, tables, "difference_by_J", cols, [""] + ["{:+.2f}"] * len(Js))
    return rows


def pivot_table(core, pivot, tables):
    base = core[(core["T"] == 4) & (core["dist"] == "poisson") & (core["relation"] == "LCY1-LCY2")
                & (core["K"] == 30) & (core["R"] == 5)]
    rows = []
    cols = ["", "J", "Saving QM", "Saving heuristic", "Heuristic - QM", "Half-width"]
    for factor, title, value in (("b", "Penalty cost $b$", 5), ("v", "Unit transshipment cost $v$", 0.5),
                                 ("z", "Unit reorder cost $z$", 1), ("level", "Average expected demand per period", 7),
                                 ("T", "Horizon $T$", 4)):
        d = pivot[pivot["factor"] == factor]
        if d.empty:
            continue
        rows.append({"heading": title})
        if factor == "T":                                   # six relations per J: the average of each horizon
            for J in sorted(d["J"].unique()):
                four = core[(core["J"] == J) & (core["dist"] == "poisson") & (core["K"] == 30) & (core["R"] == 5)]
                for label, e in (("4 (base)", four), ("8", d[d["J"] == J])):
                    if len(e):
                        rows.append({"": label, "J": int(J), "Saving QM": e["saving_qm_pct"].mean(),
                                     "Saving heuristic": e["saving_alg_pct"].mean(),
                                     "Heuristic - QM": e["alg_minus_qm_pct"].mean(),
                                     "Half-width": e["alg_minus_qm_hw_pct"].mean()})
            continue
        levels = [(float(r[factor]), r, "") for _, r in d.iterrows()]
        levels += [(float(value), r, " (base)") for _, r in base.iterrows()]
        for x, r, mark in sorted(levels, key=lambda a: (a[1]["J"], a[0])):
            rows.append({"": f"{x:g}{mark}", "J": int(r["J"]), "Saving QM": r["saving_qm_pct"],
                         "Saving heuristic": r["saving_alg_pct"], "Heuristic - QM": r["alg_minus_qm_pct"],
                         "Half-width": r["alg_minus_qm_hw_pct"]})
    if rows:
        save(rows, tables, "pivot", cols, ["", "{:d}", "{:.2f}", "{:.2f}", "{:+.2f}", "{:.2f}"])
    return rows


def times(tm, tables):
    rows = []
    for (T, J), d in tm.groupby(["T", "J"]):
        r = d.iloc[0]
        rows.append({"T": int(T), "J": int(J), "Decision at k=1": r["seconds_decision_k1_alg"],
                     "Decision, average": r["seconds_per_decision_alg"], "Precomputation": r["seconds_precompute_alg"],
                     "QM decision": r["seconds_per_decision_qm"], "Single-location problems": r["seconds_single"],
                     "States of SDP-2": f"{float(r['sdp_states']):.1e}"})
    cols = ["T", "J", "Decision at k=1", "Decision, average", "Precomputation", "QM decision",
            "Single-location problems", "States of SDP-2"]
    save(rows, tables, "times", cols, ["{:d}", "{:d}", "{:.3f}", "{:.3f}", "{:.3f}", "{:.4f}", "{:.2f}", ""])
    return rows


def two_locations(core, tables):
    """J = 2 with the optimum of Ntest2.py. The gap of the heuristic and of the policy without transshipment come
    from Ntest2.py. The gap of the quasi-myopic policy is the gap of the heuristic minus the paired difference
    heuristic - QM of this test bed, expressed in per cent of the optimal cost."""
    path = HERE / "results_Ntest2" / "core.csv"
    if not path.exists():
        return []
    a = pd.read_csv(path)
    keys = ["T", "dist", "cv", "relation", "K", "z", "R", "v", "h", "b", "level"]
    b = core[core["J"] == 2]
    for d in (a, b):
        d["cv"] = d["cv"].fillna(-1)
    m = b.merge(a[keys + ["etc_opt", "gap_pct", "gap_nt_pct"]], on=keys, how="inner")
    if m.empty:
        return []
    m["gap_qm"] = m["gap_pct"] - m["alg_minus_qm_pct"] * m["etc_nt_exact"] / m["etc_opt"]
    rows = []

    def add(label, d):
        rows.append({"": label, "Instances": len(d), "Heuristic": d["gap_pct"].mean(), "Quasi-myopic": d["gap_qm"].mean(),
                     "No transshipment": d["gap_nt_pct"].mean(), "Better (%)": 100 * (d["gap_pct"] < d["gap_qm"]).mean()})

    for label, test in KR_CLASSES:
        add(label, m[test(m["K"] / m["R"])])
    add("All instances", m)
    cols = ["", "Instances", "Heuristic", "Quasi-myopic", "No transshipment", "Better (%)"]
    save(rows, tables, "two_locations", cols, ["", "{:d}", "{:.2f}", "{:.2f}", "{:.2f}", "{:.0f}"])
    return rows


def summary_large(core, large, tables):
    """Heuristic minus quasi-myopic on the instances of the block large, next to the same instances of the block
    core: rows are the classes of K/R and the relations, columns the numbers of locations."""
    same = core[(core["dist"] == "poisson") & core["relation"].isin(large["relation"].unique())]
    both = pd.concat([same, large])
    Js = sorted(both["J"].unique())
    rows = []

    def add(label, d):
        r = {"": str(label)}
        for J in Js:
            e = d[d["J"] == J]
            r[f"J={J}"] = e["alg_minus_qm_pct"].mean() if len(e) else np.nan
        rows.append(r)

    rows.append({"heading": "Ratio of the fixed costs"})
    for label, test in KR_CLASSES:
        add(label, both[test(both["KR"])])
    rows.append({"heading": "Demand patterns"})
    for level, d in both.groupby("relation", sort=True):
        add(level, d)
    rows.append({"heading": "All"})
    add("All instances", both)
    save(rows, tables, "difference_large", [""] + [f"J={J}" for J in Js], [""] + ["{:+.2f}"] * len(Js))
    print(pd.DataFrame([r for r in rows if "heading" not in r]).round(2).to_string(index=False))


def summary(core, tm, tables):
    out = [f"block core: {len(core)} instances, J in {sorted(int(x) for x in core['J'].unique())}"]
    for J, d in core.groupby("J"):
        diff, hw = d["alg_minus_qm_pct"], d["alg_minus_qm_hw_pct"]
        out.append(f"  J = {J}: {len(d)} instances on {int(d['paths'].iloc[0])} paths. Saving against no transshipment: "
                   f"QM {d['saving_qm_pct'].mean():.2f}%, heuristic {d['saving_alg_pct'].mean():.2f}%. Heuristic minus QM "
                   f"{diff.mean():+.2f}% on average, from {diff.min():+.2f}% to {diff.max():+.2f}%. Heuristic cheaper in "
                   f"{(diff < 0).sum()} instances, significantly in {(diff + hw < 0).sum()}; dearer significantly in "
                   f"{(diff - hw > 0).sum()}. Average half-width {hw.mean():.2f}%.")
        strong = d[d["KR"] >= 5]
        weak = d[d["KR"] <= 2]
        out.append(f"         K/R >= 5: heuristic minus QM {strong['alg_minus_qm_pct'].mean():+.2f}% "
                   f"({len(strong)} instances); K/R <= 2: {weak['alg_minus_qm_pct'].mean():+.2f}% ({len(weak)} instances)")
    if len(tm):
        out.append("block time, seconds of one decision of the heuristic at k = 1:")
        for (T, J), d in tm.groupby(["T", "J"]):
            r = d.iloc[0]
            out.append(f"  T = {T}, J = {J}: {r['seconds_decision_k1_alg']:.3f} s; average decision "
                       f"{r['seconds_per_decision_alg']:.3f} s; QM {r['seconds_per_decision_qm']:.4f} s; "
                       f"SDP-2 would need {float(r['sdp_states']):.1e} states per period")
    (tables / "summary.txt").write_text("\n".join(out) + "\n")
    print("\n".join(out))


def figures(core, tm, folder: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; no figures")
        return
    Js = sorted(core["J"].unique())
    x = np.arange(len(Js))
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.bar(x - 0.2, [core[core["J"] == J]["saving_qm_pct"].mean() for J in Js], 0.4, color="0.7", label="Quasi-myopic")
    ax.bar(x + 0.2, [core[core["J"] == J]["saving_alg_pct"].mean() for J in Js], 0.4, color="0.2", label="Heuristic")
    ax.set_xticks(x, [str(J) for J in Js])
    ax.set_xlabel("Number of locations")
    ax.set_ylabel("Saving against no transshipment (%)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(folder / "saving_by_J.png", dpi=200)
    plt.close(fig)

    ratios = sorted(core["KR"].unique())
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for J, style in zip(Js, ("o-", "s-", "^-", "d-", "v-", "x-")):
        ax.plot(range(len(ratios)), [core[(core["J"] == J) & (core["KR"] == r)]["alg_minus_qm_pct"].mean()
                                     for r in ratios], style, label=f"J = {J}")
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xticks(range(len(ratios)), [f"{r:g}" for r in ratios])
    ax.set_xlabel("K / R")
    ax.set_ylabel("Heuristic minus quasi-myopic (%)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(folder / "difference_by_KR.png", dpi=200)
    plt.close(fig)

    if len(tm):
        fig, ax = plt.subplots(figsize=(5, 3.6))
        for T, style in zip(sorted(tm["T"].unique()), ("o-", "s--", "^:")):
            d = tm[tm["T"] == T].sort_values("J")
            ax.semilogy(d["J"], d["seconds_decision_k1_alg"], style, color="k", label=f"Heuristic, T = {T}")
            ax.semilogy(d["J"], d["seconds_per_decision_qm"], style, color="0.6", label=f"Quasi-myopic, T = {T}")
        ax.set_xticks(sorted(int(J) for J in tm["J"].unique()))
        ax.set_xlabel("Number of locations")
        ax.set_ylabel("Seconds for one decision")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(folder / "time_by_J.png", dpi=200)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    folder = HERE / ("results_Ntest3_quick" if a.quick else "results_Ntest3")
    core, pivot, tm = read(folder, "core"), read(folder, "pivot"), read(folder, "time")
    large = read(folder, "large")
    if core.empty:
        print("no results of the block core; run Ntest3.py first")
        return
    tables, figs = folder / "tables", folder / "figures"
    tables.mkdir(exist_ok=True)
    figs.mkdir(exist_ok=True)
    rows = by_locations(core, large, tables)
    print(pd.DataFrame([r for r in rows if "heading" not in r]).round(2).to_string(index=False))
    if len(large):
        summary_large(core, large, tables)
    rows = by_factor(core, tables)
    print(pd.DataFrame([r for r in rows if "heading" not in r]).round(2).to_string(index=False))
    rows = difference_by_J(core, tables)
    print(pd.DataFrame([r for r in rows if "heading" not in r]).round(2).to_string(index=False))
    if not pivot.empty:
        pivot_table(core, pivot, tables)
    if not tm.empty:
        print(pd.DataFrame(times(tm, tables)).to_string(index=False))
    rows = two_locations(core, tables)
    if rows:
        print(pd.DataFrame(rows).round(2).to_string(index=False))
    summary(core, tm, tables)
    figures(core, tm, figs)
    print("tables ->", tables, "\nfigures ->", figs)


if __name__ == "__main__":
    main()
