"""
Ntest3analysis.py : tables and figures of the main test bed (Ntest3.py), from results_Ntest3_v2.

Run this file after Ntest3.py. It uses compatible blocks with experiment metadata and writes tables and figures
inside the results folder:
    tables/by_locations.csv, .tex     block core for every J; then the block large with the same instances of
                                      the block core, so that J = 2 to 10 are compared on common instances
    tables/by_factor.csv, .tex        block core, all J together: every level of every factor, three classes of K/R
    tables/reduction_by_J.csv, .tex   reduction against QM for every J (columns) and every level of every factor
                                      (rows), block core
    tables/reduction_large.csv, .tex  the same common cases for J = 2 to 10
    tables/patterns.csv, .tex         block patterns: the 17 other pairs, for every J
    tables/pivot.csv, .tex            block pivot: every level of b, v and z next to the base, for every J
    tables/level.csv, .tex            block level: demand levels 5, 7, 10, 15, for every J
    tables/horizon.csv, .tex          block horizon: T = 4 against T = 8 on the same instances
    tables/times.csv, .tex            block time: seconds of one decision and of the single-location problems, and
                                      the states that SDP-2 would need
    tables/summary.txt                the numbers quoted in the text
    figures/saving_by_J.png, reduction_by_KR.png, time_by_J.png
Measures
    saving              cost without transshipment minus cost of the policy, in per cent of the exact cost
                        without transshipment, on the same demand paths
    reduction           cost of QM minus cost of the heuristic, in per cent of the cost of QM: positive when the
                        heuristic is cheaper
    better              share of the instances in which the heuristic is cheaper on the simulated paths
    sig. better/worse   share in which the 95% confidence interval of the paired difference excludes zero
All numbers are measured; nothing is fitted.

    python Ntest3analysis.py            the full results
    python Ntest3analysis.py --quick    the results of Ntest3.py --quick
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from Ntest3 import OUT_DIR, FIELDS, SCHEMA_VERSION

HERE = Path(__file__).resolve().parent
KR_CLASSES = [("$K/R\\le2$", lambda x: x <= 2), ("$2<K/R\\le4$", lambda x: (x > 2) & (x <= 4)),
              ("$K/R\\ge5$", lambda x: x >= 5)]
COLS = ["", "Instances", "Saving QM", "Saving heuristic", "Reduction", "Better (%)", "Sig. better (%)",
        "Sig. worse (%)"]
FORMATS = ["", "{:d}", "{:.2f}", "{:.2f}", "{:+.2f}", "{:.0f}", "{:.0f}", "{:.0f}"]
FACTORS = (("Fixed reorder cost $K$", "K"), ("Fixed transshipment cost $R$", "R"),
           ("Demand distribution", "distribution"), ("Demand patterns", "relation"))
MATCH = ["T", "dist", "cv", "relation", "K", "z", "R", "v", "h", "b", "level"]


def read(folder: Path, name: str) -> pd.DataFrame:
    path = folder / f"{name}.csv"
    if not path.exists():
        return pd.DataFrame()
    metadata_path = path.with_suffix(".meta.json")
    if not metadata_path.exists():
        raise ValueError(f"{path} has no experiment metadata. Analyse a results_Ntest3_v2 run instead of legacy files.")
    metadata = json.loads(metadata_path.read_text())
    if metadata["shared"]["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"Incompatible results schema in {path}")
    d = pd.read_csv(path)
    if list(d.columns) != FIELDS or d["id"].duplicated().any():
        raise ValueError(f"Incompatible columns or duplicate instances in {path}")
    if len(d) != len(metadata["instances"]):
        print(f"{name}: {len(d)}/{len(metadata['instances'])} instances completed; comparisons use matched cases.")
    d["cv"] = d["cv"].fillna(0.0)
    d["distribution"] = np.where(d["dist"] == "poisson", "Poisson", "Normal, $\\rho=" + d["cv"].astype(str) + "$")
    d["KR"] = d["K"] / d["R"]
    d.attrs["experiment"] = metadata["shared"]
    return d


def check_compatible_blocks(blocks):
    configurations = [d.attrs["experiment"] for d in blocks.values() if not d.empty]
    if configurations and any(c != configurations[0] for c in configurations[1:]):
        raise ValueError("Blocks use different source versions, sampling settings or environments; do not combine them.")


def line(label, d: pd.DataFrame) -> dict:
    x = d["reduction_qm_pct"]
    difference, hw = d["difference_qm_alg"], d["difference_qm_alg_hw"]
    return {"": str(label), "Instances": len(d), "Saving QM": d["saving_qm_pct"].mean(),
            "Saving heuristic": d["saving_alg_pct"].mean(), "Reduction": x.mean(),
            "Better (%)": 100 * (difference > 0).mean(), "Sig. better (%)": 100 * (difference - hw > 0).mean(),
            "Sig. worse (%)": 100 * (difference + hw < 0).mean()}


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
            if isinstance(x, (float, np.floating)) and np.isnan(x):
                cells.append("--")
            else:
                cells.append(fmt.format(x) if isinstance(x, (int, float, np.integer, np.floating)) and fmt else str(x))
        out.append(" & ".join(cells) + " \\\\")
    path.write_text("\n".join(out) + "\n")


def save(rows, tables: Path, name: str, columns=COLS, formats=FORMATS):
    rows = [r for r in rows if r.get("Instances", 1) > 0]            # a class without instances is left out
    body = [r for r in rows if "heading" not in r]
    if not body:
        return rows
    pd.DataFrame(body).reindex(columns=columns).to_csv(tables / f"{name}.csv", index=False)
    to_tex(rows, tables / f"{name}.tex", columns, formats)
    return rows


def show(rows):
    body = [r for r in rows if "heading" not in r]
    if body:
        print(pd.DataFrame(body).round(2).to_string(index=False), "\n")


def same_instances(core: pd.DataFrame, other: pd.DataFrame, match_locations=True) -> pd.DataFrame:
    """Match within the same J unless a cross-location comparison is explicitly requested."""
    columns = (["J"] if match_locations else []) + MATCH
    keys = other[columns].drop_duplicates()
    return core.merge(keys, on=columns, how="inner", validate="many_to_one")


def common_location_cases(data, locations=None):
    """Keep only factor combinations represented at every compared number of locations."""
    if data.empty:
        return data
    locations = sorted(data["J"].unique()) if locations is None else list(locations)
    data = data[data["J"].isin(locations)]
    keys = data[["J"] + MATCH].drop_duplicates()
    counts = keys.groupby(MATCH, dropna=False)["J"].nunique().reset_index(name="location_count")
    common = counts[counts["location_count"] == len(locations)][MATCH]
    return data.merge(common, on=MATCH, how="inner", validate="many_to_one")


def matched_large(core, large):
    if large.empty:
        return core.iloc[:0].copy()
    locations = sorted(set(core["J"]) | set(large["J"]))
    candidates = pd.concat([same_instances(core, large, match_locations=False), large], ignore_index=True)
    return common_location_cases(candidates, locations)


def matched_horizons(core, horizon):
    """Match both sides on J and all other factors, including when a run is only partly complete."""
    four = same_instances(core, horizon.assign(T=4))
    eight = same_instances(horizon, four.assign(T=8))
    return pd.concat([four, eight], ignore_index=True)


def by_locations(core, large, tables):
    rows = [{"heading": "Block core"}] + [line(J, d) for J, d in core.groupby("J")] + [line("All", core)]
    if len(large):
        rows += [{"heading": "Block large"}] + [line(J, d) for J, d in large.groupby("J")]
        both = matched_large(core, large)
        if not both.empty:
            rows += [{"heading": "Common instances across all compared numbers of locations"}]
            rows += [line(J, d) for J, d in both.groupby("J")]
    return save(rows, tables, "by_locations")


def by_factor(core, tables):
    rows = []
    for title, col in FACTORS:
        rows.append({"heading": title})
        rows += [line(level, d) for level, d in core.groupby(col, sort=True)]
    rows.append({"heading": "Ratio of the fixed costs"})
    rows += [line(label, core[test(core["KR"])]) for label, test in KR_CLASSES]
    rows += [{"heading": "All"}, line("All instances", core)]
    return save(rows, tables, "by_factor")


def reduction_by_J(d, tables, name):
    """Reduction against QM: rows are the levels of the factors, columns the numbers of locations."""
    Js = sorted(d["J"].unique())
    rows = []

    def add(label, e):
        r = {"": str(label)}
        for J in Js:
            x = e[e["J"] == J]["reduction_qm_pct"]
            r[f"J={J}"] = x.mean() if len(x) else np.nan
        rows.append(r)

    for title, col in FACTORS:
        rows.append({"heading": title})
        for level, e in d.groupby(col, sort=True):
            add(level, e)
    rows.append({"heading": "Ratio of the fixed costs"})
    for label, test in KR_CLASSES:
        add(label, d[test(d["KR"])])
    rows.append({"heading": "All"})
    add("All instances", d)
    return save(rows, tables, name, [""] + [f"J={J}" for J in Js], [""] + ["{:+.2f}"] * len(Js))


def by_J_table(d, tables, name, row_col, title, labels=None):
    """Rows: the levels of one column. For every J the saving of QM, the saving of the heuristic and the reduction."""
    Js = sorted(d["J"].unique())
    cols = [""] + [f"{m} J={J}" for J in Js for m in ("QM", "ALG", "Red.")]
    rows = [{"heading": title}]
    for level, e in d.groupby(row_col, sort=True):
        r = {"": labels(level) if labels else str(level)}
        for J in Js:
            x = e[e["J"] == J]
            if len(x):
                r[f"QM J={J}"], r[f"ALG J={J}"] = x["saving_qm_pct"].mean(), x["saving_alg_pct"].mean()
                r[f"Red. J={J}"] = x["reduction_qm_pct"].mean()
        rows.append(r)
    return save(rows, tables, name, cols, [""] + ["{:.2f}", "{:.2f}", "{:+.2f}"] * len(Js))


def pivot_table(core, pivot, tables):
    """Every level of b, v and z next to the base instance, for every J and both cost settings."""
    rows = []
    cols = ["", "J", "K", "R", "Saving QM", "Saving heuristic", "Reduction", "Half-width"]
    for factor, title, value in (("b", "Penalty cost $b$", 5), ("v", "Unit transshipment cost $v$", 0.5),
                                 ("z", "Unit reorder cost $z$", 1)):
        d = pivot[pivot["factor"] == factor]
        if d.empty:
            continue
        rows.append({"heading": title})
        base = same_instances(core, d.assign(**{factor: value}))
        base = base[base["J"].isin(d["J"].unique())]
        both = pd.concat([d.assign(mark=""), base.assign(mark=" (base)")])
        for _, r in both.sort_values(["J", "K", factor]).iterrows():
            rows.append({"": f"{r[factor]:g}{r['mark']}", "J": int(r["J"]), "K": int(r["K"]), "R": int(r["R"]),
                         "Saving QM": r["saving_qm_pct"], "Saving heuristic": r["saving_alg_pct"],
                         "Reduction": r["reduction_qm_pct"], "Half-width": r["reduction_qm_hw_pct"]})
    return save(rows, tables, "pivot", cols, ["", "{:d}", "{:d}", "{:d}", "{:.2f}", "{:.2f}", "{:+.2f}", "{:.2f}"])


def times(tm, tables):
    rows = []
    for (T, J), d in tm.groupby(["T", "J"]):
        r = d.iloc[0]
        rows.append({"T": int(T), "J": int(J), "Decision at k=1": r["seconds_decision_k1_alg"],
                     "Decision, average": r["seconds_per_decision_alg"], "Precomputation": r["seconds_precompute_alg"],
                     "Model construction, total (included)": r["seconds_build_alg"],
                     "QM decision": r["seconds_per_decision_qm"], "Single-location problems": r["seconds_single"],
                     "States of SDP-2": f"{float(r['sdp_states']):.1e}"})
    cols = ["T", "J", "Decision at k=1", "Decision, average", "Precomputation", "Model construction, total (included)", "QM decision",
            "Single-location problems", "States of SDP-2"]
    return save(rows, tables, "times", cols, ["{:d}", "{:d}", "{:.3f}", "{:.3f}", "{:.3f}", "{:.3f}", "{:.4f}", "{:.2f}", ""])


def summary(blocks: dict, tables: Path):
    out = []
    for name, d in blocks.items():
        if d.empty or name == "time":
            continue
        out.append(f"block {name}: {len(d)} instances")
        for J, e in d.groupby("J"):
            x, hw = e["reduction_qm_pct"], e["reduction_qm_hw_pct"]
            difference, difference_hw = e["difference_qm_alg"], e["difference_qm_alg_hw"]
            strong, weak = e[e["KR"] >= 5], e[e["KR"] <= 2]
            text = (f"  J = {J}: {len(e)} instances on {int(e['paths'].iloc[0])} paths. Saving against no "
                    f"transshipment: QM {e['saving_qm_pct'].mean():.2f}%, heuristic {e['saving_alg_pct'].mean():.2f}%. "
                    f"Reduction against QM {x.mean():+.2f}% on average, from {x.min():+.2f}% to {x.max():+.2f}%. "
                    f"Heuristic cheaper in {(difference > 0).sum()} instances, significantly in "
                    f"{(difference - difference_hw > 0).sum()}; dearer significantly in "
                    f"{(difference + difference_hw < 0).sum()}. Average ratio half-width {hw.mean():.2f}%.")
            text += (f" Fixed sampling: ALG meets the precision target in "
                     f"{int(e['precision_attained_alg'].sum())}/{len(e)} instances; "
                     f"{int(e['decisions_at_limit'].sum())} planning solves reached the time limit; "
                     f"largest final MIP gap {e['mip_gap_max'].max():.4g}.")
            if len(strong):
                text += f" K/R >= 5: {strong['reduction_qm_pct'].mean():+.2f}% ({len(strong)} instances)."
            if len(weak):
                text += f" K/R <= 2: {weak['reduction_qm_pct'].mean():+.2f}% ({len(weak)} instances)."
            out.append(text)
    tm = blocks.get("time", pd.DataFrame())
    if len(tm):
        out.append("block time, seconds of one decision of the heuristic at k = 1:")
        for (T, J), d in tm.groupby(["T", "J"]):
            r = d.iloc[0]
            out.append(f"  T = {T}, J = {J}: {r['seconds_decision_k1_alg']:.3f} s; average decision "
                       f"{r['seconds_per_decision_alg']:.3f} s; QM {r['seconds_per_decision_qm']:.4f} s; "
                       f"SDP-2 would need {float(r['sdp_states']):.1e} states per period")
    (tables / "summary.txt").write_text("\n".join(out) + "\n")
    print("\n".join(out))


def figures(core, large, tm, folder: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; no figures")
        return
    d = matched_large(core, large) if len(large) else common_location_cases(core)
    if d.empty:
        print("No factor combinations common to all compared location counts; cross-location figures are skipped.")
        for name in ("saving_by_J.png", "reduction_by_KR.png"):
            (folder / name).unlink(missing_ok=True)
        # Timing figures still have a valid interpretation independently of cost matching.
        time_figure(tm, folder)
        return
    Js = sorted(d["J"].unique())
    x = np.arange(len(Js))
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.bar(x - 0.2, [d[d["J"] == J]["saving_qm_pct"].mean() for J in Js], 0.4, color="0.7", label="Quasi-myopic")
    ax.bar(x + 0.2, [d[d["J"] == J]["saving_alg_pct"].mean() for J in Js], 0.4, color="0.2", label="Heuristic")
    ax.set_xticks(x, [str(J) for J in Js])
    ax.set_xlabel("Number of locations")
    ax.set_ylabel("Saving against no transshipment (%)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(folder / "saving_by_J.png", dpi=200)
    plt.close(fig)

    ratios = sorted(d["KR"].unique())
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for J, style in zip(Js, ("o-", "s-", "^-", "d-", "v-", "x-", "+-")):
        ax.plot(range(len(ratios)), [d[(d["J"] == J) & (d["KR"] == r)]["reduction_qm_pct"].mean() for r in ratios],
                style, label=f"J = {J}")
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xticks(range(len(ratios)), [f"{r:g}" for r in ratios])
    ax.set_xlabel("K / R")
    ax.set_ylabel("Reduction against quasi-myopic (%)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(folder / "reduction_by_KR.png", dpi=200)
    plt.close(fig)

    time_figure(tm, folder)


def time_figure(tm, folder):
    if len(tm):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 3.6))
        for T, style in zip(sorted(tm["T"].unique()), ("o-", "s--", "^:")):
            e = tm[tm["T"] == T].sort_values("J")
            ax.semilogy(e["J"], e["seconds_decision_k1_alg"], style, color="k", label=f"Heuristic at k=1, T = {T}")
            ax.semilogy(e["J"], e["seconds_per_decision_qm"], style, color="0.6", label=f"QM average, T = {T}")
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
    ap.add_argument("--output-dir", type=Path, help="results folder to analyse")
    a = ap.parse_args()
    folder = a.output_dir if a.output_dir is not None else Path(str(OUT_DIR) + ("_quick" if a.quick else ""))
    names = ("core", "large", "patterns", "pivot", "level", "horizon", "time")
    b = {n: read(folder, n) for n in names}
    check_compatible_blocks(b)
    core = b["core"]
    if core.empty:
        print("no results of the block core; run Ntest3.py first")
        return
    tables, figs = folder / "tables", folder / "figures"
    tables.mkdir(exist_ok=True)
    figs.mkdir(exist_ok=True)
    show(by_locations(core, b["large"], tables))
    show(by_factor(core, tables))
    show(reduction_by_J(core, tables, "reduction_by_J"))
    if len(b["large"]):
        both = matched_large(core, b["large"])
        if not both.empty:
            show(reduction_by_J(both, tables, "reduction_large"))
        else:
            print("No common cases across all location counts; the matched large-instance table is skipped.")
            for suffix in ("csv", "tex"):
                (tables / f"reduction_large.{suffix}").unlink(missing_ok=True)
    if len(b["patterns"]):
        show(by_J_table(b["patterns"], tables, "patterns", "relation", "Pairs of demand patterns"))
    if len(b["pivot"]):
        show(pivot_table(core, b["pivot"], tables))
    if len(b["level"]):
        seven = same_instances(core, b["level"].assign(level=7))
        show(by_J_table(pd.concat([b["level"], seven]), tables, "level", "level", "Average expected demand per period",
                        lambda x: f"{int(x)}"))
    if len(b["horizon"]):
        both = matched_horizons(core, b["horizon"])
        show(by_J_table(both, tables, "horizon", "T", "Horizon $T$", lambda x: f"{int(x)}"))
    if len(b["time"]):
        show(times(b["time"], tables))
    summary(b, tables)
    figures(core, b["large"], b["time"], figs)
    print("tables ->", tables, "\nfigures ->", figs)


if __name__ == "__main__":
    main()
