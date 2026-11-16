"""
NOTINUSE_guessingOptimalStructure/round2_report.py : aggregates out/round2.csv into the numbers and the LaTeX rows of the numerical section.

    python round2_report.py
"""
from __future__ import annotations

import json
import os

import pandas as pd

from common import OUT

TOL = 1e-6


def main():
    d = pd.read_csv(os.path.join(OUT, "round2.csv"))
    d["V5"] = pd.to_numeric(d["V5"], errors="coerce").fillna(0)
    d["ratio"] = d["K"] / d["R"]
    d["sender_possible"] = d["K"] >= d["R"] + d["v"]
    sets = [("A", "Test bed II, $T=4$"), ("B", "Test bed II, $T=8$"), ("C", "Other costs, $T=4$"),
            ("D", "Outside the test beds, $T=4$")]
    print("instances", d.instance.nunique(), "instance-periods", len(d), "states", int(d.n_states.sum()))
    rows = []
    for key, label in sets + [("all", "All")]:
        g = d if key == "all" else d[d.set == key]
        if not len(g):
            continue
        proved = int(g[["V1", "V2", "V3", "V4a", "V4b", "V4c", "V5"]].sum().sum())
        rows.append(dict(
            label=label, instances=g.instance.nunique(), periods=len(g), states=int(g.n_states.sum()),
            transship=int(g.n_transship.sum()), V1=int(g.V1.sum()), V2=int(g.V2.sum()), V3=int(g.V3.sum()),
            V4=int(g[["V4a", "V4b", "V4c"]].sum().sum()), V5=int(g.V5.sum()), proved=proved,
            Q1=int(g.Q1.sum()), Q2=int(g.Q2.sum()),
            sender=int(g.n_sender_reorders.sum()), receiver=int(g.n_receiver_reorders.sum()),
            sender_when_impossible=int(g.loc[~g.sender_possible, "n_sender_reorders"].sum()),
            reorder_states=int((g.n_both + g.n_only).sum()),
            reorder_below=int((g.n_both_below + g.n_only_below).sum()),
            ship=int((g.n_ship_12 + g.n_ship_21).sum()), ship_beyond=int((g.n_ship_12_beyond + g.n_ship_21_beyond).sum()),
            lines=int(g.lines.sum()), interval_viol=int(g.interval_violations.sum()),
            periods_interval_viol=int((g.interval_violations > 0).sum()),
            instances_interval_viol=int(g.groupby("instance").interval_violations.sum().gt(0).sum()),
            kC=int((g.kappa_C > g.K + TOL).sum()), kCq=int((g.kappa_Cq > g.K + TOL).sum()),
            kC_inst=int(g.assign(f=g.kappa_C > g.K + TOL).groupby("instance").f.any().sum()),
            kCq_inst=int(g.assign(f=g.kappa_Cq > g.K + TOL).groupby("instance").f.any().sum()),
            thr_inc=int(((g.thr1_increases > 0) | (g.thr2_increases > 0)).sum()),
            multi=int(g.S_only_multivalued.sum()), outside=int(g.levels_outside_window.sum()),
            max_targets=int(g.max_targets.max())))
    r = pd.DataFrame(rows)
    pd.set_option("display.width", 250, "display.max_columns", 50)
    print(r.T.to_string())
    print("\nLaTeX rows, Table of checks")
    for x in rows:
        print(f"{x['label']} & {x['instances']} & {x['states']:,} & {x['transship']:,} & {x['V1']} & {x['V2']} & "
              f"{x['V3']} & {x['V4']} & {x['V5']} & {x['Q1']} & {x['Q2']} \\\\")
    print("\nLaTeX rows, descriptive table")
    for x in rows:
        print(f"{x['label']} & {x['instances']} & {100 * x['reorder_below'] / max(1, x['reorder_states']):.1f} & "
              f"{100 * x['ship_beyond'] / max(1, x['ship']):.1f} & {x['sender']:,} & {x['instances_interval_viol']} & "
              f"{100 * x['interval_viol'] / x['lines']:.2f} & {x['kC_inst']} & {x['kCq_inst']} \\\\")
    # by ratio K/R, sets A and B
    ab = d[d.set.isin(["A", "B"])]
    print("\nby K/R (sets A and B): instances, share of lines with interval violation, instances with any, "
          "instances with kappa_Cq > K, sender-reorder states")
    for lab, m in (("K/R <= 2", ab.ratio <= 2), ("2 < K/R <= 4", (ab.ratio > 2) & (ab.ratio <= 4)), ("K/R >= 5", ab.ratio >= 5)):
        g = ab[m]
        print(lab, g.instance.nunique(), f"{100 * g.interval_violations.sum() / g.lines.sum():.2f}",
              int(g.groupby('instance').interval_violations.sum().gt(0).sum()),
              int(g.assign(f=g.kappa_Cq > g.K + TOL).groupby('instance').f.any().sum()), int(g.n_sender_reorders.sum()))
    print("\nlargest K/(R+v) without sender reorder check: sender states where K < R + v:",
          int(d.loc[~d.sender_possible, 'n_sender_reorders'].sum()), "of instances", d.loc[~d.sender_possible].instance.nunique())
    print("min C1_00", d.C1_00.min(), "max seconds", d.seconds.max(), "window sizes", (d.hi - d.lo + 1).min(), (d.hi - d.lo + 1).max(),
          "states per instance-period", d.n_states.min(), d.n_states.max())
    ex = json.load(open(os.path.join(OUT, "round2_examples.json")))
    print(len(ex), "violation examples")
    for e in ex[:20]:
        print(e)


if __name__ == "__main__":
    main()
