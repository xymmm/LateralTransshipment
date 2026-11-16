"""
NOTINUSE_guessingOptimalStructure/round5_report.py : tables for the stress test of round5_extreme.py.

    python round5_report.py

Reads out/round5b.csv and prints the numbers used in the paper.
"""
from __future__ import annotations

import json
import os

import pandas as pd

from common import OUT

d = pd.read_csv(os.path.join(OUT, "round5b.csv"))
d["KleRv"] = d.K <= d.R + d.v
inst = d.groupby("instance").first()
print(f"instances {d.instance.nunique()}  (corners {inst[inst.set == 'corners'].shape[0]}, random2 {inst[inst.set == 'random2'].shape[0]})")
print(f"instance-periods {len(d)}   states {int(d.n_states.sum()):,}   transshipment states {int(d.n_transship.sum()):,}")
print(f"instance-periods before the last period {int((d.t < d['T']).sum())}")
print("ranges: K", d.K.min(), d.K.max(), "| R", d.R.min(), d.R.max(), "| v", d.v.min(), d.v.max(), "| b", d.b.min(), d.b.max(),
      "| h", d.h.min(), d.h.max(), "| T", sorted(d["T"].unique()), "| window width", (d.hi - d.lo).min(), (d.hi - d.lo).max())
print("demand models:", inst.groupby(["dist", "cv"], dropna=False).size().to_dict())
print("instances with K <= R + v:", int(inst.KleRv.sum()), " with K < R + v:", int((inst.K < inst.R + inst.v).sum()))
print("seconds: total", round(inst.seconds.sum() / 3600, 2), "h   max", inst.seconds.max())

print("\n-- proved statements (each must be 0)")
V = ["V1", "V2", "V3", "V4a", "V4b", "V4c", "V5t"]
print(d[V].fillna(0).sum().astype(int).to_dict())
print("V5 of round2 (compares with a solver that reorders in a tie):", int(d.V5.fillna(0).sum()), "flags; confirmed by V5t:",
      int(d.V5t.fillna(0).sum()))
print("levels outside the window:", int(d.levels_outside_window.sum()))
print("reorder states:", int((d.n_both + d.n_only).sum()), " below their level:", int((d.n_both_below + d.n_only_below).sum()))
print("transshipments beyond the level:", int((d.n_ship_12_beyond + d.n_ship_21_beyond).sum()), "of",
      int((d.n_ship_12 + d.n_ship_21).sum()))
print("sender reorders:", int(d.n_sender_reorders.sum()), " receiver reorders:", int(d.n_receiver_reorders.sum()))
sub = d[d.K < d.R + d.v]
print("instances with K < R + v:", sub.instance.nunique(), " transshipment states:", int(sub.n_transship.sum()),
      " sender reorders:", int(sub.n_sender_reorders.sum()))

print("\n-- threshold property")
for col in ("Q1", "Q1_reached", "Q2"):
    bad = d[d[col] > 0]
    print(f"{col}: instance-periods {len(bad)}, instances {bad.instance.nunique()}; with K <= R + v: instances "
          f"{bad[bad.KleRv].instance.nunique()}; with K > R + v: instances {bad[~bad.KleRv].instance.nunique()}; "
          f"in the last period: {int((bad.t == bad['T']).sum())}")
bad = d[d.Q1_reached > 0]
print(bad[["instance", "t", "T", "demand", "dist", "cv", "K", "R", "v", "b", "h", "mu1", "mu2", "Q1", "Q1_reached", "Q2"]].to_string())

print("\n-- further regularities: instance-periods and instances with an exception")
for col, den in (("E1", "curve_pairs"), ("E2", "curve_pairs"), ("E3", "line_pairs"), ("E4", "line_pairs"), ("E5", None),
                 ("E6", "cells"), ("E7", "cells")):
    bad = d[d[col] > 0]
    extra = f"   exceptions {int(d[col].sum()):,} of {int(d[den].sum()):,}" if den else f"   exceptions {int(d[col].sum()):,}"
    print(f"{col}: instance-periods {len(bad)} of {len(d)}, instances {bad.instance.nunique()}{extra}")
print("interval of each line fails: lines", int(d.interval_violations.sum()), "of", int(d.lines.sum()), " instances",
      d[d.interval_violations > 0].instance.nunique())
print("K-convexity in a coordinate fails (kappa > K): C~", d[d.kappa_C > d.K + 1e-6].instance.nunique(), " C~~",
      d[d.kappa_Cq > d.K + 1e-6].instance.nunique())

print("\n-- ten types of action: states and instances")
types = ["none", "r1", "r2", "r12", "s12", "s21", "s12_recv", "s21_recv", "s12_send", "s21_send"]
tot = d[[f"n_{k}" for k in types]].sum()
print({k: int(tot[f"n_{k}"]) for k in types})
print({k: d[d[f"n_{k}"] > 0].instance.nunique() for k in types})
print("instance-periods by number of types:", d.n_types.value_counts().sort_index().to_dict())
print("sender ships all its stock when it reorders: exceptions", int(d.E5.sum()), "of", int((d.n_s12_send + d.n_s21_send).sum()))

ex = json.load(open(os.path.join(OUT, "round5b_examples.json")))
import collections
print("\nexamples by check:", dict(collections.Counter(e["check"] for e in ex)))
err = os.path.join(OUT, "round5b_errors.json")
if os.path.exists(err):
    e = json.load(open(err))
    print("errors:", len(e), collections.Counter(x["error"][:70] for x in e))
