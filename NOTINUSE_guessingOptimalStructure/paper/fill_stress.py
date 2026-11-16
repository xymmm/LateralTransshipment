"""
NOTINUSE_guessingOptimalStructure/paper/fill_stress.py : numbers of the stress test for Section 9 of the paper.

    python fill_stress.py

Reads ../out/round5b.csv and prints every number that the section on the stress test quotes, as a dictionary.
"""
from __future__ import annotations

import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
d = pd.read_csv(os.path.join(HERE, "..", "out", "round5b.csv"))
inst = d.groupby("instance").first()
kle = d.K <= d.R + d.v
klt = d.K < d.R + d.v


def n_inst(mask):
    return int(d[mask].instance.nunique())


N = dict(
    NSTRESS=d.instance.nunique(),
    NCORNERS=int((inst.set == "corners").sum()),
    NRANDOM=int((inst.set == "random2").sum()),
    NIP=len(d),
    NSTATESB=int(d.n_states.sum()),
    NKLE=n_inst(kle),
    NKLT=n_inst(klt),
    NTRANS=int(d.n_transship.sum()),
    NBEYOND=int((d.n_ship_12_beyond + d.n_ship_21_beyond).sum()),
    NSHIPS=int((d.n_ship_12 + d.n_ship_21).sum()),
    NREORDER=int((d.n_both + d.n_only).sum()),
    NBELOW=int((d.n_both_below + d.n_only_below).sum()),
    NSENDER=int(d.n_sender_reorders.sum()),
    NSENDER_KLT=int(d[klt].n_sender_reorders.sum()),
    NTRANS_KLT=int(d[klt].n_transship.sum()),
    NRECEIVER=int(d.n_receiver_reorders.sum()),
    NLAST=2 * int((d.t == d["T"]).sum()),
    V=d[["V1", "V2", "V3", "V4a", "V4b", "V4c", "V5t"]].fillna(0).sum().astype(int).to_dict(),
    V5_round2=int(d.V5.fillna(0).sum()),
    OUTSIDE=int(d.levels_outside_window.sum()),
    Q1_ip=int((d.Q1 > 0).sum()), Q1_inst=n_inst(d.Q1 > 0), Q1_inst_kle=n_inst((d.Q1 > 0) & kle), Q1_inst_kgt=n_inst((d.Q1 > 0) & ~kle),
    Q1r_ip=int((d.Q1_reached > 0).sum()), Q1r_inst=n_inst(d.Q1_reached > 0), Q1r_inst_kle=n_inst((d.Q1_reached > 0) & kle),
    Q1r_inst_kgt=n_inst((d.Q1_reached > 0) & ~kle),
    Q2_ip=int((d.Q2 > 0).sum()), Q2_inst=n_inst(d.Q2 > 0), Q2_inst_kle=n_inst((d.Q2 > 0) & kle),
    Q_last=int(((d.Q1 > 0) & (d.t == d["T"])).sum()),
    NIP_before_last=int((d.t < d["T"]).sum()),
    E1=n_inst(d.E1 > 0), E2=n_inst(d.E2 > 0), E3=n_inst(d.E3 > 0), E4=n_inst(d.E4 > 0), E5=n_inst(d.E5 > 0),
    E6=n_inst(d.E6 > 0), E7=n_inst(d.E7 > 0),
    EINT=n_inst(d.interval_violations > 0), EKC=n_inst(d.kappa_C > d.K + 1e-6), EKCQ=n_inst(d.kappa_Cq > d.K + 1e-6),
    INT_lines=int(d.interval_violations.sum()), LINES=int(d.lines.sum()),
    NTEN=int((d.n_types == 10).sum()),
    types_inst={k: n_inst(d[f"n_{k}"] > 0) for k in ["none", "r1", "r2", "r12", "s12", "s21", "s12_recv", "s21_recv", "s12_send", "s21_send"]},
    E5_states=int(d.E5.sum()),
    sender_states=int((d.n_s12_send + d.n_s21_send).sum()),
    hours=round(float(inst.seconds.sum()) / 3600, 1),
    demand_models=inst.groupby(["dist", "cv"], dropna=False).size().to_dict(),
    T_counts=inst["T"].value_counts().sort_index().to_dict(),
    max_window=int((d.hi - d.lo + 1).max()),
)
N["NNOTBELOW"] = N["NREORDER"] - N["NBELOW"]
N["NMINTYPE"] = min(N["types_inst"].values())
print(json.dumps({k: (v if not isinstance(v, dict) else {str(a): b for a, b in v.items()}) for k, v in N.items()}, indent=1))
bad = d[d.Q1_reached > 0]
if len(bad):
    print(bad[["instance", "t", "T", "demand", "dist", "cv", "K", "R", "v", "b", "h", "mu1", "mu2", "Q1", "Q1_reached", "Q2"]].to_string())
