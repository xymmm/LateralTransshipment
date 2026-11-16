"""
NOTINUSE_guessingOptimalStructure/conjecture_sweep2.py : extension of the sweep of conjecture_search.py (family asym2) to larger K
for Poisson demand and normal demand with rho = 0.3.

    python conjecture_sweep2.py
"""
from __future__ import annotations

import multiprocessing as mp

import numpy as np

from conjecture_search import Instance3, run

if __name__ == "__main__":
    todo = [Instance3("ASYM2", 2, float(K), 2, d, c, b=20.0, mu1=[1, 30], mu2=[1, 12])
            for (d, c) in (("poisson", None), ("normal", 0.3)) for K in np.arange(55, 80.01, 1.0)]
    with mp.get_context("spawn").Pool(10) as pool:
        res = pool.map(run, todo)
    for rows in res:
        r = rows[0]
        if "error" in r:
            print("ERROR", r)
            continue
        if r["threshold_violations"] or r["Q_violations"]:
            print(r["dist"], r["cv"], "K", r["K"], "threshold", r["threshold_violations"], r["first_threshold"],
                  "Q", r["Q_violations"], r["worst"])
    print("done", len(res))
