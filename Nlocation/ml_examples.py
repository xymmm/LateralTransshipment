"""
ml_examples.py : the five solvers on four instances.

    python ml_examples.py            all instances
    python ml_examples.py --only C   one instance (A, B, C or D)
Writes results_examples/examples.csv and results_examples/examples.txt (examples_C.csv and so on with --only).

Instances: T periods, opening inventory 0, z = 1, v = 0.5, h = 1, b = 5.
  Integer demand (step = 1), K = 5, R = 1
    A   3 locations, T = 4, Poisson, expected demand  (1, 2, 1, 2), (2, 1, 2, 1), (1, 1, 1, 1)
    B   5 locations, T = 4, Poisson, expected demand  (0.5, 1, 1.5, 1), (1.5, 1, 0.5, 1), (1, 1, 1, 1),
                                                      (0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5)
  Real-valued demand (step < 1: demand values and inventory levels are multiples of step, quantities whole units)
    C   3 locations, T = 3, gamma with coefficient of variation 0.5, step 0.5, K = 5, R = 1,
        expected demand  (1, 2, 1), (2, 1, 2), (1, 1, 1)
    D   2 locations, T = 4, normal with coefficient of variation 0.3, step 0.5, K = 20, R = 10,
        expected demand  (6, 13, 7, 2), (13, 7, 1, 7).
        The exact value is also computed with the steps 1, 0.25 and 0.1 on the same inventory levels.

Inventory levels
    max_inventory = the total expected demand of all locations over the horizon, rounded up. One location can
                    then hold the stock of the whole system.
    min_inventory = -D, with D the largest truncated demand of one period at any location, rounded up.
    The exact value is solved again with max_inventory raised by 25% and with min_inventory lowered by 25%; the
    table reports the relative change. No change means that the inventory levels are large enough.

What is reported
    SDP1, SDP2, SDP2quick   ETC = C_1(i_1), the exact expected total cost of the optimal policy, and the time of
                            the backward induction. SDP1 and SDP2 list every transshipment vector W; they are run
                            when the number of vectors of one period is below --limit, and the number is reported
                            otherwise.
    milp                    the objective of the static plan from the opening inventory and its solution time.
                            The objective is the approximate cost of a plan that is never revised; it is not an ETC.
    milpRL                  ETC of the rolling-horizon algorithm by simulation, with the half-width of its 95%
                            confidence interval, and its times: precomputation, one decision at k = 1, all decisions.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

warnings.filterwarnings("ignore", category=RuntimeWarning)

import ml_milpRL  # noqa: E402
from ml_common import largest_demand, per_unit, total_demand_quantile  # noqa: E402
from ml_milp import solve_static  # noqa: E402
from ml_milpRL import milpRL  # noqa: E402
from ml_SDP1 import SDP1  # noqa: E402
from ml_SDP2 import SDP2  # noqa: E402
from ml_SDP2quick import SDP2quick  # noqa: E402

SMALL = dict(K=5.0, z=1.0, R=1.0, v=0.5, h=1.0, b=5.0)
PAPER = dict(K=20.0, z=1.0, R=10.0, v=0.5, h=1.0, b=5.0)
INSTANCES = {
    "A": dict(means=[[1, 2, 1, 2], [2, 1, 2, 1], [1, 1, 1, 1]], costs=SMALL, demand={}, max_paths=1_000_000),
    "B": dict(means=[[0.5, 1, 1.5, 1], [1.5, 1, 0.5, 1], [1, 1, 1, 1], [0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]],
              costs=SMALL, demand={}, max_paths=20_000),
    "C": dict(means=[[1, 2, 1], [2, 1, 2], [1, 1, 1]], costs=SMALL,
              demand=dict(distribution="gamma", cv=0.5, step=0.5), max_paths=10_000),
    "D": dict(means=[[6, 13, 7, 2], [13, 7, 1, 7]], costs=PAPER,
              demand=dict(distribution="normal", cv=0.3, step=0.5), max_paths=3_000, other_steps=(1.0, 0.25, 0.1)),
}                                                    # max_paths: simulation stops at a 0.1% half-width or here


def vectors_per_period(J: int, lo: int, hi: int, m: int = 1) -> int:
    """Number of transshipment vectors W that SDP1 and SDP2 list in one period: the sum of |R(i)| over all states.
    With m steps per unit there are m^J classes of states, each with the whole-unit levels lo..hi."""
    per_location = sum(math.comb(max(s, 0) + J - 1, J - 1) for s in range(lo, hi + 1))
    return (m * per_location) ** J


def run(name: str, limit: float):
    inst = INSTANCES[name]
    mus, costs, demand = inst["means"], inst["costs"], inst["demand"]
    step = demand.get("step", 1.0)
    m = per_unit(step)
    J, T = len(mus), len(mus[0])
    D = largest_demand(mus, **demand)
    total = float(np.sum(mus))
    lo, hi = -int(math.ceil(D)), int(math.ceil(total))
    law = demand.get("distribution", "poisson") + (f" with cv {demand['cv']}" if "cv" in demand else "")
    rows, log = [], []

    def say(text):
        print(text, flush=True)
        log.append(text)

    def add(method, etc, seconds, note="", half_width=np.nan, step_=step, states=None):
        rows.append(dict(instance=name, J=J, T=T, demand=law, step=step_, min_inventory=lo, max_inventory=hi,
                         states=((hi - lo + 1) * per_unit(step_)) ** J if states is None else states,
                         method=method, ETC=etc, half_width=half_width, seconds=seconds, note=note))

    say(f"\n===== instance {name}: {J} locations, T = {T}, {law} demand, step {step:g}, expected demand {mus}")
    say(f"costs {costs}")
    say(f"total expected demand {total:g}; largest truncated demand of a period D = {D:g}; "
        f"0.9999 quantile of the total demand {total_demand_quantile(mus, **demand):g}")
    say(f"inventory levels [{lo}, {hi}{'' if m == 1 else f' + 1 - {step:g}'}] in steps of {step:g}: "
        f"{(hi - lo + 1) * m} levels per location, {((hi - lo + 1) * m) ** J:,} states")

    # ---- SDP2quick, and the check of the inventory levels
    quick = SDP2quick(mus, lo, hi, **costs, **demand)
    opt = quick.value()
    hi2, lo2 = int(math.ceil(1.25 * hi)), int(math.floor(1.25 * lo))
    up = SDP2quick(mus, lo, hi2, **costs, **demand)
    change_hi, seconds_hi, states_hi = abs(up.value() - opt) / up.value(), up.seconds, up.states
    del up
    down = SDP2quick(mus, lo2, hi, **costs, **demand)
    change_lo = abs(down.value() - opt) / down.value()
    del down
    add("SDP2quick", opt, quick.seconds, f"first decision {quick.action()}")
    say(f"SDP2quick   ETC {opt:.4f}   {quick.seconds:.2f}s   first decision {quick.action()}")
    say(f"            max_inventory {hi} -> {hi2} ({states_hi:,} states, {seconds_hi:.2f}s): relative change {change_hi:.1e}; "
        f"min_inventory {lo} -> {lo2}: relative change {change_lo:.1e}")
    rows[-1].update(change_max_plus_25pct=change_hi, change_min_minus_25pct=change_lo)

    # ---- SDP2 and SDP1, following the formulation
    count = vectors_per_period(J, lo, hi, m)
    if count <= limit:
        two = SDP2(mus, lo, hi, **costs, **demand)
        add("SDP2", two.value(), two.seconds, f"largest difference from SDP2quick {float(np.abs(two.C - quick.C).max()):.1e}")
        say(f"SDP2        ETC {two.value():.4f}   {two.seconds:.2f}s   second stage {two.seconds_stage['reorder']:.2f}s, "
            f"first stage {two.seconds_stage['transshipment']:.2f}s; {count:,} transshipment vectors per period; "
            f"largest difference from SDP2quick over all states and periods {float(np.abs(two.C - quick.C).max()):.1e}")
        del two
        one = SDP1(mus, lo, hi, **costs, **demand)
        add("SDP1", one.value(), one.seconds, f"largest difference from SDP2quick {float(np.abs(one.C - quick.C).max()):.1e}")
        say(f"SDP1        ETC {one.value():.4f}   {one.seconds:.2f}s   "
            f"largest difference from SDP2quick over all states and periods {float(np.abs(one.C - quick.C).max()):.1e}")
        del one
    else:
        for method in ("SDP2", "SDP1"):
            add(method, np.nan, np.nan, f"not run: {count:.2e} transshipment vectors per period")
        say(f"SDP2, SDP1  not run: they would list {count:.2e} transshipment vectors in every period")
    del quick

    # ---- the exact value with other steps, on the same inventory levels
    for other in inst.get("other_steps", ()):
        r = SDP2quick(mus, lo, hi, **costs, **dict(demand, step=other))
        add("SDP2quick", r.value(), r.seconds, f"step {other:g} in the place of {step:g}", step_=other, states=r.states)
        say(f"SDP2quick   with step {other:g}: ETC {r.value():.4f}   {r.states:,} states   {r.seconds:.2f}s   "
            f"first decision {r.action()}")
        del r

    # ---- the static model, one solve
    plan = solve_static(mus, **costs, **demand)
    add("milp (static plan)", plan.obj, plan.seconds, "objective of the plan, not an ETC")
    say(f"milp        objective of the static plan {plan.obj:.4f}   {plan.seconds:.3f}s   "
        f"reorders by period {plan.Q.tolist()}")

    # ---- the rolling-horizon algorithm
    ml_milpRL.clear_store()
    r = milpRL(mus, **costs, **demand, max_paths=inst["max_paths"])
    gap = 100.0 * (r.etc - opt) / opt
    add("milpRL", r.etc, r.seconds_total,
        f"gap {gap:.2f}%; {r.n_paths} paths; precomputation {r.seconds_precompute:.4f}s; decision at k=1 "
        f"{r.seconds_decision_k1:.3f}s; {r.n_decisions} decisions, mean {r.seconds_per_decision:.4f}s", r.half_width)
    rows[-1].update(gap_pct=gap, paths=r.n_paths, seconds_precompute=r.seconds_precompute,
                    seconds_decision_k1=r.seconds_decision_k1, seconds_milp=r.seconds_milp,
                    seconds_refine=r.seconds_refine, seconds_build=r.seconds_build, decisions=r.n_decisions,
                    seconds_per_decision=r.seconds_per_decision, precision_attained=r.precision_attained)
    say(f"milpRL      ETC {r.etc:.4f} +- {r.half_width:.4f} on {r.n_paths:,} paths "
        f"({'0.1% reached' if r.precision_attained else 'path limit reached'});   gap to the optimum {gap:.2f}%")
    say(f"            first decision {r.first_decision}")
    say(f"            precomputation {r.seconds_precompute:.4f}s ({r.entries_computed} cumulative demands computed, "
        f"{r.entries_reused} reused); models {r.seconds_build:.3f}s; decision at k = 1: {r.seconds_decision_k1:.3f}s; "
        f"{r.n_decisions:,} decisions: model solves {r.seconds_milp:.1f}s, refinement {r.seconds_refine:.1f}s, "
        f"mean {r.seconds_per_decision:.4f}s per decision; whole run {r.seconds_total:.1f}s")
    return rows, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=sorted(INSTANCES), default=None)
    ap.add_argument("--limit", type=float, default=5e9,
                    help="SDP1 and SDP2 are run when the transshipment vectors of one period are at most this many")
    args = ap.parse_args()
    out = HERE / "results_examples"
    out.mkdir(exist_ok=True)
    rows, log = [], []
    for name in ([args.only] if args.only else sorted(INSTANCES)):
        r, l = run(name, args.limit)
        rows += r
        log += l
    suffix = f"_{args.only}" if args.only else ""
    pd.DataFrame(rows).to_csv(out / f"examples{suffix}.csv", index=False)
    (out / f"examples{suffix}.txt").write_text("\n".join(log) + "\n")
    print("\nresults ->", out)


if __name__ == "__main__":
    main()
