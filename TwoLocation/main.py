"""
Example driver: solve one instance with SDP1 and SDP2 using the same parameter set.

Run:  python main.py
"""
import numpy as np

from lt_sdp import SDP1, SDP2

# ---------------- one parameter set, shared by both solvers ----------------
params = dict(
    demand_mean1=[4, 6, 8, 6],      # location 1, one mean per period
    demand_mean2=[4, 6, 8, 6],      # location 2
    min_inventory=-20,
    max_inventory=60,
    K=20,                           # fixed ordering cost
    z=1,                            # unit ordering cost
    R=5,                            # fixed transshipment cost
    v=0.5,                          # unit transshipment cost
    h=0.25,                         # holding cost
    b=5,                            # backorder cost
    distribution="poisson",         # "poisson" or "normal"
    cv=None,                        # required when distribution="normal": sd = cv * mean, scalar or one per period
    tail=1e-4,
    max_quantity=None,              # orders run over 0..max_quantity at each location; None means the number of grid levels
    no_initial_transship=False,
    no_initial_order=False,
)


def summary(res, i1=0, i2=0):
    W, QA, QB = res.action(i1, i2, period=1)
    print(f"{res.method}: C_1({i1},{i2}) = {res.value(i1, i2):.6f}   action (W, QA, QB) = ({W}, {QA}, {QB})   "
          f"time {res.seconds:.2f}s   orders beyond grid: {res.orders_beyond_grid()}")


if __name__ == "__main__":
    r1, r2 = SDP1(**params), SDP2(**params)
    summary(r1); summary(r2)
    print(f"max |C_SDP1 - C_SDP2| over all periods and states = {np.max(np.abs(r1.C - r2.C)):.2e}")

