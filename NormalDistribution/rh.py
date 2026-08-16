"""
rh.py
─────
Receding-horizon control for the two-location transshipment problem, and its
Monte-Carlo evaluation.

At each period k the leading order quantity is taken from MILP-1 and the leading
transshipment is refined against the plan-based value function using the EXACT
expected residual cost of the fixed plan (paper eqs 17-19).  This is the
corrected refinement: the future is integrated in closed form via the normal
loss functions, not substituted by a sampled demand path.

Evaluation samples integer demand from the same discretised-normal law used by
the SDPs, so the receding-horizon cost and the SDP optimum are directly
comparable.  MILP solves are memoised on (k, i1, i2) to make the simulation
affordable.
"""

import numpy as np
from normal_demand import normal_pmf, loss, loss_complement
from milp1 import solve_milp1


def _trans_cost(W, R, v):
    return (R + v * abs(W)) if W != 0 else 0.0


def _order_cost(Q, K, z):
    return (K + z * Q) if Q > 0 else 0.0


def plan_value_of_W(Wk, i1_prev, i2_prev, plan, mu1_res, mu2_res, cv,
                    R, v, h, b):
    """
    Plan-based value  C_tilde^pi_k  as a function of the leading transshipment
    Wk, holding every later decision of the plan fixed (eqs 17-18).  Only the
    Wk-dependent terms are returned; the additive ordering / later-transship
    constants are omitted because they do not affect the argmin over Wk.
    """
    mu1 = np.asarray(mu1_res, float); mu2 = np.asarray(mu2_res, float)
    sig1 = cv * mu1; sig2 = cv * mu2
    Wlater = plan["W"].copy(); Wlater[0] = 0.0          # leading W is the decision
    Q1 = plan["Q1"]; Q2 = plan["Q2"]
    Tr = len(mu1)

    # cumulative available before demand at each residual period
    cumQ1 = np.cumsum(Q1); cumQ2 = np.cumsum(Q2)
    cumWlater = np.cumsum(Wlater)
    total = _trans_cost(Wk, R, v)
    muC1 = np.cumsum(mu1); muC2 = np.cumsum(mu2)
    sigC1 = np.sqrt(np.cumsum(sig1 ** 2)); sigC2 = np.sqrt(np.cumsum(sig2 ** 2))
    for t in range(Tr):
        X1 = i1_prev + cumQ1[t] - Wk - cumWlater[t]
        X2 = i2_prev + cumQ2[t] + Wk + cumWlater[t]
        total += (h * loss_complement(X1, muC1[t], sigC1[t])
                  + b * loss(X1, muC1[t], sigC1[t])
                  + h * loss_complement(X2, muC2[t], sigC2[t])
                  + b * loss(X2, muC2[t], sigC2[t]))
    return float(total)


def refine_W(i1_prev, i2_prev, plan, mu1_res, mu2_res, cv, R, v, h, b):
    """One-dimensional search for the leading transshipment (eqs 19-20)."""
    lo = -max(0, i2_prev); hi = max(0, i1_prev)
    best_W, best_v = 0, np.inf
    for W in range(lo, hi + 1):
        val = plan_value_of_W(W, i1_prev, i2_prev, plan, mu1_res, mu2_res,
                              cv, R, v, h, b)
        if val < best_v:
            best_v, best_W = val, W
    return best_W


class RHPolicy:
    """
    Receding-horizon policy with memoised MILP solves.

    refine = True  reproduces Algorithm 1 (leading order from MILP, leading
                   transshipment refined against the plan-based value).
    refine = False is the ablation that implements the MILP's leading
                   transshipment directly (used to isolate Delta_eval).
    """

    def __init__(self, mu1, mu2, cv, K, z, R, v, h, b,
                 n_regions=10, refine=True):
        self.mu1 = np.asarray(mu1, float); self.mu2 = np.asarray(mu2, float)
        self.cv = cv
        self.K, self.z, self.R, self.v, self.h, self.b = K, z, R, v, h, b
        self.n_regions = n_regions
        self.refine = refine
        self.T = len(mu1)
        self._cache = {}
        self._dec = {}

    def _plan(self, k, i1, i2):
        key = (k, i1, i2)
        if key not in self._cache:
            self._cache[key] = solve_milp1(
                i1, i2, self.mu1[k:], self.mu2[k:], self.cv,
                self.K, self.z, self.R, self.v, self.h, self.b,
                n_regions=self.n_regions)
        return self._cache[key]

    def _decision(self, k, i1, i2):
        """Leading (Q1, Q2, W) at state (i1,i2) in period k, memoised."""
        key = (k, i1, i2)
        if key not in self._dec:
            plan = self._plan(k, i1, i2)
            Q1k = int(round(plan["Q1"][0])); Q2k = int(round(plan["Q2"][0]))
            if self.refine:
                Wk = refine_W(i1, i2, plan, self.mu1[k:], self.mu2[k:],
                              self.cv, self.R, self.v, self.h, self.b)
            else:
                Wk = int(round(plan["W"][0]))
                Wk = max(-max(0, i2), min(max(0, i1), Wk))
            self._dec[key] = (Q1k, Q2k, Wk)
        return self._dec[key]

    def rollout(self, i0, demand1, demand2):
        """One rollout under a given integer demand path. Returns total cost."""
        i1, i2 = i0
        cost = 0.0
        for k in range(self.T):
            Q1k, Q2k, Wk = self._decision(k, i1, i2)
            cost += (_trans_cost(Wk, self.R, self.v)
                     + _order_cost(Q1k, self.K, self.z)
                     + _order_cost(Q2k, self.K, self.z))
            a1 = i1 - Wk + Q1k; a2 = i2 + Wk + Q2k
            i1 = a1 - int(demand1[k]); i2 = a2 - int(demand2[k])
            cost += (self.h * max(i1, 0) + self.b * max(-i1, 0)
                     + self.h * max(i2, 0) + self.b * max(-i2, 0))
        return cost

    def evaluate(self, i0, n_paths=2000, seed=0):
        """Monte-Carlo estimate of expected total cost from state i0."""
        rng = np.random.default_rng(seed)
        pmf1 = [normal_pmf(m, self.cv * m) for m in self.mu1]
        pmf2 = [normal_pmf(m, self.cv * m) for m in self.mu2]
        # sample every path's demand for every period in one vectorised call
        D1 = np.column_stack([rng.choice(len(pmf1[t]), size=n_paths, p=pmf1[t])
                              for t in range(self.T)])
        D2 = np.column_stack([rng.choice(len(pmf2[t]), size=n_paths, p=pmf2[t])
                              for t in range(self.T)])
        costs = np.empty(n_paths)
        for n in range(n_paths):
            costs[n] = self.rollout(i0, D1[n], D2[n])
        mean = costs.mean()
        half = 1.96 * costs.std(ddof=1) / np.sqrt(n_paths)
        return mean, half, costs


if __name__ == "__main__":
    mu1 = [6, 7, 8, 7]; mu2 = [6, 7, 8, 7]; cv = 0.30
    K, z, R, v, h, b = 40, 1, 15, 0.5, 1, 8
    pol = RHPolicy(mu1, mu2, cv, K, z, R, v, h, b, refine=True)
    m, half, _ = pol.evaluate((0, 0), n_paths=500, seed=1)
    print(f"RH cost from (0,0): {m:.3f} +/- {half:.3f}   (cache size {len(pol._cache)})")
