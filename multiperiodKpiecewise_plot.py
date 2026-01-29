import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ==========================================
# 1. Parameter Setup
# ==========================================
T = 4
h = 1.0  # Holding cost
b = 10.0  # Penalty cost (High to prevent stockout)
K_order = 50.0  # LARGE Fixed ordering cost (Encourage avoiding orders)
z_order = 0.0  # Unit ordering cost = 0
R_trans = 5.0  # SMALL Fixed transshipment cost (Encourage transshipment)
v_trans = 0.0  # Unit transshipment cost = 0

# Non-Stationary Demand: Independent Normal Distributions
# Period 1: Mean 10 (High)
# Period 2: Mean 5  (Low)
# Period 3: Mean 8  (Medium)
# Period 4: Mean 5  (Low)
demand_params = [
    {'mu': 10.0, 'sigma': 3.0},  # T=1
    {'mu': 5.0, 'sigma': 2.0},  # T=2
    {'mu': 8.0, 'sigma': 3.0},  # T=3
    {'mu': 5.0, 'sigma': 2.0}  # T=4
]

# State Space limits (Optimization: only calculate needed range)
min_inv = -10
max_inv = 20
state_values = np.arange(min_inv, max_inv + 1)
n_states = len(state_values)
state_to_idx = {v: i for i, v in enumerate(state_values)}

# Action limits
max_Q = 20
max_W = 15

# Value Function
V = np.zeros((T + 2, n_states, n_states))
Policy = {}


# Pre-calculate discretized Normal demands for Expectation (Speed up)
# We use 5 representative points for probability mass
def get_demand_dist(mu, sigma):
    # Approximating Normal with discrete points
    x = np.linspace(mu - 2 * sigma, mu + 2 * sigma, 5).astype(int)
    x = np.maximum(0, x)  # No negative demand
    # Calculate probabilities
    probs_raw = norm.pdf(x, mu, sigma)
    probs = probs_raw / np.sum(probs_raw)
    return x, probs


# ==========================================
# 2. Helper Functions
# ==========================================
def get_transship_cost(W):
    return (R_trans + v_trans * abs(W)) if W != 0 else 0


def get_order_cost(Q):
    return (K_order + z_order * Q) if Q > 0 else 0


def get_immediate_loss(inv):
    return h * max(0, inv) + b * max(0, -inv)


# ==========================================
# 3. Backward Induction
# ==========================================
print("Starting SDP Calculation (T=4)...")

for t in range(T, 0, -1):
    print(f"  Calculating Period {t}...")
    Policy[t] = {}
    mu = demand_params[t - 1]['mu']
    sigma = demand_params[t - 1]['sigma']
    d_vals, d_probs = get_demand_dist(mu, sigma)

    # We only need full DP for states we might visit.
    # To save time in this snippet, we calculate all, but focus on the 'imbalanced'
    # states that will appear in our simulation.

    for idx1, val1 in enumerate(state_values):
        for idx2, val2 in enumerate(state_values):

            # --- Stage 1: Search W ---
            best_W_val = float('inf')
            best_action = (0, 0, 0)

            # Constraints
            limit_pos = max(0, val1)
            limit_neg = -max(0, val2)
            w_start = max(limit_neg, -max_W)
            w_end = min(limit_pos, max_W)

            if w_start > w_end:
                possible_Ws = [0]
            else:
                possible_Ws = range(w_start, w_end + 1)

            for W in possible_Ws:
                t_cost = get_transship_cost(W)
                i1_tilde = val1 - W
                i2_tilde = val2 + W

                # --- Stage 2: Search Q ---
                # Since z=0, we likely order BIG or NOTHING.
                # Simplification: Brute force optimal Q pair

                best_Q_val = float('inf')
                best_Qs = (0, 0)

                # Search heuristic: Q is likely 0 or brings us to target S
                # Search 0, and range [max(0, mu - inv), mu + 3sigma - inv]
                # To be safe and show the graph, we scan range 0..15
                q_scan = range(0, 16)

                for Q1 in q_scan:
                    for Q2 in q_scan:
                        c_order = get_order_cost(Q1) + get_order_cost(Q2)

                        exp_fut = 0
                        # Vectorized expectation
                        for d1, p1 in zip(d_vals, d_probs):
                            for d2, p2 in zip(d_vals, d_probs):
                                prob = p1 * p2
                                ni1 = i1_tilde + Q1 - d1
                                ni2 = i2_tilde + Q2 - d2

                                # Immediate cost
                                imm = get_immediate_loss(ni1) + get_immediate_loss(ni2)

                                # Future Value
                                n_idx1 = np.clip(state_to_idx.get(ni1, 0 if ni1 < min_inv else n_states - 1), 0,
                                                 n_states - 1)
                                n_idx2 = np.clip(state_to_idx.get(ni2, 0 if ni2 < min_inv else n_states - 1), 0,
                                                 n_states - 1)

                                exp_fut += prob * (imm + V[t + 1][n_idx1, n_idx2])

                        val_Q = c_order + exp_fut
                        if val_Q < best_Q_val:
                            best_Q_val = val_Q
                            best_Qs = (Q1, Q2)

                total = t_cost + best_Q_val
                if total < best_W_val:
                    best_W_val = total
                    best_action = (W, best_Qs[0], best_Qs[1])

            V[t][idx1, idx2] = best_W_val
            Policy[t][(idx1, idx2)] = best_action

# ==========================================
# 4. Simulation Trace
# ==========================================
print("\n=== Simulation Trace ===")
# Start at (0,0)
curr_i1, curr_i2 = 0, 0
trace_res = []

# Force demands to create imbalance in Period 1
# P1: Loc 1 gets LOW demand (surplus), Loc 2 gets HIGH demand (shortage)
forced_demands = [
    (2, 14),  # T=1: Force Loc 2 to run out, Loc 1 to have leftovers
    (5, 5),  # T=2: Mean demand
    (8, 8),  # T=3
    (5, 5)  # T=4
]

print(f"{'T':<3} {'Start State':<15} {'Action (W, Q1, Q2)':<20} {'After Trans':<15} {'Demand':<10} {'End State':<15}")

for t in range(1, T + 1):
    # 1. Get Action
    idx1 = np.clip(state_to_idx.get(curr_i1, 0), 0, n_states - 1)
    idx2 = np.clip(state_to_idx.get(curr_i2, 0), 0, n_states - 1)
    W, Q1, Q2 = Policy[t].get((idx1, idx2), (0, 0, 0))

    # 2. Apply Transshipment
    mid_i1 = curr_i1 - W
    mid_i2 = curr_i2 + W

    # 3. Apply Demand (Forced)
    d1, d2 = forced_demands[t - 1]

    end_i1 = mid_i1 + Q1 - d1
    end_i2 = mid_i2 + Q2 - d2

    trace_res.append({
        't': t, 'start': (curr_i1, curr_i2), 'action': (W, Q1, Q2),
        'mid': (mid_i1, mid_i2), 'demand': (d1, d2), 'end': (end_i1, end_i2)
    })

    print(
        f"{t:<3} {str((curr_i1, curr_i2)):<15} {str((W, Q1, Q2)):<20} {str((mid_i1, mid_i2)):<15} {str((d1, d2)):<10} {str((end_i1, end_i2)):<15}")

    curr_i1, curr_i2 = end_i1, end_i2

# ==========================================
# 5. Plotting (Extended for Visualization)
# ==========================================
# We want to show the graph for T=2, where Transshipment actually happened.
# The state was trace_res[1]['start'] -> e.g., (13, 1)

plot_t = 2
plot_state = trace_res[1]['start']
p_idx1 = state_to_idx.get(plot_state[0])
p_idx2 = state_to_idx.get(plot_state[1])

# Re-calculate G(W) curve for this specific state to plot it
mu = demand_params[plot_t - 1]['mu']
sigma = demand_params[plot_t - 1]['sigma']
d_vals, d_probs = get_demand_dist(mu, sigma)

G_curve_W = []
G_curve_Val = []

# --- KEY CHANGE: Extend the plotting range artificially ---
# Instead of strictly limiting by current inventory, we plot a wider range
# to visualize the theoretical shape of the cost function.
# Theoretical range: [-10, 10] even if inventory doesn't allow it fully.
w_range_plot = range(-10, 11)

print(f"Plotting G(W) for state {plot_state} over range {list(w_range_plot)}")

for W in w_range_plot:
    # 1. Calculate Transshipment Cost
    t_cost = get_transship_cost(W)

    # 2. Virtual Post-Transshipment State
    # Note: We allow negative intermediate inventory here just for cost calculation visualization
    # In real logic, this action would be infeasible if i - W < 0.
    # But mathematically, the cost function G(W) is defined.
    i1_tilde = plot_state[0] - W
    i2_tilde = plot_state[1] + W

    # Solve Stage 2 for this W
    best_Q_val = float('inf')
    q_scan = range(0, 16)

    for Q1 in q_scan:
        for Q2 in q_scan:
            c_order = get_order_cost(Q1) + get_order_cost(Q2)
            exp_fut = 0
            for d1, p1 in zip(d_vals, d_probs):
                for d2, p2 in zip(d_vals, d_probs):
                    prob = p1 * p2
                    ni1 = i1_tilde + Q1 - d1
                    ni2 = i2_tilde + Q2 - d2

                    imm = get_immediate_loss(ni1) + get_immediate_loss(ni2)

                    # Boundary clamping for Value Function lookup
                    n_idx1 = np.clip(state_to_idx.get(ni1, 0 if ni1 < min_inv else n_states - 1), 0, n_states - 1)
                    n_idx2 = np.clip(state_to_idx.get(ni2, 0 if ni2 < min_inv else n_states - 1), 0, n_states - 1)

                    exp_fut += prob * (imm + V[plot_t + 1][n_idx1, n_idx2])

            val_Q = c_order + exp_fut
            if val_Q < best_Q_val: best_Q_val = val_Q

    G_curve_W.append(W)
    G_curve_Val.append(t_cost + best_Q_val)

# Plot
plt.figure(figsize=(10, 6))

# Split into left and right for color clarity
left_W = [w for w in G_curve_W if w < 0]
left_G = [G_curve_Val[i] for i, w in enumerate(G_curve_W) if w < 0]
right_W = [w for w in G_curve_W if w > 0]
right_G = [G_curve_Val[i] for i, w in enumerate(G_curve_W) if w > 0]
zero_val = G_curve_Val[G_curve_W.index(0)]

plt.plot(left_W, left_G, 'b-o', linewidth=2, label='Negative Branch (W<0)')
plt.plot(right_W, right_G, 'r-o', linewidth=2, label='Positive Branch (W>0)')

# Mark Discontinuity
plt.scatter(0, zero_val, color='green', s=150, zorder=5, label='No Transship (W=0)')

# Draw vertical dashed line to show the "Jump"
min_val_plot = min(min(G_curve_Val), zero_val)
max_val_plot = max(G_curve_Val)
plt.vlines(0, min_val_plot, max_val_plot, colors='gray', linestyles='--', alpha=0.5)

# Mark Optimal W (from the feasible set in simulation)
# Note: The true optimal might be constrained, but here we show the global minimum of the curve
min_idx = np.argmin(G_curve_Val)
opt_W = G_curve_W[min_idx]
opt_Val = G_curve_Val[min_idx]
plt.scatter(opt_W, opt_Val, color='gold', marker='*', s=300, zorder=6, label=f'Global Min W={opt_W}')

# Annotate
plt.title(f'Piecewise K-Convex Structure of G(W)\nPeriod {plot_t}, State {plot_state}', fontsize=14)
plt.xlabel('Transshipment Quantity W', fontsize=12)
plt.ylabel('Total Expected Cost', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

filename = 'sdp_piecewise_k_convex_full.png'
plt.savefig(filename)
print(f"Full plot saved to {filename}")