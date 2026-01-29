import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ==========================================
# 1. 参数设置 (Parameters)
# ==========================================
h = 1.0  # Unit holding cost
b = 5.0  # Unit penalty cost
K_order = 20.0  # Fixed ordering cost
z = 0.0  # Unit ordering cost
R_trans = 50.0  # Fixed transshipment cost
v_trans = 0.0  # Unit transshipment cost

i1_init = 0  # Location 1 库存
i2_init = 0  # Location 2 库存

mu = 10.0
sigma = 4.0

W_range = np.arange(-20, 21, 1)


# ==========================================
# 2. 核心函数
# ==========================================

def expected_loss_L(y):
    """
    L(y) = E[h*(y-D)+ + b*(D-y)+]
    """
    z_score = (y - mu) / sigma
    pdf = norm.pdf(z_score)
    cdf = norm.cdf(z_score)

    # E[ (D-y)+ ] = sigma*pdf + (mu-y)*(1-cdf)
    ESC = sigma * pdf + (mu - y) * (1 - cdf)
    # E[ (y-D)+ ] = ESC + (y - mu)
    E_Inv = ESC + (y - mu)

    return h * E_Inv + b * ESC


def get_replenishment_cost_curve(current_inv, q_range):
    """
    Return the cost curve for a range of Qs given current inventory
    Cost(Q) = K*I(Q>0) + z*Q + L(current_inv + Q)
    """
    costs = []
    for q in q_range:
        if q == 0:
            c = expected_loss_L(current_inv)
        else:
            c = K_order + z * q + expected_loss_L(current_inv + q)
        costs.append(c)
    return costs


def solve_second_stage_single_loc(current_inv):
    # Find min cost over Q in [0, 50]
    best_cost = float('inf')
    # Check q=0
    c0 = expected_loss_L(current_inv)
    best_cost = c0

    # Check q > 0
    # We scan a reasonable range. If current_inv is -20, we might need Q up to 40 (target ~15)
    for q in range(1, 60):
        c = K_order + z * q + expected_loss_L(current_inv + q)
        if c < best_cost:
            best_cost = c
    return best_cost


# ==========================================
# 3. 计算 G(W)
# ==========================================
G_values = []
for W in W_range:
    # 1. Transshipment cost
    if W == 0:
        trans_cost = 0
    else:
        trans_cost = R_trans + v_trans * abs(W)

    # 2. Post-transshipment states
    i1_tilde = i1_init - W
    i2_tilde = i2_init + W

    # 3. Second stage optimal costs
    cost1 = solve_second_stage_single_loc(i1_tilde)
    cost2 = solve_second_stage_single_loc(i2_tilde)

    G_values.append(trans_cost + cost1 + cost2)

# ==========================================
# 4. 计算 Q-Cost Curves (Visualizing 2nd request)
# ==========================================
# Scenario A: W=0 (Base case)
# Loc 1 inv = 0, Loc 2 inv = 0
q_plot_range = np.arange(0, 41)
cost_curve_w0_loc1 = get_replenishment_cost_curve(0, q_plot_range)  # Since inv=0
cost_curve_w0_loc2 = get_replenishment_cost_curve(0, q_plot_range)  # Same

# Scenario B: W=10 (Transshipment case)
# Loc 1 inv = -10, Loc 2 inv = 10
cost_curve_w10_loc1 = get_replenishment_cost_curve(-10, q_plot_range)
cost_curve_w10_loc2 = get_replenishment_cost_curve(10, q_plot_range)

# ==========================================
# 5. 绘图
# ==========================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot 1: G(W) - Transshipment Objective
ax = axes[0]
ax.plot(W_range, G_values, 'o-', markersize=4, label='Total Cost $G_t(W)$')
# Highlight W=0
idx_0 = list(W_range).index(0)
ax.plot([0], [G_values[idx_0]], 'go', markersize=10, label='No Transshipment')
ax.set_title('Stage 1: Transshipment Objective $G_t(W)$', fontsize=14)
ax.set_xlabel('Transshipment Quantity $W$')
ax.set_ylabel('Total Expected Cost')
ax.grid(True, alpha=0.3)
ax.legend()

# Plot 2: Cost vs Q (at W=0, Inventory=0)
ax = axes[1]
ax.plot(q_plot_range, cost_curve_w0_loc1, 'b-', label='Location 1 (Inv=0)')
# ax.plot(q_plot_range, cost_curve_w0_loc2, 'r--', label='Location 2 (Inv=0)') # Identical
ax.set_title('Stage 2: Cost vs Replenishment $Q$\n(At $W=0$, Initial Inv=0)', fontsize=14)
ax.set_xlabel('Replenishment Quantity $Q$')
ax.set_ylabel('Expected Cost (Order + Holding + Penalty)')
ax.grid(True, alpha=0.3)
ax.legend()

# Plot 3: Cost vs Q (at W=10, Inv1=-10, Inv2=10)
ax = axes[2]
ax.plot(q_plot_range, cost_curve_w10_loc1, 'b-', label='Location 1 (Inv=-10)')
ax.plot(q_plot_range, cost_curve_w10_loc2, 'r-', label='Location 2 (Inv=10)')
ax.set_title('Stage 2: Cost vs Replenishment $Q$\n(At $W=10$, Imbalanced)', fontsize=14)
ax.set_xlabel('Replenishment Quantity $Q$')
ax.set_ylabel('Expected Cost')
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig('sdp2_analysis_plots.png')
plt.show()