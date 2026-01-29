import numpy as np
import time
from scipy.stats import norm

# ==========================================
# 1. 高速配置 (Optimized Setup)
# ==========================================
T = 4
h = 1.0;
b = 10.0
K_order = 50.0;
R_trans = 5.0
# 扩大状态空间以展示速度 (即使范围变大，速度依然很快)
min_inv, max_inv = -20, 30
state_values = np.arange(min_inv, max_inv + 1)
n_states = len(state_values)
state_idx_map = {v: i for i, v in enumerate(state_values)}

# 预计算需求分布 (Pre-calc Probabilities)
demand_params = [
    {'mu': 10.0, 'sigma': 3.0}, {'mu': 5.0, 'sigma': 2.0},
    {'mu': 8.0, 'sigma': 3.0}, {'mu': 5.0, 'sigma': 2.0}
]


def get_demand_probs(mu, sigma):
    # 离散化需求，使用向量运算
    d_range = np.arange(0, 21)  # 假设最大需求20
    probs = norm.pdf(d_range, mu, sigma)
    probs /= probs.sum()
    return d_range, probs


# 初始化价值函数
V = np.zeros((n_states, n_states))
Policy = {}


# ==========================================
# 2. 核心加速函数 (Vectorized Helpers)
# ==========================================

def precompute_post_decision_cost(t, V_next):
    """
    【核心优化】
    在进入状态循环前，一次性计算所有可能的 (y1, y2) 状态下的
    [即时成本 + 期望未来成本]。
    y = i + Q - W (订货后的库存水平)
    """
    mu = demand_params[t - 1]['mu']
    sigma = demand_params[t - 1]['sigma']
    d_vals, d_probs = get_demand_probs(mu, sigma)

    # 1. 计算即时损失 L(y) table (Vectorized)
    # L(y) = Sum_d P(d) * [h*max(0, y-d) + b*max(0, d-y)]
    # 创建 (N_states, N_demands) 矩阵广播计算
    y_grid = state_values[:, None]  # (N, 1)
    d_grid = d_vals[None, :]  # (1, D)

    ending_inv = y_grid - d_grid
    loss_matrix = h * np.maximum(0, ending_inv) + b * np.maximum(0, -ending_inv)
    expected_loss = loss_matrix @ d_probs  # (N,) Matrix vector product

    # 2. 计算期望未来价值 E[V(y-d)]
    # 这是一个卷积操作，但为了边界处理，我们用广播映射
    # 我们需要映射 Next_State_Index
    next_state_indices = np.clip(
        np.searchsorted(state_values, ending_inv),
        0, n_states - 1
    )
    # V_next_lookup: (N_y, N_d) -> 对应每个 y 和 d 的 V 值
    # 对于两个地点，我们需要做两次独立的期望计算吗？
    # 不，状态是耦合的。我们需要计算 2D 期望。
    # E[V(y1-d1, y2-d2)] = Sum_d1 Sum_d2 P(d1)P(d2) V(y1-d1, y2-d2)

    # 这是一个 2D 卷积。为了速度，我们可以利用 Kronecker 积或分离计算
    # 这里用矩阵乘法加速：
    # 设 P 为转移概率矩阵 (N, N)，则 E_V = P @ V @ P.T (假设两地需求独立且同分布)
    # 如果分布不同，则是 P1 @ V @ P2.T

    # 构建转移矩阵 P (N_states x N_states)
    # P[i, j] = prob that state i becomes state j (given demand)
    # y - d = next_s  =>  d = y - next_s
    P = np.zeros((n_states, n_states))
    for d, p in zip(d_vals, d_probs):
        # 向量化填充：对角线偏移
        # 这是一个 Toeplitz 矩阵结构，可以用 scipy.linalg.toeplitz 加速，这里手动循环也很快
        # 意思是：如果我有 y，需求是 d，我会变成 y-d
        # 所以 P[row, col] 表示从 index row 变到 index col 的概率
        # row_val - d = col_val
        # col_idx = row_idx - d (approx)

        # 这里的实现：简单平移
        start = d
        end = n_states
        if start < n_states:
            # Shift value logic matching index logic
            # indices shift by d roughly
            # accurately: P[i, j] where state[j] ~ state[i] - d
            idx_shift = int(d)  # assuming step 1
            # P[k, k-d] += p
            # Handle boundaries carefully (clamping)

            # Simple safe loop for matrix construction
            for i in range(n_states):
                target = state_values[i] - d
                j = np.clip(np.searchsorted(state_values, target), 0, n_states - 1)
                P[i, j] += p

    # 2D Expected Future Value calculation using Matrix Multiplication
    # EV = P1 @ V @ P2.T
    # 假设两地分布相同（本例每期只有一组mu/sigma）
    E_Future_V = P @ V_next @ P.T

    # Total G_matrix[i, j] = Cost if we achieve post-order level (state[i], state[j])
    # = L(state[i]) + L(state[j]) + E_Future_V[i, j]
    L_vec = expected_loss
    Total_Post_Decision_Cost = L_vec[:, None] + L_vec[None, :] + E_Future_V

    return Total_Post_Decision_Cost


def solve_stage_2_fast(i1_idx, i2_idx, cost_matrix):
    """
    给定转运后的状态，利用预计算矩阵找最优 Q。
    """
    # 当前状态索引
    base_i1 = i1_idx
    base_i2 = i2_idx

    # 搜索范围：我们不需要搜无限大，只需搜到合理的高水位
    # 假设最大订货量 20
    max_q_idx = 20

    # 切片获取相关的 Cost 区域
    # 我们要找 Q1, Q2 使得 cost_matrix[base_i1 + Q1, base_i2 + Q2] + K*I(Q) 最小
    # 注意边界检查
    limit_1 = min(n_states, base_i1 + max_q_idx + 1)
    limit_2 = min(n_states, base_i2 + max_q_idx + 1)

    sub_matrix = cost_matrix[base_i1:limit_1, base_i2:limit_2]

    # 添加订货成本 K
    # 这是一个 broadcasting 加法
    # Cost = MatrixVal + (K if row > 0) + (K if col > 0)

    # 快速找到不订货的成本 (0,0)
    cost_no_order = sub_matrix[0, 0]

    # 找到订货后的最小值
    # 加上 K
    k_mat_1 = np.full(sub_matrix.shape[0], K_order)
    k_mat_1[0] = 0
    k_mat_2 = np.full(sub_matrix.shape[1], K_order)
    k_mat_2[0] = 0

    total_cost_grid = sub_matrix + k_mat_1[:, None] + k_mat_2[None, :]

    # argmin
    min_idx_flat = np.argmin(total_cost_grid)
    q1_opt, q2_opt = np.unravel_index(min_idx_flat, total_cost_grid.shape)
    min_val = total_cost_grid[q1_opt, q2_opt]

    return min_val, (q1_opt, q2_opt)


# ==========================================
# 3. 主循环 (高速版)
# ==========================================
print("=== High Performance SDP-2 Solver Started ===")
start_time = time.time()

for t in range(T, 0, -1):
    print(f"Processing Period {t}...", end=" ")
    t_start = time.time()
    Policy[t] = {}

    # 1. 预计算本期的 Post-Decision Cost Matrix (Vectorized!)
    # 这步把最慢的 O(N^2 * D^2) 变成了矩阵乘法
    G_matrix = precompute_post_decision_cost(t, V)

    new_V = np.zeros((n_states, n_states))

    # 2. 状态循环
    for i1 in range(n_states):
        for i2 in range(n_states):
            val1 = state_values[i1]
            val2 = state_values[i2]

            # --- Stage 1: Search W (利用分段 K-Convexity) ---

            # A. 评估 W=0
            # 直接查表 G_matrix 找最优 Q
            base_cost, best_Qs = solve_stage_2_fast(i1, i2, G_matrix)

            best_W = 0
            min_total_cost = base_cost
            best_action = (0, best_Qs[0], best_Qs[1])

            # B. 搜索正向分支 W > 0 (1->2)
            # 只有当 i1 > 0 时才可能运
            max_w = min(15, max(0, val1))  # 限制最大搜索范围

            curr_w = 1
            while curr_w <= max_w:
                # 转运后的索引
                # i1 -> i1 - w, i2 -> i2 + w
                idx1_new = i1 - curr_w
                idx2_new = i2 + curr_w

                # 边界检查
                if idx1_new < 0 or idx2_new >= n_states:
                    break

                trans_cost = R_trans  # z=0
                # 快速求解 Stage 2 (Lookup)
                s2_cost, s2_qs = solve_stage_2_fast(idx1_new, idx2_new, G_matrix)
                total = trans_cost + s2_cost

                if total < min_total_cost:
                    min_total_cost = total
                    best_W = curr_w
                    best_action = (curr_w, s2_qs[0], s2_qs[1])
                elif total > min_total_cost + K_order:
                    # 【K-Convex 剪枝】: 如果成本上升超过 K，这碗到底了，停止！
                    break

                curr_w += 1

            # C. 搜索负向分支 W < 0 (2->1)
            # 逻辑同上
            max_w_neg = min(15, max(0, val2))
            curr_w = 1
            while curr_w <= max_w_neg:
                idx1_new = i1 + curr_w
                idx2_new = i2 - curr_w

                if idx1_new >= n_states or idx2_new < 0:
                    break

                trans_cost = R_trans
                s2_cost, s2_qs = solve_stage_2_fast(idx1_new, idx2_new, G_matrix)
                total = trans_cost + s2_cost

                if total < min_total_cost:
                    min_total_cost = total
                    best_W = -curr_w
                    best_action = (-curr_w, s2_qs[0], s2_qs[1])
                elif total > min_total_cost + K_order:
                    # 【K-Convex 剪枝】
                    break

                curr_w += 1

            new_V[i1, i2] = min_total_cost
            Policy[t][(i1, i2)] = best_action

    V = new_V  # 更新 Value Function
    print(f"Done in {time.time() - t_start:.2f}s")

print(f"Total Computation Time: {time.time() - start_time:.2f}s")

# ==========================================
# 4. 验证算例 (Trace)
# ==========================================
print("\n=== Simulation Trace (Non-Stationary) ===")
curr_i1, curr_i2 = 0, 0
# 强制复现之前的场景
forced_demands = [(2, 14), (5, 5), (8, 8), (5, 5)]

print(f"{'T':<3} {'State':<15} {'Action (W, Q1, Q2)':<20} {'After Trans':<15} {'Demand':<10} {'End State':<15}")

for t in range(1, T + 1):
    idx1 = np.clip(state_idx_map.get(curr_i1, 0), 0, n_states - 1)
    idx2 = np.clip(state_idx_map.get(curr_i2, 0), 0, n_states - 1)

    W, Q1, Q2 = Policy[t].get((idx1, idx2), (0, 0, 0))

    mid_i1 = curr_i1 - W
    mid_i2 = curr_i2 + W
    d1, d2 = forced_demands[t - 1]
    end_i1 = mid_i1 + Q1 - d1
    end_i2 = mid_i2 + Q2 - d2

    print(
        f"{t:<3} {str((curr_i1, curr_i2)):<15} {str((W, Q1, Q2)):<20} {str((mid_i1, mid_i2)):<15} {str((d1, d2)):<10} {str((end_i1, end_i2)):<15}")
    curr_i1, curr_i2 = end_i1, end_i2