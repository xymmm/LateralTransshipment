from algorithm import algorithm
from lt_sdp import SDP2

params = dict(
    demand_mean1=[4, 6, 8, 6],
    demand_mean2=[4, 6, 8, 6],
    K=20, z=1, R=5, v=0.5, h=0.25, b=5,
    distribution="poisson",      # 或 "normal"，同时把 cv 设为 0.25 之类
    cv=None,
)

s = SDP2(**params, min_inventory=-20, max_inventory=60)
r = algorithm(**params)

print(f"SDP-2:       C_1(0,0) = {s.value():.4f}   action {s.action()}   time {s.seconds:.2f}s")
print(f"Algorithm 1: ETC = {r.etc:.4f} +- {r.half_width:.4f}   first decision {r.first_decision}   "
      f"paths {r.n_paths}   time {r.seconds_total:.1f}s")
print(f"gap = {100 * (r.etc - s.value()) / s.value():.3f}%")