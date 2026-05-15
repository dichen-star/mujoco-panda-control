"""
Day 5 - 奇异位形可视化

扫描 joint4 从 -π/2 到 0（手臂从弯曲到完全伸直），
观察可操作度 μ 和最小奇异值的变化，理解奇异位形。
"""
import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

from src.kinematics.forward import panda_jacobian
from src.kinematics.jacobian import manipulability, smallest_singular_value


# 扫描 joint4 从弯曲到伸直
N = 200
theta4_range = np.linspace(-np.pi/2, -0.0698, N)

mu_list = []
sigma_min_list = []

# 一个远离奇异的工作姿态（类似 home 姿态）
base_theta = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785])

for theta4 in theta4_range:
    theta = base_theta.copy()
    theta[3] = theta4  # 改变 joint4
    
    J = panda_jacobian(theta)
    mu_list.append(manipulability(J))
    sigma_min_list.append(smallest_singular_value(J))

mu_list = np.array(mu_list)
sigma_min_list = np.array(sigma_min_list)

# 画图
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(theta4_range, mu_list, 'b-', linewidth=2)
axes[0].set_xlabel('joint4 angle (rad)')
axes[0].set_ylabel('Manipulability mu')
axes[0].set_title('Manipulability vs joint4 angle')
axes[0].grid(True, alpha=0.3)
axes[0].axvline(x=-0.0698, color='r', linestyle='--', label='joint4 limit (-0.07)')
axes[0].legend()

axes[1].plot(theta4_range, sigma_min_list, 'g-', linewidth=2)
axes[1].set_xlabel('joint4 angle (rad)')
axes[1].set_ylabel('Smallest singular value sigma_6')
axes[1].set_title('Smallest singular value vs joint4 angle')
axes[1].grid(True, alpha=0.3)
axes[1].axvline(x=-0.0698, color='r', linestyle='--', label='joint4 limit')
axes[1].legend()

plt.tight_layout()

# 保存
output = 'logs/day5_singularity.png'
plt.savefig(output, dpi=120, bbox_inches='tight')
print(f"图已保存到: {output}")
print(f"\n关键观察:")
print(f"  joint4 = -pi/2 (弯曲) -> mu = {mu_list[0]:.4f}, sigma_min = {sigma_min_list[0]:.4f}")
print(f"  joint4 = -0.07 (接近伸直) -> mu = {mu_list[-1]:.4f}, sigma_min = {sigma_min_list[-1]:.4f}")
print(f"  -> 接近伸直时可操作度急剧下降，这就是奇异位形")
