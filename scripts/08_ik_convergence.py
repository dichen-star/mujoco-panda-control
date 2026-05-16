"""
Day 6 - IK 收敛过程可视化

观察一个中等难度目标的 IK 收敛轨迹：
- 误差如何指数下降
- 7 个关节角如何平滑演化
"""
import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

from src.kinematics.forward import panda_fk
from src.kinematics.inverse import panda_ik, PANDA_HOME_THETA, PANDA_JOINT_LOWS, PANDA_JOINT_HIGHS


# 选一个 HOME 初值能顺利收敛的中等难度目标
# seed=42 经测试是一次性收敛的好例子
np.random.seed(0)
theta_truth = np.random.uniform(
    PANDA_JOINT_LOWS + 0.5,
    PANDA_JOINT_HIGHS - 0.5
)
T_target = panda_fk(theta_truth)

print("Target joint angles (ground truth):")
print("  {}".format(theta_truth))
print("Target end-effector position: {}".format(T_target[:3, 3]))
print()

# 跑 IK：限制 max_iter=50 让图更清爽
theta_solved, success, info = panda_ik(
    T_target,
    theta_init=PANDA_HOME_THETA,
    max_iter=50,
)

print("IK Result:")
print("  Converged       = {}".format(success))
print("  Iterations      = {}".format(info['iters']))
print("  Final pos error = {:.2e} m".format(info['final_pos_err']))
print("  Final rot error = {:.2e} rad".format(info['final_rot_err']))
print()
print("Solved joint angles:")
print("  {}".format(theta_solved))
print()
print("Note: solved theta != ground truth because Panda is 7-DoF redundant.")
print("Both produce the same end-effector pose, just from different configurations.")

# 提取历史数据
history = info['history']
n_iters = len(history['pos_err'])
iters = np.arange(n_iters)
theta_history = np.array(history['theta'][:n_iters])

# 画图
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ============== 左图：误差对数收敛 ==============
ax = axes[0]
ax.semilogy(iters, history['pos_err'], 'b-', linewidth=2,
            marker='o', markersize=4, label='Position error (m)')
ax.semilogy(iters, history['rot_err'], 'r-', linewidth=2,
            marker='s', markersize=4, label='Rotation error (rad)')
ax.axhline(y=1e-4, color='gray', linestyle='--', alpha=0.7,
           label='Convergence threshold (1e-4)')
ax.set_xlabel('Iteration', fontsize=11)
ax.set_ylabel('Error (log scale)', fontsize=11)
ax.set_title('IK Convergence: Error vs Iteration\n(DLS with lambda=0.01)',
             fontsize=12)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3, which='both')

# 标注关键点
if success:
    ax.annotate('Converged at iter {}'.format(info['iters']),
                xy=(info['iters'] - 1, history['pos_err'][-1]),
                xytext=(info['iters'] * 0.5, 1e-2),
                arrowprops=dict(arrowstyle='->', color='green', alpha=0.7),
                fontsize=10, color='green')

# ============== 右图：7 个关节角随迭代变化 ==============
ax = axes[1]
colors = plt.cm.tab10(np.arange(7))
for i in range(7):
    ax.plot(iters, theta_history[:, i],
            color=colors[i], linewidth=2,
            label='joint{}'.format(i + 1))

# 标出 HOME 初值（起点）和最终解
ax.scatter([0] * 7, theta_history[0], c='black', s=50, marker='o',
           zorder=5, label='Initial (HOME)' if False else None)
ax.scatter([n_iters - 1] * 7, theta_history[-1], c='black', s=80,
           marker='X', zorder=5)

ax.set_xlabel('Iteration', fontsize=11)
ax.set_ylabel('Joint angle (rad)', fontsize=11)
ax.set_title('Joint Angles Trajectory During IK\n(start: HOME pose, end: solution)',
             fontsize=12)
ax.legend(loc='upper right', fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)

plt.tight_layout()

# 保存
output_path = 'logs/day6_ik_convergence.png'
plt.savefig(output_path, dpi=120, bbox_inches='tight')
print("\nFigure saved to: {}".format(output_path))

plt.show()