"""
Day 6 - IK 收敛过程可视化
观察一个中等难度目标的 IK 收敛轨迹
"""
import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

from src.kinematics.forward import panda_fk
from src.kinematics.inverse import panda_ik, PANDA_HOME_THETA, PANDA_JOINT_LOWS, PANDA_JOINT_HIGHS


# 选一个中等难度的随机目标
np.random.seed(7)
theta_truth = np.random.uniform(
    PANDA_JOINT_LOWS + 0.5,
    PANDA_JOINT_HIGHS - 0.5
)
T_target = panda_fk(theta_truth)

print("Target joint angles = {}".format(theta_truth))
print("Target end-effector position = {}".format(T_target[:3, 3]))

# 跑 IK
theta_solved, success, info = panda_ik(T_target, theta_init=PANDA_HOME_THETA)

print("\nIK Result:")
print("  Converged = {}, Iterations = {}".format(success, info['iters']))
print("  Final position error = {:.2e}".format(info['final_pos_err']))
print("  Final rotation error = {:.2e}".format(info['final_rot_err']))

# 提取历史数据
history = info['history']
n_iters = len(history['pos_err'])
iters = np.arange(n_iters)
theta_history = np.array(history['theta'][:n_iters])

# 画图
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图：误差对数收敛
axes[0].semilogy(iters, history['pos_err'], 'b-', linewidth=2, label='Position error (m)')
axes[0].semilogy(iters, history['rot_err'], 'r-', linewidth=2, label='Rotation error (rad)')
axes[0].axhline(y=1e-4, color='gray', linestyle='--', alpha=0.5, label='Convergence threshold')
axes[0].set_xlabel('Iteration')
axes[0].set_ylabel('Error (log scale)')
axes[0].set_title('IK Convergence: Error vs Iteration')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 右图：7 个关节角随迭代变化
colors = plt.cm.tab10(np.arange(7))
for i in range(7):
    axes[1].plot(iters, theta_history[:, i], color=colors[i], 
                 label='joint{}'.format(i+1), linewidth=1.5)
axes[1].set_xlabel('Iteration')
axes[1].set_ylabel('Joint angle (rad)')
axes[1].set_title('Joint Angles Trajectory During IK')
axes[1].legend(loc='upper right', fontsize=8, ncol=2)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('logs/day6_ik_convergence.png', dpi=120, bbox_inches='tight')
print("\nFigure saved to: logs/day6_ik_convergence.png")
plt.show()