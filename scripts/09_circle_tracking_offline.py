"""
Day 7 - 圆轨迹跟踪：完整管线演示

流程：
  1. 生成圆形目标轨迹（100 个 SE(3) 位姿）
  2. 对每个目标调用 IK，得到关节角序列
  3. 在 MuJoCo 中实时仿真执行
  4. 记录目标 vs 实际末端轨迹

第一周综合演示：综合运用 FK/Jacobian/IK 三件套
"""
import time
import numpy as np
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

from src.kinematics.forward import panda_fk
from src.kinematics.inverse import panda_ik, PANDA_HOME_THETA, PANDA_JOINT_LOWS, PANDA_JOINT_HIGHS


# ============== 参数 ==============
PANDA_XML = "assets/mujoco_menagerie/franka_emika_panda/panda.xml"

# 圆轨迹参数
CIRCLE_CENTER = np.array([0.5, 0.0, 0.5])  # 圆心 (x, y, z)
CIRCLE_RADIUS = 0.15                        # 半径（米）
CIRCLE_PERIOD = 10.0                         # 一周时间（秒）
N_POINTS = 100                              # 轨迹点数

# 仿真参数
SIM_DURATION = CIRCLE_PERIOD                # 仿真总时长


# ============== 1. 生成目标轨迹 ==============
def generate_circle_targets(center, radius, n_points):
    """
    生成 XY 平面上的圆形目标位姿序列
    
    末端朝向保持固定（朝下，类似 HOME 姿态的朝向）
    """
    # HOME 姿态末端朝下，作为固定朝向参考
    T_home = panda_fk(PANDA_HOME_THETA)
    R_target = T_home[:3, :3]  # 用 HOME 时末端的朝向
    
    targets = []
    for i in range(n_points):
        angle = 2 * np.pi * i / n_points
        pos = center + radius * np.array([np.cos(angle), np.sin(angle), 0])
        
        T = np.eye(4)
        T[:3, :3] = R_target
        T[:3, 3] = pos
        targets.append(T)
    
    return targets


# ============== 2. 离线求解 IK 序列 ==============
def solve_ik_sequence(targets):
    """对轨迹上每个目标位姿求解 IK，返回关节角序列"""
    print("Solving IK for {} targets...".format(len(targets)))
    theta_sequence = []
    
    # 第一个目标从 HOME 开始
    theta_current = PANDA_HOME_THETA.copy()
    
    n_success = 0
    total_iters = 0
    
    for i, T_target in enumerate(targets):
        # 用上一个解作为下一个初值（连续目标技巧）
        theta_solved, success, info = panda_ik(
            T_target, 
            theta_init=theta_current,
            max_iter=100,
            tol_pos=1e-4,
            tol_rot=1e-3,
        )
        
        if success:
            n_success += 1
            total_iters += info['iters']
            theta_current = theta_solved  # 更新为下一个初值
        else:
            # 失败时用 HOME 重启一次
            theta_solved, success, info = panda_ik(
                T_target,
                theta_init=PANDA_HOME_THETA,
                max_iter=200,
            )
            if success:
                n_success += 1
                total_iters += info['iters']
                theta_current = theta_solved
            else:
                print("  WARN: target {} failed both attempts".format(i))
        
        theta_sequence.append(theta_solved.copy())
    
    print("IK Stats:")
    print("  Success: {}/{}".format(n_success, len(targets)))
    print("  Avg iterations: {:.1f}".format(total_iters / max(n_success, 1)))
    
    return np.array(theta_sequence)


# ============== 3. MuJoCo 仿真执行 ==============
def run_mujoco_simulation(theta_sequence, targets):
    """在 MuJoCo 里按关节角序列执行，并实时显示"""
    model = mujoco.MjModel.from_xml_path(PANDA_XML)
    data = mujoco.MjData(model)
    
    # 找到 hand body
    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    
    # 初始化到 HOME
    data.qpos[:7] = PANDA_HOME_THETA
    data.qpos[7:] = 0.04
    data.ctrl[:7] = PANDA_HOME_THETA
    mujoco.mj_forward(model, data)
    
    # 记录数据
    actual_ee_positions = []
    target_ee_positions = []
    
    n_points = len(theta_sequence)
    dt_per_point = SIM_DURATION / n_points
    
    print("\nLaunching MuJoCo viewer...")
    print("Robot will draw a circle of radius {} m at center {}".format(
        CIRCLE_RADIUS, CIRCLE_CENTER))
    print("Total duration: {:.1f} s".format(SIM_DURATION))
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 静止 1 秒让你看清 HOME 姿态
        t_pause_start = time.time()
        while viewer.is_running() and time.time() - t_pause_start < 1.0:
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)
        
        # 跟踪圆轨迹
        sim_start = time.time()
        for i, theta_target in enumerate(theta_sequence):
            if not viewer.is_running():
                break
            
            point_start = time.time()
            
            # 设置控制目标
            data.ctrl[:7] = theta_target
            
            # 仿真直到下一个点的时间
            while (time.time() - point_start) < dt_per_point:
                if not viewer.is_running():
                    break
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(0.002)  # 让 viewer 喘口气
            
            # 记录当前末端位置
            actual_ee_positions.append(data.xpos[hand_id].copy())
            target_ee_positions.append(targets[i][:3, 3])
        
        # 最后再静止 1 秒
        t_end = time.time()
        while viewer.is_running() and time.time() - t_end < 1.0:
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)
    
    print("\nSimulation complete.")
    return np.array(actual_ee_positions), np.array(target_ee_positions)


# ============== 4. 可视化轨迹对比 ==============
def visualize_trajectories(actual_pos, target_pos):
    """3D 对比图 + 误差分析"""
    fig = plt.figure(figsize=(14, 6))
    
    # 左图：3D 轨迹对比
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot(target_pos[:, 0], target_pos[:, 1], target_pos[:, 2],
             'b-', linewidth=2, label='Target', alpha=0.7)
    ax1.plot(actual_pos[:, 0], actual_pos[:, 1], actual_pos[:, 2],
             'r--', linewidth=2, label='Actual', alpha=0.7)
    ax1.scatter(target_pos[0, 0], target_pos[0, 1], target_pos[0, 2],
                c='green', s=100, marker='o', label='Start', zorder=10)
    ax1.scatter(target_pos[-1, 0], target_pos[-1, 1], target_pos[-1, 2],
                c='red', s=100, marker='X', label='End', zorder=10)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('Circle Trajectory: Target vs Actual')
    ax1.legend()
    
    # 右图：跟踪误差随时间变化
    ax2 = fig.add_subplot(1, 2, 2)
    n = len(actual_pos)
    t = np.linspace(0, SIM_DURATION, n)
    errors = np.linalg.norm(actual_pos - target_pos, axis=1) * 1000  # 转毫米
    
    ax2.plot(t, errors, 'g-', linewidth=2)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Tracking error (mm)')
    ax2.set_title('Position Tracking Error vs Time\n(Max: {:.2f} mm, Mean: {:.2f} mm)'.format(
        np.max(errors), np.mean(errors)))
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('logs/day7_circle_tracking.png', dpi=120, bbox_inches='tight')
    print("\nFigure saved to: logs/day7_circle_tracking.png")
    plt.show()


# ============== 主流程 ==============
if __name__ == "__main__":
    print("=" * 60)
    print("Day 7: Circle Trajectory Tracking Demo")
    print("=" * 60)
    
    # 1. 生成目标
    print("\n[1/4] Generating circle targets...")
    targets = generate_circle_targets(CIRCLE_CENTER, CIRCLE_RADIUS, N_POINTS)
    print("  Generated {} target poses".format(len(targets)))
    
    # 2. 离线求 IK
    print("\n[2/4] Solving IK sequence...")
    theta_sequence = solve_ik_sequence(targets)
    
    # 3. MuJoCo 仿真
    print("\n[3/4] Running MuJoCo simulation...")
    actual_pos, target_pos = run_mujoco_simulation(theta_sequence, targets)
    
    # 4. 可视化
    print("\n[4/4] Visualizing trajectories...")
    visualize_trajectories(actual_pos, target_pos)
    
    print("\n" + "=" * 60)
    print("Day 7 demo complete!")
    print("=" * 60)