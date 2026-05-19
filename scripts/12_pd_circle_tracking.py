"""
Day 9 - PD + 重力补偿控制器：圆轨迹跟踪

对比目标：
- Day 7 MuJoCo 默认位置伺服器: 稳态误差 ~20-40mm
- 今天 PD + G(q):                稳态误差 < 5mm
"""
import time
import numpy as np
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

from src.kinematics.forward import panda_fk
from src.kinematics.inverse import panda_ik, PANDA_HOME_THETA
from src.controllers.pd_gravity import (
    compute_pd_gravity_torque,
    apply_torque,
    disable_actuators,
    PANDA_KP_DEFAULT,
    PANDA_KD_DEFAULT,
)


PANDA_XML = "assets/mujoco_menagerie/franka_emika_panda/panda.xml"

# 圆轨迹参数
CIRCLE_CENTER = np.array([0.5, 0.0, 0.5])
CIRCLE_RADIUS = 0.15
CIRCLE_PERIOD = 10.0   # 一周时间（秒）
N_POINTS = 200          # 轨迹采样点数

# 控制参数
USE_VIEWER = True       # 是否实时显示
SIM_DURATION = CIRCLE_PERIOD + 2.0   # 1秒预热 + 10秒圆 + 1秒收尾


def generate_trajectory():
    """生成圆轨迹的关节空间序列 (q_d, qdot_d)"""
    T_home = panda_fk(PANDA_HOME_THETA)
    R_target = T_home[:3, :3]
    
    print("Generating trajectory by offline IK...")
    
    times = np.linspace(0, CIRCLE_PERIOD, N_POINTS)
    q_d_list = []
    theta_current = PANDA_HOME_THETA.copy()
    
    for t in times:
        angle = 2 * np.pi * t / CIRCLE_PERIOD
        pos = CIRCLE_CENTER + CIRCLE_RADIUS * np.array(
            [np.cos(angle), np.sin(angle), 0]
        )
        T_target = np.eye(4)
        T_target[:3, :3] = R_target
        T_target[:3, 3] = pos
        
        theta_solved, success, _ = panda_ik(
            T_target, theta_init=theta_current, max_iter=100
        )
        if success:
            theta_current = theta_solved
        q_d_list.append(theta_current.copy())
    
    q_d_array = np.array(q_d_list)
    
    # 数值微分得到 qdot_d
    dt = CIRCLE_PERIOD / N_POINTS
    qdot_d_array = np.gradient(q_d_array, dt, axis=0)
    
    print("Trajectory generated: {} points, {:.3f}s per point".format(
        N_POINTS, dt))
    return times, q_d_array, qdot_d_array


def interpolate_target(t, times, q_d_array, qdot_d_array):
    """
    在 t 时刻查询目标 (q_d, qdot_d)
    
    简单线性插值，超出范围时 qdot_d = 0
    """
    T_total = times[-1]
    
    if t <= 0:
        return q_d_array[0].copy(), np.zeros(7)
    if t >= T_total:
        return q_d_array[-1].copy(), np.zeros(7)
    
    idx = np.searchsorted(times, t) - 1
    idx = np.clip(idx, 0, len(times) - 2)
    alpha = (t - times[idx]) / (times[idx + 1] - times[idx])
    
    q_d = (1 - alpha) * q_d_array[idx] + alpha * q_d_array[idx + 1]
    qdot_d = (1 - alpha) * qdot_d_array[idx] + alpha * qdot_d_array[idx + 1]
    
    return q_d, qdot_d


def run_simulation(times_traj, q_d_array, qdot_d_array, Kp, Kd):
    """
    用 PD + G 控制器跑圆轨迹仿真
    
    Returns: 时间序列、目标末端位置、实际末端位置、关节误差
    """
    model = mujoco.MjModel.from_xml_path(PANDA_XML)
    data = mujoco.MjData(model)

    # 完全禁用 actuator，让 qfrc_applied 成为唯一力矩源
    disable_actuators(model)

    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    
    # 初始化到轨迹起点附近（不是 HOME，避免初始 jump）
    data.qpos[:7] = q_d_array[0]
    data.qpos[7:] = 0.04
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    
    dt = model.opt.timestep
    n_steps = int(SIM_DURATION / dt)
    
    # 记录数据
    record_times = []
    record_target_ee = []
    record_actual_ee = []
    record_q_err = []
    record_tau = []
    
    print("\nRunning simulation with PD + Gravity Compensation...")
    print("  dt = {:.4f}s, total steps = {}".format(dt, n_steps))
    print("  Kp = {}".format(Kp))
    print("  Kd = {}".format(Kd))
    
    if USE_VIEWER:
        viewer = mujoco.viewer.launch_passive(model, data)
        viewer.cam.lookat = np.array([0.4, 0.0, 0.5])
        viewer.cam.distance = 1.6
        viewer.cam.azimuth = 130
        viewer.cam.elevation = -25
    
    t_sim = 0.0
    real_time_start = time.time()
    
    # 1 秒静止预热 + 圆轨迹 + 1 秒静止收尾
    WARMUP_TIME = 1.0     # 仿真开始时静止 1 秒
    COOLDOWN_TIME = 1.0   # 圆结束后静止 1 秒
    
    try:
        for step in range(n_steps):
            # ============== 时间映射 ==============
            t_traj = t_sim - WARMUP_TIME
            
            if t_traj < 0:
                # 预热阶段：保持在轨迹第一个点
                q_d = q_d_array[0].copy()
                qdot_d = np.zeros(7)
            elif t_traj > CIRCLE_PERIOD:
                # 收尾阶段：保持在轨迹最后一个点
                q_d = q_d_array[-1].copy()
                qdot_d = np.zeros(7)
            else:
                # 正常跟踪圆轨迹
                q_d, qdot_d = interpolate_target(
                    t_traj, times_traj, q_d_array, qdot_d_array)
            
            # ============== 控制 ==============
            tau = compute_pd_gravity_torque(
                model, data, q_d, qdot_d, Kp, Kd)
            
            apply_torque(data, tau)
            
            mujoco.mj_step(model, data)
            t_sim += dt
            
            # ============== 记录 ==============
            if step % 4 == 0:
                actual_ee = data.xpos[hand_id].copy()
                
                # 计算目标末端位置（在圆上的对应点）
                if t_traj < 0:
                    target_pos = CIRCLE_CENTER + CIRCLE_RADIUS * np.array([1, 0, 0])
                elif t_traj > CIRCLE_PERIOD:
                    target_pos = CIRCLE_CENTER + CIRCLE_RADIUS * np.array([1, 0, 0])
                else:
                    angle = 2 * np.pi * t_traj / CIRCLE_PERIOD
                    target_pos = CIRCLE_CENTER + CIRCLE_RADIUS * np.array(
                        [np.cos(angle), np.sin(angle), 0])
                
                record_times.append(t_sim)
                record_target_ee.append(target_pos)
                record_actual_ee.append(actual_ee)
                record_q_err.append(q_d - data.qpos[:7].copy())
                record_tau.append(tau.copy())
            
            # 实时显示
            if USE_VIEWER and step % 10 == 0:
                if not viewer.is_running():
                    break
                viewer.sync()
                real_elapsed = time.time() - real_time_start
                if t_sim > real_elapsed:
                    time.sleep(min(0.01, t_sim - real_elapsed))
    
    finally:
        if USE_VIEWER:
            viewer.close()
    
    print("Simulation complete.")
    return (np.array(record_times), 
            np.array(record_target_ee),
            np.array(record_actual_ee),
            np.array(record_q_err),
            np.array(record_tau))


def visualize_results(times, target_ee, actual_ee, q_err, tau):
    """可视化跟踪结果"""
    fig = plt.figure(figsize=(15, 10))
    
    # 3D 轨迹（等比例坐标轴）
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.plot(target_ee[:, 0], target_ee[:, 1], target_ee[:, 2],
             'b-', linewidth=2, label='Target', alpha=0.8)
    ax1.plot(actual_ee[:, 0], actual_ee[:, 1], actual_ee[:, 2],
             'r--', linewidth=2, label='Actual (PD+G)', alpha=0.8)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('Circle Trajectory: Target vs Actual\n(PD + Gravity Compensation)')
    ax1.legend()
    
    # 强制等比例坐标轴（避免 Z 轴被自动放大）
    center = CIRCLE_CENTER
    range_half = CIRCLE_RADIUS * 1.3
    ax1.set_xlim(center[0] - range_half, center[0] + range_half)
    ax1.set_ylim(center[1] - range_half, center[1] + range_half)
    ax1.set_zlim(center[2] - range_half, center[2] + range_half)
    ax1.set_box_aspect([1, 1, 1])
    
    # 跟踪误差
    ax2 = fig.add_subplot(2, 2, 2)
    pos_errors = np.linalg.norm(actual_ee - target_ee, axis=1) * 1000  # mm
    ax2.plot(times, pos_errors, 'g-', linewidth=2)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Position error (mm)')
    
    # 稳态：去掉预热（前 1.5 秒）和收尾（最后 1.5 秒）
    steady_idx = (times > 1.5) & (times < CIRCLE_PERIOD + 0.5)
    if np.any(steady_idx):
        mean_err = np.mean(pos_errors[steady_idx])
        max_err = np.max(pos_errors[steady_idx])
    else:
        mean_err = np.mean(pos_errors)
        max_err = np.max(pos_errors)
    
    ax2.set_title('Position Tracking Error vs Time\n'
                  '(Steady state: Mean {:.2f} mm, Max {:.2f} mm)'.format(
                      mean_err, max_err))
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=mean_err, color='orange', linestyle='--', 
                alpha=0.6, label='Mean (steady)')
    ax2.legend()
    
    # 关节误差
    ax3 = fig.add_subplot(2, 2, 3)
    for i in range(7):
        ax3.plot(times, q_err[:, i] * 1000, 
                 linewidth=1.5, label='joint{}'.format(i+1))
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Joint position error (mrad)')
    ax3.set_title('Joint-space tracking errors')
    ax3.legend(loc='best', ncol=2, fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # 力矩
    ax4 = fig.add_subplot(2, 2, 4)
    for i in range(7):
        ax4.plot(times, tau[:, i], linewidth=1.5, label='joint{}'.format(i+1))
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Joint torque (N·m)')
    ax4.set_title('Control torques over time')
    ax4.legend(loc='best', ncol=2, fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('logs/day9_pd_gravity_tracking.png', 
                dpi=120, bbox_inches='tight')
    print("\nFigure saved to: logs/day9_pd_gravity_tracking.png")
    
    print("\n===== Summary =====")
    print("Mean steady-state error: {:.3f} mm".format(mean_err))
    print("Max steady-state error: {:.3f} mm".format(max_err))
    
    if mean_err < 5.0:
        print("Result: PASS (target < 5mm)")
    else:
        print("Result: Acceptable but room for improvement (try CTC in Day 10)")
    
    plt.show()


if __name__ == "__main__":
    print("=" * 60)
    print("Day 9: PD + Gravity Compensation - Circle Tracking")
    print("=" * 60)
    
    times_traj, q_d_array, qdot_d_array = generate_trajectory()
    
    times, target_ee, actual_ee, q_err, tau = run_simulation(
        times_traj, q_d_array, qdot_d_array,
        PANDA_KP_DEFAULT, PANDA_KD_DEFAULT
    )
    
    visualize_results(times, target_ee, actual_ee, q_err, tau)
    
    print("\n" + "=" * 60)
    print("Day 9 complete!")
    print("=" * 60)