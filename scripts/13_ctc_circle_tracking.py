"""
Day 10 - CTC 控制器：圆轨迹跟踪

对比目标：
- Day 9 PD + G(q):  稳态误差 0.23 mm
- 今天 CTC:          稳态误差 < 0.1 mm（目标）
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
from src.controllers.pd_gravity import disable_actuators, apply_torque
from src.controllers.ctc import (
    compute_ctc_torque,
    PANDA_CTC_KP,
    PANDA_CTC_KD,
)


PANDA_XML = "assets/mujoco_menagerie/franka_emika_panda/panda.xml"

# 圆轨迹参数（与 Day 9 一致，便于对比）
CIRCLE_CENTER = np.array([0.5, 0.0, 0.5])
CIRCLE_RADIUS = 0.15
CIRCLE_PERIOD = 10.0
N_POINTS = 200

# 仿真参数
USE_VIEWER = True
WARMUP_TIME = 1.0
COOLDOWN_TIME = 1.0
SIM_DURATION = WARMUP_TIME + CIRCLE_PERIOD + COOLDOWN_TIME


def generate_trajectory():
    """生成圆轨迹 (q_d, qdot_d, qddot_d) 序列"""
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
    
    # 再次微分得到 qddot_d
    qddot_d_array = np.gradient(qdot_d_array, dt, axis=0)
    
    print("Trajectory generated: {} points, {:.3f}s per point".format(
        N_POINTS, dt))
    return times, q_d_array, qdot_d_array, qddot_d_array


def interpolate_target(t, times, q_d_array, qdot_d_array, qddot_d_array):
    """在 t 时刻查询目标 (q_d, qdot_d, qddot_d)"""
    T_total = times[-1]
    
    if t <= 0:
        return q_d_array[0].copy(), np.zeros(7), np.zeros(7)
    if t >= T_total:
        return q_d_array[-1].copy(), np.zeros(7), np.zeros(7)
    
    idx = np.searchsorted(times, t) - 1
    idx = np.clip(idx, 0, len(times) - 2)
    alpha = (t - times[idx]) / (times[idx + 1] - times[idx])
    
    q_d = (1 - alpha) * q_d_array[idx] + alpha * q_d_array[idx + 1]
    qdot_d = (1 - alpha) * qdot_d_array[idx] + alpha * qdot_d_array[idx + 1]
    qddot_d = (1 - alpha) * qddot_d_array[idx] + alpha * qddot_d_array[idx + 1]
    
    return q_d, qdot_d, qddot_d


def run_simulation(times_traj, q_d_array, qdot_d_array, qddot_d_array, Kp, Kd):
    """用 CTC 控制器跑圆轨迹仿真"""
    model = mujoco.MjModel.from_xml_path(PANDA_XML)
    data = mujoco.MjData(model)
    
    disable_actuators(model)
    
    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    
    # 初始化到轨迹起点
    data.qpos[:7] = q_d_array[0]
    data.qpos[7:] = 0.04
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    
    dt = model.opt.timestep
    n_steps = int(SIM_DURATION / dt)
    
    record_times = []
    record_target_ee = []
    record_actual_ee = []
    record_q_err = []
    record_tau = []
    
    print("\nRunning simulation with Computed Torque Control...")
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
    
    try:
        for step in range(n_steps):
            t_traj = t_sim - WARMUP_TIME
            
            if t_traj < 0:
                q_d = q_d_array[0].copy()
                qdot_d = np.zeros(7)
                qddot_d = np.zeros(7)
            elif t_traj > CIRCLE_PERIOD:
                q_d = q_d_array[-1].copy()
                qdot_d = np.zeros(7)
                qddot_d = np.zeros(7)
            else:
                q_d, qdot_d, qddot_d = interpolate_target(
                    t_traj, times_traj, q_d_array, qdot_d_array, qddot_d_array)
            
            tau = compute_ctc_torque(
                model, data, q_d, qdot_d, qddot_d, Kp, Kd)
            
            apply_torque(data, tau)
            
            mujoco.mj_step(model, data)
            t_sim += dt
            
            if step % 4 == 0:
                actual_ee = data.xpos[hand_id].copy()
                
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
    """可视化 CTC 结果"""
    fig = plt.figure(figsize=(15, 10))
    
    # 3D 轨迹
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.plot(target_ee[:, 0], target_ee[:, 1], target_ee[:, 2],
             'b-', linewidth=2, label='Target', alpha=0.8)
    ax1.plot(actual_ee[:, 0], actual_ee[:, 1], actual_ee[:, 2],
             'r--', linewidth=2, label='Actual (CTC)', alpha=0.8)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('Circle Trajectory: Target vs Actual\n(Computed Torque Control)')
    ax1.legend()
    
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
    
    steady_idx = (times > WARMUP_TIME + 0.5) & (times < WARMUP_TIME + CIRCLE_PERIOD - 0.5)
    if np.any(steady_idx):
        mean_err = np.mean(pos_errors[steady_idx])
        max_err = np.max(pos_errors[steady_idx])
    else:
        mean_err = np.mean(pos_errors)
        max_err = np.max(pos_errors)
    
    ax2.set_title('Position Tracking Error vs Time\n'
                  '(Steady state: Mean {:.3f} mm, Max {:.3f} mm)'.format(
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
    plt.savefig('logs/day10_ctc_tracking.png', 
                dpi=120, bbox_inches='tight')
    print("\nFigure saved to: logs/day10_ctc_tracking.png")
    
    print("\n===== Summary =====")
    print("Mean steady-state error: {:.3f} mm".format(mean_err))
    print("Max steady-state error: {:.3f} mm".format(max_err))
    
    if mean_err < 0.1:
        print("Result: EXCELLENT (target < 0.1mm)")
    elif mean_err < 0.5:
        print("Result: GOOD (better than Day 9 PD+G)")
    else:
        print("Result: Acceptable but check tuning")
    
    plt.show()


if __name__ == "__main__":
    print("=" * 60)
    print("Day 10: Computed Torque Control - Circle Tracking")
    print("=" * 60)
    
    times_traj, q_d_array, qdot_d_array, qddot_d_array = generate_trajectory()
    
    times, target_ee, actual_ee, q_err, tau = run_simulation(
        times_traj, q_d_array, qdot_d_array, qddot_d_array,
        PANDA_CTC_KP, PANDA_CTC_KD
    )
    
    visualize_results(times, target_ee, actual_ee, q_err, tau)
    
    print("\n" + "=" * 60)
    print("Day 10 complete!")
    print("=" * 60)