"""
Day 11 - 笛卡尔阻抗控制：柔顺性演示

实验设计：
1. 让 Panda 末端"挂"在目标位置 x_d
2. 在仿真中模拟外力推动末端
3. 观察不同刚度下末端的柔顺响应
4. 对比 SOFT / MEDIUM / STIFF 三种刚度
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
from src.controllers.impedance import (
    compute_impedance_torque,
    PANDA_K_CART_DEFAULT, PANDA_D_CART_DEFAULT,
    PANDA_K_CART_STIFF, PANDA_D_CART_STIFF,
    PANDA_K_CART_SOFT, PANDA_D_CART_SOFT,
)


PANDA_XML = "assets/mujoco_menagerie/franka_emika_panda/panda.xml"

# 目标位置：让 Panda 末端"悬停"在这里
TARGET_POS = np.array([0.5, 0.0, 0.5])
TARGET_QDOT = np.zeros(3)

# 仿真参数
SIM_DURATION = 8.0       # 总时长
USE_VIEWER = True

# 外力施加（模拟有人推末端）
PUSH_START_TIME = 2.0    # 2s 时开始推
PUSH_END_TIME = 4.0      # 4s 时停止推
PUSH_FORCE = np.array([15.0, 0.0, 0.0])  # X 方向推 15 N


def get_initial_q():
    """求初始关节角，让末端在 TARGET_POS"""
    T_target = np.eye(4)
    T_home = panda_fk(PANDA_HOME_THETA)
    T_target[:3, :3] = T_home[:3, :3]  # 用 HOME 朝向
    T_target[:3, 3] = TARGET_POS
    
    q_init, success, _ = panda_ik(T_target, theta_init=PANDA_HOME_THETA)
    if not success:
        print("WARN: IK 没收敛，用 HOME 初始化")
        return PANDA_HOME_THETA.copy()
    return q_init


def run_impedance_simulation(K_cart, D_cart, label="medium"):
    """运行阻抗控制仿真，记录末端响应"""
    model = mujoco.MjModel.from_xml_path(PANDA_XML)
    data = mujoco.MjData(model)
    
    disable_actuators(model)
    
    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    
    # 初始化到目标位置
    q_init = get_initial_q()
    data.qpos[:7] = q_init
    data.qpos[7:] = 0.04
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    
    dt = model.opt.timestep
    n_steps = int(SIM_DURATION / dt)
    
    record_times = []
    record_ee_pos = []
    record_ee_x_err = []
    record_tau_norm = []
    record_external_force = []
    
    print("\nRunning {} impedance simulation...".format(label))
    print("  K_cart = {}".format(K_cart))
    print("  D_cart = {}".format(D_cart))
    print("  Push: {} N during [{}, {}] s".format(
        PUSH_FORCE, PUSH_START_TIME, PUSH_END_TIME))
    
    viewer = None
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
            # 模拟外力推动
            external_force = np.zeros(3)
            if PUSH_START_TIME <= t_sim < PUSH_END_TIME:
                external_force = PUSH_FORCE
            
            # 把外力施加在 hand body 上
            # MuJoCo xfrc_applied[body_id] = [Fx, Fy, Fz, Tx, Ty, Tz]
            data.xfrc_applied[hand_id, :3] = external_force
            data.xfrc_applied[hand_id, 3:] = 0
            
            # 阻抗控制
            tau = compute_impedance_torque(
                model, data, TARGET_POS, TARGET_QDOT, K_cart, D_cart
            )
            
            apply_torque(data, tau)
            
            mujoco.mj_step(model, data)
            t_sim += dt
            
            if step % 4 == 0:
                ee_pos = data.xpos[hand_id].copy()
                x_err = ee_pos - TARGET_POS  # 偏离目标
                
                record_times.append(t_sim)
                record_ee_pos.append(ee_pos)
                record_ee_x_err.append(x_err)
                record_tau_norm.append(np.linalg.norm(tau))
                record_external_force.append(external_force.copy())
            
            if USE_VIEWER and step % 10 == 0:
                if not viewer.is_running():
                    break
                viewer.sync()
                real_elapsed = time.time() - real_time_start
                if t_sim > real_elapsed:
                    time.sleep(min(0.01, t_sim - real_elapsed))
    
    finally:
        if viewer is not None:
            viewer.close()
    
    print("Simulation complete.")
    
    return (np.array(record_times),
            np.array(record_ee_pos),
            np.array(record_ee_x_err),
            np.array(record_tau_norm),
            np.array(record_external_force))


def compare_stiffnesses():
    """对比三种刚度的响应"""
    results = {}
    
    print("=" * 60)
    print("Stiffness Comparison: SOFT vs MEDIUM vs STIFF")
    print("=" * 60)
    
    # 三种刚度依次跑
    for label, K, D in [
        ("Soft (K=50)", PANDA_K_CART_SOFT, PANDA_D_CART_SOFT),
        ("Medium (K=200)", PANDA_K_CART_DEFAULT, PANDA_D_CART_DEFAULT),
        ("Stiff (K=1000)", PANDA_K_CART_STIFF, PANDA_D_CART_STIFF),
    ]:
        times, ee_pos, ee_err, tau_norm, ext_f = run_impedance_simulation(
            K, D, label=label
        )
        results[label] = {
            'times': times,
            'ee_pos': ee_pos,
            'ee_err': ee_err,
            'tau_norm': tau_norm,
            'ext_f': ext_f,
        }
    
    return results


def visualize_comparison(results):
    """可视化三种刚度对比"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    colors = {
        'Soft (K=50)': 'blue',
        'Medium (K=200)': 'green',
        'Stiff (K=1000)': 'red',
    }
    
    # ===== 左上：X 方向位移 =====
    ax = axes[0, 0]
    for label, data in results.items():
        x_displacement = data['ee_err'][:, 0] * 1000  # mm
        ax.plot(data['times'], x_displacement, 
                color=colors[label], linewidth=2, label=label)
    
    # 标注外力施加区间
    ax.axvspan(PUSH_START_TIME, PUSH_END_TIME, 
               alpha=0.2, color='yellow', label='External push')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('X displacement from target (mm)')
    ax.set_title('Cartesian Impedance: End-effector displacement\n'
                 'Under external push (15 N in +X)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linewidth=0.5)
    
    # ===== 右上：位移幅值（说明柔顺度）=====
    ax = axes[0, 1]
    for label, data in results.items():
        displacement = np.linalg.norm(data['ee_err'], axis=1) * 1000  # mm
        ax.plot(data['times'], displacement, 
                color=colors[label], linewidth=2, label=label)
    ax.axvspan(PUSH_START_TIME, PUSH_END_TIME, alpha=0.2, color='yellow')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Total displacement (mm)')
    ax.set_title('Total displacement magnitude\n'
                 'Higher K = Smaller displacement (less compliant)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # ===== 左下：最大位移对比柱状图 =====
    ax = axes[1, 0]
    labels = list(results.keys())
    max_disp = [np.max(np.linalg.norm(results[l]['ee_err'], axis=1)) * 1000 
                for l in labels]
    bar_colors = [colors[l] for l in labels]
    
    bars = ax.bar(labels, max_disp, color=bar_colors)
    ax.set_ylabel('Max displacement during push (mm)')
    ax.set_title('Stiffness vs Max Displacement\n'
                 '(Compliance characteristic)')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, max_disp):
        ax.text(bar.get_x() + bar.get_width()/2, val + 1,
                '{:.1f} mm'.format(val),
                ha='center', va='bottom', fontsize=10)
    
    # ===== 右下：力矩量级 =====
    ax = axes[1, 1]
    for label, data in results.items():
        ax.plot(data['times'], data['tau_norm'], 
                color=colors[label], linewidth=2, label=label)
    ax.axvspan(PUSH_START_TIME, PUSH_END_TIME, alpha=0.2, color='yellow')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('||τ|| (N·m)')
    ax.set_title('Control torque magnitude\n'
                 'Higher K = More torque needed to resist push')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('logs/day11_impedance_comparison.png', 
                dpi=120, bbox_inches='tight')
    print("\nFigure saved to: logs/day11_impedance_comparison.png")
    
    # 打印数值对比
    print("\n===== Summary =====")
    print("{:<20} {:<25} {:<25}".format(
        "Stiffness", "Max displacement (mm)", "Steady error (mm)"))
    print("-" * 70)
    for label in labels:
        max_d = np.max(np.linalg.norm(results[label]['ee_err'], axis=1)) * 1000
        # 稳态：取最后 1 秒
        last_sec_mask = results[label]['times'] > SIM_DURATION - 1.0
        steady = np.mean(np.linalg.norm(
            results[label]['ee_err'][last_sec_mask], axis=1)) * 1000
        print("{:<20} {:<25.2f} {:<25.3f}".format(label, max_d, steady))
    
    print("\nObservation: Stiffer K -> smaller displacement but larger torque")
    
    plt.show()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Day 11: Cartesian Impedance Control - Compliance Demo")
    print("=" * 60)
    
    results = compare_stiffnesses()
    
    visualize_comparison(results)
    
    print("\n" + "=" * 60)
    print("Day 11 complete!")
    print("=" * 60)