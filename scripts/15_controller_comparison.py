"""
Day 12 - 四种控制器综合对比
======================================
同一圆轨迹 + 同一外力扰动下，PK:
  1. MuJoCo 默认位置伺服 (baseline)
  2. PD + 重力补偿        (Day 9)
  3. 计算力矩控制 CTC     (Day 10)
  4. 笛卡尔阻抗控制        (Day 11)

三幕剧:
  0-5s   自由跟踪
  5-7s   施加 15N 横向外力 (+Y)
  7-10s  撤力恢复
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import mujoco

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

from src.kinematics.forward import panda_fk
from src.kinematics.inverse import panda_ik, PANDA_HOME_THETA
from src.controllers.pd_gravity import (
    disable_actuators, apply_torque,
    compute_pd_gravity_torque, compute_gravity,
    PANDA_KP_DEFAULT, PANDA_KD_DEFAULT,
)
from src.controllers.ctc import (
    compute_ctc_torque, PANDA_CTC_KP, PANDA_CTC_KD,
)
from src.controllers.impedance import compute_impedance_torque

PANDA_XML = "assets/mujoco_menagerie/franka_emika_panda/panda.xml"

# ---------- 任务参数（与 Day 9-11 一致）----------
CIRCLE_CENTER = np.array([0.5, 0.0, 0.5])
CIRCLE_RADIUS = 0.15
CIRCLE_PERIOD = 10.0
SIM_DURATION = 10.0

# ---------- 扰动参数 ----------
DISTURB_FORCE = np.array([0.0, 15.0, 0.0])   # +Y 方向 15N
DISTURB_START = 5.0
DISTURB_END = 7.0

# ---------- 阻抗刚度（中等，可被推动才看得出柔顺）----------
IMP_K = np.array([300.0, 300.0, 300.0])
IMP_D = np.array([35.0, 35.0, 35.0])


def circle_pose(t):
    """解析圆轨迹：返回末端目标位置 x_d 和速度 xdot_d"""
    w = 2 * np.pi / CIRCLE_PERIOD
    a = w * t
    x = CIRCLE_CENTER + CIRCLE_RADIUS * np.array([np.cos(a), np.sin(a), 0])
    xdot = CIRCLE_RADIUS * w * np.array([-np.sin(a), np.cos(a), 0])
    return x, xdot


def build_joint_trajectory():
    """离线 IK 预生成关节轨迹 q_d(t)，并差分得到 qdot_d, qddot_d"""
    print("Building joint trajectory by offline IK (warm-start)...")
    T_home = panda_fk(PANDA_HOME_THETA)
    R_target = T_home[:3, :3]

    N = 400
    ts = np.linspace(0, SIM_DURATION, N)
    q_d = np.zeros((N, 7))
    theta = PANDA_HOME_THETA.copy()
    for i, t in enumerate(ts):
        x, _ = circle_pose(t)
        T = np.eye(4)
        T[:3, :3] = R_target
        T[:3, 3] = x
        result = panda_ik(T, theta_init=theta)              # warm-start
        theta = result[0] if isinstance(result, tuple) else result
        q_d[i] = theta

    dt = ts[1] - ts[0]
    qdot_d = np.gradient(q_d, dt, axis=0)
    qddot_d = np.gradient(qdot_d, dt, axis=0)
    return ts, q_d, qdot_d, qddot_d


def interp_row(ts, arr, t):
    """按时间线性插值一行（7维）"""
    return np.array([np.interp(t, ts, arr[:, j]) for j in range(arr.shape[1])])


def run_controller(ctrl_type, traj):
    """跑单个控制器，返回 (times, ee_err, max_disturb, recover_time)"""
    ts_traj, q_d_arr, qdot_d_arr, qddot_d_arr = traj

    # 关键：每个控制器都重新加载全新 model
    model = mujoco.MjModel.from_xml_path(PANDA_XML)
    data = mujoco.MjData(model)
    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")

    # 后三个控制器需要禁用 actuator；伺服保留内置 actuator
    if ctrl_type != "servo":
        disable_actuators(model)

    # 初始位形 = home
    data.qpos[:7] = PANDA_HOME_THETA
    data.qpos[7:] = 0.04
    mujoco.mj_forward(model, data)

    dt = model.opt.timestep
    n_steps = int(SIM_DURATION / dt)

    times, errs = [], []
    for k in range(n_steps):
        t = k * dt

        # 期望轨迹
        q_d = interp_row(ts_traj, q_d_arr, t)
        qdot_d = interp_row(ts_traj, qdot_d_arr, t)
        qddot_d = interp_row(ts_traj, qddot_d_arr, t)
        x_d, xdot_d = circle_pose(t)

        # 施加扰动
        if DISTURB_START <= t < DISTURB_END:
            data.xfrc_applied[hand_id, :3] = DISTURB_FORCE
        else:
            data.xfrc_applied[hand_id, :3] = 0.0

        # 按控制器类型计算力矩 / 控制量
        if ctrl_type == "servo":
            data.ctrl[:7] = q_d            # 内置位置伺服
        elif ctrl_type == "pd_g":
            tau = compute_pd_gravity_torque(
                model, data, q_d, qdot_d, PANDA_KP_DEFAULT, PANDA_KD_DEFAULT)
            apply_torque(data, tau)
        elif ctrl_type == "ctc":
            tau = compute_ctc_torque(
                model, data, q_d, qdot_d, qddot_d, PANDA_CTC_KP, PANDA_CTC_KD)
            apply_torque(data, tau)
        elif ctrl_type == "impedance":
            tau = compute_impedance_torque(
                model, data, x_d, xdot_d, IMP_K, IMP_D)
            apply_torque(data, tau)

        mujoco.mj_step(model, data)

        # 记录末端跟踪误差（mm）
        ee = data.xpos[hand_id].copy()
        err_mm = np.linalg.norm(ee - x_d) * 1000.0
        times.append(t)
        errs.append(err_mm)

    times = np.array(times)
    errs = np.array(errs)

    # 指标
    free_mask = (times >= 1.0) & (times < DISTURB_START)   # 跳过 1s 启动瞬态
    rms = np.sqrt(np.mean(errs[free_mask] ** 2))
    disturb_mask = (times >= DISTURB_START) & (times < DISTURB_END)
    max_disturb = errs[disturb_mask].max()

    # 恢复时间：7s 后误差首次回到 2mm 内
    recover_time = np.nan
    after = times >= DISTURB_END
    for tt, ee in zip(times[after], errs[after]):
        if ee < 2.0:
            recover_time = tt - DISTURB_END
            break

    return times, errs, rms, max_disturb, recover_time


def main():
    print("=" * 60)
    print("Day 12: Four-Controller Comparison")
    print("=" * 60)

    traj = build_joint_trajectory()

    configs = [
        ("servo",     "Position Servo (baseline)", "#888888"),
        ("pd_g",      "PD + Gravity",              "#1f77b4"),
        ("ctc",       "Computed Torque (CTC)",     "#2ca02c"),
        ("impedance", "Cartesian Impedance",       "#d62728"),
    ]

    results = {}
    print("\n%-26s %-12s %-14s %-12s" %
          ("Controller", "RMS(mm)", "MaxDisturb(mm)", "Recover(s)"))
    print("-" * 66)
    for ctype, label, color in configs:
        times, errs, rms, maxd, rec = run_controller(ctype, traj)
        results[ctype] = (times, errs, label, color, rms, maxd, rec)
        rec_str = "%.2f" % rec if not np.isnan(rec) else "not back"
        print("%-26s %-12.3f %-14.1f %-12s" % (label, rms, maxd, rec_str))

    # ---------- 画图 ----------
    fig, axes = plt.subplots(2, 1, figsize=(11, 9))

    # 上图：误差时序（全部叠加）
    ax = axes[0]
    for ctype, _, _ in configs:
        times, errs, label, color, *_ = results[ctype]
        ax.plot(times, errs, label=label, color=color, lw=1.8)
    ax.axvspan(DISTURB_START, DISTURB_END, color="orange", alpha=0.15,
               label="Disturbance 15N")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Tracking error (mm)")
    ax.set_title("Tracking Error: Free → Disturbance → Recovery")
    ax.set_yscale("log")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # 下图：三指标柱状
    ax = axes[1]
    labels = [results[c][2] for c, _, _ in configs]
    rms_vals = [results[c][4] for c, _, _ in configs]
    maxd_vals = [results[c][5] for c, _, _ in configs]
    colors = [results[c][3] for c, _, _ in configs]
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w / 2, rms_vals, w, label="RMS error (free)", color=colors, alpha=0.6)
    ax.bar(x + w / 2, maxd_vals, w, label="Max disturb dev", color=colors, alpha=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace(" ", "\n") for l in labels], fontsize=8)
    ax.set_ylabel("Error (mm)")
    ax.set_yscale("log")
    ax.set_title("RMS Tracking vs Max Disturbance Deviation")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = "logs/day12_controller_comparison.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print("\nFigure saved to:", out)
    print("=" * 60)
    print("Day 12 complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()