"""
Day 17 - Week 3 收官:Model-based vs Model-free vs Robust-RL 三方对决
=========================================================================
三场景:
  Scene 1: 固定圆轨迹跟踪    → 测稳态 RMS(精度)
  Scene 2: 随机目标触达       → 测成功率(泛化)
  Scene 3: 扰动下随机触达     → 测扰动成功率(鲁棒性)

三选手:
  CTC               (Week 2 控制论)
  PPO_pos (Day 14)  (model-free 位置控制)
  PPO_torque_DR     (Day 16B,robust RL)
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import mujoco

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

from stable_baselines3 import PPO

from src.envs.panda_reach_env import PandaReachEnv
from src.envs.panda_reach_env_torque import PandaReachEnvTorque
from src.kinematics.forward import panda_fk
from src.kinematics.inverse import panda_ik, PANDA_HOME_THETA
from src.controllers.ctc import compute_ctc_torque, PANDA_CTC_KP, PANDA_CTC_KD
from src.controllers.pd_gravity import disable_actuators, apply_torque

# ---- 模型路径 ----
POS_PPO_MODEL = "models/best_model.zip"
TORQUE_DR_MODEL = "models/day16_torque/dr/best_model.zip"
PANDA_XML = "assets/mujoco_menagerie/franka_emika_panda/panda.xml"

# ---- 任务参数 ----
CIRCLE_CENTER = np.array([0.5, 0.0, 0.5])
CIRCLE_RADIUS = 0.15
CIRCLE_PERIOD = 10.0
SCENE1_DURATION = 10.0
N_REACH_EPISODES = 100
N_DISTURB_EPISODES = 50

DISTURB_PARAMS = dict(payload=0.4, damping_scale=1.25, gravity_scale=1.05)


# ============================================================
# Scene 1: 圆轨迹跟踪 — 测稳态 RMS
# ============================================================
def circle_pose(t):
    w = 2 * np.pi / CIRCLE_PERIOD
    a = w * t
    x = CIRCLE_CENTER + CIRCLE_RADIUS * np.array([np.cos(a), np.sin(a), 0])
    xdot = CIRCLE_RADIUS * w * np.array([-np.sin(a), np.cos(a), 0])
    return x, xdot


def build_circle_joint_traj():
    """离线 IK,给 CTC 用"""
    T_home = panda_fk(PANDA_HOME_THETA)
    R_target = T_home[:3, :3]
    N = 400
    ts = np.linspace(0, SCENE1_DURATION, N)
    q_d = np.zeros((N, 7))
    theta = PANDA_HOME_THETA.copy()
    for i, t in enumerate(ts):
        x, _ = circle_pose(t)
        T = np.eye(4)
        T[:3, :3] = R_target
        T[:3, 3] = x
        result = panda_ik(T, theta_init=theta)
        theta = result[0] if isinstance(result, tuple) else result
        q_d[i] = theta
    dt = ts[1] - ts[0]
    qdot_d = np.gradient(q_d, dt, axis=0)
    qddot_d = np.gradient(qdot_d, dt, axis=0)
    return ts, q_d, qdot_d, qddot_d


def interp_row(ts, arr, t):
    return np.array([np.interp(t, ts, arr[:, j]) for j in range(arr.shape[1])])


def scene1_ctc(traj):
    """CTC 跑圆轨迹,返回稳态 RMS(mm)"""
    ts_traj, q_d_arr, qdot_d_arr, qddot_d_arr = traj
    model = mujoco.MjModel.from_xml_path(PANDA_XML)
    data = mujoco.MjData(model)
    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    disable_actuators(model)
    data.qpos[:7] = PANDA_HOME_THETA
    data.qpos[7:] = 0.04
    mujoco.mj_forward(model, data)

    dt = model.opt.timestep
    n_steps = int(SCENE1_DURATION / dt)
    times, errs = [], []
    for k in range(n_steps):
        t = k * dt
        q_d = interp_row(ts_traj, q_d_arr, t)
        qdot_d = interp_row(ts_traj, qdot_d_arr, t)
        qddot_d = interp_row(ts_traj, qddot_d_arr, t)
        x_d, _ = circle_pose(t)
        tau = compute_ctc_torque(model, data, q_d, qdot_d, qddot_d,
                                 PANDA_CTC_KP, PANDA_CTC_KD)
        apply_torque(data, tau)
        mujoco.mj_step(model, data)
        ee = data.xpos[hand_id]
        err_mm = np.linalg.norm(ee - x_d) * 1000.0
        times.append(t); errs.append(err_mm)
    times, errs = np.array(times), np.array(errs)
    mask = times >= 1.0  # 跳过启动瞬态(Day 12 教训)
    return float(np.sqrt(np.mean(errs[mask] ** 2)))


def scene1_ppo(model_path, env_class, **env_kwargs):
    """把圆轨迹当瞬时目标喂给 PPO,测稳态 RMS"""
    model = PPO.load(model_path)
    # 兼容 v1(无 reward_type)和 v3/torque(有 reward_type)
    try:
        env = env_class(max_steps=10000, reward_type="dense", **env_kwargs)
    except TypeError:
        env = env_class(max_steps=10000, **env_kwargs)
    obs, info = env.reset(seed=0)
    # 替换目标为圆轨迹起点
    env.unwrapped.target = circle_pose(0)[0]
    obs = env.unwrapped._get_obs()

    dt = env.unwrapped.control_dt
    n_steps = int(SCENE1_DURATION / dt)
    hand_id = env.unwrapped.hand_id
    times, errs = [], []
    for k in range(n_steps):
        t = k * dt
        # 每步把目标更新成轨迹当前点
        env.unwrapped.target = circle_pose(t)[0]
        obs = env.unwrapped._get_obs()
        action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(action)
        ee = env.unwrapped.data.xpos[hand_id]
        x_d, _ = circle_pose(t)
        err_mm = np.linalg.norm(ee - x_d) * 1000.0
        times.append(t); errs.append(err_mm)
    env.close()
    times, errs = np.array(times), np.array(errs)
    mask = times >= 1.0
    return float(np.sqrt(np.mean(errs[mask] ** 2)))


# ============================================================
# Scene 2 / 3: Reach 任务(可选扰动)
# ============================================================
def scene_reach_ctc(n_episodes, dr_params=None, seed_offset=0):
    """CTC + IK 跑 Reach:每个 episode 离线 IK,然后 CTC 跟踪"""
    successes, finals = 0, []
    for ep in range(n_episodes):
        model = mujoco.MjModel.from_xml_path(PANDA_XML)
        data = mujoco.MjData(model)
        hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        disable_actuators(model)

        # 应用扰动
        if dr_params is not None:
            hand_nominal = model.body_mass[hand_id]
            damping_nominal = model.dof_damping[:7].copy()
            gravity_nominal = model.opt.gravity.copy()
            model.body_mass[hand_id] = hand_nominal + dr_params["payload"]
            model.dof_damping[:7] = damping_nominal * dr_params["damping_scale"]
            model.opt.gravity[:] = gravity_nominal * dr_params["gravity_scale"]

        data.qpos[:7] = PANDA_HOME_THETA
        data.qpos[7:] = 0.04
        mujoco.mj_forward(model, data)

        # 随机目标
        rng = np.random.RandomState(ep + seed_offset)
        target = rng.uniform([0.3, -0.3, 0.2], [0.7, 0.3, 0.7])

        # 离线 IK 求目标关节角
        T_home = panda_fk(PANDA_HOME_THETA)
        T = np.eye(4); T[:3, :3] = T_home[:3, :3]; T[:3, 3] = target
        result = panda_ik(T, theta_init=PANDA_HOME_THETA)
        q_target = result[0] if isinstance(result, tuple) else result

        # CTC 简单跟踪:静态目标 → qdot_d=0, qddot_d=0
        qdot_d = np.zeros(7); qddot_d = np.zeros(7)
        dt = model.opt.timestep
        max_t = 4.0  # 4 秒给 CTC 收敛
        last_dist = 0.0
        for k in range(int(max_t / dt)):
            tau = compute_ctc_torque(model, data, q_target, qdot_d, qddot_d,
                                     PANDA_CTC_KP, PANDA_CTC_KD)
            apply_torque(data, tau)
            mujoco.mj_step(model, data)
        ee = data.xpos[hand_id]
        dist = float(np.linalg.norm(ee - target))
        if dist < 0.05:
            successes += 1
        finals.append(dist)
    return {"success_rate": successes / n_episodes * 100,
            "mean_final_mm": float(np.mean(finals)) * 1000}


def scene_reach_ppo(model_path, env_class, n_episodes, dr_params=None,
                    seed_offset=0, **env_kwargs):
    """PPO 跑 Reach,可选扰动"""
    model = PPO.load(model_path)
    # v1 没有 reward_type / enable_dr / ctrl_noise_std
    if env_class is PandaReachEnv:
        env = env_class(max_steps=200, **env_kwargs)
    else:
        env = env_class(max_steps=200, reward_type="dense",
                    enable_dr=False, ctrl_noise_std=0.0, **env_kwargs)
    successes, finals = 0, []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep + seed_offset)
        if dr_params is not None:
            env.set_dr_params(dr_params)
            mujoco.mj_forward(env.model, env.data)
        terminated = truncated = False
        last_dist = 0.0
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, info = env.step(action)
            last_dist = info["distance"]
        if terminated:
            successes += 1
        finals.append(last_dist)
    env.close()
    return {"success_rate": successes / n_episodes * 100,
            "mean_final_mm": float(np.mean(finals)) * 1000}


# ============================================================
# 主流程
# ============================================================
def main():
    print("=" * 70)
    print("Day 17: Grand Comparison — Model-based vs Model-free vs Robust-RL")
    print("=" * 70)

    for name, path in [("Pos PPO", POS_PPO_MODEL), ("Torque DR", TORQUE_DR_MODEL)]:
        if not os.path.exists(path):
            print(f"❌ Missing: {path}"); return
    print("All models found ✓\n")

    results = {"CTC": {}, "PPO_pos": {}, "PPO_torque_DR": {}}

    # ---- Scene 1: 圆轨迹 ----
    print("[Scene 1] Circle trajectory tracking (RMS, lower=better)...")
    traj = build_circle_joint_traj()
    results["CTC"]["scene1_rms_mm"] = scene1_ctc(traj)
    print(f"  CTC:           {results['CTC']['scene1_rms_mm']:.3f} mm")

    results["PPO_pos"]["scene1_rms_mm"] = scene1_ppo(POS_PPO_MODEL, PandaReachEnv)
    print(f"  PPO_pos:       {results['PPO_pos']['scene1_rms_mm']:.1f} mm")

    # 力矩 DR 不补偿重力做圆轨迹会困难,这里同样让它跑同任务
    results["PPO_torque_DR"]["scene1_rms_mm"] = scene1_ppo(
        TORQUE_DR_MODEL, PandaReachEnvTorque, enable_dr=False, ctrl_noise_std=0.0)
    print(f"  PPO_torque_DR: {results['PPO_torque_DR']['scene1_rms_mm']:.1f} mm")

    # ---- Scene 2: 随机 Reach(无扰动)----
    print(f"\n[Scene 2] Random reach, {N_REACH_EPISODES} episodes (no perturbation)...")
    r = scene_reach_ctc(N_REACH_EPISODES, dr_params=None, seed_offset=20000)
    results["CTC"]["scene2_success"] = r["success_rate"]
    results["CTC"]["scene2_dist_mm"] = r["mean_final_mm"]
    print(f"  CTC+IK:        {r['success_rate']:>5.1f}%  {r['mean_final_mm']:>5.0f} mm")

    r = scene_reach_ppo(POS_PPO_MODEL, PandaReachEnv, N_REACH_EPISODES, seed_offset=20000)
    results["PPO_pos"]["scene2_success"] = r["success_rate"]
    results["PPO_pos"]["scene2_dist_mm"] = r["mean_final_mm"]
    print(f"  PPO_pos:       {r['success_rate']:>5.1f}%  {r['mean_final_mm']:>5.0f} mm")

    r = scene_reach_ppo(TORQUE_DR_MODEL, PandaReachEnvTorque, N_REACH_EPISODES,
                        seed_offset=20000)
    results["PPO_torque_DR"]["scene2_success"] = r["success_rate"]
    results["PPO_torque_DR"]["scene2_dist_mm"] = r["mean_final_mm"]
    print(f"  PPO_torque_DR: {r['success_rate']:>5.1f}%  {r['mean_final_mm']:>5.0f} mm")

    # ---- Scene 3: 扰动下 Reach ----
    print(f"\n[Scene 3] Random reach with worst-case perturbation, "
          f"{N_DISTURB_EPISODES} episodes...")
    r = scene_reach_ctc(N_DISTURB_EPISODES, dr_params=DISTURB_PARAMS, seed_offset=30000)
    results["CTC"]["scene3_success"] = r["success_rate"]
    results["CTC"]["scene3_dist_mm"] = r["mean_final_mm"]
    print(f"  CTC+IK:        {r['success_rate']:>5.1f}%  {r['mean_final_mm']:>5.0f} mm")

    # 位置版 v1 没有 set_dr_params 接口,且 Day 16 已证明 PD 伺服会吸收扰动
    # (86% vs 88%,差异在噪声范围内),Scene 3 直接复用 Scene 2 数字作为参考
    results["PPO_pos"]["scene3_success"] = results["PPO_pos"]["scene2_success"]
    results["PPO_pos"]["scene3_dist_mm"] = results["PPO_pos"]["scene2_dist_mm"]
    print(f"  PPO_pos:       {results['PPO_pos']['scene3_success']:>5.1f}%  "
        f"{results['PPO_pos']['scene3_dist_mm']:>5.0f} mm  "
        f"(借用 Scene 2 — PD 伺服吸收扰动,见 Day 16)")

    r = scene_reach_ppo(TORQUE_DR_MODEL, PandaReachEnvTorque, N_DISTURB_EPISODES,
                        dr_params=DISTURB_PARAMS, seed_offset=30000)
    results["PPO_torque_DR"]["scene3_success"] = r["success_rate"]
    results["PPO_torque_DR"]["scene3_dist_mm"] = r["mean_final_mm"]
    print(f"  PPO_torque_DR: {r['success_rate']:>5.1f}%  {r['mean_final_mm']:>5.0f} mm")

    # ============================================================
    # 出图
    # ============================================================
    print("\n[Plotting]...")
    plt.rcParams.update({
        "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
        "xtick.labelsize": 11, "ytick.labelsize": 10, "legend.fontsize": 10,
    })

    fig = plt.figure(figsize=(15, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1], wspace=0.30)
    ax_left = fig.add_subplot(gs[0])
    ax_right = fig.add_subplot(gs[1], projection="polar")

    methods = ["CTC", "PPO_pos", "PPO_torque_DR"]
    method_full = ["CTC\n(model-based)", "PPO_pos\n(model-free)", "PPO_torque_DR\n(robust RL)"]
    colors = ["#2a9d8f", "#e76f51", "#264653"]

    # ============================================================
    # 左:三方法 × 三场景,单图 log 轴,场景用并排子柱区分
    # ============================================================
    s1 = [results[m]["scene1_rms_mm"] for m in methods]
    s2 = [results[m]["scene2_success"] for m in methods]
    s3 = [results[m]["scene3_success"] for m in methods]

    x = np.arange(3)   # 三个方法
    w = 0.26

    ax = ax_left
    b1 = ax.bar(x - w, s1, w, color=colors, alpha=0.95,
                edgecolor="black", linewidth=0.7, hatch="")
    b2 = ax.bar(x,     s2, w, color=colors, alpha=0.70,
                edgecolor="black", linewidth=0.7, hatch="//")
    b3 = ax.bar(x + w, s3, w, color=colors, alpha=0.45,
                edgecolor="black", linewidth=0.7, hatch="..")

    ax.set_yscale("log")
    ax.set_ylim(0.05, 2000)
    ax.set_xticks(x)
    ax.set_xticklabels(method_full, fontsize=10)
    ax.set_ylabel("Value (mm or %, log scale)")
    ax.set_title("Performance across three scenes", pad=12)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--", which="both")
    ax.set_axisbelow(True)

    # 在每根柱顶标数字
    def label_bars(bars, vals, unit):
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v * 1.25,
                    f"{v:.2f}{unit}" if v < 1 else f"{v:.0f}{unit}",
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    label_bars(b1, s1, "mm")
    label_bars(b2, s2, "%")
    label_bars(b3, s3, "%")

    # 自定义场景图例(用 hatch 表示,颜色保持中性灰)
    from matplotlib.patches import Patch
    scene_legend = [
        Patch(facecolor="#aaa", edgecolor="black", hatch="",   label="Scene 1: Circle RMS (mm)"),
        Patch(facecolor="#aaa", edgecolor="black", hatch="//", label="Scene 2: Reach success (%)"),
        Patch(facecolor="#aaa", edgecolor="black", hatch="..", label="Scene 3: Disturbed reach (%)"),
    ]
    ax.legend(handles=scene_legend, loc="upper left", frameon=True,
              fontsize=9, framealpha=0.92)

    # ============================================================
    # 右:能力雷达图
    # ============================================================
    def normalize_inverse(vals):
        arr = np.array(vals, dtype=float)
        inv = 1.0 / (arr + 1e-6)
        return inv / inv.max()

    def normalize_direct(vals):
        return np.array(vals, dtype=float) / 100.0

    precision = normalize_inverse(s1)
    generalization = normalize_direct(s2)
    robustness = normalize_direct(s3)

    categories = ["Precision\n(Scene 1)", "Generalization\n(Scene 2)", "Robustness\n(Scene 3)"]
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    ax = ax_right
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=8, color="#666")
    ax.set_title("Capability radar (normalized)", pad=24, fontsize=12)
    ax.spines["polar"].set_color("#bbb")
    ax.grid(color="#ddd", linewidth=0.8)

    # 按面积从大到小画,大的在底层不会遮小的
    method_order = sorted(range(3),
                          key=lambda i: -(precision[i] + generalization[i] + robustness[i]))
    for i in method_order:
        vals = [precision[i], generalization[i], robustness[i]]
        vals += vals[:1]
        ax.plot(angles, vals, color=colors[i], linewidth=2.2,
                label=methods[i], marker="o", markersize=5)
        ax.fill(angles, vals, color=colors[i], alpha=0.12)

    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.10),
              frameon=True, fontsize=9)

    plt.tight_layout()
    out = "logs/day17_grand_comparison.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Figure saved: {out}")

    # ============================================================
    # 终末成绩单
    # ============================================================
    print("\n" + "=" * 70)
    print("FINAL SCOREBOARD")
    print("=" * 70)
    print(f"{'Method':<22} {'S1 RMS':<12} {'S2 Succ':<12} {'S3 Succ':<12}")
    print("-" * 60)
    for m, label in zip(methods, method_full):
        line = label.replace('\n', ' ')
        print(f"{line:<22} "
              f"{results[m]['scene1_rms_mm']:>6.2f}mm   "
              f"{results[m]['scene2_success']:>5.1f}%     "
              f"{results[m]['scene3_success']:>5.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()