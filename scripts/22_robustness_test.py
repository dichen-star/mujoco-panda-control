"""
Day 16 - 鲁棒性对比:Day 14 标称模型 vs Day 16 DR 模型
========================================================
测试场景:把环境参数固定到特定偏移值,看每个模型在偏移下的表现

测试矩阵:
- 标称环境 (no perturbation)
- 加重 payload (0.3 kg)
- 重阻尼 (×1.3)
- 重力偏移 (×1.05)
- 全部叠加(worst case)

每个场景 × 每个模型 × 50 episodes
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

from stable_baselines3 import PPO
from src.envs.panda_reach_env_v3 import PandaReachEnvV3

# 模型路径
NOMINAL_MODEL = "models/best_model.zip"      # Day 14 的
DR_MODEL = "models/day16/best_model.zip"     # Day 16 的

# 测试场景
NOMINAL_DR = dict(payload=0.0, damping_scale=1.0, gravity_scale=1.0)

SCENARIOS = [
    ("Nominal",         dict(payload=0.0, damping_scale=1.0, gravity_scale=1.0)),
    ("Payload 0.3kg",   dict(payload=0.3, damping_scale=1.0, gravity_scale=1.0)),
    ("Damping ×1.3",    dict(payload=0.0, damping_scale=1.3, gravity_scale=1.0)),
    ("Gravity ×1.05",   dict(payload=0.0, damping_scale=1.0, gravity_scale=1.05)),
    ("Worst case",      dict(payload=0.4, damping_scale=1.25, gravity_scale=1.05)),
]

N_EPISODES = 50


def eval_in_scenario(model, dr_params, n_episodes, seed_base):
    """在指定 DR 偏移下评估模型"""
    # enable_dr=False 是关键:我们手动设置 DR,不让它随机
    env = PandaReachEnvV3(
        max_steps=200,
        reward_type="dense",
        enable_dr=False,
        ctrl_noise_std=0.0,
    )

    successes = 0
    finals = []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed_base + ep)
        # 手动设置 DR 参数(覆盖 reset 后的标称值)
        env.set_dr_params(dr_params)
        import mujoco
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
    return {
        "success_rate": successes / n_episodes * 100,
        "mean_final_mm": float(np.mean(finals)) * 1000,
    }


def main():
    print("=" * 70)
    print(f"Day 16: Robustness comparison")
    print(f"  Scenarios: {len(SCENARIOS)}")
    print(f"  Episodes per scenario: {N_EPISODES}")
    print("=" * 70)

    # 检查模型存在
    for name, path in [("Nominal (Day 14)", NOMINAL_MODEL), ("DR (Day 16)", DR_MODEL)]:
        if not os.path.exists(path):
            print(f"❌ Model not found: {path}")
            return
        print(f"  ✓ {name}: {path}")

    nominal_model = PPO.load(NOMINAL_MODEL)
    dr_model = PPO.load(DR_MODEL)

    # ---- 跑所有场景 ----
    results = {"Nominal (Day 14)": {}, "DR (Day 16)": {}}
    print(f"\n{'Scenario':<20} {'Nominal Success':<18} {'DR Success':<14}")
    print("-" * 56)
    for sc_name, dr_params in SCENARIOS:
        nom_stats = eval_in_scenario(nominal_model, dr_params, N_EPISODES, seed_base=10000)
        dr_stats = eval_in_scenario(dr_model, dr_params, N_EPISODES, seed_base=10000)
        results["Nominal (Day 14)"][sc_name] = nom_stats
        results["DR (Day 16)"][sc_name] = dr_stats
        print(f"{sc_name:<20} {nom_stats['success_rate']:>6.1f}% "
              f"({nom_stats['mean_final_mm']:>5.0f}mm)  "
              f"{dr_stats['success_rate']:>6.1f}% ({dr_stats['mean_final_mm']:>5.0f}mm)")

    # ---- 画图 ----
    plt.rcParams.update({
        "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    })

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    scenarios = [s[0] for s in SCENARIOS]
    x = np.arange(len(scenarios))
    w = 0.36

    # 左:成功率对比
    ax = axes[0]
    nom_success = [results["Nominal (Day 14)"][s]["success_rate"] for s in scenarios]
    dr_success = [results["DR (Day 16)"][s]["success_rate"] for s in scenarios]
    b1 = ax.bar(x - w/2, nom_success, w, color="#d62728", alpha=0.85,
                edgecolor="black", linewidth=0.8, label="Nominal (Day 14)")
    b2 = ax.bar(x + w/2, dr_success, w, color="#1f77b4", alpha=0.85,
                edgecolor="black", linewidth=0.8, label="DR (Day 16)")
    ax.set_xticks(x); ax.set_xticklabels(scenarios, rotation=15, ha="right")
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 115)
    ax.set_title(f"Robustness comparison ({N_EPISODES} ep/scenario)", pad=12)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, axis="y", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)

    for bars, vals in [(b1, nom_success), (b2, dr_success)]:
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v + 2,
                    f"{v:.0f}%", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")

    # 右:终末距离对比
    ax = axes[1]
    nom_dist = [results["Nominal (Day 14)"][s]["mean_final_mm"] for s in scenarios]
    dr_dist = [results["DR (Day 16)"][s]["mean_final_mm"] for s in scenarios]
    ax.bar(x - w/2, nom_dist, w, color="#d62728", alpha=0.85,
           edgecolor="black", linewidth=0.8, label="Nominal (Day 14)")
    ax.bar(x + w/2, dr_dist, w, color="#1f77b4", alpha=0.85,
           edgecolor="black", linewidth=0.8, label="DR (Day 16)")
    ax.set_xticks(x); ax.set_xticklabels(scenarios, rotation=15, ha="right")
    ax.set_ylabel("Mean final distance (mm)")
    ax.set_yscale("log")
    ax.set_title("Mean final distance (log scale)", pad=12)
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, axis="y", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)

    plt.tight_layout()
    out = "logs/day16_robustness.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nFigure saved: {out}")
    print("=" * 70)


if __name__ == "__main__":
    main()