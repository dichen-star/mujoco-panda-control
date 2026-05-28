"""
Day 16B - 力矩控制版鲁棒性对比
=================================
力矩标称模型 vs 力矩 DR 模型,5 个偏移场景 × 50 episodes。
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import mujoco

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

from stable_baselines3 import PPO
from src.envs.panda_reach_env_torque import PandaReachEnvTorque

NOMINAL_MODEL = "models/day16_torque/nominal/best_model.zip"
DR_MODEL = "models/day16_torque/dr/best_model.zip"

# (内部名, 显示名, DR 偏移参数)
SCENARIOS = [
    ("Nominal",       "No perturbation",            dict(payload=0.0, damping_scale=1.0,  gravity_scale=1.0)),
    ("Payload",       "Payload +0.3kg",             dict(payload=0.3, damping_scale=1.0,  gravity_scale=1.0)),
    ("Damping",       "Damping x1.3",               dict(payload=0.0, damping_scale=1.3,  gravity_scale=1.0)),
    ("Gravity",       "Gravity x1.05",              dict(payload=0.0, damping_scale=1.0,  gravity_scale=1.05)),
    ("Worst",         "All combined\n(worst case)", dict(payload=0.4, damping_scale=1.25, gravity_scale=1.05)),
]
N_EPISODES = 50


def eval_in_scenario(model, dr_params, n_episodes, seed_base):
    env = PandaReachEnvTorque(
        max_steps=200, reward_type="dense",
        enable_dr=False, ctrl_noise_std=0.0,
    )
    successes, finals = 0, []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed_base + ep)
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


def main():
    print("=" * 70)
    print("Day 16B: Torque-control robustness comparison")
    print("=" * 70)
    for name, path in [("Nominal torque", NOMINAL_MODEL), ("DR torque", DR_MODEL)]:
        if not os.path.exists(path):
            print(f"❌ Model not found: {path}\n   先跑 scripts/24_train_dr_torque.py")
            return
        print(f"  ✓ {name}: {path}")

    nominal_model = PPO.load(NOMINAL_MODEL)
    dr_model = PPO.load(DR_MODEL)

    results = {"Nominal": {}, "DR": {}}
    print(f"\n{'Scenario':<16} {'Nominal':<20} {'DR':<20}")
    print("-" * 56)
    for key, disp, dr_params in SCENARIOS:
        nom = eval_in_scenario(nominal_model, dr_params, N_EPISODES, 10000)
        dr = eval_in_scenario(dr_model, dr_params, N_EPISODES, 10000)
        results["Nominal"][key] = nom
        results["DR"][key] = dr
        print(f"{disp.splitlines()[0]:<16} {nom['success_rate']:>5.1f}% ({nom['mean_final_mm']:>5.0f}mm)   "
              f"{dr['success_rate']:>5.1f}% ({dr['mean_final_mm']:>5.0f}mm)")

    # ---- 画图 ----
    plt.rcParams.update({
        "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
        "xtick.labelsize": 9.5, "ytick.labelsize": 10, "legend.fontsize": 10,
    })
    C_NOM, C_DR = "#d62728", "#1f77b4"
    keys = [s[0] for s in SCENARIOS]
    disp = [s[1] for s in SCENARIOS]
    x = np.arange(len(keys))
    w = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # 左:成功率(缩放到 0-40,差异才看得清)
    ax = axes[0]
    nom_s = [results["Nominal"][k]["success_rate"] for k in keys]
    dr_s = [results["DR"][k]["success_rate"] for k in keys]
    b1 = ax.bar(x - w/2, nom_s, w, color=C_NOM, alpha=0.85,
                edgecolor="black", linewidth=0.8, label="Trained w/o DR (nominal)")
    b2 = ax.bar(x + w/2, dr_s, w, color=C_DR, alpha=0.85,
                edgecolor="black", linewidth=0.8, label="Trained w/ DR")
    ax.set_xticks(x); ax.set_xticklabels(disp)
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 40)
    ax.set_title(f"Torque control: robustness under parameter shift\n({N_EPISODES} episodes / scenario)", pad=12)
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, axis="y", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)
    for bars, vals in [(b1, nom_s), (b2, dr_s)]:
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v + 0.6, f"{v:.0f}%",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    # 右:终末距离(线性轴 + 成功线)
    ax = axes[1]
    nom_d = [results["Nominal"][k]["mean_final_mm"] for k in keys]
    dr_d = [results["DR"][k]["mean_final_mm"] for k in keys]
    b1 = ax.bar(x - w/2, nom_d, w, color=C_NOM, alpha=0.85,
                edgecolor="black", linewidth=0.8, label="Trained w/o DR (nominal)")
    b2 = ax.bar(x + w/2, dr_d, w, color=C_DR, alpha=0.85,
                edgecolor="black", linewidth=0.8, label="Trained w/ DR")
    ax.axhline(50, color="green", ls="--", lw=1.3, label="50 mm success threshold")
    ax.set_xticks(x); ax.set_xticklabels(disp)
    ax.set_ylabel("Mean final distance (mm)")
    ax.set_ylim(0, 360)
    ax.set_title("Mean final distance to target (linear scale)", pad=12)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, axis="y", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)
    for bars, vals in [(b1, nom_d), (b2, dr_d)]:
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v + 4, f"{v:.0f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.text(0.5, 0.005,
             "DR training randomized 4 dims (payload/damping/gravity/control noise); "
             "test fixes the 3 reproducible physical dims. Control noise is training-only.",
             ha="center", fontsize=8.5, color="#555", style="italic")

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    out = "logs/day16_torque_robustness.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nFigure saved: {out}")
    print("=" * 70)


if __name__ == "__main__":
    main()