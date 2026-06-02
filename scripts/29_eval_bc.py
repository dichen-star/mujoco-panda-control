"""
Day 18 - 评估行为克隆
=================================
同一组 100 个目标上对比：专家(IK) / BC(模仿) / PPO(Day14, RL)。
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

from stable_baselines3 import PPO
from src.envs.panda_reach_env import PandaReachEnv
from src.kinematics.forward import panda_fk
from src.kinematics.inverse import panda_ik
from src.imitation.bc_policy import BCPolicy

BC_MODEL = "models/bc/bc_reach.pt"
PPO_MODEL = "models/best_model.zip"
N_EP = 100
SEED_OFF = 40000   # held-out 目标，三方共用


def expert_action(env):
    base = env.unwrapped
    q_now = base.data.qpos[:7]
    T_home = panda_fk(q_now)
    T = np.eye(4)
    T[:3, :3] = T_home[:3, :3]
    T[:3, 3] = base.target
    res = panda_ik(T, theta_init=q_now)
    q_target = res[0] if isinstance(res, tuple) else res
    return np.clip((q_target - q_now) / base.action_scale, -1.0, 1.0)


def eval_policy(env, action_fn, n_ep, seed_off):
    succ, finals = 0, []
    for ep in range(n_ep):
        obs, info = env.reset(seed=ep + seed_off)
        terminated = truncated = False
        last = 0.0
        while not (terminated or truncated):
            a = action_fn(obs, env)
            obs, r, terminated, truncated, info = env.step(a)
            last = info["distance"]
        succ += int(terminated)
        finals.append(last)
    return succ / n_ep * 100, float(np.mean(finals)) * 1000


def main():
    print("=" * 60)
    print(f"Day 18: BC evaluation ({N_EP} episodes, shared targets)")
    print("=" * 60)

    env = PandaReachEnv(max_steps=200)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    bc = BCPolicy(obs_dim=obs_dim, act_dim=act_dim)
    bc.load_state_dict(torch.load(BC_MODEL, map_location="cpu"))
    bc.eval()

    ppo = PPO.load(PPO_MODEL)

    policies = [
        ("Expert (IK)",     lambda obs, env: expert_action(env),          "#2a9d8f"),
        ("BC (imitation)",  lambda obs, env: bc.predict(obs),             "#e76f51"),
        ("PPO (RL, Day14)", lambda obs, env: ppo.predict(obs, deterministic=True)[0], "#264653"),
    ]

    results, colors = {}, {}
    for name, fn, color in policies:
        sr, fd = eval_policy(env, fn, N_EP, SEED_OFF)
        results[name] = (sr, fd)
        colors[name] = color
        print(f"  {name:<18} success {sr:5.1f}%   final {fd:5.0f} mm")
    env.close()

    # ---- 画图（两面板，保持简洁）----
    plt.rcParams.update({
        "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
    })
    names = list(results.keys())
    cols = [colors[n] for n in names]
    sr = [results[n][0] for n in names]
    fd = [results[n][1] for n in names]
    x = np.arange(len(names))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    b = ax.bar(x, sr, 0.6, color=cols, edgecolor="black", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Success rate (%)"); ax.set_ylim(0, 115)
    ax.set_title(f"Reach success ({N_EP} episodes)")
    ax.grid(True, axis="y", alpha=0.25, linestyle="--"); ax.set_axisbelow(True)
    for bar, v in zip(b, sr):
        ax.text(bar.get_x() + bar.get_width()/2, v + 2, f"{v:.0f}%",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax = axes[1]
    b = ax.bar(x, fd, 0.6, color=cols, edgecolor="black", linewidth=0.8)
    ax.axhline(50, color="green", ls="--", lw=1.3, label="50 mm threshold")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Mean final distance (mm)")
    ax.set_ylim(0, max(fd) * 1.30)        # 留头部空间，最高柱+标签不顶到边
    ax.set_title("Mean final distance")
    ax.legend(loc="upper left", frameon=True)   # 图例移到左上，避开最高柱
    ax.grid(True, axis="y", alpha=0.25, linestyle="--"); ax.set_axisbelow(True)
    for bar, v in zip(b, fd):
        ax.text(bar.get_x() + bar.get_width()/2, v + max(fd)*0.02, f"{v:.0f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    out = "logs/day18_bc_eval.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nFigure saved: {out}")
    print("=" * 60)


if __name__ == "__main__":
    main()