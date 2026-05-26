"""
Day 14 - PPO 模型详细评估
=================================
1. 100 episodes 跑成功率、平均距离、平均步数
2. 与 Day 13 的 random / greedy IK baseline 横向对比
3. 输出 logs/day14_ppo_eval.png(三方对比柱状图 + 成功 episode 距离曲线)
"""
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

from stable_baselines3 import PPO
from src.envs.panda_reach_env import PandaReachEnv
from src.kinematics.forward import panda_fk
from src.kinematics.inverse import panda_ik


def eval_policy(env, policy_fn, n_episodes=100, seed_offset=0):
    """通用评估函数:policy_fn(obs, env) -> action"""
    successes, returns, finals, steps_list = 0, [], [], []
    dist_traces = []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep + seed_offset)
        ep_return, last_dist, steps = 0.0, 0.0, 0
        trace = []
        terminated = truncated = False
        while not (terminated or truncated):
            action = policy_fn(obs, env)
            obs, r, terminated, truncated, info = env.step(action)
            ep_return += r
            last_dist = info["distance"]
            trace.append(last_dist)
            steps += 1
        if terminated:
            successes += 1
            dist_traces.append(trace)
        returns.append(ep_return)
        finals.append(last_dist)
        steps_list.append(steps)
    return {
        "success_rate": successes / n_episodes,
        "mean_return": float(np.mean(returns)),
        "mean_final_dist_mm": float(np.mean(finals)) * 1000,
        "mean_steps": float(np.mean(steps_list)),
        "success_traces": dist_traces,
    }


def random_policy(obs, env):
    return env.action_space.sample()


def greedy_ik_policy(obs, env):
    """每步用 IK 求目标关节角,朝它走一步"""
    base = env.unwrapped
    target_pos = base.target
    T_home = panda_fk(base.data.qpos[:7])
    T_target = np.eye(4)
    T_target[:3, :3] = T_home[:3, :3]
    T_target[:3, 3] = target_pos
    result = panda_ik(T_target, theta_init=base.data.qpos[:7])
    q_target = result[0] if isinstance(result, tuple) else result
    delta = q_target - base.data.qpos[:7]
    return np.clip(delta / base.action_scale, -1.0, 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/best_model.zip")
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()

    print("=" * 60)
    print(f"Day 14: PPO evaluation ({args.episodes} episodes per policy)")
    print("=" * 60)

    env = PandaReachEnv(max_steps=200)

    # 1) Random baseline
    print("\n[1/3] Random policy ...")
    rand_stats = eval_policy(env, random_policy, n_episodes=args.episodes, seed_offset=10000)
    print(f"    success {rand_stats['success_rate']*100:.1f}%  "
          f"return {rand_stats['mean_return']:.2f}  "
          f"final {rand_stats['mean_final_dist_mm']:.1f}mm")

    # 2) Greedy IK baseline (oracle)
    print("\n[2/3] Greedy IK policy (oracle baseline) ...")
    ik_stats = eval_policy(env, greedy_ik_policy, n_episodes=args.episodes, seed_offset=20000)
    print(f"    success {ik_stats['success_rate']*100:.1f}%  "
          f"return {ik_stats['mean_return']:.2f}  "
          f"final {ik_stats['mean_final_dist_mm']:.1f}mm")

    # 3) PPO
    print(f"\n[3/3] PPO policy ({args.model}) ...")
    model = PPO.load(args.model)

    def ppo_policy(obs, env):
        action, _ = model.predict(obs, deterministic=True)
        return action

    ppo_stats = eval_policy(env, ppo_policy, n_episodes=args.episodes, seed_offset=30000)
    print(f"    success {ppo_stats['success_rate']*100:.1f}%  "
          f"return {ppo_stats['mean_return']:.2f}  "
          f"final {ppo_stats['mean_final_dist_mm']:.1f}mm  "
          f"avg_steps {ppo_stats['mean_steps']:.1f}")

    env.close()

    # ---------- 画对比图 ----------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 左:三方指标对比
    ax = axes[0]
    labels = ["Random", "Greedy IK\n(oracle)", "PPO\n(learned)"]
    success = [rand_stats["success_rate"]*100,
               ik_stats["success_rate"]*100,
               ppo_stats["success_rate"]*100]
    finals = [rand_stats["mean_final_dist_mm"],
              ik_stats["mean_final_dist_mm"],
              ppo_stats["mean_final_dist_mm"]]
    colors = ["#888888", "#2ca02c", "#d62728"]
    x = np.arange(3)
    w = 0.35
    ax2 = ax.twinx()
    bars1 = ax.bar(x - w/2, success, w, color=colors, alpha=0.6, label="Success rate (%)")
    bars2 = ax2.bar(x + w/2, finals, w, color=colors, alpha=1.0, label="Final dist (mm)")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Success rate (%)", color="gray")
    ax2.set_ylabel("Mean final distance (mm)")
    ax2.set_yscale("log")
    ax.set_title(f"Policy comparison ({args.episodes} episodes)")
    ax.set_ylim(0, 105)
    for b, v in zip(bars1, success):
        ax.text(b.get_x()+b.get_width()/2, v+2, f"{v:.0f}%", ha="center", fontsize=9)
    for b, v in zip(bars2, finals):
        ax2.text(b.get_x()+b.get_width()/2, v*1.1, f"{v:.0f}", ha="center", fontsize=9)

    # 右:PPO 成功 episode 的距离曲线(展示收敛行为)
    ax = axes[1]
    traces = ppo_stats["success_traces"][:30]
    for trace in traces:
        ax.plot(np.array(trace)*1000, color="#d62728", alpha=0.25, lw=1)
    ax.axhline(50, color="black", ls="--", lw=1, label="5cm success threshold")
    ax.set_xlabel("Step")
    ax.set_ylabel("Distance to target (mm)")
    ax.set_title(f"PPO success episodes: distance vs step (n={len(traces)})")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = "logs/day14_ppo_eval.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nFigure saved: {out}")
    print("=" * 60)
    print("Day 14 done.")
    print("=" * 60)


if __name__ == "__main__":
    main()