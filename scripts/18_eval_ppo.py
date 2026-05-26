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

    # ---------- 画对比图(v2:可读性优化)----------
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    })

    fig = plt.figure(figsize=(15, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2], wspace=0.30)
    ax_left = fig.add_subplot(gs[0])
    ax_right = fig.add_subplot(gs[1])

    # ============ 左:三方对比柱状图(双 y 轴拆分,避免量级冲突)============
    labels = ["Random", "Greedy IK\n(oracle)", "PPO\n(learned)"]
    success = [rand_stats["success_rate"]*100,
               ik_stats["success_rate"]*100,
               ppo_stats["success_rate"]*100]
    finals = [rand_stats["mean_final_dist_mm"],
              ik_stats["mean_final_dist_mm"],
              ppo_stats["mean_final_dist_mm"]]
    colors = ["#9e9e9e", "#2ca02c", "#d62728"]

    x = np.arange(3)
    w = 0.36

    ax2 = ax_left.twinx()

    bars_succ = ax_left.bar(x - w/2, success, w,
                            color=colors, alpha=0.55,
                            edgecolor="black", linewidth=0.8,
                            label="Success rate")
    bars_dist = ax2.bar(x + w/2, finals, w,
                       color=colors, alpha=1.0,
                       edgecolor="black", linewidth=0.8,
                       label="Final distance")

    ax_left.set_xticks(x)
    ax_left.set_xticklabels(labels)
    ax_left.set_ylabel("Success rate (%)", color="#444")
    ax_left.set_ylim(0, 115)                                  # 给 label 留空间
    ax_left.tick_params(axis="y", labelcolor="#444")
    ax_left.set_title(f"Policy comparison (n={args.episodes} episodes)",
                      pad=12)
    ax_left.grid(True, axis="y", alpha=0.25, linestyle="--")
    ax_left.set_axisbelow(True)

    ax2.set_ylabel("Mean final distance (mm)", color="#222")
    ax2.set_yscale("log")
    ax2.set_ylim(1, max(finals) * 3)                          # log 上限留空间
    ax2.tick_params(axis="y", labelcolor="#222")

    # 柱顶标签(success 在上,distance 在数字旁,不再相互压住)
    for b, v in zip(bars_succ, success):
        ax_left.text(b.get_x() + b.get_width()/2, v + 2.5,
                     f"{v:.0f}%", ha="center", va="bottom",
                     fontsize=10, fontweight="bold", color="#444")
    for b, v in zip(bars_dist, finals):
        ax2.text(b.get_x() + b.get_width()/2, v * 1.15,
                 f"{v:.0f}mm", ha="center", va="bottom",
                 fontsize=10, fontweight="bold", color="#222")

    # 合并图例
    h1, l1 = ax_left.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax_left.legend(h1 + h2, l1 + l2, loc="upper left",
                   frameon=True, framealpha=0.9)

    # ============ 右:PPO 成功 episodes 收敛曲线 ============
    traces = ppo_stats["success_traces"][:30]
    for trace in traces:
        ax_right.plot(np.array(trace) * 1000,
                      color="#d62728", alpha=0.25, lw=1.2)

    # 画一条中位数粗线,显示"典型收敛行为"
    max_len = max(len(t) for t in traces)
    # 把每条 trace padding 到 max_len(后段用 nan 不参与中位数)
    padded = np.full((len(traces), max_len), np.nan)
    for i, t in enumerate(traces):
        padded[i, :len(t)] = np.array(t) * 1000
    median_curve = np.nanmedian(padded, axis=0)
    ax_right.plot(median_curve, color="#7f0000", lw=2.5,
                  label=f"Median (n={len(traces)})")

    ax_right.axhline(50, color="black", ls="--", lw=1.2,
                     label="5cm success threshold")

    ax_right.set_xlabel("Step within episode")
    ax_right.set_ylabel("Distance to target (mm)")
    ax_right.set_yscale("log")
    ax_right.set_title("PPO convergence: individual successes + median",
                       pad=12)
    ax_right.legend(loc="upper right", frameon=True, framealpha=0.9)
    ax_right.grid(True, alpha=0.25, linestyle="--")
    ax_right.set_axisbelow(True)

    plt.tight_layout()
    out = "logs/day14_ppo_eval.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nFigure saved: {out}")


if __name__ == "__main__":
    main()