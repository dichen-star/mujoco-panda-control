"""Day 21 - 多模态任务 BC vs DP 对比 + 轨迹可视化"""
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')
from src.envs.obstacle2d_env import (Obstacle2DEnv, expert_action,
                                      GOAL, OBSTACLE, OBSTACLE_R, GOAL_R, START_JITTER)
from src.imitation.bc_policy import BCPolicy
from src.imitation.diffusion_policy import DiffusionPolicy

BC_MODEL = "models/bc/bc_obstacle.pt"
DP_MODEL = "models/bc/dp_obstacle.pt"
K = 60
T = 50


def run_expert(env, side, seed):
    obs, info = env.reset(seed=seed); traj = [env.pos.copy()]; idx = 0; term = trunc = False; info = {}
    while not (term or trunc):
        a, idx = expert_action(env.pos, side, idx, env.step_size)
        obs, r, term, trunc, info = env.step(a); traj.append(env.pos.copy())
    return np.array(traj), info.get("success", False), info.get("collision", False)


def run_chunked(env, predict_fn, H, seed):
    obs, info = env.reset(seed=seed); traj = [env.pos.copy()]; term = trunc = False; info = {}
    while not (term or trunc):
        chunk = predict_fn(obs).reshape(H, -1)
        for a in chunk:
            obs, r, term, trunc, info = env.step(a); traj.append(env.pos.copy())
            if term or trunc:
                break
    return np.array(traj), info.get("success", False), info.get("collision", False)


def summarize(trajs):
    s = sum(ok for _, ok, _ in trajs); c = sum(col for _, _, col in trajs)
    return s / len(trajs) * 100, c / len(trajs) * 100


def main():
    d = np.load("data/demos_obstacle.npz"); H = int(d["H"]); od = 4
    bc = BCPolicy(od, H * 2); bc.load_state_dict(torch.load(BC_MODEL, map_location="cpu")); bc.eval()
    dp = DiffusionPolicy(od, H * 2, T=T); dp.load_state_dict(torch.load(DP_MODEL, map_location="cpu")); dp.eval()
    env = Obstacle2DEnv()

    exp = [run_expert(env, -1 if ep % 2 == 0 else 1, seed=10000 + ep) for ep in range(K)]
    bc_t = [run_chunked(env, lambda o: bc.predict(o), H, seed=20000 + ep) for ep in range(K)]
    dp_t = [run_chunked(env, lambda o: dp.predict(o), H, seed=30000 + ep) for ep in range(K)]

    print("=" * 56)
    for name, trajs in [("Expert", exp), ("BC (chunked)", bc_t), ("DP (chunked)", dp_t)]:
        sr, cr = summarize(trajs)
        print(f"  {name:<14} success {sr:5.1f}%   collision {cr:5.1f}%")
    print("=" * 56)

    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12})
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharex=True, sharey=True)
    panels = [("Expert demos (bimodal)", exp),
              ("BC: averages -> into wall", bc_t),
              ("Diffusion: samples one mode", dp_t)]
    for ax, (title, trajs) in zip(axes, panels):
        ax.add_patch(Circle(OBSTACLE, OBSTACLE_R, color="#888", alpha=0.45, zorder=1))
        ax.add_patch(Circle(GOAL, GOAL_R, color="#2a9d8f", alpha=0.5, zorder=1))
        ax.scatter([0], [2], marker="*", s=180, color="#2a9d8f", zorder=3)
        ax.plot([-START_JITTER, START_JITTER], [0, 0], color="gray", lw=4, alpha=0.5, zorder=1)
        for traj, ok, col in trajs:
            ax.plot(traj[:, 0], traj[:, 1], color=("#2a9d8f" if ok else "#d62728"),
                    alpha=0.35, lw=1.2, zorder=2)
        sr, cr = summarize(trajs)
        ax.set_title(f"{title}\nsuccess {sr:.0f}% / collision {cr:.0f}%")
        ax.set_aspect("equal"); ax.set_xlim(-1.0, 1.0); ax.set_ylim(-0.1, 2.15)
        ax.set_xlabel("x"); ax.grid(True, alpha=0.2)
    axes[0].set_ylabel("y")
    plt.tight_layout()
    out = "logs/day21_multimodal.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Figure saved: {out}")


if __name__ == "__main__":
    main()