# scripts/make_demo_gif.py  —— 可选：生成 README 动态横幅
import os, sys
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')
from src.envs.obstacle2d_env import Obstacle2DEnv, GOAL, OBSTACLE, OBSTACLE_R, GOAL_R
from src.imitation.diffusion_policy import DiffusionPolicy

H, T = 8, 50
dp = DiffusionPolicy(4, H * 2, T=T)
dp.load_state_dict(torch.load("models/bc/dp_obstacle.pt", map_location="cpu")); dp.eval()
env = Obstacle2DEnv()

trajs = []
for ep in range(6):                       # 采几条 DP 轨迹（左右都会出现）
    obs, _ = env.reset(seed=30000 + ep); traj = [env.pos.copy()]; term = trunc = False
    while not (term or trunc):
        for a in dp.predict(obs).reshape(H, -1):
            obs, r, term, trunc, info = env.step(a); traj.append(env.pos.copy())
            if term or trunc: break
    trajs.append(np.array(traj))

maxlen = max(len(t) for t in trajs)
fig, ax = plt.subplots(figsize=(5, 5))

def draw(frame):
    ax.clear()
    ax.add_patch(Circle(OBSTACLE, OBSTACLE_R, color="#888", alpha=0.45))
    ax.add_patch(Circle(GOAL, GOAL_R, color="#2a9d8f", alpha=0.5))
    ax.scatter([0], [2], marker="*", s=180, color="#2a9d8f")
    for t in trajs:
        k = min(frame, len(t) - 1)
        ax.plot(t[:k + 1, 0], t[:k + 1, 1], color="#2a9d8f", alpha=0.5, lw=1.5)
        ax.scatter(t[k, 0], t[k, 1], color="#e76f51", s=30, zorder=3)
    ax.set_xlim(-1, 1); ax.set_ylim(-0.1, 2.15); ax.set_aspect("equal")
    ax.set_title("Diffusion Policy: multimodal detour"); ax.axis("off")

FuncAnimation(fig, draw, frames=maxlen, interval=80).save(
    "logs/day21_multimodal.gif", writer=PillowWriter(fps=12))
print("Saved logs/day21_multimodal.gif")