"""Day 21 - 采集多模态专家示范（动作分块形式）"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')
from src.envs.obstacle2d_env import Obstacle2DEnv, expert_action

N_EPISODES = 400
H = 8
OUT = "data/demos_obstacle.npz"


def main():
    os.makedirs("data", exist_ok=True)
    env = Obstacle2DEnv()
    obs_chunks, act_chunks = [], []
    n_succ = 0
    for ep in range(N_EPISODES):
        side = -1 if ep % 2 == 0 else 1     # 50/50 双模态
        obs, info = env.reset(seed=ep)
        idx = 0
        ep_obs, ep_act = [], []
        term = trunc = False
        while not (term or trunc):
            a, idx = expert_action(env.pos, side, idx, env.step_size)
            ep_obs.append(obs.astype(np.float32))
            ep_act.append(a)
            obs, r, term, trunc, info = env.step(a)
        n_succ += int(info["success"])
        # 切 chunk：obs_t -> 未来 H 步动作（末尾用最后动作填充）
        L = len(ep_act)
        for t in range(L):
            chunk = ep_act[t:t + H]
            while len(chunk) < H:
                chunk.append(ep_act[-1])
            obs_chunks.append(ep_obs[t])
            act_chunks.append(np.concatenate(chunk).astype(np.float32))

    obs_arr = np.asarray(obs_chunks, np.float32)
    act_arr = np.asarray(act_chunks, np.float32)
    np.savez_compressed(OUT, obs=obs_arr, act=act_arr, H=H)
    print(f"Collected {len(obs_arr)} (obs, chunk) pairs from {N_EPISODES} episodes")
    print(f"Chunk dim = {act_arr.shape[1]} (H={H} x act=2) | Expert success {n_succ/N_EPISODES*100:.1f}%")
    print(f"Saved to {OUT}")


if __name__ == "__main__":
    main()