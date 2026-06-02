"""
Day 18 - 采集专家示范数据
=================================
专家 = Week 1 的 greedy IK 策略（每步用 IK 解目标关节角，朝它走一步）。
跑 N 个 episode，记录每一步的 (observation, expert_action) 对，存成 npz。
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

from src.envs.panda_reach_env import PandaReachEnv
from src.kinematics.forward import panda_fk
from src.kinematics.inverse import panda_ik

N_EPISODES = 300
OUT = "data/demos_reach.npz"


def expert_action(env):
    """greedy IK 专家：朝 IK 解方向走一步，归一化并裁剪到 [-1,1]"""
    base = env.unwrapped
    q_now = base.data.qpos[:7]
    T_home = panda_fk(q_now)
    T = np.eye(4)
    T[:3, :3] = T_home[:3, :3]   # 保持 home 姿态朝向
    T[:3, 3] = base.target
    res = panda_ik(T, theta_init=q_now)
    q_target = res[0] if isinstance(res, tuple) else res
    delta = q_target - q_now
    return np.clip(delta / base.action_scale, -1.0, 1.0)


def main():
    os.makedirs("data", exist_ok=True)
    env = PandaReachEnv(max_steps=200)

    obs_list, act_list = [], []
    n_success = 0
    for ep in range(N_EPISODES):
        obs, info = env.reset(seed=ep)
        terminated = truncated = False
        ep_ok = False
        while not (terminated or truncated):
            a = expert_action(env).astype(np.float32)
            obs_list.append(obs.astype(np.float32))   # 记录“看到的状态”
            act_list.append(a)                          # 和“专家此刻的动作”
            obs, r, terminated, truncated, info = env.step(a)
            if terminated:
                ep_ok = True
        n_success += int(ep_ok)

    env.close()
    obs_arr = np.asarray(obs_list, dtype=np.float32)
    act_arr = np.asarray(act_list, dtype=np.float32)
    np.savez_compressed(OUT, obs=obs_arr, act=act_arr)

    print("=" * 60)
    print(f"Collected {len(obs_arr)} (obs, action) pairs "
          f"from {N_EPISODES} episodes")
    print(f"Expert success rate: {n_success / N_EPISODES * 100:.1f}%")
    print(f"Saved to {OUT}  ({obs_arr.nbytes/1e6:.1f} MB)")
    print("=" * 60)


if __name__ == "__main__":
    main()