"""
Day 13 - 验证 PandaReachEnv 接口与物理健康度
==============================================
1. Gymnasium 官方 env_checker(暴露接口问题)
2. Random policy 跑通 5 episode(不崩溃)
3. Greedy IK policy(应该收敛,验证奖励设计)
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from src.envs.panda_reach_env import PandaReachEnv
from src.kinematics.inverse import panda_ik


def test_1_env_checker():
    print("\n[1/3] Gymnasium env_checker ...")
    env = PandaReachEnv()
    check_env(env, skip_render_check=True)
    env.close()
    print("    ✅ 接口合规")


def test_2_random_policy(n_episodes=5):
    print(f"\n[2/3] Random policy × {n_episodes} episodes ...")
    env = PandaReachEnv(max_steps=200)
    returns, finals = [], []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep)
        ep_return, last_dist = 0.0, 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            action = env.action_space.sample()
            obs, r, terminated, truncated, info = env.step(action)
            ep_return += r
            last_dist = info["distance"]
        returns.append(ep_return)
        finals.append(last_dist)
        print(f"    ep{ep}: return={ep_return:7.2f}  final_dist={last_dist*1000:6.1f}mm")
    env.close()
    print(f"    平均 return: {np.mean(returns):.2f}  平均终末距离: {np.mean(finals)*1000:.1f}mm")
    print("    ✅ 跑完不崩(数值应该很差,因为是瞎走)")


def test_3_greedy_policy(n_episodes=5):
    """用 IK 求 target_q,然后每步朝它走一小步——应该接近最优"""
    print(f"\n[3/3] Greedy IK policy × {n_episodes} episodes ...")
    env = PandaReachEnv(max_steps=200)
    successes, finals = 0, []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=100 + ep)

        # 用 IK 把目标点解成关节角(姿态固定为 home 朝向)
        from src.kinematics.forward import panda_fk
        T_home = panda_fk(env.unwrapped.data.qpos[:7])
        T_target = np.eye(4)
        T_target[:3, :3] = T_home[:3, :3]
        T_target[:3, 3] = info["target"]
        result = panda_ik(T_target, theta_init=env.unwrapped.data.qpos[:7])
        q_target = result[0] if isinstance(result, tuple) else result

        ep_return, last_dist = 0.0, 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            # 朝 q_target 方向走一步,大小不超过 action_scale
            current_q = env.unwrapped.data.qpos[:7]
            delta = q_target - current_q
            action = np.clip(delta / env.unwrapped.action_scale, -1.0, 1.0)
            obs, r, terminated, truncated, info = env.step(action)
            ep_return += r
            last_dist = info["distance"]
        if terminated:
            successes += 1
        finals.append(last_dist)
        flag = "✓" if terminated else "✗"
        print(f"    ep{ep}: return={ep_return:7.2f}  final_dist={last_dist*1000:6.1f}mm  {flag}")
    env.close()
    print(f"    成功率: {successes}/{n_episodes}  平均终末距离: {np.mean(finals)*1000:.1f}mm")
    print("    ✅ 应该绝大多数 episode 都能成功触达")


def main():
    print("=" * 60)
    print("Day 13: PandaReachEnv health check")
    print("=" * 60)
    test_1_env_checker()
    test_2_random_policy()
    test_3_greedy_policy()
    print("\n" + "=" * 60)
    print("Day 13 done.  Tomorrow: PPO!")
    print("=" * 60)


if __name__ == "__main__":
    main()