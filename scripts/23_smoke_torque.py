"""Day 16B - 力矩环境冒烟测试：确认接口正确、物理不发散"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

from src.envs.panda_reach_env_torque import PandaReachEnvTorque

env = PandaReachEnvTorque(max_steps=200, reward_type="dense", enable_dr=False)
obs, info = env.reset(seed=0)
print("reset OK. obs shape:", obs.shape, "| target:", info["target"])

# 1) 零动作：只有重力补偿，机械臂应大致悬停，不应飞出/出 NaN
max_dist = 0.0
for _ in range(200):
    obs, r, term, trunc, info = env.step(np.zeros(7, dtype=np.float32))
    max_dist = max(max_dist, info["distance"])
    if not np.all(np.isfinite(obs)):
        print("❌ NaN/Inf in obs — 物理发散，接口或补偿有问题"); sys.exit(1)
    if term or trunc:
        break
print(f"零动作 200 步：物理稳定 ✓  末端最大偏移 {max_dist*1000:.0f}mm "
      f"(悬停应该 < ~300mm)")

# 2) 随机动作：不应发散
obs, info = env.reset(seed=1)
for _ in range(200):
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    if not np.all(np.isfinite(obs)):
        print("❌ 随机动作下物理发散"); sys.exit(1)
    if term or trunc: break
env.close()
print("随机动作 200 步：物理稳定 ✓")
print("\n✅ 接口冒烟测试通过，可以开训练。")