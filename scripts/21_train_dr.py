"""
Day 16 - 训练 DR 模型
=================================
和 Day 14 完全相同的超参数,只是环境替换成 v3 with DR enabled。
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

from src.envs.panda_reach_env_v3 import PandaReachEnvV3

LOG_DIR = "logs/day16/dr_training"
MODEL_DIR = "models/day16"
TB_LOG = "logs/tensorboard"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

N_ENVS = 8
TOTAL_TIMESTEPS = 300_000  # DR 训练比标称多一点,因为任务变难了
SEED = 42

# DR 配置(训练时用)
DR_CONFIG = dict(
    enable_dr=True,
    payload_range=(0.0, 0.5),         # 0 到 500g
    damping_scale_range=(0.7, 1.3),   # ±30%
    gravity_scale_range=(0.95, 1.05), # ±5%
    ctrl_noise_std=0.02,              # 2% 动作噪声
)


def make_env(rank, dr_config, log_subdir, seed=0):
    def _init():
        env = PandaReachEnvV3(
            max_steps=200,
            reward_type="dense",
            **dr_config,
        )
        env = Monitor(env, filename=os.path.join(log_subdir, f"monitor_{rank}"))
        env.reset(seed=seed + rank)
        return env
    return _init


def main():
    print("=" * 70)
    print("Day 16: PPO + Domain Randomization")
    print(f"  payload:  {DR_CONFIG['payload_range']} kg")
    print(f"  damping:  ×{DR_CONFIG['damping_scale_range']}")
    print(f"  gravity:  ×{DR_CONFIG['gravity_scale_range']}")
    print(f"  ctrl_std: {DR_CONFIG['ctrl_noise_std']}")
    print("=" * 70)

    train_env = DummyVecEnv([
        make_env(i, DR_CONFIG, LOG_DIR, seed=SEED) for i in range(N_ENVS)
    ])
    # 评估时也用 DR,但范围更小(防止评估结果太抖)
    eval_dr_config = dict(DR_CONFIG)
    eval_env = DummyVecEnv([
        make_env(99, eval_dr_config, LOG_DIR + "_eval", seed=SEED + 1000)
    ])

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        seed=SEED,
        tensorboard_log=TB_LOG,
        policy_kwargs=dict(net_arch=[256, 256]),
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=max(10_000 // N_ENVS, 1),
        n_eval_episodes=20,  # 比 Day 14 的 10 多一倍,因为 DR 评估方差大
        deterministic=True,
        verbose=1,
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=eval_callback,
        tb_log_name="day16_dr",
        progress_bar=True,
    )

    model.save(os.path.join(MODEL_DIR, "ppo_dr_final.zip"))
    train_env.close()
    eval_env.close()
    print(f"\n✅ Best model: {MODEL_DIR}/best_model.zip")
    print(f"✅ Final model: {MODEL_DIR}/ppo_dr_final.zip")


if __name__ == "__main__":
    main()