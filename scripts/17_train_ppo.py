"""
Day 14 - PPO 训练 PandaReach
====================================
关键设计:
- DummyVecEnv + 8 并行环境(MuJoCo 单步快,subproc 反而慢)
- 标准 PPO 超参数(SB3 默认 + 微调)
- TensorBoard 监控 + 自动评估回调
- 保存最佳模型到 models/ppo_panda_reach_best.zip
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy

from src.envs.panda_reach_env import PandaReachEnv

# ---------- 路径 ----------
LOG_DIR = "logs/ppo_panda_reach"
MODEL_DIR = "models"
TB_LOG = "logs/tensorboard"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(TB_LOG, exist_ok=True)

# ---------- 训练参数 ----------
N_ENVS = 8
TOTAL_TIMESTEPS = 250_000
SEED = 42


def make_env(rank, seed=0):
    """工厂函数:返回一个被 Monitor 包装的环境"""
    def _init():
        env = PandaReachEnv(max_steps=200)
        env = Monitor(env, filename=os.path.join(LOG_DIR, f"monitor_{rank}"))
        env.reset(seed=seed + rank)
        return env
    return _init


def main():
    print("=" * 60)
    print(f"Day 14: PPO training on PandaReach")
    print(f"  - {N_ENVS} parallel envs (DummyVecEnv)")
    print(f"  - {TOTAL_TIMESTEPS:,} total timesteps")
    print(f"  - tensorboard: tensorboard --logdir {TB_LOG}")
    print("=" * 60)

    # ---------- 训练环境(并行)----------
    train_env = DummyVecEnv([make_env(i, seed=SEED) for i in range(N_ENVS)])

    # ---------- 评估环境(单个,固定 seed)----------
    eval_env = DummyVecEnv([make_env(99, seed=SEED + 1000)])

    # ---------- PPO 模型 ----------
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=2048,           # 每个 env 收集 2048 步 → 总共 16384 步/更新
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,           # Reach 任务足够稠密,不需要额外探索
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        seed=SEED,
        tensorboard_log=TB_LOG,
        policy_kwargs=dict(net_arch=[256, 256]),  # 2x256 MLP
    )

    # ---------- 回调:训练中定期评估,保存最佳模型 ----------
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=max(10_000 // N_ENVS, 1),   # ~每 10k 步评估一次
        n_eval_episodes=10,
        deterministic=True,
        render=False,
        verbose=1,
    )

    # ---------- 训练 ----------
    print("\n--- training start ---\n")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=eval_callback,
        tb_log_name="ppo_panda_reach",
        progress_bar=True,
    )
    print("\n--- training done ---")

    # ---------- 保存最终模型 ----------
    final_path = os.path.join(MODEL_DIR, "ppo_panda_reach_final.zip")
    model.save(final_path)
    print(f"Final model saved: {final_path}")
    print(f"Best model:        {MODEL_DIR}/best_model.zip")

    # ---------- 最终评估 ----------
    print("\n--- final evaluation (20 episodes) ---")
    mean_r, std_r = evaluate_policy(model, eval_env, n_eval_episodes=20, deterministic=True)
    print(f"Mean return: {mean_r:.2f} ± {std_r:.2f}")

    train_env.close()
    eval_env.close()
    print("\n" + "=" * 60)
    print("Day 14 training complete.")
    print("Next: run scripts/18_eval_ppo.py for detailed evaluation.")
    print("=" * 60)


if __name__ == "__main__":
    main()