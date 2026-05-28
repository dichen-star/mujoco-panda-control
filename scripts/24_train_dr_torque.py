"""
Day 16B - 力矩控制版：训练标称模型 + DR 模型
================================================
和 Day 16 (21_train_dr.py) 结构相同，两处不同：
  1. 环境换成 PandaReachEnvTorque（力矩动作空间）
  2. 一次训两个模型：nominal (enable_dr=False) 和 dr (enable_dr=True)
     这样才能在"动力学暴露"的动作空间下做对照实验
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

from src.envs.panda_reach_env_torque import PandaReachEnvTorque

LOG_ROOT = "logs/day16_torque"
MODEL_ROOT = "models/day16_torque"
TB_LOG = "logs/tensorboard"
os.makedirs(LOG_ROOT, exist_ok=True)
os.makedirs(MODEL_ROOT, exist_ok=True)

N_ENVS = 8
TOTAL_TIMESTEPS = 300_000
SEED = 42

DR_RANGES = dict(
    payload_range=(0.0, 0.5),
    damping_scale_range=(0.7, 1.3),
    gravity_scale_range=(0.95, 1.05),
    ctrl_noise_std=0.02,
)


def make_env(rank, enable_dr, log_subdir, seed=0):
    def _init():
        if enable_dr:
            env = PandaReachEnvTorque(
                max_steps=200, reward_type="dense",
                enable_dr=True, **DR_RANGES,
            )
        else:
            # 标称：关闭 DR，所有参数固定在标称值
            env = PandaReachEnvTorque(
                max_steps=200, reward_type="dense",
                enable_dr=False, ctrl_noise_std=0.0,
            )
        env = Monitor(env, filename=os.path.join(log_subdir, f"monitor_{rank}"))
        env.reset(seed=seed + rank)
        return env
    return _init


def train_one(tag, enable_dr):
    print("\n" + "=" * 70)
    print(f"Torque control training: {tag}  (enable_dr={enable_dr})")
    print("=" * 70)

    log_dir = os.path.join(LOG_ROOT, tag)
    model_dir = os.path.join(MODEL_ROOT, tag)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    train_env = DummyVecEnv([
        make_env(i, enable_dr, log_dir, seed=SEED) for i in range(N_ENVS)
    ])
    eval_env = DummyVecEnv([
        make_env(99, enable_dr, os.path.join(log_dir, "eval"), seed=SEED + 1000)
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
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=max(10_000 // N_ENVS, 1),
        n_eval_episodes=20,
        deterministic=True,
        verbose=1,
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=eval_callback,
        tb_log_name=f"day16torque_{tag}",
        progress_bar=True,
    )
    model.save(os.path.join(model_dir, "final.zip"))
    train_env.close()
    eval_env.close()
    print(f"✅ {tag} done. best → {model_dir}/best_model.zip")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["nominal", "dr", "both"], default="both")
    args = parser.parse_args()

    if args.only in ("nominal", "both"):
        train_one("nominal", enable_dr=False)
    if args.only in ("dr", "both"):
        train_one("dr", enable_dr=True)

    print("\n" + "=" * 70)
    print("Done. Next: python scripts/25_robustness_torque.py")
    print("=" * 70)


if __name__ == "__main__":
    main()