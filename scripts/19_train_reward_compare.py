"""
Day 15 - 三方对比:sparse / sparse+curriculum / dense
========================================================
跑三个独立 PPO 训练,共享同一超参数,只在 reward + curriculum 上不同。
"""
import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CallbackList

from src.envs.panda_reach_env_v2 import PandaReachEnvV2
from src.envs.curriculum_callback import CurriculumCallback

LOG_ROOT = "logs/day15"
MODEL_DIR = "models/day15"
TB_LOG = "logs/tensorboard"
os.makedirs(LOG_ROOT, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

N_ENVS = 8
TOTAL_TIMESTEPS = 300_000
SEED = 42

# Curriculum 计划:0-50k scale=0.2,50-150k scale=0.5,150k+ scale=1.0
CURRICULUM_SCHEDULE = [(0, 0.2), (50_000, 0.5), (150_000, 1.0)]


def make_env(rank, reward_type, init_scale, log_subdir, seed=0):
    def _init():
        env = PandaReachEnvV2(
            max_steps=200,
            reward_type=reward_type,
            target_range_scale=init_scale,
        )
        env = Monitor(env, filename=os.path.join(log_subdir, f"monitor_{rank}"))
        env.reset(seed=seed + rank)
        return env
    return _init


def train_one(exp_name, reward_type, use_curriculum):
    print("\n" + "=" * 70)
    print(f"Experiment: {exp_name}")
    print(f"  reward_type    = {reward_type}")
    print(f"  use_curriculum = {use_curriculum}")
    print("=" * 70)

    log_dir = os.path.join(LOG_ROOT, exp_name)
    os.makedirs(log_dir, exist_ok=True)

    init_scale = CURRICULUM_SCHEDULE[0][1] if use_curriculum else 1.0

    train_env = DummyVecEnv([
        make_env(i, reward_type, init_scale, log_dir, seed=SEED)
        for i in range(N_ENVS)
    ])
    # eval 环境永远用完整工作空间(scale=1.0)和 dense reward 评估真实能力
    # 注:即便训练用 sparse,评估时我们看的是"是否成功",所以 reward_type 不影响 success
    eval_env = DummyVecEnv([
        make_env(99, reward_type, 1.0,
                 os.path.join(log_dir, "eval"), seed=SEED + 1000)
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
        ent_coef=0.01 if reward_type == "sparse" else 0.0,  # sparse 需要更多探索
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        seed=SEED,
        tensorboard_log=TB_LOG,
        policy_kwargs=dict(net_arch=[256, 256]),
    )

    callbacks = [
        EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(MODEL_DIR, exp_name),
            log_path=log_dir,
            eval_freq=max(10_000 // N_ENVS, 1),
            n_eval_episodes=10,
            deterministic=True,
            verbose=1,
        ),
    ]
    if use_curriculum:
        callbacks.append(CurriculumCallback(CURRICULUM_SCHEDULE, verbose=1))

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=CallbackList(callbacks),
        tb_log_name=f"day15_{exp_name}",
        progress_bar=True,
    )

    model.save(os.path.join(MODEL_DIR, exp_name, "final.zip"))
    train_env.close()
    eval_env.close()
    print(f"✅ {exp_name} done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["A", "B", "C", "all"], default="all",
                        help="只跑某个实验(用于断点续训)")
    args = parser.parse_args()

    experiments = [
        ("A_sparse_no_curriculum", "sparse", False),
        ("B_sparse_curriculum",    "sparse", True),
        ("C_dense_no_curriculum",  "dense",  False),
    ]

    for name, rtype, curr in experiments:
        if args.only == "all" or args.only in name:
            train_one(name, rtype, curr)

    print("\n" + "=" * 70)
    print("All experiments done.  Run scripts/20_compare_rewards.py to analyze.")
    print("=" * 70)


if __name__ == "__main__":
    main()