"""
Day 19 - DAgger (Dataset Aggregation)
========================================
滚动当前策略收集它自己漂到的状态，用专家(IK)在这些状态上补标注，聚合后重训。
迭代数轮，成功率从 BC 的 ~52% 向专家 ~94% 收敛。
"""
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

from src.envs.panda_reach_env import PandaReachEnv
from src.kinematics.forward import panda_fk
from src.kinematics.inverse import panda_ik
from src.imitation.bc_policy import BCPolicy

SEED_DATA = "data/demos_reach.npz"
OUT_MODEL = "models/bc/dagger_reach.pt"
FIG = "logs/day19_dagger_curve.png"

N_DAGGER = 5             # DAgger 迭代轮数
ROLLOUT_EPISODES = 30    # 每轮用当前策略采集多少 episode 的状态
EVAL_EPISODES = 50
BC_EPOCHS = 60
BATCH = 256
LR = 1e-3
SEED = 0


def expert_action(env):
    base = env.unwrapped
    q_now = base.data.qpos[:7]
    T_home = panda_fk(q_now)
    T = np.eye(4); T[:3, :3] = T_home[:3, :3]; T[:3, 3] = base.target
    res = panda_ik(T, theta_init=q_now)
    q_target = res[0] if isinstance(res, tuple) else res
    return np.clip((q_target - q_now) / base.action_scale, -1.0, 1.0).astype(np.float32)


def train_bc(obs_arr, act_arr, obs_dim, act_dim):
    """在聚合数据上重训 BC，返回验证集最优模型"""
    obs_mean = obs_arr.mean(0); obs_std = obs_arr.std(0) + 1e-6
    X = torch.as_tensor(obs_arr, dtype=torch.float32)
    Y = torch.as_tensor(act_arr, dtype=torch.float32)
    n = len(X); idx = torch.randperm(n)
    n_val = max(1, int(0.1 * n))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    loader = DataLoader(TensorDataset(X[tr_idx], Y[tr_idx]), batch_size=BATCH, shuffle=True)
    Xv, Yv = X[val_idx], Y[val_idx]

    model = BCPolicy(obs_dim=obs_dim, act_dim=act_dim)
    model.set_normalizer(obs_mean, obs_std)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    loss_fn = torch.nn.MSELoss()

    best_val, best_state = float("inf"), None
    for ep in range(BC_EPOCHS):
        model.train()
        for xb, yb in loader:
            opt.zero_grad(); loss_fn(model(xb), yb).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            v = loss_fn(model(Xv), Yv).item()
        if v < best_val:
            best_val = v
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best_val


def evaluate(model, n_ep, seed_off):
    env = PandaReachEnv(max_steps=200)
    succ = 0
    for ep in range(n_ep):
        obs, info = env.reset(seed=ep + seed_off)
        term = trunc = False
        while not (term or trunc):
            obs, r, term, trunc, info = env.step(model.predict(obs))
        succ += int(term)
    env.close()
    return succ / n_ep * 100


def rollout_and_label(model, n_ep, seed_off):
    """用当前策略 rollout：记录(访问到的状态, 专家在该状态的动作)"""
    env = PandaReachEnv(max_steps=200)
    obs_new, act_new = [], []
    for ep in range(n_ep):
        obs, info = env.reset(seed=ep + seed_off)
        term = trunc = False
        while not (term or trunc):
            a_exp = expert_action(env)        # 标签：专家在当前状态的动作
            obs_new.append(obs.astype(np.float32))
            act_new.append(a_exp)
            obs, r, term, trunc, info = env.step(model.predict(obs))  # 前进：学生动作
    env.close()
    return np.asarray(obs_new, np.float32), np.asarray(act_new, np.float32)


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    os.makedirs("models/bc", exist_ok=True)

    d = np.load(SEED_DATA)
    obs_buf, act_buf = d["obs"].copy(), d["act"].copy()
    obs_dim, act_dim = obs_buf.shape[1], act_buf.shape[1]
    print(f"Seed dataset: {len(obs_buf)} pairs")

    sizes, succ_hist, val_hist = [], [], []

    print("\n[iter 0] train initial BC ...")
    model, vbest = train_bc(obs_buf, act_buf, obs_dim, act_dim)
    s = evaluate(model, EVAL_EPISODES, seed_off=50000)
    sizes.append(len(obs_buf)); succ_hist.append(s); val_hist.append(vbest)
    print(f"         dataset {len(obs_buf):6d}  val_mse {vbest:.4f}  success {s:.1f}%")

    for it in range(1, N_DAGGER + 1):
        o_new, a_new = rollout_and_label(model, ROLLOUT_EPISODES, seed_off=it * 1000)
        obs_buf = np.concatenate([obs_buf, o_new], axis=0)
        act_buf = np.concatenate([act_buf, a_new], axis=0)
        model, vbest = train_bc(obs_buf, act_buf, obs_dim, act_dim)
        s = evaluate(model, EVAL_EPISODES, seed_off=50000)
        sizes.append(len(obs_buf)); succ_hist.append(s); val_hist.append(vbest)
        print(f"[iter {it}] +{len(o_new):5d} states  dataset {len(obs_buf):6d}  "
              f"val_mse {vbest:.4f}  success {s:.1f}%")

    torch.save(model.state_dict(), OUT_MODEL)
    print(f"\nSaved final DAgger policy to {OUT_MODEL}")

    # ---- 学习曲线 ----
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12})
    fig, ax = plt.subplots(figsize=(8, 5))
    iters = list(range(len(succ_hist)))
    ax.plot(iters, succ_hist, "-o", color="#9b5de5", lw=2.2, markersize=7, label="DAgger policy")
    ax.axhline(94, color="#2a9d8f", ls="--", lw=1.6, label="Expert (IK) 94%")
    ax.axhline(89, color="#264653", ls=":", lw=1.6, label="PPO (Day14) 89%")
    ax.axhline(52, color="#e76f51", ls="-.", lw=1.4, label="BC start 52%")
    ax.set_xlabel("DAgger iteration"); ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 100); ax.set_xticks(iters)
    ax.set_title("DAgger closes the covariate-shift gap")
    ax.grid(True, alpha=0.25, linestyle="--"); ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=True)
    for i, sc in zip(iters, succ_hist):
        ax.annotate(f"{sc:.0f}%", (i, sc), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG, dpi=140, bbox_inches="tight")
    print(f"Figure saved: {FIG}")

    np.savez("logs/day19_dagger_metrics.npz",
             sizes=np.array(sizes), success=np.array(succ_hist), val=np.array(val_hist))


if __name__ == "__main__":
    main()