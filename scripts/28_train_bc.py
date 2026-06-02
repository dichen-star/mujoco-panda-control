"""
Day 18 - 训练行为克隆策略
=================================
纯监督学习：(obs → expert_action) 的 MSE 回归。CPU 上秒级完成。
"""
import os
import sys
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

from src.imitation.bc_policy import BCPolicy

DATA = "data/demos_reach.npz"
OUT = "models/bc/bc_reach.pt"
EPOCHS = 100
BATCH = 256
LR = 1e-3
SEED = 42


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    os.makedirs("models/bc", exist_ok=True)

    d = np.load(DATA)
    obs, act = d["obs"], d["act"]
    print(f"Dataset: {obs.shape[0]} pairs | obs_dim={obs.shape[1]} "
          f"act_dim={act.shape[1]}")

    # 归一化参数（只从训练数据统计）
    obs_mean = obs.mean(axis=0)
    obs_std = obs.std(axis=0) + 1e-6

    X = torch.as_tensor(obs, dtype=torch.float32)
    Y = torch.as_tensor(act, dtype=torch.float32)

    # 90/10 划分训练/验证
    n = len(X)
    idx = torch.randperm(n)
    n_val = int(0.1 * n)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    loader = DataLoader(TensorDataset(X[tr_idx], Y[tr_idx]),
                        batch_size=BATCH, shuffle=True)
    Xv, Yv = X[val_idx], Y[val_idx]

    model = BCPolicy(obs_dim=obs.shape[1], act_dim=act.shape[1])
    model.set_normalizer(obs_mean, obs_std)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    loss_fn = torch.nn.MSELoss()

    best_val, best_state = float("inf"), None

    print("\n--- training ---")
    for ep in range(EPOCHS):
        model.train()
        tot = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)

        model.eval()
        with torch.no_grad():
            vloss = loss_fn(model(Xv), Yv).item()
        if vloss < best_val:
            best_val = vloss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (ep + 1) % 10 == 0:
            print(f"epoch {ep+1:3d}  train_mse {tot/len(tr_idx):.5f}  "
                  f"val_mse {vloss:.5f}  (best {best_val:.5f})")

    model.load_state_dict(best_state)   # 用验证集最优快照，而非最后一轮
    torch.save(model.state_dict(), OUT)
    print(f"\nBest val MSE: {best_val:.5f}")
    print(f"Saved best-val BC policy to {OUT}")


if __name__ == "__main__":
    main()