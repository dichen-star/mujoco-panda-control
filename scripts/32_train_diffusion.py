"""
Day 20 - 训练 Diffusion Policy（在 Day 18 的专家示范上，与 BC 同数据）
"""
import os
import sys
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')
from src.imitation.diffusion_policy import DiffusionPolicy

DATA = "data/demos_reach.npz"
OUT = "models/bc/diffusion_reach.pt"
EPOCHS = 200
BATCH = 256
LR = 1e-3
T = 50
SEED = 0


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    os.makedirs("models/bc", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    d = np.load(DATA)
    obs, act = d["obs"], d["act"]
    obs_mean, obs_std = obs.mean(0), obs.std(0) + 1e-6
    X = torch.as_tensor(obs, dtype=torch.float32)
    Y = torch.as_tensor(act, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X, Y), batch_size=BATCH, shuffle=True)

    model = DiffusionPolicy(obs_dim=obs.shape[1], act_dim=act.shape[1], T=T).to(device)
    model.set_normalizer(obs_mean, obs_std)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"device={device}  T={T}  pairs={len(X)}")
    print("--- training (denoise MSE 应下降并趋稳，绝对值不像 BC 的动作 MSE 那样可直接解读) ---")
    for ep in range(EPOCHS):
        model.train(); tot = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            B = yb.shape[0]
            t = torch.randint(0, T, (B,), device=device)
            eps = torch.randn_like(yb)
            abar = model.abar[t].unsqueeze(-1)
            a_t = torch.sqrt(abar) * yb + torch.sqrt(1 - abar) * eps
            loss = ((model.eps(a_t, xb, t) - eps) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * B
        if (ep + 1) % 20 == 0:
            print(f"epoch {ep+1:3d}  denoise_mse {tot/len(X):.4f}")

    torch.save(model.state_dict(), OUT)
    print(f"\nSaved Diffusion Policy to {OUT}")


if __name__ == "__main__":
    main()